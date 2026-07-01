"""
trainer.py - Lightweight training loop for claim-level UQ heads.

Replaces the HuggingFace Trainer glue with a small, explicit loop that natively
handles variable-length per-claim labels and, crucially, selects the best model
by a SAMPLE-LEVEL (trace-level) validation metric — the same quantity we report.

Per epoch:
  1. Train on claim BCE (+ multi-task trace BCE for heads that emit a trace logit).
  2. Run inference on validation, collect per-claim probabilities and the head's
     trace logit (if any), aggregate claims -> trace via a validation-selected
     rule, and score AUROC/PR-AUC/ECE/Acc at the TRACE level.
  3. Keep the checkpoint with the best trace metric; early-stop on patience.

The chosen aggregation rule (and the mix weights) are persisted so evaluation
reproduces the exact validation-selected readout.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.cached_features import CachedFeatureDataset, ChunkAwareSampler
from models.aggregation import aggregate_batch, select_aggregation
from scripts.metrics import compute_all_metrics
from utils.common import collate_claim_traces

log = logging.getLogger(__name__)


@dataclass
class TrainState:
    best_metric: float = float("-inf")
    best_epoch: int = -1
    best_state: Optional[dict] = None
    best_aggregation: dict = field(default_factory=lambda: {"method": "conc", "mix_weights": None})
    history: List[dict] = field(default_factory=list)
    no_improve: int = 0


def _amp_dtype(name: str):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(name, torch.bfloat16)


def _move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            out[k] = [t.to(device, non_blocking=True) for t in v]
        else:
            out[k] = v
    return out


@torch.no_grad()
def collect_predictions(model, loader: DataLoader, device: torch.device, amp_dtype) -> dict:
    """Run the model over a loader; return per-trace claim probs/types/labels + trace logits."""
    model.eval()
    claim_probs_per_trace: List[np.ndarray] = []
    claim_types_per_trace: List[np.ndarray] = []
    claim_labels_per_trace: List[np.ndarray] = []
    trace_labels: List[int] = []
    trace_logits: List[Optional[float]] = []
    k_hops: List[int] = []

    use_amp = device.type == "cuda" and amp_dtype != torch.float32
    for batch in loader:
        batch = _move_batch(batch, device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            out = model(
                features=batch["features"],
                attention_mask=batch["attention_mask"],
                claim_masks=batch["claim_masks"],
                claim_types=batch["claim_types"],
            )
        claim_logits = out.claim_logits.float()           # (B, max_claims, C)
        probs = torch.sigmoid(claim_logits.squeeze(-1))   # (B, max_claims)
        has_trace = out.trace_logit is not None
        if has_trace:
            tprob = torch.sigmoid(out.trace_logit.float().reshape(-1))

        for i, labels_i in enumerate(batch["claim_labels"]):
            n = int(labels_i.shape[0])
            if n == 0:
                continue
            claim_probs_per_trace.append(probs[i, :n].detach().cpu().numpy())
            claim_types_per_trace.append(batch["claim_types"][i][:n].detach().cpu().numpy())
            claim_labels_per_trace.append(labels_i.detach().cpu().numpy().astype(int))
            trace_labels.append(int(batch["trace_labels"][i].item()))
            trace_logits.append(float(tprob[i].item()) if has_trace else None)
            k_hops.append(int(batch["k_hops"][i].item()))

    return {
        "claim_probs": claim_probs_per_trace,
        "claim_types": claim_types_per_trace,
        "claim_labels": claim_labels_per_trace,
        "trace_labels": np.asarray(trace_labels, dtype=int),
        "trace_probs": trace_logits,          # list; None entries for claim-only heads
        "k_hops": np.asarray(k_hops, dtype=int),
    }


def score_trace_level(preds: dict, aggregation: dict, objective: str = "auroc") -> tuple:
    """Return (metrics_dict, trace_scores) at the sample level for a fixed aggregation.

    If the head emits a learned trace probability, it is blended with the
    aggregated claim score by simple averaging — this uses the multi-task signal
    while staying comparable to claim-only heads (which skip the blend).
    """
    agg_scores = aggregate_batch(
        preds["claim_probs"], preds["claim_types"],
        method=aggregation["method"], mix_weights=aggregation.get("mix_weights"),
    )
    trace_probs = preds["trace_probs"]
    if any(tp is not None for tp in trace_probs):
        learned = np.array([tp if tp is not None else np.nan for tp in trace_probs], dtype=np.float64)
        blended = np.where(np.isnan(learned), agg_scores, 0.5 * (agg_scores + learned))
        final_scores = blended
    else:
        final_scores = agg_scores
    metrics = compute_all_metrics(preds["trace_labels"], final_scores)
    return metrics, final_scores


def _build_loader(ds: CachedFeatureDataset, batch_size: int, shuffle: bool,
                  num_workers: int, seed: int) -> DataLoader:
    sampler = None
    if shuffle and not ds._preload_all:
        sampler = ChunkAwareSampler(ds, seed=seed, drop_last=False, batch_size=batch_size)
        shuffle = False
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=num_workers, collate_fn=collate_claim_traces,
        pin_memory=torch.cuda.is_available(), drop_last=False,
        persistent_workers=num_workers > 0,
    )


def train_loop(
    model,
    train_ds: CachedFeatureDataset,
    val_ds: CachedFeatureDataset,
    *,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    max_grad_norm: float,
    grad_accum_steps: int,
    amp_dtype_name: str,
    metric_for_best_model: str,
    early_stopping_patience: int,
    num_workers: int,
    seed: int,
    device: torch.device,
    on_epoch_end=None,
) -> TrainState:
    amp_dtype = _amp_dtype(amp_dtype_name)
    train_loader = _build_loader(train_ds, batch_size, True, num_workers, seed)
    val_loader = _build_loader(val_ds, batch_size, False, num_workers, seed)

    optimizer = torch.optim.AdamW(model.trainable_params(), lr=learning_rate, weight_decay=weight_decay)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, grad_accum_steps)))
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp = device.type == "cuda" and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32

    state = TrainState()
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        if isinstance(getattr(train_loader, "sampler", None), ChunkAwareSampler):
            train_loader.sampler.set_epoch(epoch)
        t0 = time.time()
        running = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            batch = _move_batch(batch, device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=autocast_enabled):
                out = model(
                    features=batch["features"],
                    attention_mask=batch["attention_mask"],
                    claim_masks=batch["claim_masks"],
                    claim_types=batch["claim_types"],
                    claim_labels=batch["claim_labels"],
                    trace_labels=batch["trace_labels"],
                )
                loss = out.loss / grad_accum_steps
            if loss is None:
                continue
            scaler.scale(loss).backward()
            if (step + 1) % grad_accum_steps == 0:
                if max_grad_norm and max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.trainable_params(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            running += float(out.loss.detach().item())

        # ----- validation (sample-level) -----
        preds = collect_predictions(model, val_loader, device, amp_dtype)
        agg = select_aggregation(
            preds["claim_probs"], preds["trace_labels"], preds["claim_types"],
            objective="prauc" if metric_for_best_model == "pr_auc" else "auroc",
        )
        val_metrics, _ = score_trace_level(preds, agg, objective=metric_for_best_model)
        metric_key = {"auroc": "roc_auc", "pr_auc": "pr_auc", "prauc": "pr_auc"}.get(
            metric_for_best_model, "roc_auc"
        )
        current = val_metrics.get(metric_key, float("nan"))
        current = current if not math.isnan(current) else float("-inf")

        train_loss = running / max(1, len(train_loader))
        epoch_s = time.time() - t0
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_auroc": val_metrics.get("roc_auc"),
            "val_pr_auc": val_metrics.get("pr_auc"),
            "val_ece": val_metrics.get("ece"),
            "val_acc": val_metrics.get("accuracy"),
            "aggregation": agg["method"],
            "epoch_s": epoch_s,
        }
        state.history.append(entry)
        log.info(
            "epoch %d | loss %.4f | val AUROC %.4f PR-AUC %.4f ECE %.4f Acc %.4f | agg=%s | %.1fs",
            epoch, train_loss, entry["val_auroc"] or 0.0, entry["val_pr_auc"] or 0.0,
            entry["val_ece"] or 0.0, entry["val_acc"] or 0.0, agg["method"], epoch_s,
        )
        if on_epoch_end is not None:
            on_epoch_end(entry)

        if current > state.best_metric:
            state.best_metric = current
            state.best_epoch = epoch
            state.best_state = {k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()}
            state.best_aggregation = agg
            state.no_improve = 0
        else:
            state.no_improve += 1
            if early_stopping_patience > 0 and state.no_improve >= early_stopping_patience:
                log.info("Early stopping at epoch %d (best epoch %d, %s=%.4f)",
                         epoch, state.best_epoch, metric_key, state.best_metric)
                break

    if state.best_state is not None:
        model.head.load_state_dict(state.best_state)
    return state
