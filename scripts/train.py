#!/usr/bin/env python3
"""
train.py - Phase 2: train a claim-level UQ head on cached frozen features.

Loads Phase-1 cached traces (no LLM in the loop), trains the head with claim BCE
(+ multi-task trace BCE when the head emits a trace logit), selects the best
checkpoint by SAMPLE-LEVEL validation AUROC, and saves:

    {output_dir}/final_model/head_weights.pth
    {output_dir}/final_model/head_config.json   (includes the validation-selected aggregation)
    {output_dir}/train_results.json             (history + efficiency)

Usage:
    python scripts/train.py --head_type spatialmind --cache_dir /path/to/cache --output_dir /path/out
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from config import Config
from data.cached_features import CachedFeatureDataset
from models.heads import build_head
from models.wrapper import ClaimUQModel
from scripts.trainer import train_loop
from spatial_constraints import CLAIM_CONSTRAINT_DIM, TRACE_CONSTRAINT_DIM
from utils.efficiency import (
    get_cpu_peak_memory_gb,
    get_gpu_peak_memory_gb,
    reset_gpu_peak_memory,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 2: train a claim-level UQ head")
    p.add_argument("--head_type", type=str, default=None)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--num_epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--loss_type", type=str, default=None, choices=["bce", "balanced_bce", "focal"])
    p.add_argument("--loss_pos_weight", type=float, default=None)
    p.add_argument("--focal_gamma", type=float, default=None)
    p.add_argument("--trace_loss_weight", type=float, default=None)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def apply_overrides(cfg: Config, args) -> Config:
    if args.head_type:
        cfg.head.head_type = args.head_type
    if args.cache_dir:
        cfg.generation.cache_dir = args.cache_dir
    if args.output_dir:
        cfg.output.output_dir = args.output_dir
    if args.num_epochs:
        cfg.training.num_epochs = args.num_epochs
    if args.batch_size:
        cfg.training.per_device_train_batch_size = args.batch_size
        cfg.training.per_device_eval_batch_size = args.batch_size
    if args.learning_rate:
        cfg.training.learning_rate = args.learning_rate
    if args.loss_type:
        cfg.training.loss_type = args.loss_type
    if args.loss_pos_weight is not None:
        cfg.training.loss_pos_weight = args.loss_pos_weight
    if args.focal_gamma is not None:
        cfg.training.focal_gamma = args.focal_gamma
    if args.trace_loss_weight is not None:
        cfg.head.trace_loss_weight = args.trace_loss_weight
    if args.max_train_samples is not None:
        cfg.dataset.max_train_samples = args.max_train_samples
    if args.max_eval_samples is not None:
        cfg.dataset.max_eval_samples = args.max_eval_samples
    if args.seed is not None:
        cfg.training.seed = args.seed
    return cfg


def main():
    args = parse_args()
    cfg = apply_overrides(Config(), args)

    torch.manual_seed(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = cfg.generation.cache_dir
    output_dir = cfg.output.output_dir
    os.makedirs(output_dir, exist_ok=True)

    log.info("=" * 70)
    log.info("SpatialMind Phase 2: Train head=%s", cfg.head.head_type)
    log.info("  cache_dir=%s", cache_dir)
    log.info("  output_dir=%s", output_dir)
    log.info("  epochs=%d batch=%d lr=%g loss=%s trace_w=%.2f",
             cfg.training.num_epochs, cfg.training.per_device_train_batch_size,
             cfg.training.learning_rate, cfg.training.loss_type, cfg.head.trace_loss_weight)
    log.info("=" * 70)

    train_ds = CachedFeatureDataset(cache_dir, "train", max_samples=cfg.dataset.max_train_samples)
    val_ds = CachedFeatureDataset(cache_dir, "validation", max_samples=cfg.dataset.max_eval_samples)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(
            "Empty train/val after pending-filtering. Run scripts/judge.py on the cache first."
        )

    feature_dim = train_ds[0]["features"].shape[-1]
    n_tr, n_cor, n_hall = train_ds.trace_label_stats()
    n_cl, n_cl_cor, n_cl_hall = train_ds.claim_label_stats()
    log.info("Train traces: %d (%.1f%% correct, %.1f%% hallucination)",
             n_tr, 100 * n_cor / max(n_tr, 1), 100 * n_hall / max(n_tr, 1))
    log.info("Train claims: %d (%.1f%% correct, %.1f%% hallucination)",
             n_cl, 100 * n_cl_cor / max(n_cl, 1), 100 * n_cl_hall / max(n_cl, 1))

    # Auto pos_weight for balanced_bce. The claim and trace tasks have OPPOSITE
    # class balance, so compute a SEPARATE pos_weight (= n_neg / n_pos) for each.
    # When loss_pos_weight is set (>0) it overrides both.
    override = float(cfg.training.loss_pos_weight)
    if cfg.training.loss_type == "balanced_bce" and override <= 0:
        claim_pos_weight = float(n_cl_hall / max(n_cl_cor, 1))
        trace_pos_weight = float(n_hall / max(n_cor, 1))
    else:
        claim_pos_weight = trace_pos_weight = max(override, 0.0) or 1.0
    log.info("balanced_bce pos_weight: claim=%.3f, trace=%.3f (loss_type=%s)",
             claim_pos_weight, trace_pos_weight, cfg.training.loss_type)

    head = build_head(
        head_type=cfg.head.head_type,
        feature_dim=feature_dim,
        num_classes=cfg.head.num_classes,
        head_dim=cfg.head.head_dim,
        n_layers=cfg.head.n_layers,
        n_heads=cfg.head.n_heads,
        dropout=cfg.head.dropout,
        max_seq_len=cfg.head.max_seq_len,
    )
    trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
    log.info("Head %s: %s trainable params, feature_dim=%d, emits_trace_logit=%s",
             cfg.head.head_type, f"{trainable:,}", feature_dim, getattr(head, "emits_trace_logit", False))

    model = ClaimUQModel(
        head=head,
        num_classes=cfg.head.num_classes,
        loss_type=cfg.training.loss_type,
        claim_pos_weight=claim_pos_weight,
        trace_pos_weight=trace_pos_weight,
        focal_gamma=cfg.training.focal_gamma,
        trace_loss_weight=cfg.head.trace_loss_weight,
    ).to(device)

    reset_gpu_peak_memory(device)
    start = time.time()
    state = train_loop(
        model, train_ds, val_ds,
        num_epochs=cfg.training.num_epochs,
        batch_size=cfg.training.per_device_train_batch_size,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_ratio=cfg.training.warmup_ratio,
        max_grad_norm=cfg.training.max_grad_norm,
        grad_accum_steps=cfg.training.grad_accum_steps,
        amp_dtype_name=cfg.training.amp_dtype,
        metric_for_best_model=cfg.training.metric_for_best_model,
        early_stopping_patience=cfg.training.early_stopping_patience,
        num_workers=cfg.training.num_workers,
        seed=cfg.training.seed,
        device=device,
        aggregation=cfg.head.aggregation,
    )
    train_time = time.time() - start
    gpu_peak = get_gpu_peak_memory_gb(device)
    cpu_peak = get_cpu_peak_memory_gb()

    # ---- save best model + config (incl. validation-selected aggregation) ----
    final_dir = os.path.join(output_dir, cfg.output.final_model_subdir)
    os.makedirs(final_dir, exist_ok=True)
    epochs_run = len(state.history)
    head_config = {
        "head_type": cfg.head.head_type,
        "feature_dim": feature_dim,
        "num_classes": cfg.head.num_classes,
        "head_dim": cfg.head.head_dim,
        "n_layers": cfg.head.n_layers,
        "n_heads": cfg.head.n_heads,
        "dropout": cfg.head.dropout,
        "max_seq_len": cfg.head.max_seq_len,
        "trace_loss_weight": cfg.head.trace_loss_weight,
        "emits_trace_logit": bool(getattr(head, "emits_trace_logit", False)),
        "aggregation": state.best_aggregation,   # {method, mix_weights}
        "trainable_params": trainable,
        "cache_dir": cache_dir,
        "cache_schema_version": int(train_ds.manifest.get("cache_schema_version", 0)),
        "constraint_method": train_ds.manifest.get("constraint_method", ""),
        "claim_constraint_dim": CLAIM_CONSTRAINT_DIM,
        "trace_constraint_dim": TRACE_CONSTRAINT_DIM,
        "seed": cfg.training.seed,
    }
    torch.save(model.head.state_dict(), os.path.join(final_dir, "head_weights.pth"))
    with open(os.path.join(final_dir, "head_config.json"), "w", encoding="utf-8") as f:
        json.dump(head_config, f, indent=2)

    efficiency = {
        "params_m": trainable / 1e6,
        "train_time_h": train_time / 3600.0,
        "epoch_s": (train_time / epochs_run) if epochs_run else 0.0,
        "epochs_completed": epochs_run,
        "cpu_memory_gb": cpu_peak,
        "gpu_memory_gb": gpu_peak,
    }
    results = {
        "head_type": cfg.head.head_type,
        "cache_dir": cache_dir,
        "trainable_params": trainable,
        "train_samples": len(train_ds),
        "eval_samples": len(val_ds),
        "best_epoch": state.best_epoch,
        "best_val_metric": state.best_metric,
        "best_aggregation": state.best_aggregation,
        "history": state.history,
        "efficiency": efficiency,
        "head_config": head_config,
    }
    with open(os.path.join(output_dir, "train_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    log.info("Saved best model to %s", final_dir)
    log.info("Best epoch %d | val %s=%.4f | agg=%s",
             state.best_epoch, cfg.training.metric_for_best_model,
             state.best_metric, state.best_aggregation.get("method"))
    log.info("Training done in %.1f min", train_time / 60.0)


if __name__ == "__main__":
    main()
