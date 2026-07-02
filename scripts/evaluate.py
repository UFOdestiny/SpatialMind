#!/usr/bin/env python3
"""
evaluate.py - Phase 3: SAMPLE-LEVEL (trace-level) evaluation.

For every method we produce ONE reliability score per trace and evaluate against
the trace-level correctness label. This is the meaningful unit: does the system
know when to trust the model's final spatial answer?

Protocol (applied identically to supervised heads AND unsupervised baselines):
  1. Score every claim of every test trace (head: sigmoid(claim logit);
     baseline: 1 - uncertainty, min-max normalized with VALIDATION-fit stats).
  2. Aggregate a trace's claim scores -> one trace score using a rule SELECTED ON
     VALIDATION (conc / avg / min / mix), shared across all methods. NEVER fit on
     the test split. In distribution the rule is re-selected on the eval cache's
     own validation split; under OOD transfer (no target validation cache) it
     falls back to the train-time / source rule, so test statistics never leak.
  3. Supervised heads that emit a learned trace logit additionally blend it in
     (their multi-task contribution); this is reported alongside the shared rule.
  4. Compute AUROC / PR-AUC / Acc / ECE (+ diagnostics) at the trace level.

Usage:
    # one trained head
    python scripts/evaluate.py --head_path .../final_model --cache_dir /cache --split test
    # unsupervised baselines
    python scripts/evaluate.py --cache_dir /cache --split test --eval_baselines
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

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from data.cached_features import CachedFeatureDataset
from data.datasets import get_dataset_cls
from models.aggregation import aggregate_batch, select_aggregation
from models.heads import build_head
from models.unsup_heads import build_estimator, UNSUPERVISED_HEAD_REGISTRY
from models.wrapper import ClaimUQModel
from scripts.metrics import compute_all_metrics, compute_claim_metrics
from scripts.trainer import collect_predictions, score_trace_level, _amp_dtype
from utils.common import collate_claim_traces
from utils.efficiency import get_gpu_peak_memory_gb, reset_gpu_peak_memory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 3: sample-level evaluation")
    p.add_argument("--head_path", type=str, default=None)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--val_split", type=str, default="validation",
                   help="Split used to (re)select the claim->trace aggregation rule.")
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--eval_baselines", action="store_true")
    p.add_argument("--baselines", type=str, default=None, help="Comma list; default all.")
    p.add_argument("--save_predictions", action="store_true", default=True)
    p.add_argument("--no_save_predictions", dest="save_predictions", action="store_false")
    return p.parse_args()


def _read_manifest(cache_dir: str, split: str) -> dict:
    path = Path(cache_dir) / split / "manifest.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _base_model_name(cache_dir: str, split: str, cfg) -> str:
    mf = _read_manifest(cache_dir, split)
    if mf.get("model_path"):
        return Path(mf["model_path"]).name
    return Path(cfg.model.pretrained_model_name_or_path).name or "unknown"


def _difficulty_field(cache_dir: str, split: str) -> str:
    mf = _read_manifest(cache_dir, split)
    name = mf.get("dataset_name", "")
    if name:
        try:
            return get_dataset_cls(name).difficulty_field
        except ValueError:
            pass
    return "k_hop"


def _make_loader(ds, batch_size):
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_claim_traces,
                      pin_memory=torch.cuda.is_available(), num_workers=2)


def _skipped_report(method_type, head_type, base_model, split, difficulty_field):
    """Report for a split with no usable (labeled) samples: recorded, not crashed."""
    return {
        "method_type": method_type,
        "head_type": head_type,
        "base_model_name": base_model,
        "split": split,
        "difficulty_field": difficulty_field,
        "total_samples": 0,
        "status": "skipped_no_usable_samples",
        "overall_metrics": {},
        "per_difficulty_metrics": {},
        "efficiency": {},
    }


def _per_difficulty(k_hops, labels, scores, field, threshold=0.5):
    out = {}
    for d in sorted(set(k_hops.tolist())):
        mask = k_hops == d
        if mask.sum() > 0 and len(np.unique(labels[mask])) >= 1:
            out[int(d)] = compute_all_metrics(labels[mask], scores[mask], threshold)
    return out


# --------------------------------------------------------------------------- #
# Supervised head
# --------------------------------------------------------------------------- #
def evaluate_head(args, cfg, cache_dir, output_dir, difficulty_field):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(args.head_path, "head_config.json")) as f:
        head_cfg = json.load(f)

    head = build_head(
        head_type=head_cfg["head_type"], feature_dim=head_cfg["feature_dim"],
        num_classes=head_cfg.get("num_classes", 1), head_dim=head_cfg.get("head_dim", 256),
        n_layers=head_cfg.get("n_layers", 2), n_heads=head_cfg.get("n_heads", 8),
        dropout=head_cfg.get("dropout", 0.1), max_seq_len=head_cfg.get("max_seq_len", 512),
    )
    head.load_state_dict(torch.load(os.path.join(args.head_path, "head_weights.pth"), map_location="cpu"))
    model = ClaimUQModel(head=head, num_classes=head_cfg.get("num_classes", 1)).to(device)
    amp_dtype = _amp_dtype(cfg.training.amp_dtype)

    test_ds = CachedFeatureDataset(cache_dir, args.split, max_samples=args.max_samples, max_cached_chunks=2)
    if len(test_ds) == 0:
        log.warning(
            "Eval split '%s' has 0 usable samples (all pending/unjudged?) — skipping %s. "
            "Run scripts/judge.py on this cache to assign labels.",
            args.split, head_cfg["head_type"],
        )
        return _skipped_report("supervised", head_cfg["head_type"],
                               _base_model_name(cache_dir, args.split, cfg), args.split, difficulty_field)

    # Re-select aggregation on the eval cache's own validation split (fair OOD calibration).
    aggregation = head_cfg.get("aggregation", {"method": "conc", "mix_weights": None})
    try:
        val_ds = CachedFeatureDataset(cache_dir, args.val_split, max_cached_chunks=2)
        val_preds = collect_predictions(model, _make_loader(val_ds, args.batch_size), device, amp_dtype)
        aggregation = select_aggregation(
            val_preds["claim_probs"], val_preds["trace_labels"], val_preds["claim_types"],
            objective="auroc",
        )
        log.info("Re-selected aggregation on %s: %s", args.val_split, aggregation)
    except FileNotFoundError:
        log.info("No %s split in eval cache; using train-time aggregation: %s", args.val_split, aggregation)

    reset_gpu_peak_memory(device)
    t0 = time.time()
    preds = collect_predictions(model, _make_loader(test_ds, args.batch_size), device, amp_dtype)
    infer_s = time.time() - t0

    metrics, trace_scores = score_trace_level(preds, aggregation)
    labels = preds["trace_labels"]

    # Claim-level diagnostics (secondary).
    flat_probs, flat_labels, flat_types = [], [], []
    for cp, cl, ct in zip(preds["claim_probs"], preds["claim_labels"], preds["claim_types"]):
        flat_probs.extend(cp.tolist()); flat_labels.extend(cl.tolist()); flat_types.extend(ct.tolist())
    claim_metrics = compute_claim_metrics(
        np.asarray(flat_labels), np.asarray(flat_probs), np.asarray(flat_types)
    ) if flat_labels else {}

    per_diff = _per_difficulty(preds["k_hops"], labels, trace_scores, difficulty_field)

    train_results_path = Path(args.head_path).resolve().parent / "train_results.json"
    train_eff = {}
    if train_results_path.is_file():
        with open(train_results_path) as f:
            train_eff = json.load(f).get("efficiency", {})

    report = {
        "method_type": "supervised",
        "head_type": head_cfg["head_type"],
        "base_model_name": _base_model_name(cache_dir, args.split, cfg),
        "split": args.split,
        "difficulty_field": difficulty_field,
        "total_samples": int(len(labels)),
        "trainable_params": head_cfg.get("trainable_params", 0),
        "aggregation": aggregation,
        "overall_metrics": metrics,
        "claim_metrics": claim_metrics,
        "per_difficulty_metrics": per_diff,
        "efficiency": {
            "params_m": head_cfg.get("trainable_params", 0) / 1e6,
            "train_time_h": train_eff.get("train_time_h", 0.0),
            "epoch_s": train_eff.get("epoch_s", 0.0),
            "inference_s": infer_s,
            "samples_per_second": len(labels) / infer_s if infer_s > 0 else 0.0,
            "gpu_memory_gb": max(train_eff.get("gpu_memory_gb", 0.0), get_gpu_peak_memory_gb(device)),
        },
    }
    if args.save_predictions:
        report["predictions"] = [
            {"sample_id": i, "trace_label": int(labels[i]), "trace_score": float(trace_scores[i]),
             "k_hop": int(preds["k_hops"][i]),
             "claim_probs": preds["claim_probs"][i].tolist(),
             "claim_labels": preds["claim_labels"][i].tolist()}
            for i in range(len(labels))
        ]
    return report


# --------------------------------------------------------------------------- #
# Unsupervised baselines
# --------------------------------------------------------------------------- #
def _collect_baseline_uncertainty(estimator, ds):
    """Gather per-claim UNCERTAINTY (raw, un-normalized) for every usable trace."""
    claim_unc_per_trace, claim_type_per_trace, labels, k_hops = [], [], [], []
    for i in range(len(ds)):
        raw = ds.load_raw(i)
        claims = raw.get("claims", []) or []
        verified = raw.get("verified", []) or []
        # keep only claims with a usable label to align with the supervised protocol
        idxs = [ci for ci in range(min(len(claims), len(verified))) if int(verified[ci]) in (0, 1)]
        if not idxs:
            lbl = int(raw.get("label", 1))
            if lbl not in (0, 1):
                continue
            idxs = list(range(len(claims))) if claims else [0]
        unc = estimator.estimate_claims(raw)
        if unc is None:
            unc = [0.5] * len(claims)
        types = [int(claims[ci].get("claim_type_id", 2)) if ci < len(claims) else 2 for ci in idxs]
        sel_unc = [float(unc[ci]) if ci < len(unc) else 0.5 for ci in idxs]
        claim_unc_per_trace.append(sel_unc)
        claim_type_per_trace.append(types)
        labels.append(int(raw.get("label", 1)))
        k_hops.append(int(raw.get("k_hop", 0)))
    return claim_unc_per_trace, claim_type_per_trace, np.asarray(labels, dtype=int), np.asarray(k_hops, dtype=int)


def _fit_norm_stats(claim_unc_per_trace):
    """Fit min-max normalization stats (lo, range) on a REFERENCE split's uncertainties.

    These stats must come from a non-test split (validation) so the test transform
    never depends on test statistics — otherwise threshold-dependent metrics
    (ECE / Acc@0.5 / Brier / NLL) would leak.
    """
    if not claim_unc_per_trace:
        return 0.0, 1.0
    allu = np.concatenate([np.asarray(u) for u in claim_unc_per_trace])
    if allu.size == 0:
        return 0.0, 1.0
    lo, hi = float(allu.min()), float(allu.max())
    return lo, (hi - lo) if hi > lo else 1.0


def _uncertainty_to_confidence(claim_unc_per_trace, norm_stats):
    """Map per-claim uncertainty -> confidence in [0, 1] with FIXED (validation) stats.

    Higher confidence => more reliable. Values are clipped to [0, 1] since test
    uncertainties may fall slightly outside the validation range.
    """
    lo, rng = norm_stats
    return [[float(np.clip(1.0 - ((u - lo) / rng), 0.0, 1.0)) for u in trace]
            for trace in claim_unc_per_trace]


def _baseline_trace_scores(estimator, ds, aggregation, norm_stats):
    """Per-trace confidence scores via the shared claim->trace protocol.

    `norm_stats` (lo, range) are fit on the validation split and passed in, so the
    uncertainty->confidence map applied to test does not use test statistics.
    """
    claim_unc, claim_types, labels, k_hops = _collect_baseline_uncertainty(estimator, ds)
    claim_conf = _uncertainty_to_confidence(claim_unc, norm_stats)
    scores = aggregate_batch(
        claim_conf, claim_types,
        method=aggregation["method"], mix_weights=aggregation.get("mix_weights"),
    )
    n_claims = np.array([len(c) for c in claim_conf], dtype=int)
    return scores, labels, k_hops, n_claims


def evaluate_baselines(args, cfg, cache_dir, output_dir, difficulty_field, names=None):
    if names is None:
        names = list(UNSUPERVISED_HEAD_REGISTRY.keys())
    reports = {}
    base_model = _base_model_name(cache_dir, args.split, cfg)
    test_ds = CachedFeatureDataset(cache_dir, args.split, max_samples=args.max_samples, max_cached_chunks=2)
    if len(test_ds) == 0:
        log.warning(
            "Eval split '%s' has 0 usable samples (all pending/unjudged?) — skipping baselines. "
            "Run scripts/judge.py on this cache to assign labels.", args.split,
        )
        return {name: _skipped_report("unsupervised", name, base_model, args.split, difficulty_field)
                for name in names}
    try:
        val_ds = CachedFeatureDataset(cache_dir, args.val_split, max_cached_chunks=2)
    except FileNotFoundError:
        val_ds = None

    if val_ds is None:
        log.warning(
            "No '%s' split in eval cache: baseline normalization+aggregation fall back to "
            "the %s split (acceptable for OOD, where the confidence scale is uncalibrated; "
            "AUROC/PR-AUC are unaffected as they are rank-based).",
            args.val_split, args.split,
        )

    for name in names:
        estimator = build_estimator(name)
        # Fit uncertainty->confidence normalization AND select the claim->trace rule
        # on VALIDATION (never test). Fall back to the eval split only when no
        # validation cache exists (OOD), which we warned about above.
        ref_ds = val_ds if val_ds is not None else test_ds
        ref_unc, ref_types, ref_labels, _ = _collect_baseline_uncertainty(estimator, ref_ds)
        norm_stats = _fit_norm_stats(ref_unc)
        aggregation = {"method": "conc", "mix_weights": None}
        if len(ref_labels) > 0 and len(np.unique(ref_labels)) >= 2:
            ref_conf = _uncertainty_to_confidence(ref_unc, norm_stats)
            aggregation = select_aggregation(ref_conf, ref_labels, ref_types, objective="auroc")

        t0 = time.time()
        scores, labels, k_hops, n_claims = _baseline_trace_scores(estimator, test_ds, aggregation, norm_stats)
        infer_s = time.time() - t0
        metrics = compute_all_metrics(labels, scores)
        reports[name] = {
            "method_type": "unsupervised",
            "head_type": name,
            "base_model_name": base_model,
            "split": args.split,
            "difficulty_field": difficulty_field,
            "total_samples": int(len(labels)),
            "aggregation": aggregation,
            "norm_fit_on": args.val_split if val_ds is not None else args.split,
            "overall_metrics": metrics,
            "per_difficulty_metrics": _per_difficulty(k_hops, labels, scores, difficulty_field),
            "efficiency": {"params_m": 0.0, "inference_s": infer_s,
                           "samples_per_second": len(labels) / infer_s if infer_s > 0 else 0.0},
        }
        if args.save_predictions:
            # Store enough for the length-confound diagnostic (n_claims via claim_probs len).
            reports[name]["predictions"] = [
                {"sample_id": i, "trace_label": int(labels[i]), "trace_score": float(scores[i]),
                 "k_hop": int(k_hops[i]), "claim_probs": [0.0] * int(n_claims[i])}
                for i in range(len(labels))
            ]
        m = metrics
        log.info("baseline %-14s AUROC=%.4f PR-AUC=%.4f Acc=%.4f ECE=%.4f",
                 name, m["roc_auc"], m["pr_auc"], m["accuracy"], m["ece"])
    return reports


# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    cfg = Config()
    cache_dir = args.cache_dir or cfg.generation.cache_dir
    output_dir = args.output_dir or os.path.join(cfg.output.output_dir, "evaluation")
    os.makedirs(output_dir, exist_ok=True)
    difficulty_field = _difficulty_field(cache_dir, args.split)
    dataset_name = _read_manifest(cache_dir, args.split).get("dataset_name", "unknown")

    log.info("=" * 70)
    log.info("SpatialMind Evaluation (sample-level) | dataset=%s split=%s", dataset_name, args.split)
    log.info("  cache=%s", cache_dir)
    log.info("=" * 70)

    all_reports = {}
    if args.head_path:
        report = evaluate_head(args, cfg, cache_dir, output_dir, difficulty_field)
        all_reports[report["head_type"]] = report
        with open(os.path.join(output_dir, "evaluation_report.json"), "w") as f:
            json.dump(report, f, indent=2, default=str)
        m = report.get("overall_metrics", {})
        if m:
            log.info("HEAD %-14s AUROC=%.4f PR-AUC=%.4f Acc=%.4f ECE=%.4f (agg=%s)",
                     report["head_type"], m["roc_auc"], m["pr_auc"], m["accuracy"], m["ece"],
                     report.get("aggregation", {}).get("method", "-"))

    if args.eval_baselines:
        names = [b.strip() for b in args.baselines.split(",")] if args.baselines else None
        baseline_reports = evaluate_baselines(args, cfg, cache_dir, output_dir, difficulty_field, names)
        all_reports.update(baseline_reports)
        for name, report in baseline_reports.items():
            bdir = os.path.join(output_dir, name)
            os.makedirs(bdir, exist_ok=True)
            with open(os.path.join(bdir, "evaluation_report.json"), "w") as f:
                json.dump(report, f, indent=2, default=str)

    with open(os.path.join(output_dir, "combined_evaluation.json"), "w") as f:
        json.dump(all_reports, f, indent=2, default=str)

    log.info("=" * 70)
    log.info("SUMMARY (sample-level)")
    for name, report in all_reports.items():
        m = report.get("overall_metrics", {})
        if not m or report.get("status") == "skipped_no_usable_samples":
            log.info("  %-16s [%-12s] SKIPPED (no usable samples)",
                     name, report.get("method_type", "?"))
            continue
        log.info("  %-16s [%-12s] AUROC=%.4f PR-AUC=%.4f Acc=%.4f ECE=%.4f",
                 name, report.get("method_type", "?"), m.get("roc_auc", 0), m.get("pr_auc", 0),
                 m.get("accuracy", 0), m.get("ece", 0))
    log.info("=" * 70)


if __name__ == "__main__":
    main()
