#!/usr/bin/env python3
"""
evaluate.py - Evaluate the SpatialMind UQ pipeline on cached test features.

Evaluates:
  1. The SpatialMind UQ head — loads a trained head and runs it on cached features
  2. Unsupervised reference baselines — runs estimators on raw cached data

Dataset-agnostic: reads dataset_name from the Phase 1 manifest and uses the
dataset class's eval_config + difficulty_field for per-difficulty breakdowns.

Usage:
    # Evaluate a single supervised head:
    python scripts/evaluate.py --head_path /path/to/final_model --cache_dir /path/to/cache

    # Evaluate all unsupervised baselines:
    python scripts/evaluate.py --cache_dir /path/to/cache --eval_baselines

    # OOD evaluation (head trained on StepGame, evaluate on bAbI features):
    python scripts/evaluate.py --head_path /path/to/final_model --cache_dir /path/to/babi_cache
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.numpy_compat import (
    configure_protobuf_python_implementation,
    patch_numpy_core_multiarray,
)

configure_protobuf_python_implementation()
patch_numpy_core_multiarray()

import numpy as np
import torch
from transformers import Trainer

from config import Config
from data.cached_features import CachedFeatureDataset
from data.datasets import get_dataset_cls
from models.heads import build_head
from models.wrapper import CachedFeatureModel
from scripts.metrics import (
    compute_all_metrics,
    compute_claim_metrics,
)
from utils.common import collate_claim_cached_features
from scripts.trainer import build_eval_training_arguments
from models.heads import BASELINE_REGISTRY
from utils.efficiency import (
    get_cpu_peak_memory_gb,
    get_gpu_peak_memory_gb,
    reset_gpu_peak_memory,
)
from utils.number_utils import first_number
from utils.wandb_utils import (
    finish_run,
    init_baseline_run,
    init_eval_run,
    load_baseline_run_metadata,
    load_run_metadata,
    log_eval_report,
    resolve_head_type,
    should_use_wandb,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def _read_manifest(cache_dir: str, split: str) -> dict:
    """Read manifest.json from a split directory."""
    manifest_path = Path(cache_dir) / split / "manifest.json"
    if not manifest_path.exists():
        return {}
    with open(manifest_path, "r") as f:
        return json.load(f)


def _get_base_model_name(cache_dir: str, split: str, cfg=None) -> str:
    """Read the base model name from the Phase 1 manifest, or fall back to config."""
    manifest = _read_manifest(cache_dir, split)
    model_path = manifest.get("model_path", "")
    if model_path:
        return Path(model_path).name
    if cfg is not None:
        return Path(cfg.model.pretrained_model_name_or_path).name or "unknown"
    return "unknown"


def _get_difficulty_field(cache_dir: str, split: str) -> str:
    """Detect the difficulty field name from the manifest's dataset_name."""
    manifest = _read_manifest(cache_dir, split)
    dataset_name = manifest.get("dataset_name", "")
    if dataset_name:
        try:
            cls = get_dataset_cls(dataset_name)
            return cls.difficulty_field
        except ValueError:
            pass
    return "k_hop"  # fallback


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the SpatialMind UQ pipeline")
    parser.add_argument("--head_path", type=str, default=None,
                        help="Path to trained head directory (head_config.json + head_weights.pth)")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Path to cached features from Phase 1")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--eval_baselines", action="store_true",
                        help="Also evaluate unsupervised baselines")
    parser.add_argument("--baselines", type=str, default=None,
                        help="Comma-separated baseline names (default: all)")
    parser.add_argument("--train_dataset", type=str, default=None,
                        help="Training dataset name (for OOD baseline evaluation)")
    parser.add_argument("--baseline_run_dir", type=str, default=None,
                        help="Directory containing baseline wandb_run.json (for OOD resume)")
    # Backward compat: kept but no longer needed in shell scripts
    parser.add_argument("--k_hop_values", type=str, default=None,
                        help="(Deprecated) Difficulty filter. Prefer using dataset eval_config.")
    # WandB integration
    parser.add_argument("--report_to", type=str, default="none",
                        help="Experiment tracking: none, wandb, tensorboard, or all")
    parser.add_argument("--wandb_project", type=str, default="spatialmind")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_log_dir", type=str, default=None,
                        help="Directory to save wandb run data locally")
    parser.add_argument("--save_predictions", action="store_true", default=True,
                        help="Save detailed predictions (inputs, labels, predictions) to JSON")
    parser.add_argument("--no_save_predictions", dest="save_predictions", action="store_false",
                        help="Disable saving detailed predictions")
    return parser.parse_args()


def _collect_claim_arrays(dataset, logits: np.ndarray):
    """Rebuild claim-level labels/type/difficulty arrays from dataset order."""
    labels = []
    claim_type_ids = []
    claim_difficulties = []
    for i in range(len(dataset)):
        sample = dataset[i]
        verified = sample.get("verified", [])
        claims = sample.get("claims", [])
        if verified:
            labels.extend([int(v) for v in verified])
            claim_difficulties.extend([int(sample.get("k_hop", 0))] * len(verified))
            for ci in range(len(verified)):
                c = claims[ci] if ci < len(claims) else {}
                claim_type_ids.append(int(c.get("claim_type_id", 2)))
        else:
            labels.append(int(sample["labels"].item()))
            claim_difficulties.append(int(sample.get("k_hop", 0)))
            claim_type_ids.append(2)

    labels = np.array(labels, dtype=np.int64)
    claim_type_ids = np.array(claim_type_ids, dtype=np.int64)
    claim_difficulties = np.array(claim_difficulties, dtype=np.int64)

    if logits.shape[0] != labels.shape[0]:
        min_len = min(logits.shape[0], labels.shape[0])
        logits = logits[:min_len]
        labels = labels[:min_len]
        claim_type_ids = claim_type_ids[:min_len]
        claim_difficulties = claim_difficulties[:min_len]

    return logits, labels, claim_type_ids, claim_difficulties


def evaluate_supervised_head(args, cfg, cache_dir, k_hop_values, output_dir, difficulty_field):
    """Evaluate a trained supervised head on cached features."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model_name = _get_base_model_name(cache_dir, args.split, cfg)

    # Load head config
    head_config_path = os.path.join(args.head_path, "head_config.json")
    with open(head_config_path, "r") as f:
        head_cfg = json.load(f)

    log.info("Head config: %s", head_cfg)

    # Load cached test dataset
    # Use LRU caching (preload_all=False) to avoid OOM on large test sets.
    test_ds = CachedFeatureDataset(
        cache_dir=cache_dir,
        split=args.split,
        k_hop_values=k_hop_values,
        max_samples=args.max_samples,
        preload_all=False,
        max_cached_chunks=2,  # Limit memory usage for large test sets
    )

    # Build and load head
    feature_dim = head_cfg["feature_dim"]
    num_classes = head_cfg.get("num_classes", 1)
    head = build_head(
        head_type=head_cfg["head_type"],
        feature_dim=feature_dim,
        num_classes=num_classes,
        head_dim=head_cfg.get("head_dim", 256),
        n_layers=head_cfg.get("n_layers", 2),
        n_heads=head_cfg.get("n_heads", 8),
        dropout=head_cfg.get("dropout", 0.1),
    ).to(device)

    head_weights_path = os.path.join(args.head_path, "head_weights.pth")
    head.load_state_dict(torch.load(head_weights_path, map_location="cpu"))
    head.to(device)

    # Build model wrapper
    model = CachedFeatureModel(head=head, num_classes=num_classes)

    # Trainer-based prediction
    eval_args = build_eval_training_arguments(
        cfg=cfg,
        output_dir=output_dir,
        per_device_eval_batch_size=args.batch_size,
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        data_collator=collate_claim_cached_features,
        compute_metrics=None,
    )

    # Fixed threshold=0.5 (standard practice in hallucination detection papers)
    threshold = 0.5

    log.info("Running prediction with HF Trainer.predict()...")
    reset_gpu_peak_memory(device)
    prediction_output = trainer.predict(test_ds)

    eval_metrics = prediction_output.metrics
    log.info("Trainer predict metrics: %s", eval_metrics)

    inference_runtime_s = first_number(
        eval_metrics.get("test_runtime"),
        eval_metrics.get("eval_runtime"),
    )
    inference_cpu_peak_gb = get_cpu_peak_memory_gb()
    inference_gpu_peak_gb = get_gpu_peak_memory_gb(device)
    log.info(
        "Inference efficiency: runtime=%.4fs, cpu_peak=%.3fGB, gpu_peak=%.3fGB",
        inference_runtime_s,
        inference_cpu_peak_gb,
        inference_gpu_peak_gb,
    )

    # Get logits and labels
    logits = prediction_output.predictions

    if logits.ndim > 1:
        logits = logits.squeeze(-1)

    logits, labels, claim_type_ids, claim_difficulties = _collect_claim_arrays(test_ds, logits)

    # Overall metrics
    overall_metrics = compute_all_metrics(labels, logits, threshold=threshold)
    claim_metrics = compute_claim_metrics(labels, logits, claim_type_ids, threshold=threshold)
    log.info("Overall metrics: %s", overall_metrics)

    # Per-difficulty metrics
    per_difficulty_metrics = compute_per_difficulty(
        claim_difficulties,
        labels,
        logits,
        difficulty_field,
        threshold=threshold,
    )

    # Load training results for efficiency info
    train_results_path = Path(args.head_path).resolve().parent / "train_results.json"
    train_results = None
    if train_results_path.is_file():
        with train_results_path.open("r", encoding="utf-8") as f:
            train_results = json.load(f)

    train_eff = (train_results or {}).get("efficiency", {})
    params_m = first_number(
        train_eff.get("params_m"),
        head_cfg.get("trainable_params", 0) / 1e6,
    )

    # Compute samples_per_second
    total_samples = len(labels)
    samples_per_second = 0.0
    if inference_runtime_s > 0 and total_samples > 0:
        samples_per_second = total_samples / inference_runtime_s

    efficiency = {
        "params_m": params_m,
        "flops_g": first_number(train_eff.get("flops_g")),
        "train_time_h": first_number(train_eff.get("train_time_h")),
        "epoch_s": first_number(train_eff.get("epoch_s")),
        "inference_s": inference_runtime_s,
        "samples_per_second": samples_per_second,
        "cpu_memory_gb": max(
            first_number(train_eff.get("cpu_memory_gb"), default=0),
            inference_cpu_peak_gb,
        ),
        "gpu_memory_gb": max(
            first_number(train_eff.get("gpu_memory_gb"), default=0),
            inference_gpu_peak_gb,
        ),
        "train_cpu_memory_gb": first_number(train_eff.get("cpu_memory_gb"), default=0),
        "train_gpu_memory_gb": first_number(train_eff.get("gpu_memory_gb"), default=0),
        "inference_cpu_memory_gb": inference_cpu_peak_gb,
        "inference_gpu_memory_gb": inference_gpu_peak_gb,
    }
    log.info("Evaluation efficiency metrics: %s", efficiency)

    report = {
        "method_type": "supervised",
        "head_type": head_cfg["head_type"],
        "base_model_name": base_model_name,
        "head_path": args.head_path,
        "split": args.split,
        "difficulty_field": difficulty_field,
        "k_hop_values": k_hop_values or "all",
        "total_samples": int(len(labels)),
        "num_classes": num_classes,
        "trainable_params": head_cfg.get("trainable_params", 0),
        "overall_metrics": overall_metrics,
        "claim_metrics": claim_metrics,
        "per_difficulty_metrics": per_difficulty_metrics,
        "threshold": threshold,
        "efficiency": efficiency,
        "trainer_metrics": eval_metrics,
    }

    # Save detailed predictions if enabled
    if getattr(args, "save_predictions", True):
        probs = 1 / (1 + np.exp(-logits))  # sigmoid
        preds = (probs >= threshold).astype(int)
        
        predictions_data = []
        claim_idx = 0
        for i in range(len(test_ds)):
            sample = test_ds[i]
            verified = sample.get("verified", [])
            claims = sample.get("claims", [])
            num_claims = len(verified) if verified else 1
            
            sample_entry = {
                "sample_id": i,
                "question": sample.get("question", ""),
                "generated_text": sample.get("generated_text", ""),
                "k_hop": int(sample.get("k_hop", 0)),
                "claims": []
            }
            
            for ci in range(num_claims):
                if claim_idx >= len(labels):
                    break
                claim_info = claims[ci] if ci < len(claims) else {}
                sample_entry["claims"].append({
                    "claim_text": claim_info.get("claim", ""),
                    "claim_type": claim_info.get("claim_type", "unknown"),
                    "claim_type_id": int(claim_info.get("claim_type_id", 2)),
                    "label": int(labels[claim_idx]),
                    "prediction": int(preds[claim_idx]),
                    "probability": float(probs[claim_idx]),
                    "logit": float(logits[claim_idx]),
                    "correct": int(labels[claim_idx] == preds[claim_idx]),
                })
                claim_idx += 1
            
            predictions_data.append(sample_entry)
        
        predictions_path = os.path.join(output_dir, "predictions.json")
        with open(predictions_path, "w", encoding="utf-8") as f:
            json.dump(predictions_data, f, indent=2, ensure_ascii=False)
        log.info("Detailed predictions saved to %s", predictions_path)
        report["predictions_path"] = predictions_path

    return report


def evaluate_baselines(cache_dir, split, k_hop_values, max_samples, difficulty_field, baseline_names=None, output_dir=None, save_predictions=True):
    """Evaluate unsupervised baselines on cached raw data."""
    import time

    results = {}
    base_model_name = _get_base_model_name(cache_dir, split)

    if baseline_names is None:
        baseline_names = ["random", "mcp", "perplexity", "mean_token_entropy", "ccp"]

    for name in baseline_names:
        canonical_name = UNSUPERVISED_HEAD_ALIASES.get(name, name)
        if canonical_name not in BASELINE_REGISTRY:
            log.warning("Unknown baseline '%s', skipping.", name)
            continue

        log.info("Evaluating baseline: %s", name)
        estimator = BASELINE_REGISTRY[canonical_name]()

        start_time = time.time()
        scores, labels, k_hops = estimator.estimate_batch(
            cache_dir=cache_dir,
            split=split,
            k_hop_values=k_hop_values,
            max_samples=max_samples,
        )
        inference_s = time.time() - start_time

        # Compute samples_per_second
        total_samples = len(labels)
        samples_per_second = total_samples / inference_s if inference_s > 0 else 0.0

        # Estimator outputs are uncertainty-like (higher => more hallucination).
        # Global label convention here is 1=correct, 0=hallucination, so invert.
        # We z-score normalize to convert to logit-like scores before negating.
        # This ensures the confidence_scores have a proper distribution for threshold=0.5.
        scores_mean = np.mean(scores)
        scores_std = np.std(scores)
        if scores_std < 1e-8:
            scores_std = 1.0
        confidence_scores = -(scores - scores_mean) / scores_std

        # Compute binary metrics (confidence_scores used as continuous predictions)
        overall_metrics = compute_all_metrics(labels, confidence_scores, threshold=0.5)
        log.info("  %s overall: %s", name, overall_metrics)

        # Per-difficulty
        per_difficulty_metrics = {}
        unique_k = sorted(set(k_hops))
        for k in unique_k:
            mask = k_hops == k
            if mask.sum() > 0:
                k_metrics = compute_all_metrics(labels[mask], confidence_scores[mask], threshold=0.5)
                per_difficulty_metrics[int(k)] = k_metrics
                log.info("    %s=%d: n=%d, acc=%.4f, f1=%.4f, roc_auc=%.4f",
                         difficulty_field, k, mask.sum(), k_metrics["accuracy"],
                         k_metrics["f1"], k_metrics["roc_auc"])

        result_entry = {
            "method_type": "unsupervised",
            "head_type": name,
            "base_model_name": base_model_name,
            "split": split,
            "difficulty_field": difficulty_field,
            "k_hop_values": k_hop_values or "all",
            "total_samples": total_samples,
            "overall_metrics": overall_metrics,
            "per_difficulty_metrics": per_difficulty_metrics,
            "efficiency": {
                "params_m": 0.0,
                "flops_g": 0.0,
                "train_time_h": 0.0,
                "inference_s": inference_s,
                "samples_per_second": samples_per_second,
            },
        }
        
        # Save predictions for baselines if enabled
        if save_predictions and output_dir:
            probs = 1 / (1 + np.exp(-confidence_scores))  # sigmoid(prob of label=1 correct)
            preds = (probs >= 0.5).astype(int)
            
            predictions_data = []
            for i in range(len(labels)):
                predictions_data.append({
                    "sample_id": i,
                    "k_hop": int(k_hops[i]),
                    "label": int(labels[i]),
                    "prediction": int(preds[i]),
                    "probability": float(probs[i]),
                    "score": float(scores[i]),
                    "confidence_score": float(confidence_scores[i]),
                    "correct": int(labels[i] == preds[i]),
                })
            
            baseline_dir = os.path.join(output_dir, name)
            os.makedirs(baseline_dir, exist_ok=True)
            predictions_path = os.path.join(baseline_dir, "predictions.json")
            with open(predictions_path, "w", encoding="utf-8") as f:
                json.dump(predictions_data, f, indent=2, ensure_ascii=False)
            log.info("Baseline %s predictions saved to %s", name, predictions_path)
            result_entry["predictions_path"] = predictions_path
        
        results[name] = result_entry

    return results


def compute_per_difficulty(difficulties_arr, labels, logits_or_scores, difficulty_field="k_hop", threshold=0.5):
    """Compute per-difficulty metrics from claim-level arrays."""
    per_difficulty_metrics = {}
    unique_d = sorted(set(difficulties_arr.tolist()))

    for d in unique_d:
        mask = difficulties_arr == d
        if mask.sum() > 0:
            d_metrics = compute_all_metrics(labels[mask], logits_or_scores[mask], threshold=threshold)
            per_difficulty_metrics[int(d)] = d_metrics
            log.info("  %s=%d: n=%d, acc=%.4f, f1=%.4f, pr_auc=%.4f, roc_auc=%.4f",
                     difficulty_field, d, mask.sum(), d_metrics["accuracy"],
                     d_metrics["f1"], d_metrics["pr_auc"], d_metrics["roc_auc"])

    return per_difficulty_metrics


def main():
    args = parse_args()
    cfg = Config()

    cache_dir = args.cache_dir or cfg.generation.cache_dir
    output_dir = args.output_dir or os.path.join(cfg.output.output_dir, "evaluation")
    os.makedirs(output_dir, exist_ok=True)

    # Backward compat: --k_hop_values still accepted but deprecated
    k_hop_values = None
    if args.k_hop_values:
        k_hop_values = [int(x) for x in args.k_hop_values.split(",")]

    # Detect difficulty field name from manifest
    difficulty_field = _get_difficulty_field(cache_dir, args.split)
    manifest = _read_manifest(cache_dir, args.split)
    dataset_name = manifest.get("dataset_name", "unknown")

    use_wandb = should_use_wandb(args.report_to)
    train_wandb_meta = load_run_metadata(args.head_path) if args.head_path else None
    train_dataset_name = train_wandb_meta.get("dataset_name", dataset_name) if train_wandb_meta else dataset_name

    # Initialize WandB for supervised head run (if enabled)
    wandb_run = None
    test_metric_prefix = f"test_{dataset_name}"
    is_ood = False
    if args.head_path and use_wandb:
        head_type = resolve_head_type(args.head_path, default="unknown")
        wandb_run, test_metric_prefix, is_ood = init_eval_run(
            enabled=use_wandb,
            project=args.wandb_project,
            entity=args.wandb_entity,
            head_type=head_type,
            train_dataset_name=train_dataset_name,
            eval_dataset_name=dataset_name,
            split=args.split,
            train_run_metadata=train_wandb_meta,
            log_dir=args.wandb_log_dir or output_dir,
        )

    log.info("=" * 70)
    log.info("SpatialMind: Evaluation")
    log.info("=" * 70)
    log.info("Dataset:     %s", dataset_name)
    log.info("Cache dir:   %s", cache_dir)
    log.info("Split:       %s", args.split)
    log.info("Difficulty:  %s (field=%s)", k_hop_values or "all", difficulty_field)
    log.info("Head path:   %s", args.head_path or "(none)")
    log.info("Baselines:   %s", args.eval_baselines)
    log.info("=" * 70)

    all_reports = {}

    # Evaluate supervised head
    if args.head_path:
        log.info("--- Supervised Head Evaluation ---")
        head_report = evaluate_supervised_head(
            args, cfg, cache_dir, k_hop_values, output_dir, difficulty_field
        )
        head_type = head_report["head_type"]
        all_reports[head_type] = head_report

        # Log to WandB
        if wandb_run is not None:
            log_eval_report(test_metric_prefix, head_report)

        # Save individual report
        head_report_path = os.path.join(output_dir, "evaluation_report.json")
        with open(head_report_path, "w") as f:
            json.dump(head_report, f, indent=2, default=str)
        log.info("Head evaluation report saved to %s", head_report_path)

    # Evaluate unsupervised baselines
    if args.eval_baselines:
        log.info("--- Unsupervised Baseline Evaluation ---")
        baseline_names = None
        if args.baselines:
            baseline_names = [b.strip() for b in args.baselines.split(",")]

        baseline_reports = evaluate_baselines(
            cache_dir=cache_dir,
            split=args.split,
            k_hop_values=k_hop_values,
            max_samples=args.max_samples,
            difficulty_field=difficulty_field,
            baseline_names=baseline_names,
            output_dir=output_dir,
            save_predictions=args.save_predictions,
        )
        all_reports.update(baseline_reports)

        # Log baselines to WandB runs
        if use_wandb:
            # Finish supervised run before creating baseline runs
            if wandb_run is not None:
                finish_run()
                wandb_run = None

            # Determine train dataset for baselines
            # For ID: train_dataset == eval_dataset (dataset_name)
            # For OOD: use --train_dataset or --baseline_run_dir to find original train dataset
            baseline_train_dataset = args.train_dataset or dataset_name

            for name, report in baseline_reports.items():
                # Try to load existing baseline run metadata for OOD resume
                baseline_meta = None
                if args.baseline_run_dir:
                    baseline_meta = load_baseline_run_metadata(args.baseline_run_dir, name)

                baseline_run, baseline_prefix, _ = init_baseline_run(
                    enabled=use_wandb,
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    baseline_name=name,
                    train_dataset_name=baseline_train_dataset,
                    eval_dataset_name=dataset_name,
                    split=args.split,
                    baseline_run_metadata=baseline_meta,
                    output_dir=output_dir,
                    log_dir=args.wandb_log_dir or output_dir,
                )
                if baseline_run is not None:
                    log_eval_report(baseline_prefix, report)
                    finish_run()

        # Save individual baseline reports
        for name, report in baseline_reports.items():
            baseline_dir = os.path.join(output_dir, name)
            os.makedirs(baseline_dir, exist_ok=True)
            baseline_report_path = os.path.join(baseline_dir, "evaluation_report.json")
            with open(baseline_report_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            log.info("Baseline report saved: %s", baseline_report_path)

    # Print summary
    log.info("=" * 70)
    log.info("EVALUATION SUMMARY")
    log.info("=" * 70)
    for method_name, report in all_reports.items():
        m = report["overall_metrics"]
        method_type = report.get("method_type", "unknown")
        log.info(
            "  %-15s [%-12s] acc=%.4f prec=%.4f rec=%.4f f1=%.4f roc=%.4f pr=%.4f ece=%.4f",
            method_name, method_type,
            m.get("accuracy", 0), m.get("precision", 0), m.get("recall", 0),
            m.get("f1", 0), m.get("roc_auc", 0), m.get("pr_auc", 0),
            m.get("ece", 0),
        )
    log.info("=" * 70)

    # Save combined report
    combined_path = os.path.join(output_dir, "combined_evaluation.json")
    with open(combined_path, "w") as f:
        json.dump(all_reports, f, indent=2, default=str)
    log.info("Combined evaluation saved to %s", combined_path)

    # Finish supervised WandB run
    if wandb_run is not None:
        finish_run()
if __name__ == "__main__":
    main()
