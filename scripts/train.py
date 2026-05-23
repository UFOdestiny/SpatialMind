#!/usr/bin/env python3
"""
train.py - Phase 2: Train binary UQ head on cached features.

Loads pre-extracted features from Phase 1 (no LLM needed), trains a
lightweight binary classification head with BCEWithLogitsLoss.

Usage:
    python scripts/train.py --head_type uq --cache_dir /path/to/cached_features
    python scripts/train.py --head_type uq --num_epochs 50 --batch_size 32
"""

import os
import sys
import json
import time
import math
import logging
import argparse
import copy
from pathlib import Path
from typing import Optional, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.numpy_compat import (
    configure_protobuf_python_implementation,
    patch_numpy_core_multiarray,
)

configure_protobuf_python_implementation()
patch_numpy_core_multiarray()

import torch
from transformers import EarlyStoppingCallback, TrainerCallback

from config import Config
from data.cached_features import CachedFeatureDataset
from models.heads import build_head
from models.wrapper import CachedFeatureModel
from utils.common import collate_claim_cached_features, make_binary_compute_metrics
from scripts.trainer import HeadTrainer, build_train_training_arguments
from utils.efficiency import (
    get_cpu_peak_memory_gb,
    get_gpu_peak_memory_gb,
    reset_gpu_peak_memory,
)
from utils.wandb_utils import (
    build_config_dict,
    finish_run,
    init_train_run,
    log_metrics,
    should_use_wandb,
    strip_wandb_from_report_to,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Suppress noisy Accelerate INFO about dataset length (not relevant for single-GPU training)
logging.getLogger("accelerate.accelerator").setLevel(logging.WARNING)


class StepHeartbeatCallback(TrainerCallback):
    """Emit trainer metrics into standard logs at each logging event."""

    def __init__(self, dataset_name: str, enable_wandb: bool):
        self.enable_wandb = enable_wandb
        self.train_prefix = f"train_{dataset_name}"
        self.valid_prefix = f"valid_{dataset_name}"

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = args, control, kwargs
        if not logs:
            return

        step = int(getattr(state, "global_step", 0))
        epoch = logs.get("epoch", state.epoch)
        if isinstance(epoch, (int, float)):
            epoch_text = f"{float(epoch):.4f}"
        else:
            epoch_text = str(epoch)

        metrics = {k: v for k, v in logs.items() if k != "total_flos"}
        log.info(
            "trainer_step_log step=%d epoch=%s metrics=%s",
            step,
            epoch_text,
            metrics,
        )

        if not self.enable_wandb:
            return

        valid_metrics = {k: v for k, v in metrics.items() if k.startswith("eval_")}
        train_metrics = {k: v for k, v in metrics.items() if not k.startswith("eval_")}
        log_metrics(self.train_prefix, train_metrics, step=step)
        log_metrics(self.valid_prefix, valid_metrics, step=step)


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2: Train binary UQ head")
    parser.add_argument("--head_type", type=str, default=None,
                        help="Head type to train (use: uq)")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for training artifacts")
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--k_hop_values", type=str, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--loss_type",
        type=str,
        default=None,
        choices=["bce", "balanced_bce", "focal"],
        help="Binary loss type for training.",
    )
    parser.add_argument(
        "--loss_pos_weight",
        type=float,
        default=None,
        help="Positive class weight for balanced_bce.",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=None,
        help="Focal loss gamma when --loss_type focal.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        help="Experiment tracking: none, wandb, tensorboard, or all",
    )
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument(
        "--wandb_log_dir",
        type=str,
        default=None,
        help="Directory to save wandb run data locally",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint dir, or 'auto' to find latest checkpoint in output_dir",
    )
    return parser.parse_args()


def apply_overrides(cfg: Config, args):
    """Apply CLI args and environment variable overrides to config."""
    env_map = {
        "CACHE_DIR": ("generation", "cache_dir"),
        "OUTPUT_DIR": ("output", "output_dir"),
        "NUM_EPOCHS": ("training", "num_epochs"),
        "BATCH_SIZE": ("training", "per_device_train_batch_size"),
        "LEARNING_RATE": ("training", "learning_rate"),
        "WARMUP_STEPS": ("training", "warmup_steps"),
        "HEAD_TYPE": ("head", "head_type"),
        "DISABLE_TQDM": ("training", "disable_tqdm"),
        "LOGGING_STRATEGY": ("training", "logging_strategy"),
        "LOGGING_STEPS": ("training", "logging_steps"),
        "DATALOADER_NUM_WORKERS": ("training", "dataloader_num_workers"),
        "LOSS_TYPE": ("training", "loss_type"),
        "LOSS_POS_WEIGHT": ("training", "loss_pos_weight"),
        "FOCAL_GAMMA": ("training", "focal_gamma"),
        "REPORT_TO": ("training", "report_to"),
        "WANDB_PROJECT": ("training", "wandb_project"),
        "WANDB_ENTITY": ("training", "wandb_entity"),
    }
    for env_key, (section, field) in env_map.items():
        val = os.environ.get(env_key)
        if val is not None:
            sub_cfg = getattr(cfg, section)
            current = getattr(sub_cfg, field)
            if isinstance(current, bool):
                parsed = val.strip().lower() in {"1", "true", "yes", "on"}
            elif isinstance(current, int):
                parsed = int(val)
            elif isinstance(current, float):
                parsed = float(val)
            else:
                parsed = val
            setattr(sub_cfg, field, parsed)

    if os.environ.get("BATCH_SIZE"):
        cfg.training.per_device_eval_batch_size = int(os.environ["BATCH_SIZE"])

    k_hop_env = os.environ.get("K_HOP_VALUES")
    if k_hop_env:
        cfg.dataset.k_hop_values = [int(x) for x in k_hop_env.split(",")]

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
    if args.warmup_steps is not None:
        cfg.training.warmup_steps = args.warmup_steps
    if args.k_hop_values:
        cfg.dataset.k_hop_values = [int(x) for x in args.k_hop_values.split(",")]
    if args.max_train_samples:
        cfg.dataset.max_train_samples = args.max_train_samples
    if args.seed is not None:
        cfg.training.seed = args.seed
    if args.loss_type:
        cfg.training.loss_type = args.loss_type
    if args.loss_pos_weight is not None:
        cfg.training.loss_pos_weight = args.loss_pos_weight
    if args.focal_gamma is not None:
        cfg.training.focal_gamma = args.focal_gamma
    # WandB overrides
    if args.report_to:
        cfg.training.report_to = args.report_to
    if args.wandb_project:
        cfg.training.wandb_project = args.wandb_project
    if args.wandb_entity:
        cfg.training.wandb_entity = args.wandb_entity

    return cfg


def list_checkpoints(output_dir: str) -> list[str]:
    """List checkpoint directories sorted by step (descending)."""
    if not os.path.isdir(output_dir):
        return []

    checkpoint_dirs = []
    for name in os.listdir(output_dir):
        if not name.startswith("checkpoint-"):
            continue
        try:
            step = int(name.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        path = os.path.join(output_dir, name)
        if os.path.isdir(path):
            checkpoint_dirs.append((step, path))

    checkpoint_dirs.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in checkpoint_dirs]


def _build_model_from_head_config(head_cfg: dict) -> CachedFeatureModel:
    """Rebuild a checkpoint model from persisted head_config.json."""
    head = build_head(
        head_type=head_cfg["head_type"],
        feature_dim=head_cfg["feature_dim"],
        num_classes=head_cfg["num_classes"],
        head_dim=head_cfg.get("head_dim", 256),
        n_layers=head_cfg.get("n_layers", 2),
        n_heads=head_cfg.get("n_heads", 8),
        dropout=head_cfg.get("dropout", 0.1),
    )
    return CachedFeatureModel(head=head, num_classes=head_cfg["num_classes"])


def ensure_hf_checkpoint_compatible(checkpoint_dir: str) -> bool:
    """Ensure checkpoint contains HF-required model file for resume.

    Legacy checkpoints from our custom trainer only stored `head_weights.pth`.
    HF `Trainer.train(resume_from_checkpoint=...)` requires `pytorch_model.bin`
    (or safetensors). We synthesize `pytorch_model.bin` from legacy artifacts.
    """
    weights_file = os.path.join(checkpoint_dir, "pytorch_model.bin")
    safe_weights_file = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.isfile(weights_file) or os.path.isfile(safe_weights_file):
        return True

    legacy_head_file = os.path.join(checkpoint_dir, "head_weights.pth")
    head_config_file = os.path.join(checkpoint_dir, "head_config.json")
    if not os.path.isfile(legacy_head_file) or not os.path.isfile(head_config_file):
        return False

    fe_file = os.path.join(checkpoint_dir, "fe_weights.pth")
    try:
        with open(head_config_file, "r", encoding="utf-8") as f:
            head_cfg = json.load(f)
        model = _build_model_from_head_config(head_cfg)
        model.head.load_state_dict(torch.load(legacy_head_file, map_location="cpu"))
        if hasattr(model, "feature_extractor") and os.path.isfile(fe_file):
            model.feature_extractor.load_state_dict(torch.load(fe_file, map_location="cpu"))
        torch.save(model.state_dict(), weights_file)
        log.info("Upgraded legacy checkpoint for HF resume: %s", checkpoint_dir)
        return True
    except (FileNotFoundError, RuntimeError, OSError, ValueError) as exc:
        log.warning("Failed to upgrade legacy checkpoint %s: %s", checkpoint_dir, exc)
        return False


def find_latest_resumable_checkpoint(output_dir: str) -> Optional[str]:
    """Find latest checkpoint that can be resumed by HF Trainer."""
    checkpoints = list_checkpoints(output_dir)
    if not checkpoints:
        log.info("No checkpoints found in %s", output_dir)
        return None
    
    log.info("Found %d checkpoint(s) in %s", len(checkpoints), output_dir)
    for checkpoint_dir in checkpoints:
        if ensure_hf_checkpoint_compatible(checkpoint_dir):
            # Validate checkpoint integrity
            weights_file = os.path.join(checkpoint_dir, "pytorch_model.bin")
            safe_weights = os.path.join(checkpoint_dir, "model.safetensors")
            trainer_state = os.path.join(checkpoint_dir, "trainer_state.json")
            
            if os.path.isfile(trainer_state):
                with open(trainer_state, "r") as f:
                    state = json.load(f)
                log.info(
                    "  Valid checkpoint: %s (step=%d, epoch=%.2f)",
                    os.path.basename(checkpoint_dir),
                    state.get("global_step", -1),
                    state.get("epoch", -1),
                )
            else:
                log.info("  Valid checkpoint: %s (no trainer_state.json)", os.path.basename(checkpoint_dir))
            
            return checkpoint_dir
        else:
            log.warning("  Skipping incompatible checkpoint: %s", os.path.basename(checkpoint_dir))
    
    return None


def train_single_head(
    cfg: Config,
    train_ds: "CachedFeatureDataset",
    val_ds: "CachedFeatureDataset",
    feature_dim: int,
    wandb_log_dir: Optional[str] = None,
    resume_from_checkpoint: Optional[str] = None,
):
    """Train a single head using pre-loaded datasets.
    
    This is the core training function that can be called multiple times
    with different head types but the same datasets (memory-efficient).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = cfg.output.output_dir
    os.makedirs(output_dir, exist_ok=True)
    cache_dir = cfg.generation.cache_dir

    # ======================== Initialize WandB (if enabled) ========================
    use_wandb = should_use_wandb(cfg.training.report_to)
    wandb_config = build_config_dict(cfg)
    wandb_config["output_dir"] = output_dir
    wandb_config["cache_dir"] = cache_dir
    wandb_run = init_train_run(
        enabled=use_wandb,
        project=cfg.training.wandb_project,
        entity=cfg.training.wandb_entity,
        head_type=cfg.head.head_type,
        dataset_name=cfg.dataset.dataset_name,
        config=wandb_config,
        tags=[cfg.head.head_type, cfg.dataset.dataset_name],
        log_dir=wandb_log_dir or output_dir,
        output_dir=output_dir,
    )

    log.info("Feature dimension: %d (from cached data)", feature_dim)

    # Log claim label distribution (memory-efficient: loads one chunk at a time)
    n_total, n_correct, n_hall = train_ds.get_claim_label_stats()
    log.info(
        "Train claims: %d total, %d correct (%.1f%%), %d hallucinations (%.1f%%)",
        n_total,
        n_correct,
        100 * n_correct / max(n_total, 1),
        n_hall,
        100 * n_hall / max(n_total, 1),
    )

    # ======================== Build head ========================
    num_classes = cfg.head.num_classes  # default 1 (binary)
    log.info("Building head: type=%s, num_classes=%d", cfg.head.head_type, num_classes)
    head = build_head(
        head_type=cfg.head.head_type,
        feature_dim=feature_dim,
        num_classes=num_classes,
        head_dim=cfg.head.head_dim,
        n_layers=cfg.head.n_layers,
        n_heads=cfg.head.n_heads,
        dropout=cfg.head.dropout,
    ).to(device)

    trainable_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    log.info("Trainable parameters: %s", f"{trainable_params:,}")

    # ======================== Build model ========================
    loss_pos_weight = float(cfg.training.loss_pos_weight)
    if cfg.training.loss_type == "balanced_bce" and loss_pos_weight <= 0:
        loss_pos_weight = float(n_hall / max(n_correct, 1))
    cfg.training.loss_pos_weight = loss_pos_weight

    model = CachedFeatureModel(
        head=head,
        num_classes=num_classes,
        loss_type=cfg.training.loss_type,
        pos_weight=loss_pos_weight,
        focal_gamma=cfg.training.focal_gamma,
    )

    # ======================== Head config (for saving) ========================
    head_config = {
        "head_type": cfg.head.head_type,
        "feature_dim": feature_dim,
        "num_classes": num_classes,
        "head_dim": cfg.head.head_dim,
        "n_layers": cfg.head.n_layers,
        "n_heads": cfg.head.n_heads,
        "dropout": cfg.head.dropout,
        "cache_dir": cache_dir,
        "trainable_params": trainable_params,
        "loss_type": cfg.training.loss_type,
        "loss_pos_weight": loss_pos_weight,
        "focal_gamma": cfg.training.focal_gamma,
    }

    # ======================== Training Arguments ========================
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    effective_batch = (
        cfg.training.per_device_train_batch_size
        * cfg.training.gradient_accumulation_steps
        * world_size
    )
    steps_per_epoch = max(1, math.ceil(len(train_ds) / max(1, effective_batch)))
    total_steps = int(cfg.training.num_epochs * steps_per_epoch)
    warmup_steps = cfg.training.warmup_steps
    if warmup_steps <= 0 and cfg.training.warmup_ratio > 0:
        warmup_steps = int(total_steps * cfg.training.warmup_ratio)
    log.info(
        "Scheduler setup: total_steps=%d, warmup_steps=%d",
        total_steps,
        warmup_steps,
    )

    training_args = build_train_training_arguments(
        cfg=cfg,
        output_dir=output_dir,
        warmup_steps=warmup_steps,
        report_to_override=(
            strip_wandb_from_report_to(cfg.training.report_to)
            if use_wandb
            else None
        ),
    )
    log.info(
        "Trainer logging config: strategy=%s, steps=%s, disable_tqdm=%s, TQDM_DISABLE=%s",
        training_args.logging_strategy,
        training_args.logging_steps,
        training_args.disable_tqdm,
        os.environ.get("TQDM_DISABLE", "<unset>"),
    )

    # ======================== Create Trainer ========================
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=cfg.training.early_stopping_patience
        ),
        StepHeartbeatCallback(
            dataset_name=cfg.dataset.dataset_name,
            enable_wandb=wandb_run is not None,
        ),
    ]
    trainer = HeadTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_claim_cached_features,
        compute_metrics=make_binary_compute_metrics(),
        callbacks=callbacks,
    )
    trainer._head_config = head_config

    # Persist a copy of training arguments for reproducibility with our custom save().
    training_args_path = os.path.join(output_dir, "training_args.json")
    with open(training_args_path, "w", encoding="utf-8") as f:
        json.dump(training_args.to_dict(), f, indent=2, default=str)
    log.info("Training arguments saved to %s", training_args_path)

    # ======================== Resolve checkpoint for resume ========================
    actual_checkpoint = None
    if resume_from_checkpoint:
        if resume_from_checkpoint.lower() == "auto":
            actual_checkpoint = find_latest_resumable_checkpoint(output_dir)
            if actual_checkpoint:
                log.info("Auto-detected checkpoint: %s", actual_checkpoint)
            else:
                log.info("No checkpoint found in %s, starting from scratch", output_dir)
        elif os.path.isdir(resume_from_checkpoint):
            if ensure_hf_checkpoint_compatible(resume_from_checkpoint):
                actual_checkpoint = resume_from_checkpoint
                log.info("Resuming from specified checkpoint: %s", actual_checkpoint)
            else:
                log.warning(
                    "Specified checkpoint is not HF-compatible and cannot be upgraded: %s. "
                    "Starting from scratch.",
                    resume_from_checkpoint,
                )
        else:
            log.warning("Checkpoint path not found: %s, starting from scratch", resume_from_checkpoint)

    # ======================== Train ========================
    if actual_checkpoint:
        log.info("Resuming training from checkpoint: %s", actual_checkpoint)
    else:
        log.info("Starting training from scratch...")
    reset_gpu_peak_memory(device)
    train_result = trainer.train(resume_from_checkpoint=actual_checkpoint)
    train_cpu_peak_gb = get_cpu_peak_memory_gb()
    train_gpu_peak_gb = get_gpu_peak_memory_gb(device)

    train_metrics = train_result.metrics
    if "total_flos" not in train_metrics:
        train_metrics["total_flos"] = trainer.state.total_flos
    log.info("Training metrics: %s", train_metrics)

    log_history = trainer.state.log_history

    # ======================== Final evaluation ========================
    eval_metrics = trainer.evaluate()
    log.info("Final eval metrics: %s", eval_metrics)
    if wandb_run is not None:
        current_step = int(getattr(trainer.state, "global_step", 0))
        log_metrics(
            f"train_{cfg.dataset.dataset_name}",
            train_metrics,
            step=current_step,
        )
        log_metrics(
            f"valid_{cfg.dataset.dataset_name}",
            eval_metrics,
            step=current_step,
        )

    train_runtime_s = float(train_metrics.get("train_runtime", 0.0) or 0.0)
    epochs_completed = float(train_metrics.get("epoch", cfg.training.num_epochs) or 0.0)
    total_flos = float(train_metrics.get("total_flos", 0.0) or 0.0)
    flops_source = "trainer"
    if total_flos <= 0.0:
        estimated_flos = (
            6.0
            * float(trainable_params)
            * float(len(train_ds))
            * max(epochs_completed, 0.0)
        )
        if estimated_flos > 0.0:
            total_flos = estimated_flos
            flops_source = "estimated_head_only"

    efficiency = {
        "params_m": trainable_params / 1e6,
        "flops_g": total_flos / 1e9 if total_flos else 0.0,
        "flops_source": flops_source,
        "train_time_h": train_runtime_s / 3600 if train_runtime_s else 0.0,
        "epoch_s": (train_runtime_s / epochs_completed) if epochs_completed else 0.0,
        "inference_s": float(eval_metrics.get("eval_runtime", 0.0) or 0.0),
        "cpu_memory_gb": train_cpu_peak_gb,
        "gpu_memory_gb": train_gpu_peak_gb,
        "epochs_completed": epochs_completed,
    }
    log.info("Efficiency metrics: %s", efficiency)

    # ======================== Save best model ========================
    if cfg.output.save_final_model:
        final_dir = os.path.join(output_dir, cfg.output.final_model_subdir)
        trainer._save(output_dir=final_dir)
        log.info("Best model saved to %s", final_dir)

    # ======================== Save consolidated results ========================
    results = {
        "head_type": cfg.head.head_type,
        "num_classes": num_classes,
        "cache_dir": cache_dir,
        "trainable_params": trainable_params,
        "k_hop_values": cfg.dataset.k_hop_values,
        "train_samples": len(train_ds),
        "eval_samples": len(val_ds),
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "efficiency": efficiency,
        "log_history": log_history,
        "head_config": head_config,
    }
    results_path = os.path.join(output_dir, "train_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Results saved to %s", results_path)

    # ======================== Finish WandB ========================
    if wandb_run is not None:
        finish_run()

    # Clean up model/trainer to free GPU memory before next head
    del trainer, model, head
    torch.cuda.empty_cache()

    return results


def train(
    cfg: Config,
    wandb_log_dir: Optional[str] = None,
    resume_from_checkpoint: Optional[str] = None,
):
    """Train the SpatialMind UQ head."""
    cache_dir = cfg.generation.cache_dir
    k_hop_values = cfg.dataset.k_hop_values if cfg.dataset.k_hop_values else None
    
    log.info("Loading cached feature datasets from %s", cache_dir)
    
    # For single-head mode, use LRU cache to save memory
    train_ds = CachedFeatureDataset(
        cache_dir=cache_dir,
        split="train",
        k_hop_values=k_hop_values,
        max_samples=cfg.dataset.max_train_samples,
        preload_all=False,
    )
    val_ds = CachedFeatureDataset(
        cache_dir=cache_dir,
        split="validation",
        k_hop_values=k_hop_values,
        max_samples=cfg.dataset.max_eval_samples,
        preload_all=False,
    )
    
    if len(train_ds) == 0:
        raise ValueError(
            "No training samples remain after pending-filtering. "
            "Likely all claim/sample labels are still -1. "
            "Run scripts/judge.py on the cache (or rerun Phase 1.5) before training."
        )
    if len(val_ds) == 0:
        raise ValueError(
            "No validation samples remain after pending-filtering. "
            "Likely all claim/sample labels are still -1. "
            "Run scripts/judge.py on the cache (or rerun Phase 1.5) before training."
        )
    
    sample = train_ds[0]
    feature_dim = sample["features"].shape[-1]
    
    return train_single_head(
        cfg=cfg,
        train_ds=train_ds,
        val_ds=val_ds,
        feature_dim=feature_dim,
        wandb_log_dir=wandb_log_dir,
        resume_from_checkpoint=resume_from_checkpoint,
    )


def main():
    args = parse_args()
    cfg = Config()
    cfg = apply_overrides(cfg, args)

    if cfg.head.head_type != "uq":
        raise ValueError("SpatialMind supports only --head_type uq.")

    log.info("=" * 70)
    log.info("SpatialMind Phase 2: Binary UQ Head Training")
    log.info("=" * 70)
    log.info("Head type:    %s", cfg.head.head_type)
    log.info("Cache dir:    %s", cfg.generation.cache_dir)
    log.info("K-hop values: %s", cfg.dataset.k_hop_values)
    log.info("Epochs:       %s", cfg.training.num_epochs)
    log.info("Batch size:   %s", cfg.training.per_device_train_batch_size)
    log.info("LR:           %s", cfg.training.learning_rate)
    log.info("Loss:         %s (pos_weight=%.4f, gamma=%.2f)",
             cfg.training.loss_type, cfg.training.loss_pos_weight, cfg.training.focal_gamma)
    log.info("Output:       %s", cfg.output.output_dir)
    if args.resume_from_checkpoint:
        log.info("Resume:       %s", args.resume_from_checkpoint)
    log.info("=" * 70)

    start_time = time.time()
    results = train(
        cfg,
        wandb_log_dir=args.wandb_log_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    elapsed = time.time() - start_time

    log.info("Training completed in %.1f minutes", elapsed / 60)
    em = results["eval_metrics"]
    log.info(
        "Eval: accuracy=%.4f, f1=%.4f, pr_auc=%.4f, roc_auc=%.4f, ece=%.4f",
        em.get("eval_accuracy", 0),
        em.get("eval_f1", 0),
        em.get("eval_pr_auc", 0),
        em.get("eval_roc_auc", 0),
        em.get("eval_ece", 0),
    )


if __name__ == "__main__":
    main()
