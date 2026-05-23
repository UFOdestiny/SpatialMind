#!/usr/bin/env python3
"""
wandb_utils.py - WandB integration utilities for SpatialMind.

This module provides decoupled WandB support that can be enabled/disabled
via config.py without modifying training code.

Naming Convention:
    - Run names: {head_type}_{dataset_name} (e.g., "uq_stepgame", "random_stepgame")
    - Metric prefixes:
        - train_{dataset}/    : training metrics
        - valid_{dataset}/    : validation metrics
        - test_{dataset}/     : in-distribution test metrics
        - OOD_test_{dataset}/ : out-of-distribution test metrics

Usage:
    1. Set report_to="wandb" in config.py (or via CLI)
    2. Set wandb_project, wandb_entity as needed
    3. Call init_run() before training/evaluation

Environment variables:
    WANDB_API_KEY: Your WandB API key (required for first use)
    WANDB_DISABLED: Set to "true" to force disable WandB
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

log = logging.getLogger(__name__)

# Global flag to track if wandb is available
_WANDB_AVAILABLE: Optional[bool] = None


def _iter_report_to_values(report_to: Any) -> Iterable[str]:
    """Normalize report_to into lowercase tokens."""
    if report_to is None:
        return []
    if isinstance(report_to, str):
        raw = report_to.strip()
        if not raw:
            return []
        if "," in raw:
            return [x.strip().lower() for x in raw.split(",") if x.strip()]
        return [raw.lower()]
    if isinstance(report_to, (list, tuple, set)):
        values = []
        for item in report_to:
            text = str(item).strip().lower()
            if text:
                values.append(text)
        return values
    text = str(report_to).strip().lower()
    return [text] if text else []


def is_wandb_available() -> bool:
    """Check if wandb package is installed."""
    global _WANDB_AVAILABLE
    if _WANDB_AVAILABLE is None:
        try:
            import wandb  # noqa: F401
            _WANDB_AVAILABLE = True
        except ImportError:
            _WANDB_AVAILABLE = False
    return _WANDB_AVAILABLE


def should_use_wandb(report_to: Any) -> bool:
    """Determine if WandB should be used based on config and environment."""
    if os.environ.get("WANDB_DISABLED", "").lower() == "true":
        return False

    report_to_values = list(_iter_report_to_values(report_to))
    if "wandb" not in report_to_values and "all" not in report_to_values:
        return False

    if not is_wandb_available():
        log.warning(
            "report_to includes 'wandb' but wandb package not installed. "
            "Install with: pip install wandb"
        )
        return False

    return True


def strip_wandb_from_report_to(report_to: Any):
    """Return Trainer report_to with wandb removed.

    We use custom WandB logging, so the HF WandB callback should stay disabled
    to avoid duplicate logs and step conflicts.
    """
    if report_to is None:
        return None
    if isinstance(report_to, str):
        raw = report_to.strip()
        lower = raw.lower()
        if lower in {"", "none", "wandb"}:
            return "none"
        if lower == "all":
            return ["tensorboard"]
        if "," in raw:
            items = [x.strip() for x in raw.split(",") if x.strip()]
            items = [x for x in items if x.lower() != "wandb"]
            if not items:
                return "none"
            return items if len(items) > 1 else items[0]
        return raw
    if isinstance(report_to, (list, tuple, set)):
        items = [str(x) for x in report_to if str(x).strip().lower() != "wandb"]
        if not items:
            return "none"
        return items
    return report_to


# =============================================================================
# Run Name Utilities
# =============================================================================


def make_run_name(head_type: str, dataset_name: str) -> str:
    """Generate consistent run name: {head_type}_{dataset_name}.
    
    Normalizes dataset_name to lowercase for consistent naming.
    """
    return f"{head_type}_{dataset_name.lower()}"


def make_metric_prefix(phase: str, dataset_name: str, is_ood: bool = False) -> str:
    """Generate metric prefix based on phase and OOD status.

    Args:
        phase: "train", "valid", or "test"
        dataset_name: Name of the dataset (will be normalized to lowercase)
        is_ood: Whether this is an OOD evaluation

    Returns:
        Metric prefix like "train_stepgame", "OOD_test_babi", etc.
    """
    dataset_lower = dataset_name.lower()
    if phase == "test" and is_ood:
        return f"OOD_test_{dataset_lower}"
    return f"{phase}_{dataset_lower}"


# =============================================================================
# Core WandB Operations
# =============================================================================


def init_run(
    *,
    project: str,
    entity: Optional[str] = None,
    run_name: str,
    group: Optional[str] = None,
    job_type: Optional[str] = None,
    run_id: Optional[str] = None,
    resume: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[list] = None,
    notes: Optional[str] = None,
    log_dir: Optional[str] = None,
) -> Optional[Any]:
    """Initialize or resume a WandB run.

    Args:
        project: WandB project name (e.g., "spatialmind")
        entity: WandB team/user name
        run_name: Display name for the run
        group: Run group for organizing related runs
        job_type: Job type label (e.g., "train", "test", "ood_test")
        run_id: Existing run id for resuming
        resume: Resume policy ("allow", "must", "never")
        config: Hyperparameters to log
        tags: List of tags
        notes: Free-form notes
        log_dir: Local directory for wandb data

    Returns:
        wandb.Run object if successful, None otherwise
    """
    if not is_wandb_available():
        log.warning("wandb not available, skipping init")
        return None

    import wandb

    init_kwargs = {
        "project": project,
        "name": run_name,
        "config": config or {},
    }

    if entity:
        init_kwargs["entity"] = entity
    if group:
        init_kwargs["group"] = group
    if job_type:
        init_kwargs["job_type"] = job_type
    if run_id:
        init_kwargs["id"] = run_id
    if resume:
        init_kwargs["resume"] = resume
    if tags:
        init_kwargs["tags"] = tags
    if notes:
        init_kwargs["notes"] = notes
    if log_dir:
        init_kwargs["dir"] = log_dir

    try:
        run = wandb.init(**init_kwargs)
        log.info(
            "WandB initialized: project=%s, entity=%s, run=%s",
            project,
            entity or "(default)",
            run.name if run else "N/A",
        )
        return run
    except Exception as e:
        log.error("Failed to initialize WandB: %s", e)
        return None


def log_metrics(
    prefix: str,
    metrics: Dict[str, Any],
    step: Optional[int] = None,
    normalize_names: bool = True,
) -> None:
    """Log metrics under a given prefix.

    Args:
        prefix: Metric prefix (e.g., "train_stepgame")
        metrics: Dictionary of metric name -> value
        step: Optional step number (logged as {prefix}/step)
        normalize_names: If True, strip "eval_"/"train_" prefixes from names
    """
    if not metrics or not is_wandb_available():
        return

    import wandb

    if wandb.run is None:
        return

    payload = {}
    for key, value in metrics.items():
        if not isinstance(value, (int, float)):
            continue
        metric_name = _normalize_metric_name(str(key)) if normalize_names else str(key)
        payload[f"{prefix}/{metric_name}"] = float(value)

    if not payload:
        return

    if step is not None:
        payload[f"{prefix}/step"] = int(step)

    wandb.log(payload)


def log_eval_report(
    prefix: str,
    report: Dict[str, Any],
    step: Optional[int] = None,
) -> None:
    """Log evaluation report metrics (overall + efficiency).

    Args:
        prefix: Metric prefix (e.g., "test_stepgame", "OOD_test_babi")
        report: Evaluation report dict with "overall_metrics" and "efficiency"
        step: Optional step number
    """
    if not report:
        return

    overall = report.get("overall_metrics", {})
    efficiency = report.get("efficiency", {})
    total_samples = report.get("total_samples", 0)

    # Compute samples_per_second if missing but we have inference_s
    inference_s = efficiency.get("inference_s", 0)
    samples_per_second = efficiency.get("samples_per_second", 0)
    if samples_per_second == 0 and inference_s > 0 and total_samples > 0:
        samples_per_second = total_samples / inference_s

    metrics = {
        "accuracy": overall.get("accuracy", 0),
        "precision": overall.get("precision", 0),
        "recall": overall.get("recall", 0),
        "f1": overall.get("f1", 0),
        "roc_auc": overall.get("roc_auc", 0),
        "pr_auc": overall.get("pr_auc", 0),
        "ece": overall.get("ece", 0),
        "inference_time_s": inference_s,
        "samples_per_second": samples_per_second,
    }

    log_metrics(prefix, metrics, step=step, normalize_names=False)

    # Also log to summary
    if not is_wandb_available():
        return

    import wandb

    if wandb.run is None:
        return

    wandb.run.summary[f"{prefix}/f1"] = overall.get("f1", 0)
    wandb.run.summary[f"{prefix}/pr_auc"] = overall.get("pr_auc", 0)
    wandb.run.summary[f"{prefix}/ece"] = overall.get("ece", 0)
    if samples_per_second > 0:
        wandb.run.summary[f"{prefix}/samples_per_second"] = samples_per_second


def log_summary(metrics: Dict[str, Any]) -> None:
    """Log summary metrics to current run."""
    if not is_wandb_available():
        return

    import wandb

    if wandb.run is not None:
        for key, value in metrics.items():
            wandb.run.summary[key] = value
        log.info("Logged %d summary metrics to WandB", len(metrics))


def finish_run() -> None:
    """Finish current WandB run gracefully."""
    if not is_wandb_available():
        return

    import wandb

    if wandb.run is not None:
        wandb.finish()
        log.info("WandB run finished")


# =============================================================================
# Metadata Persistence (for resuming runs in evaluation)
# =============================================================================


def save_run_metadata(output_dir: str, metadata: Dict[str, Any]) -> Optional[str]:
    """Persist WandB run identity so evaluation can resume the train run."""
    if not output_dir:
        return None
    meta_path = Path(output_dir) / "wandb_run.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    log.info("WandB run metadata saved to %s", meta_path)
    return str(meta_path)


def load_run_metadata(head_path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Load run metadata saved during training (if available)."""
    if not head_path:
        return None
    meta_path = Path(head_path).resolve().parent / "wandb_run.json"
    if not meta_path.exists():
        return None
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        log.warning("Failed to read WandB metadata from %s: %s", meta_path, e)
        return None


def resolve_head_type(head_path: Optional[str], default: str = "unknown") -> str:
    """Resolve head type from `<head_path>/head_config.json` if available."""
    if not head_path:
        return default
    head_config_path = Path(head_path) / "head_config.json"
    if not head_config_path.exists():
        return default
    try:
        with head_config_path.open("r", encoding="utf-8") as f:
            head_cfg = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        log.warning("Failed to parse %s: %s", head_config_path, e)
        return default
    return str(head_cfg.get("head_type", default))


# =============================================================================
# High-Level Init Functions (train.py / evaluate.py)
# =============================================================================


def init_train_run(
    *,
    enabled: bool,
    project: str,
    entity: Optional[str],
    head_type: str,
    dataset_name: str,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[list] = None,
    log_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Optional[Any]:
    """Initialize training run and persist metadata for evaluation resume.

    Creates a run named "{head_type}_{dataset_name}" and saves metadata
    so evaluate.py can resume the same run for test/OOD metrics.
    """
    if not enabled:
        return None

    run_name = make_run_name(head_type, dataset_name)
    run = init_run(
        project=project,
        entity=entity,
        run_name=run_name,
        group=run_name,
        job_type="train",
        config=config,
        tags=tags or [head_type, dataset_name],
        log_dir=log_dir,
    )

    if run is not None and output_dir:
        save_run_metadata(
            output_dir,
            {
                "id": getattr(run, "id", None),
                "name": run_name,
                "group": run_name,
                "project": project,
                "entity": entity,
                "head_type": head_type,
                "dataset_name": dataset_name,
            },
        )

    return run


def init_eval_run(
    *,
    enabled: bool,
    project: str,
    entity: Optional[str],
    head_type: str,
    train_dataset_name: str,
    eval_dataset_name: str,
    split: str = "test",
    train_run_metadata: Optional[Dict[str, Any]] = None,
    log_dir: Optional[str] = None,
) -> Tuple[Optional[Any], str, bool]:
    """Initialize or resume evaluation run.

    For supervised heads: resumes the training run if metadata is available.
    For unsupervised baselines: creates a new run.

    Returns:
        (wandb_run, metric_prefix, is_ood)
    """
    # Normalize dataset names to lowercase for consistent comparison
    train_ds_lower = train_dataset_name.lower()
    eval_ds_lower = eval_dataset_name.lower()
    is_ood = train_ds_lower != eval_ds_lower
    metric_prefix = make_metric_prefix(split, eval_ds_lower, is_ood=is_ood)

    if not enabled:
        return None, metric_prefix, is_ood

    run_name = make_run_name(head_type, train_ds_lower)
    run_id = None
    resume_mode = None
    job_type = "ood_test" if is_ood else "test"

    # Resume training run if metadata available
    if train_run_metadata and train_run_metadata.get("id"):
        run_id = train_run_metadata["id"]
        resume_mode = "allow"
        run_name = train_run_metadata.get("name", run_name)

    run = init_run(
        project=project,
        entity=entity,
        run_name=run_name,
        group=run_name,
        job_type=job_type,
        run_id=run_id,
        resume=resume_mode,
        config={
            "head_type": head_type,
            "train_dataset": train_dataset_name,
            "eval_dataset": eval_dataset_name,
            "split": split,
            "is_ood": is_ood,
        },
        tags=["evaluation", head_type, eval_dataset_name, split],
        log_dir=log_dir,
    )

    if run is not None:
        if run_id:
            log.info(
                "Resumed WandB run id=%s for %s eval (prefix=%s)",
                run_id,
                "OOD" if is_ood else "ID",
                metric_prefix,
            )
        else:
            log.info("Created new WandB run for baseline: %s", run_name)

    return run, metric_prefix, is_ood


def init_baseline_run(
    *,
    enabled: bool,
    project: str,
    entity: Optional[str],
    baseline_name: str,
    train_dataset_name: str,
    eval_dataset_name: str,
    split: str = "test",
    baseline_run_metadata: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
    log_dir: Optional[str] = None,
) -> Tuple[Optional[Any], str, bool]:
    """Initialize or resume a baseline evaluation run.

    For ID evaluation: creates a new run named "{baseline_name}_{train_dataset_name}"
    For OOD evaluation: resumes the existing run and logs under OOD_test_* prefix

    Args:
        baseline_name: Name of the baseline (e.g., "random", "mcp")
        train_dataset_name: Dataset used to define the baseline (e.g., "stepgame")
        eval_dataset_name: Dataset being evaluated (e.g., "babi" for OOD)
        baseline_run_metadata: Previously saved run metadata for resuming
        output_dir: Where to save run metadata for future OOD evaluations

    Returns:
        (wandb_run, metric_prefix, is_ood)
    """
    # Normalize dataset names to lowercase for consistent comparison
    train_ds_lower = train_dataset_name.lower()
    eval_ds_lower = eval_dataset_name.lower()
    is_ood = train_ds_lower != eval_ds_lower
    metric_prefix = make_metric_prefix(split, eval_ds_lower, is_ood=is_ood)

    if not enabled:
        return None, metric_prefix, is_ood

    run_name = make_run_name(baseline_name, train_ds_lower)
    run_id = None
    resume_mode = None
    job_type = "baseline_ood_test" if is_ood else "baseline_test"

    # Resume existing run if metadata available
    if baseline_run_metadata and baseline_run_metadata.get("id"):
        run_id = baseline_run_metadata["id"]
        resume_mode = "allow"
        run_name = baseline_run_metadata.get("name", run_name)

    run = init_run(
        project=project,
        entity=entity,
        run_name=run_name,
        group=run_name,
        job_type=job_type,
        run_id=run_id,
        resume=resume_mode,
        config={
            "method": baseline_name,
            "train_dataset": train_dataset_name,
            "eval_dataset": eval_dataset_name,
            "split": split,
            "is_ood": is_ood,
        },
        tags=["evaluation", "baseline", baseline_name, eval_dataset_name, split],
        log_dir=log_dir,
    )

    # Save metadata for future OOD evaluations (only on first ID evaluation)
    if run is not None and output_dir and not is_ood:
        save_run_metadata(
            output_dir,
            {
                "id": getattr(run, "id", None),
                "name": run_name,
                "group": run_name,
                "project": project,
                "entity": entity,
                "baseline_name": baseline_name,
                "dataset_name": train_dataset_name,
            },
        )

    return run, metric_prefix, is_ood


def load_baseline_run_metadata(output_dir: str, baseline_name: str) -> Optional[Dict[str, Any]]:
    """Load baseline run metadata saved during ID evaluation."""
    if not output_dir:
        return None
    meta_path = Path(output_dir) / baseline_name / "wandb_run.json"
    if not meta_path.exists():
        return None
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        log.warning("Failed to read baseline WandB metadata from %s: %s", meta_path, e)
        return None


# =============================================================================
# Helper Functions
# =============================================================================


def _normalize_metric_name(metric_name: str) -> str:
    """Strip trainer-specific prefixes so names stay stable across phases."""
    if metric_name.startswith("eval_"):
        return metric_name[len("eval_"):]
    if metric_name.startswith("train_"):
        return metric_name[len("train_"):]
    return metric_name


def build_config_dict(cfg) -> Dict[str, Any]:
    """Build a flat config dict for WandB from Config dataclass.

    Args:
        cfg: Config object (from config.py)

    Returns:
        Flat dictionary suitable for wandb.init(config=...)
    """
    config_dict = {}

    if hasattr(cfg, "training"):
        t = cfg.training
        config_dict.update(
            {
                "epochs": t.num_epochs,
                "batch_size": t.per_device_train_batch_size,
                "learning_rate": t.learning_rate,
                "weight_decay": t.weight_decay,
                "warmup_ratio": t.warmup_ratio,
                "lr_scheduler": t.lr_scheduler_type,
                "bf16": t.bf16,
                "early_stopping_patience": t.early_stopping_patience,
            }
        )

    if hasattr(cfg, "head"):
        h = cfg.head
        config_dict.update(
            {
                "head_type": h.head_type,
                "head_dim": h.head_dim,
                "n_layers": h.n_layers,
                "n_heads": h.n_heads,
                "dropout": h.dropout,
            }
        )

    if hasattr(cfg, "dataset"):
        d = cfg.dataset
        config_dict.update(
            {
                "dataset_name": d.dataset_name,
                "max_train_samples": d.max_train_samples,
                "max_eval_samples": d.max_eval_samples,
                "k_hop_values": d.k_hop_values,
            }
        )

    if hasattr(cfg, "model"):
        m = cfg.model
        model_path = getattr(m, "pretrained_model_name_or_path", "")
        if model_path:
            config_dict["model_name"] = model_path.rstrip("/").split("/")[-1]

    return config_dict

def log_baseline_report_to_independent_wandb_run(**kwargs) -> bool:
    """Deprecated: Use init_baseline_run() + log_eval_report() + finish_run()."""
    enabled = kwargs.get("enabled", False)
    if not enabled:
        return False

    project = kwargs.get("project", "")
    entity = kwargs.get("entity")
    dataset_name = kwargs.get("dataset_name", "")
    split = kwargs.get("split", "test")
    baseline_name = kwargs.get("baseline_name", "")
    report = kwargs.get("report", {})
    log_dir = kwargs.get("log_dir")

    run, prefix, _ = init_baseline_run(
        enabled=enabled,
        project=project,
        entity=entity,
        baseline_name=baseline_name,
        train_dataset_name=dataset_name,
        eval_dataset_name=dataset_name,
        split=split,
        log_dir=log_dir,
    )

    if run is None:
        return False

    log_eval_report(prefix, report)
    finish_run()
    return True


def load_train_global_step(head_path: Optional[str]) -> Optional[int]:
    """Load max trainer global_step from sibling train_results.json (if present).

    Note: This is kept for backward compatibility but step-based logging
    is no longer used for eval metrics.
    """
    if not head_path:
        return None

    train_results_path = Path(head_path).resolve().parent / "train_results.json"
    if not train_results_path.is_file():
        return None

    try:
        with train_results_path.open("r", encoding="utf-8") as f:
            train_results = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        log.warning("Failed to parse %s: %s", train_results_path, e)
        return None

    max_step = None
    for entry in train_results.get("log_history", []):
        if not isinstance(entry, dict):
            continue
        raw_step = entry.get("step")
        if isinstance(raw_step, (int, float)):
            step = int(raw_step)
        elif isinstance(raw_step, str) and raw_step.strip().isdigit():
            step = int(raw_step.strip())
        else:
            continue
        if step < 0:
            continue
        if max_step is None or step > max_step:
            max_step = step
    return max_step
