"""
config.py - Central configuration for the public SpatialMind codebase.

All tunable parameters are defined here. Paths can be overridden via environment
variables.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict


# =============================================================================
# Environment Variable Helpers
# =============================================================================
def _get_env(key: str, default: str = "") -> str:
    """Get environment variable with fallback."""
    return os.environ.get(key, default)


def _get_env_int(key: str, default: int) -> int:
    """Get integer environment variable with fallback."""
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _get_env_float(key: str, default: float) -> float:
    """Get float environment variable with fallback."""
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


# =============================================================================
# Base paths (override via environment variables)
# =============================================================================
PROJECT_DIR = _get_env("PROJECT_DIR", str(Path(__file__).parent))
WORKSPACE_ROOT = _get_env("SPATIALMIND_ROOT", f"{PROJECT_DIR}/artifacts")

# Derived paths
MODELS_ROOT = _get_env("MODELS_ROOT", f"{WORKSPACE_ROOT}/models")
DATASETS_ROOT = _get_env("DATASETS_ROOT", f"{WORKSPACE_ROOT}/datasets")
RESULTS_ROOT = _get_env("RESULTS_ROOT", f"{WORKSPACE_ROOT}/results")
LOGS_ROOT = _get_env("LOGS_ROOT", f"{WORKSPACE_ROOT}/logs")
HF_CACHE = _get_env("HF_CACHE", f"{MODELS_ROOT}/.hf_cache")

# HuggingFace token
HF_TOKEN = _get_env("HF_TOKEN", "")

# =============================================================================
# Global seed for reproducibility (used across all modules)
# =============================================================================
GLOBAL_SEED: int = 2026


# =============================================================================
# Path configuration (from environment variables)
# =============================================================================
@dataclass
class PathConfig:
    models_root: str = field(default_factory=lambda: MODELS_ROOT)
    datasets_root: str = field(default_factory=lambda: DATASETS_ROOT)
    results_root: str = field(default_factory=lambda: RESULTS_ROOT)
    logs_root: str = field(default_factory=lambda: LOGS_ROOT)
    hf_cache_dir: Optional[str] = field(default_factory=lambda: HF_CACHE)


# =============================================================================
# Download configuration
# =============================================================================
@dataclass
class DownloadModelEntry:
    repo_id: str = ""
    local_name: str = ""
    size: str = ""
    requires_auth: bool = False


@dataclass
class DownloadConfig:
    # Load HF token from environment variable when needed.
    hf_token: Optional[str] = field(default_factory=lambda: HF_TOKEN or None)
    models: List[DownloadModelEntry] = field(default_factory=list)
    datasets: List[Dict[str, Optional[str]]] = field(default_factory=list)

    use_symlinks: bool = False
    resume_download: bool = True


# =============================================================================
# Base LLM model configuration
# =============================================================================
@dataclass
class ModelConfig:
    pretrained_model_name_or_path: str = field(
        default_factory=lambda: f"{MODELS_ROOT}/{_get_env('MODEL_NAME', 'backbone-model')}"
    )
    device_map: str = "cuda"
    torch_dtype: str = "float16"
    trust_remote_code: bool = True
    hf_cache_dir: Optional[str] = field(default_factory=lambda: HF_CACHE)
    hf_token: Optional[str] = field(default_factory=lambda: HF_TOKEN or None)
    model_max_length: int = 2048
    padding_side: str = "left"


# =============================================================================
# Feature extractor configuration
# =============================================================================
@dataclass
class FeatureConfig:
    hidden_state_layers: str = "-1"
    top_n_probs: int = 4
    temperature: float = 1.0
    # Attention feature extraction: use the last few layers by default for efficiency.
    attention_layers: str = "-1,-2,-3"  # "all" or e.g. "-1,-2,-3"; empty = no attention features
    attention_heads: str = "all"  # "all" or comma-separated head indices
    attn_history_sz: int = 3  # Default lookback window
    pool_attention_layers: bool = False  # Keep per-layer channels by default


# =============================================================================
# Head configuration
# =============================================================================
@dataclass
class HeadConfig:
    head_type: str = "uq"
    num_classes: int = 1  # Binary: 1=correct/non-hallucination, 0=hallucination
    # These shape parameters are used by the supported supervised heads.
    head_dim: int = 256
    n_layers: int = 2
    n_heads: int = 8
    dropout: float = 0.1


# =============================================================================
# Dataset configuration
# =============================================================================
@dataclass
class DatasetConfig:
    granularity: str = "claim"
    dataset_name: str = field(default_factory=lambda: _get_env("DATASET_NAME", "stepgame").lower())
    dataset_path: str = field(default_factory=lambda: f"{DATASETS_ROOT}/{_get_env('DATASET_NAME', 'StepGame')}")
    val_split_ratio: float = 0.1
    max_train_samples: int = 0
    max_eval_samples: int = 0
    hf_cache_dir: Optional[str] = field(default_factory=lambda: HF_CACHE)
    k_hop_values: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8, 9])


# =============================================================================
# Training configuration
# =============================================================================
@dataclass
class TrainingConfig:
    num_epochs: int = field(default_factory=lambda: _get_env_int("NUM_EPOCHS", 100))
    learning_rate: float = field(default_factory=lambda: _get_env_float("LEARNING_RATE", 2e-4))
    warmup_steps: int = 0
    warmup_ratio: float = 0.1
    weight_decay: float = 0.1
    per_device_train_batch_size: int = field(default_factory=lambda: _get_env_int("BATCH_SIZE", 32))
    per_device_eval_batch_size: int = field(default_factory=lambda: _get_env_int("BATCH_SIZE", 32))
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    loss_type: str = field(default_factory=lambda: _get_env("LOSS_TYPE", "bce"))
    loss_pos_weight: float = field(default_factory=lambda: _get_env_float("LOSS_POS_WEIGHT", 1.0))
    focal_gamma: float = field(default_factory=lambda: _get_env_float("FOCAL_GAMMA", 2.0))
    lr_scheduler_type: str = "linear"
    fp16: bool = False
    bf16: bool = True
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "pr_auc"
    greater_is_better: Optional[bool] = None
    early_stopping_patience: int = 10
    seed: int = GLOBAL_SEED
    report_to: str = field(default_factory=lambda: _get_env("REPORT_TO", "none"))
    wandb_project: str = field(default_factory=lambda: _get_env("WANDB_PROJECT", "spatialmind"))
    wandb_entity: Optional[str] = field(default_factory=lambda: _get_env("WANDB_ENTITY", "") or None)
    wandb_run_name: Optional[str] = None
    dataloader_num_workers: int = 1
    dataloader_pin_memory: bool = True
    dataloader_prefetch_factor: int = 2
    dataloader_persistent_workers: bool = True
    dataloader_drop_last: bool = True
    logging_strategy: str = "epoch"
    logging_steps: int = 100
    disable_tqdm: bool = True


# =============================================================================
# Generation configuration (Phase 1: LLM generation + feature caching)
# =============================================================================
@dataclass
class GenerationConfig:
    cache_dir: str = field(default_factory=lambda: f"{WORKSPACE_ROOT}/cached_features")
    max_new_tokens: int = field(default_factory=lambda: _get_env_int("GEN_MAX_NEW_TOKENS", 256))
    do_sample: bool = False
    temperature: float = 1.0
    batch_size: int = field(default_factory=lambda: _get_env_int("GEN_BATCH_SIZE", 8))
    skip_existing: bool = True
    chunk_size: int = 10000
    save_hidden_states: bool = True
    save_token_probs: bool = True
    save_attention_weights: bool = True
    backend: str = field(default_factory=lambda: _get_env("BACKEND", "vllm"))


# =============================================================================
# vLLM engine configuration (used when backend="vllm")
# =============================================================================
@dataclass
class VLLMConfig:
    """vLLM engine configuration for Phase 1 acceleration."""
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = field(
        default_factory=lambda: _get_env_float("VLLM_GPU_MEMORY_UTILIZATION", 0.25)
    )
    max_model_len: int = field(
        default_factory=lambda: _get_env_int("VLLM_MAX_MODEL_LEN", 2048)
    )
    enforce_eager: bool = False
    swap_space: int = 4
    dtype: str = "auto"
    seed: int = GLOBAL_SEED
    attention_backend: str = field(
        default_factory=lambda: _get_env("VLLM_ATTENTION_BACKEND", "FLASHINFER")
    )


# =============================================================================
# Judge configuration (Phase 1.5: LLM-as-judge for free-form tasks)
# =============================================================================
@dataclass
class JudgeConfig:
    """Config for scripts/judge.py — LLM-based correctness evaluation."""
    judge_model_path: str = field(
        default_factory=lambda: f"{MODELS_ROOT}/{_get_env('JUDGE_MODEL_NAME', 'judge-model')}"
    )
    judge_backend: str = field(default_factory=lambda: _get_env("BACKEND", "vllm"))
    judge_max_new_tokens: int = 64
    judge_batch_size: int = 64


# =============================================================================
# Output configuration
# =============================================================================
@dataclass
class OutputConfig:
    output_dir: str = field(default_factory=lambda: RESULTS_ROOT)
    log_dir: str = field(default_factory=lambda: LOGS_ROOT)
    log_level: str = "INFO"
    save_final_model: bool = True
    final_model_subdir: str = "final_model"


# =============================================================================
# Top-level config
# =============================================================================
@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    download: DownloadConfig = field(default_factory=DownloadConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    vllm: VLLMConfig = field(default_factory=VLLMConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def get_default_config() -> Config:
    return Config()
