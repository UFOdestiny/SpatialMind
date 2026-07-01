"""
config.py - Central configuration for the SpatialMind codebase.

SpatialMind performs claim-level uncertainty quantification (hallucination
detection) for spatial reasoning in white-box LLMs. A frozen backbone LLM
generates a Reasoning/Conclusion trace; SpatialMind decomposes the trace into
ordered spatial claims, scores each claim's correctness from frozen internal
features, and aggregates the claim scores into a single trace-level (sample-level)
reliability score used for the final evaluation.

All tunable parameters live here. Every path can be overridden via environment
variables so the project is portable across machines by editing only jobs/common.sh
(shell) or the env vars below (python).
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# =============================================================================
# Environment Variable Helpers
# =============================================================================
def _get_env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _get_env_int(key: str, default: int) -> int:
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _get_env_float(key: str, default: float) -> float:
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _get_env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


# =============================================================================
# Base paths (override via environment variables)
#
# Default runtime root is <repo>/spatialmind, which in this deployment is a
# symlink to fast /blue scratch storage. jobs/common.sh sets these explicitly.
# =============================================================================
PROJECT_DIR = _get_env("PROJECT_DIR", str(Path(__file__).parent))
WORKSPACE_ROOT = _get_env("SPATIALMIND_ROOT", f"{PROJECT_DIR}/spatialmind")

MODELS_ROOT = _get_env("MODELS_ROOT", f"{WORKSPACE_ROOT}/models")
DATASETS_ROOT = _get_env("DATASETS_ROOT", f"{WORKSPACE_ROOT}/datasets")
RESULTS_ROOT = _get_env("RESULTS_ROOT", f"{WORKSPACE_ROOT}/results")
LOGS_ROOT = _get_env("LOGS_ROOT", f"{WORKSPACE_ROOT}/logs")
CACHE_ROOT = _get_env("CACHE_ROOT", f"{WORKSPACE_ROOT}/cache")
HF_CACHE = _get_env("HF_CACHE", f"{MODELS_ROOT}/.hf_cache")

HF_TOKEN = _get_env("HF_TOKEN", "")

# =============================================================================
# Global seed for reproducibility (used across all modules)
# =============================================================================
GLOBAL_SEED: int = 2026


# =============================================================================
# Path configuration
# =============================================================================
@dataclass
class PathConfig:
    models_root: str = field(default_factory=lambda: MODELS_ROOT)
    datasets_root: str = field(default_factory=lambda: DATASETS_ROOT)
    results_root: str = field(default_factory=lambda: RESULTS_ROOT)
    logs_root: str = field(default_factory=lambda: LOGS_ROOT)
    cache_root: str = field(default_factory=lambda: CACHE_ROOT)
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
    hf_token: Optional[str] = field(default_factory=lambda: HF_TOKEN or None)
    models: List[DownloadModelEntry] = field(default_factory=list)
    datasets: List[Dict[str, Optional[str]]] = field(default_factory=list)
    use_symlinks: bool = False
    resume_download: bool = True


# =============================================================================
# Base LLM model configuration (frozen backbone that generates the trace)
# =============================================================================
@dataclass
class ModelConfig:
    pretrained_model_name_or_path: str = field(
        default_factory=lambda: f"{MODELS_ROOT}/{_get_env('MODEL_NAME', 'Llama-3.1-8B-Instruct')}"
    )
    device_map: str = "cuda"
    torch_dtype: str = "float16"
    trust_remote_code: bool = True
    hf_cache_dir: Optional[str] = field(default_factory=lambda: HF_CACHE)
    hf_token: Optional[str] = field(default_factory=lambda: HF_TOKEN or None)
    model_max_length: int = 2048
    padding_side: str = "left"


# =============================================================================
# Frozen feature extractor configuration
#
# The frozen per-token feature x_t = [ h_t^last ; l_t ; a_t ] concatenates the
# final-layer hidden state, top-k decoding log-probabilities, and short-range
# attention-lookback features (see the paper's "Frozen Trace Features").
# =============================================================================
@dataclass
class FeatureConfig:
    hidden_state_layers: str = "-1"
    top_n_probs: int = 4
    temperature: float = 1.0
    # Attention-lookback features: use the last few layers by default for efficiency.
    attention_layers: str = "-1,-2,-3"  # "all" | comma list | "" (disable)
    attention_heads: str = "all"        # "all" | comma-separated head indices
    attn_history_sz: int = 3            # lookback window
    pool_attention_layers: bool = False


# =============================================================================
# Head configuration
#
# num_classes=1 => binary per-claim correctness (1=supported, 0=hallucinated).
# `aggregation` selects how per-claim probabilities collapse to one trace score
# at evaluation time; "mix" is a validation-selected blend applied uniformly to
# every method (including baselines) so the sample-level comparison is fair.
# =============================================================================
@dataclass
class HeadConfig:
    head_type: str = field(default_factory=lambda: _get_env("HEAD_TYPE", "spatialmind"))
    num_classes: int = 1
    head_dim: int = 256
    n_layers: int = 2
    n_heads: int = 8
    dropout: float = 0.1
    max_seq_len: int = 512
    # Multi-task: the SpatialMind head emits a learned trace-level logit in
    # addition to per-claim logits. `trace_loss_weight` scales the trace-level
    # BCE term added to the per-claim BCE. 0 disables the multi-task head.
    trace_loss_weight: float = field(default_factory=lambda: _get_env_float("TRACE_LOSS_WEIGHT", 0.5))
    # Default claim->trace aggregation used when a head has no learned trace head.
    aggregation: str = field(default_factory=lambda: _get_env("AGGREGATION", "mix"))


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
    num_epochs: int = field(default_factory=lambda: _get_env_int("NUM_EPOCHS", 40))
    learning_rate: float = field(default_factory=lambda: _get_env_float("LEARNING_RATE", 2e-4))
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    per_device_train_batch_size: int = field(default_factory=lambda: _get_env_int("BATCH_SIZE", 64))
    per_device_eval_batch_size: int = field(default_factory=lambda: _get_env_int("BATCH_SIZE", 64))
    max_grad_norm: float = 1.0
    # Loss on per-claim logits. bce | balanced_bce | focal.
    loss_type: str = field(default_factory=lambda: _get_env("LOSS_TYPE", "bce"))
    loss_pos_weight: float = field(default_factory=lambda: _get_env_float("LOSS_POS_WEIGHT", 1.0))
    focal_gamma: float = field(default_factory=lambda: _get_env_float("FOCAL_GAMMA", 2.0))
    grad_accum_steps: int = field(default_factory=lambda: _get_env_int("GRAD_ACCUM_STEPS", 1))
    amp_dtype: str = field(default_factory=lambda: _get_env("AMP_DTYPE", "bfloat16"))  # bfloat16 | float16 | float32
    # Model selection: the metric is measured at the SAMPLE (trace) level on the
    # validation split, matching the reported evaluation protocol.
    metric_for_best_model: str = field(default_factory=lambda: _get_env("BEST_METRIC", "auroc"))
    early_stopping_patience: int = field(default_factory=lambda: _get_env_int("EARLY_STOP_PATIENCE", 8))
    seed: int = GLOBAL_SEED
    num_workers: int = field(default_factory=lambda: _get_env_int("DATALOADER_NUM_WORKERS", 2))


# =============================================================================
# Generation configuration (Phase 1: LLM generation + feature caching)
# =============================================================================
@dataclass
class GenerationConfig:
    cache_dir: str = field(default_factory=lambda: f"{CACHE_ROOT}/cached_features")
    max_new_tokens: int = field(default_factory=lambda: _get_env_int("GEN_MAX_NEW_TOKENS", 256))
    do_sample: bool = False
    temperature: float = 1.0
    batch_size: int = field(default_factory=lambda: _get_env_int("GEN_BATCH_SIZE", 32))
    skip_existing: bool = True
    chunk_size: int = field(default_factory=lambda: _get_env_int("GEN_CHUNK_SIZE", 10000))
    save_hidden_states: bool = True
    save_token_probs: bool = True
    save_attention_weights: bool = True
    backend: str = field(default_factory=lambda: _get_env("BACKEND", "vllm"))


# =============================================================================
# vLLM engine configuration (used when backend="vllm")
# =============================================================================
@dataclass
class VLLMConfig:
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = field(
        default_factory=lambda: _get_env_float("VLLM_GPU_MEMORY_UTILIZATION", 0.30)
    )
    max_model_len: int = field(default_factory=lambda: _get_env_int("VLLM_MAX_MODEL_LEN", 2048))
    enforce_eager: bool = False
    swap_space: int = 4
    dtype: str = "auto"
    seed: int = GLOBAL_SEED
    attention_backend: str = field(
        default_factory=lambda: _get_env("VLLM_ATTENTION_BACKEND", "FLASHINFER")
    )


# =============================================================================
# Judge configuration (Phase 1.5: LLM-as-judge for free-form / claim labeling)
# =============================================================================
@dataclass
class JudgeConfig:
    judge_model_path: str = field(
        default_factory=lambda: f"{MODELS_ROOT}/{_get_env('JUDGE_MODEL_NAME', 'Mistral-Small-3.2-24B-Instruct-2506')}"
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
