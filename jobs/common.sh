#!/usr/bin/env bash
###############################################################################
# common.sh - Shared configuration and utilities for SpatialMind pipelines.
#
# Source this from a pipeline script:
#   source "$(dirname "$0")/common.sh"
#
# ALL global defaults live here. To port the project to a new machine, edit only
# the "Path Configuration" and "Model & Dataset Configuration" blocks below.
#
# By design, ALL runtime artifacts (logs / results / cache / HF cache) are written
# UNDER  <repo>/spatialmind , which in this deployment is a symlink to fast /blue
# scratch storage. Models and datasets are read through spatialmind/{models,datasets}.
###############################################################################

###############################################################################
# Path Configuration
###############################################################################
_SCRIPT_DIR="/home/dy23a.fsu/popllm/SpatialMind/jobs"
_PROJECT_ROOT="$(dirname "${_SCRIPT_DIR}")"

PROJECT_DIR="${PROJECT_DIR:-${_PROJECT_ROOT}}"
# Runtime root: everything (logs/results/cache) is written here, as required.
SPATIALMIND_ROOT="${SPATIALMIND_ROOT:-${PROJECT_DIR}/spatialmind}"

# Models & datasets are read via the spatialmind/ symlinks to shared storage.
MODELS_ROOT="${MODELS_ROOT:-${SPATIALMIND_ROOT}/models}"
DATASETS_ROOT="${DATASETS_ROOT:-${SPATIALMIND_ROOT}/datasets}"

# Job artifacts (base + per-job subdir keyed by SLURM_JOB_ID).
BASE_RESULTS_ROOT="${BASE_RESULTS_ROOT:-${SPATIALMIND_ROOT}/results}"
BASE_LOGS_ROOT="${BASE_LOGS_ROOT:-${SPATIALMIND_ROOT}/logs}"
RESULTS_ROOT="${RESULTS_ROOT:-${BASE_RESULTS_ROOT}/${SLURM_JOB_ID:-manual_run}}"
LOGS_ROOT="${LOGS_ROOT:-${BASE_LOGS_ROOT}/${SLURM_JOB_ID:-manual_run}}"

# Cached frozen features (shared across jobs; keyed by dataset/model).
CACHE_ROOT="${CACHE_ROOT:-${SPATIALMIND_ROOT}/cache}"
HF_CACHE="${HF_CACHE:-${MODELS_ROOT}/.hf_cache}"

###############################################################################
# vLLM / GPU Configuration (centralized)
###############################################################################
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.30}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-2048}"
VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER}"
HF_TOKEN="${HF_TOKEN:-}"

###############################################################################
# Model & Dataset Configuration
#
# Backbones present on disk: Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3,
# gemma-2-9b-it. Judge: Mistral-Small-3.2-24B-Instruct-2506.
# Claim extraction uses the strict Reasoning/Conclusion format + regex (no extra
# model); set CLAIM_EXTRACTOR_MODEL_NAME to enable an LLM extractor if desired.
###############################################################################
MODEL_NAME="${MODEL_NAME:-Llama-3.1-8B-Instruct}"
DATASET_NAME="${DATASET_NAME:-StepGame}"
JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-Mistral-Small-3.2-24B-Instruct-2506}"
CLAIM_EXTRACTOR_MODEL_NAME="${CLAIM_EXTRACTOR_MODEL_NAME:-}"
CLAIM_EXTRACTOR_BACKEND="${CLAIM_EXTRACTOR_BACKEND:-vllm}"
CLAIM_EXTRACTOR_MAX_NEW_TOKENS="${CLAIM_EXTRACTOR_MAX_NEW_TOKENS:-256}"
CLAIM_LABELER_MAX_NEW_TOKENS="${CLAIM_LABELER_MAX_NEW_TOKENS:-128}"
# Stage-2 reasoning judge uses an analysis-first prompt; needs room for the CoT.
JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-256}"
DEFER_CLAIM_EXTRACTION="${DEFER_CLAIM_EXTRACTION:-1}"

# Judge rerun control.
JUDGE_PENDING_SKIP_THRESHOLD="${JUDGE_PENDING_SKIP_THRESHOLD:-10}"
FORCE_JUDGE="${FORCE_JUDGE:-0}"
FORCE_OOD_JUDGE_DATASETS="${FORCE_OOD_JUDGE_DATASETS:-}"

# Download mappings: "local_name:repo_id".
DOWNLOAD_MODELS=(
    "Llama-3.1-8B-Instruct:meta-llama/Llama-3.1-8B-Instruct"
    "Mistral-7B-Instruct-v0.3:mistralai/Mistral-7B-Instruct-v0.3"
    "gemma-2-9b-it:google/gemma-2-9b-it"
    "Mistral-Small-3.2-24B-Instruct-2506:mistralai/Mistral-Small-3.2-24B-Instruct-2506"
)
DOWNLOAD_DATASETS=(
    "StepGame:ZhengyanShi/StepGame"
    "spartqa:tasksource/spartqa-mchoice"
    "babi:facebook/babi_qa"
    "SpaRTUN:tasksource/SpaRTUN"
    "SpaceNLI:tasksource/SpaceNLI"
)

# Cache directory (optionally namespaced by CACHE_SUBDIR, e.g. "example").
CACHE_SUBDIR="${CACHE_SUBDIR:-}"
if [[ -n "${CACHE_SUBDIR}" ]]; then
    CACHE_DIR="${CACHE_ROOT}/cached_features/${CACHE_SUBDIR}/${DATASET_NAME}/${MODEL_NAME}"
else
    CACHE_DIR="${CACHE_ROOT}/cached_features/${DATASET_NAME}/${MODEL_NAME}"
fi

###############################################################################
# Head Zoo
#
# Our method + full baseline zoo + ablations. Every head is claim-level and is
# aggregated to the trace (sample) level by the shared evaluation protocol.
###############################################################################
ALL_HEAD_TYPES=(
    # Our method
    "spatialmind"
    # Supervised baselines
    "saplma"
    "factoscope"
    "lookback_lens"
    "uhead"
    "luh_light"
    "linear"
    "mlp"
    "gated_mlp"
    "cnn"
    # SpatialMind ablations (cumulative)
    "abl_base"
    "abl_cross"
    "abl_type"
    "abl_scope"
    # SpatialMind ablations (leave-one-out)
    "abl_no_cross"
    "abl_no_type"
    "abl_no_scope"
    "abl_no_bank"
)

# Unsupervised baselines are evaluated jointly via `evaluate.py --eval_baselines`.

###############################################################################
# Training Configuration
###############################################################################
TRAIN_EPOCHS="${TRAIN_EPOCHS:-${NUM_EPOCHS:-30}}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${BATCH_SIZE:-1024}}"
TRAIN_LEARNING_RATE="${TRAIN_LEARNING_RATE:-${LEARNING_RATE:-0.0002}}"
LOSS_TYPE="${LOSS_TYPE:-balanced_bce}"        # bce | balanced_bce | focal (auto per-level pos_weight)
LOSS_POS_WEIGHT="${LOSS_POS_WEIGHT:-0}"       # 0 = auto per-level; >0 overrides
FOCAL_GAMMA="${FOCAL_GAMMA:-2.0}"
TRACE_LOSS_WEIGHT="${TRACE_LOSS_WEIGHT:-0.5}"  # multi-task trace objective weight
BEST_METRIC="${BEST_METRIC:-auroc}"            # sample-level model-selection metric

# Multi-GPU controls.
MAX_HEADS_PER_GPU="${MAX_HEADS_PER_GPU:-1}"
MAX_PARALLEL_HEADS="${MAX_PARALLEL_HEADS:-}"
EVAL_MAX_HEADS_PER_GPU="${EVAL_MAX_HEADS_PER_GPU:-${MAX_HEADS_PER_GPU:-1}}"
EVAL_MAX_PARALLEL_HEADS="${EVAL_MAX_PARALLEL_HEADS:-}"

###############################################################################
# Generation Configuration (Phase 1)
###############################################################################
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-32}"
FREE_FORM_GEN_BATCH_SIZE="${FREE_FORM_GEN_BATCH_SIZE:-8}"
BACKEND="${BACKEND:-vllm}"
GEN_MAX_NEW_TOKENS="${GEN_MAX_NEW_TOKENS:-256}"
# Traces per cache chunk. Small => bounded host RAM during training (chunks are
# tens of GB at full scale). 2500 ~ 6-10 GB/chunk for feature_dim ~4.4k.
GEN_CHUNK_SIZE="${GEN_CHUNK_SIZE:-2500}"
# Host-RAM controls for training data loading. With small chunks (GEN_CHUNK_SIZE
# ~2500 => ~6-10 GB each) parallel workers overlap I/O with GPU compute safely:
# peak RAM ~ (workers+1) x MAX_CACHED_CHUNKS x chunk. Set workers=0 only for very
# large chunks (tens of GB). Applies to ALL head trainings.
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
MAX_CACHED_CHUNKS="${MAX_CACHED_CHUNKS:-1}"
GEN_MAX_TRAIN="${GEN_MAX_TRAIN:-0}"   # 0 = no limit
GEN_MAX_VAL="${GEN_MAX_VAL:-0}"
GEN_MAX_TEST="${GEN_MAX_TEST:-0}"
GEN_MAX_OOD="${GEN_MAX_OOD:-${GEN_MAX_TEST}}"          # OOD test samples
GEN_MAX_OOD_VAL="${GEN_MAX_OOD_VAL:-${GEN_MAX_OOD}}"  # OOD validation (calibration) samples

###############################################################################
# OOD Evaluation Configuration
###############################################################################
# In-distribution training set is StepGame; OOD transfer targets:
OOD_DATASETS="${OOD_DATASETS:-spartqa babi SpaRTUN SpaceNLI}"

###############################################################################
# Environment Setup
###############################################################################
setup_environment() {
    export HF_TOKEN="${HF_TOKEN:-}"
    export XDG_RUNTIME_DIR="/tmp/runtime-${USER:-$(whoami)}"
    export HF_HOME="${HF_CACHE}"
    export DISABLE_TQDM=1
    export PYTHONUNBUFFERED=1
    export USE_TF=0 USE_FLAX=0
    export TRANSFORMERS_NO_TF=1 TRANSFORMERS_NO_FLAX=1 TRANSFORMERS_NO_JAX=1
    export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND}"

    # Export path + hyperparameter env vars so config.py resolves identically.
    export SPATIALMIND_ROOT MODELS_ROOT DATASETS_ROOT RESULTS_ROOT LOGS_ROOT CACHE_ROOT HF_CACHE
    export MODEL_NAME DATASET_NAME JUDGE_MODEL_NAME JUDGE_MAX_NEW_TOKENS
    export VLLM_GPU_MEMORY_UTILIZATION VLLM_MAX_MODEL_LEN
    export TRAIN_EPOCHS NUM_EPOCHS="${TRAIN_EPOCHS}"
    export BATCH_SIZE="${TRAIN_BATCH_SIZE}" LEARNING_RATE="${TRAIN_LEARNING_RATE}"
    export LOSS_TYPE LOSS_POS_WEIGHT FOCAL_GAMMA TRACE_LOSS_WEIGHT BEST_METRIC
    export BACKEND GEN_MAX_NEW_TOKENS GEN_CHUNK_SIZE
    export DATALOADER_NUM_WORKERS MAX_CACHED_CHUNKS

    # Compiler caches on persistent storage.
    export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/.torchinductor"
    export TRITON_CACHE_DIR="${CACHE_ROOT}/.triton"
    mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

    # Create all runtime directories under the spatialmind root.
    mkdir -p "${RESULTS_ROOT}" "${LOGS_ROOT}" "${HF_CACHE}" "${CACHE_DIR}" \
             "${LOGS_ROOT}/data" "${LOGS_ROOT}/train" "${LOGS_ROOT}/eval" \
             "${LOGS_ROOT}/download" "${LOGS_ROOT}/figure"

    # HPC modules + conda (adjust for your cluster).
    module load cuda/12.8.1 2>/dev/null || true
    module load gcc 2>/dev/null || true
    module load conda 2>/dev/null || true
    conda activate "${CONDA_ENV:-llm}" 2>/dev/null || true

    if command -v python >/dev/null 2>&1; then
        export PYTHON_BIN="python"
    elif command -v python3 >/dev/null 2>&1; then
        export PYTHON_BIN="python3"
    else
        echo "[ERROR] Neither python nor python3 found in PATH."
        return 127
    fi
    cd "${PROJECT_DIR}"
}

###############################################################################
# GPU Detection
###############################################################################
detect_gpus() {
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        IFS=',' read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
        NUM_GPUS=${#GPU_IDS[@]}
    else
        NUM_GPUS=$( (nvidia-smi -L 2>/dev/null || true) | wc -l )
        NUM_GPUS=${NUM_GPUS//[[:space:]]/}
        if [ "${NUM_GPUS}" -gt 0 ]; then
            GPU_IDS=($(seq 0 $((NUM_GPUS - 1))))
        else
            GPU_IDS=(0); NUM_GPUS=0
        fi
    fi
    echo "Detected GPUs: [${GPU_IDS[*]}] (total: ${NUM_GPUS})"
    if [ "${NUM_GPUS}" -le 0 ]; then
        echo "[WARN] No visible GPUs; running CPU-compatible single-worker mode."
    fi
}

###############################################################################
# Utility helpers
###############################################################################
print_header() {
    echo ""
    echo "###################################################################"
    echo "#  $1"
    echo "###################################################################"
    echo ""
}

print_step() {
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

format_duration() {
    local s=$1 m h
    m=$((s / 60)); h=$((m / 60)); m=$((m % 60)); s=$((s % 60))
    if [ ${h} -gt 0 ]; then echo "${h}h ${m}m"
    elif [ ${m} -gt 0 ]; then echo "${m}m ${s}s"
    else echo "${s}s"; fi
}

# --- cache-readiness helpers (used to skip completed stages) --- #
cache_split_ready() {
    local cache_dir="$1" split="$2"
    local manifest="${cache_dir}/${split}/manifest.json"
    [[ -f "${manifest}" ]] || return 1
    local expected
    expected=$(grep -o '"num_chunks"[[:space:]]*:[[:space:]]*[0-9]\+' "${manifest}" | head -n1 | grep -o '[0-9]\+' || true)
    [[ -n "${expected}" && "${expected}" -gt 0 ]] || return 1
    local existing
    existing=$(find "${cache_dir}/${split}" -maxdepth 1 -type f -name 'chunk_*.pt' 2>/dev/null | wc -l)
    [[ "${existing}" -ge "${expected}" ]]
}

cache_splits_ready() {
    local cache_dir="$1" splits_csv="$2" split
    local splits=(); IFS=',' read -ra splits <<< "${splits_csv}"
    for split in "${splits[@]}"; do
        split="${split// /}"; [[ -n "${split}" ]] || continue
        cache_split_ready "${cache_dir}" "${split}" || return 1
    done
    return 0
}

manifest_total_pending() {
    local manifest="$1/$2/manifest.json"
    [[ -f "${manifest}" ]] || return 1
    local p; p=$(grep -o '"total_pending"[[:space:]]*:[[:space:]]*[0-9]\+' "${manifest}" | head -n1 | grep -o '[0-9]\+' || true)
    [[ -n "${p}" ]] || return 1
    echo "${p}"
}

# Pending CLAIM labels (verified == -1), e.g. reasoning claims awaiting Stage-2.
# Distinct from sample-level total_pending: an exact-match dataset has
# total_pending=0 but total_claim_pending>0 until the reasoning judge runs.
manifest_total_claim_pending() {
    local manifest="$1/$2/manifest.json"
    [[ -f "${manifest}" ]] || return 1
    local p; p=$(grep -o '"total_claim_pending"[[:space:]]*:[[:space:]]*[0-9]\+' "${manifest}" | head -n1 | grep -o '[0-9]\+' || true)
    [[ -n "${p}" ]] || return 1
    echo "${p}"
}

should_run_judge_for_splits() {
    local cache_dir="$1" splits_csv="$2" context="${3:-Judge}" ignore_force="${4:-0}"
    local threshold="${JUDGE_PENDING_SKIP_THRESHOLD:-0}"
    if [[ "${ignore_force}" != "1" && "${FORCE_JUDGE:-0}" == "1" ]]; then
        echo "  [JUDGE] ${context}: FORCE_JUDGE=1."; return 0
    fi
    local split splits=(); IFS=',' read -ra splits <<< "${splits_csv}"
    for split in "${splits[@]}"; do
        split="${split// /}"; [[ -n "${split}" ]] || continue
        if ! cache_split_ready "${cache_dir}" "${split}"; then
            echo "  [JUDGE] ${context}/${split}: cache not ready, will run judge."; return 0
        fi
        local pending
        if ! pending=$(manifest_total_pending "${cache_dir}" "${split}"); then
            echo "  [JUDGE] ${context}/${split}: no total_pending, will run judge."; return 0
        fi
        # Also trigger on pending CLAIM labels (reasoning claims awaiting Stage-2),
        # which are the common case for exact-match datasets (sample pending=0).
        local claim_pending
        if claim_pending=$(manifest_total_claim_pending "${cache_dir}" "${split}"); then
            if [[ "${claim_pending}" -gt "${threshold}" ]]; then
                echo "  [JUDGE] ${context}/${split}: claim_pending=${claim_pending} > ${threshold}, will run judge."; return 0
            fi
        else
            # Legacy manifest without the key: run judge to be safe.
            echo "  [JUDGE] ${context}/${split}: no total_claim_pending key, will run judge."; return 0
        fi
        if [[ "${pending}" -gt "${threshold}" ]]; then
            echo "  [JUDGE] ${context}/${split}: pending=${pending} > ${threshold}, will run judge."; return 0
        fi
        echo "  [JUDGE] ${context}/${split}: pending=${pending}, claim_pending=${claim_pending} <= ${threshold}, skip."
    done
    return 1
}
