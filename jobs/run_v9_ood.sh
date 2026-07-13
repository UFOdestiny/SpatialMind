#!/usr/bin/env bash
###############################################################################
# run_v9_ood.sh - Phase 3b: OOD transfer for the constraint_guided_v9 heads.
#
# For each OOD dataset: guided generate -> guided judge -> rebuild native
# constraint view -> evaluate every trained head + baselines under strict
# validation-adapted calibration (no test-stat leakage). Reuses the real
# pipeline functions from p1.sh/p3.sh so behaviour matches a full run.
###############################################################################
set -uo pipefail
SCRIPT_DIR="/home/dy23a.fsu/popllm/SpatialMind/jobs"
cd /home/dy23a.fsu/popllm/SpatialMind

# --- pin everything to the v9 namespace / job dir ---
export DATASET_NAME="StepGame"
export MODEL_NAME="Llama-3.1-8B-Instruct"
export JUDGE_MODEL_NAME="Mistral-Small-3.2-24B-Instruct-2506"
export CACHE_SUBDIR="constraint_guided_v9"
export RESULTS_ROOT="/home/dy23a.fsu/popllm/SpatialMind/spatialmind/results/constraint_guided_v9_20260712"
export LOGS_ROOT="/home/dy23a.fsu/popllm/SpatialMind/spatialmind/logs/constraint_guided_v9_20260712"
export GEN_MAX_OOD="${GEN_MAX_OOD:-500}"
export GEN_MAX_OOD_VAL="${GEN_MAX_OOD_VAL:-250}"
export CALIBRATE="${CALIBRATE:-standard}"
export STRUCT_CALIB_C="${STRUCT_CALIB_C:-0.01}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"

HEAD_TYPES="spatialmind constraint_only spatialmind_neural constraint_no_conflict constraint_no_context constraint_no_entailment constraint_no_repair uhead factoscope mlp"
OOD_LIST=(spartqa babi SpaRTUN SpaceNLI)

source "${SCRIPT_DIR}/common.sh"
read -ra ALL_HEAD_TYPES <<< "${HEAD_TYPES}"
setup_environment
detect_gpus

export MODEL_PATH="${MODELS_ROOT}/${MODEL_NAME}"
export JUDGE_MODEL_PATH="${MODELS_ROOT}/${JUDGE_MODEL_NAME}"

source "${SCRIPT_DIR}/p1.sh"
source "${SCRIPT_DIR}/p3.sh"

mkdir -p "${LOGS_ROOT}/data" "${LOGS_ROOT}/eval"

echo "=== Phase 3b OOD start $(date) ==="
echo "OOD: ${OOD_LIST[*]} | heads: ${ALL_HEAD_TYPES[*]}"

for ood in "${OOD_LIST[@]}"; do
    echo "########## OOD: ${ood} $(date) ##########"
    prepare_ood_cache "${ood}" || { echo "[FAIL] prepare ${ood}"; continue; }
    ood_cache="$(_ood_cache_dir "${ood}")"

    # Rebuild native constraint view to match v9 ID schema (Phase 1.6 parity).
    for sp in validation test; do
        python scripts/rebuild_constraint_cache.py --cache_dir "${ood_cache}" --split "${sp}" \
            >> "${LOGS_ROOT}/data/rebuild_ood_${ood}.log" 2>&1 || echo "[WARN] rebuild ${ood}/${sp}"
    done

    if ! cache_split_ready "${ood_cache}" "test" || ! cache_split_ready "${ood_cache}" "validation"; then
        echo "[FAIL] ${ood}: cache not ready (strict: need validation+test)"; continue
    fi

    for ht in "${ALL_HEAD_TYPES[@]}"; do
        out="${RESULTS_ROOT}/eval_ood/${ood}/${ht}"
        [[ -f "${out}/evaluation_report.json" ]] && { echo "[SKIP] ${ood}/${ht}"; continue; }
        eval_head "${ht}" "${ood_cache}" "${out}" || echo "[FAIL] eval ${ood}/${ht}"
    done
    if [[ ! -f "${RESULTS_ROOT}/eval_ood/${ood}/baselines/combined_evaluation.json" ]]; then
        eval_baselines "${ood_cache}" "${RESULTS_ROOT}/eval_ood/${ood}/baselines" || echo "[FAIL] baselines ${ood}"
    fi
    echo "########## OOD ${ood} done $(date) ##########"
done

echo "=== Phase 3b OOD done $(date) ==="
