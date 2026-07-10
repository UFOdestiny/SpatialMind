#!/usr/bin/env bash
###############################################################################
# _pipeline_body.sh - Shared full-pipeline body for the per-backbone runners.
#
# A per-backbone pipeline*.sh sets MODEL_NAME (+ optional overrides), then:
#   source "${SCRIPT_DIR}/_pipeline_body.sh"
# This runs the full flow: (download) -> generate/judge -> train zoo -> evaluate
# ID+OOD -> summary. Every artifact is written under <repo>/spatialmind.
#
# Resume: set RESUME_JOB_ID=<id> to reuse an existing job's results/logs dir.
###############################################################################

set -euo pipefail
PIPELINE_START=$(date +%s)

DATASET_NAME="${DATASET_NAME:-StepGame}"
JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-Mistral-Small-3.2-24B-Instruct-2506}"

source "${SCRIPT_DIR}/common.sh"

# Resume support: pin RESULTS_ROOT/LOGS_ROOT to a prior job id.
if [[ -n "${RESUME_JOB_ID:-}" ]]; then
    RESULTS_ROOT="${BASE_RESULTS_ROOT}/${RESUME_JOB_ID}"
    LOGS_ROOT="${BASE_LOGS_ROOT}/${RESUME_JOB_ID}"
    echo "[RESUME] Using job ${RESUME_JOB_ID}"
fi

read -ra ALL_HEAD_TYPES <<< "${HEAD_TYPES:-${ALL_HEAD_TYPES[*]}}"
OOD_DATASETS=(); read -r -a OOD_DATASETS <<< "${OOD_DATASETS:-spartqa babi SpaRTUN SpaceNLI}"

RUN_LOG="${LOGS_ROOT}/pipeline${RESUME_JOB_ID:+_resume}.log"
mkdir -p "${LOGS_ROOT}"
exec > >(tee -a "${RUN_LOG}") 2>&1

setup_environment
detect_gpus

MODEL_PATH="${MODELS_ROOT}/${MODEL_NAME}"
JUDGE_MODEL_PATH="${MODELS_ROOT}/${JUDGE_MODEL_NAME}"
CLAIM_EXTRACTOR_MODEL_PATH=""
[[ -n "${CLAIM_EXTRACTOR_MODEL_NAME:-}" ]] && CLAIM_EXTRACTOR_MODEL_PATH="${MODELS_ROOT}/${CLAIM_EXTRACTOR_MODEL_NAME}"
if [[ -d "${DATASETS_ROOT}/${DATASET_NAME}/hf_dataset" ]]; then
    DATASET_PATH="${DATASETS_ROOT}/${DATASET_NAME}/hf_dataset"
else
    DATASET_PATH="${DATASETS_ROOT}/${DATASET_NAME}"
fi

for p in "${MODEL_PATH}" "${JUDGE_MODEL_PATH}"; do
    [[ -d "${p}" ]] || { echo "[ERROR] missing model dir: ${p} (run with SKIP_DOWNLOAD=0)"; }
done

print_header "SpatialMind -- Full Pipeline (${MODEL_NAME})"
echo "Job: ${SLURM_JOB_ID:-manual} | Node: $(hostname) | $(date)"
echo "Backbone: ${MODEL_NAME} | Judge: ${JUDGE_MODEL_NAME}"
echo "Heads (${#ALL_HEAD_TYPES[@]}): ${ALL_HEAD_TYPES[*]}"
echo "OOD: ${OOD_DATASETS[*]:-none} | Cache: ${CACHE_DIR}"

source "${SCRIPT_DIR}/p1.sh"
source "${SCRIPT_DIR}/p2.sh"
source "${SCRIPT_DIR}/p3.sh"

SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-1}"
[[ "${SKIP_DOWNLOAD}" != "1" ]] && { run_phase0 || echo "[WARN] downloads had failures"; }

run_phase1 "train,validation,test" "${GEN_MAX_TRAIN},${GEN_MAX_VAL},${GEN_MAX_TEST}" \
    || { echo "[ERROR] Phase 1 failed, aborting."; exit 1; }
run_phase2 || echo "[WARN] Phase 2 had failures"
run_phase3 || echo "[WARN] Phase 3 had failures"

print_step "Summary"
"${PYTHON_BIN:-python}" utils/results.py --results-root "${RESULTS_ROOT}" \
    --title "SpatialMind Summary (${MODEL_NAME})" --figure-dir "${LOGS_ROOT}/figure" || true

echo "Results: ${RESULTS_ROOT}"
echo "Logs:    ${LOGS_ROOT}"
echo "Cache:   ${CACHE_DIR}"
echo "Total time: $(format_duration $(( $(date +%s) - PIPELINE_START )))"
echo "End: $(date)"
