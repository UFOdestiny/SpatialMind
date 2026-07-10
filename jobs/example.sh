#!/usr/bin/env bash
###############################################################################
# example.sh - SpatialMind quick-test pipeline (small samples, all phases).
#
#   sbatch jobs/example.sh
#   bash jobs/example.sh                          # run without SLURM
#   GEN_MAX_TRAIN=2000 TRAIN_EPOCHS=15 bash jobs/example.sh
#
# Small-scale end-to-end smoke test on StepGame(ID) -> spartqa/babi(OOD).
# Uses a separate cache namespace (CACHE_SUBDIR=example) so it never collides
# with a full run. All artifacts land under <repo>/spatialmind.
###############################################################################

#SBATCH --job-name=smind-ex
#SBATCH --account=fsu-compsci-dept
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200gb
#SBATCH --time=6-23:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.log

set -euo pipefail
PIPELINE_START=$(date +%s)
SCRIPT_DIR="/home/dy23a.fsu/popllm/SpatialMind/jobs"

# ---- quick-test configuration ----
DATASET_NAME="${DATASET_NAME:-StepGame}"
MODEL_NAME="${MODEL_NAME:-Llama-3.1-8B-Instruct}"
JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-Mistral-Small-3.2-24B-Instruct-2506}"
CACHE_SUBDIR="example"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-30}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1024}"
GEN_MAX_TRAIN="${GEN_MAX_TRAIN:-20000}"
GEN_MAX_VAL="${GEN_MAX_VAL:-5000}"
GEN_MAX_TEST="${GEN_MAX_TEST:-5000}"
GEN_MAX_OOD="${GEN_MAX_OOD:-5000}"
# Quick-test head subset (full zoo runs in pipeline*.sh).
HEAD_TYPES="${HEAD_TYPES:-spatialmind uhead saplma factoscope lookback_lens luh_light mlp}"

source "${SCRIPT_DIR}/common.sh"
read -ra ALL_HEAD_TYPES <<< "${HEAD_TYPES}"
OOD_DATASETS=(); read -r -a OOD_DATASETS <<< "${OOD_DATASETS:-spartqa babi SpaRTUN SpaceNLI}"

RUN_LOG="${LOGS_ROOT}/example.log"; mkdir -p "${LOGS_ROOT}"
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

print_header "SpatialMind -- Quick-Test Pipeline"
echo "Job: ${SLURM_JOB_ID:-manual} | Node: $(hostname) | $(date)"
echo "Backbone: ${MODEL_NAME} | Judge: ${JUDGE_MODEL_NAME}"
echo "Heads: ${ALL_HEAD_TYPES[*]}"
echo "Samples: train=${GEN_MAX_TRAIN} val=${GEN_MAX_VAL} test=${GEN_MAX_TEST} ood=${GEN_MAX_OOD}"
echo "OOD: ${OOD_DATASETS[*]:-none}"
echo "Cache: ${CACHE_DIR}"

source "${SCRIPT_DIR}/p1.sh"
source "${SCRIPT_DIR}/p2.sh"
source "${SCRIPT_DIR}/p3.sh"

SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-1}"
[[ "${SKIP_DOWNLOAD}" != "1" ]] && { run_phase0 || echo "[WARN] downloads had failures"; }

run_phase1 "train,validation,test" "${GEN_MAX_TRAIN},${GEN_MAX_VAL},${GEN_MAX_TEST}" \
    || { echo "[ERROR] Phase 1 failed"; exit 1; }
run_phase2 || echo "[WARN] Phase 2 had failures"
run_phase3 || echo "[WARN] Phase 3 had failures"

print_step "Summary"
"${PYTHON_BIN:-python}" utils/results.py --results-root "${RESULTS_ROOT}" \
    --title "SpatialMind Quick-Test Summary" --figure-dir "${LOGS_ROOT}/figure" || true

echo "Results: ${RESULTS_ROOT}"
echo "Logs:    ${LOGS_ROOT}"
echo "Cache:   ${CACHE_DIR}"
echo "Total time: $(format_duration $(( $(date +%s) - PIPELINE_START )))"
echo "End: $(date)"
