#!/usr/bin/env bash
###############################################################################
# p1.sh - Phase 0 (download) + Phase 1 (generate -> claim-extract -> judge)
#
# Sourced by pipeline scripts (provides run_phase0 / run_phase1), or run directly:
#   sbatch jobs/p1.sh
#   DATASET_NAME=StepGame MODEL_NAME=Llama-3.1-8B-Instruct bash jobs/p1.sh
###############################################################################

#SBATCH --job-name=sm-p1
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200gb
#SBATCH --time=6-23:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.log

_P1_SOURCED=0
[[ "${BASH_SOURCE[0]}" != "${0}" ]] && _P1_SOURCED=1

###############################################################################
# Phase 0: downloads
###############################################################################
download_model() {
    local repo_id="$1" local_name="$2" force="${3:-0}"
    local dir="${MODELS_ROOT}/${local_name}"
    if [[ "${force}" != "1" && -f "${dir}/config.json" ]] && \
       ( ls "${dir}"/*.safetensors >/dev/null 2>&1 || ls "${dir}"/*.bin >/dev/null 2>&1 ); then
        echo "  [SKIP] ${local_name}"; return 0
    fi
    echo "  Downloading model ${repo_id} -> ${local_name}"
    local flag=""; [[ "${force}" == "1" ]] && flag="--force"
    "${PYTHON_BIN:-python}" utils/download_models.py --repo_id "${repo_id}" --local_name "${local_name}" ${flag} \
        2>&1 | tee "${LOGS_ROOT}/download/model_${local_name}.log"
    return ${PIPESTATUS[0]}
}

download_dataset() {
    local repo_id="$1" local_name="$2" force="${3:-0}"
    local dir="${DATASETS_ROOT}/${local_name}"
    if [[ "${force}" != "1" && -d "${dir}" ]]; then echo "  [SKIP] ${local_name}"; return 0; fi
    echo "  Downloading dataset ${repo_id} -> ${local_name}"
    local flag=""; [[ "${force}" == "1" ]] && flag="--force"
    "${PYTHON_BIN:-python}" utils/download_dataset.py --repo_id "${repo_id}" --local_name "${local_name}" ${flag} \
        2>&1 | tee "${LOGS_ROOT}/download/dataset_${local_name}.log"
    return ${PIPESTATUS[0]}
}

_lookup() {  # _lookup <local_name> <array-name>
    local target="$1"; shift
    local entry
    for entry in "$@"; do
        [[ "${entry%%:*}" == "${target}" ]] && { echo "${entry#*:}"; return 0; }
    done
    echo ""; return 1
}

run_phase0() {
    print_step "Phase 0: Download Models & Datasets"
    local fail=0
    declare -A models=(["${MODEL_NAME}"]=1)
    [[ -n "${JUDGE_MODEL_NAME:-}" ]] && models["${JUDGE_MODEL_NAME}"]=1
    [[ -n "${CLAIM_EXTRACTOR_MODEL_NAME:-}" ]] && models["${CLAIM_EXTRACTOR_MODEL_NAME}"]=1
    declare -A datasets=(["${DATASET_NAME}"]=1)
    for ood in "${OOD_DATASETS[@]:-}"; do [[ -n "${ood}" ]] && datasets["${ood}"]=1; done

    echo "--- Models ---"
    for m in "${!models[@]}"; do
        local repo; repo=$(_lookup "${m}" "${DOWNLOAD_MODELS[@]}")
        [[ -z "${repo}" ]] && { echo "  [WARN] no repo for model ${m}"; fail=$((fail+1)); continue; }
        download_model "${repo}" "${m}" "${FORCE_DOWNLOAD:-0}" || fail=$((fail+1))
    done
    echo "--- Datasets ---"
    for d in "${!datasets[@]}"; do
        local repo; repo=$(_lookup "${d}" "${DOWNLOAD_DATASETS[@]}")
        [[ -z "${repo}" ]] && { echo "  [WARN] no repo for dataset ${d}"; fail=$((fail+1)); continue; }
        download_dataset "${repo}" "${d}" "${FORCE_DOWNLOAD:-0}" || fail=$((fail+1))
    done
    return ${fail}
}

###############################################################################
# Phase 1: generate (+ optional deferred claim extraction) + judge
###############################################################################
run_generation() {
    local splits="${1:-train,validation,test}"
    local max_samples="${2:-${GEN_MAX_TRAIN},${GEN_MAX_VAL},${GEN_MAX_TEST}}"
    local gen_log="${LOGS_ROOT}/data/generate_${DATASET_NAME}.log"

    if cache_splits_ready "${CACHE_DIR}" "${splits}"; then
        echo "  [SKIP] Generation cache complete for ${splits}."
    else
        local claim_args=()
        if [[ -n "${CLAIM_EXTRACTOR_MODEL_PATH:-}" && -d "${CLAIM_EXTRACTOR_MODEL_PATH}" && "${DEFER_CLAIM_EXTRACTION}" == "1" ]]; then
            claim_args+=(--defer_claim_extraction)
        fi
        "${PYTHON_BIN:-python}" scripts/generate.py \
            --dataset "${DATASET_NAME}" \
            --split "${splits}" \
            --max_samples "${max_samples}" \
            --max_new_tokens "${GEN_MAX_NEW_TOKENS:-256}" \
            --backend "${BACKEND:-vllm}" \
            --model_path "${MODEL_PATH}" \
            --dataset_path "${DATASET_PATH}" \
            --cache_dir "${CACHE_DIR}" \
            --batch_size "${GEN_BATCH_SIZE:-32}" \
            --free_form_batch_size "${FREE_FORM_GEN_BATCH_SIZE:-8}" \
            --skip_existing \
            "${claim_args[@]}" \
            2>&1 | tee "${gen_log}"
        [[ ${PIPESTATUS[0]} -ne 0 ]] && { echo "  [FAILED] Generation"; return 1; }
    fi

    # Deferred LLM claim extraction (optional).
    if [[ -n "${CLAIM_EXTRACTOR_MODEL_PATH:-}" && -d "${CLAIM_EXTRACTOR_MODEL_PATH}" && "${DEFER_CLAIM_EXTRACTION}" == "1" ]]; then
        echo "  Running claim extraction (stage 2)..."
        "${PYTHON_BIN:-python}" scripts/claim_extract.py \
            --cache_dir "${CACHE_DIR}" --split "${splits}" \
            --claim_extractor_model "${CLAIM_EXTRACTOR_MODEL_PATH}" \
            --backend "${CLAIM_EXTRACTOR_BACKEND:-vllm}" \
            --max_new_tokens "${CLAIM_EXTRACTOR_MAX_NEW_TOKENS:-256}" \
            --batch_size "${GEN_BATCH_SIZE:-32}" --skip_existing \
            2>&1 | tee "${LOGS_ROOT}/data/claim_extract_${DATASET_NAME}.log"
    fi
    return 0
}

run_judge() {
    local splits="${1:-train,validation,test}"
    "${PYTHON_BIN:-python}" scripts/judge.py \
        --cache_dir "${CACHE_DIR}" --split "${splits}" \
        --judge_model "${JUDGE_MODEL_PATH}" --judge_backend "${BACKEND:-vllm}" \
        --judge_max_new_tokens "${JUDGE_MAX_NEW_TOKENS:-256}" \
        2>&1 | tee "${LOGS_ROOT}/data/judge_${DATASET_NAME}.log"
    return ${PIPESTATUS[0]}
}

run_phase1() {
    local splits="${1:-train,validation,test}"
    local max_samples="${2:-${GEN_MAX_TRAIN},${GEN_MAX_VAL},${GEN_MAX_TEST}}"
    local t0; t0=$(date +%s)
    print_step "Phase 1: Generate & Cache Features"
    run_generation "${splits}" "${max_samples}" || return $?
    echo ""
    print_step "Phase 1.5: LLM-as-Judge (claim + trace labels)"
    if should_run_judge_for_splits "${CACHE_DIR}" "${splits}" "phase1"; then
        run_judge "${splits}"
    else
        echo "  [SKIP] Judge: all splits below pending threshold."
    fi
    echo "Phase 1 completed in $(format_duration $(( $(date +%s) - t0 )))."
}

###############################################################################
# Standalone execution
###############################################################################
if [[ ${_P1_SOURCED} -eq 0 ]]; then
    set -euo pipefail
    SCRIPT_DIR="/home/dy23a.fsu/popllm/SpatialMind/jobs"
    source "${SCRIPT_DIR}/common.sh"
    OOD_DATASETS=(); read -r -a OOD_DATASETS <<< "${OOD_DATASETS:-}"
    RUN_LOG="${LOGS_ROOT}/p1.log"; mkdir -p "${LOGS_ROOT}"
    exec > >(tee -a "${RUN_LOG}") 2>&1
    setup_environment

    MODEL_PATH="${MODELS_ROOT}/${MODEL_NAME}"
    JUDGE_MODEL_PATH="${MODELS_ROOT}/${JUDGE_MODEL_NAME}"
    CLAIM_EXTRACTOR_MODEL_PATH=""
    [[ -n "${CLAIM_EXTRACTOR_MODEL_NAME:-}" ]] && CLAIM_EXTRACTOR_MODEL_PATH="${MODELS_ROOT}/${CLAIM_EXTRACTOR_MODEL_NAME}"
    if [[ -d "${DATASETS_ROOT}/${DATASET_NAME}/hf_dataset" ]]; then
        DATASET_PATH="${DATASETS_ROOT}/${DATASET_NAME}/hf_dataset"
    else
        DATASET_PATH="${DATASETS_ROOT}/${DATASET_NAME}"
    fi

    print_header "SpatialMind - Phase 0 & 1"
    echo "Dataset: ${DATASET_NAME} | Backbone: ${MODEL_NAME} | Judge: ${JUDGE_MODEL_NAME}"
    echo "Cache:   ${CACHE_DIR}"

    SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-1}"
    if [[ "${SKIP_DOWNLOAD}" != "1" ]]; then run_phase0 || echo "[WARN] downloads had failures"; fi
    run_phase1 "train,validation,test" "${GEN_MAX_TRAIN},${GEN_MAX_VAL},${GEN_MAX_TEST}"
fi
