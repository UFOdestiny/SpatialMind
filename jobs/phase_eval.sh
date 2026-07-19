#!/usr/bin/env bash
###############################################################################
# phase_eval.sh - sample-level evaluation (ID + OOD) for heads + baselines.
#
# Sourced by run scripts (provides run_phase3), or run directly:
#   sbatch jobs/phase_eval.sh
#   HEAD_TYPES="spatialmind uhead" bash jobs/phase_eval.sh
#
# Layout produced (consumed by utils/results.py):
#   ${RESULTS_ROOT}/eval/<head>/evaluation_report.json                (ID head)
#   ${RESULTS_ROOT}/eval/baselines/<name>/evaluation_report.json      (ID baselines)
#   ${RESULTS_ROOT}/eval_ood/<dataset>/<head>/evaluation_report.json   (OOD head)
###############################################################################

#SBATCH --job-name=sm-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120gb
#SBATCH --time=6-23:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.log

_P3_SOURCED=0
[[ "${BASH_SOURCE[0]}" != "${0}" ]] && _P3_SOURCED=1

_eval_gpu() { echo "${CUDA_VISIBLE_DEVICES%%,*}"; }

is_eval_complete() {
    [[ -f "${RESULTS_ROOT}/eval/$1/evaluation_report.json" ]]
}
is_ood_eval_complete() {
    [[ -f "${RESULTS_ROOT}/eval_ood/$1/$2/evaluation_report.json" ]]
}

eval_head() {
    local head_type="$1" cache_dir="${2:-${CACHE_DIR}}" out_dir="$3"
    local log="${LOGS_ROOT}/eval/eval_$(basename "${out_dir}").log"
    mkdir -p "${out_dir}" "${LOGS_ROOT}/eval"
    if [[ ! -d "${RESULTS_ROOT}/train/${head_type}/final_model" ]]; then
        echo "  [SKIP] no trained model for ${head_type}"; return 1
    fi
    CUDA_VISIBLE_DEVICES="$(_eval_gpu)" "${PYTHON_BIN:-python}" scripts/evaluate.py \
        --head_path "${RESULTS_ROOT}/train/${head_type}/final_model" \
        --cache_dir "${cache_dir}" --output_dir "${out_dir}" \
        --split test --batch_size "${TRAIN_BATCH_SIZE:-128}" \
        --calibrate "${CALIBRATE:-standard}" --struct_calib_C "${STRUCT_CALIB_C:-0.01}" \
        2>&1 | tee "${log}"
    return ${PIPESTATUS[0]}
}

eval_baselines() {
    local cache_dir="${1:-${CACHE_DIR}}" out_dir="$2"
    mkdir -p "${out_dir}" "${LOGS_ROOT}/eval"
    CUDA_VISIBLE_DEVICES="$(_eval_gpu)" "${PYTHON_BIN:-python}" scripts/evaluate.py \
        --cache_dir "${cache_dir}" --output_dir "${out_dir}" \
        --split test --eval_baselines --batch_size "${TRAIN_BATCH_SIZE:-128}" \
        --calibrate "${CALIBRATE:-standard}" --struct_calib_C "${STRUCT_CALIB_C:-0.01}" \
        2>&1 | tee "${LOGS_ROOT}/eval/eval_$(basename "${out_dir}")_baselines.log"
    return ${PIPESTATUS[0]}
}

# OOD cache dir mirrors the ID layout under CACHE_ROOT.
_ood_cache_dir() {
    local ood="$1"
    if [[ -n "${CACHE_SUBDIR}" ]]; then
        echo "${CACHE_ROOT}/cached_features/${CACHE_SUBDIR}/${ood}/${MODEL_NAME}"
    else
        echo "${CACHE_ROOT}/cached_features/${ood}/${MODEL_NAME}"
    fi
}

_ood_dataset_path() {
    local ood="$1"
    if [[ -d "${DATASETS_ROOT}/${ood}/hf_dataset" ]]; then echo "${DATASETS_ROOT}/${ood}/hf_dataset";
    else echo "${DATASETS_ROOT}/${ood}"; fi
}

prepare_ood_cache() {
    # Generate + judge BOTH the OOD validation and test splits. The validation
    # split calibrates the OOD transfer exactly like ID (aggregation rule for the
    # head, uncertainty->confidence normalization for baselines) so that no test
    # statistics ever leak, and OOD ECE/Acc are properly calibrated.
    local ood="$1" ood_cache; ood_cache="$(_ood_cache_dir "${ood}")"
    mkdir -p "${ood_cache}"
    local ood_splits="validation,test"
    if ! cache_splits_ready "${ood_cache}" "${ood_splits}"; then
        echo "  Generating OOD validation+test cache for ${ood}..."
        "${PYTHON_BIN:-python}" scripts/generate.py \
            --dataset "${ood}" --split "${ood_splits}" \
            --max_samples "${GEN_MAX_OOD_VAL:-${GEN_MAX_OOD:-0}},${GEN_MAX_OOD:-0}" \
            --max_new_tokens "${GEN_MAX_NEW_TOKENS:-768}" --backend "${BACKEND:-vllm}" \
            --model_path "${MODEL_PATH}" --dataset_path "$(_ood_dataset_path "${ood}")" \
            --cache_dir "${ood_cache}" --batch_size "${GEN_BATCH_SIZE:-32}" \
            --free_form_batch_size "${FREE_FORM_GEN_BATCH_SIZE:-8}" --skip_existing \
            2>&1 | tee "${LOGS_ROOT}/data/generate_ood_${ood}.log"
    fi
    local ignore_force=0
    [[ -n "${FORCE_OOD_JUDGE_DATASETS:-}" ]] && ignore_force=1
    local forced=0
    for d in ${FORCE_OOD_JUDGE_DATASETS:-}; do [[ "${d}" == "${ood}" ]] && forced=1; done
    if [[ "${forced}" == "1" ]] || should_run_judge_for_splits "${ood_cache}" "${ood_splits}" "ood/${ood}" "${ignore_force}"; then
        "${PYTHON_BIN:-python}" scripts/judge.py \
            --cache_dir "${ood_cache}" --split "${ood_splits}" \
            --judge_model "${JUDGE_MODEL_PATH}" --judge_backend "${BACKEND:-vllm}" \
            --judge_max_new_tokens "${JUDGE_MAX_NEW_TOKENS:-512}" \
            2>&1 | tee "${LOGS_ROOT}/data/judge_ood_${ood}.log"
    fi
}

run_phase3() {
    local t0; t0=$(date +%s); local fail=0

    print_step "Phase 3: In-Distribution Evaluation"
    for ht in "${ALL_HEAD_TYPES[@]}"; do
        if is_eval_complete "${ht}"; then echo "[SKIP] ${ht}: already evaluated"; continue; fi
        eval_head "${ht}" "${CACHE_DIR}" "${RESULTS_ROOT}/eval/${ht}" || fail=$((fail+1))
    done
    if [[ ! -f "${RESULTS_ROOT}/eval/baselines/combined_evaluation.json" ]]; then
        eval_baselines "${CACHE_DIR}" "${RESULTS_ROOT}/eval/baselines" || fail=$((fail+1))
    else
        echo "[SKIP] baselines: already evaluated"
    fi

    if [[ ${#OOD_DATASETS[@]} -gt 0 ]]; then
        print_step "Phase 3b: OOD Evaluation (${OOD_DATASETS[*]})"
        for ood in "${OOD_DATASETS[@]}"; do
            [[ "${ood}" == "${DATASET_NAME}" ]] && { echo "skip ${ood} (== train dataset)"; continue; }
            prepare_ood_cache "${ood}"
            local ood_cache; ood_cache="$(_ood_cache_dir "${ood}")"
            if ! cache_split_ready "${ood_cache}" "test"; then
                echo "  [FAILED] OOD cache missing for ${ood}"; fail=$((fail+1)); continue
            fi
            if ! cache_split_ready "${ood_cache}" "validation"; then
                echo "  [FAILED] OOD ${ood}: strict protocol requires validation; no test fallback."
                fail=$((fail+1)); continue
            fi
            for ht in "${ALL_HEAD_TYPES[@]}"; do
                if is_ood_eval_complete "${ood}" "${ht}"; then echo "  [SKIP] OOD ${ood}/${ht}"; continue; fi
                eval_head "${ht}" "${ood_cache}" "${RESULTS_ROOT}/eval_ood/${ood}/${ht}" || fail=$((fail+1))
            done
            if [[ ! -f "${RESULTS_ROOT}/eval_ood/${ood}/baselines/combined_evaluation.json" ]]; then
                eval_baselines "${ood_cache}" "${RESULTS_ROOT}/eval_ood/${ood}/baselines" || fail=$((fail+1))
            fi
        done
    fi
    echo "Phase 3 done in $(format_duration $(( $(date +%s) - t0 ))). Failures ${fail}."
    return ${fail}
}

if [[ ${_P3_SOURCED} -eq 0 ]]; then
    set -euo pipefail
    SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    source "${SCRIPT_DIR}/common.sh"
    [[ -n "${HEAD_TYPES:-}" ]] && read -ra ALL_HEAD_TYPES <<< "${HEAD_TYPES}"
    OOD_DATASETS=(); read -r -a OOD_DATASETS <<< "${OOD_DATASETS:-}"
    RUN_LOG="${LOGS_ROOT}/phase_eval.log"; mkdir -p "${LOGS_ROOT}"
    exec > >(tee -a "${RUN_LOG}") 2>&1
    setup_environment; detect_gpus
    MODEL_PATH="${MODELS_ROOT}/${MODEL_NAME}"
    JUDGE_MODEL_PATH="${MODELS_ROOT}/${JUDGE_MODEL_NAME}"
    print_header "SpatialMind - Phase 3: Evaluate"
    echo "Cache: ${CACHE_DIR} | Heads: ${ALL_HEAD_TYPES[*]} | OOD: ${OOD_DATASETS[*]:-none}"
    run_phase3
    "${PYTHON_BIN:-python}" utils/results.py --results-root "${RESULTS_ROOT}" \
        --title "SpatialMind Summary" --figure-dir "${LOGS_ROOT}/figure" || true
fi
