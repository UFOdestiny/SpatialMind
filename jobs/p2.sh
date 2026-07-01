#!/usr/bin/env bash
###############################################################################
# p2.sh - Phase 2: train the head zoo on cached frozen features.
#
# Sourced by pipelines (provides run_phase2), or run directly:
#   sbatch jobs/p2.sh
#   HEAD_TYPES="spatialmind uhead" bash jobs/p2.sh
###############################################################################

#SBATCH --job-name=sm-p2
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120gb
#SBATCH --time=6-23:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.log

_P2_SOURCED=0
[[ "${BASH_SOURCE[0]}" != "${0}" ]] && _P2_SOURCED=1

is_head_complete() {
    [[ -f "${RESULTS_ROOT}/train/$1/final_model/head_weights.pth" ]]
}

train_head() {
    local head_type="$1" gpu_id="${2:-0}"
    local out="${RESULTS_ROOT}/train/${head_type}"
    local log="${LOGS_ROOT}/train/train_${head_type}.log"
    mkdir -p "${out}" "${LOGS_ROOT}/train"
    echo "Training head: ${head_type} (GPU ${gpu_id})"
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN:-python}" scripts/train.py \
        --head_type "${head_type}" \
        --cache_dir "${CACHE_DIR}" \
        --output_dir "${out}" \
        --num_epochs "${TRAIN_EPOCHS:-40}" \
        --batch_size "${TRAIN_BATCH_SIZE:-64}" \
        --learning_rate "${TRAIN_LEARNING_RATE:-0.0002}" \
        --loss_type "${LOSS_TYPE:-bce}" \
        --loss_pos_weight "${LOSS_POS_WEIGHT:-1.0}" \
        --focal_gamma "${FOCAL_GAMMA:-2.0}" \
        --trace_loss_weight "${TRACE_LOSS_WEIGHT:-0.5}" \
        2>&1 | tee "${log}"
    return ${PIPESTATUS[0]}
}

run_phase2() {
    local t0; t0=$(date +%s)
    local fail=0 skip=0
    local heads_to_train=()
    for ht in "${ALL_HEAD_TYPES[@]}"; do
        if is_head_complete "${ht}"; then echo "[SKIP] ${ht}: already trained"; skip=$((skip+1));
        else heads_to_train+=("${ht}"); fi
    done
    [[ ${#heads_to_train[@]} -eq 0 ]] && { echo "All heads trained."; return 0; }

    print_step "Phase 2: Training (${#heads_to_train[@]} heads, ${TRAIN_EPOCHS:-40} epochs)"

    if [[ ${NUM_GPUS:-1} -gt 1 ]]; then
        # Multi-GPU: dynamic slot pool (one head per slot, round-robin GPUs).
        local -a slots=()
        local per_gpu="${MAX_HEADS_PER_GPU:-1}"; [[ "${per_gpu}" -lt 1 ]] && per_gpu=1
        if [[ -n "${MAX_PARALLEL_HEADS:-}" ]]; then
            for ((s=0; s<MAX_PARALLEL_HEADS; s++)); do slots+=("${GPU_IDS[$((s % NUM_GPUS))]}"); done
        else
            for g in "${GPU_IDS[@]}"; do for ((s=0; s<per_gpu; s++)); do slots+=("${g}"); done; done
        fi
        local max=${#slots[@]} next=0 total=${#heads_to_train[@]}
        declare -A pid2head pid2slot
        _start() {
            local idx="$1" slot="$2" g="${slots[$2]}" ht="${heads_to_train[$1]}"
            mkdir -p "${RESULTS_ROOT}/train/${ht}" "${LOGS_ROOT}/train"
            CUDA_VISIBLE_DEVICES="${g}" "${PYTHON_BIN:-python}" scripts/train.py \
                --head_type "${ht}" --cache_dir "${CACHE_DIR}" \
                --output_dir "${RESULTS_ROOT}/train/${ht}" \
                --num_epochs "${TRAIN_EPOCHS:-40}" --batch_size "${TRAIN_BATCH_SIZE:-64}" \
                --learning_rate "${TRAIN_LEARNING_RATE:-0.0002}" --loss_type "${LOSS_TYPE:-bce}" \
                --loss_pos_weight "${LOSS_POS_WEIGHT:-1.0}" --focal_gamma "${FOCAL_GAMMA:-2.0}" \
                --trace_loss_weight "${TRACE_LOSS_WEIGHT:-0.5}" \
                > "${LOGS_ROOT}/train/train_${ht}.log" 2>&1 &
            local pid=$!; pid2head["${pid}"]=${idx}; pid2slot["${pid}"]=${slot}
            echo "[START] slot ${slot} (GPU ${g}): ${ht}"
        }
        for ((s=0; s<max && next<total; s++)); do _start ${next} ${s}; next=$((next+1)); done
        while [[ ${#pid2head[@]} -gt 0 ]]; do
            local fp="" st=0
            if wait -n -p fp 2>/dev/null; then st=0; else st=$?; [[ -z "${fp}" ]] && break; fi
            local slot="${pid2slot[${fp}]}" ht="${heads_to_train[${pid2head[${fp}]}]}"
            [[ ${st} -eq 0 ]] && echo "[OK] ${ht}" || { echo "[FAILED] ${ht} (exit ${st})"; fail=$((fail+1)); }
            unset pid2head["${fp}"] pid2slot["${fp}"]
            if [[ ${next} -lt ${total} ]]; then _start ${next} ${slot}; next=$((next+1)); fi
        done
    else
        for ht in "${heads_to_train[@]}"; do
            train_head "${ht}" "${GPU_IDS[0]:-0}" || fail=$((fail+1))
        done
    fi
    echo "Phase 2 done in $(format_duration $(( $(date +%s) - t0 ))). Skipped ${skip}, failures ${fail}."
    return ${fail}
}

if [[ ${_P2_SOURCED} -eq 0 ]]; then
    set -euo pipefail
    SCRIPT_DIR="/home/dy23a.fsu/popllm/SpatialMind/jobs"
    source "${SCRIPT_DIR}/common.sh"
    [[ -n "${HEAD_TYPES:-}" ]] && read -ra ALL_HEAD_TYPES <<< "${HEAD_TYPES}"
    RUN_LOG="${LOGS_ROOT}/p2.log"; mkdir -p "${LOGS_ROOT}"
    exec > >(tee -a "${RUN_LOG}") 2>&1
    setup_environment; detect_gpus
    print_header "SpatialMind - Phase 2: Train Heads"
    echo "Cache: ${CACHE_DIR} | Heads: ${ALL_HEAD_TYPES[*]}"
    run_phase2
fi
