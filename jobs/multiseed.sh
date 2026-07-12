#!/usr/bin/env bash
# Repeat head training/evaluation on one immutable native cache.
# Example:
#   CACHE_SUBDIR=constraint_v3 SEEDS="2026 2027 2028 2029 2030" sbatch jobs/multiseed.sh

#SBATCH --job-name=sm-seeds
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
SCRIPT_DIR="/home/dy23a.fsu/popllm/SpatialMind/jobs"
source "${SCRIPT_DIR}/common.sh"
setup_environment
detect_gpus

SEEDS_CONFIG="${SEEDS:-2026 2027 2028 2029 2030}"
HEADS_CONFIG="${HEAD_TYPES:-spatialmind constraint_only spatialmind_neural mlp uhead factoscope}"
read -ra SEED_LIST <<< "${SEEDS_CONFIG}"
read -ra HEAD_LIST <<< "${HEADS_CONFIG}"

for seed in "${SEED_LIST[@]}"; do
    for head in "${HEAD_LIST[@]}"; do
        out="${RESULTS_ROOT}/seed_${seed}/train/${head}"
        eval_out="${RESULTS_ROOT}/seed_${seed}/eval/${head}"
        if [[ ! -f "${out}/final_model/head_weights.pth" ]]; then
            CUDA_VISIBLE_DEVICES="${GPU_IDS[0]:-0}" "${PYTHON_BIN}" scripts/train.py \
                --head_type "${head}" --cache_dir "${CACHE_DIR}" --output_dir "${out}" \
                --num_epochs "${TRAIN_EPOCHS}" --batch_size "${TRAIN_BATCH_SIZE}" \
                --learning_rate "${TRAIN_LEARNING_RATE}" --loss_type "${LOSS_TYPE}" \
                --loss_pos_weight "${LOSS_POS_WEIGHT}" --focal_gamma "${FOCAL_GAMMA}" \
                --trace_loss_weight "${TRACE_LOSS_WEIGHT}" --seed "${seed}"
        fi
        if [[ ! -f "${eval_out}/evaluation_report.json" ]]; then
            CUDA_VISIBLE_DEVICES="${GPU_IDS[0]:-0}" "${PYTHON_BIN}" scripts/evaluate.py \
                --head_path "${out}/final_model" --cache_dir "${CACHE_DIR}" \
                --output_dir "${eval_out}" --split test --batch_size "${TRAIN_BATCH_SIZE}" \
                --calibrate standard
        fi
    done
done

"${PYTHON_BIN}" scripts/aggregate_seeds.py --results_root "${RESULTS_ROOT}" \
    --reference spatialmind --output "${RESULTS_ROOT}/multiseed_summary.json"
