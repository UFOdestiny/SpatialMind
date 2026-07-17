#!/usr/bin/env bash
#SBATCH --job-name=smind-sample-sparp
#SBATCH --account=fsu-compsci-dept
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200gb
#SBATCH --time=1-00:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.log
###############################################################################
# Sampling baselines (Semantic Entropy + SelfCheckGPT + P(True)) for the two
# SpaRP probe datasets. Sizes match run_sparp_probe.sh (val 1000 / test 1500).
#   sbatch jobs/sbatch_sampling_sparp.sh   OR   bash jobs/sbatch_sampling_sparp.sh
###############################################################################
set -uo pipefail
SCRIPT_DIR="/home/dy23a.fsu/popllm/SpatialMind/jobs"
cd /home/dy23a.fsu/popllm/SpatialMind
export MODEL_NAME="Llama-3.1-8B-Instruct"; export RUN_TAG="llama"
export DATASET_NAME="StepGame"; export CACHE_SUBDIR="constraint_guided_v11_llama"
source "${SCRIPT_DIR}/common.sh"; setup_environment
export VLLM_WORKER_MULTIPROC_METHOD=spawn
RESULTS_ROOT="/home/dy23a.fsu/popllm/SpatialMind/spatialmind/results/constraint_guided_v11_llama"
MODEL_PATH="${MODELS_ROOT}/${MODEL_NAME}"
for ds in SpaRP_PS1 SpaRP_PS3; do
    out="${RESULTS_ROOT}/eval_ood/${ds}/baselines_sampling"
    [[ -f "${out}/combined_evaluation.json" ]] && { echo "[SKIP] ${ds}"; continue; }
    echo "===== sampling baselines: ${ds} $(date) ====="
    python scripts/sampling_baselines.py \
        --model_path "${MODEL_PATH}" --dataset_name "${ds}" \
        --dataset_path "${DATASETS_ROOT}/${ds}/hf_dataset" \
        --out_dir "${out}" --max_val 1000 --max_test 1500 \
        --K "${K:-10}" --temp 0.7 --gpu_frac 0.85 --max_len 2048 || echo "[FAIL] ${ds}"
done
echo "done $(date)"
