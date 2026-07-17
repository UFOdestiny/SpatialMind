#!/usr/bin/env bash
#SBATCH --job-name=smind-sample-llama-v11
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
# Sampling-based UQ baselines (Semantic Entropy + P(True)) for the Llama v11
# backbone, across all five headline datasets. Independent of the main pipeline;
# writes into a SEPARATE baselines_sampling/ dir so it never clobbers the
# calibrated single-pass baselines. K=10 stochastic decodes per sample.
#
# Sizes MUST match the main v11 run (val 2000 / test 3000) so sample_id aligns.
#
#   sbatch jobs/sbatch_sampling_llama_v11.sh
###############################################################################
set -uo pipefail
SCRIPT_DIR="/home/dy23a.fsu/popllm/SpatialMind/jobs"
cd /home/dy23a.fsu/popllm/SpatialMind

export MODEL_NAME="Llama-3.1-8B-Instruct"
export RUN_TAG="llama"
export DATASET_NAME="StepGame"          # placeholder; per-dataset loop below
export CACHE_SUBDIR="constraint_guided_v11_${RUN_TAG}"

source "${SCRIPT_DIR}/common.sh"
setup_environment
# Safeguard: force spawn so vLLM workers can init CUDA even if any library
# touched a CUDA context in the parent process.
export VLLM_WORKER_MULTIPROC_METHOD=spawn

RESULTS_ROOT="/home/dy23a.fsu/popllm/SpatialMind/spatialmind/results/constraint_guided_v11_${RUN_TAG}"
MODEL_PATH="${MODELS_ROOT}/${MODEL_NAME}"

K="${K:-10}"
TEMP="${TEMP:-0.7}"
MAX_VAL="${MAX_VAL:-2000}"
MAX_TEST="${MAX_TEST:-3000}"
GPU_FRAC="${GPU_FRAC:-0.85}"

# dataset_name : results_subdir (id -> eval, others -> eval_ood/<name>)
declare -A DS_DIR=( [StepGame]="eval" [spartqa]="eval_ood/spartqa" \
                    [babi]="eval_ood/babi" [SpaRTUN]="eval_ood/SpaRTUN" \
                    [SpaceNLI]="eval_ood/SpaceNLI" )

for ds in StepGame spartqa babi SpaRTUN SpaceNLI; do
    out="${RESULTS_ROOT}/${DS_DIR[$ds]}/baselines_sampling"
    if [[ -f "${out}/combined_evaluation.json" ]]; then
        echo "[SKIP] ${ds} (already done)"; continue
    fi
    echo "===== sampling baselines: ${ds} $(date) ====="
    python scripts/sampling_baselines.py \
        --model_path "${MODEL_PATH}" \
        --dataset_name "${ds}" \
        --dataset_path "${DATASETS_ROOT}/${ds}/hf_dataset" \
        --out_dir "${out}" \
        --max_val "${MAX_VAL}" --max_test "${MAX_TEST}" \
        --K "${K}" --temp "${TEMP}" --gpu_frac "${GPU_FRAC}" \
        --max_len "${VLLM_MAX_MODEL_LEN:-2048}" \
        || echo "[FAIL] ${ds}"
done
echo "===== sampling baselines done $(date) ====="
