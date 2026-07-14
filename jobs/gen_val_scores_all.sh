#!/usr/bin/env bash
###############################################################################
# gen_val_scores_all.sh - fill in VALIDATION-split predictions for ALL candidate
# UQ signals (supervised heads + unsupervised baselines), for every backbone and
# dataset, so a multi-signal fusion combiner can be fit honestly (no test peek).
#
# Eval-only from existing caches (no generation). Idempotent: skips done work.
###############################################################################
set -uo pipefail
SCRIPT_DIR="/home/dy23a.fsu/popllm/SpatialMind/jobs"
cd /home/dy23a.fsu/popllm/SpatialMind
source "${SCRIPT_DIR}/common.sh"

# backbone tag -> MODEL_NAME
declare -A MODELS=(
  [20260712]=Llama-3.1-8B-Instruct
  [mistral7b]=Mistral-7B-Instruct-v0.3
  [gemma2]=gemma-2-9b-it
  [phi4reason]=Phi-4-reasoning
)
# supervised heads to score on validation (must have been trained in the run)
SUP_HEADS="constraint_no_conflict constraint_only spatialmind_neural spatialmind uhead factoscope mlp"
# unsupervised baselines to score on validation
BASE_LIST="constraint_rule,ccp,mcp,perplexity,token_entropy,random"
declare -A CN=( [id]=StepGame [spartqa]=spartqa [babi]=babi [SpaRTUN]=SpaRTUN [SpaceNLI]=SpaceNLI [SpartQA_YN]=SpartQA_YN )

setup_environment
detect_gpus

for tag in "${!MODELS[@]}"; do
  MODEL_NAME="${MODELS[$tag]}"
  if [[ "$tag" == "20260712" ]]; then SUB="constraint_guided_v10"; else SUB="constraint_guided_v10_${tag}"; fi
  R="/home/dy23a.fsu/popllm/SpatialMind/spatialmind/results/constraint_guided_v10_${tag}"
  [[ "$tag" == "20260712" ]] && R="/home/dy23a.fsu/popllm/SpatialMind/spatialmind/results/constraint_guided_v10_20260712"
  LG="/home/dy23a.fsu/popllm/SpatialMind/spatialmind/logs/valscores_${tag}"; mkdir -p "$LG"
  echo "############ backbone=$tag model=$MODEL_NAME ############"
  for ds in id spartqa babi SpaRTUN SpaceNLI SpartQA_YN; do
    dname="${CN[$ds]}"
    if [[ "$ds" == "id" ]]; then cache="${CACHE_ROOT}/cached_features/${SUB}/StepGame/${MODEL_NAME}";
    else cache="${CACHE_ROOT}/cached_features/${SUB}/${ds}/${MODEL_NAME}"; fi
    [[ -d "$cache/validation" ]] || { echo "[skip] $tag/$ds no val cache"; continue; }

    # supervised heads
    for ht in $SUP_HEADS; do
      hp="${R}/train/${ht}/final_model"
      [[ -f "${hp}/head_weights.pth" ]] || continue
      out="${R}/val_scores/${dname}/${ht}"
      [[ -f "${out}/evaluation_report.json" ]] && continue
      mkdir -p "$out"
      CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
        --head_path "${hp}" --cache_dir "${cache}" --output_dir "${out}" \
        --split validation --batch_size 256 --calibrate standard --struct_calib_C 0.01 \
        > "${LG}/val_${ds}_${ht}.log" 2>&1
      echo "[$([[ $? -eq 0 ]] && echo OK || echo FAIL)] $tag/$ds/$ht"
    done

    # unsupervised baselines (val split)
    bout="${R}/val_scores_baselines/${dname}"
    if [[ ! -f "${bout}/combined_evaluation.json" ]]; then
      mkdir -p "$bout"
      CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
        --cache_dir "${cache}" --output_dir "${bout}" --split validation \
        --eval_baselines --baselines "${BASE_LIST}" --batch_size 256 \
        --calibrate standard --struct_calib_C 0.01 \
        > "${LG}/val_${ds}_baselines.log" 2>&1
      echo "[$([[ $? -eq 0 ]] && echo OK || echo FAIL)] $tag/$ds/baselines"
    fi
  done
done
echo "### gen_val_scores_all DONE $(date)"
