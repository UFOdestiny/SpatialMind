#!/usr/bin/env bash
###############################################################################
# run_v9_train.sh - Phase 2 training driver for the constraint_guided_v9 cache.
# Sequential single-GPU training of the full head zoo for the novelty method.
###############################################################################
set -uo pipefail
cd /home/dy23a.fsu/popllm/SpatialMind

CACHE="spatialmind/cache/cached_features/constraint_guided_v9/StepGame/Llama-3.1-8B-Instruct"
JOB="constraint_guided_v9_20260712"
RES="spatialmind/results/${JOB}"
LOG="spatialmind/logs/${JOB}"
mkdir -p "${RES}/train" "${LOG}/train"

EPOCHS="${TRAIN_EPOCHS:-30}"
BATCH="${TRAIN_BATCH_SIZE:-128}"
LR="${TRAIN_LEARNING_RATE:-0.0002}"

# main method, pure controls, constraint ablations, strong baselines
HEADS=(
  spatialmind
  constraint_only
  spatialmind_neural
  constraint_no_conflict
  constraint_no_context
  constraint_no_entailment
  constraint_no_repair
  uhead
  factoscope
  mlp
)

echo "=== Phase 2 v9 training start $(date) ==="
echo "cache=${CACHE}"
echo "heads=${HEADS[*]}"

for ht in "${HEADS[@]}"; do
  out="${RES}/train/${ht}"
  if [[ -f "${out}/final_model/head_weights.pth" ]]; then
    echo "[SKIP] ${ht}: already trained"; continue
  fi
  mkdir -p "${out}"
  echo "[TRAIN] ${ht} $(date)"
  CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --head_type "${ht}" \
    --cache_dir "${CACHE}" \
    --output_dir "${out}" \
    --num_epochs "${EPOCHS}" \
    --batch_size "${BATCH}" \
    --learning_rate "${LR}" \
    --loss_type bce \
    --trace_loss_weight 0.5 \
    > "${LOG}/train/train_${ht}.log" 2>&1
  rc=$?
  if [[ ${rc} -eq 0 ]]; then echo "[OK] ${ht}"; else echo "[FAIL] ${ht} rc=${rc}"; fi
done

echo "=== Phase 2 v9 training done $(date) ==="
