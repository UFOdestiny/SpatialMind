#!/usr/bin/env bash
###############################################################################
# run_v9_eval_id.sh - Phase 3 (ID test) evaluation for constraint_guided_v9.
# Evaluates every trained head + the deterministic/statistical baselines on the
# StepGame test split. OOD transfer is handled separately.
###############################################################################
set -uo pipefail
cd /home/dy23a.fsu/popllm/SpatialMind

CACHE="spatialmind/cache/cached_features/constraint_guided_v9/StepGame/Llama-3.1-8B-Instruct"
JOB="constraint_guided_v9_20260712"
RES="spatialmind/results/${JOB}"
LOG="spatialmind/logs/${JOB}"
BATCH="${TRAIN_BATCH_SIZE:-128}"
CALIBRATE="${CALIBRATE:-standard}"
STRUCT_C="${STRUCT_CALIB_C:-0.01}"
mkdir -p "${LOG}/eval"

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

echo "=== Phase 3 (ID test) start $(date) ==="

for ht in "${HEADS[@]}"; do
  if [[ ! -f "${RES}/train/${ht}/final_model/head_weights.pth" ]]; then
    echo "[SKIP] ${ht}: not trained"; continue
  fi
  out="${RES}/eval/${ht}"
  mkdir -p "${out}"
  echo "[EVAL] ${ht} $(date)"
  CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
    --head_path "${RES}/train/${ht}/final_model" \
    --cache_dir "${CACHE}" --output_dir "${out}" \
    --split test --batch_size "${BATCH}" \
    --calibrate "${CALIBRATE}" --struct_calib_C "${STRUCT_C}" \
    > "${LOG}/eval/eval_${ht}.log" 2>&1
  rc=$?
  [[ ${rc} -eq 0 ]] && echo "[OK] ${ht}" || echo "[FAIL] ${ht} rc=${rc}"
done

# Statistical + deterministic constraint baselines (single pass).
echo "[EVAL] baselines $(date)"
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
  --cache_dir "${CACHE}" --output_dir "${RES}/eval/baselines" \
  --split test --eval_baselines --batch_size "${BATCH}" \
  --calibrate "${CALIBRATE}" --struct_calib_C "${STRUCT_C}" \
  > "${LOG}/eval/eval_baselines.log" 2>&1
rc=$?
[[ ${rc} -eq 0 ]] && echo "[OK] baselines" || echo "[FAIL] baselines rc=${rc}"

echo "=== Phase 3 (ID test) done $(date) ==="
