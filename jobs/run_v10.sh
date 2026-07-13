#!/usr/bin/env bash
###############################################################################
# run_v10.sh - Scaled, submission-standard end-to-end run.
#
# Reuses the real pipeline functions (p1.sh generation/judge/rebuild/audit,
# p3.sh eval) so behaviour is identical to a full pipeline, only larger:
#   ID  StepGame : train 5000 / val 1000 / test 2000
#   OOD (4 sets) : val 1000 / test 2000  (auto-capped to availability, e.g. babi)
#
# Stages (each idempotent / skippable):
#   1  Phase 1  : generate -> judge -> rebuild constraints -> leakage audit (ID)
#   2  Phase 2  : train the head zoo
#   3  Phase 3  : ID test eval (heads + baselines)
#   4  Phase 3b : OOD gen+judge+rebuild+eval (heads + baselines) x4
#   5  val_scores: validation-split predictions for the fusion base heads
#   6  fusion   : symbolizability-gated stacked fusion (validation-selected)
###############################################################################
set -uo pipefail
SCRIPT_DIR="/home/dy23a.fsu/popllm/SpatialMind/jobs"
cd /home/dy23a.fsu/popllm/SpatialMind

# --- v10 namespace / scaled sizes ---
export DATASET_NAME="StepGame"
export MODEL_NAME="Llama-3.1-8B-Instruct"
export JUDGE_MODEL_NAME="Mistral-Small-3.2-24B-Instruct-2506"
export CACHE_SUBDIR="constraint_guided_v10"
export RESULTS_ROOT="/home/dy23a.fsu/popllm/SpatialMind/spatialmind/results/constraint_guided_v10_20260712"
export LOGS_ROOT="/home/dy23a.fsu/popllm/SpatialMind/spatialmind/logs/constraint_guided_v10_20260712"

export GEN_MAX_TRAIN="${GEN_MAX_TRAIN:-5000}"
export GEN_MAX_VAL="${GEN_MAX_VAL:-1000}"
export GEN_MAX_TEST="${GEN_MAX_TEST:-2000}"
export GEN_MAX_OOD="${GEN_MAX_OOD:-2000}"
export GEN_MAX_OOD_VAL="${GEN_MAX_OOD_VAL:-1000}"

export CALIBRATE="${CALIBRATE:-standard}"
export STRUCT_CALIB_C="${STRUCT_CALIB_C:-0.01}"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-30}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
export TRAIN_LEARNING_RATE="${TRAIN_LEARNING_RATE:-0.0002}"

HEAD_TYPES="spatialmind constraint_only spatialmind_neural constraint_no_conflict constraint_no_context constraint_no_entailment constraint_no_repair uhead factoscope mlp"
# SpartQA_YN = machine-generated block-world yes/no (high symbolizability, ~8 rel/story);
# added to broaden the symbolizability spectrum beyond the original 4 OOD sets.
OOD_LIST=(spartqa babi SpaRTUN SpaceNLI SpartQA_YN)
FUSION_CON="constraint_no_conflict"
FUSION_NEU="mlp"

source "${SCRIPT_DIR}/common.sh"
read -ra ALL_HEAD_TYPES <<< "${HEAD_TYPES}"
setup_environment
detect_gpus
export MODEL_PATH="${MODELS_ROOT}/${MODEL_NAME}"
export JUDGE_MODEL_PATH="${MODELS_ROOT}/${JUDGE_MODEL_NAME}"
export DATASET_PATH
if [[ -d "${DATASETS_ROOT}/${DATASET_NAME}/hf_dataset" ]]; then
    DATASET_PATH="${DATASETS_ROOT}/${DATASET_NAME}/hf_dataset"
else
    DATASET_PATH="${DATASETS_ROOT}/${DATASET_NAME}"
fi
export CLAIM_EXTRACTOR_MODEL_PATH=""

source "${SCRIPT_DIR}/p1.sh"
source "${SCRIPT_DIR}/p3.sh"

ID_CACHE="${CACHE_DIR}"
mkdir -p "${LOGS_ROOT}/data" "${LOGS_ROOT}/train" "${LOGS_ROOT}/eval" "${RESULTS_ROOT}/train"

echo "########################################################"
echo "### run_v10 start $(date)"
echo "### ID cache: ${ID_CACHE}"
echo "### sizes: train=${GEN_MAX_TRAIN} val=${GEN_MAX_VAL} test=${GEN_MAX_TEST} ood=${GEN_MAX_OOD}/${GEN_MAX_OOD_VAL}"
echo "########################################################"

###############################################################################
# Stage 1: Phase 1 (ID data)
###############################################################################
echo "===== STAGE 1: Phase 1 (ID) $(date) ====="
run_phase1 "train,validation,test" "${GEN_MAX_TRAIN},${GEN_MAX_VAL},${GEN_MAX_TEST}" \
    || { echo "[FATAL] Phase 1 failed"; exit 1; }

###############################################################################
# Stage 2: train head zoo
###############################################################################
echo "===== STAGE 2: train heads $(date) ====="
for ht in "${ALL_HEAD_TYPES[@]}"; do
    out="${RESULTS_ROOT}/train/${ht}"
    if [[ -f "${out}/final_model/head_weights.pth" ]]; then echo "[SKIP] train ${ht}"; continue; fi
    mkdir -p "${out}"
    echo "[TRAIN] ${ht} $(date)"
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
        --head_type "${ht}" --cache_dir "${ID_CACHE}" --output_dir "${out}" \
        --num_epochs "${TRAIN_EPOCHS}" --batch_size "${TRAIN_BATCH_SIZE}" \
        --learning_rate "${TRAIN_LEARNING_RATE}" --loss_type bce --trace_loss_weight 0.5 \
        > "${LOGS_ROOT}/train/train_${ht}.log" 2>&1
    echo "[$( [[ $? -eq 0 ]] && echo OK || echo FAIL )] train ${ht}"
done

###############################################################################
# Stage 3: ID test eval + baselines
###############################################################################
echo "===== STAGE 3: ID eval $(date) ====="
for ht in "${ALL_HEAD_TYPES[@]}"; do
    out="${RESULTS_ROOT}/eval/${ht}"
    [[ -f "${out}/evaluation_report.json" ]] && { echo "[SKIP] eval ${ht}"; continue; }
    eval_head "${ht}" "${ID_CACHE}" "${out}" || echo "[FAIL] eval ${ht}"
done
[[ -f "${RESULTS_ROOT}/eval/baselines/combined_evaluation.json" ]] || \
    eval_baselines "${ID_CACHE}" "${RESULTS_ROOT}/eval/baselines" || echo "[FAIL] ID baselines"

###############################################################################
# Stage 4: OOD gen+judge+rebuild+eval
###############################################################################
echo "===== STAGE 4: OOD $(date) ====="
for ood in "${OOD_LIST[@]}"; do
    echo "---------- OOD ${ood} $(date) ----------"
    prepare_ood_cache "${ood}" || { echo "[FAIL] prepare ${ood}"; continue; }
    ood_cache="$(_ood_cache_dir "${ood}")"
    for sp in validation test; do
        python scripts/rebuild_constraint_cache.py --cache_dir "${ood_cache}" --split "${sp}" \
            >> "${LOGS_ROOT}/data/rebuild_ood_${ood}.log" 2>&1 || echo "[WARN] rebuild ${ood}/${sp}"
    done
    if ! cache_split_ready "${ood_cache}" "test" || ! cache_split_ready "${ood_cache}" "validation"; then
        echo "[FAIL] ${ood}: cache not ready"; continue
    fi
    for ht in "${ALL_HEAD_TYPES[@]}"; do
        out="${RESULTS_ROOT}/eval_ood/${ood}/${ht}"
        [[ -f "${out}/evaluation_report.json" ]] && { echo "[SKIP] ${ood}/${ht}"; continue; }
        eval_head "${ht}" "${ood_cache}" "${out}" || echo "[FAIL] eval ${ood}/${ht}"
    done
    [[ -f "${RESULTS_ROOT}/eval_ood/${ood}/baselines/combined_evaluation.json" ]] || \
        eval_baselines "${ood_cache}" "${RESULTS_ROOT}/eval_ood/${ood}/baselines" || echo "[FAIL] baselines ${ood}"
done

###############################################################################
# Stage 5: validation-split predictions for fusion base heads
###############################################################################
echo "===== STAGE 5: val_scores for fusion $(date) ====="
declare -A CN=( [StepGame]=StepGame [spartqa]=spartqa [babi]=babi [SpaRTUN]=SpaRTUN [SpaceNLI]=SpaceNLI [SpartQA_YN]=SpartQA_YN )
for ds in StepGame "${OOD_LIST[@]}"; do
    if [[ "${ds}" == "StepGame" ]]; then cache="${ID_CACHE}"; else cache="$(_ood_cache_dir "${ds}")"; fi
    for ht in "${FUSION_CON}" "${FUSION_NEU}"; do
        hp="${RESULTS_ROOT}/train/${ht}/final_model"
        [[ -f "${hp}/head_weights.pth" ]] || { echo "[skip] ${ht} not trained"; continue; }
        out="${RESULTS_ROOT}/val_scores/${CN[$ds]}/${ht}"
        [[ -f "${out}/evaluation_report.json" ]] && { echo "[SKIP] val ${ds}/${ht}"; continue; }
        mkdir -p "${out}"
        CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
            --head_path "${hp}" --cache_dir "${cache}" --output_dir "${out}" \
            --split validation --batch_size "${TRAIN_BATCH_SIZE}" \
            --calibrate "${CALIBRATE}" --struct_calib_C "${STRUCT_CALIB_C}" \
            > "${LOGS_ROOT}/eval/val_${ds}_${ht}.log" 2>&1
        echo "[$( [[ $? -eq 0 ]] && echo OK || echo FAIL )] val ${ds}/${ht}"
    done
done

###############################################################################
# Stage 6: gated fusion (validation-selected, no test peeking)
###############################################################################
echo "===== STAGE 6: gated fusion $(date) ====="
python scripts/gated_fusion.py \
    --results_root "${RESULTS_ROOT}" \
    --cache_root "${CACHE_ROOT}/cached_features/${CACHE_SUBDIR}" \
    --model "${MODEL_NAME}" \
    --con_head "${FUSION_CON}" --neu_head "${FUSION_NEU}" \
    --mode auto --out_subdir fusion \
    --datasets "id:StepGame,spartqa:spartqa,babi:babi,SpaRTUN:SpaRTUN,SpaceNLI:SpaceNLI,SpartQA_YN:SpartQA_YN" \
    2>&1 | tee "${LOGS_ROOT}/eval/fusion.log"

echo "########################################################"
echo "### run_v10 done $(date)"
echo "########################################################"
