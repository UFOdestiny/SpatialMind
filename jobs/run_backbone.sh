#!/usr/bin/env bash
###############################################################################
# run_backbone.sh - full SpatialMind pipeline for an ARBITRARY backbone.
#
# End-to-end, submission-standard run. Reuses the real phase functions
# (phase_data.sh generation/judge/rebuild/audit, phase_eval.sh eval) so behaviour
# matches the phase scripts exactly. The backbone, cache namespace, results root
# and logs root are all parameterized so multiple backbones can be run WITHOUT
# overwriting each other's artifacts.
#
# Usage (do not run directly; use the per-model wrappers run_<model>.sh):
#   MODEL_NAME=Mistral-7B-Instruct-v0.3 RUN_TAG=mistral bash jobs/run_backbone.sh
#
# Required env in:  MODEL_NAME, RUN_TAG
#
# Scale (same for every backbone):
#   ID  StepGame : train 5000 / val 1000 / test 2000
#   OOD (5 sets) : val 1000 / test 2000  (auto-capped to availability, e.g. babi)
#
# Stages (each idempotent / skippable):
#   1  data   : generate -> judge -> rebuild constraints -> leakage audit (ID)
#   2  train  : train the head zoo
#   3  eval   : ID test eval (heads + baselines)
#   4  ood    : OOD gen+judge+rebuild+eval (heads + baselines) x5
#   5  val    : validation-split predictions for ALL fusion signals
#   6  fusion : multi-signal applicability-aware fusion (validation-selected)
###############################################################################
set -uo pipefail
SCRIPT_DIR="/home/dy23a.fsu/popllm/SpatialMind/jobs"
cd /home/dy23a.fsu/popllm/SpatialMind

: "${MODEL_NAME:?set MODEL_NAME (a dir under spatialmind/models)}"
: "${RUN_TAG:?set RUN_TAG (short slug for the namespace, e.g. mistral)}"

# --- namespace: keyed by RUN_TAG so backbones never collide ---
export DATASET_NAME="StepGame"
export JUDGE_MODEL_NAME="Mistral-Small-3.2-24B-Instruct-2506"
export CACHE_SUBDIR="constraint_guided_v10_${RUN_TAG}"
export RESULTS_ROOT="/home/dy23a.fsu/popllm/SpatialMind/spatialmind/results/constraint_guided_v10_${RUN_TAG}"
export LOGS_ROOT="/home/dy23a.fsu/popllm/SpatialMind/spatialmind/logs/constraint_guided_v10_${RUN_TAG}"

# --- scaled sample sizes ---
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

# Head zoo trained on the ID cache. The fusion combiner draws on the supervised
# heads plus the unsupervised baselines below.
HEAD_TYPES="spatialmind constraint_only spatialmind_neural constraint_no_conflict constraint_no_context constraint_no_entailment constraint_no_repair uhead factoscope mlp"
OOD_LIST=(spartqa babi SpaRTUN SpaceNLI SpartQA_YN)
# Supervised heads scored on validation (must be a subset of HEAD_TYPES).
FUSION_SUP_HEADS="constraint_no_conflict constraint_only spatialmind_neural spatialmind uhead factoscope mlp"
# Unsupervised baselines scored on validation for fusion.
FUSION_BASELINES="constraint_rule,ccp,mcp,perplexity,token_entropy,random"

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

source "${SCRIPT_DIR}/phase_data.sh"
source "${SCRIPT_DIR}/phase_eval.sh"

ID_CACHE="${CACHE_DIR}"
mkdir -p "${LOGS_ROOT}/data" "${LOGS_ROOT}/train" "${LOGS_ROOT}/eval" "${RESULTS_ROOT}/train"

echo "########################################################"
echo "### run_backbone start $(date)"
echo "### backbone : ${MODEL_NAME}"
echo "### tag      : ${RUN_TAG}"
echo "### cache    : ${ID_CACHE}"
echo "### results  : ${RESULTS_ROOT}"
echo "### sizes: train=${GEN_MAX_TRAIN} val=${GEN_MAX_VAL} test=${GEN_MAX_TEST} ood=${GEN_MAX_OOD}/${GEN_MAX_OOD_VAL}"
echo "########################################################"

###############################################################################
# Stage 1: data phase (ID)
###############################################################################
echo "===== STAGE 1: data (ID) $(date) ====="
run_phase1 "train,validation,test" "${GEN_MAX_TRAIN},${GEN_MAX_VAL},${GEN_MAX_TEST}" \
    || { echo "[FATAL] data phase failed"; exit 1; }

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
# Stage 5: validation-split predictions for ALL fusion signals
#
# The multi-signal fusion combiner is fit on validation only, so every signal it
# may route to needs a validation-split prediction. Supervised heads -> per-head
# reports; unsupervised baselines -> a combined report. Test is never touched.
###############################################################################
echo "===== STAGE 5: val_scores for fusion $(date) ====="
declare -A CN=( [StepGame]=StepGame [spartqa]=spartqa [babi]=babi [SpaRTUN]=SpaRTUN [SpaceNLI]=SpaceNLI [SpartQA_YN]=SpartQA_YN )
for ds in StepGame "${OOD_LIST[@]}"; do
    if [[ "${ds}" == "StepGame" ]]; then cache="${ID_CACHE}"; else cache="$(_ood_cache_dir "${ds}")"; fi
    [[ -d "${cache}/validation" ]] || { echo "[skip] ${ds}: no val cache"; continue; }
    dname="${CN[$ds]}"
    # supervised heads
    for ht in ${FUSION_SUP_HEADS}; do
        hp="${RESULTS_ROOT}/train/${ht}/final_model"
        [[ -f "${hp}/head_weights.pth" ]] || { echo "[skip] ${ht} not trained"; continue; }
        out="${RESULTS_ROOT}/val_scores/${dname}/${ht}"
        [[ -f "${out}/evaluation_report.json" ]] && { echo "[SKIP] val ${ds}/${ht}"; continue; }
        mkdir -p "${out}"
        CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
            --head_path "${hp}" --cache_dir "${cache}" --output_dir "${out}" \
            --split validation --batch_size "${TRAIN_BATCH_SIZE}" \
            --calibrate "${CALIBRATE}" --struct_calib_C "${STRUCT_CALIB_C}" \
            > "${LOGS_ROOT}/eval/val_${ds}_${ht}.log" 2>&1
        echo "[$( [[ $? -eq 0 ]] && echo OK || echo FAIL )] val ${ds}/${ht}"
    done
    # unsupervised baselines
    bout="${RESULTS_ROOT}/val_scores_baselines/${dname}"
    if [[ ! -f "${bout}/combined_evaluation.json" ]]; then
        mkdir -p "${bout}"
        CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
            --cache_dir "${cache}" --output_dir "${bout}" --split validation \
            --eval_baselines --baselines "${FUSION_BASELINES}" --batch_size "${TRAIN_BATCH_SIZE}" \
            --calibrate "${CALIBRATE}" --struct_calib_C "${STRUCT_CALIB_C}" \
            > "${LOGS_ROOT}/eval/val_${ds}_baselines.log" 2>&1
        echo "[$( [[ $? -eq 0 ]] && echo OK || echo FAIL )] val ${ds}/baselines"
    fi
done

###############################################################################
# Stage 6: multi-signal fusion (validation-selected, no test peeking)
###############################################################################
echo "===== STAGE 6: fusion $(date) ====="
python scripts/fusion.py \
    --results_root "${RESULTS_ROOT}" \
    --cache_root "${CACHE_ROOT}/cached_features/${CACHE_SUBDIR}" \
    --model "${MODEL_NAME}" \
    --out_subdir fusion \
    --datasets "id:StepGame,spartqa:spartqa,babi:babi,SpaRTUN:SpaRTUN,SpaceNLI:SpaceNLI,SpartQA_YN:SpartQA_YN" \
    2>&1 | tee "${LOGS_ROOT}/eval/fusion.log"

echo "########################################################"
echo "### run_backbone (${RUN_TAG}) done $(date)"
echo "########################################################"
