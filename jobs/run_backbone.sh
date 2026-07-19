#!/usr/bin/env bash
###############################################################################
# run_backbone.sh - full submission pipeline for one backbone.
#
# Results are namespaced by RUN_TAG and every stage is resumable.
# The headline datasets are StepGame (ID), SpaRTQA, SpaRTUN, SpaceNLI, and SpaRP.
#
# Required env in:  MODEL_NAME, RUN_TAG
###############################################################################
set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

: "${MODEL_NAME:?set MODEL_NAME (a dir under spatialmind/models)}"
: "${RUN_TAG:?set RUN_TAG (short slug for the namespace, e.g. llama)}"

export DATASET_NAME="StepGame"
export JUDGE_MODEL_NAME="Mistral-Small-3.2-24B-Instruct-2506"
export CACHE_SUBDIR="constraint_guided_${RUN_TAG}"

# --- scaled sample sizes (5k / 2k / 3k) ---
export GEN_MAX_TRAIN="${GEN_MAX_TRAIN:-5000}"
export GEN_MAX_VAL="${GEN_MAX_VAL:-2000}"
export GEN_MAX_TEST="${GEN_MAX_TEST:-3000}"
export GEN_MAX_OOD="${GEN_MAX_OOD:-3000}"
export GEN_MAX_OOD_VAL="${GEN_MAX_OOD_VAL:-2000}"

export CALIBRATE="${CALIBRATE:-standard}"
export STRUCT_CALIB_C="${STRUCT_CALIB_C:-0.01}"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-30}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
export TRAIN_LEARNING_RATE="${TRAIN_LEARNING_RATE:-0.0002}"

# Head zoo trained on the ID cache.
HEAD_TYPES="spatialmind constraint_only spatialmind_neural constraint_no_conflict constraint_no_context constraint_no_entailment constraint_no_repair uhead factoscope mlp"
# Final headline lineup: StepGame(ID) + SpaRTQA, SpaRTUN, SpaceNLI, SpaRP (OOD).
# bAbI and SpartQA-YN dropped (single-pass information ceiling / near-chance).
OOD_LIST=(spartqa SpaRTUN SpaceNLI SpaRP_PS3)
FUSION_SUP_HEADS="constraint_no_conflict constraint_only spatialmind_neural spatialmind uhead factoscope mlp"
FUSION_BASELINES="constraint_rule,ccp,mcp,perplexity,token_entropy,random"
# Sampling-based SOTA baselines (K stochastic decodes each), run after the heads.
SAMPLING_K="${SAMPLING_K:-10}"

source "${SCRIPT_DIR}/common.sh"
export RESULTS_ROOT="${RESULTS_ROOT:-${BASE_RESULTS_ROOT}/${CACHE_SUBDIR}}"
export LOGS_ROOT="${LOGS_ROOT:-${BASE_LOGS_ROOT}/${CACHE_SUBDIR}}"
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
echo "### tag      : ${RUN_TAG}   namespace: ${CACHE_SUBDIR}"
echo "### cache    : ${ID_CACHE}"
echo "### sizes: train=${GEN_MAX_TRAIN} val=${GEN_MAX_VAL} test=${GEN_MAX_TEST} ood=${GEN_MAX_OOD}/${GEN_MAX_OOD_VAL}"
echo "########################################################"

echo "===== STAGE 1: data (ID) $(date) ====="
run_phase1 "train,validation,test" "${GEN_MAX_TRAIN},${GEN_MAX_VAL},${GEN_MAX_TEST}" \
    || { echo "[FATAL] data phase failed"; exit 1; }

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

echo "===== STAGE 3: ID eval $(date) ====="
for ht in "${ALL_HEAD_TYPES[@]}"; do
    out="${RESULTS_ROOT}/eval/${ht}"
    [[ -f "${out}/evaluation_report.json" ]] && { echo "[SKIP] eval ${ht}"; continue; }
    eval_head "${ht}" "${ID_CACHE}" "${out}" || echo "[FAIL] eval ${ht}"
done
[[ -f "${RESULTS_ROOT}/eval/baselines/combined_evaluation.json" ]] || \
    eval_baselines "${ID_CACHE}" "${RESULTS_ROOT}/eval/baselines" || echo "[FAIL] ID baselines"

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

echo "===== STAGE 5: val_scores for fusion $(date) ====="
declare -A CN=( [StepGame]=StepGame [spartqa]=spartqa [babi]=babi [SpaRTUN]=SpaRTUN [SpaceNLI]=SpaceNLI )
for ds in StepGame "${OOD_LIST[@]}"; do
    if [[ "${ds}" == "StepGame" ]]; then cache="${ID_CACHE}"; else cache="$(_ood_cache_dir "${ds}")"; fi
    [[ -d "${cache}/validation" ]] || { echo "[skip] ${ds}: no val cache"; continue; }
    dname="${CN[$ds]:-$ds}"
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

echo "===== STAGE 6: fusion $(date) ====="
python scripts/fusion.py \
    --results_root "${RESULTS_ROOT}" \
    --cache_root "${CACHE_ROOT}/cached_features" \
    --cache_subdir "${CACHE_SUBDIR}" \
    --model "${MODEL_NAME}" \
    --out_subdir fusion \
    --datasets "id:StepGame,spartqa:spartqa,SpaRTUN:SpaRTUN,SpaceNLI:SpaceNLI,SpaRP_PS3:SpaRP_PS3" \
    2>&1 | tee "${LOGS_ROOT}/eval/fusion.log"

###############################################################################
# Stage 7: sampling-based SOTA baselines (Semantic Entropy / SelfCheckGPT /
# P(True)). K stochastic decodes per sample; written to baselines_sampling/ so
# they never clobber the single-pass baselines. Scored against the same greedy
# trace label at benchmark time (scripts/benchmark_fair.py).
###############################################################################
echo "===== STAGE 7: sampling baselines $(date) ====="
export VLLM_WORKER_MULTIPROC_METHOD=spawn
declare -A SAMP_DIR=( [StepGame]="eval" [spartqa]="eval_ood/spartqa" \
                      [babi]="eval_ood/babi" \
                      [SpaRTUN]="eval_ood/SpaRTUN" [SpaceNLI]="eval_ood/SpaceNLI" \
                      [SpaRP_PS1]="eval_ood/SpaRP_PS1" [SpaRP_PS3]="eval_ood/SpaRP_PS3" )
declare -A SAMP_DS=( [StepGame]="StepGame" [spartqa]="spartqa" [babi]="babi" \
                     [SpaRTUN]="SpaRTUN" [SpaceNLI]="SpaceNLI" \
                     [SpaRP_PS1]="SpaRP_PS1" [SpaRP_PS3]="SpaRP_PS3" )
for key in StepGame spartqa babi SpaRTUN SpaceNLI SpaRP_PS1 SpaRP_PS3; do
    out="${RESULTS_ROOT}/${SAMP_DIR[$key]}/baselines_sampling"
    [[ -f "${out}/combined_evaluation.json" ]] && { echo "[SKIP] sampling ${key}"; continue; }
    python scripts/sampling_baselines.py \
        --model_path "${MODEL_PATH}" --dataset_name "${SAMP_DS[$key]}" \
        --dataset_path "${DATASETS_ROOT}/${SAMP_DS[$key]}/hf_dataset" \
        --out_dir "${out}" --max_val "${GEN_MAX_VAL}" --max_test "${GEN_MAX_TEST}" \
        --K "${SAMPLING_K}" --temp 0.7 --gpu_frac 0.85 --max_len "${VLLM_MAX_MODEL_LEN:-2048}" \
        > "${LOGS_ROOT}/eval/sampling_${key}.log" 2>&1 || echo "[FAIL] sampling ${key}"
done

echo "===== FINAL benchmark (fair, unified labels) $(date) ====="
python scripts/benchmark_fair.py --root "${RESULTS_ROOT}" 2>&1 | tee "${LOGS_ROOT}/eval/benchmark_fair.log"

echo "########################################################"
echo "### run_backbone (${RUN_TAG}) done $(date)"
echo "########################################################"
