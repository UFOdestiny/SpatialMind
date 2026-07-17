#!/usr/bin/env bash
###############################################################################
# run_sparp_probe.sh - SMALL-SAMPLE probe on the two new SpaRP spatial datasets.
#
# Tests whether SpatialMind (WITHOUT sampling signals) beats the sampling SOTA
# (Semantic Entropy / P(True) / SelfCheckGPT) on fresh spatial benchmarks.
#
# Treated as OOD transfer from the already-trained Llama v11 head zoo:
#   generate -> judge -> rebuild -> eval(heads+baselines) -> val_scores -> fusion.
# Sampling baselines run separately via scripts/sampling_baselines.py.
#
# Small scale for a fast verdict: val 1000 / test 1500.
#   bash jobs/run_sparp_probe.sh
###############################################################################
set -uo pipefail
SCRIPT_DIR="/home/dy23a.fsu/popllm/SpatialMind/jobs"
cd /home/dy23a.fsu/popllm/SpatialMind

export MODEL_NAME="Llama-3.1-8B-Instruct"
export RUN_TAG="llama"
export DATASET_NAME="StepGame"
export JUDGE_MODEL_NAME="Mistral-Small-3.2-24B-Instruct-2506"
export CACHE_SUBDIR="constraint_guided_v11_${RUN_TAG}"
export RESULTS_ROOT="/home/dy23a.fsu/popllm/SpatialMind/spatialmind/results/constraint_guided_v11_${RUN_TAG}"
export LOGS_ROOT="/home/dy23a.fsu/popllm/SpatialMind/spatialmind/logs/constraint_guided_v11_${RUN_TAG}"

# small-sample OOD sizes
export GEN_MAX_OOD="${GEN_MAX_OOD:-1500}"
export GEN_MAX_OOD_VAL="${GEN_MAX_OOD_VAL:-1000}"
export CALIBRATE="${CALIBRATE:-standard}"; export STRUCT_CALIB_C="${STRUCT_CALIB_C:-0.01}"

HEAD_TYPES="spatialmind constraint_only spatialmind_neural constraint_no_conflict constraint_no_context constraint_no_entailment constraint_no_repair uhead factoscope mlp"
SPARP_LIST=(SpaRP_PS1 SpaRP_PS3)
FUSION_SUP_HEADS="constraint_no_conflict constraint_only spatialmind_neural spatialmind uhead factoscope mlp"
FUSION_BASELINES="constraint_rule,ccp,mcp,perplexity,token_entropy,random"

source "${SCRIPT_DIR}/common.sh"
read -ra ALL_HEAD_TYPES <<< "${HEAD_TYPES}"
setup_environment
detect_gpus
export MODEL_PATH="${MODELS_ROOT}/${MODEL_NAME}"
export JUDGE_MODEL_PATH="${MODELS_ROOT}/${JUDGE_MODEL_NAME}"
export CLAIM_EXTRACTOR_MODEL_PATH=""
source "${SCRIPT_DIR}/phase_data.sh"
source "${SCRIPT_DIR}/phase_eval.sh"

echo "########## SpaRP probe start $(date) ##########"

for ood in "${SPARP_LIST[@]}"; do
    echo "===== OOD ${ood} $(date) ====="
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

    # validation-split scores for fusion signals
    for ht in ${FUSION_SUP_HEADS}; do
        hp="${RESULTS_ROOT}/train/${ht}/final_model"
        [[ -f "${hp}/head_weights.pth" ]] || continue
        out="${RESULTS_ROOT}/val_scores/${ood}/${ht}"
        [[ -f "${out}/evaluation_report.json" ]] && continue
        mkdir -p "${out}"
        CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
            --head_path "${hp}" --cache_dir "${ood_cache}" --output_dir "${out}" \
            --split validation --batch_size 256 --calibrate "${CALIBRATE}" --struct_calib_C "${STRUCT_CALIB_C}" \
            > "${LOGS_ROOT}/eval/val_${ood}_${ht}.log" 2>&1 || echo "[FAIL] val ${ood}/${ht}"
    done
    bout="${RESULTS_ROOT}/val_scores_baselines/${ood}"
    if [[ ! -f "${bout}/combined_evaluation.json" ]]; then
        mkdir -p "${bout}"
        CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
            --cache_dir "${ood_cache}" --output_dir "${bout}" --split validation \
            --eval_baselines --baselines "${FUSION_BASELINES}" --batch_size 256 \
            --calibrate "${CALIBRATE}" --struct_calib_C "${STRUCT_CALIB_C}" \
            > "${LOGS_ROOT}/eval/val_${ood}_baselines.log" 2>&1 || echo "[FAIL] val ${ood}/baselines"
    fi
done

# Fusion for the two SpaRP targets (reads trained heads + baselines + val scores)
echo "===== fusion $(date) ====="
python scripts/fusion.py \
    --results_root "${RESULTS_ROOT}" \
    --cache_root "${CACHE_ROOT}/cached_features" --cache_subdir "${CACHE_SUBDIR}" \
    --model "${MODEL_NAME}" --out_subdir fusion \
    --datasets "SpaRP_PS1:SpaRP_PS1,SpaRP_PS3:SpaRP_PS3" \
    2>&1 | tee "${LOGS_ROOT}/eval/fusion_sparp.log"

echo "########## SpaRP probe done $(date) ##########"
