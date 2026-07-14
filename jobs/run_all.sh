#!/usr/bin/env bash
###############################################################################
# run_all.sh - run every backbone SEQUENTIALLY on a single GPU.
# One model fully finishes (data->train->eval->ood->val->fusion) before the next
# starts. Each writes to its own constraint_guided_v10_<tag> namespace so runs
# never touch each other. Fully resumable: every stage is idempotent and skips
# completed work on re-run. Edit the wrapper list to run a subset.
###############################################################################
set -uo pipefail
JOBS="/home/dy23a.fsu/popllm/SpatialMind/jobs"
for wrapper in run_llama.sh run_mistral.sh run_gemma.sh run_phi4.sh; do
    echo "############################################################"
    echo "### SEQUENTIAL: ${wrapper} start $(date)"
    echo "############################################################"
    bash "${JOBS}/${wrapper}" || echo "[WARN] ${wrapper} exited non-zero; continuing to next backbone"
    echo "### SEQUENTIAL: ${wrapper} end $(date)"
done
echo "### ALL BACKBONES DONE $(date)"
