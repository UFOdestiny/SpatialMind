#!/usr/bin/env bash
###############################################################################
# run_v10_all_backbones.sh - run the three extra backbones SEQUENTIALLY.
# Single GPU: one model fully finishes (gen->train->eval->fusion) before the
# next starts. Each writes to its own constraint_guided_v10_<tag> namespace,
# so the original Llama v10 run is never touched. Fully resumable: every stage
# is idempotent and skips completed work on re-run.
###############################################################################
set -uo pipefail
JOBS="/home/dy23a.fsu/popllm/SpatialMind/jobs"
for wrapper in run_v10_mistral7b.sh run_v10_gemma2.sh run_v10_phi4.sh; do
    echo "############################################################"
    echo "### SEQUENTIAL: ${wrapper} start $(date)"
    echo "############################################################"
    bash "${JOBS}/${wrapper}" || echo "[WARN] ${wrapper} exited non-zero; continuing to next backbone"
    echo "### SEQUENTIAL: ${wrapper} end $(date)"
done
echo "### ALL BACKBONES DONE $(date)"
