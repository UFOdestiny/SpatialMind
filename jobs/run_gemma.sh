#!/usr/bin/env bash
# Full pipeline, backbone = gemma-2-9b-it. Isolated namespace.
set -uo pipefail
export MODEL_NAME="gemma-2-9b-it"
export RUN_TAG="gemma2"
bash /home/dy23a.fsu/popllm/SpatialMind/jobs/run_backbone.sh
