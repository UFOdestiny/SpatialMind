#!/usr/bin/env bash
# Full pipeline, backbone = Phi-4-reasoning. Isolated namespace.
set -uo pipefail
export MODEL_NAME="Phi-4-reasoning"
export RUN_TAG="phi4reason"
bash /home/dy23a.fsu/popllm/SpatialMind/jobs/run_backbone.sh
