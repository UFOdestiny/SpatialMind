#!/usr/bin/env bash
# v11 scaled run, backbone = gemma-2-9b-it.
set -uo pipefail
export MODEL_NAME="gemma-2-9b-it"
export RUN_TAG="gemma"
bash /home/dy23a.fsu/popllm/SpatialMind/jobs/run_backbone_v11.sh
