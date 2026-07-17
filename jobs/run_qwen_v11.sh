#!/usr/bin/env bash
# v11 scaled run, backbone = Qwen3-8B. Fourth backbone (replaces phi-4).
set -uo pipefail
export MODEL_NAME="Qwen3-8B"
export RUN_TAG="qwen"
bash /home/dy23a.fsu/popllm/SpatialMind/jobs/run_backbone_v11.sh
