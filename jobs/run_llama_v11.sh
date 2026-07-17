#!/usr/bin/env bash
# v11 scaled run, backbone = Llama-3.1-8B-Instruct (primary).
set -uo pipefail
export MODEL_NAME="Llama-3.1-8B-Instruct"
export RUN_TAG="llama"
bash /home/dy23a.fsu/popllm/SpatialMind/jobs/run_backbone_v11.sh
