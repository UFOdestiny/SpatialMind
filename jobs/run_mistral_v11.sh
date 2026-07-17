#!/usr/bin/env bash
# v11 scaled run, backbone = Mistral-7B-Instruct-v0.3.
set -uo pipefail
export MODEL_NAME="Mistral-7B-Instruct-v0.3"
export RUN_TAG="mistral"
bash /home/dy23a.fsu/popllm/SpatialMind/jobs/run_backbone_v11.sh
