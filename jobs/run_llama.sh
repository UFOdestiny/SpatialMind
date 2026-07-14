#!/usr/bin/env bash
# Full pipeline, backbone = Llama-3.1-8B-Instruct (primary). Isolated namespace.
set -uo pipefail
export MODEL_NAME="Llama-3.1-8B-Instruct"
export RUN_TAG="20260712"
bash /home/dy23a.fsu/popllm/SpatialMind/jobs/run_backbone.sh
