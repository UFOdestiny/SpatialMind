#!/usr/bin/env bash
# Full pipeline, backbone = Mistral-7B-Instruct-v0.3. Isolated namespace.
set -uo pipefail
export MODEL_NAME="Mistral-7B-Instruct-v0.3"
export RUN_TAG="mistral7b"
bash /home/dy23a.fsu/popllm/SpatialMind/jobs/run_backbone.sh
