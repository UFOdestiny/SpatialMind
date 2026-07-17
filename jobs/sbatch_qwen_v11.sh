#!/usr/bin/env bash
#SBATCH --job-name=smind-qwen-v11
#SBATCH --account=fsu-compsci-dept
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200gb
#SBATCH --time=6-23:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.log
###############################################################################
# sbatch jobs/sbatch_qwen_v11.sh
# v11 scaled run, backbone = Qwen3-8B. Namespace constraint_guided_v11_qwen.
###############################################################################
set -uo pipefail
export MODEL_NAME="Qwen3-8B"
export RUN_TAG="qwen"
bash /home/dy23a.fsu/popllm/SpatialMind/jobs/run_backbone_v11.sh
