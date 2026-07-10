#!/usr/bin/env bash
###############################################################################
# pipeline2.sh - Full SpatialMind pipeline with Mistral-7B-Instruct-v0.3.
#
#   sbatch jobs/pipeline2.sh
#   RESUME_JOB_ID=<id> sbatch jobs/pipeline2.sh
###############################################################################

#SBATCH --job-name=mistral
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200gb
#SBATCH --time=6-23:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.log

SCRIPT_DIR="/home/dy23a.fsu/popllm/SpatialMind/jobs"
export MODEL_NAME="${MODEL_NAME:-Mistral-7B-Instruct-v0.3}"
source "${SCRIPT_DIR}/_pipeline_body.sh"
