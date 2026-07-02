#!/usr/bin/env bash
###############################################################################
# pipeline1.sh - Full SpatialMind pipeline with Llama-3.1-8B-Instruct (primary).
#
#   sbatch jobs/pipeline1.sh
#   RESUME_JOB_ID=<id> sbatch jobs/pipeline1.sh   # resume a prior run
###############################################################################

#SBATCH --job-name=llama
#SBATCH --account=fsu-compsci-dept
#SBATCH --qos=fsu-compsci-dept
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250gb
#SBATCH --time=6-23:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.log

SCRIPT_DIR="/home/dy23a.fsu/popllm/SpatialMind/jobs"
export MODEL_NAME="${MODEL_NAME:-Llama-3.1-8B-Instruct}"
source "${SCRIPT_DIR}/_pipeline_body.sh"
