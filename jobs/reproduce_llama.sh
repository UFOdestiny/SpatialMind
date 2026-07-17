#!/usr/bin/env bash
#SBATCH --job-name=smind-llama
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
# End-to-end reproduction for backbone = Llama-3.1-8B-Instruct.
# Full pipeline: data -> train head zoo -> ID/OOD eval -> val scores -> fusion
#   -> sampling SOTA baselines -> fair benchmark. Datasets: StepGame(ID) +
#   SpaRTQA, SpaRTUN, SpaceNLI, SpaRP (OOD). Namespace constraint_guided_v11_llama.
# Idempotent/resumable. Run:  sbatch jobs/reproduce_llama.sh   (or bash ...)
###############################################################################
set -uo pipefail
export MODEL_NAME="Llama-3.1-8B-Instruct"
export RUN_TAG="llama"
bash /home/dy23a.fsu/popllm/SpatialMind/jobs/run_backbone_v11.sh
