#!/usr/bin/env bash
#SBATCH --job-name=smind-phi
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
# End-to-end reproduction for backbone = gemma-2-9b-it.
# Full pipeline: data -> train head zoo -> ID/OOD eval -> val scores -> fusion
#   -> sampling SOTA baselines -> fair benchmark. Datasets: StepGame(ID) +
#   SpaRTQA, SpaRTUN, SpaceNLI, SpaRP (OOD). Namespace constraint_guided_v11_gemma.
# Idempotent/resumable. Run:  sbatch jobs/reproduce_phi.sh   (or bash ...)
###############################################################################
set -uo pipefail
export MODEL_NAME="Phi-4-reasoning"
export RUN_TAG="phi"
bash /home/dy23a.fsu/popllm/SpatialMind/jobs/run_backbone_v11.sh
