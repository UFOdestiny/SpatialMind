#!/usr/bin/env bash
#SBATCH --job-name=smind-mistral
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200gb
#SBATCH --time=6-23:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.log
###############################################################################
# End-to-end reproduction for backbone = Mistral-7B-Instruct-v0.3.
# Full pipeline: data -> train head zoo -> ID/OOD eval -> val scores -> fusion
#   -> sampling SOTA baselines -> fair benchmark. Datasets: StepGame(ID) +
#   SpaRTQA, SpaRTUN, SpaceNLI, SpaRP (OOD).
# Idempotent/resumable. Run:  sbatch jobs/reproduce_mistral.sh   (or bash ...)
###############################################################################
set -uo pipefail
export MODEL_NAME="Mistral-7B-Instruct-v0.3"
export RUN_TAG="mistral"
bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/run_backbone.sh"
