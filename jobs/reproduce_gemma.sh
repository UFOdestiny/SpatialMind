#!/usr/bin/env bash
#SBATCH --job-name=smind-gemma
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
#   SpaRTQA, SpaRTUN, SpaceNLI, SpaRP (OOD).
# Idempotent/resumable. Run:  sbatch jobs/reproduce_gemma.sh   (or bash ...)
###############################################################################
set -uo pipefail
export MODEL_NAME="gemma-2-9b-it"
export RUN_TAG="gemma"
bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/run_backbone.sh"
