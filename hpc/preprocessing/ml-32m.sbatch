#!/usr/bin/env bash
#SBATCH --job-name=preprocess-ml-32m
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/%x-%j.log
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
scripts/preprocessing/ml-32m.sh
