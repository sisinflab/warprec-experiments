#!/usr/bin/env bash
#SBATCH --job-name=preprocess-lfm2b
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/%x-%j.log
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
scripts/preprocessing/lfm2b.sh
