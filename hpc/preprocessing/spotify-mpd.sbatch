#!/usr/bin/env bash
#SBATCH --job-name=preprocess-spotify-mpd
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%j.log
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
scripts/preprocessing/spotify-mpd.sh
