#!/usr/bin/env bash
#SBATCH --job-name=memory-subsets
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --output=logs/%x-%j.log
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
"${ENVS_DIR:-envs}/warprec/bin/python" scripts/memory/measure_memory.py subsets "$@"
