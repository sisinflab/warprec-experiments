#!/usr/bin/env bash
#SBATCH --job-name=memory-measure
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=300G
#SBATCH --output=logs/%x-%j.log
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
PYTHON="${ENVS_DIR:-envs}/warprec/bin/python"
CONFIG="${1:?usage: hpc/memory/measure.sbatch <config> <dataset> <fractions>}"
DATASET="${2:?usage: hpc/memory/measure.sbatch <config> <dataset> <fractions>}"
shift 2
"$PYTHON" scripts/memory/measure_memory.py measure "$CONFIG" "$DATASET" "$@"
"$PYTHON" scripts/memory/measure_memory.py fit "$CONFIG" "$DATASET"
