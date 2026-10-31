#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
USAGE="usage: hpc/benchmark/submit.sh <account> <config> <dataset> [sbatch options]"
ACCOUNT="${1:?$USAGE}"
CONFIG="${2:?$USAGE}"
DATASET="${3:?$USAGE}"
shift 3

if [ ! -f "$CONFIG" ]; then
    echo "Missing configuration $CONFIG." >&2
    exit 1
fi
CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
RUN_CONFIG="$CONFIG"
if [ -f "${CONFIG%.yml}-$DATASET.yml" ]; then
    RUN_CONFIG="${CONFIG%.yml}-$DATASET.yml"
fi
PYTHON="${ENVS_DIR:-$REPO_ROOT/envs}/warprec/bin/python"
GPUS="$("$PYTHON" "$REPO_ROOT/scripts/benchmark/common/gpus.py" "$RUN_CONFIG")"
EXPERIMENT="$(basename "$(dirname "$RUN_CONFIG")")"
JOB_NAME="${EXPERIMENT%%-*}-$DATASET-$(basename "$RUN_CONFIG" .yml)"

cd "$REPO_ROOT"
mkdir -p logs
set -x
sbatch -A "$ACCOUNT" ${BENCH_PARTITION:+"--partition=$BENCH_PARTITION"} --gres="gpu:$GPUS" \
    --job-name="$JOB_NAME" "$@" hpc/benchmark/run.sbatch "$CONFIG" "$DATASET"
