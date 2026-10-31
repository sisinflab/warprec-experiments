#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
USAGE="usage: scripts/benchmark/run.sh <config> <dataset> [runner options]"
CONFIG="${1:?$USAGE}"
DATASET="${2:?$USAGE}"
shift 2

ENV_PREFIX="${ENVS_DIR:-$REPO_ROOT/envs}/warprec"
RUNNER="$REPO_ROOT/scripts/benchmark/runners/warprec/run.py"
PYTHON="$ENV_PREFIX/bin/python"
if [ ! -f "$CONFIG" ]; then
    echo "Missing configuration $CONFIG." >&2
    exit 1
fi
CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
DATASET_CONFIG="${CONFIG%.yml}-$DATASET.yml"
if [ -f "$DATASET_CONFIG" ]; then
    CONFIG="$DATASET_CONFIG"
fi
EXPERIMENT="$(basename "$(dirname "$CONFIG")")"
NAME="$(basename "$CONFIG" .yml)"
if [ ! -x "$PYTHON" ]; then
    echo "Missing environment $ENV_PREFIX: run scripts/setup/install_env.sh warprec first." >&2
    exit 1
fi

RUN_DIR="${BENCH_RESULTS_DIR:-$REPO_ROOT/results}/$EXPERIMENT/$DATASET/$NAME/$(date +%Y%m%d-%H%M%S)-${SLURM_JOB_ID:-$$}"
mkdir -p "$RUN_DIR"

GPUS="$("$PYTHON" "$REPO_ROOT/scripts/benchmark/common/gpus.py" "$CONFIG")"
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((GPUS - 1)))"
fi
VISIBLE="$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | head -n "$GPUS" | paste -sd, -)"
if [ "$(echo "$VISIBLE" | tr ',' '\n' | wc -l)" -lt "$GPUS" ]; then
    echo "The configuration needs $GPUS GPUs, the run sees $CUDA_VISIBLE_DEVICES." >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES="$VISIBLE"
export PATH="$ENV_PREFIX/bin:$PATH"
export PYTHONNOUSERSITE=1

echo "run: $EXPERIMENT $NAME on $DATASET, on $(hostname), $(nproc) CPUs, GPUs $CUDA_VISIBLE_DEVICES"
echo "configuration: $CONFIG"
echo "results: $RUN_DIR"

"$PYTHON" "$REPO_ROOT/scripts/benchmark/common/preload.py" 2>&1 | tee "$RUN_DIR/run.log" \
    || echo "Preloading the data libraries failed: the run goes on without it." | tee -a "$RUN_DIR/run.log"

MONITOR_PIDS=()
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=index,memory.used \
        --format=csv,noheader,nounits -lms 1000 > "$RUN_DIR/gpu-memory.log" 2>/dev/null &
    MONITOR_PIDS+=($!)
fi
if [ -r /proc/meminfo ]; then
    while true; do
        awk '/^MemTotal:/ {total = $2} /^MemAvailable:/ {available = $2} END {print total - available}' /proc/meminfo
        sleep 1
    done > "$RUN_DIR/ram.log" 2>/dev/null &
    MONITOR_PIDS+=($!)
fi

TIME_LIMIT="${BENCH_TIME_LIMIT:-86400}"
STARTED=$(date +%s)
set +e
timeout --signal=TERM --kill-after=120 "$TIME_LIMIT" \
    "$PYTHON" -u "$RUNNER" --config "$CONFIG" --dataset "$DATASET" \
    --output-dir "$RUN_DIR" "$@" > >(tee -a "$RUN_DIR/run.log") 2>&1 &
RUNNER_PID=$!

trap 'echo "run.sh: SIGTERM received, passed on to the runner"; kill -TERM "$RUNNER_PID" 2>/dev/null' TERM
wait "$RUNNER_PID"
EXIT_CODE=$?
while kill -0 "$RUNNER_PID" 2>/dev/null; do
    wait "$RUNNER_PID"
    EXIT_CODE=$?
done
set -e

for pid in ${MONITOR_PIDS[@]+"${MONITOR_PIDS[@]}"}; do
    kill "$pid" 2>/dev/null || true
done
"$PYTHON" "$REPO_ROOT/scripts/benchmark/common/finalize.py" \
    --output-dir "$RUN_DIR" --exit-code "$EXIT_CODE" \
    --elapsed "$(( $(date +%s) - STARTED ))" --time-limit "$TIME_LIMIT"
exit "$EXIT_CODE"
