#!/usr/bin/env bash
#SBATCH --job-name=benchmark
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --output=logs/%x-%j.log
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
JOB_TMPDIR=""
TMPDIR_LINK=""
if [ -n "${BENCH_SCRATCH_DIR:-}" ]; then
    export BENCH_RAY_SPILL_DIR="${BENCH_RAY_SPILL_DIR:-$BENCH_SCRATCH_DIR/ray-spill}"
    export BENCH_RAY_TMPDIR="${BENCH_RAY_TMPDIR:-/tmp}"
    JOB_TMPDIR="$BENCH_SCRATCH_DIR/tmp/${SLURM_JOB_ID:-$$}"
    TMPDIR_LINK="/tmp/bench-${SLURM_JOB_ID:-$$}"
    mkdir -p "$JOB_TMPDIR"
    ln -sfn "$JOB_TMPDIR" "$TMPDIR_LINK"
    export TMPDIR="$TMPDIR_LINK"
    export XDG_CACHE_HOME="$JOB_TMPDIR/cache"
    export CUDA_CACHE_PATH="$JOB_TMPDIR/cache/nv"
    export MPLCONFIGDIR="$JOB_TMPDIR/cache/matplotlib"
fi
scripts/benchmark/run.sh "$@" &
RUN_PID=$!
trap 'echo "run.sbatch: SIGTERM received, passed on to run.sh"; kill -TERM "$RUN_PID" 2>/dev/null' TERM
set +e
wait "$RUN_PID"
EXIT_CODE=$?
while kill -0 "$RUN_PID" 2>/dev/null; do
    wait "$RUN_PID"
    EXIT_CODE=$?
done
if [ -n "$JOB_TMPDIR" ]; then
    rm -f "$TMPDIR_LINK"
    rm -rf "$JOB_TMPDIR"
fi
exit "$EXIT_CODE"
