#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW_DIR="$REPO_ROOT/datasets/raw/ml-1m"
OUT_DIR="$REPO_ROOT/datasets/processed/ml-1m"

if [ ! -f "$RAW_DIR/ml-1m.zip" ]; then
    echo "Missing $RAW_DIR/ml-1m.zip: run scripts/download/datasets.sh ml-1m first." >&2
    exit 1
fi

unzip -o -q "$RAW_DIR/ml-1m.zip" -d "$RAW_DIR"

mkdir -p "$OUT_DIR"
{
    printf 'user_id\tmovie_id\trating\ttimestamp\n'
    awk -F'::' -v OFS='\t' '{ print $1, $2, $3, $4 }' "$RAW_DIR/ml-1m/ratings.dat"
} > "$OUT_DIR/ratings.tsv.tmp"
mv "$OUT_DIR/ratings.tsv.tmp" "$OUT_DIR/ratings.tsv"

ls -l "$OUT_DIR/ratings.tsv"
