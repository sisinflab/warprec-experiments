#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW_DIR="$REPO_ROOT/datasets/raw/ml-32m"
OUT_DIR="$REPO_ROOT/datasets/processed/ml-32m"

if [ ! -f "$RAW_DIR/ml-32m.zip" ]; then
    echo "Missing $RAW_DIR/ml-32m.zip: run scripts/download/datasets.sh ml-32m first." >&2
    exit 1
fi

unzip -o -q "$RAW_DIR/ml-32m.zip" -d "$RAW_DIR"

mkdir -p "$OUT_DIR"
cp "$RAW_DIR/ml-32m/ratings.csv" "$OUT_DIR/ratings.csv.tmp"
mv "$OUT_DIR/ratings.csv.tmp" "$OUT_DIR/ratings.csv"

ls -l "$OUT_DIR/ratings.csv"
