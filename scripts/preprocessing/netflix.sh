#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW_DIR="$REPO_ROOT/datasets/raw/netflix"
OUT_DIR="$REPO_ROOT/datasets/processed/netflix"

if [ ! -f "$RAW_DIR/nf_prize_dataset.tar.gz" ]; then
    echo "Missing $RAW_DIR/nf_prize_dataset.tar.gz: run scripts/download/datasets.sh netflix first." >&2
    exit 1
fi

tar -xzf "$RAW_DIR/nf_prize_dataset.tar.gz" -C "$RAW_DIR"
tar -xf "$RAW_DIR/download/training_set.tar" -C "$RAW_DIR/download"

mkdir -p "$OUT_DIR"
{
    printf 'customer_id\tmovie_id\trating\tdate\n'
    find "$RAW_DIR/download/training_set" -name 'mv_*.txt' | sort | xargs awk -F, -v OFS='\t' '
        { sub(/\r$/, "") }
        FNR == 1 { movie = $0; sub(/:$/, "", movie); next }
        { print $1, movie, $2, $3 }
    '
} > "$OUT_DIR/ratings.tsv.tmp"
mv "$OUT_DIR/ratings.tsv.tmp" "$OUT_DIR/ratings.tsv"

ls -l "$OUT_DIR/ratings.tsv"
