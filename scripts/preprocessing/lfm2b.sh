#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW_DIR="$REPO_ROOT/datasets/raw/lfm2b"
OUT_DIR="$REPO_ROOT/datasets/processed/lfm2b"

if [ ! -f "$RAW_DIR/LFM-2b.zip" ]; then
    echo "Missing $RAW_DIR/LFM-2b.zip: place the LFM-2b archive there first." >&2
    exit 1
fi

unzip -o "$RAW_DIR/LFM-2b.zip" -d "$RAW_DIR"

mkdir -p "$OUT_DIR"
bzip2 -dc "$RAW_DIR/LFM-2b/listening-events.tsv.bz2" > "$OUT_DIR/listening-events.tsv.tmp"
mv "$OUT_DIR/listening-events.tsv.tmp" "$OUT_DIR/listening-events.tsv"

ls -l "$OUT_DIR/listening-events.tsv"
