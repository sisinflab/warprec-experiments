#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW_DIR="$REPO_ROOT/datasets/raw/spotify-mpd"
OUT_DIR="$REPO_ROOT/datasets/processed/spotify-mpd"

if [ ! -f "$RAW_DIR/spotify_million_playlist_dataset.zip" ]; then
    echo "Missing $RAW_DIR/spotify_million_playlist_dataset.zip: place the archive there first." >&2
    exit 1
fi

unzip -o -q "$RAW_DIR/spotify_million_playlist_dataset.zip" -d "$RAW_DIR"

mkdir -p "$OUT_DIR"
python3 - "$RAW_DIR/data" "$OUT_DIR/playlists.tsv.tmp" <<'EOF'
import json
import sys
from pathlib import Path

data_dir, out_path = sys.argv[1:]

slices = sorted(
    Path(data_dir).glob("mpd.slice.*.json"),
    key=lambda path: int(path.name.split(".")[2].split("-")[0]),
)

with open(out_path, "w", encoding="utf-8") as out:
    out.write("pid\ttrack_uri\tpos\n")
    for path in slices:
        with open(path, encoding="utf-8") as f:
            playlists = json.load(f)["playlists"]
        for playlist in playlists:
            pid = playlist["pid"]
            out.writelines(
                f"{pid}\t{track['track_uri']}\t{track['pos']}\n"
                for track in playlist["tracks"]
            )
EOF
mv "$OUT_DIR/playlists.tsv.tmp" "$OUT_DIR/playlists.tsv"

ls -l "$OUT_DIR/playlists.tsv"
