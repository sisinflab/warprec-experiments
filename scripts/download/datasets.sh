#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW_DIR="$REPO_ROOT/datasets/raw"

SOURCES="
ml-1m|https://files.grouplens.org/datasets/movielens/ml-1m.zip|a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20
ml-32m|https://files.grouplens.org/datasets/movielens/ml-32m.zip|e4a68655d7386b8f95f2f2424b2ff975dfdd15ffd59e0d864a14dca43e99d6ee
netflix|https://archive.org/download/nf_prize_dataset.tar/nf_prize_dataset.tar.gz|864c7da830766febcc8cb0a4e47b1057183aff6b7ad253425e129d80950e954d
"

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

download() {
    local dataset="$1" url="$2" expected="$3"
    local file="$RAW_DIR/$dataset/$(basename "$url")"

    mkdir -p "$RAW_DIR/$dataset"
    if [ -f "$file" ] && [ "$(sha256_of "$file")" = "$expected" ]; then
        echo "$dataset: already downloaded"
        return
    fi

    curl -fL --insecure --retry 3 -C - -o "$file.part" "$url"
    if [ "$(sha256_of "$file.part")" != "$expected" ]; then
        echo "$dataset: the downloaded file does not match its SHA-256, remove $file.part and retry." >&2
        exit 1
    fi
    mv "$file.part" "$file"
    echo "$dataset: saved in $file"
}

if [ "$#" -eq 0 ]; then
    set -- ml-1m ml-32m netflix
fi

for dataset in "$@"; do
    line="$(echo "$SOURCES" | grep "^$dataset|" || true)"
    if [ -z "$line" ]; then
        echo "Unknown dataset '$dataset'. Available: ml-1m, ml-32m, netflix." >&2
        exit 1
    fi
    IFS='|' read -r name url sha256 <<< "$line"
    download "$name" "$url" "$sha256"
done
