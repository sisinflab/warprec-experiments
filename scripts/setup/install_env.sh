#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FRAMEWORK="${1:?usage: scripts/setup/install_env.sh <framework>}"
ENV_FILE="$REPO_ROOT/environments/$FRAMEWORK.yml"
ENV_PREFIX="${ENV_PREFIX:-$REPO_ROOT/envs/$FRAMEWORK}"

export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$REPO_ROOT/.cache/conda-pkgs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$REPO_ROOT/.cache/pip}"
export CONDA_OVERRIDE_CUDA="${CONDA_OVERRIDE_CUDA:-12.0}"

if [ ! -f "$ENV_FILE" ]; then
    echo "No environments/$FRAMEWORK.yml. Available:" >&2
    ls "$REPO_ROOT/environments" | sed -n 's/\.yml$//p' | sed 's/^/  /' >&2
    exit 1
fi

if ! ls "$REPO_ROOT/frameworks/$FRAMEWORK"/* >/dev/null 2>&1; then
    echo "frameworks/$FRAMEWORK is empty: run 'git submodule update --init --recursive' first." >&2
    exit 1
fi

if [ -e "$ENV_PREFIX" ]; then
    echo "$ENV_PREFIX already exists: remove it to reinstall." >&2
    exit 1
fi

CONDA=""
for candidate in "${MAMBA_EXE:-}" micromamba mamba conda; do
    if [ -n "$candidate" ] && command -v "$candidate" >/dev/null 2>&1; then
        CONDA="$candidate"
        break
    fi
done
if [ -z "$CONDA" ]; then
    echo "No micromamba, mamba or conda found on the PATH." >&2
    exit 1
fi

cd "$REPO_ROOT/environments"
case "$(basename "$CONDA")" in
    micromamba) "$CONDA" create -y -p "$ENV_PREFIX" -f "$FRAMEWORK.yml" ;;
    *) "$CONDA" env create -p "$ENV_PREFIX" -f "$FRAMEWORK.yml" ;;
esac

"$CONDA" run -p "$ENV_PREFIX" python -c "
import sys
print('python', sys.version.split()[0])
try:
    import torch
    print('torch', torch.__version__, '| CUDA build', torch.version.cuda)
except ImportError:
    print('torch not installed')
"
