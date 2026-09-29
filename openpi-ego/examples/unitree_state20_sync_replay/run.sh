#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CLIENT_ENV="$SCRIPT_DIR/.venv"
UNITREE_PYTHON="${UNITREE_PYTHON:-/home/zh/miniconda3/envs/unitree_deploy/bin/python}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache-openpi-unitree-state20-sync}"

if [[ ! -x "$UNITREE_PYTHON" ]]; then
  echo "找不到 Unitree Python: $UNITREE_PYTHON" >&2
  exit 2
fi

if [[ ! -x "$CLIENT_ENV/bin/python" ]]; then
  uv venv --no-project --offline --python "$UNITREE_PYTHON" --system-site-packages "$CLIENT_ENV"
fi

export PYTHONPATH="$OPENPI_ROOT/packages/openpi-client/src:/home/zh/unitree-deploy${PYTHONPATH:+:$PYTHONPATH}"
exec "$CLIENT_ENV/bin/python" "$SCRIPT_DIR/replay_state_chunks.py" "$@"
