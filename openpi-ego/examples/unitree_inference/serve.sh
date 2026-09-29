#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache-openpi-unitree}"

if [[ "${1:-}" == "--list" || -z "${1:-}" ]]; then
  exec uv run --project "$ROOT_DIR" python "$ROOT_DIR/scripts/serve_unitree_policy.py" --list
fi

CONFIG="$1"
CHECKPOINT="${2:-}"
PORT="${3:-8000}"
ARGS=(--config "$CONFIG" --port "$PORT")
if [[ -n "$CHECKPOINT" ]]; then
  ARGS+=(--checkpoint "$CHECKPOINT")
fi

exec uv run --project "$ROOT_DIR" python "$ROOT_DIR/scripts/serve_unitree_policy.py" "${ARGS[@]}"
