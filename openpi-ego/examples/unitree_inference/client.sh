#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CLIENT_ENV="$SCRIPT_DIR/.venv"
UNITREE_PYTHON="${UNITREE_PYTHON:-/home/zh/miniconda3/envs/unitree_deploy/bin/python}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache-openpi-unitree-client}"

MODE="${1:-}"
PROMPT="${2:-}"
HOST="${3:-127.0.0.1}"
PORT="${4:-8000}"
ROBOT_TYPE="${5:-auto}"
EXTRA_ARGS=("${@:6}")

case "$MODE" in
  sync) SCRIPT="sync_inference.py" ;;
  async) SCRIPT="async_inference.py" ;;
  temporal_ensembling) SCRIPT="temporal_ensembling_inference.py" ;;
  temporal_smoothing) SCRIPT="temporal_smoothing_inference.py" ;;
  rtc) SCRIPT="rtc_inference.py" ;;
  *)
    echo "用法: $0 {sync|async|temporal_ensembling|temporal_smoothing|rtc} '任务指令' [host] [port] [auto|unitree_g1_dex1|unitree_g1_brainco]" >&2
    exit 2
    ;;
esac
if [[ -z "$PROMPT" ]]; then
  echo "必须提供任务指令。" >&2
  exit 2
fi
if [[ ! -x "$UNITREE_PYTHON" ]]; then
  echo "找不到 Unitree Python: $UNITREE_PYTHON" >&2
  exit 2
fi

if [[ ! -x "$CLIENT_ENV/bin/python" ]]; then
  uv venv --no-project --offline --python "$UNITREE_PYTHON" --system-site-packages "$CLIENT_ENV"
fi

export PYTHONPATH="$ROOT_DIR/packages/openpi-client/src:/home/zh/unitree-deploy${PYTHONPATH:+:$PYTHONPATH}"
exec "$CLIENT_ENV/bin/python" "$SCRIPT_DIR/$SCRIPT" \
  --host "$HOST" \
  --port "$PORT" \
  --robot-type "$ROBOT_TYPE" \
  --prompt "$PROMPT" \
  "${EXTRA_ARGS[@]}"
