#!/usr/bin/env bash
# ============================================================================
# 耳机壳分割链路的入口包装: 自动 source 环境变量 + 用本链路自己的 venv 解释器。
#
# 用法 (位置参数与 tools/seg_object.py 完全一致):
#   bash tools/seg_run.sh 20260920_111300 --prompt "earphone case" --stage check
#   bash tools/seg_run.sh 20260920_111300 --prompt "earphone case" --stage frames
#   bash tools/seg_run.sh 20260920_111300 --prompt "earphone case" --stage image
#   bash tools/seg_run.sh 20260920_111300 --prompt "earphone case" --stage video
#   bash tools/seg_run.sh 20260920_111300 --prompt "earphone case" --stage render
#   bash tools/seg_run.sh 20260920_111300 20260920_111342 --prompt "earphone case" --stage all
#
# 等价于:
#   source tools/seg_env.sh
#   .venv/bin/python tools/seg_object.py ...
# ============================================================================

set -euo pipefail

WORK="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$WORK/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
    echo "✗ 没有 $PY —— 先装环境: bash $WORK/tools/seg_setup.sh" >&2
    exit 2
fi

# shellcheck source=/dev/null
source "$WORK/tools/seg_env.sh"

exec "$PY" "$WORK/tools/seg_object.py" "$@"
