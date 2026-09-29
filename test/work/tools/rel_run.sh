#!/usr/bin/env bash
# ============================================================================
# step3 (右手末端相对分割物体的位姿) 的入口包装: 用本链路自己的 venv 解释器。
#
# 为什么必须包一层: cv2 / h5py / pyarrow 只装在 test/work/.venv 里, 系统 anaconda
# python 没有 cv2。链路里既要 import xrhand / tools/overlay.py (step1 那套), 又要
# import ego_relation_policy 的 stereo_depth.py (它的骨架), 两边都只在 .venv 里能过。
#
# 用法 (位置参数与 tools/rel_object.py 完全一致):
#   bash tools/rel_run.sh 20260920_111342 --stage check
#   bash tools/rel_run.sh 20260920_111342 --stage all
#   bash tools/rel_run.sh 20260920_111342 --stage render --frames 50,120,260
#   PHANTOM_STAGE3_CMD='...' bash tools/rel_run.sh 20260920_111342 --stage step3 --frames 25,79,138,190,214
#   bash tools/rel_run.sh 20260920_111342 --stage relation --frames 50,120,260
#
# 等价于:
#   .venv/bin/python tools/rel_object.py ...
#
# 不 source tools/seg_env.sh: 那条链的环境变量是给 SAM2 / HuggingFace 用的
# (HF_ENDPOINT / HF_HOME / SAM2_BUILD_CUDA), 本链路一个都不需要, 也不该被它污染。
# ============================================================================

set -euo pipefail

WORK="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$WORK/.venv/bin/python"

# Make CUDA/cuDNN libraries installed in this venv visible to PyTorch/ONNX
# Runtime/Piper.  The paths are derived from WORK so the launcher is not
# dependent on the caller's current directory.
CUDA_SITE="$WORK/.venv/lib/python3.11/site-packages/nvidia"
export LD_LIBRARY_PATH="$CUDA_SITE/cublas/lib:$CUDA_SITE/cudnn/lib:$CUDA_SITE/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"

if [[ ! -x "$PY" ]]; then
    echo "✗ 没有 $PY —— 先装环境: bash $WORK/tools/seg_setup.sh" >&2
    exit 2
fi

exec "$PY" "$WORK/tools/rel_object.py" "$@"
