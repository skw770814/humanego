#!/usr/bin/env bash
# ============================================================================
# pipeline 自包含入口: 用 test/pipeline/.venv 跑 xrpipe。
#
#   bash run.sh step1 20260920_111342
#   bash run.sh step2 --stems 20260920_111342 --prompt "small white earbud case"
#   bash run.sh step3 --stems 20260920_111342 --only-keep
#   bash run.sh step4 --stems 20260920_111342 --dataset demo
#   bash run.sh all   --stems 20260920_111342,20260920_111300 --dataset smoke \
#                     --prompt "small white earbud case" --prompt "<另一段的类别>"
#
# 环境 = pipeline/.venv 的 CUDA/cuDNN 路径（供 torch /
# onnxruntime / pyrender 用) + tools/seg_env.sh 的 HuggingFace 变量 (SAM2 权重下载;
# 本机直连 huggingface.co 超时, 走 hf-mirror)。
#
# 不设 PIPER_GL_PLATFORM: xrrel/piper.py 的 "auto" 已经处理了 (有 DISPLAY 走 X11,
# 真 headless 才退回 egl), 从外面再设一次只会覆盖掉那个判断。
# ============================================================================

set -euo pipefail

PIPE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$PIPE/.venv/bin/python"

CUDA_SITE="$PIPE/.venv/lib/python3.11/site-packages/nvidia"
export LD_LIBRARY_PATH="$CUDA_SITE/cublas/lib:$CUDA_SITE/cudnn/lib:$CUDA_SITE/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="$PIPE/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_TELEMETRY=1
export SAM2_BUILD_CUDA=0
export SEG_MODELS_DIR="$HF_HOME"

if [[ ! -x "$PY" ]]; then
    echo "✗ 没有 $PY —— 先装环境: bash $PIPE/tools/seg_setup.sh" >&2
    exit 2
fi

exec env PYTHONPATH="$PIPE${PYTHONPATH:+:$PYTHONPATH}" "$PY" -m xrpipe "$@"
