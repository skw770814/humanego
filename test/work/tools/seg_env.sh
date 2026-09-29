#!/usr/bin/env bash
# 用法: source tools/seg_env.sh
#
# 分割链路需要的环境变量。每个新 shell 都要先 source 一次 —— 主要是 HF_ENDPOINT:
# 本机 huggingface.co 直连超时 (20 s 无响应), hf-mirror.com 0.57 s 就通。
# HF_HOME 指到 test/work/.cache 下, 与 ego_relation_policy 的 models/ 约定同构。

# shellcheck disable=SC2155
export SEG_WORK="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="$SEG_WORK/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_TELEMETRY=1

# sam2 的 sdist 安装与运行都不需要 CUDA 扩展; 显式关掉, 防止误编译。
export SAM2_BUILD_CUDA=0

# 复制的 sam2_video.py 从环境变量读 cache_dir 的兜底值
export SEG_MODELS_DIR="$HF_HOME"

echo "seg env:"
echo "  HF_ENDPOINT=$HF_ENDPOINT"
echo "  HF_HOME=$HF_HOME"
echo "  venv python=$SEG_WORK/.venv/bin/python"
