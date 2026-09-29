#!/usr/bin/env bash
# ============================================================================
# 耳机壳分割链路的环境安装 (幂等, 可重复执行)
#
# 目标: 在 test/work 下建一个独立 venv, 只装这条链路要用的东西。
#       不跑 ego_relation_policy 的 `uv sync --group perception` —— 那会把
#       mujoco / cotracker / orient-anything / xformers 一起拖进来。
#
# 版本对齐 ego_relation_policy/pyproject.toml 的 perception 组:
#   torch==2.5.1 / torchvision==0.20.1 (pytorch-cu121 索引), sam2>=1.0,
#   transformers>=4.40,<5
#
# 本机事实 (2026-09-21 实测):
#   GPU   RTX 4060 Laptop 8 GB, compute_cap 8.9 (Ada/sm_89) —— cu121 轮子覆盖
#   HF    huggingface.co 直连超时; hf-mirror.com 通 (0.57 s) => 必须设 HF_ENDPOINT
#
# 用法:
#   bash tools/seg_setup.sh            # 建 venv + 装依赖 (不含权重)
#   source tools/seg_env.sh            # 之后每个 shell 先 source 这个拿环境变量
# ============================================================================

set -euo pipefail

WORK="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORK"

VENV="$WORK/.venv"
PY="$VENV/bin/python"
UV="${UV:-uv}"

echo "=== [1/5] venv: $VENV (python 3.11) ==="
if [[ ! -x "$PY" ]]; then
    # uv 里已经缓存了 cpython-3.11.15, 不必联网
    UV_PYTHON_DOWNLOADS=never "$UV" venv --python 3.11 --seed "$VENV"
else
    echo "    已存在, 跳过"
fi
"$PY" -V

echo
echo "=== [2/5] torch 2.5.1 + torchvision 0.20.1 (cu121) ==="
# 关键: 不能加 --no-deps —— nvidia-* 运行期库要一起装。
# 用 +cu121 本地版本号钉死, 避免与 PyPI 上同版本号的 cu124 默认构建混淆。
if ! "$UV" pip install --python "$PY" \
        --index-url https://download.pytorch.org/whl/cu121 \
        "torch==2.5.1+cu121" "torchvision==0.20.1+cu121" 2>/dev/null; then
    echo "    单独 pytorch 索引失败, 加 PyPI 作补充索引重试"
    "$UV" pip install --python "$PY" \
        --index-url https://download.pytorch.org/whl/cu121 \
        --extra-index-url https://pypi.org/simple \
        --index-strategy unsafe-best-match \
        "torch==2.5.1+cu121" "torchvision==0.20.1+cu121"
fi

echo
echo "=== [3/5] 数值栈 + 构建后端 ==="
# sam2 是 sdist, 后面要用 --no-build-isolation, 要求目标环境里已有构建后端
"$UV" pip install --python "$PY" --index-url https://pypi.org/simple \
    "numpy>=1.24.4,<2.3" "pillow>=10" "setuptools>=70" wheel

echo
echo "=== [4/5] sam2 1.1.0 (sdist, 关 CUDA 扩展) ==="
# setup.py 默认 SAM2_BUILD_CUDA=1 会去编译 CUDAExtension (本机无 nvcc) => 必须关掉。
# --no-build-isolation: 否则隔离环境会按 [build-system].requires 再装一份 torch。
# --no-deps: sam2 的真实依赖下一步单独装。
SAM2_BUILD_CUDA=0 SAM2_BUILD_ALLOW_ERRORS=1 "$UV" pip install --python "$PY" \
    --index-url https://pypi.org/simple --no-deps --no-build-isolation sam2==1.1.0

echo
echo "=== [5/5] 其余运行期依赖 ==="
"$UV" pip install --python "$PY" --index-url https://pypi.org/simple \
    hydra-core iopath tqdm \
    "transformers>=4.40,<5" \
    opencv-python-headless imageio imageio-ffmpeg python-box pyyaml matplotlib scipy

echo
echo "=== 安装完成, 自检 ==="
"$PY" - <<'EOF'
import torch, torchvision, sam2, transformers, cv2, hydra, iopath
from pathlib import Path
print(f"  torch        {torch.__version__}")
print(f"  torchvision  {torchvision.__version__}")
print(f"  cuda ready   {torch.cuda.is_available()}"
      + (f"  -> {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
print(f"  sam2         {getattr(sam2, '__version__', '(无版本号)')}  @ {Path(sam2.__path__[0])}")
print(f"  transformers {transformers.__version__}")
print(f"  cv2          {cv2.__version__}")
cfg = Path(sam2.__path__[0]) / "configs" / "sam2" / "sam2_hiera_t.yaml"
assert cfg.is_file(), f"sam2 包里缺 hydra 配置: {cfg}"
text = cfg.read_text()
assert "feat_sizes: [64, 64]" in text, "sam2_hiera_t.yaml 的 feat_sizes 不是 [64, 64]"
assert "image_size: 1024" in text, "sam2_hiera_t.yaml 的 image_size 不是 1024"
print(f"  hydra cfg    {cfg}  feat_sizes=[64,64] image_size=1024  OK")
EOF
