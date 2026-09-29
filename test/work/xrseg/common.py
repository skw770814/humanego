"""公共小件: 路径约定 / 提示词归一化 / 内存预检。

目录约定 (沿用复制过来的 sam2_video.py 与 DINOSAM.py 的命名):

    out/seg_<stem>/
      frames/00000.jpg          eye0 JPEG 帧缓存, 就是 SAM2 video 的输入
      masks/obj1/00000.png      逐帧 mask, 0/255
      mask_obj1.png             提示帧上的 DINO+SAM2 mask —— 上游 process_and_save
                                的命名, 同时是 SAM2 video add_new_mask 的初值
      prompt_frame.png          提示帧核验图 (人工确认 DINO 框住的是不是耳机壳)
      still_f####.png           抽帧静帧 (带放大内嵌)
      quality.png               面积比 / score / 手部重叠曲线
      report.json metrics.npz
      seg_overlay_<stem>.mp4    主产物 2160x810
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PKG = Path(__file__).resolve().parent  # test/work/xrseg
WORK = PKG.parent  # test/work
DATA = WORK.parent  # test/ (CameraRecord_*.mp4 / trackingData_*.txt)

DEFAULT_OUTDIR = WORK / "out"
DEFAULT_CFG = PKG / "cfg" / "DINOSAM.yaml"

# 双目布局 (见 xrhand/camera.py): 2160x810 = 左半 eye0 + 右半 eye1
EYE_W, EYE_H = 1080, 810
FULL_W, FULL_H = 2160, 810

INSTANCE_ID = "obj1"
# 物体实例 id 的**规范顺序**: obj1..objN。与参考的 `object_catalog` 同一个命名法
# (参考是 obj1..obj3), 顺序即 state 里各物体的拼接顺序。
MAX_OBJECTS = 3
INSTANCE_IDS = tuple(f"obj{i}" for i in range(1, MAX_OBJECTS + 1))

# SAM2 video predictor 在 init_state 里会把每帧物化成 (3,1024,1024) float32
# (sam2/utils/misc.py::load_video_frames) —— 12.58 MB/帧, 与帧数线性相关。
SAM2_FRAME_TENSOR_BYTES = 3 * 1024 * 1024 * 4

# 逐物体的叠加配色 (BGR), 照抄上游 `sam2_video._COLORS`。
MASK_COLORS = {
    "obj1": (60, 190, 80),
    "obj2": (40, 40, 235),
    "obj3": (20, 220, 245),
}
COLOR = MASK_COLORS[INSTANCE_ID]  # BGR, 同上游 _COLORS["obj1"]


def instance_color(instance_id: str) -> tuple[int, int, int]:
    """物体的叠加色; 规范表里没有的 (自定义 instance_id) 退回 obj1 的绿。"""
    return MASK_COLORS.get(str(instance_id), COLOR)


def known_instance_ids(names) -> list[str]:
    """把一组目录名排成规范顺序 (obj1..objN 在前, 其余字典序在后)。"""
    present = {str(name) for name in names}
    known = [i for i in INSTANCE_IDS if i in present]
    return known + sorted(present - set(INSTANCE_IDS))


def normalize_prompt(text: str) -> str:
    """GroundingDINO 的文本提示规范: 小写、词间单空格、结尾 " ."。

    上游 cfg/DINOSAM.yaml 里写的是 "a pretty cat ." 这种形态。
    """
    words = " ".join(str(text).strip().lower().replace(".", " ").split())
    if not words:
        raise ValueError("提示词为空")
    return words + " ."


def models_cache_dir() -> Path | None:
    """HF 权重缓存目录; 没设环境变量就返回 None (交给 huggingface_hub 默认)。"""
    value = os.environ.get("HUGGINGFACE_HUB_CACHE") or ""
    return Path(value) if value else None


def mem_available_bytes() -> int:
    """/proc/meminfo 的 MemAvailable —— 比 psutil 少一个依赖。"""
    with open("/proc/meminfo", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return 0


def human_gb(value: float) -> str:
    return f"{value / 1e9:.1f} GB"


def sam2_cuda_ext_available() -> bool:
    """sam2._C (CUDA 扩展) 在不在。

    本机没装 nvcc, sdist 装 sam2 时用 SAM2_BUILD_CUDA=0 跳过了编译, 所以是 False。
    影响: `sam2/utils/misc.py::fill_holes_in_mask_scores` 会 `from sam2 import _C` 失败,
    被那个函数自己的 `except Exception` 吞掉 -> 只 warn 一次、不填洞 (见 plan R4)。
    别的路径 (图像/视频推理本体) 都是纯 PyTorch, 不受影响。
    """
    import importlib.util

    try:
        return importlib.util.find_spec("sam2._C") is not None
    except (ImportError, ValueError):
        return False


def preflight_memory(frame_count: int, *, offload_video_to_cpu: bool) -> str | None:
    """SAM2 帧张量的落点预检 (plan R1: 本机 RAM 才是瓶颈, 显存反而空)。

    offload_video_to_cpu=True 时这 12.58 MB/帧 落在**系统内存**, 而本机 15 GB RAM
    常被浏览器占到只剩 1-2 GB; 返回一句给用户看的告警, 不够就返回 None。
    """
    need = frame_count * SAM2_FRAME_TENSOR_BYTES
    if not offload_video_to_cpu:
        return None  # 进显存, 不占 RAM
    avail = mem_available_bytes()
    if avail >= need * 1.4:  # 留 40% 给解码/激活/JPEG 缓冲
        return None
    return (
        f"SAM2 会把 {frame_count} 帧物化成 {human_gb(need)} 的张量, 且 offload_video_to_cpu=True "
        f"决定它落在**系统内存**里; 当前 MemAvailable 只有 {human_gb(avail)} (需要约 "
        f"{human_gb(need * 1.4)})。先关掉浏览器/其他大内存进程, 或改用 --video-in-vram "
        f"(把张量放进 8 GB 显存, 见 plan R1)。"
    )


@dataclass(frozen=True)
class SegPaths:
    """一条采集的分割产物目录。"""

    stem: str
    outdir: Path

    @classmethod
    def for_stem(cls, stem: str, outdir: str | Path = DEFAULT_OUTDIR) -> "SegPaths":
        return cls(stem=stem, outdir=Path(outdir) / f"seg_{stem}")

    # ---- 输入 (采集原始数据, 在 test/ 下) ----
    @property
    def mp4(self) -> Path:
        return DATA / f"CameraRecord_{self.stem}.mp4"

    @property
    def tracking(self) -> Path:
        return DATA / f"trackingData_{self.stem}.txt"

    # ---- 帧缓存与 mask ----
    @property
    def frames_dir(self) -> Path:
        return self.outdir / "frames"

    @property
    def masks_dir(self) -> Path:
        return self.outdir / "masks"

    def instance_ids(self) -> list[str]:
        """`masks/` 下**实际存在**的物体目录, 按规范顺序 (obj1..objN)。

        多物体时"有哪些物体"的唯一事实来源 —— 不去读 report.json / prompt.json,
        免得盘上的 mask 与报告里的清单不一致时两边各说各话。
        """
        if not self.masks_dir.is_dir():
            return []
        return known_instance_ids(p.name for p in self.masks_dir.iterdir() if p.is_dir())

    def instance_mask_dir(self, instance_id: str = INSTANCE_ID) -> Path:
        return self.masks_dir / str(instance_id)

    def mask(self, frame: int, instance_id: str = INSTANCE_ID) -> Path:
        return self.instance_mask_dir(instance_id) / f"{frame:05d}.png"

    def prompt_mask(self, instance_id: str = INSTANCE_ID) -> Path:
        """提示帧上的 mask, 上游命名 mask_{instance_id}.png。"""
        return self.outdir / f"mask_{instance_id}.png"

    @property
    def prompt_json(self) -> Path:
        return self.outdir / "prompt.json"

    @property
    def prompt_frame_png(self) -> Path:
        return self.outdir / "prompt_frame.png"

    def prompt_frame_for(self, instance_id: str = INSTANCE_ID) -> Path:
        """提示帧核验图。obj1 沿用上游那个文件名, 其余物体带 instance_id 后缀。"""
        if str(instance_id) == INSTANCE_ID:
            return self.prompt_frame_png
        return self.outdir / f"prompt_frame_{instance_id}.png"

    # ---- 其余产物 ----
    @property
    def quality_png(self) -> Path:
        return self.outdir / "quality.png"

    @property
    def report_json(self) -> Path:
        return self.outdir / "report.json"

    @property
    def metrics_npz(self) -> Path:
        return self.outdir / "metrics.npz"

    @property
    def overlay_video(self) -> Path:
        return self.outdir / f"seg_overlay_{self.stem}.mp4"

    def still(self, frame: int) -> Path:
        return self.outdir / f"still_f{frame:04d}.png"

    def frame_jpg(self, frame: int) -> Path:
        return self.frames_dir / f"{frame:05d}.jpg"
