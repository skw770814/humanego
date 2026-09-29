"""视频读写, 走 imageio_ffmpeg 自带的静态 ffmpeg, 不依赖 cv2。

环境里没有系统 ffmpeg/ffprobe, 也没有 cv2 (pip 无网络), 所以统一用
imageio_ffmpeg.get_ffmpeg_exe() 拿到静态 ffmpeg 二进制, 用 rawvideo 管道交换 numpy 数组。

视频布局 (plan §1.4): 2160x810 左右并排立体, 每眼 1080x810 (4:3)。
"""

from __future__ import annotations

import re
import subprocess
from typing import Iterator, Optional

import numpy as np

try:
    import imageio_ffmpeg

    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:  # pragma: no cover
    FFMPEG = "ffmpeg"


def probe(path: str) -> dict:
    """读取视频基本信息: 尺寸 / 帧率 / 帧数 / 时长。

    用 ffmpeg -i 的 stderr 文本解析 —— 环境没有 ffprobe。
    帧数以解码计数为准 (probe 报告的 nb_frames 在部分封装里缺失)。
    """
    p = subprocess.run(
        [FFMPEG, "-hide_banner", "-i", path],
        capture_output=True,
        text=True,
        errors="replace",
    )
    err = p.stderr
    out: dict = {"path": path}

    m = re.search(r"Stream #\d+:\d+.*?: Video: .*?, (\d+)x(\d+)", err)
    if m:
        out["width"], out["height"] = int(m.group(1)), int(m.group(2))

    m = re.search(r"(\d+(?:\.\d+)?) fps", err)
    if m:
        out["fps"] = float(m.group(1))

    m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", err)
    if m:
        h, mi, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
        out["duration_s"] = h * 3600 + mi * 60 + s

    m = re.search(r"Video: (h264|hevc|mpeg4)", err)
    if m:
        out["codec"] = m.group(1)

    return out


def count_frames(path: str) -> int:
    """精确解码计数 (probe 的 nb_frames 不可靠时用)。"""
    p = subprocess.run(
        [FFMPEG, "-hide_banner", "-i", path, "-f", "null", "-"],
        capture_output=True,
        text=True,
        errors="replace",
    )
    hits = re.findall(r"frame=\s*(\d+)", p.stderr)
    return int(hits[-1]) if hits else 0


def frame_pts(path: str) -> np.ndarray:
    """取每帧的 pts (秒)。用于 plan §2.3 的时长漂移校验。"""
    p = subprocess.run(
        [
            FFMPEG,
            "-hide_banner",
            "-i",
            path,
            "-vf",
            "showinfo",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        errors="replace",
    )
    return np.array(
        [float(x) for x in re.findall(r"pts_time:([0-9.]+)", p.stderr)], dtype=np.float64
    )


def iter_frames(
    path: str, width: int, height: int, scale: float = 1.0, max_frames: Optional[int] = None
) -> Iterator[np.ndarray]:
    """逐帧解码为 RGB uint8 (H,W,3)。

    scale<1 时让 ffmpeg 做缩放, 便于快速预览 / 相位相关计算。
    """
    vf = [] if scale == 1.0 else [f"scale=iw*{scale}:ih*{scale}"]
    ow, oh = int(round(width * scale)), int(round(height * scale))

    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-i", path]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    if max_frames is not None:
        cmd += ["-frames:v", str(max_frames)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)
    nbytes = ow * oh * 3
    try:
        while True:
            buf = proc.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(oh, ow, 3)
    finally:
        if proc.stdout:
            proc.stdout.close()
        proc.wait()


class VideoWriter:
    """把 numpy 帧编码成 mp4。

    libx264 + yuv420p + faststart, 保证能被常见播放器直接打开。
    """

    def __init__(self, path: str, width: int, height: int, fps: float, crf: int = 18):
        self.width, self.height, self.fps = width, height, fps
        cmd = [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            path,
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.path = path
        self._n = 0

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(
                f"帧尺寸 {frame.shape} != 期望 {(self.height, self.width, 3)}"
            )
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        self._n += 1

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        err = self.proc.stderr.read().decode("utf-8", "replace") if self.proc.stderr else ""
        self.proc.wait()
        if self.proc.returncode != 0:
            raise RuntimeError(f"ffmpeg 编码失败 (rc={self.proc.returncode}):\n{err}")

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def n_written(self) -> int:
        return self._n


def save_png(frame: np.ndarray, path: str) -> None:
    from PIL import Image

    Image.fromarray(frame).save(path)


if __name__ == "__main__":
    import sys

    for p in sys.argv[1:]:
        info = probe(p)
        info["n_frames"] = count_frames(p)
        pts = frame_pts(p)
        if pts.size:
            info["pts_first"] = float(pts[0])
            info["pts_last"] = float(pts[-1])
            info["pts_step_median"] = float(np.median(np.diff(pts)))
        for k, v in info.items():
            print(f"  {k}: {v}")
        print()
