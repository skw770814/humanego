"""把双目 mp4 的左半 eye0 落成 JPEG 帧缓存 (SAM2 video 的输入)。

适配: 上游 `sam2_video.py::_prepare_jpeg_frames` 是从已经 staged 好的
`all_data/*/rgb.png` 拷贝成 `%05d.jpg`; 本仓库没有 staged 目录, 就直接从
CameraRecord_*.mp4 解码 —— 用 xrhand.video.iter_frames (ffmpeg rawvideo 管道,
已在本仓库验证过 2160x810 布局), 取左半 `[:, :EYE_W]`。

命名必须是 `%05d.jpg`: SAM2 的 load_video_frames_from_jpg_images 只认 .jpg,
且按 int(stem) 排序。JPEG 质量 95 与上游一致。

颜色: iter_frames 给的是 RGB, cv2.imwrite 期望 BGR, 所以写盘前翻一次通道 ——
这样 cv2.imread 读回来的就是正确的 BGR, 上游 DINOSAM.py 里
`Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))` 那条路才成立。
"""

from __future__ import annotations

import cv2
import numpy as np

from xrhand.video import iter_frames

from xrseg.common import EYE_W, FULL_H, FULL_W, SegPaths


def prepare_eye0_frames(
    paths: SegPaths,
    *,
    max_frames: int | None = None,
    quality: int = 95,
    force: bool = False,
) -> int:
    """解码 mp4 左半并写 `frames/%05d.jpg`, 返回写出的帧数。"""
    destination = paths.frames_dir
    destination.mkdir(parents=True, exist_ok=True)

    written = 0
    for index, frame_rgb in enumerate(
        iter_frames(str(paths.mp4), FULL_W, FULL_H, max_frames=max_frames)
    ):
        output = paths.frame_jpg(index)
        if force or not output.is_file():
            # eye0 = 画面左半 (u < 1080), 见 tools/eye_order_check.py 的视差符号判定
            bgr = np.ascontiguousarray(frame_rgb[:, :EYE_W][:, :, ::-1])
            if not cv2.imwrite(str(output), bgr, [cv2.IMWRITE_JPEG_QUALITY, quality]):
                raise RuntimeError(f"无法写入帧缓存: {output}")
        written = index + 1

    if written == 0:
        raise RuntimeError(f"没有从 {paths.mp4} 解出任何帧")

    # 上次跑没加 --max-frames 时留下的多余帧要清掉, 否则 SAM2 会多传播一段
    for stale in destination.glob("*.jpg"):
        if int(stale.stem) >= written:
            stale.unlink()
    return written
