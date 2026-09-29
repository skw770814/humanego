"""右手 26 点骨架叠加层。

复用现成的两件东西, 不重复造:
  - `tools/overlay.py::Reel` —— 已经处理了 "视频帧号 → tracking 记录" 的 lag 映射
    (lag 取自 out/align_<stem>.json), 以及 26 点在两半画面上的投影;
  - `xrhand/render.py::draw_hand` —— 手掌多边形 + 骨骼 + 关节点, 传入 half 时自己
    会加 `half * EYE_W` 偏移, 所以可以直接画在整幅 2160 宽的画面上。

tools/ 不是包 (没有 __init__.py), 所以按文件路径 importlib 载入 overlay.py。
缺 out/align_<stem>.json 或 out/calib.json 时抛 HandUnavailable, 由 CLI 降级成
--no-hand 并告警, 不中断整条链路 (plan R7)。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from xrhand import render as R
from xrhand.camera import EYE_W, CameraParams

from xrseg.common import WORK


class HandUnavailable(RuntimeError):
    """拿不到投影所需的产物 —— 调用方应降级, 而不是报错退出。"""


def _load_overlay_module():
    """按路径载入 tools/overlay.py (它自己会 sys.path.insert(0, WORK))。"""
    path = WORK / "tools" / "overlay.py"
    if not path.is_file():
        raise HandUnavailable(f"找不到 {path}")
    name = "xrseg_tools_overlay"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class HandLayer:
    """一条采集的右手骨架绘制器。"""

    def __init__(self, stem: str, *, calib: str | Path | None = None, lag: int | None = None):
        out = WORK / "out"
        align_json = out / f"align_{stem}.json"
        if not align_json.is_file():
            raise HandUnavailable(
                f"缺 {align_json} (帧对齐产物), 先跑 python -m xrhand.align; "
                f"或用 --no-hand 只出 mask 叠加"
            )

        params = CameraParams()
        self.calib_note = "标称参数 (未标定)"
        calib_path = Path(calib) if calib else WORK / "config" / "calib.json"
        if calib_path.is_file():
            import json

            with open(calib_path, encoding="utf-8") as handle:
                payload = json.load(handle)
            params = CameraParams.from_json(payload["params"])
            self.calib_note = f"{calib_path.name} (d_f={params.d_f:+.3f})"
        else:
            raise HandUnavailable(
                f"缺 {calib_path} —— 用标称参数画出来的骨架会明显偏, 不如不画; "
                f"先跑 python tools/make_calib.py, 或用 --no-hand"
            )

        overlay = _load_overlay_module()
        self.reel = overlay.Reel(stem, lag, params)
        self.stem = stem
        self.lag = self.reel.lag
        self.n_frames = self.reel.n_frames
        self._hull_cache: dict[tuple[int, int], np.ndarray | None] = {}

    # ------------------------------------------------------------ 画
    def draw(
        self, frame_rgb: np.ndarray, frame: int, *, radius: int = 4
    ) -> Image.Image:
        """在整幅 (2160x810) RGB 帧上叠右手骨架, 返回 RGBA 图。"""
        img = Image.fromarray(frame_rgb).convert("RGBA")
        draw = ImageDraw.Draw(img)
        for half in range(2):
            R.draw_hand(
                draw,
                self.reel.U[frame, half],
                self.reel.V[frame, half],
                self.reel.Z[frame, half],
                self.reel.VALID[frame],
                half,
                radius=radius,
                alpha_fill=True,
                img_for_poly=img,
            )
        return img

    # ------------------------------------------------------------ 几何量
    def hull(self, frame: int, half: int = 0) -> np.ndarray | None:
        """该帧右手可见关节的凸包 (整幅坐标, 已含半幅偏移); 点不够时 None。"""
        key = (frame, half)
        if key in self._hull_cache:
            return self._hull_cache[key]
        from scipy.spatial import ConvexHull, QhullError

        usable = self.reel.usable[frame, half]
        u = self.reel.U[frame, half][usable]
        v = self.reel.V[frame, half][usable]
        result = None
        if u.size >= 3:
            points = np.stack([u + half * EYE_W, v], axis=1)
            try:
                result = points[ConvexHull(points).vertices]
            except QhullError:  # 三点共线/退化成一条线
                result = None
        self._hull_cache[key] = result
        return result

    def overlap_ratio(self, frame: int, support: np.ndarray) -> float:
        """mask ∩ 右手凸包 / mask —— 判断"框太松把手指吃进去了"(plan R5)。

        用 cv2.fillPoly 而不是 matplotlib 的 contains_points: 后者对 50 万个像素点
        要几百毫秒, 452 帧下来就是好几分钟。
        """
        import cv2

        area = float(support.sum())
        if area <= 0:
            return 0.0
        hull = self.hull(frame, 0)  # mask 只画在左半 eye0 上
        if hull is None:
            return 0.0
        canvas = np.zeros(support.shape, dtype=np.uint8)
        cv2.fillPoly(canvas, [np.round(hull).astype(np.int32)], 1)
        return float(np.count_nonzero((canvas > 0) & support)) / area

    def summary(self, frame: int) -> str:
        left = self.reel.in_frac[frame, 0] * 100
        right = self.reel.in_frac[frame, 1] * 100
        return f"hand lag {self.lag} | in-frame L {left:.0f}% R {right:.0f}%"
