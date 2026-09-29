"""可视化: 全幅 mask 叠加视频 / 静帧 / 质量曲线 / 提示帧核验图。

版面是从复制的 `sam2_video.py::_render_video` 搬过来的 (52 px 表头 + 画面 +
28+25N px 表尾, 24 号灰底, mask 填充 `addWeighted(0.66, 0.34)` + `drawContours`),
这里多两层:

  - mask 只叠在**左半 eye0** 上 (只有 eye0 跑了分割), 右半保持原图;
  - 两半都叠右手骨架 (`xrseg.hand_overlay`), 与 out/overlay_*.mp4 的画法一致。

编码走 `xrhand.video.VideoWriter` (libx264 + yuv420p + faststart), 与仓库里已有的
`out/overlay_*.mp4` 同一套, 不用上游那份 mp4v —— mp4v 体积大且兼容性差。
帧解码同样走 `xrhand.video.iter_frames` (整幅 2160x810 RGB)。

上游的 `_render_metrics_chart` 依赖 CoTracker 的可见性曲线, 本仓库没有 CoTracker,
所以质量曲线换成: 面积比 (蓝) + 每帧 score (红) + mask∩右手凸包占比 (橙),
空 mask 段用 axvspan 标出来。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from xrhand.video import VideoWriter, iter_frames

from xrseg.common import (
    EYE_H,
    EYE_W,
    FULL_H,
    FULL_W,
    INSTANCE_ID,
    SegPaths,
    instance_color,
)

HEADER_H = 52
FOOTER_H = 28 + 25 * 2  # 默认行数 (2): 只有一个物体时就是原来那个高度
PANEL_H = 360  # 静帧/核验图里放大面板的高度, 同 tools/overlay.py
BARS_BG = 24
TEXT_LIGHT = (240, 240, 240)
TEXT_DIM = (185, 185, 185)


# ------------------------------------------------------------------ 单帧绘制


def load_object_masks(paths: SegPaths, frame: int, instance_ids: list[str]) -> dict[str, np.ndarray]:
    """读这一帧所有物体的 mask (`{instance_id: uint8 HxW}`), 缺的当全零。"""
    out: dict[str, np.ndarray] = {}
    for instance_id in instance_ids:
        path = paths.mask(frame, instance_id)
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path.is_file() else None
        if mask is None:
            raise FileNotFoundError(path)
        out[instance_id] = mask
    return out


def _mask_overlay_bgr(frame_rgb: np.ndarray,
                      masks: dict[str, np.ndarray] | np.ndarray) -> np.ndarray:
    """把各物体的 mask 按各自配色叠到左半, 返回 RGB (输入输出都是 RGB, 中间借 BGR 走 cv2)。

    也收单个 `ndarray` (= 只有 obj1 的那份 mask) —— `xrrel/render.py` 是"原样调用"
    这个函数的, 单物体那条链路因此一个字都不用改。
    """
    if isinstance(masks, np.ndarray):
        masks = {INSTANCE_ID: masks}
    bgr = np.ascontiguousarray(frame_rgb[:, :, ::-1]).copy()
    left = bgr[:, :EYE_W]
    for instance_id, mask in masks.items():
        color = instance_color(instance_id)
        overlay = left.copy()
        overlay[mask > 127] = color
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(left, contours, -1, color, 2, cv2.LINE_AA)
        left[:] = cv2.addWeighted(left, 0.66, overlay, 0.34, 0.0)
    return np.ascontiguousarray(bgr[:, :, ::-1])


def _union_mask(masks: dict[str, np.ndarray]) -> np.ndarray:
    """所有物体的并集 —— 放大面板/静帧按它取外接框。"""
    if not masks:
        raise ValueError("没有物体 mask")
    union = np.zeros_like(next(iter(masks.values())))
    for mask in masks.values():
        union = np.maximum(union, mask)
    return union


def _draw_frame(
    frame_rgb: np.ndarray,
    masks: dict[str, np.ndarray],
    hand,
    frame: int,
    *,
    radius: int = 4,
) -> np.ndarray:
    rgb = _mask_overlay_bgr(frame_rgb, masks)
    if hand is not None:
        rgb = np.asarray(hand.draw(rgb, frame, radius=radius).convert("RGB"))
    return rgb


def _bars(rgb: np.ndarray, header: str, lines: list[str],
          *, footer_h: int | None = None) -> np.ndarray:
    """表头 + 画面 + 表尾, 与复制版 _render_video 的版面一致。"""
    if footer_h is None:
        footer_h = max(FOOTER_H, 28 + 25 * len(lines))
    header_bar = np.full((HEADER_H, rgb.shape[1], 3), BARS_BG, dtype=np.uint8)
    cv2.putText(header_bar, header, (14, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                TEXT_LIGHT, 1, cv2.LINE_AA)
    footer_bar = np.full((footer_h, rgb.shape[1], 3), BARS_BG, dtype=np.uint8)
    for index, line in enumerate(lines):
        cv2.putText(footer_bar, line, (14, 24 + index * 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, TEXT_DIM, 1, cv2.LINE_AA)
    return np.vstack((header_bar, rgb, footer_bar))


def _zoom_panel(bgr: np.ndarray, mask: np.ndarray, pad: int = 70) -> np.ndarray | None:
    """按 mask 外接框从**左半**裁一块放大 —— 1080 宽里差几十像素肉眼看不出来。"""
    ys, xs = np.nonzero(mask > 127)
    if ys.size == 0:
        return None
    x1 = int(max(0, xs.min() - pad))
    x2 = int(min(EYE_W, xs.max() + pad + 1))
    y1 = int(max(0, ys.min() - pad))
    y2 = int(min(EYE_H, ys.max() + pad + 1))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    crop = bgr[y1:y2, x1:x2]
    scale = PANEL_H / crop.shape[0]
    return cv2.resize(
        crop, (max(1, int(round(crop.shape[1] * scale))), PANEL_H),
        interpolation=cv2.INTER_LANCZOS4,
    )


def _with_inset(rgb: np.ndarray, mask: np.ndarray, caption: str = "") -> np.ndarray:
    """画面 + 下方放大面板 + 一行说明 (照 tools/overlay.py 的画布布局)。

    画布跟随**输入自身的尺寸**, 不再假定是整幅双目帧: `render_stills` 喂进来的是
    2160x810 (整幅, 左半叠 mask), `render_prompt_frame` 喂进来的是 1080x810 (只有
    eye0 —— 帧缓存本来就是切了左半的), 两种都得能画。
    """
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    height, width = bgr.shape[:2]
    panel = _zoom_panel(bgr, mask)
    if panel is None:
        return rgb
    cap_h = 28 if caption else 0
    canvas = np.full((height + 8 + PANEL_H + cap_h, width, 3), 20, dtype=np.uint8)
    canvas[:height] = bgr
    panel_w = min(panel.shape[1], width - 16)
    canvas[height + 4:height + 4 + PANEL_H, 8:8 + panel_w] = panel[:, :panel_w]
    if caption:
        cv2.putText(canvas, caption, (14, height + 8 + PANEL_H + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
    return canvas[:, :, ::-1]


# ------------------------------------------------------------------ 视频


def render_overlay_video(
    paths: SegPaths,
    *,
    prompt: str,
    fps: float,
    hand=None,
    scores: np.ndarray | None = None,
    instance_ids: list[str] | None = None,
    frame_count: int | None = None,
    prompt_frame: int = 0,
    radius: int = 4,
    crf: int = 18,
    progress_every: int = 50,
) -> Path:
    """主产物: 2160x810 + 上下字幕条。左半 eye0 逐物体叠 mask, 两半叠右手骨架。

    `scores` 是 `(帧数, 物体数)`; 只给一维时当成单物体。字幕每个物体一行 —— 行数决定
    表尾高度, 所以只有一个物体时尺寸与旧版逐像素相同。
    """
    if instance_ids is None:
        instance_ids = paths.instance_ids() or [INSTANCE_ID]
    instance_ids = [str(i) for i in instance_ids]
    object_count = len(instance_ids)
    if frame_count is None:
        frame_count = len(sorted(paths.instance_mask_dir(instance_ids[0]).glob("*.png")))
    if frame_count <= 0:
        raise RuntimeError(f"没有 mask 可渲染: {paths.instance_mask_dir(instance_ids[0])}")

    score_table = None if scores is None else np.asarray(scores, dtype=np.float64)
    if score_table is not None and score_table.ndim == 1:
        score_table = score_table[:, None]

    # 表尾行数 = 物体数 + 1 (手部一行): 单物体时尺寸与旧版逐像素相同。
    footer_h = 28 + 25 * (object_count + 1)
    height = HEADER_H + EYE_H + footer_h
    paths.overlay_video.parent.mkdir(parents=True, exist_ok=True)
    # area/initial 的基准取**第一个非空帧**(通常就是提示帧): 提示帧选在 N>0 时,
    # 前 N 帧是硬写的空帧, 拿第 0 帧当分母会得到一堆几万倍的比值, 白占满字幕。
    # 逐物体各算各的基准。
    initial_area = np.ones(object_count, dtype=np.int64)
    initial_frame = np.zeros(object_count, dtype=np.int64)
    done = np.zeros(object_count, dtype=bool)
    for probe in range(frame_count):
        for j, instance_id in enumerate(instance_ids):
            if done[j]:
                continue
            probe_mask = cv2.imread(str(paths.mask(probe, instance_id)), cv2.IMREAD_GRAYSCALE)
            count = int(np.count_nonzero(probe_mask > 127)) if probe_mask is not None else 0
            if count:
                initial_area[j], initial_frame[j], done[j] = count, probe, True
        if done.all():
            break
    print(f"║ 渲染 {frame_count} 帧 -> {paths.overlay_video} "
          f"({FULL_W}x{height} @ {fps:g} fps)  物体 {instance_ids}  "
          f"area 基准 = " + ", ".join(
              f"{i}@f{int(initial_frame[j])}({int(initial_area[j])}px)"
              for j, i in enumerate(instance_ids)
          ))
    writer = VideoWriter(str(paths.overlay_video), FULL_W, height, fps, crf=crf)
    try:
        for frame, frame_rgb in enumerate(iter_frames(str(paths.mp4), FULL_W, FULL_H)):
            if frame >= frame_count:
                break
            masks = load_object_masks(paths, frame, instance_ids)
            rgb = _draw_frame(frame_rgb, masks, hand, frame, radius=radius)
            lines = []
            for j, instance_id in enumerate(instance_ids):
                area = int(np.count_nonzero(masks[instance_id] > 127))
                score = float(score_table[frame, j]) if score_table is not None else float("nan")
                ratio = f"{area / initial_area[j]:5.2f}" if area else "   --"
                score_text = f"{score:.3f}" if np.isfinite(score) else "  --"
                note = "  (提示帧之前, 目标未出现)" if frame < prompt_frame else ""
                lines.append(
                    f"{instance_id} {prompt} | mask {area:6d}px | "
                    f"area/initial {ratio} | score {score_text}{note}"
                )
            lines.append(hand.summary(frame) if hand is not None else "hand overlay: off")
            composite = _bars(
                rgb,
                f"SAM2 Video masks | frame {frame:03d}/{frame_count - 1:03d} | "
                f"{frame / fps:05.2f}s | prompt f{prompt_frame}",
                lines,
                footer_h=footer_h,
            )
            writer.write(composite)
            if progress_every and frame and frame % progress_every == 0:
                print(f"║   frame {frame:4d}/{frame_count - 1}")
    finally:
        writer.close()
    print(f"║ -> {paths.overlay_video} ({paths.overlay_video.stat().st_size / 1e6:.1f} MB)")
    return paths.overlay_video


# ------------------------------------------------------------------ 静帧


def auto_still_frames(areas: np.ndarray, prompt_frame: int, count: int = 6) -> list[int]:
    """抽几帧给人工判读: 首/末个非空帧、面积最大/最小、再在非空区间里均匀铺几帧。

    提示帧 > 0 时它前面的帧是硬写的空帧 (目标还没出现), 拿它们占名额没有信息量,
    所以均匀铺帧的范围从**第一个非空帧**起算。
    """
    areas = np.asarray(areas, dtype=np.float64).reshape(-1)
    total = areas.size
    start = min(max(int(prompt_frame), 0), total - 1)
    nonempty = np.flatnonzero(areas > 0)
    if nonempty.size:
        start = min(start, int(nonempty[0]))
    picks = {int(v) for v in np.linspace(start, total - 1, count).round().astype(int)}
    if nonempty.size:
        picks |= {
            int(nonempty[0]),  # 目标刚出现
            int(nonempty[-1]),  # 最后一次框住
            int(nonempty[np.argmax(areas[nonempty])]),
            int(nonempty[np.argmin(areas[nonempty])]),
        }
    return sorted(picks)


def render_stills(
    paths: SegPaths,
    *,
    frames: list[int],
    prompt: str,
    fps: float,
    hand=None,
    scores: np.ndarray | None = None,
    instance_ids: list[str] | None = None,
    radius: int = 4,
) -> list[Path]:
    if instance_ids is None:
        instance_ids = paths.instance_ids() or [INSTANCE_ID]
    instance_ids = [str(i) for i in instance_ids]
    wanted = set(int(f) for f in frames)
    if not wanted:
        return []
    top = max(wanted)
    written: list[Path] = []
    for frame, frame_rgb in enumerate(iter_frames(str(paths.mp4), FULL_W, FULL_H)):
        if frame > top:
            break
        if frame not in wanted:
            continue
        masks = load_object_masks(paths, frame, instance_ids)
        rgb = _draw_frame(frame_rgb, masks, hand, frame, radius=radius)
        areas = {i: int(np.count_nonzero(m > 127)) for i, m in masks.items()}
        caption = (
            f"frame {frame}  {frame / fps:.2f}s  "
            + "  ".join(f"{i} {areas[i]}px" for i in instance_ids)
            + f"  prompt {prompt!r}"
        )
        out = _with_inset(rgb, _union_mask(masks), caption)
        path = paths.still(frame)
        if not cv2.imwrite(str(path), np.ascontiguousarray(out[:, :, ::-1])):
            raise RuntimeError(f"无法写入静帧: {path}")
        written.append(path)
        print(f"║ -> {path}")
    return written


# ------------------------------------------------------------------ 质量曲线


def render_quality_chart(
    paths: SegPaths,
    *,
    prompt: str,
    areas: np.ndarray,
    scores: np.ndarray,
    hand_overlap: np.ndarray | None = None,
    instance_ids: list[str] | None = None,
    fps: float = 30.0,
    prompt_frame: int = 0,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from xrseg.sam2_video import _true_runs

    areas = np.asarray(areas, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if areas.ndim == 1:
        areas = areas[:, None]
    if scores.ndim == 1:
        scores = scores[:, None]
    if instance_ids is None:
        instance_ids = [INSTANCE_ID] * areas.shape[1]
    instance_ids = [str(i) for i in instance_ids]
    frames = np.arange(areas.shape[0])

    figure, axis = plt.subplots(figsize=(14, 4.2))
    colors = ("#1976d2", "#d32f2f", "#ef6c00", "#6a1b9a")
    curves = []
    for j, instance_id in enumerate(instance_ids):
        # 归一化基准取**第一个非空帧**: 提示帧 > 0 时第 0 帧是空帧, 拿它当分母 (上游的
        # areas[0]) 会让整条曲线变成几万倍的直线。逐物体各算各的基准。
        nonempty = np.flatnonzero(areas[:, j] > 0)
        baseline = float(areas[nonempty[0], j]) if nonempty.size else 1.0
        baseline_frame = int(nonempty[0]) if nonempty.size else 0
        curve = areas[:, j] / baseline
        curves.append(curve)
        color = colors[j % len(colors)]
        axis.plot(frames, curve, color=color, linewidth=1.2,
                  label=f"{instance_id} area / area @ frame {baseline_frame}")
        axis.plot(frames, scores[:, j], color=color, linewidth=0.8, linestyle=":",
                  label=f"{instance_id} mean mask score")
    if hand_overlap is not None and np.isfinite(hand_overlap).any():
        axis.plot(frames, hand_overlap, color="#455a64", linewidth=1.0, linestyle="--",
                  label=f"mask ∩ right-hand hull / mask ({instance_ids[0]})")
    for start, end in _true_runs(areas.max(axis=1) == 0):
        axis.axvspan(start, end, color="#b71c1c", alpha=0.15)
    axis.axvline(prompt_frame, color="#2e7d32", linestyle=":", linewidth=1.0,
                 label=f"prompt frame {prompt_frame}")
    axis.axhline(0.1, color="#555555", linestyle="--", linewidth=0.8, alpha=0.7)
    axis.set_xlabel("camera frame (eye0)")
    axis.set_ylabel("normalized signal")
    finite = np.concatenate([
        np.concatenate([c[np.isfinite(c)] for c in curves]),
        scores.ravel()[np.isfinite(scores.ravel())],
        np.asarray([1.15]),
    ])
    axis.set_ylim(-0.02, min(float(np.max(finite)) * 1.1, 6.0))
    axis.grid(alpha=0.2)
    axis.legend(loc="upper right", fontsize=7, ncol=2)
    axis.set_title(
        f"{paths.stem} | {', '.join(f'{i} {prompt}' for i in instance_ids)} | "
        f"prompt f{prompt_frame} | red bands = 全部物体都空的帧 | "
        f"{areas.shape[0]} frames @ {fps:g} fps"
    )
    figure.tight_layout()
    paths.quality_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(paths.quality_png, dpi=160)
    plt.close(figure)
    print(f"║ -> {paths.quality_png}")
    return paths.quality_png


# ------------------------------------------------------------------ 提示帧核验图


def render_prompt_frame(
    paths: SegPaths,
    *,
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    boxes: np.ndarray,
    confidences: np.ndarray,
    prompt: str,
    frame: int,
    box_threshold: float,
    avg_confidence: float,
    latency_s: float,
    instance_id: str = INSTANCE_ID,
) -> Path:
    """DINO 框 + 每框 score + 归一化提示词 + 最终 mask —— 跑全片之前先看这张。"""
    bgr = np.ascontiguousarray(frame_rgb[:, :, ::-1]).copy()
    left = bgr[:, :EYE_W]
    support = mask > 127
    overlay = left.copy()
    overlay[support] = (230, 200, 60)  # BGR 青蓝, 与绿色框区分开
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(left, contours, -1, (230, 200, 60), 2, cv2.LINE_AA)
    bgr[:, :EYE_W] = cv2.addWeighted(left, 0.7, overlay, 0.3, 0.0)

    for box, confidence in zip(boxes, confidences):
        x1, y1, x2, y2 = (int(round(float(v))) for v in box)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (60, 200, 60), 2)
        cv2.putText(bgr, f"{confidence:.3f}", (x1, max(y1 - 7, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 200, 60), 1, cv2.LINE_AA)

    area = int(np.count_nonzero(support))
    lines = [
        f"DINO+SAM2 prompt frame | {paths.stem} | frame {frame} | {instance_id}",
        f"prompt {prompt!r}   box_threshold={box_threshold}   boxes={len(boxes)}   "
        f"avg conf {avg_confidence:.3f}",
        f"mask {area} px   latency {latency_s * 1000:.0f} ms   "
        f"-> 确认框住的是目标物体、没吃掉手指, 再跑 --stage video",
    ]
    for index, line in enumerate(lines):
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            cv2.putText(bgr, line, (15 + dx, 26 + index * 26 + dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(bgr, line, (15, 26 + index * 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (0, 255, 255), 1, cv2.LINE_AA)

    out = _with_inset(bgr[:, :, ::-1], mask, f"zoom: DINO box -> SAM2 mask ({instance_id})")
    destination = paths.prompt_frame_for(instance_id)
    if not cv2.imwrite(str(destination), np.ascontiguousarray(out[:, :, ::-1])):
        raise RuntimeError(f"无法写入 {destination}")
    print(f"║ -> {destination}")
    return destination
