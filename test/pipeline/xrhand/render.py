"""把投影后的关键点画到视频帧上。

约定 (plan §3.2):
  - 左右两半各画一遍, 左半用 eye_of_half(0) 对应的外参, 右半用 eye_of_half(1)。
  - 不可见/超出画面的点用**不同样式**标出 —— 避免把"看不见"误读成"识别不准"。
  - 角上实时显示 head 角速度/线速度: head 快速转动时追踪误差会被放大,
    这些帧的偏差不应算到手部追踪头上 (plan §1.2b 的误差归属)。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import gripper as gr
from . import skeleton as sk
from .camera import EYE_H, EYE_W

# 画面上认为"在画面内"的余量 (超出则视为被裁掉)
MARGIN = 4

COLOR_BONE = (255, 255, 255)
COLOR_SHADOW = (0, 0, 0)
COLOR_OFFSCREEN = (255, 0, 0)
COLOR_INVALID = (128, 128, 128)


def _font(size: int):
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _disc(d: ImageDraw.ImageDraw, xy, r, fill, outline=COLOR_SHADOW, w=2):
    x, y = xy
    d.ellipse([x - r, y - r, x + r, y + r], fill=fill, outline=outline, width=w)


def draw_hand(
    d: ImageDraw.ImageDraw,
    u_eye: np.ndarray,
    v: np.ndarray,
    z: np.ndarray,
    valid: np.ndarray,
    half: int,
    radius: int = 5,
    alpha_fill: bool = True,
    img_for_poly: Optional[Image.Image] = None,
    *,
    joints=None,
    edges=None,
):
    """在指定半幅上画一只手。

    u_eye 是单眼局部坐标, 这里加上 half*EYE_W 变成整幅坐标。

    `joints` / `edges` 是**仅关键字**的取子集参数, 默认 None = 原来的全 26 点 +
    `sk.EDGES`。二值爪那边就是靠它**复用这个函数本身**来画那 5 个关键点 ——
    配色 (`sk.JOINT_COLOR`)、半径规则、画外红三角、无效点空心圈全都与原方法一致,
    不另写一份。掌心填充与 `inside`/`good` 仍按**全 26 点**算, 不受子集影响。
    """
    x0 = half * EYE_W
    U = u_eye + x0

    inside = (
        (z > 0)
        & (U >= x0 + MARGIN)
        & (U < x0 + EYE_W - MARGIN)
        & (v >= MARGIN)
        & (v < EYE_H - MARGIN)
    )
    good = valid & inside

    # 掌心半透明填充
    # 注意: 这里原先写成 `img_for_poly.alpha_composite(ov) if ... else None`,
    # 是把条件表达式当语句用 —— 无论条件真假都**不会执行** alpha_composite,
    # 所以掌心填充从来没真正合成过。改成正常的 if 语句。
    # 另外 overlay 只建掌心 bbox 那么大再贴过去: 建整幅 (2160x810 RGBA = 7 MB)
    # 在整片渲染时每帧要分配两次, 纯属浪费。
    if alpha_fill and img_for_poly is not None and img_for_poly.mode == "RGBA":
        if good[sk.PALM_POLY].all():
            poly = [(float(U[j]), float(v[j])) for j in sk.PALM_POLY]
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            x1 = int(max(x0, min(xs) - 2))
            x2 = int(min(x0 + EYE_W, max(xs) + 3))
            y1 = int(max(0, min(ys) - 2))
            y2 = int(min(EYE_H, max(ys) + 3))
            if x2 > x1 and y2 > y1:
                ov = Image.new("RGBA", (x2 - x1, y2 - y1), (0, 0, 0, 0))
                ImageDraw.Draw(ov).polygon(
                    [(px - x1, py - y1) for px, py in poly], fill=(255, 255, 255, 40)
                )
                img_for_poly.alpha_composite(ov, (x1, y1))

    # 骨骼
    for a, b in (sk.EDGES if edges is None else edges):
        if not (good[a] and good[b]):
            continue
        d.line(
            [(float(U[a]), float(v[a])), (float(U[b]), float(v[b]))],
            fill=COLOR_BONE,
            width=3,
        )

    # 关节
    for j in (range(sk.NUM_JOINTS) if joints is None else joints):
        if not valid[j]:
            if inside[j]:
                _disc(d, (float(U[j]), float(v[j])), 4, COLOR_INVALID, w=1)
            continue
        if not inside[j]:
            # 在画面外: 贴边画一个红三角提示, 不要让人以为点丢了
            cx = min(max(float(U[j]), x0 + 6), x0 + EYE_W - 7)
            cy = min(max(float(v[j]), 6), EYE_H - 7)
            d.polygon(
                [(cx, cy - 7), (cx - 6, cy + 5), (cx + 6, cy + 5)],
                fill=COLOR_OFFSCREEN,
                outline=COLOR_SHADOW,
            )
            continue
        r = 7 if j in sk.TIPS else (8 if j == sk.WRIST else 5)
        _disc(d, (float(U[j]), float(v[j])), r, sk.JOINT_COLOR[j])


def draw_hud(
    d: ImageDraw.ImageDraw,
    lines: list[str],
    half: int,
    font,
    y0: int = 8,
    color=(255, 255, 0),
):
    x0 = half * EYE_W + 10
    y = y0
    for ln in lines:
        # 先描黑边再写字, 保证在任何背景上都可读
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            d.text((x0 + dx, y + dy), ln, font=font, fill=(0, 0, 0))
        d.text((x0, y), ln, font=font, fill=color)
        y += 18


def draw_center_cross(d: ImageDraw.ImageDraw, half: int, color=(0, 255, 255)):
    """画单眼画面中心 (cx, cy 所在处), 用于人工核对标定偏移。"""
    x0 = half * EYE_W
    cx, cy = x0 + EYE_W / 2 - 0.5, EYE_H / 2 - 0.5
    d.line([(cx - 15, cy), (cx + 15, cy)], fill=color, width=1)
    d.line([(cx, cy - 15), (cx, cy + 15)], fill=color, width=1)
    d.rectangle([x0 + 0.5, 0.5, x0 + EYE_W - 0.5, EYE_H - 0.5], outline=color, width=1)


def gauge(frac: float, width: int = 100) -> str:
    """0..1 -> 文本条形图。"""
    n = int(np.clip(frac, 0, 1) * width)
    return "[" + "#" * n + "." * (width - n) + "]"


# ================================================================ 二值爪

# 5 个关键点**不在这里定颜色** —— 它们由 draw_hand 用 skeleton.JOINT_COLOR 画,
# 与 26 点渲染里的同名点同色同半径 (腕白 r8、指根/指尖按指色);
# HumanEgo 自己那套 (腕白/指根青/拇指尖红/食指尖绿) 里, 红绿与本仓库 FINGER_COLORS
# 恰好对上, 只有"指根用青色"这一条不同 —— 用户要求"完全照原样", 所以跟随本仓库。
# 这里只留夹爪独有的颜色。
COLOR_MID_LINE = (0, 255, 255)
# 状态色同上游 COLOR_GRASP (grasp.py:499 附近, 原为 BGR): 张开金, 闭合橙红
COLOR_GRASP_OPEN = (255, 215, 0)
COLOR_GRASP_CLOSED = (255, 69, 0)
# 坐标架 X 红 / Y 绿 / Z 蓝 (AriaHandsOps.py:1085 的 colors 列表, BGR -> RGB 后相同)
COLOR_AXIS = ((255, 0, 0), (0, 255, 0), (0, 0, 255))

PANEL_H = 178  # 判据波形面板高度; VideoWriter 建好尺寸不能改, 所以是常量
COLOR_PANEL_BG = (18, 18, 18)
COLOR_PANEL_GRID = (58, 58, 58)
COLOR_PANEL_CURVE = (0, 230, 120)
COLOR_PANEL_BAND = (52, 20, 10)


def _text(d: ImageDraw.ImageDraw, xy, s: str, font, fill, outline=COLOR_SHADOW):
    """描边文字 —— 与 draw_hud 一样先描黑边, 保证在任何背景上都读得出来。

    注意: 这里只写 ASCII。`_font()` 加载的 DejaVu / Liberation 都没有中文字形,
    写了中文会整行变成豆腐块 (现有 HUD 也全是英文)。
    """
    x, y = xy
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        d.text((x + dx, y + dy), s, font=font, fill=outline)
    d.text((x, y), s, font=font, fill=fill)


def _arrow_tip(d: ImageDraw.ImageDraw, p0, p1, color, size: float = 9.0):
    """在 p1 处朝 p0->p1 方向画一个小三角。

    cv2 的 arrowedLine 在 PIL 里没有对应物 (AriaHandsOps.py:1088 用的是它),
    用"线 + 端点小三角"代替, 视觉上等价。
    """
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    v = p1 - p0
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return
    u = v / n
    perp = np.asarray([-u[1], u[0]])
    base = p1 - size * u
    d.polygon(
        [tuple(p1), tuple(base + size * 0.45 * perp), tuple(base - size * 0.45 * perp)],
        fill=color,
        outline=COLOR_SHADOW,
    )


def draw_gripper(
    d: ImageDraw.ImageDraw,
    half: int,
    *,
    key_u: np.ndarray,
    key_v: np.ndarray,
    key_z: np.ndarray,
    key_valid: np.ndarray,
    mid_u: float,
    mid_v: float,
    mid_z: float,
    mid_valid: bool,
    axis_u: np.ndarray,
    axis_v: np.ndarray,
    axis_z: np.ndarray,
    axis_valid: bool,
    closed: bool,
    pinch_m: float,
    closure: float,
):
    """在指定半幅上画二值爪的**夹爪那一层**: 中点 + 连线 + 6DoF 坐标架 + 状态字。

    那 5 个关键点本身**不在这里画** —— 由 `draw_hand(..., joints=gripper.KEYPOINTS,
    edges=gripper.BONES)` 走原方法来画 (配色 `sk.JOINT_COLOR`、半径规则、画外红三角
    都与 26 点渲染里的同名点逐像素一致)。这里只画夹爪独有的东西, 所以仍然需要
    腕/两指尖的位置来连线和定位。

    入参全是**单眼局部坐标** (与 draw_hand 一致), 这里自己加 half*EYE_W。
    5 点的顺序 = `gripper.KEYPOINTS` = (腕, 拇指根, 食指根, 拇指尖, 食指尖);
    axis_* 是 (3,) 的 x/y/z 轴端点。
    """
    x0 = half * EYE_W
    U = np.asarray(key_u, dtype=np.float64) + x0
    v = np.asarray(key_v, dtype=np.float64)
    z = np.asarray(key_z, dtype=np.float64)
    valid = np.asarray(key_valid, dtype=bool)
    inside = (
        (z > 0)
        & (U >= x0 + MARGIN)
        & (U < x0 + EYE_W - MARGIN)
        & (v >= MARGIN)
        & (v < EYE_H - MARGIN)
    )
    good = valid & inside

    state_color = COLOR_GRASP_CLOSED if closed else COLOR_GRASP_OPEN

    # --- 夹爪在位时: 中点->腕 连线、两指尖虚线、坐标架 ---
    mid_px = (float(mid_u) + x0, float(mid_v))
    mid_ok = bool(
        mid_valid
        and mid_z > 0
        and x0 + MARGIN <= mid_px[0] < x0 + EYE_W - MARGIN
        and MARGIN <= mid_px[1] < EYE_H - MARGIN
    )
    if mid_ok:
        if good[gr.K_WRIST]:
            d.line(
                [mid_px, (float(U[gr.K_WRIST]), float(v[gr.K_WRIST]))],
                fill=COLOR_MID_LINE,
                width=1,
            )
        if good[gr.K_TTIP] and good[gr.K_ITIP]:
            # 两指尖之间用虚线连起来 —— 这条线的长度就是判据信号本身。
            # PIL 没有 dash 参数, 按段切着手画。
            p0 = np.asarray([float(U[gr.K_TTIP]), float(v[gr.K_TTIP])])
            p1 = np.asarray([float(U[gr.K_ITIP]), float(v[gr.K_ITIP])])
            length = float(np.linalg.norm(p1 - p0))
            if length > 1.0:
                step = 10.0
                unit = (p1 - p0) / length
                for s in np.arange(0.0, length, step):
                    e = min(s + step * 0.6, length)
                    d.line(
                        [tuple(p0 + unit * s), tuple(p0 + unit * e)],
                        fill=state_color,
                        width=2,
                    )
        # 坐标架 (轴长 x 0.06 / y 0.10 / z 0.06 m, 见 gripper.AXIS_LENGTHS)
        if axis_valid:
            for k in range(3):
                if not (axis_z[k] > 0):
                    continue
                p = (float(axis_u[k]) + x0, float(axis_v[k]))
                d.line([mid_px, p], fill=COLOR_AXIS[k], width=2)
                _arrow_tip(d, mid_px, p, COLOR_AXIS[k])
                # 轴名标在端点右下 (HumanEgo `_draw_axis` 的 (+8,+8)), 用带描边的
                # _text 保证压在亮/暗背景上都读得出; 颜色跟轴走 (X红/Y绿/Z蓝)
                _text(
                    d,
                    (p[0] + 8, p[1] + 8),
                    "XYZ"[k],
                    _font(13),
                    COLOR_AXIS[k],
                )
        _disc(d, mid_px, 6, state_color, w=1)
        d.line([(mid_px[0] - 9, mid_px[1]), (mid_px[0] + 9, mid_px[1])],
               fill=COLOR_SHADOW, width=1)
        _text(
            d,
            (mid_px[0] + 11, mid_px[1] - 26),
            f"{'CLOSED' if closed else 'OPEN'}  d={pinch_m:.3f}m  c={closure:.2f}",
            _font(13),
            state_color,
        )


def draw_grasp_panel(
    width: int,
    *,
    frame: int,
    distance: np.ndarray,
    closure: np.ndarray,
    closed: np.ndarray,
    calibration: dict,
    fps: float = 30.0,
    height: int = PANEL_H,
) -> Image.Image:
    """全片判据波形面板 (贴在画面下方), 版面移植 grasp.py:341-496 的 _draw_waveform_panel。

    上游那份是 cv2 画在 BGR 图上; 这里整条链是 RGB + PIL, 而且系统 anaconda python
    没有 cv2 (而 `tools/overlay.py` 就是用它跑的), 所以改用 PIL 原语重画:
    曲线/网格/阈值线/数字行/游标位置与含义都对应, 少的是 cv2 的箭头和抗锯齿。

    面板上只有 ASCII —— `_font()` 没有中文字形。
    """
    closed = np.asarray(closed, dtype=bool)
    closure = np.asarray(closure, dtype=np.float64)
    n = int(closure.size)
    if n == 0:
        return Image.new("RGB", (width, height), COLOR_PANEL_BG)
    frame = int(np.clip(frame, 0, n - 1))

    close_hi = float(calibration["close_hi"])
    open_lo = float(calibration["open_lo"])
    q95 = float(calibration["open_baseline_m"])
    q10 = float(calibration["closed_reference_m"])
    span = float(calibration["usable_range_m"])

    img = Image.new("RGB", (width, height), COLOR_PANEL_BG)
    d = ImageDraw.Draw(img)
    f_title, f_note = _font(15), _font(12)

    d.text(
        (12, 6),
        f"BINARY GRIPPER (5 keypoints) | pos = midpoint(thumb_tip, index_tip) "
        f"| episode grasp waveform | {n} frames @ {fps:g} fps",
        font=f_title,
        fill=(235, 235, 235),
    )
    d.text(
        (12, 28),
        f"closure = (q95 - d) / span  (thick line)   |   dashed: close_hi "
        f"{close_hi:.2f} -> {calibration['close_distance_m']:.4f} m , open_lo "
        f"{open_lo:.2f} -> {calibration['open_distance_m']:.4f} m   |   digital row = "
        f"ADAPT state   |   right axis = pinch distance (m)",
        font=f_note,
        fill=(185, 185, 185),
    )

    # ---- 版面 ----
    left, right = 78, max(width - 330, 200)
    top, bottom = 48, 128
    dig_y, dig_h = 138, 13

    def x_of(i: int) -> float:
        return left + (right - left) * (i / max(n - 1, 1))

    def y_of(value: float) -> float:
        return top + (1.0 - float(np.clip(value, 0.0, 1.0))) * (bottom - top)

    # 闭合区间底色 (先画, 免得盖住曲线)
    for start, end, value in gr.bool_runs(closed):
        if value:
            d.rectangle([x_of(start), top, x_of(max(end - 1, start)), bottom],
                        fill=COLOR_PANEL_BAND)

    # 网格 + 刻度
    for value in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = y_of(value)
        d.line([(left, y), (right, y)], fill=COLOR_PANEL_GRID, width=1)
        label = f"{value:.2f}"
        d.text((left - 8 - len(label) * 7, y - 7), label, font=f_note, fill=(150, 150, 150))
    for i in range(0, n, 100):
        x = x_of(i)
        d.line([(x, top), (x, bottom)], fill=(38, 38, 38), width=1)
        # 描边: 帧号会落在闭合区间的深色底上, 不描边就糊住了
        _text(d, (x + 3, bottom + 3), str(i), _font(11), (120, 120, 120))

    # 阈值线: close_hi 决定"何时判闭合", open_lo 决定"何时判张开"。PIL 没虚线,
    # 按段切开手画, 免得和实线曲线混淆。
    for value, color in ((close_hi, COLOR_GRASP_CLOSED), (open_lo, COLOR_GRASP_OPEN)):
        y = y_of(value)
        for x in np.arange(left, right, 14):
            d.line([(x, y), (min(x + 8, right), y)], fill=color, width=1)

    # 米制右轴: closure=0 对应 q95, closure=1 对应 q95-span
    for value in (0.0, 0.5, 1.0):
        d.text((right + 10, y_of(value) - 7), f"{q95 - value * span:.4f} m",
               font=f_note, fill=(150, 150, 150))

    # 主曲线
    points = [(float(x_of(i)), float(y_of(closure[i]))) for i in range(n)]
    d.line(points, fill=COLOR_PANEL_CURVE, width=2, joint="curve")

    # 数字状态行
    d.rectangle([left, dig_y, right, dig_y + dig_h], outline=(80, 80, 80))
    for start, end, value in gr.bool_runs(closed):
        if value:
            d.rectangle([x_of(start), dig_y + 1, x_of(max(end - 1, start)), dig_y + dig_h - 1],
                        fill=COLOR_GRASP_CLOSED)
    d.text((left - 8 - 5 * 7, dig_y), "ADAPT", font=f_note, fill=(200, 200, 200))

    # 游标
    cx = x_of(frame)
    d.line([(cx, top), (cx, dig_y + dig_h)], fill=(255, 255, 255), width=1)
    _disc(d, (cx, y_of(closure[frame])), 4,
          COLOR_GRASP_CLOSED if closed[frame] else COLOR_GRASP_OPEN, w=1)

    d.text(
        (12, 156),
        f"q95(open) {q95:.4f} m   q10(closed) {q10:.4f} m   span {span:.4f} m "
        f"(raw range {calibration['dynamic_range_m']:.4f} m, min_range "
        f"{calibration['min_range_m']:.3f} m)   median {calibration['median_window']} "
        f"confirm {calibration['confirm_ticks']} min_state {calibration['min_state_ticks']} "
        f"| frame {frame}  d={distance[frame]:.4f} m  closure={closure[frame]:.3f}"
        + ("  [range too flat -> always OPEN]" if calibration["dynamic_range_too_flat"] else ""),
        font=f_note,
        fill=(185, 185, 185),
    )
    return img
