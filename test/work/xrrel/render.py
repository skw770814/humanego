"""可视化: 三层同框 + 逐帧实时相对位姿。

用户对可视化的要求是原话: 「可视化要求有 step1 5 关键点和 step2 物体分割, 并实时打印相对位姿」。
所以画面从上到下是**三层叠出来的**, 每层都用现成实现, 不重画:

    ┌────────────────────── 2160 x 810 双目原图 ──────────────────────┐
    │  左半 eye0:  step2 mask 叠色 + 轮廓                              │
    │              + step3 物体点云 / 物体系坐标架 / 中点到物体的连线   │
    │              + step1 的 5 个关键点 + 夹爪 6DoF + CLOSED/OPEN     │
    │  右半 eye1:  step1 的 5 个关键点 (mask 只在 eye0, 见 xrseg)      │
    ├────────────────── 178 px 抓取判据波形面板 (step1 原物) ──────────┤
    ├────────────────── 168 px 相对位姿面板 (本文件新写) ──────────────┤
    └──────────────────────────────────────────────────────────────────┘
                        = 2160 x 1156

  1. step2 分割: `xrseg/render.py::_mask_overlay_bgr(frame_rgb, mask)` **原样调用**
     (mask 填充 addWeighted(0.66,0.34) + drawContours, 只叠左半) —— 与
     `out/seg_<stem>/seg_overlay_*.mp4` 是同一条代码路径。
  2. step1 关键点: `tools/overlay.py::draw_frame(reel, rgb, f, radius, inset=False,
     mode="gripper")` **原样调用** —— 5 个关键点走 `draw_hand` 本体 + 夹爪那层, 并且
     它返回的就是 2160x988 (含抓取面板), 所以下面只需要再贴我们那块 168 的面板。
  3. step3: 本文件。物体点云 (相机0 反投影的逆投回去, `lift.project_camera0`)、
     物体系的 XYZ 坐标架 (红/绿/蓝, 与 step1 的夹爪坐标架同一套 `render.COLOR_AXIS`)、
     中点->物体质心的连线与米数、每半幅右下角一块紧凑文字块、以及底部那块相对位姿面板
     (左半数值 + 右半三条曲线与游标)。

「实时」是两处都做到: **画面上**逐帧刷新 (文字块 + 面板游标) 且**终端里**逐帧打印同一组数
(`--print-frames`, 默认开), 打印行与 npz 同帧一致。
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from xrhand import render as R
from xrhand.video import VideoWriter, iter_frames

from . import EYE_F, EYE_H, EYE_W, FULL_H, FULL_W, RelPaths
from .lift import load_clouds, load_mask, project_camera0

REL_PANEL_H = 168  # 相对位姿面板高度; 与 R.PANEL_H(178) 相加得最终画布高
VIDEO_H = FULL_H + R.PANEL_H + REL_PANEL_H  # 810 + 178 + 168 = 1156

COLOR_CLOUD = (90, 220, 255)  # 物体点云 (浅橙)
COLOR_LINK = (255, 255, 255)  # 中点 -> 物体质心 连线
COLOR_OBJECT_ORIGIN = (255, 215, 0)
COLOR_REL_BG = (14, 20, 26)
COLOR_REL_GRID = (56, 66, 74)
COLOR_NO_OBS = (34, 34, 38)  # 无观测帧阴影
COLOR_INVALID = (44, 34, 24)  # 有观测但位姿无效 (门控丢帧) 阴影
COLOR_LATCH_BAND = (52, 20, 10)  # 手推帧底色 (本帧没有测量, 位姿由手推出)
COLOR_TEXT = (238, 238, 238)
COLOR_TEXT_DIM = (176, 176, 176)

AXIS_LEN_M = 0.05  # 物体系坐标架轴长 (米) —— 物体约 6~8 cm, 这个长度看得出朝向


# ---------------------------------------------------------------- 相机0 -> 像素


def _warn_if_calibrated_f(reel) -> str | None:
    """点云用标称 K 反投影, 画回去也必须用标称 K。标定增量非零时显式说明。"""
    eff = float(reel.proj.p.eff_f)
    if abs(eff - EYE_F) > 1e-9:
        return (
            f"标定把焦距改成了 eff_f={eff:.4f} (标称 {EYE_F}); 物体点云仍按标称 K 投影 "
            f"(点云本身就是标称 K 反投影出来的), 因此点云与叠加层的像素可能差一点点"
        )
    return None


def _project_camera0(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    u, v = project_camera0(points)
    return u, v, points[:, 2]


def _inside_half(u: np.ndarray, v: np.ndarray, z: np.ndarray, half: int, margin: float = 2.0):
    x0 = half * EYE_W
    return (
        (z > 0)
        & (u >= x0 + margin)
        & (u < x0 + EYE_W - margin)
        & (v >= margin)
        & (v < EYE_H - margin)
    )


# ---------------------------------------------------------------- step3 那一层


def draw_relative(
    image: Image.Image,
    *,
    frame: int,
    mask_cloud: np.ndarray,
    T_camera0_object: np.ndarray,
    T_camera0_midpoint: np.ndarray,
    T_object_midpoint: np.ndarray | None,
    fields: dict,
) -> Image.Image:
    """在整幅 2160x810 上叠 step3 层 (物体点云 / 物体系 / 连线 / 两块文字)。

    全部几何都在**相机0 (左眼)** 系, 所以只画在左半; 右半只放文字块 (右半是 eye1,
    我们既没有它的分割也没有它的深度)。
    """
    img = image.convert("RGB")
    d = ImageDraw.Draw(img)
    font = R._font(15)
    font_small = R._font(13)

    # --- 物体点云 (相机0 -> 左半像素) ---
    if mask_cloud.shape[0]:
        u, v, z = _project_camera0(mask_cloud)
        keep = _inside_half(u, v, z, 0)
        if keep.any():
            d.point([(float(x), float(y)) for x, y in zip(u[keep], v[keep])], fill=COLOR_CLOUD)

    # --- 物体系坐标架 + 原点 ---
    origin_cam = np.asarray(T_camera0_object, dtype=np.float64)[:3, 3]
    rotation = np.asarray(T_camera0_object, dtype=np.float64)[:3, :3]
    axis_cam = np.stack([origin_cam + AXIS_LEN_M * rotation[:, k] for k in range(3)])
    u0, v0, z0 = _project_camera0(origin_cam[None])
    ua, va, za = _project_camera0(axis_cam)
    if _inside_half(u0, v0, z0, 0)[0]:
        base = (float(u0[0]), float(v0[0]))
        for k in range(3):
            if not _inside_half(ua[k : k + 1], va[k : k + 1], za[k : k + 1], 0)[0]:
                continue
            tip = (float(ua[k]), float(va[k]))
            d.line([base, tip], fill=R.COLOR_AXIS[k], width=2)
            R._arrow_tip(d, base, tip, R.COLOR_AXIS[k])
            R._text(d, (tip[0] + 6, tip[1] + 4), "o" + "XYZ"[k], font_small, R.COLOR_AXIS[k])
        R._disc(d, base, 6, COLOR_OBJECT_ORIGIN, w=1)

    # --- 中点 -> 物体原点 连线 (长度就是相对距离) ---
    mid_cam = np.asarray(T_camera0_midpoint, dtype=np.float64)[:3, 3]
    um, vm, zm = _project_camera0(mid_cam[None])
    if _inside_half(um, vm, zm, 0)[0] and _inside_half(u0, v0, z0, 0)[0]:
        mid_px = (float(um[0]), float(vm[0]))
        origin_px = (float(u0[0]), float(v0[0]))
        d.line([mid_px, origin_px], fill=COLOR_LINK, width=1)
        if T_object_midpoint is not None:
            length = float(np.linalg.norm(T_object_midpoint[:3, 3]))
            centre = ((mid_px[0] + origin_px[0]) / 2 + 6, (mid_px[1] + origin_px[1]) / 2 - 18)
            R._text(d, centre, f"{length:.3f} m", font_small, COLOR_LINK)

    # --- 每半幅右下角的紧凑文字块 (两半都放, 便于全屏观看) ---
    # 按**实测行宽**右对齐到本半幅右边缘 (留 12 px), 不能用固定偏移: 状态那行实测 605 px
    # (位姿行 424), 原来按 430 留白 -> 左半那份越过接缝 175 px 压到 eye1 画面上 step1 的
    # 手部状态文字上, 右半那份被画面右边缘切掉 175 px。
    lines = _text_block(fields)
    block_w = max(d.textlength(text, font=font) for text, _ in lines)
    # 最后两行 (相对位姿 + 状态) 的位置与改动前逐像素相同: 多出来的物体行往上排。
    top = EYE_H - 52 - 20 * (len(lines) - 2)
    for half in range(2):
        x = half * EYE_W + EYE_W - 12 - block_w
        for index, (text, color) in enumerate(lines):
            R._text(d, (x, top + index * 20), text, font, color)

    return img


def _text_block(fields: dict) -> list[tuple[str, tuple[int, int, int]]]:
    """紧凑文字: 其余物体各一行 (多物体时) + 相对位姿 + 状态。字全是 ASCII
    (`R._text` 的字体没有中文)。N=1 时就是原来的两行, 位置也不变。"""
    extra = [
        (
            f"{instance_id}  |t|={distance:.4f} m" if valid else f"{instance_id}  n/a",
            COLOR_TEXT if valid else COLOR_TEXT_DIM,
        )
        for instance_id, distance, valid in fields.get("others", [])
    ]
    if fields["T_object_midpoint"] is None:
        return extra + [
            ("REL  n/a  (pose invalid this frame)", COLOR_TEXT_DIM),
            (fields["status"], COLOR_TEXT_DIM),
        ]
    t = fields["T_object_midpoint"][:3, 3]
    rpy = fields["rpy"]
    state_color = (
        # 颜色跟**夹爪状态**走, 不跟 latched 走: 锁存要到"闭合 + 有物体位姿可就地锚定 +
        # 距手 < latch_distance_m"三条都满足才成立 (例如 f243-273 握住了但距手 252~270 mm,
        # 锁存被拒), 拿 latched 上色会把那一段握持画成"张开"。
        R.COLOR_GRASP_CLOSED
        if fields["closed"]
        else (R.COLOR_GRASP_OPEN if fields["observed"] else COLOR_TEXT_DIM)
    )
    return extra + [
        (f"REL t=[{t[0]:+.4f} {t[1]:+.4f} {t[2]:+.4f}] m  |t|={fields['distance']:.4f} m",
         COLOR_TEXT),
        (f"RPY=[{rpy[0]:+.2f} {rpy[1]:+.2f} {rpy[2]:+.2f}] deg   {fields['status']}",
         state_color),
    ]


# ---------------------------------------------------------------- 相对位姿面板


def draw_rel_panel(
    width: int,
    *,
    frame: int,
    relative: np.ndarray,
    valid: np.ndarray,
    latched: np.ndarray,
    observed: np.ndarray,
    distance: np.ndarray,
    height: int = REL_PANEL_H,
    source: str = "",
) -> Image.Image:
    """底部相对位姿面板: 左半逐帧数值, 右半 x/y/z 三条曲线 + 游标 + 阴影区间。

    阴影不是装饰, 是如实标注:
      - **无观测帧** (`metrics.npz` 的 `areas==0`, 见 plan 风险 3): 灰底;
      - **有观测但位姿被门控丢帧**: 棕底;
      - **手推帧** (latched: 握持期间物体位姿一律由 「手 x T_hand_object」 给出, 不跑拟合):
        橙红底色带。锁存期间这三条相对平移曲线是**平**的。 (照 HumanEgo-main 的锁存,
        见 objectpose 的循环顶部)
    """
    relative = np.asarray(relative, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    latched = np.asarray(latched, dtype=bool)
    observed = np.asarray(observed, dtype=bool)
    n = len(valid)
    img = Image.new("RGB", (width, height), COLOR_REL_BG)
    d = ImageDraw.Draw(img)
    if n == 0:
        return img
    frame = int(np.clip(frame, 0, n - 1))
    font_title, font_note = R._font(15), R._font(12)

    d.text(
        (12, 6),
        "STEP3 RELATIVE POSE   T_object_right_midpoint = inv(T_camera0_object) @ "
        "T_camera0_right_midpoint   (right EEF = thumb/index tip midpoint, HumanEgo frame)   "
        "|   camera0 = left eye, same pixel frame as the SAM2 mask and the stereo depth",
        font=font_title,
        fill=(232, 232, 232),
    )
    d.text(
        (12, 28),
        "shaded: grey = no object observation (SAM2 areas==0) , brown = observed but pose "
        "rejected/invalid ; band: latched (pose carried by the hand, no measurement that frame)   |   "
        "curves = relative translation x/y/z (m)   |   right axis = metres"
        + (f"   |   {source}" if source else ""),
        font=font_note,
        fill=COLOR_TEXT_DIM,
    )

    left, right = 92, max(width - 320, 260)
    top, bottom = 50, 130

    def x_of(index: int) -> float:
        return left + (right - left) * (index / max(n - 1, 1))

    curves = relative[:, :3, 3]  # (n,3) 相对平移
    # 有效性是**逐帧**的 (`valid` 一整帧为真/假), 掩码压成一维 (n,)。
    # 原来写成 `np.isfinite(curves) & valid[:, None]` 得到 (n,3), 下面 `bool(finite[index])`
    # (278/288/300/305) 必抛 ValueError —— render 第一次真正跑起来时崩的就是这一行。
    finite = np.isfinite(curves).all(axis=1) & valid
    magnitude = (
        float(np.percentile(np.abs(curves[finite]), 98)) if finite.any() else 0.0
    )
    limit = max(magnitude * 1.15, 0.01)

    def y_of(value: float) -> float:
        """+limit 在顶, 0 在中间, -limit 在底。"""
        unit = 0.5 * (float(np.clip(value / limit, -1.0, 1.0)) + 1.0)
        return bottom - unit * (bottom - top)

    # --- 阴影: 先画, 免得盖住曲线 ---
    _shade(d, np.nonzero(~observed)[0], x_of, top, bottom, COLOR_NO_OBS)
    _shade(d, np.nonzero(observed & ~valid)[0], x_of, top, bottom, COLOR_INVALID)
    _shade(d, np.nonzero(latched)[0], x_of, top, bottom, COLOR_LATCH_BAND)

    # --- 网格 (0 线与 ±limit) ---
    for value in (0.0, limit / 2, -limit / 2, limit, -limit):
        y = y_of(value)
        d.line([(left, y), (right, y)],
               fill=(90, 90, 90) if value == 0.0 else COLOR_REL_GRID, width=1)
        label = f"{value:+.4f}"
        d.text((left - 8 - len(label) * 7, y - 7), label, font=font_note, fill=(150, 150, 150))
    for index in range(0, n, 50):
        x = x_of(index)
        d.line([(x, top), (x, bottom)], fill=(30, 36, 42), width=1)
        R._text(d, (x + 3, bottom + 3), str(index), R._font(11), (120, 120, 120))

    # --- 三条曲线: 断开处不连线, 免得把无效段画成"扫过去" ---
    for axis in range(3):
        run: list[tuple[float, float]] = []
        for index in range(n):
            if bool(finite[index]):
                run.append((float(x_of(index)), float(y_of(curves[index, axis]))))
            else:
                if len(run) > 1:
                    d.line(run, fill=R.COLOR_AXIS[axis], width=2, joint="curve")
                run = []
        if len(run) > 1:
            d.line(run, fill=R.COLOR_AXIS[axis], width=2, joint="curve")

    # --- 右轴: 当前值 ---
    if bool(finite[frame]):
        for axis in range(3):
            value = float(curves[frame, axis])
            d.text(
                (right + 12, top + axis * 22),
                f"{'xyz'[axis]} {value:+.4f} m",
                font=font_note,
                fill=R.COLOR_AXIS[axis],
            )
    # --- 游标 ---
    cx = x_of(frame)
    d.line([(cx, top), (cx, bottom)], fill=(255, 255, 255), width=1)
    if bool(finite[frame]):
        for axis in range(3):
            R._disc(d, (cx, y_of(float(curves[frame, axis]))), 4, R.COLOR_AXIS[axis], w=1)

    # --- 底部数值行 (与面板上的曲线同源, 也是终端打印的同一组数) ---
    if bool(finite[frame]):
        t = curves[frame]
        rpy = np.degrees(_rotation_vector(relative[frame][:3, :3]))
        detail = (
            f"frame {frame}   t=[{t[0]:+.4f} {t[1]:+.4f} {t[2]:+.4f}] m   "
            f"|t|={float(np.linalg.norm(t)):.4f} m   "
            f"rotvec=[{rpy[0]:+.2f} {rpy[1]:+.2f} {rpy[2]:+.2f}] deg   "
            f"dist(mid->obj)={float(distance[frame]):.4f} m"
        )
    else:
        detail = (
            f"frame {frame}   relative pose INVALID "
            f"(object observed={int(observed[frame])}, pose valid={int(valid[frame])}, "
            f"latched={int(latched[frame])})"
        )
    d.text((12, 146), detail, font=font_note, fill=COLOR_TEXT)
    return img


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(np.asarray(rotation, dtype=np.float64)).as_rotvec()


def _shade(d, indices, x_of, top, bottom, color) -> None:
    """把一组帧号压成连续区间再填色 (逐帧填矩形在 274 帧上会画 274 个矩形)。"""
    if len(indices) == 0:
        return
    indices = np.asarray(indices, dtype=int)
    cuts = np.flatnonzero(np.diff(indices) != 1)
    starts = np.concatenate([[0], cuts + 1])
    ends = np.concatenate([cuts, [len(indices) - 1]])
    for start, end in zip(starts, ends):
        d.rectangle(
            [x_of(int(indices[start])), top, x_of(int(indices[end])), bottom], fill=color
        )


# ---------------------------------------------------------------- 主循环


def _first_object(value: np.ndarray, object_ndim: int) -> np.ndarray:
    """多物体时取第 0 个物体, 老的单物体产物原样返回。

    `object_ndim` = 带物体轴时该数组的维数 (位姿 4 维 `(T,N,4,4)`, 逐帧标量 2 维 `(T,N)`)。
    这一层 (点云 / 物体系坐标架 / 相对位姿面板) 是**单物体**的画法: 一条曲线、一块文字、
    一个坐标架。多物体时其余物体在终端逐行打印、也在报告里, 画面这一路只画 obj1 ——
    于是 N=1 时画出来的东西与改动前逐像素相同。
    """
    value = np.asarray(value)
    return value[:, 0] if value.ndim == object_ndim else value


def _other_objects(distances, valid, instance_ids, frame) -> list[tuple[str, float, bool]]:
    """除 obj1 外其余物体在这一帧的 (instance_id, 距离, 是否有效) —— 给文字块用。"""
    out: list[tuple[str, float, bool]] = []
    for j in range(1, len(instance_ids)):
        distance = float(distances[frame, j])
        out.append((str(instance_ids[j]), distance, bool(valid[frame, j])))
    return out


def _fields(index, relative, valid, latched, observed, distance, confidence, residual,
            closed=None, others=None) -> dict:
    T_rel = relative[index] if bool(valid[index]) else None
    if latched[index]:
        # `latched` = 本帧位姿是「手 x T_hand_object」推出来的, **不是**一次测量
        status = "HAND-PUSHED (no measurement)"
    elif valid[index]:
        status = "OBSERVED" if observed[index] else "NO-OBS"
    elif observed[index]:
        status = "POSE-REJECTED"  # 有观测但被门控丢了 (或内点不足)
    else:
        status = "NO-OBS"
    return {
        "T_object_midpoint": T_rel,
        "rpy": np.degrees(_rotation_vector(T_rel[:3, :3])) if T_rel is not None else np.zeros(3),
        "distance": float(distance[index]) if np.isfinite(distance[index]) else float("nan"),
        "latched": bool(latched[index]),
        "observed": bool(observed[index]),
        "closed": bool(closed[index]) if closed is not None else bool(latched[index]),
        # conf=/res= 只对**有测量**的帧有意义: 手推帧本帧没跑拟合, 它们的 confidence /
        # residual_m 记的是 nan (见 objectpose 循环顶部), 打出来只会是 conf=nan res=nan。
        "status": (
            status if latched[index]
            else f"{status}  conf={confidence[index]:.2f} res={residual[index]:.4f} m"
        ),
        # 多物体时其余物体各一行 (N=1 时是空表 -> 文字块与改动前逐像素相同)
        "others": list(others or []),
    }


def _print_line(frame: int, fields: dict) -> None:
    """终端那一路的「实时打印相对位姿」—— 与画面上的文字块、npz 同帧同值。"""
    if fields["T_object_midpoint"] is None:
        print(f"  f={frame:04d} REL 无效  {fields['status']}", flush=True)
        return
    t = fields["T_object_midpoint"][:3, 3]
    rpy = fields["rpy"]
    print(
        f"  f={frame:04d} t_rel=[{t[0]:+.4f} {t[1]:+.4f} {t[2]:+.4f}] m  "
        f"RPY=[{rpy[0]:+7.2f} {rpy[1]:+7.2f} {rpy[2]:+7.2f}] deg  "
        f"|t|={fields['distance']:.4f}  {fields['status']}",
        flush=True,
    )


def run(
    paths: RelPaths,
    *,
    reel=None,
    frames=None,
    radius: int = 5,
    frame_print: bool = True,
    verbose: bool = True,
) -> dict:
    """出 `render/rel_<stem>.mp4` (2160x1156); `frames` 里的帧另外存静帧 PNG。

    `frames=None` = 只出视频。视频**始终整片** (帧号是时间轴的一部分, 抽帧会让面板曲线
    与画面对不上), `frames` 只挑哪几帧另外存 PNG —— 与 `tools/seg_object.py --frames`
    的约定一致。
    """
    from .relation import load_overlay, load_reel

    if reel is None:
        _, _, reel = load_reel(paths.stem)

    with np.load(paths.relation_npz) as archive:
        # 多物体: 这些键都带物体轴 (位姿 (T,N,4,4), 逐帧标量 (T,N))。这一层只画 obj1,
        # 所以取第 0 个物体; 老的单物体产物维数不同, `_first_object` 原样放行。
        relative = _first_object(archive["T_object_right_midpoint"].astype(np.float64), 4)
        valid = _first_object(archive["valid"].astype(bool), 2)
        latched = _first_object(archive["latched"].astype(bool), 2)
        observed = _first_object(archive["object_observed"].astype(bool), 2)
        distance = _first_object(archive["distance_m"].astype(np.float64), 2)
        confidence = _first_object(archive["confidence"].astype(np.float64), 2)
        residual = _first_object(archive["residual_m"].astype(np.float64), 2)
        T_object_camera0 = _first_object(archive["T_camera0_object"].astype(np.float64), 4)
        T_mid_camera0 = archive["T_camera0_right_midpoint"].astype(np.float64)
        # 其余物体只用于文字块那一行 (距离), 不走曲线/面板
        others_distance = archive["distance_m"].astype(np.float64)
        others_valid = archive["valid"].astype(bool)
        instance_ids = (
            [str(v) for v in np.atleast_1d(archive["instance_ids"])]
            if "instance_ids" in archive.files else []
        )

    clouds = load_clouds(paths)
    # 夹爪二值状态 (与 step1 同源, 用于状态色)
    gripper_closed = np.asarray(reel.G_CLOSED, dtype=bool)
    n_frames = len(relative)
    warning = _warn_if_calibrated_f(reel)
    if warning and verbose:
        print(f"    [warn] {warning}")

    from xrseg.render import _mask_overlay_bgr  # 按需 import: 它在 .venv 里才可用

    overlay = load_overlay()
    wanted_stills = set(int(f) for f in (frames or []))
    video = VideoWriter(str(paths.video), FULL_W, VIDEO_H, reel.fps, crf=18)
    written = 0
    still_paths: list[str] = []
    stats = {"no_obs": 0, "invalid": 0, "printed": 0}
    try:
        # `iter_frames` 逐帧吐**裸 ndarray** (xrhand/video.py:95 的 `Iterator[np.ndarray]`),
        # 帧号要自己 enumerate —— 别写成 `for frame, frame_rgb in iter_frames(...)`。
        for frame, frame_rgb in enumerate(iter_frames(str(paths.mp4), FULL_W, FULL_H)):
            if frame >= n_frames:
                break
            mask = load_mask(paths, frame)
            if mask is None:
                mask = np.zeros((EYE_H, EYE_W), dtype=np.uint8)
            # ① step2: 物体分割叠色 (原样调用)
            rgb = _mask_overlay_bgr(frame_rgb, mask)
            # ② step1: 5 个关键点 + 夹爪 + 抓取面板 (原样调用, 返回 2160x988)
            image = overlay.draw_frame(reel, rgb, frame, radius, inset=False, mode="gripper")
            # ③ step3: 点云 / 物体系 / 连线 / 文字
            start, end = int(clouds["offset"][frame]), int(clouds["offset"][frame + 1])
            fields = _fields(
                frame, relative, valid, latched, observed, distance,
                confidence, residual, closed=gripper_closed,
                others=(
                    _other_objects(others_distance, others_valid, instance_ids, frame)
                    if others_distance.ndim == 2 else None
                ),
            )
            image = draw_relative(
                image,
                frame=frame,
                mask_cloud=clouds["points"][start:end],
                T_camera0_object=T_object_camera0[frame],
                T_camera0_midpoint=T_mid_camera0[frame],
                T_object_midpoint=fields["T_object_midpoint"],
                fields=fields,
            )
            # ④ 底部相对位姿面板
            panel = draw_rel_panel(
                FULL_W,
                frame=frame,
                relative=relative,
                valid=valid,
                latched=latched,
                observed=observed,
                distance=distance,
                source=f"stem {paths.stem}",
            )
            canvas = Image.new("RGB", (FULL_W, VIDEO_H), COLOR_REL_BG)
            canvas.paste(image, (0, 0))
            canvas.paste(panel, (0, FULL_H + R.PANEL_H))
            video.write(np.asarray(canvas))
            written += 1

            stats["no_obs"] += int(not observed[frame])
            stats["invalid"] += int(not valid[frame])
            if frame_print:
                _print_line(frame, fields)
                stats["printed"] += 1
            if frame in wanted_stills:
                target = paths.still(frame)
                canvas.save(target)
                still_paths.append(str(target))
            if verbose and written % 50 == 0:
                print(f"    渲染 {written}/{n_frames} 帧", flush=True)
    finally:
        video.close()

    report = {
        "video": str(paths.video),
        "frames_rendered": int(written),
        "size": [FULL_W, VIDEO_H],
        "layers": [
            "step2: xrseg.render._mask_overlay_bgr (SAM2 object mask, left half)",
            "step1: tools/overlay.draw_frame(mode='gripper') "
            "(5 keypoints W/TB/IB/T/I + gripper 6DoF + CLOSED/OPEN + grasp panel)",
            "step3: xrrel.render (object cloud / object frame / midpoint->object link / "
            "per-half pose text / relative-pose panel)",
        ],
        "panel_heights": {"grasp_panel": R.PANEL_H, "relative_panel": REL_PANEL_H},
        "no_observation_frames": int(stats["no_obs"]),
        "invalid_pose_frames": int(stats["invalid"]),
        "terminal_lines_printed": int(stats["printed"]),
        "stills": still_paths,
        "notes": [warning] if warning else [],
    }
    if verbose:
        print(
            f"    -> {paths.video} ({written} 帧, {FULL_W}x{VIDEO_H}, "
            f"无观测 {stats['no_obs']} 帧, 位姿无效 {stats['invalid']} 帧)"
        )
        if still_paths:
            print(f"    静帧: {', '.join(still_paths)}")
    return report
