"""Step2 完整关系可视化：所有物体与手指中点 TCP 同框。

同一个视频画布从上到下包含：

* 2160x810 双目原图：左眼同时叠加每个物体的彩色 mask、点云、物体 TCP、
  手指中点到物体的连线和距离；两眼均叠加 Step1 五关键点与手指中点 TCP。
* 178 px Step1 抓取判据面板。
* 每个物体各 168 px 的相对位姿曲线面板，纵向堆叠在同一视频中。

全部投影几何仍使用 camera0（左眼）坐标系；这里只做诊断渲染，不改 action/reference。
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from xrhand import render as R
from xrhand.video import VideoWriter, iter_frames

from . import EYE_F, EYE_H, EYE_W, FULL_H, FULL_W, RelPaths
from .lift import load_clouds, load_mask, project_camera0

REL_PANEL_H = 168  # 每个物体一块；多物体时在同一视频画布内纵向堆叠
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


def _object_rgb(instance_id: str) -> tuple[int, int, int]:
    """`xrseg` 的实例色是 BGR；PIL 图层需要 RGB。"""
    from xrseg.common import instance_color

    blue, green, red = instance_color(instance_id)
    return int(red), int(green), int(blue)


def draw_relative(
    image: Image.Image,
    *,
    frame: int,
    mask_cloud: np.ndarray,
    T_camera0_object: np.ndarray,
    T_camera0_midpoint: np.ndarray,
    T_object_midpoint: np.ndarray | None,
    fields: dict,
    instance_id: str = "obj1",
    object_color: tuple[int, int, int] | None = None,
    valid: bool = True,
    draw_text: bool = True,
) -> Image.Image:
    """把一个物体的点云、TCP 和手到物体连线叠到同一双目画面。

    点云只要求本帧有深度观测；TCP/连线必须要求关系位姿有效，避免无效帧把单位阵
    当成真实物体 TCP 画在相机原点。多物体时调用方对每个物体依次调用本函数。
    """
    del frame
    img = image.convert("RGB")
    d = ImageDraw.Draw(img)
    font_small = R._font(13)
    color = tuple(object_color or _object_rgb(instance_id))

    if mask_cloud.shape[0]:
        u, v, z = _project_camera0(mask_cloud)
        keep = _inside_half(u, v, z, 0)
        if keep.any():
            d.point([(float(x), float(y)) for x, y in zip(u[keep], v[keep])], fill=color)

    # 无效帧只保留 mask/点云与状态文字，不画由无效矩阵产生的伪 TCP。
    if valid and T_object_midpoint is not None:
        origin_cam = np.asarray(T_camera0_object, dtype=np.float64)[:3, 3]
        rotation = np.asarray(T_camera0_object, dtype=np.float64)[:3, :3]
        axis_cam = np.stack([origin_cam + AXIS_LEN_M * rotation[:, k] for k in range(3)])
        u0, v0, z0 = _project_camera0(origin_cam[None])
        ua, va, za = _project_camera0(axis_cam)
        origin_inside = bool(_inside_half(u0, v0, z0, 0)[0])
        if origin_inside:
            base = (float(u0[0]), float(v0[0]))
            for k in range(3):
                if not _inside_half(ua[k:k + 1], va[k:k + 1], za[k:k + 1], 0)[0]:
                    continue
                tip = (float(ua[k]), float(va[k]))
                d.line([base, tip], fill=R.COLOR_AXIS[k], width=2)
                R._arrow_tip(d, base, tip, R.COLOR_AXIS[k])
                R._text(d, (tip[0] + 6, tip[1] + 4),
                        f"{instance_id}:{'XYZ'[k]}", font_small, R.COLOR_AXIS[k])
            R._disc(d, base, 7, color, w=2)
            R._text(d, (base[0] + 8, base[1] - 18), f"{instance_id} TCP", font_small, color)

        mid_cam = np.asarray(T_camera0_midpoint, dtype=np.float64)[:3, 3]
        um, vm, zm = _project_camera0(mid_cam[None])
        if bool(_inside_half(um, vm, zm, 0)[0]) and origin_inside:
            mid_px = (float(um[0]), float(vm[0]))
            origin_px = (float(u0[0]), float(v0[0]))
            d.line([mid_px, origin_px], fill=color, width=2)
            length = float(np.linalg.norm(T_object_midpoint[:3, 3]))
            centre = ((mid_px[0] + origin_px[0]) / 2 + 6,
                      (mid_px[1] + origin_px[1]) / 2 - 18)
            R._text(d, centre, f"{instance_id} {length:.3f} m", font_small, color)

    if draw_text:
        img = draw_fields_text(img, [fields])
    return img


def _text_block(fields: dict) -> list[tuple[str, tuple[int, int, int]]]:
    """一个物体的两行实时关系文字；多物体时由 `draw_fields_text` 合并绘制。"""
    instance_id = str(fields.get("instance_id", "obj1"))
    if fields["T_object_midpoint"] is None:
        return [
            (f"{instance_id} REL n/a (pose invalid this frame)", COLOR_TEXT_DIM),
            (f"{instance_id} {fields['status']}", COLOR_TEXT_DIM),
        ]
    t = fields["T_object_midpoint"][:3, 3]
    rpy = fields["rpy"]
    state_color = (
        R.COLOR_GRASP_CLOSED
        if fields["closed"]
        else (R.COLOR_GRASP_OPEN if fields["observed"] else COLOR_TEXT_DIM)
    )
    return [
        (f"{instance_id} REL t=[{t[0]:+.4f} {t[1]:+.4f} {t[2]:+.4f}] m  "
         f"|t|={fields['distance']:.4f} m", COLOR_TEXT),
        (f"{instance_id} RPY=[{rpy[0]:+.2f} {rpy[1]:+.2f} {rpy[2]:+.2f}] deg   "
         f"{fields['status']}", state_color),
    ]


def draw_fields_text(image: Image.Image, fields_by_object: list[dict]) -> Image.Image:
    """在同一视频画面内同时列出所有物体的相对位姿与状态。"""
    img = image.convert("RGB")
    d = ImageDraw.Draw(img)
    font = R._font(15)
    lines = [line for fields in fields_by_object for line in _text_block(fields)]
    if not lines:
        return img
    block_w = max(d.textlength(text, font=font) for text, _ in lines)
    top = max(8, EYE_H - 12 - 20 * len(lines))
    for half in range(2):
        x = half * EYE_W + EYE_W - 12 - block_w
        for index, (text, color) in enumerate(lines):
            R._text(d, (x, top + index * 20), text, font, color)
    return img


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
    instance_id: str = "obj1",
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
        f"STEP2 RELATIVE POSE [{instance_id}]   "
        "T_object_right_midpoint = inv(T_camera0_object) @ T_camera0_right_midpoint   "
        "(right EEF = thumb/index tip midpoint, HumanEgo frame)   "
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
            f"{instance_id} frame {frame}   t=[{t[0]:+.4f} {t[1]:+.4f} {t[2]:+.4f}] m   "
            f"|t|={float(np.linalg.norm(t)):.4f} m   "
            f"rotvec=[{rpy[0]:+.2f} {rpy[1]:+.2f} {rpy[2]:+.2f}] deg   "
            f"dist(mid->obj)={float(distance[frame]):.4f} m"
        )
    else:
        detail = (
            f"{instance_id} frame {frame}   relative pose INVALID "
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


def _with_object_axis(value: np.ndarray, object_ndim: int, name: str) -> np.ndarray:
    """把旧单物体 `(T,...)` 与新多物体 `(T,N,...)` 统一成带物体轴的形状。"""
    array = np.asarray(value)
    if array.ndim == object_ndim - 1:
        array = array[:, None]
    if array.ndim != object_ndim:
        raise ValueError(f"{name} 维数应为 {object_ndim - 1} 或 {object_ndim}, 拿到 {array.shape}")
    return array


def video_height(object_count: int) -> int:
    """同一个视频画布：原图+抓取面板，再为每个物体纵向追加一块关系面板。"""
    count = int(object_count)
    if count <= 0:
        raise ValueError(f"物体数必须 > 0, 拿到 {object_count}")
    return FULL_H + R.PANEL_H + REL_PANEL_H * count


def _fields(index, relative, valid, latched, observed, distance, confidence, residual,
            closed=None, *, instance_id: str = "obj1") -> dict:
    T_rel = relative[index] if bool(valid[index]) else None
    if latched[index]:
        status = "HAND-PUSHED (no measurement)"
    elif valid[index]:
        status = "OBSERVED" if observed[index] else "NO-OBS"
    elif observed[index]:
        status = "POSE-REJECTED"
    else:
        status = "NO-OBS"
    return {
        "instance_id": str(instance_id),
        "T_object_midpoint": T_rel,
        "rpy": np.degrees(_rotation_vector(T_rel[:3, :3])) if T_rel is not None else np.zeros(3),
        "distance": float(distance[index]) if np.isfinite(distance[index]) else float("nan"),
        "latched": bool(latched[index]),
        "observed": bool(observed[index]),
        "closed": bool(closed[index]) if closed is not None else bool(latched[index]),
        "status": (
            status if latched[index]
            else f"{status}  conf={confidence[index]:.2f} res={residual[index]:.4f} m"
        ),
    }


def _print_line(frame: int, fields: dict) -> None:
    """可选的逐帧终端输出；pipeline Step2 默认关闭，只保留视频进度。"""
    instance_id = fields.get("instance_id", "obj1")
    if fields["T_object_midpoint"] is None:
        print(f"  f={frame:04d} {instance_id} REL 无效  {fields['status']}", flush=True)
        return
    t = fields["T_object_midpoint"][:3, 3]
    rpy = fields["rpy"]
    print(
        f"  f={frame:04d} {instance_id} t_rel=[{t[0]:+.4f} {t[1]:+.4f} {t[2]:+.4f}] m  "
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
    """生成一个包含所有物体的 `render/rel_<stem>.mp4`。

    每帧同时显示全部物体的 mask、点云、TCP、手到物体连线和距离；底部在同一画布内
    为每个物体纵向堆叠一块 168 px 关系面板。`frames` 只控制额外导出的静帧，视频始终
    保留完整时间轴。
    """
    from .relation import load_overlay, load_reel

    paths.ensure()
    if reel is None:
        _, _, reel = load_reel(paths.stem)

    with np.load(paths.relation_npz) as archive:
        relative = _with_object_axis(
            archive["T_object_right_midpoint"].astype(np.float64), 4,
            "T_object_right_midpoint",
        )
        valid = _with_object_axis(archive["valid"].astype(bool), 2, "valid")
        latched = _with_object_axis(archive["latched"].astype(bool), 2, "latched")
        observed = _with_object_axis(
            archive["object_observed"].astype(bool), 2, "object_observed"
        )
        distance = _with_object_axis(
            archive["distance_m"].astype(np.float64), 2, "distance_m"
        )
        confidence = _with_object_axis(
            archive["confidence"].astype(np.float64), 2, "confidence"
        )
        residual = _with_object_axis(
            archive["residual_m"].astype(np.float64), 2, "residual_m"
        )
        T_object_camera0 = _with_object_axis(
            archive["T_camera0_object"].astype(np.float64), 4, "T_camera0_object"
        )
        T_mid_camera0 = archive["T_camera0_right_midpoint"].astype(np.float64)
        instance_ids = (
            [str(v) for v in np.atleast_1d(archive["instance_ids"])]
            if "instance_ids" in archive.files else ["obj1"]
        )

    n_frames, object_count = valid.shape
    if len(instance_ids) != object_count:
        raise ValueError(
            f"instance_ids 有 {len(instance_ids)} 个, 关系数组有 {object_count} 个物体"
        )
    arrays = {
        "relative": relative,
        "latched": latched,
        "observed": observed,
        "distance": distance,
        "confidence": confidence,
        "residual": residual,
        "T_camera0_object": T_object_camera0,
    }
    for name, array in arrays.items():
        if array.shape[:2] != (n_frames, object_count):
            raise ValueError(f"{name} 形状 {array.shape} 与 {(n_frames, object_count)} 不一致")
    if T_mid_camera0.shape != (n_frames, 4, 4):
        raise ValueError(f"T_camera0_right_midpoint 形状错误: {T_mid_camera0.shape}")

    clouds_by_object = {
        instance_id: load_clouds(paths, instance_id) for instance_id in instance_ids
    }
    for instance_id, clouds in clouds_by_object.items():
        if len(clouds["offset"]) != n_frames + 1:
            raise ValueError(
                f"{instance_id} 点云 offset 长度 {len(clouds['offset'])} != {n_frames + 1}"
            )

    gripper_closed = np.asarray(reel.G_CLOSED, dtype=bool)
    if len(gripper_closed) < n_frames:
        raise ValueError(f"手状态只有 {len(gripper_closed)} 帧, 关系数据有 {n_frames} 帧")
    warning = _warn_if_calibrated_f(reel)
    if warning and verbose:
        print(f"    [warn] {warning}")

    from xrseg.render import _mask_overlay_bgr

    overlay = load_overlay()
    wanted_stills = set(int(f) for f in (frames or []))
    output_height = video_height(object_count)
    video = VideoWriter(str(paths.video), FULL_W, output_height, reel.fps, crf=18)
    written = 0
    still_paths: list[str] = []
    per_object = {
        instance_id: {"no_observation_frames": 0, "invalid_pose_frames": 0}
        for instance_id in instance_ids
    }
    printed = 0
    try:
        for frame, frame_rgb in enumerate(iter_frames(str(paths.mp4), FULL_W, FULL_H)):
            if frame >= n_frames:
                break
            masks = {}
            for instance_id in instance_ids:
                mask = load_mask(paths, frame, instance_id)
                masks[instance_id] = (
                    mask if mask is not None else np.zeros((EYE_H, EYE_W), dtype=np.uint8)
                )
            # ① 所有物体 mask 同时叠在左眼。
            rgb = _mask_overlay_bgr(frame_rgb, masks)
            # ② Step1 的五关键点、手指中点 TCP 坐标架与抓取面板。
            image = overlay.draw_frame(reel, rgb, frame, radius, inset=False, mode="gripper")

            # ③ 所有物体点云、TCP、手->物体连线同时叠在同一画面。
            fields_by_object = []
            for j, instance_id in enumerate(instance_ids):
                clouds = clouds_by_object[instance_id]
                cloud_start = int(clouds["offset"][frame])
                cloud_end = int(clouds["offset"][frame + 1])
                fields = _fields(
                    frame,
                    relative[:, j], valid[:, j], latched[:, j], observed[:, j],
                    distance[:, j], confidence[:, j], residual[:, j],
                    closed=gripper_closed, instance_id=instance_id,
                )
                fields_by_object.append(fields)
                image = draw_relative(
                    image,
                    frame=frame,
                    mask_cloud=clouds["points"][cloud_start:cloud_end],
                    T_camera0_object=T_object_camera0[frame, j],
                    T_camera0_midpoint=T_mid_camera0[frame],
                    T_object_midpoint=fields["T_object_midpoint"],
                    fields=fields,
                    instance_id=instance_id,
                    object_color=_object_rgb(instance_id),
                    valid=bool(valid[frame, j]),
                    draw_text=False,
                )
                per_object[instance_id]["no_observation_frames"] += int(
                    not observed[frame, j]
                )
                per_object[instance_id]["invalid_pose_frames"] += int(not valid[frame, j])
                if frame_print:
                    _print_line(frame, fields)
                    printed += 1
            image = draw_fields_text(image, fields_by_object)

            # ④ 同一个视频画布内为每个物体纵向追加一块关系曲线面板。
            canvas = Image.new("RGB", (FULL_W, output_height), COLOR_REL_BG)
            canvas.paste(image, (0, 0))
            panel_top = FULL_H + R.PANEL_H
            for j, instance_id in enumerate(instance_ids):
                panel = draw_rel_panel(
                    FULL_W,
                    frame=frame,
                    relative=relative[:, j],
                    valid=valid[:, j],
                    latched=latched[:, j],
                    observed=observed[:, j],
                    distance=distance[:, j],
                    source=f"stem {paths.stem}",
                    instance_id=instance_id,
                )
                canvas.paste(panel, (0, panel_top + j * REL_PANEL_H))

            video.write(np.asarray(canvas))
            written += 1
            if frame in wanted_stills:
                target = paths.still(frame)
                canvas.save(target)
                still_paths.append(str(target))
            if verbose and written % 50 == 0:
                print(f"    渲染 {written}/{n_frames} 帧", flush=True)
    finally:
        video.close()

    no_observation_total = sum(v["no_observation_frames"] for v in per_object.values())
    invalid_total = sum(v["invalid_pose_frames"] for v in per_object.values())
    report = {
        "video": str(paths.video),
        "frames_rendered": int(written),
        "size": [FULL_W, output_height],
        "instance_ids": instance_ids,
        "object_count": int(object_count),
        "layers": [
            "step2: all SAM2 object masks (left eye, per-instance colors)",
            "step1: five hand keypoints + fingertip-midpoint TCP + grasp panel",
            "step2: all object clouds/TCPs + midpoint-to-object links/distances",
            "step2: one relative-pose panel per object in the same video canvas",
        ],
        "panel_heights": {
            "grasp_panel": R.PANEL_H,
            "relative_panel_each": REL_PANEL_H,
            "relative_panels": int(object_count),
        },
        "per_object": per_object,
        "no_observation_object_frames": int(no_observation_total),
        "invalid_pose_object_frames": int(invalid_total),
        "terminal_lines_printed": int(printed),
        "stills": still_paths,
        "notes": [warning] if warning else [],
    }
    if verbose:
        details = ", ".join(
            f"{instance_id}: 无观测 {stats['no_observation_frames']} / "
            f"位姿无效 {stats['invalid_pose_frames']}"
            for instance_id, stats in per_object.items()
        )
        print(
            f"    -> {paths.video} ({written} 帧, {FULL_W}x{output_height}; {details})"
        )
        if still_paths:
            print(f"    静帧: {', '.join(still_paths)}")
    return report
