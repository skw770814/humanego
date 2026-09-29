"""相对位姿: `T_object_right_midpoint = inv(T_cam0_object) @ T_cam0_midpoint`。

三段拼起来:

  1. **手的中点系 (世界系)** —— 直接把 step1 的那条链路原样跑一遍: `tools/overlay.py`
     的 `Reel.compute_gripper()`。它内部就是 `xrhand/gripper.py` 的
     `pinch_distance` / `midpoint`(AriaHands.py:359) / `midpoint_frame`
     (MidpointFrameBuilder.build) / `smooth_midpoint_frame`(AriaHandsOptimizer.py:267-274),
     并且把结果存在 `reel.G_MID` (N,3) / `reel.G_R` (N,3,3) / `reel.G_CLOSED` 里。
     **这里不重算、不复制**, 所以 step1 与 step3 看到的末端是同一个末端, 逐像素同级。

  2. **世界系 -> 相机0 (左眼)** —— 见 `camera_from_world`。zrrel 自己按 `Projector`
     (xrhand/camera.py:120-178) 的**同一顺序**拼 4x4, 因为 `Projector` 只给逐点的
     `(u,v,z)`, 而算相对位姿需要整个 4x4。`self_check_camera0` 用随机点与
     `Projector.project` 逐点比对把它锁住 (<1e-9)。

     **为什么必须自己拼、不能用别的等价写法**: `head_to_cam` 在**每眼外参之前**减了一个
     固定的 `extra_t`(camera.py:152), 所以"从视差直接反投影出来的点"与"手经 Projector 算
     出来的点"天然差一个常向量。物体点云活在左眼相机系 (见 xrrel/lift.py), 手也必须走
     **half=0 / eye=0** 这条链, 两边才同系。第 2 轮就是在这上面栽的, 所以这次锁死。

  3. **相对位姿** —— 照 `encoding.py:474-491`:
         T_object_right_midpoint = compose(invert(T_camera0_object), T_camera0_midpoint)
         T_right_midpoint_object = compose(invert(T_camera0_midpoint), T_camera0_object)
     `invert`/`compose` 直接用 `contracts/se3.py:60-72` (真 SE(3) 逆: Rᵀ, -Rᵀt)。

自然语言描述 (`relation_text`) 照 `encoding.py:300-348`: 距离分档
`contact/near/reachable/far` (bins 取他 `configs/default.yaml:168` 的
`relations.distance_bins_m=[0.06,0.15,0.35]`)、方向词取相机轴 (左右/上下/前后)、
靠近/远离用 `relations.approach_speed_m_s=0.04`。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from . import INSTANCE_ID, OUT, WORK, RelPaths

# tools/ 不是包 (没有 __init__.py), 所以按文件路径 importlib 载入 overlay.py ——
# 与 xrseg/hand_overlay.py:29-43 同一个做法, 不重复造。
_OVERLAY_MODULE = "xrrel_tools_overlay"


def load_overlay():
    path = WORK / "tools" / "overlay.py"
    if not path.is_file():
        raise FileNotFoundError(f"找不到 {path}")
    if _OVERLAY_MODULE in sys.modules:
        return sys.modules[_OVERLAY_MODULE]
    spec = importlib.util.spec_from_file_location(_OVERLAY_MODULE, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_OVERLAY_MODULE] = module
    spec.loader.exec_module(module)  # 它自己会 sys.path.insert(0, WORK)
    return module


def load_params(calib: str | Path | None = None):
    """相机参数: 默认读 out/calib.json (与 tools/overlay.py / xrseg/hand_overlay.py 同约定)。

    **读不到就报错, 不用标称值兜底**: `extra_t` 全零会让骨架整体偏 37~163 px
    (overlay.py 文件头记着这个教训), 拿它算相对位姿得到的是错的位置, 不如直接停。
    """
    from xrhand.camera import CameraParams

    path = Path(calib) if calib else OUT / "calib.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"缺 {path} —— 标称参数会让投影整体偏, 别用来算相对位姿; "
            f"先跑 python tools/make_calib.py"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = CameraParams.from_json(payload["params"])
    return params, path


def load_reel(stem: str, *, calib=None, lag=None, gripper: bool = True):
    """建 `overlay.Reel` (帧对齐 + 26 点投影) 并算好夹爪那条链。

    返回 (overlay 模块, params, reel)。`reel.compute_gripper()` 之后
    `reel.G_MID/G_R/G_POSE_VALID/G_CLOSED/G_CLOSURE/G_CALIB` 才存在。
    """
    overlay = load_overlay()
    params, _ = load_params(calib)
    reel = overlay.Reel(stem, lag, params)
    if gripper:
        reel.compute_gripper()
    return overlay, params, reel


# ---------------------------------------------------------------- 世界 -> 相机0


def _se3(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = np.asarray(rotation, dtype=np.float64)
    out[:3, 3] = np.asarray(translation, dtype=np.float64)
    return out


def camera_from_world(extrinsics, params, head_pos, head_quat, eye: int = 0) -> np.ndarray:
    """`T_camera_eye_world` (4x4) —— `Projector` 那条链的矩阵形式, 逐段对应。

    `Projector` 用行向量写 (`p @ M`), 这里换成列向量 (`M.T @ p`), 所以每段的旋转取转置:

        world_to_head : p_head = (p_world - t_head) @ R      ->  R_headᵀ (p_world - t_head)
        head_to_cam   : p_dev  = p_head @ Rz.T               ->  Rz p_head
                        p_dev  = (p_dev - extra_t) @ extra_R.T -> extra_R (p_dev - extra_t)
                        p_cam  = (p_dev - t_E) @ R_E.T       ->  R_E (p_dev - t_E)

    `extrinsic_subtract_t=False` 那一支 (`p_cam = p_dev @ R.T + t`) 也照样支持。
    """
    from xrhand.camera import rotz
    from xrhand.io_tracking import quat_to_mat

    head_pos = np.asarray(head_pos, dtype=np.float64)
    rotation_head = quat_to_mat(np.asarray(head_quat, dtype=np.float64))
    if params.flip_head_quat:
        rotation_head = rotation_head.T

    rotation_eye = np.asarray(extrinsics, dtype=np.float64)[eye][:3, :3]
    translation_eye = np.asarray(extrinsics, dtype=np.float64)[eye][:3, 3]

    T_head_world = _se3(rotation_head.T, -rotation_head.T @ head_pos)
    T_dev_head = _se3(rotz(params.rz_device_deg), np.zeros(3))
    extra_R = np.asarray(params.extra_R, dtype=np.float64)
    extra_t = np.asarray(params.extra_t, dtype=np.float64)
    T_dev2_dev = _se3(extra_R, -extra_R @ extra_t)
    if params.extrinsic_subtract_t:
        T_cam_dev2 = _se3(rotation_eye, -rotation_eye @ translation_eye)
    else:
        T_cam_dev2 = _se3(rotation_eye, translation_eye)
    return T_cam_chain(T_cam_dev2, T_dev2_dev, T_dev_head, T_head_world)


def T_cam_chain(*transforms) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    for transform in transforms:
        out = out @ np.asarray(transform, dtype=np.float64)
    return out


def self_check_camera0(reel, *, eye: int = 0, samples: int = 64, seed: int = 20260921) -> dict:
    """随机世界点: `camera_from_world` + `cam_to_pixel` 应逐点等于 `Projector.project`。

    这一步是防止"物体点云一个系、手另一个系"那个老坑 (第 2 轮)。误差要求 < 1e-9。
    """
    rng = np.random.default_rng(seed)
    record = reel.records[reel.ridx[len(reel.ridx) // 2]]
    points = rng.normal(scale=0.35, size=(samples, 3)) + np.asarray(record.head_pos)
    transform = camera_from_world(reel.header.extrinsics, reel.params, record.head_pos,
                                  record.head_quat, eye=eye)
    p_cam = points @ transform[:3, :3].T + transform[:3, 3]
    u_chain, v_chain, z_chain = reel.proj.cam_to_pixel(p_cam)
    u_ref, v_ref, z_ref, _ = reel.proj.project(points, record.head_pos, record.head_quat, eye)

    rotation = transform[:3, :3]
    report = {
        "eye": int(eye),
        "samples": int(samples),
        "max_pixel_error": float(np.max(np.abs(np.stack([u_chain - u_ref, v_chain - v_ref])))),
        "max_depth_error": float(np.max(np.abs(z_chain - z_ref))),
        "det_rotation": float(np.linalg.det(rotation)),
        "max_orthogonality_error": float(np.max(np.abs(rotation.T @ rotation - np.eye(3)))),
        "extra_t_applied_m": np.asarray(reel.params.extra_t, dtype=float).tolist(),
        "flip_head_quat": bool(reel.params.flip_head_quat),
        "extrinsic_subtract_t": bool(reel.params.extrinsic_subtract_t),
        "eye0_is_left_half": bool(reel.params.eye0_is_left_half),
    }
    if report["max_pixel_error"] > 1e-9 or report["max_depth_error"] > 1e-9:
        raise AssertionError(
            f"T_camera0_world 与 Projector 不一致 (最大像素误差 "
            f"{report['max_pixel_error']:.3e}, 深度误差 {report['max_depth_error']:.3e}) —— "
            f"物体点云与手会落在不同坐标系"
        )
    if abs(report["det_rotation"] - 1.0) > 1e-12:
        raise AssertionError(f"det(R)={report['det_rotation']!r} 不是 +1")
    return report


# ---------------------------------------------------------------- 手的状态


def hand_states(reel, n_frames: int) -> dict:
    """把 `Reel` 的逐帧结果整理成 (N,4,4) 的相机系位姿。

    `T_camera0_midpoint` 只在 `G_POSE_VALID` 的帧上有意义 (5 点齐且几何不退化),
    其余帧是单位阵并配 `valid=false`。
    """
    world_mid = np.asarray(reel.G_MID, dtype=np.float64)
    world_rot = np.asarray(reel.G_R, dtype=np.float64)
    pose_valid = np.asarray(reel.G_POSE_VALID, dtype=bool)
    closed = np.asarray(reel.G_CLOSED, dtype=bool)

    T_cam_world = np.repeat(np.eye(4)[None], n_frames, axis=0)
    T_cam_mid = np.repeat(np.eye(4)[None], n_frames, axis=0)
    T_world_mid = np.repeat(np.eye(4)[None], n_frames, axis=0)
    valid = np.zeros(n_frames, dtype=bool)
    for frame in range(n_frames):
        record = reel.records[reel.ridx[frame]]
        T_cam_world[frame] = camera_from_world(
            reel.header.extrinsics, reel.params, record.head_pos, record.head_quat, eye=0
        )
        if not pose_valid[frame]:
            continue
        T_world_mid[frame] = _se3(world_rot[frame], world_mid[frame])
        T_cam_mid[frame] = T_cam_world[frame] @ T_world_mid[frame]
        valid[frame] = True
    return {
        "T_camera0_world": T_cam_world,
        "T_camera0_midpoint": T_cam_mid,
        "T_world_midpoint": T_world_mid,
        "valid": valid,
        "closed": closed,
        "key_valid": np.asarray(reel.G_KEY_VALID, dtype=bool),
        "closure": np.asarray(reel.G_CLOSURE, dtype=np.float64),
        "pinch_m": np.asarray(reel.G_PINCH, dtype=np.float64),
        "judge": np.asarray(reel.G_JUDGE, dtype=bool),
        "timestamps_ns": np.asarray(
            [reel.records[reel.ridx[f]].time_stamp_ns for f in range(n_frames)], dtype=np.int64
        ),
        "grasp_calibration": dict(reel.G_CALIB),
    }


# ---------------------------------------------------------------- 自然语言 (照 encoding.py:300-348)


def _distance_word(distance: float, bins) -> str:
    labels = ("contact", "near", "reachable", "far")
    return labels[min(int(np.searchsorted(np.asarray(bins), distance, side="right")), len(labels) - 1)]


def _direction_word(relative_translation: np.ndarray) -> str:
    axis = int(np.argmax(np.abs(relative_translation)))
    if axis == 0:
        return "right" if relative_translation[axis] > 0 else "left"
    if axis == 1:
        return "below" if relative_translation[axis] > 0 else "above"
    return "behind" if relative_translation[axis] > 0 else "in front"


def relation_text(
    delta_camera: np.ndarray,
    distance: float,
    previous_distance: float | None,
    dt: float,
    bins,
    approach_speed_m_s: float,
) -> str:
    """末端相对物体的方向/远近/趋离, 与 `encoding.py:333-345` 同一套词表。"""
    motion = "steady"
    if previous_distance is not None and dt > 0:
        speed = (distance - previous_distance) / max(dt, 1e-6)
        if speed < -approach_speed_m_s:
            motion = "approaching"
        elif speed > approach_speed_m_s:
            motion = "leaving"
    return f"{_distance_word(distance, bins)} {_direction_word(delta_camera)} {motion}"


# ---------------------------------------------------------------- 相对位姿


def relative_poses(T_camera0_object: np.ndarray, T_camera0_midpoint: np.ndarray):
    """照 `encoding.py:474-491`。返回 (T_object_right_midpoint, T_right_midpoint_object)。

    **物体轴恒在**: `T_camera0_object` 是 `(T, N, 4, 4)` (N=1 时也是), 手的位姿是
    `(T, 4, 4)` —— 只有一只手, 所以"逐物体的相对位姿"就是**同一个**手位姿与每个物体各算
    一次。参考是"手 × 物体"的外积, 我们这里退化成物体轴一维 (见 README 的范围说明)。

    单独喂 `(T, 4, 4)` 的老产物也认 (自动补一个物体轴), 于是老 npz 读进来还是 N=1。
    """
    from ego_relation.contracts.se3 import compose, invert

    objects = np.asarray(T_camera0_object, dtype=np.float64)
    if objects.ndim == 3:
        objects = objects[:, None]
    n, n_obj = objects.shape[:2]
    forward = np.empty((n, n_obj, 4, 4), dtype=np.float64)
    backward = np.empty((n, n_obj, 4, 4), dtype=np.float64)
    for frame in range(n):
        hand = np.asarray(T_camera0_midpoint[frame], dtype=np.float64)
        hand_inv = invert(hand)  # 逐物体同一个逆, 提到物体循环外面
        for j in range(n_obj):
            forward[frame, j] = compose(invert(objects[frame, j]), hand)
            backward[frame, j] = compose(hand_inv, objects[frame, j])
    return forward, backward


def _instance_table(archive) -> tuple[list[str], list[str]]:
    """npz 里的物体表: (有序的 `obj1..objN`, 对应类别)。

    step2 多物体后 `objectpose.save` 写的是数组; 老产物 (单物体) 没有这两个键 —— 退回
    规范里的第一个物体, 于是老 npz 读进来仍是 N=1、逐位同旧行为。
    """
    if "instance_ids" in archive.files:
        ids = [str(v) for v in np.atleast_1d(archive["instance_ids"])]
    else:
        ids = [INSTANCE_ID]
    if "categories" in archive.files:
        categories = [str(v) for v in np.atleast_1d(archive["categories"])]
        if len(categories) != len(ids):
            categories = (categories + [INSTANCE_ID] * len(ids))[: len(ids)]
    else:
        categories = list(ids)
    return ids, categories


def run(paths: RelPaths, hands, *, verbose: bool = True, frame_print: bool = False,
        frames=None) -> dict:
    """算相对位姿并落 `relation/rel_<stem>.npz` + `relation/relation_report.json`。

    `hands` = `hand_states()` 的返回值 (由 CLI 算一次, pose 与 relation 两个 stage 共用,
    保证锁存用的手位姿与这里报出来的手位姿是同一个)。
    """
    with np.load(paths.object_npz) as archive:
        T_camera0_object = archive["T_camera0_object"].astype(np.float64)
        object_valid = archive["valid"].astype(bool)
        object_observed = archive["observed"].astype(bool)
        latched = archive["latched"].astype(bool)
        confidence = archive["confidence"].astype(np.float32)
        residual_m = archive["residual_m"].astype(np.float32)
        instance_ids, categories = _instance_table(archive)

    # 老产物是 (T,4,4) / (T,), 补一个长度为 1 的物体轴 -> 后面的代码只认 (T,N,...)
    if T_camera0_object.ndim == 3:
        T_camera0_object = T_camera0_object[:, None]

    def _object_axis(value):
        value = np.asarray(value)
        return value[:, None] if value.ndim == 1 else value

    object_valid = _object_axis(object_valid)
    object_observed = _object_axis(object_observed)
    latched = _object_axis(latched)
    confidence = _object_axis(confidence)
    residual_m = _object_axis(residual_m)

    T_camera0_midpoint = hands["T_camera0_midpoint"]
    n, n_obj = T_camera0_object.shape[:2]
    if len(T_camera0_midpoint) != n:
        raise ValueError(f"物体位姿 {n} 帧与手 {len(T_camera0_midpoint)} 帧不一致")
    if n_obj != len(instance_ids):
        raise ValueError(f"物体位姿有 {n_obj} 个物体, 但 instance_ids 是 {instance_ids}")

    forward, backward = relative_poses(T_camera0_object, T_camera0_midpoint)
    # 两端都有效才是一次可用的相对位姿 (逐物体: 物体有效 ∧ 手有效)
    valid = object_valid & hands["valid"][:, None]

    cfg, _ = _config()
    bins = cfg.relations.distance_bins_m
    approach = cfg.relations.approach_speed_m_s
    fps = 1.0 / max(float(_fps(paths)), 1e-9)

    # 自然语言描述逐物体各算一条 (同一个手, 所以只有物体那端不同)
    texts: list[list[str]] = []
    distances = np.full((n, n_obj), np.nan, dtype=np.float64)
    for j in range(n_obj):
        rows: list[str] = []
        previous: float | None = None
        for frame in range(n):
            if not valid[frame, j]:
                rows.append("")
                continue
            delta = T_camera0_midpoint[frame][:3, 3] - T_camera0_object[frame, j][:3, 3]
            distance = float(np.linalg.norm(delta))
            distances[frame, j] = distance
            rows.append(relation_text(delta, distance, previous, fps, bins, approach))
            previous = distance
        texts.append(rows)

    np.savez_compressed(
        paths.relation_npz,
        instance_ids=np.asarray(instance_ids),
        categories=np.asarray(categories),
        frame_index=np.arange(n, dtype=np.int32),
        timestamp_ns=hands["timestamps_ns"],
        valid=valid,
        object_valid=object_valid,
        object_observed=object_observed,
        hand_valid=hands["valid"],
        confidence=confidence,
        residual_m=residual_m,
        latched=latched,
        grasp_closed=hands["closed"],
        distance_m=distances.astype(np.float32),
        T_camera0_world=hands["T_camera0_world"],
        T_camera0_object=T_camera0_object,
        T_camera0_right_midpoint=T_camera0_midpoint,
        T_object_right_midpoint=forward,
        T_right_midpoint_object=backward,
    )

    report = _report(paths, hands, valid, forward, distances, texts, latched,
                     object_observed, fps, instance_ids, categories)
    paths.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    if verbose:
        _print_summary(report)
    if frame_print:
        # 「实时打印相对位姿」的终端那一路 (渲染时逐帧同步打); 逐物体各一行
        wanted = set(int(f) for f in frames) if frames else None
        for frame in range(n):
            if wanted is not None and frame not in wanted:
                continue
            for j, instance_id in enumerate(instance_ids):
                tag = f"{instance_id} " if n_obj > 1 else ""
                if not valid[frame, j]:
                    print(f"  f={frame:04d} {tag}REL 无效 "
                          f"(物体可见={int(object_observed[frame, j])} "
                          f"手有效={int(hands['valid'][frame])})")
                    continue
                _print_frame(frame, forward[frame, j], object_observed[frame, j],
                             latched[frame, j], prefix=tag)
    if verbose:
        print(f"    -> {paths.relation_npz}  +  {paths.report_json}")
    return {
        "valid": valid,
        "T_object_right_midpoint": forward,
        "distances": distances,
        "texts": texts,
        "report": report,
    }


def _fps(paths: RelPaths) -> float:
    from xrhand.video import probe

    return float(probe(str(paths.mp4))["fps"])


def _config():
    from .adapter import load_ego_config

    return load_ego_config()


def _rel_rpy(transform: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(transform[:3, :3]).as_euler("xyz", degrees=True)


def _print_frame(frame: int, T_object_mid: np.ndarray, observed, latched,
                 *, prefix: str = "") -> None:
    t = T_object_mid[:3, 3]
    rpy = _rel_rpy(T_object_mid)
    print(
        f"  f={frame:04d} {prefix}t_rel=[{t[0]:+.4f} {t[1]:+.4f} {t[2]:+.4f}] m  "
        f"RPY=[{rpy[0]:+7.2f} {rpy[1]:+7.2f} {rpy[2]:+7.2f}] deg  "
        f"vis={int(observed)} latched={int(latched)}"
    )


def _runs(frames) -> list[list[int]]:
    frames = sorted(int(f) for f in frames)
    if not frames:
        return []
    out = [[frames[0], frames[0]]]
    for frame in frames[1:]:
        if frame == out[-1][1] + 1:
            out[-1][1] = frame
        else:
            out.append([frame, frame])
    return out


def _object_report(valid, forward, distances, texts, latched, observed, instance_id) -> dict:
    """单个物体的统计块。`valid`/`forward`/… 都是已经切好的 (n,) / (n,4,4)。"""
    n = len(valid)
    finite = np.isfinite(distances)
    rel_t = np.linalg.norm(forward[:, :3, 3], axis=1)
    # 手推帧 (latched: 本帧没有测量, 位姿由「手 x T_hand_object」推出) 的相对位姿应近似恒定
    # —— 帧间变化量是最直接的证据。注意这几帧**不是**一次测量, 别拿它当精度的证明。
    latched_steps = np.linalg.norm(np.diff(forward[latched][:, :3, 3], axis=0), axis=1) if latched.sum() > 1 else np.zeros(0)
    unlatched = valid & ~latched
    unlatched_steps = (
        np.linalg.norm(np.diff(forward[unlatched][:, :3, 3], axis=0), axis=1)
        if unlatched.sum() > 1
        else np.zeros(0)
    )
    return {
        "instance_id": str(instance_id),
        "valid_frames": int(valid.sum()),
        "valid_ratio": float(valid.mean()),
        "object_observed_frames": int(observed.sum()),
        "latched_frames": int(latched.sum()),
        "object_no_observation_runs": _runs(np.nonzero(~observed)[0]),
        "invalid_relative_runs": _runs(np.nonzero(~valid)[0]),
        "distance_m": {
            "median": float(np.nanmedian(distances)) if finite.any() else None,
            "p10": float(np.nanpercentile(distances, 10)) if finite.any() else None,
            "p90": float(np.nanpercentile(distances, 90)) if finite.any() else None,
            "min": float(np.nanmin(distances)) if finite.any() else None,
            "max": float(np.nanmax(distances)) if finite.any() else None,
        },
        "relative_translation_norm_m": {
            "median": float(np.median(rel_t[valid])) if valid.any() else None,
        },
        "latch_stability_m_per_frame": {
            "latched_median": float(np.median(latched_steps)) if len(latched_steps) else None,
            "unlatched_median": float(np.median(unlatched_steps)) if len(unlatched_steps) else None,
        },
        "relation_text_samples": {
            str(frame): texts[frame]
            for frame in np.linspace(0, n - 1, min(6, n)).astype(int)
            if texts[frame]
        },
    }


def _report(paths, hands, valid, forward, distances, texts, latched, observed, fps,
            instance_ids, categories) -> dict:
    """整份报告: **逐物体一块** (`objects`), 顶层是"所有物体一起算数"的约简。

    顶层那几个计数用 `all(axis=1)` / `any(axis=1)` 约简 —— N=1 时与逐物体的数字逐位相同,
    N>1 时含义写进 `reduction` 那行, 免得"274 帧有效"被误读成"每个物体都有效"。
    注意 step2 的 `state_valid` 用的正是同一条严格口径 (全部物体有效 ∧ 手有效)。
    """
    n = len(valid)
    objects = {
        str(instance_id): _object_report(
            valid[:, j], forward[:, j], distances[:, j], texts[j], latched[:, j],
            observed[:, j], instance_id,
        )
        for j, instance_id in enumerate(instance_ids)
    }
    det_forward = np.linalg.det(forward[:, :, :3, :3])
    finite = np.isfinite(distances)
    rel_t = np.linalg.norm(forward[:, :, :3, 3], axis=2)
    every_valid = valid.all(axis=1)
    every_observed = observed.all(axis=1)
    return {
        "stem": paths.stem,
        "frames": int(n),
        "fps": float(fps),
        "instance_ids": [str(v) for v in instance_ids],
        "categories": [str(v) for v in categories],
        "reduction": "顶层计数: 有效/有观测 = 全部物体 (all), 手推 = 任一个物体 (any); "
                     "N=1 时与 objects 里那一块相同",
        "relative_pose": "T_object_right_midpoint = inv(T_camera0_object) @ T_camera0_right_midpoint",
        "eef_frame": "HumanEgo MidpointFrameBuilder (原点=两指尖中点, x=食指根-拇指根)",
        "camera_frame": "camera0 = 左眼 (eye0), 与 SAM2 mask / 深度同一个像素系",
        "valid_frames": int(every_valid.sum()),
        "valid_ratio": float(every_valid.mean()),
        "object_observed_frames": int(every_observed.sum()),
        "latched_frames": int(latched.any(axis=1).sum()),
        "object_no_observation_runs": _runs(np.nonzero(~every_observed)[0]),
        "invalid_relative_runs": _runs(np.nonzero(~every_valid)[0]),
        "distance_m": {
            "median": float(np.nanmedian(distances)) if finite.any() else None,
            "p10": float(np.nanpercentile(distances, 10)) if finite.any() else None,
            "p90": float(np.nanpercentile(distances, 90)) if finite.any() else None,
            "min": float(np.nanmin(distances)) if finite.any() else None,
            "max": float(np.nanmax(distances)) if finite.any() else None,
        },
        "relative_translation_norm_m": {
            "median": float(np.median(rel_t[valid])) if valid.any() else None,
        },
        "det_rotation_min_max": [float(det_forward.min()), float(det_forward.max())],
        "grasp": {
            "closed_frames": int(hands["closed"].sum()),
            "closed_runs": _runs(np.nonzero(hands["closed"])[0]),
            "calibration": hands["grasp_calibration"],
        },
        "objects": objects,
    }


def _print_summary(report) -> None:
    for instance_id, block in report["objects"].items():
        print(f"    [{instance_id}] 相对位姿: {block['valid_frames']}/{report['frames']} 帧有效 "
              f"(手推 {block['latched_frames']} 帧 —— 那几帧物体位姿由手推得, 不是测量; 物体无观测 "
              f"{len(block['object_no_observation_runs'])} 段)")
        distance = block["distance_m"]
        if distance["median"] is not None:
            print(f"    [{instance_id}] |t| = 末端到物体质心: 中位 {distance['median']:.4f} m  "
                  f"[p10 {distance['p10']:.4f}, p90 {distance['p90']:.4f}]")
        stability = block["latch_stability_m_per_frame"]
        if stability["latched_median"] is not None:
            print(f"    [{instance_id}] 手推段帧间位移中位 {stability['latched_median']:.5f} m "
                  f"(其余有效帧 {stability['unlatched_median']:.5f} m)")
    first = next(iter(report["objects"].values()))
    for frame, text in first["relation_text_samples"].items():
        print(f"    f={frame}: {text}")
