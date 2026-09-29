"""mask + 深度 -> 左眼相机系点云。

反投影的三行来自 HumanEgo `DepthLifter.py:112` `unproject_pixel`:

    X = (u - cx) * Z / fx ,  Y = (v - cy) * Z / fy ,  Z = depth

它正是 `xrhand.camera.Projector.cam_to_pixel` 的逆 (`u = f X/Z + cx`), 所以这条链
与 step1 的手部投影自洽 —— 前提是**用原左眼的 K**, 见下。

**必须用原左眼 K, 不能用 `K_rectified`**: 深度经他 `_restore_original_left_depth` 已经
映回**原左图像素** (他注释原话 "This preserves alignment with HumanEgo masks and
keypoints"), 而我们的 SAM2 mask 也是在原左眼像素上算的 (step2 只跑左半)。三者
(mask / K / depth) 同像素系, 反投影才落回同一个相机系。拿 `K_rectified` 会把点云整体
挪位 (两眼已平行时 R_left≈I, 只有 cx 上的一点差, 但那是系统性偏移)。

丢边缘: mask 沿轮廓腐蚀几像素 —— 物体边缘的视差最脏 (前景/背景混在一个 SGBM 块里),
而且 SAM2 的边缘本身就有 1~2 px 抖动。
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from . import EYE_CX, EYE_CY, EYE_F, INSTANCE_ID, RelPaths

# 腐蚀核半径 (像素)。默认 3 = 与 ego_relation_policy `configs/default.yaml` 的
# `depth.patch_radius_px` 同一个量级。
DEFAULT_ERODE_PX = 3
DEFAULT_STRIDE = 4


def _runs(frames: list[int]) -> list[list[int]]:
    """连续帧号压成 [起, 止] 区间, 报告里写区间比罗列一长串帧号可读。"""
    if not frames:
        return []
    ordered = sorted(int(f) for f in frames)
    out = [[ordered[0], ordered[0]]]
    for frame in ordered[1:]:
        if frame == out[-1][1] + 1:
            out[-1][1] = frame
        else:
            out.append([frame, frame])
    return out


def unproject_left(uv: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
    """原左眼像素 + 沿光轴的深度 -> 左眼相机系点 (逆投影, HumanEgo DepthLifter.py:112)。"""
    uv = np.asarray(uv, dtype=np.float64)
    depth_m = np.asarray(depth_m, dtype=np.float64)
    x = (uv[:, 0] - EYE_CX) * depth_m / EYE_F
    y = (uv[:, 1] - EYE_CY) * depth_m / EYE_F
    return np.stack([x, y, depth_m], axis=1)


def project_camera0(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`unproject_left` 的**严格逆**: 左眼相机系点 -> 原左眼像素 (u, v) + z。

    画图用它, 而不是 `Projector.cam_to_pixel`: 点云是用**标称 K** 反投影出来的, 画回去
    也必须用同一个 K, 否则点云和 mask 会差一点点。`Projector` 用的是 `eff_f = f + d_f`
    (标定增量); 当前 `out/calib.json` 里 d_f=d_cx=d_cy=0, 两者逐位相同 —— 上面对不上时
    渲染阶段会显式警告, 而不是默默画偏。
    """
    points = np.asarray(points, dtype=np.float64)
    z = points[:, 2]
    safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = EYE_F * points[:, 0] / safe + EYE_CX
    v = EYE_F * points[:, 1] / safe + EYE_CY
    return u, v


def load_mask(paths: RelPaths, frame: int, instance_id: str | None = None) -> np.ndarray | None:
    """读 step2 的 mask (uint8, {0,255}, 810x1080)。文件不存在返回 None。

    `instance_id=None` = 规范里的第一个物体 (obj1), 与旧行为的单物体一致。
    """
    path = paths.seg_mask(frame, instance_id) if instance_id else paths.seg_mask(frame)
    if not path.is_file():
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return mask


def load_depth_m(paths: RelPaths, frame: int) -> np.ndarray | None:
    """读深度 (他同格式: uint16 毫米, 无效 0) -> float32 米 (无效仍是 0)。"""
    path = paths.depth(frame)
    if not path.is_file():
        return None
    mm = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mm is None:
        return None
    return mm.astype(np.float32) / 1000.0


def frame_cloud(
    mask: np.ndarray,
    depth_m: np.ndarray,
    *,
    erode_px: int = DEFAULT_ERODE_PX,
    stride: int = DEFAULT_STRIDE,
) -> dict:
    """一帧的 mask 点云。

    返回 dict: points/uvs/valid/area_px/median_depth_m/n_points/bad_depth_px
    `valid=False` 表示这一帧没有可用观测 (空 mask, 或 mask 内没有任何有效深度),
    调用方如实记录, 不硬凑。
    """
    support = mask > 127
    area_px = int(support.sum())
    empty = {
        "points": np.zeros((0, 3), np.float32),
        "uvs": np.zeros((0, 2), np.float32),
        "valid": False,
        "area_px": area_px,
        "median_depth_m": float("nan"),
        "n_points": 0,
        "observed_px": 0,
    }
    if area_px == 0:
        return empty

    if erode_px > 0:
        size = 2 * int(erode_px) + 1
        kernel = np.ones((size, size), dtype=np.uint8)
        eroded = cv2.erode(support.astype(np.uint8), kernel) > 0
        if eroded.sum() < 16:  # 腐蚀没了就退回原 mask (物体太小时别把点全丢掉)
            eroded = support

    # 按 stride 抽稀 (每 stride 行/列取一个), 免得 274 帧全稠密点云爆内存
    step = max(1, int(stride))
    grid = np.zeros_like(eroded)
    grid[::step, ::step] = True
    pick = eroded & grid

    vs, us = np.nonzero(pick)
    if us.size == 0:
        return empty
    uv = np.stack([us, vs], axis=1).astype(np.float64)
    z = depth_m[vs, us].astype(np.float64)
    good = z > 0.0
    if not good.any():
        return {**empty, "observed_px": int(eroded.sum())}

    points = unproject_left(uv[good], z[good]).astype(np.float32)
    return {
        "points": points,
        "uvs": uv[good].astype(np.float32),
        "valid": True,
        "area_px": area_px,
        "median_depth_m": float(np.median(z[good])),
        "n_points": int(points.shape[0]),
        "observed_px": int(eroded.sum()),
    }


def build_clouds(
    paths: RelPaths, frames, *, erode_px: int = DEFAULT_ERODE_PX, stride: int = DEFAULT_STRIDE,
    verbose: bool = True, instance_ids: Sequence[str] | None = None,
) -> dict:
    """逐帧点云 -> 一个扁平打包的 npz (点云用 offset 索引, 不逐帧存对象数组)。

    多物体: `instance_ids` 给几个就产出几份 `clouds_<instance_id>.npz`, 每次调用只算
    指定的那几个; 不给就是规范里的第一个物体 (obj1), 与旧行为逐位一致。返回的 dict
    在单物体时和以前一样 (顶层就是那份 payload), 多物体时另加 `"objects"` 逐物体一份。
    """
    ids = [str(v) for v in (instance_ids or [INSTANCE_ID])]
    if len(ids) == 1:
        return _build_one(paths, frames, instance_id=ids[0], erode_px=erode_px,
                          stride=stride, verbose=verbose)
    payloads = {
        instance_id: _build_one(paths, frames, instance_id=instance_id, erode_px=erode_px,
                                stride=stride, verbose=verbose)
        for instance_id in ids
    }
    first = payloads[ids[0]]
    return {**first, "objects": payloads}


def _build_one(
    paths: RelPaths, frames, *, instance_id: str, erode_px: int, stride: int, verbose: bool,
) -> dict:
    frames = [int(f) for f in frames]
    chunks: list[np.ndarray] = []
    uvs: list[np.ndarray] = []
    offsets = [0]
    valid = np.zeros(len(frames), dtype=bool)
    area_px = np.zeros(len(frames), dtype=np.int32)
    median_depth = np.full(len(frames), np.nan, dtype=np.float32)
    n_points = np.zeros(len(frames), dtype=np.int32)
    missing_mask: list[int] = []
    depth_missing: list[int] = []
    empty_mask: list[int] = []

    for i, frame in enumerate(frames):
        mask = load_mask(paths, frame, instance_id)
        if mask is None:
            missing_mask.append(frame)
            offsets.append(offsets[-1])
            continue
        if not bool((mask > 127).any()):
            # 空 mask (step2 的无观测帧: 文件在, 但是全 0 图) —— 不需要深度,
            # 也不记进 depth_missing (否则 depth_missing 会被无观测帧灌满, 失去意义)
            empty_mask.append(frame)
            offsets.append(offsets[-1])
            continue
        depth_m = load_depth_m(paths, frame)
        if depth_m is None:
            depth_missing.append(frame)
            depth_m = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.float32)

        info = frame_cloud(mask, depth_m, erode_px=erode_px, stride=stride)
        valid[i] = info["valid"]
        area_px[i] = info["area_px"]
        median_depth[i] = info["median_depth_m"]
        n_points[i] = info["n_points"]
        if info["n_points"]:
            chunks.append(info["points"])
            uvs.append(info["uvs"])
        offsets.append(offsets[-1] + info["n_points"])

    points = np.concatenate(chunks) if chunks else np.zeros((0, 3), np.float32)
    uv_all = np.concatenate(uvs) if uvs else np.zeros((0, 2), np.float32)
    payload = {
        "instance_id": np.asarray(instance_id),
        "frame_index": np.asarray(frames, dtype=np.int32),
        "points": points.astype(np.float32),
        "uvs": uv_all.astype(np.float32),
        "offset": np.asarray(offsets, dtype=np.int64),
        "valid": valid,
        "area_px": area_px,
        "median_depth_m": median_depth,
        "n_points": n_points,
    }
    if verbose:
        print(
            f"    [{instance_id}] 点云: {int(valid.sum())}/{len(frames)} 帧有观测, "
            f"共 {points.shape[0]} 点 (stride={stride}, erode={erode_px}px)"
        )
        if empty_mask:
            print(f"    [{instance_id}] 空 mask (step2 无观测): "
                  f"{len(empty_mask)} 帧 {_runs(empty_mask)[:6]}")
        if missing_mask:
            print(f"    [{instance_id}] [warn] {len(missing_mask)} 帧缺 mask 文件: "
                  f"{missing_mask[:8]}…")
        if depth_missing:
            print(f"    [{instance_id}] [warn] {len(depth_missing)} 帧缺深度文件: "
                  f"{depth_missing[:8]}…")
    np.savez_compressed(paths.clouds_npz_for(instance_id), **payload)
    return {
        **payload,
        "missing_mask": missing_mask,
        "depth_missing": depth_missing,
        "empty_mask": empty_mask,
    }


def load_clouds(paths: RelPaths, instance_id: str | None = None) -> dict:
    with np.load(paths.clouds_npz_for(instance_id or INSTANCE_ID)) as handle:
        return {key: handle[key] for key in handle.files}


def cloud_at(clouds: dict, i: int) -> np.ndarray:
    """第 i 个 (在 frame_index 里的位置) 的点云 (N,3)。"""
    start, end = int(clouds["offset"][i]), int(clouds["offset"][i + 1])
    return clouds["points"][start:end]
