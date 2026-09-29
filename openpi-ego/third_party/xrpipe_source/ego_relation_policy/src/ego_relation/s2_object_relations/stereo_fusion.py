from __future__ import annotations

import json
from itertools import permutations, product
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from ego_relation.config import ProjectConfig
from ego_relation.contracts.se3 import compose, transform_points


def _proper_cube_symmetries() -> tuple[np.ndarray, ...]:
    rotations = []
    for permutation in permutations(range(3)):
        for signs in product((-1.0, 1.0), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.float64)
            matrix[np.arange(3), permutation] = signs
            if np.linalg.det(matrix) > 0.5:
                rotations.append(matrix)
    return tuple(rotations)


_CUBE_SYMMETRIES = _proper_cube_symmetries()


def _sample_depth(depth: np.ndarray, points_uv: np.ndarray, radius: int) -> np.ndarray:
    height, width = depth.shape
    values = np.zeros(len(points_uv), dtype=np.float64)
    for index, (u, v) in enumerate(points_uv):
        x, y = int(round(float(u))), int(round(float(v)))
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        patch = depth[y0:y1, x0:x1]
        valid = patch[patch > 0]
        if len(valid):
            values[index] = float(np.median(valid))
    return values


def _unproject(K: np.ndarray, uv: np.ndarray, depth: np.ndarray) -> np.ndarray:
    x = (uv[:, 0] - K[0, 2]) / K[0, 0] * depth
    y = (uv[:, 1] - K[1, 2]) / K[1, 1] * depth
    return np.stack([x, y, depth], axis=1)


def _rigid_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vh = np.linalg.svd(covariance)
    rotation = vh.T @ u.T
    if np.linalg.det(rotation) < 0:
        vh[-1] *= -1
        rotation = vh.T @ u.T
    translation = target_center - rotation @ source_center
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    residual = np.linalg.norm(transform_points(transform, source) - target, axis=1)
    return transform, residual


def robust_rigid_fit(source: np.ndarray, target: np.ndarray, threshold_m: float = 0.035):
    transform, residual = _rigid_fit(source, target)
    keep = residual <= max(threshold_m, float(np.median(residual) * 2.5))
    if keep.sum() >= 3 and keep.sum() < len(source):
        transform, residual_kept = _rigid_fit(source[keep], target[keep])
        residual = np.linalg.norm(transform_points(transform, source) - target, axis=1)
        keep = residual <= threshold_m
    return transform, residual, keep


def _points_inside_mask(mask: np.ndarray, points_uv: np.ndarray, dilation_px: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.uint8)
    if dilation_px > 0:
        size = 2 * dilation_px + 1
        mask = cv2.dilate(mask, np.ones((size, size), dtype=np.uint8))
    points = np.rint(np.asarray(points_uv, dtype=np.float64)).astype(np.int32)
    height, width = mask.shape
    inside_image = (
        (points[:, 0] >= 0)
        & (points[:, 0] < width)
        & (points[:, 1] >= 0)
        & (points[:, 1] < height)
    )
    supported = np.zeros(len(points), dtype=bool)
    supported[inside_image] = mask[points[inside_image, 1], points[inside_image, 0]] > 127
    return supported


def _rotation_step_deg(previous: np.ndarray, candidate: np.ndarray) -> float:
    relative = candidate[:3, :3] @ previous[:3, :3].T
    return float(np.degrees(Rotation.from_matrix(relative).magnitude()))


def _nearest_symmetric_rotation(previous: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    equivalent = [candidate @ symmetry for symmetry in _CUBE_SYMMETRIES]
    distances = [
        Rotation.from_matrix(previous.T @ rotation).magnitude()
        for rotation in equivalent
    ]
    return equivalent[int(np.argmin(distances))]


def _smooth_pose(
    previous: np.ndarray,
    candidate: np.ndarray,
    *,
    translation_alpha: float,
    rotation_alpha: float,
) -> np.ndarray:
    result = np.asarray(candidate, dtype=np.float64).copy()
    result[:3, 3] = (
        (1.0 - translation_alpha) * previous[:3, 3]
        + translation_alpha * candidate[:3, 3]
    )
    relative = Rotation.from_matrix(previous[:3, :3].T @ candidate[:3, :3])
    result[:3, :3] = previous[:3, :3] @ Rotation.from_rotvec(
        rotation_alpha * relative.as_rotvec()
    ).as_matrix()
    return result


def _adaptive_translation_alpha(
    base_alpha: float,
    keypoint_motion_px: float,
    motion_deadband_px: float,
    full_response_px: float,
) -> float:
    response_range = full_response_px - motion_deadband_px
    if not np.isfinite(keypoint_motion_px) or response_range <= 0:
        return float(base_alpha)
    motion_ratio = np.clip(
        (keypoint_motion_px - motion_deadband_px) / response_range,
        0.0,
        1.0,
    )
    return float(base_alpha + (1.0 - base_alpha) * motion_ratio)


def _adaptive_translation_measurement(
    recent_translations: list[np.ndarray],
    motion_ratio: float,
) -> np.ndarray:
    latest = recent_translations[-1]
    robust_median = np.median(np.stack(recent_translations), axis=0)
    blend = float(np.clip(motion_ratio, 0.0, 1.0))
    return (1.0 - blend) * robust_median + blend * latest


def _load_sam2_gate(episode_dir: Path, instance_ids: np.ndarray, frame_count: int):
    metrics_path = episode_dir / "qa" / "sam2_video_metrics.npz"
    mask_root = (
        episode_dir
        / "perception"
        / "humanego_session"
        / "preprocess"
        / "sam2_video_masks"
    )
    if not metrics_path.is_file() or not mask_root.is_dir():
        return None
    with np.load(metrics_path, allow_pickle=False) as archive:
        metric_ids = [str(value) for value in archive["instance_ids"]]
        areas = archive["areas"][:frame_count]
        scores = archive["scores"][:frame_count]
    requested = [str(value) for value in instance_ids]
    if len(areas) < frame_count or any(instance_id not in metric_ids for instance_id in requested):
        return None
    indices = [metric_ids.index(instance_id) for instance_id in requested]
    return {
        "mask_root": mask_root,
        "areas": areas[:, indices],
        "scores": scores[:, indices],
    }


def fuse_stereo_tracks(
    cfg: ProjectConfig,
    episode_dir: str | Path,
    *,
    sam2_gate_enabled: bool | None = None,
) -> Path:
    """CoTracker 身份对应 + 每帧双目深度，估计相对初始帧的 3D 刚体变化。"""
    episode_dir = Path(episode_dir).resolve()
    preprocess = episode_dir / "perception" / "humanego_session" / "preprocess"
    tracks_data = json.loads((preprocess / "cotracker_results.json").read_text(encoding="utf-8"))
    with np.load(episode_dir / "entities" / "objects_initial.npz", allow_pickle=False) as archive:
        instance_ids = archive["instance_ids"]
        categories = archive["categories"]
        initial_poses = archive["T_camera0_object"].astype(np.float64)
    K = np.load(episode_dir / "camera" / "K.npy")
    camera_poses = np.load(episode_dir / "camera" / "T_camera_to_camera0.npy")
    depth_dir = episode_dir / "depth" / "stereo_mm"
    if not depth_dir.is_dir():
        raise FileNotFoundError(f"缺少双目深度 {depth_dir}，稳定方案要求先运行 depth")

    frame_count = len(camera_poses)
    object_count = len(instance_ids)
    maximum_keypoints = max(
        len(tracks_data[str(instance_id)]["tracks"][0])
        for instance_id in instance_ids
    )
    poses = np.repeat(initial_poses[None], frame_count, axis=0)
    valid = np.zeros((frame_count, object_count), dtype=bool)
    confidence = np.zeros((frame_count, object_count), dtype=np.float32)
    residual_m = np.full((frame_count, object_count), np.nan, dtype=np.float32)
    keypoint_count = np.zeros((frame_count, object_count), dtype=np.int16)
    keypoint_accepted = np.zeros(
        (frame_count, object_count, maximum_keypoints),
        dtype=bool,
    )
    sam2_supported_count = np.full((frame_count, object_count), -1, dtype=np.int16)
    sam2_gate_reliable = np.zeros((frame_count, object_count), dtype=bool)
    temporal_rejected = np.zeros((frame_count, object_count), dtype=bool)
    rotation_rejected = np.zeros((frame_count, object_count), dtype=bool)
    measurement_translation_step_m = np.full(
        (frame_count, object_count),
        np.nan,
        dtype=np.float32,
    )
    effective_translation_alpha = np.full(
        (frame_count, object_count),
        cfg.perception.object_translation_smoothing,
        dtype=np.float32,
    )
    keypoint_motion_px = np.full(
        (frame_count, object_count),
        np.nan,
        dtype=np.float32,
    )
    if sam2_gate_enabled is None:
        sam2_gate_enabled = cfg.perception.sam2_pose_gate_enabled
    sam2_gate = (
        _load_sam2_gate(episode_dir, instance_ids, frame_count)
        if sam2_gate_enabled
        else None
    )

    for object_index, instance_id_value in enumerate(instance_ids):
        instance_id = str(instance_id_value)
        tracks = np.asarray(tracks_data[instance_id]["tracks"], dtype=np.float64)[:frame_count]
        visibility = np.asarray(tracks_data[instance_id]["visibility"], dtype=np.float64)[:frame_count]
        points_by_frame: list[np.ndarray] = []
        point_valid_by_frame: list[np.ndarray] = []
        for frame in range(frame_count):
            depth_mm = cv2.imread(str(depth_dir / f"{frame:05d}.png"), cv2.IMREAD_UNCHANGED)
            if depth_mm is None:
                raise FileNotFoundError(depth_dir / f"{frame:05d}.png")
            depths = _sample_depth(depth_mm.astype(np.float32) / 1000.0, tracks[frame], cfg.depth.patch_radius_px)
            point_valid = (visibility[frame] >= cfg.perception.visibility_threshold) & (depths > 0)
            if sam2_gate is not None:
                area = int(sam2_gate["areas"][frame, object_index])
                initial_area = max(int(sam2_gate["areas"][0, object_index]), 1)
                area_ratio = area / initial_area
                mask_reliable = bool(
                    cfg.perception.sam2_pose_gate_minimum_area_ratio
                    <= area_ratio
                    <= cfg.perception.sam2_pose_gate_maximum_area_ratio
                    and sam2_gate["scores"][frame, object_index]
                    >= cfg.perception.sam2_pose_gate_minimum_score
                )
                if mask_reliable:
                    mask_path = (
                        sam2_gate["mask_root"] / instance_id / f"{frame:05d}.png"
                    )
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        raise FileNotFoundError(mask_path)
                    mask_support = _points_inside_mask(
                        mask,
                        tracks[frame],
                        cfg.perception.sam2_pose_gate_dilation_px,
                    )
                    sam2_supported_count[frame, object_index] = int(
                        np.count_nonzero(point_valid & mask_support)
                    )
                    sam2_gate_reliable[frame, object_index] = True
                    point_valid &= mask_support
                else:
                    sam2_supported_count[frame, object_index] = 0
                    point_valid[:] = False
            keypoint_count[frame, object_index] = int(point_valid.sum())
            keypoint_accepted[frame, object_index, : len(point_valid)] = point_valid
            points_camera = _unproject(K, tracks[frame], depths)
            points_camera0 = transform_points(camera_poses[frame], points_camera)
            points_by_frame.append(points_camera0)
            point_valid_by_frame.append(point_valid)

        reference_frame = next(
            (
                frame
                for frame, point_valid in enumerate(point_valid_by_frame)
                if int(point_valid.sum()) >= cfg.depth.minimum_keypoint_depths
            ),
            None,
        )
        if reference_frame is None:
            continue
        reference_points = points_by_frame[reference_frame]
        reference_valid = point_valid_by_frame[reference_frame]
        last_pose = initial_poses[object_index].copy()
        last_measurement_pose: np.ndarray | None = None
        last_measurement_frame: int | None = None
        recent_translations: list[np.ndarray] = []
        for frame in range(frame_count):
            common = reference_valid & point_valid_by_frame[frame]
            count = int(common.sum())
            if count < cfg.depth.minimum_keypoint_depths:
                poses[frame, object_index] = last_pose
                continue
            delta, residual, inliers = robust_rigid_fit(reference_points[common], points_by_frame[frame][common])
            inlier_count = int(inliers.sum())
            if (
                inlier_count < cfg.depth.minimum_keypoint_depths
                or inlier_count / count < cfg.perception.minimum_pose_inlier_ratio
            ):
                poses[frame, object_index] = last_pose
                continue
            candidate = compose(delta, initial_poses[object_index])
            if str(categories[object_index]) in cfg.perception.rotation_symmetry_categories:
                candidate[:3, :3] = _nearest_symmetric_rotation(
                    last_pose[:3, :3],
                    candidate[:3, :3],
                )
            if last_measurement_frame is not None and last_measurement_pose is not None:
                measurement_gap = max(frame - last_measurement_frame, 1)
                gap_scale = min(measurement_gap, 5)
                if measurement_gap == 1:
                    motion_points = (
                        point_valid_by_frame[last_measurement_frame]
                        & point_valid_by_frame[frame]
                    )
                    if int(motion_points.sum()) >= cfg.depth.minimum_keypoint_depths:
                        keypoint_motion_px[frame, object_index] = float(
                            np.median(
                                np.linalg.norm(
                                    tracks[frame, motion_points]
                                    - tracks[last_measurement_frame, motion_points],
                                    axis=1,
                                )
                            )
                        )
                translation_step = float(
                    np.linalg.norm(
                        candidate[:3, 3] - last_measurement_pose[:3, 3]
                    )
                )
                measurement_translation_step_m[frame, object_index] = translation_step
                rotation_step = _rotation_step_deg(last_measurement_pose, candidate)
                if translation_step > (
                    cfg.perception.maximum_object_translation_step_m * gap_scale
                ):
                    temporal_rejected[frame, object_index] = True
                    poses[frame, object_index] = last_pose
                    continue
                rotation_gap_scale = min(np.sqrt(gap_scale), 2.0)
                if rotation_step > (
                    cfg.perception.maximum_object_rotation_step_deg
                    * rotation_gap_scale
                ):
                    rotation_rejected[frame, object_index] = True
                    candidate[:3, :3] = last_pose[:3, :3]
            last_measurement_pose = candidate.copy()
            last_measurement_frame = frame
            recent_translations.append(candidate[:3, 3].copy())
            translation_window = max(
                int(cfg.perception.object_translation_median_window),
                1,
            )
            recent_translations = recent_translations[-translation_window:]
            translation_alpha = _adaptive_translation_alpha(
                cfg.perception.object_translation_smoothing,
                float(keypoint_motion_px[frame, object_index]),
                cfg.perception.object_translation_motion_deadband_px,
                cfg.perception.object_translation_full_response_px,
            )
            motion_ratio = (
                (translation_alpha - cfg.perception.object_translation_smoothing)
                / max(1.0 - cfg.perception.object_translation_smoothing, 1e-6)
            )
            candidate[:3, 3] = _adaptive_translation_measurement(
                recent_translations,
                motion_ratio,
            )
            effective_translation_alpha[frame, object_index] = translation_alpha
            candidate = _smooth_pose(
                last_pose,
                candidate,
                translation_alpha=translation_alpha,
                rotation_alpha=cfg.perception.object_rotation_smoothing,
            )
            last_pose = candidate
            poses[frame, object_index] = candidate
            valid[frame, object_index] = True
            confidence[frame, object_index] = float(inliers.sum() / count)
            residual_m[frame, object_index] = float(np.median(residual[inliers]))

    output = episode_dir / "entities" / "objects_stereo_track.npz"
    np.savez_compressed(
        output,
        instance_ids=instance_ids,
        T_camera0_object=poses,
        valid=valid,
        confidence=confidence,
        residual_m=residual_m,
        keypoint_count=keypoint_count,
        keypoint_accepted=keypoint_accepted,
        sam2_supported_count=sam2_supported_count,
        sam2_gate_reliable=sam2_gate_reliable,
        temporal_rejected=temporal_rejected,
        rotation_rejected=rotation_rejected,
        measurement_translation_step_m=measurement_translation_step_m,
        effective_translation_alpha=effective_translation_alpha,
        keypoint_motion_px=keypoint_motion_px,
    )
    report = {
        "frames": frame_count,
        "objects": object_count,
        "valid_ratio_per_object": {
            str(instance_ids[index]): float(valid[:, index].mean()) for index in range(object_count)
        },
        "median_residual_m_per_object": {
            str(instance_ids[index]): (
                float(np.nanmedian(residual_m[:, index])) if np.isfinite(residual_m[:, index]).any() else None
            )
            for index in range(object_count)
        },
        "method": (
            "CoTracker identity + SAM2 object-region gate + local median stereo depth + "
            "robust Kabsch delta + temporal SE(3) gate/smoothing + OA-v2 initial pose"
        ),
        "sam2_pose_gate_available": sam2_gate is not None,
        "sam2_pose_gate_reliable_ratio_per_object": {
            str(instance_ids[index]): float(sam2_gate_reliable[:, index].mean())
            for index in range(object_count)
        },
        "temporal_rejected_frames_per_object": {
            str(instance_ids[index]): int(temporal_rejected[:, index].sum())
            for index in range(object_count)
        },
        "rotation_rejected_frames_per_object": {
            str(instance_ids[index]): int(rotation_rejected[:, index].sum())
            for index in range(object_count)
        },
        "stabilization": {
            "minimum_pose_inlier_ratio": cfg.perception.minimum_pose_inlier_ratio,
            "maximum_translation_step_m": cfg.perception.maximum_object_translation_step_m,
            "maximum_rotation_step_deg": cfg.perception.maximum_object_rotation_step_deg,
            "translation_smoothing": cfg.perception.object_translation_smoothing,
            "translation_median_window": (
                cfg.perception.object_translation_median_window
            ),
            "translation_motion_deadband_px": (
                cfg.perception.object_translation_motion_deadband_px
            ),
            "translation_full_response_px": (
                cfg.perception.object_translation_full_response_px
            ),
            "rotation_smoothing": cfg.perception.object_rotation_smoothing,
            "rotation_symmetry_categories": list(
                cfg.perception.rotation_symmetry_categories
            ),
        },
    }
    (episode_dir / "qa" / "stereo_object_track_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output
