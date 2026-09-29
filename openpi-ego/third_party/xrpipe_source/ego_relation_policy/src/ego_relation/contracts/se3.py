from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


UNITY_TO_CV = np.diag([1.0, -1.0, 1.0])
# Tracking/head/hands in this PICO schema are Unity LH (+Z forward).  Camera
# SDK extrinsics, however, are a native OpenXR device pose whose local camera
# axes are already CV optical axes.  Unity -> OpenXR therefore flips Z.
UNITY_TO_OPENXR = np.diag([1.0, 1.0, -1.0])


def normalize_rotation(rotation: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(np.asarray(rotation, dtype=np.float64))
    result = u @ vh
    if np.linalg.det(result) < 0:
        u[:, -1] *= -1
        result = u @ vh
    return result


def matrix_from_pose7(pose: np.ndarray) -> np.ndarray:
    """[x,y,z,qx,qy,qz,qw] 到右手系 SE(3)，不做坐标轴变换。"""
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (7,):
        raise ValueError(f"pose7 应为 (7,)，实际 {pose.shape}")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    result[:3, 3] = pose[:3]
    return result


def unity_pose7_to_cv(pose: np.ndarray) -> np.ndarray:
    """Unity LH(x右,y上,z前) pose 转 CV RH(x右,y下,z前) pose。"""
    raw = matrix_from_pose7(pose)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = UNITY_TO_CV @ raw[:3, :3] @ UNITY_TO_CV
    result[:3, 3] = UNITY_TO_CV @ raw[:3, 3]
    result[:3, :3] = normalize_rotation(result[:3, :3])
    return result


def unity_pose7_to_openxr(pose: np.ndarray) -> np.ndarray:
    """Convert a pose between two Unity LH frames to OpenXR RH frames."""
    raw = matrix_from_pose7(pose)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = UNITY_TO_OPENXR @ raw[:3, :3] @ UNITY_TO_OPENXR
    result[:3, 3] = UNITY_TO_OPENXR @ raw[:3, 3]
    result[:3, :3] = normalize_rotation(result[:3, :3])
    return result


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    return points @ transform[:3, :3].T + transform[:3, 3]


def invert(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


def compose(*transforms: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    for transform in transforms:
        result = result @ np.asarray(transform, dtype=np.float64)
    return result


def rotation_to_6d(rotation: np.ndarray) -> np.ndarray:
    """HumanEgo/OpenPI grouped-columns: [R[:,0], R[:,1]]。"""
    rotation = normalize_rotation(rotation)
    return np.concatenate([rotation[:, 0], rotation[:, 1]])


def rotation_from_6d(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    a = value[:3]
    b = value[3:6]
    x = a / max(np.linalg.norm(a), 1e-12)
    y = b - np.dot(x, b) * x
    y = y / max(np.linalg.norm(y), 1e-12)
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=1)


def transform_to_vec9(transform: np.ndarray, translation_scale: float = 1.0) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    if translation_scale <= 0:
        raise ValueError("translation_scale 必须大于 0")
    return np.concatenate([transform[:3, 3] / translation_scale, rotation_to_6d(transform[:3, :3])]).astype(
        np.float32
    )


def vec9_to_transform(value: np.ndarray, translation_scale: float = 1.0) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = value[:3] * translation_scale
    result[:3, :3] = rotation_from_6d(value[3:9])
    return result


def rotation_angle_deg(rotation: np.ndarray) -> float:
    return float(np.rad2deg(Rotation.from_matrix(normalize_rotation(rotation)).magnitude()))


def transform_to_pose6(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    return np.concatenate([transform[:3, 3], Rotation.from_matrix(transform[:3, :3]).as_rotvec()])


def nearest_indices(source_timestamps: np.ndarray, target_timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source_timestamps, dtype=np.int64)
    target = np.asarray(target_timestamps, dtype=np.int64)
    if source.ndim != 1 or target.ndim != 1 or len(source) == 0:
        raise ValueError("时间戳必须是一维非空数组")
    right = np.searchsorted(source, target, side="left")
    right = np.clip(right, 0, len(source) - 1)
    left = np.clip(right - 1, 0, len(source) - 1)
    choose_right = np.abs(source[right] - target) < np.abs(source[left] - target)
    index = np.where(choose_right, right, left).astype(np.int64)
    gap_ms = np.abs(source[index] - target).astype(np.float64) / 1e6
    return index, gap_ms
