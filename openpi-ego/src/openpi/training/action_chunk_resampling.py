"""Task-progress-aware interpolation for absolute EEF action chunks."""

from __future__ import annotations

import math

import numpy as np


def required_source_horizon(output_horizon: int, source_step_scale: float) -> int:
    """Return the dense source length needed for scaled output sample times."""
    if output_horizon < 1:
        raise ValueError("output_horizon must be positive")
    if not math.isfinite(source_step_scale) or source_step_scale <= 0:
        raise ValueError("source_step_scale must be finite and positive")
    return math.ceil((output_horizon - 1) * source_step_scale) + 1


def _rotation6d_to_matrix(values: np.ndarray) -> np.ndarray:
    vectors = np.asarray(values, dtype=np.float64).reshape(*np.asarray(values).shape[:-1], 2, 3)
    first = vectors[..., 0, :]
    second_raw = vectors[..., 1, :]
    first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-12)
    second = second_raw - np.sum(first * second_raw, axis=-1, keepdims=True) * first
    second = second / np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-12)
    return np.stack((first, second, np.cross(first, second)), axis=-1)


def _matrix_to_rotation6d(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation)
    return np.concatenate((rotation[..., :, 0], rotation[..., :, 1]), axis=-1)


def _matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to scalar-first quaternions without SciPy."""
    rotation = np.asarray(rotation, dtype=np.float64)
    m00, m01, m02 = rotation[..., 0, 0], rotation[..., 0, 1], rotation[..., 0, 2]
    m10, m11, m12 = rotation[..., 1, 0], rotation[..., 1, 1], rotation[..., 1, 2]
    m20, m21, m22 = rotation[..., 2, 0], rotation[..., 2, 1], rotation[..., 2, 2]
    q_abs = np.sqrt(
        np.maximum(
            np.stack(
                (
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ),
                axis=-1,
            ),
            0.0,
        )
    )
    candidates = np.stack(
        (
            np.stack((q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01), axis=-1),
            np.stack((m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20), axis=-1),
            np.stack((m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21), axis=-1),
            np.stack((m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2), axis=-1),
        ),
        axis=-2,
    )
    candidates /= 2.0 * np.maximum(q_abs[..., None], 0.1)
    best = np.argmax(q_abs, axis=-1)
    quaternion = np.take_along_axis(candidates, best[..., None, None], axis=-2)[..., 0, :]
    return quaternion / np.maximum(np.linalg.norm(quaternion, axis=-1, keepdims=True), 1e-12)


def _quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion = quaternion / np.maximum(np.linalg.norm(quaternion, axis=-1, keepdims=True), 1e-12)
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)


def _slerp(rotation0: np.ndarray, rotation1: np.ndarray, amount: np.ndarray) -> np.ndarray:
    quaternion0 = _matrix_to_quaternion(rotation0)
    quaternion1 = _matrix_to_quaternion(rotation1)
    dot = np.sum(quaternion0 * quaternion1, axis=-1, keepdims=True)
    quaternion1 = np.where(dot < 0.0, -quaternion1, quaternion1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    amount = np.asarray(amount, dtype=np.float64)

    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    weight0 = np.sin((1.0 - amount) * theta) / np.maximum(sin_theta, 1e-12)
    weight1 = np.sin(amount * theta) / np.maximum(sin_theta, 1e-12)
    spherical = weight0 * quaternion0 + weight1 * quaternion1
    linear = (1.0 - amount) * quaternion0 + amount * quaternion1
    quaternion = np.where(dot > 0.9995, linear, spherical)
    return _quaternion_to_matrix(quaternion)


def resample_absolute_eef_actions(
    source_actions: np.ndarray,
    source_is_pad: np.ndarray,
    *,
    output_horizon: int,
    source_step_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample dense EEF18/EEF20/EEF30 chunks at scaled fractional source-frame offsets.

    The final axis layout must be ``left EEF9 + right EEF9 + optional absolute
    tail``. XYZ and an optional tail use linear interpolation. Each EEF
    rotation uses SLERP.
    Leading batch dimensions are preserved.
    """
    source_actions = np.asarray(source_actions)
    source_is_pad = np.asarray(source_is_pad, dtype=np.bool_)
    if source_actions.ndim < 2 or source_actions.shape[-1] not in (18, 20, 30):
        raise ValueError(f"Expected EEF18, EEF20, or EEF30 source actions, got {source_actions.shape}")
    if source_is_pad.shape != source_actions.shape[:-1]:
        raise ValueError(
            f"source_is_pad shape {source_is_pad.shape} does not match actions {source_actions.shape[:-1]}"
        )
    expected_source = required_source_horizon(output_horizon, source_step_scale)
    if source_actions.shape[-2] != expected_source:
        raise ValueError(f"Expected {expected_source} source steps, got {source_actions.shape[-2]}")

    target_offsets = np.arange(output_horizon, dtype=np.float64) * source_step_scale
    lower = np.floor(target_offsets).astype(np.int64)
    upper = np.minimum(lower + 1, expected_source - 1)
    amount = (target_offsets - lower).reshape(*(1 for _ in source_actions.shape[:-2]), output_horizon, 1)
    lower_actions = source_actions[..., lower, :].astype(np.float64)
    upper_actions = source_actions[..., upper, :].astype(np.float64)
    output = lower_actions + amount * (upper_actions - lower_actions)

    for pose_start in (0, 9):
        rotation0 = _rotation6d_to_matrix(lower_actions[..., pose_start + 3 : pose_start + 9])
        rotation1 = _rotation6d_to_matrix(upper_actions[..., pose_start + 3 : pose_start + 9])
        output[..., pose_start + 3 : pose_start + 9] = _matrix_to_rotation6d(
            _slerp(rotation0, rotation1, amount)
        )

    valid_source_count = np.sum(~source_is_pad, axis=-1, dtype=np.int64)
    last_valid_offset = np.maximum(valid_source_count - 1, 0)
    output_is_pad = target_offsets > last_valid_offset[..., None]
    return output.astype(source_actions.dtype, copy=False), output_is_pad
