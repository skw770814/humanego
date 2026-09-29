from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp

from ego_relation.contracts.se3 import transform_to_vec9, vec9_to_transform


def _valid_window(requested: int, length: int, minimum: int = 3) -> int:
    window = min(int(requested), length if length % 2 else length - 1)
    if window % 2 == 0:
        window -= 1
    return window if window >= minimum else 1


def _robust_threshold(values: np.ndarray, minimum: float, scale: float = 8.0) -> float:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return max(float(minimum), median + scale * 1.4826 * mad)


def _rotation_midpoint(first: Rotation, second: Rotation) -> Rotation:
    keyframes = Rotation.concatenate((first, second))
    return Slerp([0.0, 1.0], keyframes)([0.5])[0]


def _residual_peaks(residual: np.ndarray, threshold: float) -> np.ndarray:
    candidates = np.asarray(residual, dtype=np.float64) > float(threshold)
    peaks = np.zeros(len(candidates), dtype=bool)
    begin = 0
    while begin < len(candidates):
        if not candidates[begin]:
            begin += 1
            continue
        end = begin + 1
        while end < len(candidates) and candidates[end]:
            end += 1
        peak = begin + int(np.argmax(residual[begin:end]))
        peaks[peak] = True
        begin = end
    return peaks


def _repair_translation_outliers(
    positions: np.ndarray,
    *,
    minimum_residual_m: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    result = np.asarray(positions, dtype=np.float64).copy()
    if len(result) < 3:
        return result, np.zeros(len(result), dtype=bool), minimum_residual_m
    midpoint = 0.5 * (result[:-2] + result[2:])
    residual = np.linalg.norm(result[1:-1] - midpoint, axis=1)
    threshold = _robust_threshold(residual, minimum_residual_m)
    outliers = np.zeros(len(result), dtype=bool)
    outliers[1:-1] = _residual_peaks(residual, threshold)
    for frame in np.flatnonzero(outliers):
        result[frame] = 0.5 * (result[frame - 1] + result[frame + 1])
    return result, outliers, threshold


def _repair_rotation_outliers(
    rotations: Rotation,
    *,
    minimum_residual_deg: float,
) -> tuple[Rotation, np.ndarray, float]:
    matrices = rotations.as_matrix().copy()
    if len(matrices) < 3:
        return Rotation.from_matrix(matrices), np.zeros(len(matrices), dtype=bool), minimum_residual_deg
    residual_deg = np.zeros(len(matrices) - 2, dtype=np.float64)
    midpoints: list[Rotation] = []
    for frame in range(1, len(matrices) - 1):
        midpoint = _rotation_midpoint(
            Rotation.from_matrix(matrices[frame - 1]),
            Rotation.from_matrix(matrices[frame + 1]),
        )
        midpoints.append(midpoint)
        residual_deg[frame - 1] = np.degrees(
            (midpoint.inv() * Rotation.from_matrix(matrices[frame])).magnitude()
        )
    threshold = _robust_threshold(residual_deg, minimum_residual_deg)
    outliers = np.zeros(len(matrices), dtype=bool)
    outliers[1:-1] = _residual_peaks(residual_deg, threshold)
    for frame in np.flatnonzero(outliers):
        matrices[frame] = midpoints[frame - 1].as_matrix()
    return Rotation.from_matrix(matrices), outliers, threshold


def _smooth_rotations(rotations: Rotation, window: int, polyorder: int) -> Rotation:
    actual = _valid_window(window, len(rotations))
    if actual == 1:
        return rotations
    quaternions = rotations.as_quat()
    for frame in range(1, len(quaternions)):
        if float(quaternions[frame] @ quaternions[frame - 1]) < 0:
            quaternions[frame] *= -1
    filtered = savgol_filter(
        quaternions,
        actual,
        min(int(polyorder), actual - 1),
        axis=0,
        mode="interp",
    )
    filtered /= np.maximum(np.linalg.norm(filtered, axis=1, keepdims=True), 1e-12)
    return Rotation.from_quat(filtered)


def _trajectory_metrics(transforms: np.ndarray, fps: float) -> dict[str, float]:
    if len(transforms) < 2:
        return {
            "translation_step_mm_p95": 0.0,
            "translation_step_mm_max": 0.0,
            "translation_speed_m_s_max": 0.0,
            "rotation_step_deg_p95": 0.0,
            "rotation_step_deg_max": 0.0,
            "rotation_speed_deg_s_max": 0.0,
        }
    translation_step = np.linalg.norm(np.diff(transforms[:, :3, 3], axis=0), axis=1)
    rotations = Rotation.from_matrix(transforms[:, :3, :3])
    rotation_step = np.degrees((rotations[:-1].inv() * rotations[1:]).magnitude())
    return {
        "translation_step_mm_p95": float(np.percentile(translation_step, 95) * 1000),
        "translation_step_mm_max": float(translation_step.max() * 1000),
        "translation_speed_m_s_max": float(translation_step.max() * fps),
        "rotation_step_deg_p95": float(np.percentile(rotation_step, 95)),
        "rotation_step_deg_max": float(rotation_step.max()),
        "rotation_speed_deg_s_max": float(rotation_step.max() * fps),
    }


def smooth_mode2_state(
    state: np.ndarray,
    *,
    fps: float,
    median_window: int = 3,
    smooth_window: int = 11,
    polyorder: int = 2,
    minimum_translation_outlier_m: float = 0.003,
    minimum_rotation_outlier_deg: float = 2.0,
) -> tuple[np.ndarray, dict]:
    state = np.asarray(state, dtype=np.float64)
    if state.ndim != 2 or state.shape[1] != 30:
        raise ValueError(f"Mode2 state must have shape (T, 30), got {state.shape}")
    if not np.isfinite(state).all():
        raise ValueError("Mode2 state contains non-finite values")
    result = state.copy()
    report: dict = {"frames": int(len(state)), "fps": float(fps), "sides": {}}
    translation_window = _valid_window(median_window, len(state))
    smooth_actual = _valid_window(smooth_window, len(state))
    for side, pose_slice in (("left", slice(0, 9)), ("right", slice(9, 18))):
        transforms = np.stack([vec9_to_transform(row) for row in state[:, pose_slice]])
        positions, translation_outliers, translation_threshold = _repair_translation_outliers(
            transforms[:, :3, 3],
            minimum_residual_m=minimum_translation_outlier_m,
        )
        if translation_window > 1:
            positions = median_filter(
                positions,
                size=(translation_window, 1),
                mode="nearest",
            )
        if smooth_actual > 1:
            positions = savgol_filter(
                positions,
                smooth_actual,
                min(int(polyorder), smooth_actual - 1),
                axis=0,
                mode="interp",
            )
        rotations, rotation_outliers, rotation_threshold = _repair_rotation_outliers(
            Rotation.from_matrix(transforms[:, :3, :3]),
            minimum_residual_deg=minimum_rotation_outlier_deg,
        )
        rotations = _smooth_rotations(rotations, smooth_window, polyorder)
        filtered = np.repeat(np.eye(4, dtype=np.float64)[None], len(state), axis=0)
        filtered[:, :3, :3] = rotations.as_matrix()
        filtered[:, :3, 3] = positions
        result[:, pose_slice] = np.stack([transform_to_vec9(value) for value in filtered])
        report["sides"][side] = {
            "translation_outlier_frames": np.flatnonzero(translation_outliers).tolist(),
            "rotation_outlier_frames": np.flatnonzero(rotation_outliers).tolist(),
            "translation_outlier_threshold_mm": float(translation_threshold * 1000),
            "rotation_outlier_threshold_deg": float(rotation_threshold),
            "before": _trajectory_metrics(transforms, fps),
            "after": _trajectory_metrics(filtered, fps),
        }
    report["filter"] = {
        "translation_median_window": int(translation_window),
        "savgol_window": int(smooth_actual),
        "savgol_polyorder": int(polyorder),
        "finger_commands": "unchanged",
    }
    return result.astype(np.float32), report


def write_smoothed_mode2(
    episode_dir: str | Path,
    *,
    fps: float = 30.0,
    median_window: int = 3,
    smooth_window: int = 11,
    polyorder: int = 2,
) -> tuple[Path, Path, Path]:
    episode_dir = Path(episode_dir).expanduser().resolve()
    mode2_dir = episode_dir / "mode2"
    state = np.load(mode2_dir / "state_abs.npy")
    smoothed, report = smooth_mode2_state(
        state,
        fps=fps,
        median_window=median_window,
        smooth_window=smooth_window,
        polyorder=polyorder,
    )
    action = np.concatenate((smoothed[1:], smoothed[-1:]), axis=0)
    state_path = mode2_dir / "state_abs_smoothed.npy"
    action_path = mode2_dir / "action_abs_smoothed.npy"
    report_path = episode_dir / "qa" / "step1_tcp_smoothing_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(state_path, smoothed)
    np.save(action_path, action)
    report["outputs"] = {
        "state": str(state_path),
        "action": str(action_path),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return state_path, action_path, report_path
