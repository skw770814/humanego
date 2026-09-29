"""Portable Mode2 labeler internalized from egodata_targeting_project.

The implementation intentionally keeps Mode2's timestamp-nearest semantics:
the uniform grid may be 30 Hz when every raw stream is faster than 30 Hz, but
poses and hand landmarks are never synthesized between source samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from ego_relation.config import ProjectConfig
from ego_relation.contracts.se3 import transform_to_vec9
from ego_relation.s1_pico_mode2.tcp import TCP_TO_INWARD_PALM


R_XR_TO_MJ = np.asarray(
    [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
)
PELVIS_TO_G1 = np.asarray(
    [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
)
MOTOR_ORDER = ("thumb_flex", "thumb_rot", "index", "middle", "ring", "pinky")
FINGERS = ("index", "middle", "ring", "pinky")
XR_WRIST = 1
XR_TIPS = {"thumb": 5, "index": 10, "middle": 15, "ring": 20, "pinky": 25}
XR_INDEX_KNUCKLE = 7
XR_MIDDLE_KNUCKLE = 12
XR_PINKY_KNUCKLE = 22
BODY_PELVIS = 0
BODY_LEFT_WRIST = 20
BODY_RIGHT_WRIST = 21
RATE = np.asarray(
    [2.5303 / 1.03, 2.6175 / 1.57, 2.2685 / 1.41, 2.2685 / 1.41, 2.2685 / 1.41, 2.2685 / 1.41],
    dtype=np.float32,
)


@dataclass
class RawMode2:
    body_ts: np.ndarray
    left_hand_ts: np.ndarray
    right_hand_ts: np.ndarray
    camera_ts: np.ndarray
    left_wrist_pos: np.ndarray
    left_wrist_quat: np.ndarray
    right_wrist_pos: np.ndarray
    right_wrist_quat: np.ndarray
    pelvis_pos: np.ndarray
    pelvis_quat: np.ndarray
    body_active: np.ndarray
    left_hand_pose: np.ndarray
    right_hand_pose: np.ndarray
    left_hand_active: np.ndarray
    right_hand_active: np.ndarray


def _quat_wxyz_from_xyzw(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return np.concatenate((value[..., 3:4], value[..., :3]), axis=-1)


def _quat_xyzw_from_wxyz(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return np.concatenate((value[..., 1:], value[..., :1]), axis=-1)


def _mat_from_quat(value: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(_quat_xyzw_from_wxyz(value)).as_matrix()


def _gravity_aligned_pelvis_rotation(pelvis_rotation: np.ndarray) -> np.ndarray:
    """Keep pelvis heading while aligning its vertical axis with gravity.

    Body-joint local axes remain OpenXR-like after the world mapping: +X is
    right, +Y is up, and -Z is forward. The returned matrix has those same
    local axes, but discards pelvis pitch and roll.
    """
    rotation = np.asarray(pelvis_rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(f"pelvis rotation must be (3, 3), got {rotation.shape}")
    world_up = np.asarray([0.0, 0.0, 1.0])
    forward = rotation @ np.asarray([0.0, 0.0, -1.0])
    forward -= float(forward @ world_up) * world_up
    norm = float(np.linalg.norm(forward))
    if norm < 1e-6:
        right_hint = rotation @ np.asarray([1.0, 0.0, 0.0])
        right_hint -= float(right_hint @ world_up) * world_up
        right_norm = float(np.linalg.norm(right_hint))
        if right_norm < 1e-6:
            forward = np.asarray([1.0, 0.0, 0.0])
        else:
            forward = np.cross(world_up, right_hint / right_norm)
    else:
        forward /= norm
    right = np.cross(forward, world_up)
    right /= max(float(np.linalg.norm(right)), 1e-12)
    back = -forward
    return np.stack((right, world_up, back), axis=1)


def _mapped_body_joint(body: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray]:
    pos = body[:, index, :3].astype(np.float64) @ R_XR_TO_MJ.T
    source_rotation = Rotation.from_quat(body[:, index, 3:7].astype(np.float64))
    mapped_rotation = Rotation.from_matrix(R_XR_TO_MJ) * source_rotation
    quat = _quat_wxyz_from_xyzw(mapped_rotation.as_quat())
    return pos, quat


def _load_raw(path: Path) -> RawMode2:
    with h5py.File(path, "r") as file:
        body = file["body_pose"][:]
        left_wrist_pos, left_wrist_quat = _mapped_body_joint(body, BODY_LEFT_WRIST)
        right_wrist_pos, right_wrist_quat = _mapped_body_joint(body, BODY_RIGHT_WRIST)
        pelvis_pos, pelvis_quat = _mapped_body_joint(body, BODY_PELVIS)
        return RawMode2(
            body_ts=file["body_timestamps_ns"][:].astype(np.int64),
            left_hand_ts=file["left_hand_timestamps_ns"][:].astype(np.int64),
            right_hand_ts=file["right_hand_timestamps_ns"][:].astype(np.int64),
            camera_ts=file["camera/timestamps_ns"][:].astype(np.int64),
            left_wrist_pos=left_wrist_pos,
            left_wrist_quat=left_wrist_quat,
            right_wrist_pos=right_wrist_pos,
            right_wrist_quat=right_wrist_quat,
            pelvis_pos=pelvis_pos,
            pelvis_quat=pelvis_quat,
            body_active=(
                file["body_pose_valid"][:].astype(bool)
                if "body_pose_valid" in file
                else np.ones(len(body), dtype=bool)
            ),
            left_hand_pose=file["left_hand_pose"][:].astype(np.float64),
            right_hand_pose=file["right_hand_pose"][:].astype(np.float64),
            left_hand_active=file["left_hand_active"][:].astype(bool),
            right_hand_active=file["right_hand_active"][:].astype(bool),
        )


def _invalid_runs(valid: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    index = 0
    while index < len(valid):
        if valid[index]:
            index += 1
            continue
        end = index
        while end < len(valid) and not valid[end]:
            end += 1
        runs.append((index, end))
        index = end
    return runs


def _jump_mask(value: np.ndarray, threshold: float) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    flat = value.reshape(len(value), -1, 3)
    mask = np.zeros(len(value), dtype=bool)
    if len(value) < 3:
        return mask
    steps = np.linalg.norm(flat[1:] - flat[:-1], axis=2).max(axis=1)
    edges = np.flatnonzero(steps > threshold)
    index = 0
    while index + 1 < len(edges):
        begin, end = int(edges[index]), int(edges[index + 1])
        if np.linalg.norm(flat[begin] - flat[end + 1], axis=1).max() <= threshold:
            mask[begin + 1 : end + 1] = True
            index += 2
        else:
            index += 1
    return mask


def _repair_short_runs(value: np.ndarray, bad: np.ndarray, maximum: int) -> np.ndarray:
    result = np.array(value, copy=True)
    for begin, end in _invalid_runs(~bad):
        if end - begin > maximum or begin == 0 or end >= len(result):
            continue
        for index in range(begin, end):
            fraction = (index - begin + 1) / (end - begin + 1)
            result[index] = (1.0 - fraction) * result[begin - 1] + fraction * result[end]
    return result


def _smooth(value: np.ndarray, window: int) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    actual = min(int(window), len(value) if len(value) % 2 else len(value) - 1)
    if actual < 3:
        return value.copy()
    return savgol_filter(value, actual, min(2, actual - 1), axis=0, mode="interp")


def _smooth_quat(value: np.ndarray, window: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).copy()
    flat = result.reshape(len(result), -1, 4)
    for index in range(1, len(flat)):
        flip = np.sum(flat[index] * flat[index - 1], axis=1) < 0
        flat[index, flip] *= -1
    flat = _smooth(flat, window)
    flat /= np.maximum(np.linalg.norm(flat, axis=2, keepdims=True), 1e-12)
    return flat.reshape(result.shape)


def _clean_pose(
    position: np.ndarray,
    quaternion: np.ndarray,
    active: np.ndarray,
    jump_threshold: float,
    cfg: ProjectConfig,
) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(position).all(axis=tuple(range(1, position.ndim)))
    bad = ~finite | ~np.asarray(active, dtype=bool) | _jump_mask(position, jump_threshold)
    fixed_position = _repair_short_runs(position, bad, cfg.mode2.max_repair_gap)
    fixed_quaternion = _repair_short_runs(quaternion, bad, cfg.mode2.max_repair_gap)
    return _smooth(fixed_position, cfg.mode2.smooth_window), _smooth_quat(
        fixed_quaternion, cfg.mode2.smooth_window
    )


def _clean(raw: RawMode2, cfg: ProjectConfig) -> RawMode2:
    left_pos, left_quat = _clean_pose(
        raw.left_wrist_pos,
        raw.left_wrist_quat,
        raw.body_active,
        cfg.mode2.max_wrist_jump_m,
        cfg,
    )
    right_pos, right_quat = _clean_pose(
        raw.right_wrist_pos,
        raw.right_wrist_quat,
        raw.body_active,
        cfg.mode2.max_wrist_jump_m,
        cfg,
    )
    pelvis_pos, pelvis_quat = _clean_pose(
        raw.pelvis_pos,
        raw.pelvis_quat,
        raw.body_active,
        cfg.mode2.max_pelvis_jump_m,
        cfg,
    )
    left_hand_pos, left_hand_quat = _clean_pose(
        raw.left_hand_pose[:, :, :3],
        raw.left_hand_pose[:, :, 3:7],
        raw.left_hand_active,
        cfg.mode2.max_hand_keypoint_jump_m,
        cfg,
    )
    right_hand_pos, right_hand_quat = _clean_pose(
        raw.right_hand_pose[:, :, :3],
        raw.right_hand_pose[:, :, 3:7],
        raw.right_hand_active,
        cfg.mode2.max_hand_keypoint_jump_m,
        cfg,
    )
    raw.left_wrist_pos, raw.left_wrist_quat = left_pos, left_quat
    raw.right_wrist_pos, raw.right_wrist_quat = right_pos, right_quat
    raw.pelvis_pos, raw.pelvis_quat = pelvis_pos, pelvis_quat
    raw.left_hand_pose = np.concatenate((left_hand_pos, left_hand_quat), axis=2)
    raw.right_hand_pose = np.concatenate((right_hand_pos, right_hand_quat), axis=2)
    return raw


def _stream_hz(timestamps: np.ndarray) -> float:
    if len(timestamps) < 2:
        return 0.0
    interval = float(np.median(np.diff(timestamps).astype(np.float64)))
    return 1e9 / interval if interval > 0 else 0.0


def _grid_ticks(
    streams: tuple[np.ndarray, ...],
    fps: float,
    source_rate_tolerance_hz: float = 0.0,
) -> np.ndarray:
    if fps <= 0:
        raise ValueError("timeline.control_hz must be positive")
    if source_rate_tolerance_hz < 0:
        raise ValueError("timeline.source_rate_tolerance_hz must be non-negative")
    rates = [_stream_hz(stream) for stream in streams]
    minimum_accepted_rate = fps - source_rate_tolerance_hz
    if min(rates) + 1e-6 < minimum_accepted_rate:
        raise ValueError(
            f"Mode2 nearest grid {fps:g} Hz exceeds raw source rate beyond "
            f"{source_rate_tolerance_hz:g} Hz tolerance: {rates}"
        )
    start = max(int(stream[0]) for stream in streams)
    end = min(int(stream[-1]) for stream in streams)
    step = int(round(1e9 / fps))
    count = max(1, int((end - start) // step) + 1)
    return start + step * np.arange(count, dtype=np.int64)


def _nearest(source: np.ndarray, ticks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    right = np.searchsorted(source, ticks, side="left")
    right = np.clip(right, 0, len(source) - 1)
    left = np.clip(right - 1, 0, len(source) - 1)
    choose_right = np.abs(source[right] - ticks) <= np.abs(source[left] - ticks)
    indices = np.where(choose_right, right, left).astype(np.int64)
    gaps_ms = np.abs(source[indices] - ticks).astype(np.float64) / 1e6
    return indices, gaps_ms


def _palm_frame(keypoints: np.ndarray, side: str) -> np.ndarray:
    wrist = keypoints[XR_WRIST]
    forward = keypoints[XR_MIDDLE_KNUCKLE] - wrist
    forward /= np.linalg.norm(forward) + 1e-9
    across = (
        keypoints[XR_INDEX_KNUCKLE] - keypoints[XR_PINKY_KNUCKLE]
        if side == "right"
        else keypoints[XR_PINKY_KNUCKLE] - keypoints[XR_INDEX_KNUCKLE]
    )
    normal = np.cross(forward, across)
    normal /= np.linalg.norm(normal) + 1e-9
    forward -= float(forward @ normal) * normal
    forward /= np.linalg.norm(forward) + 1e-9
    lateral = np.cross(normal, forward)
    return np.stack((forward, lateral, normal), axis=1)


def _quat_xyzw_to_rot(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    value /= max(float(np.linalg.norm(value)), 1e-12)
    return Rotation.from_quat(value).as_matrix()


class GeometricRetargeter:
    def __init__(self, assets: Path, side: str):
        self.side = side
        with np.load(assets / f"fk_tables_{side}.npz") as archive:
            self.arc_commands = archive["arc_cmds"]
            self.arcs = np.stack([archive[f"arc_{finger}"] for finger in FINGERS])
            self.grid_commands = archive["grid_cmds"]
            self.thumb_grid = archive["thumb_grid"]
            self.robot_palm = archive["robot_palm"]

    @staticmethod
    def _tips(pose: np.ndarray) -> np.ndarray:
        indices = [XR_TIPS[finger] for finger in ("thumb", *FINGERS)]
        result = np.empty((len(pose), 5, 3), dtype=np.float64)
        for frame in range(len(pose)):
            wrist_rotation = _quat_xyzw_to_rot(pose[frame, XR_WRIST, 3:7])
            relative = pose[frame, indices, :3] - pose[frame, XR_WRIST, :3]
            result[frame] = relative @ wrist_rotation
        return result

    def retarget(self, pose: np.ndarray, ticks_ns: np.ndarray) -> np.ndarray:
        valid = np.isfinite(pose).all(axis=(1, 2)) & (np.abs(pose).sum(axis=(1, 2)) > 0)
        if not valid.any():
            raise ValueError(f"{self.side}: no valid hand tracking frames")
        tips = self._tips(pose)
        spread = np.where(valid, np.linalg.norm(tips, axis=2).sum(axis=1), -np.inf)
        calibration = int(np.argmax(spread))
        robot_open = np.vstack((self.thumb_grid[0, 0], self.arcs[:, 0]))
        scale = np.linalg.norm(robot_open, axis=1) / np.maximum(
            np.linalg.norm(tips[calibration], axis=1), 1e-6
        )
        wrist_rotation = _quat_xyzw_to_rot(pose[calibration, XR_WRIST, 3:7])
        human_palm_wrist = wrist_rotation.T @ _palm_frame(pose[calibration, :, :3], self.side)
        alignment = self.robot_palm @ human_palm_wrist.T
        target = np.einsum("ij,tfj->tfi", alignment, tips) * scale[None, :, None]
        command = np.zeros((len(pose), 6), dtype=np.float32)
        for frame in np.flatnonzero(valid):
            distance = ((self.arcs - target[frame, 1:, None, :]) ** 2).sum(axis=2)
            finger_indices = distance.argmin(axis=1)
            finger_commands = self.arc_commands[finger_indices]
            finger_tips = self.arcs[np.arange(4), finger_indices]
            cost = ((self.thumb_grid - target[frame, 0]) ** 2).sum(axis=2)
            for finger in (0, 1):
                human_distance = np.linalg.norm(tips[frame, 0] - tips[frame, finger + 1])
                morphology_scale = 0.5 * (scale[0] + scale[finger + 1])
                goal = -0.003 if human_distance < 0.025 else morphology_scale * human_distance
                weight = 10.0 if human_distance < 0.025 else 3.0
                cost += weight * (
                    np.linalg.norm(self.thumb_grid - finger_tips[finger], axis=2) - goal
                ) ** 2
            first, second = np.unravel_index(cost.argmin(), cost.shape)
            command[frame] = [self.grid_commands[first], self.grid_commands[second], *finger_commands]
        good = np.flatnonzero(valid)
        source = good[np.clip(np.searchsorted(good, np.arange(len(pose)), side="right") - 1, 0, None)]
        command = command[source]
        intervals = np.diff(ticks_ns.astype(np.float64)) * 1e-9
        for frame in range(1, len(command)):
            step = RATE * max(intervals[frame - 1], 1e-4)
            command[frame] = np.clip(command[frame], command[frame - 1] - step, command[frame - 1] + step)
        return np.clip(command, 0.0, 1.0).astype(np.float32)


def build_mode2_labels(cfg: ProjectConfig, source: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    if cfg.mode2.hand_retargeter != "geometric":
        raise ValueError("portable Mode2 currently requires hand_retargeter=geometric")
    source = Path(source).resolve()
    raw = _clean(_load_raw(source), cfg)
    streams = (raw.body_ts, raw.left_hand_ts, raw.right_hand_ts, raw.camera_ts)
    ticks = _grid_ticks(
        streams,
        cfg.timeline.control_hz,
        cfg.timeline.source_rate_tolerance_hz,
    )
    body_indices, body_gaps = _nearest(raw.body_ts, ticks)
    left_indices, left_gaps = _nearest(raw.left_hand_ts, ticks)
    right_indices, right_gaps = _nearest(raw.right_hand_ts, ticks)
    camera_indices, camera_gaps = _nearest(raw.camera_ts, ticks)
    left_hand_pose = raw.left_hand_pose[left_indices]
    right_hand_pose = raw.right_hand_pose[right_indices]
    pelvis_positions = raw.pelvis_pos[body_indices]
    pelvis_rotations = np.stack(
        [_mat_from_quat(value) for value in raw.pelvis_quat[body_indices]]
    )
    world_reference_origin = pelvis_positions[0].copy()
    world_reference_rotation = _gravity_aligned_pelvis_rotation(pelvis_rotations[0])
    pelvis_level_rotations = np.stack(
        [_gravity_aligned_pelvis_rotation(value) for value in pelvis_rotations]
    )
    reference_forward = world_reference_rotation @ np.asarray([0.0, 0.0, -1.0])
    pelvis_forwards = np.einsum(
        "tij,j->ti", pelvis_level_rotations, np.asarray([0.0, 0.0, -1.0])
    )
    yaw_delta_deg = np.degrees(
        np.arctan2(
            reference_forward[0] * pelvis_forwards[:, 1]
            - reference_forward[1] * pelvis_forwards[:, 0],
            pelvis_forwards[:, :2] @ reference_forward[:2],
        )
    )
    pelvis_delta = pelvis_positions - world_reference_origin
    pelvis_motion_qa = {
        "horizontal_displacement_m_max": float(
            np.linalg.norm(pelvis_delta[:, :2], axis=1).max()
        ),
        "vertical_displacement_m_max": float(np.abs(pelvis_delta[:, 2]).max()),
        "yaw_change_deg_max": float(np.abs(yaw_delta_deg).max()),
    }

    state_eef: dict[str, np.ndarray] = {}
    for side, wrist_position, wrist_quaternion, hand_pose in (
        ("left", raw.left_wrist_pos[body_indices], raw.left_wrist_quat[body_indices], left_hand_pose),
        ("right", raw.right_wrist_pos[body_indices], raw.right_wrist_quat[body_indices], right_hand_pose),
    ):
        vectors = np.zeros((len(ticks), 9), dtype=np.float32)
        for frame in range(len(ticks)):
            wrist = np.eye(4)
            wrist[:3, :3] = _mat_from_quat(wrist_quaternion[frame])
            wrist[:3, 3] = wrist_position[frame]
            target = np.eye(4)
            relative_position = world_reference_rotation.T @ (
                wrist[:3, 3] - world_reference_origin
            )
            target[:3, 3] = PELVIS_TO_G1 @ relative_position
            keypoints_world = hand_pose[frame, :, :3] @ PELVIS_TO_G1.T
            keypoints_pelvis = (
                keypoints_world - world_reference_origin
            ) @ world_reference_rotation
            keypoints_g1 = keypoints_pelvis @ PELVIS_TO_G1.T
            target[:3, :3] = _palm_frame(keypoints_g1, side) @ TCP_TO_INWARD_PALM[side].T
            vectors[frame] = transform_to_vec9(target)
        state_eef[side] = vectors

    assets = cfg.paths.assets_dir / "revo2"
    left_hand = GeometricRetargeter(assets, "left").retarget(left_hand_pose, ticks)
    right_hand = GeometricRetargeter(assets, "right").retarget(right_hand_pose, ticks)
    state = np.concatenate(
        (state_eef["left"], state_eef["right"], left_hand, right_hand), axis=1
    ).astype(np.float32)
    action = np.concatenate((state[1:], state[-1:]), axis=0)
    body_valid = raw.body_active[body_indices] & (
        body_gaps <= cfg.timeline.max_tracking_gap_ms
    )
    left_valid = raw.left_hand_active[left_indices] & (
        left_gaps <= cfg.timeline.max_tracking_gap_ms
    )
    right_valid = raw.right_hand_active[right_indices] & (
        right_gaps <= cfg.timeline.max_tracking_gap_ms
    )
    camera_valid = camera_gaps <= cfg.timeline.max_camera_gap_ms
    valid = body_valid & left_valid & right_valid & camera_valid
    rates = {
        "body_hz": _stream_hz(raw.body_ts),
        "left_hand_hz": _stream_hz(raw.left_hand_ts),
        "right_hand_hz": _stream_hz(raw.right_hand_ts),
        "camera_hz": _stream_hz(raw.camera_ts),
    }
    return {
        "state": state,
        "action": action,
        "ticks_ns": ticks,
        "body_match": body_indices,
        "left_hand_match": left_indices,
        "right_hand_match": right_indices,
        "cam_match": camera_indices,
        "body_gap_ms": body_gaps.astype(np.float32),
        "left_hand_gap_ms": left_gaps.astype(np.float32),
        "right_hand_gap_ms": right_gaps.astype(np.float32),
        "cam_gap_ms": camera_gaps.astype(np.float32),
        "body_valid": body_valid,
        "left_hand_valid": left_valid,
        "right_hand_valid": right_valid,
        "camera_valid": camera_valid,
        "valid": valid,
    }, {
        "mode": "mode2_state",
        "mapping": "retarget_brainco_static_world_gravity_to_g1",
        "reference_frame": "initial_pelvis_origin_world_gravity_initial_pelvis_yaw",
        "engine": "ego_relation.native_mode2_v4_30hz_qa",
        "source_rates_hz": rates,
        "pelvis_motion_qa": pelvis_motion_qa,
    }
