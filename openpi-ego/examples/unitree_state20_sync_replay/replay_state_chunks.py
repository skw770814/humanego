#!/usr/bin/env python3
# ruff: noqa: E402, RUF001
"""Replay episode-20 EEF states through the existing synchronous IK path."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
OPENPI_ROOT = SCRIPT_DIR.parents[1]
INFERENCE_DIR = OPENPI_ROOT / "examples/unitree_inference"
DEFAULT_DATASET_DIR = OPENPI_ROOT / "dataset/pick_bottle_put_in_box_have_state"
DEFAULT_PARQUET = DEFAULT_DATASET_DIR / "data/chunk-000/episode_000020.parquet"

sys.path.insert(0, str(INFERENCE_DIR))

from g1_kinematics import G1Kinematics
from g1_kinematics import rot6d_to_matrix
from policy_adapter import EEFPolicyAdapter
from strategies import SynchronousStrategy

EXPECTED_STATE_NAMES = (
    *(f"kLeftWrist_{index}" for index in range(9)),
    *(f"kRightWrist_{index}" for index in range(9)),
    "kLeftHandThumb",
    "kLeftHandThumbAux",
    "kLeftHandIndex",
    "kLeftHandMiddle",
    "kLeftHandRing",
    "kLeftHandPinky",
    "kRightHandThumb",
    "kRightHandThumbAux",
    "kRightHandIndex",
    "kRightHandMiddle",
    "kRightHandRing",
    "kRightHandPinky",
)
EXPECTED_DEPLOY_ARM_MOTORS = (
    "kLeftShoulderPitch",
    "kLeftShoulderRoll",
    "kLeftShoulderYaw",
    "kLeftElbow",
    "kLeftWristRoll",
    "kLeftWristPitch",
    "kLeftWristyaw",
    "kRightShoulderPitch",
    "kRightShoulderRoll",
    "kRightShoulderYaw",
    "kRightElbow",
    "kRightWristRoll",
    "kRightWristPitch",
    "kRightWristYaw",
)
EXPECTED_DEPLOY_HAND_MOTORS = (
    "kLeftHandThumb",
    "kLeftHandThumbAux",
    "kLeftHandIndex",
    "kLeftHandMiddle",
    "kLeftHandRing",
    "kLeftHandPinky",
    "kRightHandThumb",
    "kRightHandThumbAux",
    "kRightHandIndex",
    "kRightHandMiddle",
    "kRightHandRing",
    "kRightHandPinky",
)
EEF_DIM = 18
MODEL_DIM = 30
ROBOT_DIM = 26
ARM_DIM = 14
FPS = 30.0
MAX_ARM_SPEED_RAD_S = 2.0 * np.pi
MAX_IK_POSITION_ERROR_M = 0.03
MAX_IK_ROTATION_ERROR_RAD = np.deg2rad(12.0)


@dataclass(frozen=True)
class StateTrajectory:
    model_states: np.ndarray
    timestamps: np.ndarray
    fps: float


@dataclass(frozen=True)
class ChunkValidation:
    start: int
    end: int
    max_position_error_m: float
    max_rotation_error_rad: float
    max_arm_speed_rad_s: float


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"找不到数据集元数据：{path}")
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _dataset_columns_to_kinematics(rot6d: np.ndarray) -> np.ndarray:
    """Repack [column0 xyz, column1 xyz] for G1Kinematics' 3x2 row-major input."""
    value = np.asarray(rot6d, dtype=np.float32)
    if value.shape[-1] != 6:
        raise ValueError(f"6D rotation 最后一维必须为 6，实际为 {value.shape}")
    return value.reshape(*value.shape[:-1], 2, 3).swapaxes(-2, -1).reshape(*value.shape[:-1], 6)


def _convert_state_layout(states: np.ndarray) -> np.ndarray:
    converted = np.asarray(states, dtype=np.float32).copy()
    converted[:, 3:9] = _dataset_columns_to_kinematics(converted[:, 3:9])
    converted[:, 12:18] = _dataset_columns_to_kinematics(converted[:, 12:18])
    return np.ascontiguousarray(converted)


def _validate_rotation_columns(states: np.ndarray) -> None:
    for side, start in (("left", 3), ("right", 12)):
        columns = states[:, start : start + 6].reshape(-1, 2, 3)
        first = columns[:, 0]
        second = columns[:, 1]
        norm_error = max(
            float(np.max(np.abs(np.linalg.norm(first, axis=1) - 1.0))),
            float(np.max(np.abs(np.linalg.norm(second, axis=1) - 1.0))),
        )
        orthogonal_error = float(np.max(np.abs(np.sum(first * second, axis=1))))
        if norm_error > 1e-4 or orthogonal_error > 1e-4:
            raise ValueError(
                f"{side} rotation 前两列不是正交单位向量：norm_error={norm_error:.6g}, dot_error={orthogonal_error:.6g}"
            )


def load_state_trajectory(parquet_path: Path, dataset_dir: Path) -> StateTrajectory:
    if not parquet_path.is_file():
        raise FileNotFoundError(f"找不到 Parquet：{parquet_path}")

    info = _read_json(dataset_dir / "meta/info.json")
    modality = _read_json(dataset_dir / "meta/modality.json")
    if float(info["fps"]) != FPS:
        raise ValueError(f"数据集 fps 必须为 {FPS:g}，实际为 {info['fps']}")
    state_feature = info["features"]["observation.state"]
    if state_feature["shape"] != [MODEL_DIM]:
        raise ValueError(f"observation.state 必须为 {MODEL_DIM} 维，实际为 {state_feature['shape']}")
    state_names = tuple(state_feature["names"][0])
    if state_names != EXPECTED_STATE_NAMES:
        raise ValueError(f"observation.state 顺序不匹配：{state_names}")
    expected_modality = {
        "wrist_left_pose": {"start": 0, "end": 9},
        "wrist_right_pose": {"start": 9, "end": 18},
        "fingers_left_qpos": {"start": 18, "end": 24},
        "fingers_right_qpos": {"start": 24, "end": 30},
    }
    if modality.get("state") != expected_modality:
        raise ValueError(f"state modality 切片不匹配：{modality.get('state')}")

    required_columns = ["observation.state", "timestamp", "frame_index", "episode_index"]
    table = pq.read_table(parquet_path, columns=required_columns)
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
    frame_indices = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
    episode_indices = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)

    frame_count = len(table)
    if states.shape != (frame_count, MODEL_DIM) or timestamps.shape != (frame_count,):
        raise ValueError(f"Parquet shape 错误：state={states.shape}, timestamps={timestamps.shape}")
    if frame_count < 2 or not np.isfinite(states).all() or not np.isfinite(timestamps).all():
        raise ValueError("Parquet 数据为空、过短或包含 NaN/Inf")
    if not np.all(episode_indices == 20):
        raise ValueError(f"只允许 episode 20，实际包含 {np.unique(episode_indices)}")
    if not np.array_equal(frame_indices, np.arange(frame_count)):
        raise ValueError("frame_index 必须从 0 连续递增")
    if timestamps[0] != 0.0 or not np.allclose(np.diff(timestamps), 1.0 / FPS, rtol=1e-5, atol=1e-6):
        raise ValueError("timestamp 必须从 0 开始并保持 30 Hz")
    if float(states[:, EEF_DIM:].min()) < 0.0 or float(states[:, EEF_DIM:].max()) > 1.0:
        raise ValueError("BrainCo state 超出 [0, 1]")

    _validate_rotation_columns(states)
    return StateTrajectory(model_states=_convert_state_layout(states), timestamps=timestamps, fps=FPS)


def _pose_error(kinematics: G1Kinematics, target: np.ndarray, arm_joints: np.ndarray) -> tuple[float, float]:
    actual = kinematics.forward(arm_joints)
    position_errors = []
    rotation_errors = []
    for start in (0, 9):
        position_errors.append(float(np.linalg.norm(actual[start : start + 3] - target[start : start + 3])))
        actual_rotation = rot6d_to_matrix(actual[start + 3 : start + 9], "columns")
        target_rotation = rot6d_to_matrix(target[start + 3 : start + 9], "columns")
        cosine = np.clip((np.trace(actual_rotation.T @ target_rotation) - 1.0) / 2.0, -1.0, 1.0)
        rotation_errors.append(float(np.arccos(cosine)))
    return max(position_errors), max(rotation_errors)


def solve_first_pose(
    target_eef: np.ndarray,
    initial_arm: np.ndarray,
    urdf_path: str | None,
) -> np.ndarray:
    kinematics = G1Kinematics(urdf_path, rotation_format="columns")
    arm_joints = np.asarray(initial_arm, dtype=np.float32).copy()
    for _ in range(50):
        arm_joints = kinematics.inverse(target_eef, arm_joints)
        position_error, rotation_error = _pose_error(kinematics, target_eef, arm_joints)
        if position_error <= 0.01 and rotation_error <= np.deg2rad(3.0):
            return arm_joints
    raise RuntimeError(
        f"首帧 IK 未收敛：position={position_error * 100:.3f} cm, rotation={np.rad2deg(rotation_error):.3f} deg"
    )


class DatasetStateChunkPolicy:
    """Return the next Parquet state slice as one synchronous policy chunk."""

    def __init__(self, states: np.ndarray, chunk_size: int) -> None:
        self._states = states
        self._chunk_size = chunk_size
        self.cursor = 0
        self.inference_count = 0

    def infer(self, observation: dict) -> dict:
        del observation
        if self.cursor >= len(self._states):
            raise RuntimeError("数据集 state 已全部取完")
        start = self.cursor
        end = min(start + self._chunk_size, len(self._states))
        self.cursor = end
        self.inference_count += 1
        return {
            "actions": self._states[start:end].copy(),
            "chunk_start": start,
            "chunk_end": end,
        }


class ValidatedIKPolicy:
    """Validate each complete IK chunk before SynchronousStrategy executes it."""

    def __init__(self, policy: EEFPolicyAdapter, states: np.ndarray, fps: float, urdf_path: str | None) -> None:
        self._policy = policy
        self._states = states
        self._fps = fps
        self._validator = G1Kinematics(urdf_path, rotation_format="columns")
        self.validations: list[ChunkValidation] = []

    def infer(self, observation: dict) -> dict:
        result = self._policy.infer(observation)
        actions = np.asarray(result["actions"], dtype=np.float32)
        start = int(result["chunk_start"])
        end = int(result["chunk_end"])
        targets = self._states[start:end, :EEF_DIM]
        if actions.shape != (end - start, ROBOT_DIM):
            raise ValueError(f"IK chunk shape 错误：{actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("IK chunk 包含 NaN/Inf")

        position_errors = []
        rotation_errors = []
        for target, action in zip(targets, actions, strict=True):
            position_error, rotation_error = _pose_error(self._validator, target, action[:ARM_DIM])
            position_errors.append(position_error)
            rotation_errors.append(rotation_error)
        max_position_error = max(position_errors)
        max_rotation_error = max(rotation_errors)

        current_arm = np.asarray(observation["state"], dtype=np.float32)[:ARM_DIM]
        arm_path = np.concatenate((current_arm[None, :], actions[:, :ARM_DIM]), axis=0)
        max_arm_speed = float(np.max(np.abs(np.diff(arm_path, axis=0))) * self._fps)
        if max_position_error > MAX_IK_POSITION_ERROR_M or max_rotation_error > MAX_IK_ROTATION_ERROR_RAD:
            raise RuntimeError(
                f"chunk [{start}:{end}] IK 回代误差过大：position={max_position_error * 100:.3f} cm, "
                f"rotation={np.rad2deg(max_rotation_error):.3f} deg"
            )
        if max_arm_speed > MAX_ARM_SPEED_RAD_S + 1e-5:
            raise RuntimeError(
                f"chunk [{start}:{end}] 最大关节速度 {max_arm_speed:.3f} rad/s 超过 {MAX_ARM_SPEED_RAD_S:.3f} rad/s"
            )

        validation = ChunkValidation(
            start=start,
            end=end,
            max_position_error_m=max_position_error,
            max_rotation_error_rad=max_rotation_error,
            max_arm_speed_rad_s=max_arm_speed,
        )
        self.validations.append(validation)
        print(
            f"chunk {len(self.validations)} [{start}:{end}] IK完成："
            f"position_max={max_position_error * 100:.3f} cm, "
            f"rotation_max={np.rad2deg(max_rotation_error):.3f} deg, "
            f"joint_speed_max={max_arm_speed:.3f} rad/s"
        )
        return result

    def reset(self) -> None:
        self._policy.reset()


def make_sync_strategy(
    trajectory: StateTrajectory,
    chunk_size: int,
    urdf_path: str | None,
) -> tuple[SynchronousStrategy, DatasetStateChunkPolicy, ValidatedIKPolicy]:
    if chunk_size <= 0:
        raise ValueError("--chunk-size 必须大于 0")
    source_policy = DatasetStateChunkPolicy(trajectory.model_states, chunk_size)
    metadata = {
        "action_space": "eef",
        "end_effector": "brainco",
        "model_dim": MODEL_DIM,
        "robot_type": "unitree_g1_brainco",
        "rotation_format": "columns",
    }
    ik_policy = EEFPolicyAdapter(source_policy, metadata, "unitree_g1_brainco", urdf_path)
    validated_policy = ValidatedIKPolicy(ik_policy, trajectory.model_states, trajectory.fps, urdf_path)
    return SynchronousStrategy(validated_policy, chunk_size), source_policy, validated_policy


def _smooth_transition(current: np.ndarray, target: np.ndarray, frame_count: int) -> np.ndarray:
    phase = np.arange(1, frame_count + 1, dtype=np.float64) / frame_count
    alpha = 3.0 * phase**2 - 2.0 * phase**3
    values = current[None, :] + alpha[:, None] * (target - current)[None, :]
    return np.ascontiguousarray(values, dtype=np.float32)


def _validate_deploy_action_layout() -> None:
    from unitree_deploy.robot.robot_configs import brainco_motors
    from unitree_deploy.robot.robot_configs import g1_motors

    if tuple(g1_motors) != EXPECTED_DEPLOY_ARM_MOTORS:
        raise RuntimeError(f"unitree-deploy 双臂顺序已变化：{tuple(g1_motors)}")
    if tuple(brainco_motors) != EXPECTED_DEPLOY_HAND_MOTORS:
        raise RuntimeError(f"unitree-deploy BrainCo 顺序已变化：{tuple(brainco_motors)}")


def dry_run(trajectory: StateTrajectory, chunk_size: int, urdf_path: str | None) -> None:
    first_arm = solve_first_pose(trajectory.model_states[0, :EEF_DIM], np.zeros(ARM_DIM), urdf_path)
    current = np.concatenate((first_arm, trajectory.model_states[0, EEF_DIM:])).astype(np.float32)
    strategy, source_policy, validated_policy = make_sync_strategy(trajectory, chunk_size, urdf_path)
    actions = []
    try:
        for _ in range(len(trajectory.model_states)):
            strategy.update_observation({"state": current})
            action = np.asarray(strategy.pop_action(), dtype=np.float32)
            if action.shape != (ROBOT_DIM,):
                raise ValueError(f"同步策略输出维度错误：{action.shape}")
            actions.append(action.copy())
            current = action
    finally:
        strategy.close()
        validated_policy.reset()

    if source_policy.cursor != len(trajectory.model_states):
        raise RuntimeError(f"只消费了 {source_policy.cursor}/{len(trajectory.model_states)} 帧")
    expected_chunks = (len(trajectory.model_states) + chunk_size - 1) // chunk_size
    if source_policy.inference_count != expected_chunks or len(validated_policy.validations) != expected_chunks:
        raise RuntimeError("同步 chunk 数量不匹配")
    if len(actions) != len(trajectory.model_states):
        raise RuntimeError("同步策略输出帧数不匹配")
    print(
        f"离线同步模拟通过：{len(actions)} 帧，{source_policy.inference_count} 个 chunk，未连接机器人，也未发送动作。"
    )


def execute(
    trajectory: StateTrajectory,
    chunk_size: int,
    urdf_path: str | None,
    network_interface: str | None,
    transition_seconds: float,
) -> None:
    if transition_seconds <= 0.0:
        raise ValueError("--transition-seconds 必须大于 0")
    _validate_deploy_action_layout()

    from robot_interface import UnitreeRobotInterface

    robot = UnitreeRobotInterface("unitree_g1_brainco", 1.0 / trajectory.fps, network_interface)
    strategy = None
    validated_policy = None
    try:
        print("正在连接 unitree_g1_brainco ...")
        robot.connect()
        observation = robot.get_observation("replay episode 20 state chunks")
        current = np.asarray(observation["state"], dtype=np.float32)
        if current.shape != (ROBOT_DIM,) or not np.isfinite(current).all():
            raise ValueError(f"机器人 state 必须为有限的 {ROBOT_DIM} 维，实际为 {current.shape}")

        first_arm = solve_first_pose(trajectory.model_states[0, :EEF_DIM], current[:ARM_DIM], urdf_path)
        first_target = np.concatenate((first_arm, trajectory.model_states[0, EEF_DIM:])).astype(np.float32)
        arm_delta = float(np.max(np.abs(first_target[:ARM_DIM] - current[:ARM_DIM])))
        hand_delta = float(np.max(np.abs(first_target[ARM_DIM:] - current[ARM_DIM:])))
        print(f"当前状态到首帧 IK 目标最大差值：arm={arm_delta:.4f} rad, BrainCo={hand_delta:.4f}")
        confirmation = input("确认机器人周围安全后，输入 EXECUTE STATE 20：").strip()
        if confirmation != "EXECUTE STATE 20":
            print("确认不匹配，未发送任何动作。")
            return

        transition_frames = max(1, round(transition_seconds * trajectory.fps))
        peak_transition_speed = 1.5 * arm_delta / transition_seconds
        if peak_transition_speed > MAX_ARM_SPEED_RAD_S:
            raise RuntimeError("首帧过渡速度过高，请增大 --transition-seconds")
        for action in _smooth_transition(current, first_target, transition_frames):
            robot.step(action)

        strategy, source_policy, validated_policy = make_sync_strategy(trajectory, chunk_size, urdf_path)
        input("已到达首帧。按 Enter 开始同步 chunk 回放；Ctrl-C 取消：")
        for frame_index in range(len(trajectory.model_states)):
            observation = robot.get_observation("replay episode 20 state chunks")
            strategy.update_observation(observation)
            action = np.asarray(strategy.pop_action(), dtype=np.float32)
            if action.shape != (ROBOT_DIM,):
                raise ValueError(f"同步策略输出维度错误：{action.shape}")
            robot.step(action)
            if (frame_index + 1) % 50 == 0:
                print(f"已执行 {frame_index + 1}/{len(trajectory.model_states)} 帧")

        if source_policy.cursor != len(trajectory.model_states):
            raise RuntimeError("真机回放未消费全部 state")
        print(
            f"同步回放完成：{len(trajectory.model_states)} 帧，"
            f"{source_policy.inference_count} 个 chunk，机器人保持最后一帧目标。"
        )
    finally:
        if strategy is not None:
            strategy.close()
        if validated_policy is not None:
            validated_policy.reset()
        robot.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用现有同步策略和 G1 IK 回放 episode 20 observation.state")
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--urdf-path", default=None)
    parser.add_argument("--execute", action="store_true", help="连接真机；默认仅离线 IK 模拟")
    parser.add_argument("--network-interface", default=None)
    parser.add_argument("--transition-seconds", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trajectory = load_state_trajectory(args.parquet.resolve(), args.dataset_dir.resolve())
    print(f"数据：{args.parquet.resolve()}")
    print(f"episode 20：{len(trajectory.model_states)} 帧 / {trajectory.fps:g} Hz")
    print("输入字段：observation.state，不读取 action")
    print("表示：左右 EEF xyz + rotation 前两列 + 左右 BrainCo6")
    if args.execute:
        execute(
            trajectory,
            args.chunk_size,
            args.urdf_path,
            args.network_interface,
            args.transition_seconds,
        )
    else:
        dry_run(trajectory, args.chunk_size, args.urdf_path)


if __name__ == "__main__":
    main()
