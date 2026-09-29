#!/usr/bin/env python3
# ruff: noqa: RUF001
"""Validate or replay exported episode 20 on a Unitree G1 with BrainCo hands."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
OPENPI_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_TRAJECTORY = OPENPI_ROOT / "dataset/exports/episode_000020_g1_replay_state.npz"
DEFAULT_URDF = Path("/home/zh/unitree-deploy/unitree_deploy/robot_devices/assets/g1/g1_body29_hand14.urdf")
ROBOT_INTERFACE_DIR = OPENPI_ROOT / "examples/unitree_inference"

EXPECTED_ARM_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
EXPECTED_HAND_NAMES = ("thumb_flex", "thumb_rot", "index", "middle", "ring", "pinky")
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
ACTION_DIM = 26
ARM_DIM = 14
HAND_DIM = 12
MAX_ARM_SPEED_RAD_S = 2.0 * np.pi


@dataclass(frozen=True)
class EpisodeTrajectory:
    actions: np.ndarray
    arm_qpos: np.ndarray
    hand_qpos: np.ndarray
    timestamps: np.ndarray
    fps: float


def _scalar(data: np.lib.npyio.NpzFile, key: str):
    value = np.asarray(data[key])
    if value.ndim != 0:
        raise ValueError(f"{key} 必须是标量，实际 shape={value.shape}")
    return value.item()


def _validate_urdf_limits(arm_qpos: np.ndarray, urdf_path: Path) -> None:
    if not urdf_path.is_file():
        raise FileNotFoundError(f"找不到 G1 URDF：{urdf_path}")

    root = ET.parse(urdf_path).getroot()
    joint_limits: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        if limit is None or "lower" not in limit.attrib or "upper" not in limit.attrib:
            continue
        joint_limits[joint.attrib["name"]] = (float(limit.attrib["lower"]), float(limit.attrib["upper"]))

    tolerance = 1e-5
    for index, name in enumerate(EXPECTED_ARM_NAMES):
        if name not in joint_limits:
            raise ValueError(f"URDF 中缺少关节限位：{name}")
        lower, upper = joint_limits[name]
        observed_min = float(arm_qpos[:, index].min())
        observed_max = float(arm_qpos[:, index].max())
        if observed_min < lower - tolerance or observed_max > upper + tolerance:
            raise ValueError(
                f"{name} 超出 URDF 限位 [{lower:.6f}, {upper:.6f}]：轨迹范围 [{observed_min:.6f}, {observed_max:.6f}]"
            )


def load_and_validate(path: Path, urdf_path: Path) -> EpisodeTrajectory:
    if not path.is_file():
        raise FileNotFoundError(f"找不到轨迹文件：{path}")

    required_keys = {
        "arm_qpos",
        "hand_left",
        "hand_right",
        "hand_qpos",
        "joint26",
        "timestamps",
        "joint_names",
        "hand_names",
        "fps",
        "episode_index",
        "arm_units",
        "hand_units",
    }
    with np.load(path, allow_pickle=False) as data:
        missing = required_keys.difference(data.files)
        if missing:
            raise ValueError(f"轨迹缺少字段：{sorted(missing)}")

        arm_qpos = np.asarray(data["arm_qpos"], dtype=np.float32)
        hand_left = np.asarray(data["hand_left"], dtype=np.float32)
        hand_right = np.asarray(data["hand_right"], dtype=np.float32)
        hand_qpos = np.asarray(data["hand_qpos"], dtype=np.float32)
        actions = np.asarray(data["joint26"], dtype=np.float32)
        timestamps = np.asarray(data["timestamps"], dtype=np.float64)
        joint_names = tuple(str(name) for name in data["joint_names"].tolist())
        hand_names = tuple(str(name) for name in data["hand_names"].tolist())
        fps = float(_scalar(data, "fps"))
        episode_index = int(_scalar(data, "episode_index"))
        arm_units = str(_scalar(data, "arm_units"))
        hand_units = str(_scalar(data, "hand_units"))

    if episode_index != 20:
        raise ValueError(f"只允许回放 episode 20，文件中为 episode {episode_index}")
    if fps <= 0.0 or not np.isfinite(fps):
        raise ValueError(f"无效 fps：{fps}")
    if joint_names != EXPECTED_ARM_NAMES:
        raise ValueError(f"双臂关节顺序不匹配：{joint_names}")
    if hand_names != EXPECTED_HAND_NAMES:
        raise ValueError(f"BrainCo 单手顺序不匹配：{hand_names}")
    if arm_units != "radian" or hand_units != "normalized_command":
        raise ValueError(f"单位不匹配：arm={arm_units}, hand={hand_units}")

    frame_count = actions.shape[0]
    expected_shapes = {
        "joint26": (frame_count, ACTION_DIM),
        "arm_qpos": (frame_count, ARM_DIM),
        "hand_left": (frame_count, 6),
        "hand_right": (frame_count, 6),
        "hand_qpos": (frame_count, HAND_DIM),
        "timestamps": (frame_count,),
    }
    actual_shapes = {
        "joint26": actions.shape,
        "arm_qpos": arm_qpos.shape,
        "hand_left": hand_left.shape,
        "hand_right": hand_right.shape,
        "hand_qpos": hand_qpos.shape,
        "timestamps": timestamps.shape,
    }
    for name, expected in expected_shapes.items():
        if actual_shapes[name] != expected:
            raise ValueError(f"{name} shape 错误：期望 {expected}，实际 {actual_shapes[name]}")
    if frame_count < 2:
        raise ValueError("轨迹至少需要两帧")

    arrays = (actions, arm_qpos, hand_left, hand_right, hand_qpos, timestamps)
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("轨迹包含 NaN 或 Inf")
    if not np.allclose(hand_qpos, np.concatenate((hand_left, hand_right), axis=1), atol=1e-7):
        raise ValueError("hand_qpos 不等于 left6 + right6")
    if not np.allclose(actions, np.concatenate((arm_qpos, hand_qpos), axis=1), atol=1e-7):
        raise ValueError("joint26 不等于 arm14 + BrainCo left6 + right6")
    if float(hand_qpos.min()) < 0.0 or float(hand_qpos.max()) > 1.0:
        raise ValueError(f"BrainCo 指令超出归一化范围 [0, 1]：[{hand_qpos.min():.6f}, {hand_qpos.max():.6f}]")

    time_deltas = np.diff(timestamps)
    expected_dt = 1.0 / fps
    if timestamps[0] != 0.0 or np.any(time_deltas <= 0.0):
        raise ValueError("timestamps 必须从 0 开始并严格递增")
    # Export timestamps originate from float32 frame times; allow their sub-microsecond rounding error.
    if not np.allclose(time_deltas, expected_dt, rtol=1e-5, atol=1e-6):
        raise ValueError(f"timestamps 不是稳定的 {fps:g} Hz")

    arm_velocity = np.abs(np.diff(arm_qpos, axis=0) / time_deltas[:, None])
    max_arm_velocity = float(arm_velocity.max())
    if max_arm_velocity > MAX_ARM_SPEED_RAD_S + 1e-5:
        raise ValueError(f"轨迹最大双臂速度 {max_arm_velocity:.6f} rad/s 超过接口上限 {MAX_ARM_SPEED_RAD_S:.6f} rad/s")

    _validate_urdf_limits(arm_qpos, urdf_path)
    return EpisodeTrajectory(
        actions=np.ascontiguousarray(actions),
        arm_qpos=np.ascontiguousarray(arm_qpos),
        hand_qpos=np.ascontiguousarray(hand_qpos),
        timestamps=timestamps,
        fps=fps,
    )


def _print_summary(trajectory: EpisodeTrajectory, path: Path) -> None:
    time_deltas = np.diff(trajectory.timestamps)
    arm_velocity = np.abs(np.diff(trajectory.arm_qpos, axis=0) / time_deltas[:, None])
    print(f"轨迹文件: {path}")
    print("episode: 20")
    print(f"帧数/频率: {len(trajectory.actions)} / {trajectory.fps:g} Hz")
    print(f"发送时长: {len(trajectory.actions) / trajectory.fps:.3f} s")
    print("动作顺序: 双臂关节 left7 + right7 + BrainCo left6 + right6")
    print(f"双臂最大速度: {arm_velocity.max():.6f} rad/s")
    print(f"BrainCo 范围: [{trajectory.hand_qpos.min():.6f}, {trajectory.hand_qpos.max():.6f}]")


def _smooth_transition(current: np.ndarray, target: np.ndarray, frame_count: int) -> np.ndarray:
    phase = np.arange(1, frame_count + 1, dtype=np.float64) / frame_count
    alpha = 3.0 * phase**2 - 2.0 * phase**3
    transition = current[None, :] + alpha[:, None] * (target - current)[None, :]
    return np.ascontiguousarray(transition, dtype=np.float32)


def _validate_deploy_action_layout() -> None:
    from unitree_deploy.robot.robot_configs import brainco_motors
    from unitree_deploy.robot.robot_configs import g1_motors

    actual_arm_motors = tuple(g1_motors)
    actual_hand_motors = tuple(brainco_motors)
    if actual_arm_motors != EXPECTED_DEPLOY_ARM_MOTORS:
        raise RuntimeError(f"unitree-deploy 双臂动作顺序已变化：{actual_arm_motors}")
    if actual_hand_motors != EXPECTED_DEPLOY_HAND_MOTORS:
        raise RuntimeError(f"unitree-deploy BrainCo 动作顺序已变化：{actual_hand_motors}")


def execute_trajectory(
    trajectory: EpisodeTrajectory,
    network_interface: str | None,
    transition_seconds: float,
) -> None:
    if transition_seconds <= 0.0:
        raise ValueError("--transition-seconds 必须大于 0")

    sys.path.insert(0, str(ROBOT_INTERFACE_DIR))
    from robot_interface import UnitreeRobotInterface

    _validate_deploy_action_layout()

    dt = 1.0 / trajectory.fps
    robot = UnitreeRobotInterface(
        robot_type="unitree_g1_brainco",
        dt=dt,
        network_interface=network_interface,
    )
    try:
        print("正在连接 unitree_g1_brainco ...")
        robot.connect()
        observation = robot.get_observation(prompt="replay exported episode 20")
        current = np.asarray(observation["state"], dtype=np.float32)
        if current.shape != (ACTION_DIM,) or not np.isfinite(current).all():
            raise ValueError(f"机器人当前 state 必须为有限的 26 维向量，实际 shape={current.shape}")

        first_action = trajectory.actions[0]
        arm_delta = float(np.max(np.abs(first_action[:ARM_DIM] - current[:ARM_DIM])))
        hand_delta = float(np.max(np.abs(first_action[ARM_DIM:] - current[ARM_DIM:])))
        print(f"当前状态到首帧最大差值: arm={arm_delta:.6f} rad, BrainCo={hand_delta:.6f}")
        confirmation = input("确认机器人周围安全后，输入 EXECUTE EPISODE 20：").strip()
        if confirmation != "EXECUTE EPISODE 20":
            print("确认不匹配，未发送任何动作。")
            return

        transition_frames = max(1, round(transition_seconds * trajectory.fps))
        transition = _smooth_transition(current, first_action, transition_frames)
        peak_transition_arm_speed = 1.5 * arm_delta / transition_seconds
        if peak_transition_arm_speed > MAX_ARM_SPEED_RAD_S:
            raise ValueError(
                f"过渡段估计峰值速度 {peak_transition_arm_speed:.6f} rad/s 超过 "
                f"{MAX_ARM_SPEED_RAD_S:.6f} rad/s；请增大 --transition-seconds"
            )

        print(f"用 {transition_frames} 步（{transition_seconds:.3f} s）平滑移动到首帧。")
        for action in transition:
            robot.step(action)

        input("已到达首帧。再次确认环境安全，按 Enter 开始回放；Ctrl-C 取消：")
        for action in trajectory.actions:
            robot.step(action)
        print("episode 20 回放完成，机器人保持在最后一帧目标位置。")
    finally:
        robot.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离线校验或真机回放 episode 20 的 G1 双臂关节 + BrainCo 轨迹。")
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--execute", action="store_true", help="连接真机并回放；默认仅离线校验")
    parser.add_argument("--network-interface", default=None)
    parser.add_argument("--transition-seconds", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trajectory = load_and_validate(args.trajectory.resolve(), args.urdf.resolve())
    _print_summary(trajectory, args.trajectory.resolve())
    if not args.execute:
        print("离线校验通过；未连接机器人，也未发送动作。")
        return
    execute_trajectory(trajectory, args.network_interface, args.transition_seconds)


if __name__ == "__main__":
    main()
