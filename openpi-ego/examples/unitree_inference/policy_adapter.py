"""Metadata-driven joint/EEF adapter around the OpenPI websocket client."""

from __future__ import annotations

import logging
import time
from typing import Protocol

from g1_kinematics import G1Kinematics
import numpy as np

ROBOT_DIMS = {"unitree_g1_dex1": 16, "unitree_g1_brainco": 26}
ACTION_DOMAINS = {
    "gripper": {"model_dim": 20, "robot_type": "unitree_g1_dex1", "end_effector": "dex1", "tail": "gripper2"},
    "brainco": {
        "model_dim": 30,
        "robot_type": "unitree_g1_brainco",
        "end_effector": "brainco",
        "tail": "brainco12",
    },
}


class Policy(Protocol):
    def infer(self, observation: dict) -> dict: ...


class EEFPolicyAdapter:
    def __init__(self, policy: Policy, metadata: dict, robot_type: str, urdf_path: str | None = None) -> None:
        self._policy = policy
        self._model_dim = int(metadata["model_dim"])
        self._model_tail = self._model_dim - 18
        self._robot_dim = ROBOT_DIMS[robot_type]
        self._robot_tail = self._robot_dim - 14
        action_domain = metadata.get("action_domain")
        if action_domain is not None:
            if action_domain not in ACTION_DOMAINS:
                raise ValueError(f"Unsupported policy action domain: {action_domain!r}")
            contract = ACTION_DOMAINS[action_domain]
            mismatches = {
                key: (actual, expected)
                for key, actual, expected in (
                    ("model_dim", self._model_dim, contract["model_dim"]),
                    ("robot_type", robot_type, contract["robot_type"]),
                    ("end_effector", metadata.get("end_effector"), contract["end_effector"]),
                    ("action_tail", metadata.get("action_tail"), contract["tail"]),
                )
                if actual != expected
            }
            if mismatches:
                raise ValueError(f"Policy action-domain metadata is inconsistent: {mismatches}")
        expected_effector = metadata["end_effector"]
        actual_effector = "brainco" if robot_type == "unitree_g1_brainco" else "dex1"
        if expected_effector not in ("none", actual_effector):
            raise ValueError(
                f"Config expects {expected_effector}, but selected robot {robot_type} provides {actual_effector}"
            )
        if self._model_tail not in (0, self._robot_tail):
            raise ValueError(
                f"Policy tail dimension {self._model_tail} is incompatible with robot tail {self._robot_tail}"
            )
        reference_frame = str(metadata.get("reference_frame", "torso_link"))
        eef_links = metadata.get("eef_links", ["left_hand_palm_link", "right_hand_palm_link"])
        if (
            not isinstance(eef_links, list | tuple)
            or len(eef_links) != 2
            or not all(isinstance(link, str) for link in eef_links)
        ):
            raise ValueError(f"Server metadata has invalid eef_links: {eef_links!r}")
        frame_convention = metadata.get("frame_convention", "T_reference_eef")
        if frame_convention != "T_reference_eef":
            raise ValueError(f"Unsupported EEF frame convention: {frame_convention!r}")
        self._kinematics = G1Kinematics(
            urdf_path,
            rotation_format=metadata["rotation_format"],
            reference_frame=reference_frame,
            left_eef_frame=eef_links[0],
            right_eef_frame=eef_links[1],
        )
        expected_urdf_sha256 = metadata.get("urdf_sha256")
        if expected_urdf_sha256 is not None and self._kinematics.urdf_sha256 != expected_urdf_sha256:
            raise ValueError(
                "Inference URDF does not match the URDF used for joint-to-EEF conversion: "
                f"client={self._kinematics.urdf_sha256}, checkpoint={expected_urdf_sha256}. "
                "Pass --urdf-path pointing to the exact same G1-D URDF file."
            )
        logging.info(
            "EEF FK/IK contract: %s -> %s, %s; URDF SHA256=%s",
            reference_frame,
            eef_links[0],
            eef_links[1],
            self._kinematics.urdf_sha256,
        )
        self._infer_count = 0

    def _joint_to_model(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.size != self._robot_dim:
            raise ValueError(f"Expected {self._robot_dim}-D robot joints, got {values.size}")
        return np.concatenate((self._kinematics.forward(values[:14]), values[14 : 14 + self._model_tail]))

    def _joint_chunk_to_model(self, chunk: np.ndarray) -> np.ndarray:
        chunk = np.asarray(chunk, dtype=np.float64)
        if chunk.ndim == 1:
            chunk = chunk[None, :]
        return np.stack([self._joint_to_model(row) for row in chunk])

    def infer(self, observation: dict) -> dict:
        observation = dict(observation)
        joint_state = np.asarray(observation["state"], dtype=np.float64).reshape(-1)
        if joint_state.size != self._robot_dim:
            raise ValueError(f"Expected {self._robot_dim}-D robot state, got {joint_state.size}")
        observation["state"] = self._joint_to_model(joint_state)
        if "prev_action_chunk" in observation:
            observation["prev_action_chunk"] = self._joint_chunk_to_model(observation["prev_action_chunk"])

        # The FK state and previous chunk use the exact training layout. For
        # EEF20 this is EEF18 + absolute left/right gripper; for EEF30 it is
        # EEF18 + absolute left/right BrainCo6.
        started = time.perf_counter()
        result = self._policy.infer(observation)
        actions = np.asarray(result["actions"], dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != self._model_dim:
            raise ValueError(f"Expected policy actions [H, {self._model_dim}], got {actions.shape}")
        converted = np.empty((actions.shape[0], self._robot_dim), dtype=np.float32)
        seed = joint_state[:14].copy()
        for index, action in enumerate(actions):
            converted[index, :14] = self._kinematics.inverse(action[:18], seed)
            seed = converted[index, :14]
            if self._model_tail:
                # Tail values are absolute end-effector commands and bypass IK.
                # In the gripper domain this copies action[18:20] directly to
                # the deploy environment's left/right gripper slots [14:16].
                converted[index, 14:] = action[18 : 18 + self._robot_tail]
            else:
                converted[index, 14:] = joint_state[14:]
        self._infer_count += 1
        if self._infer_count % 50 == 0:
            elapsed_ms = (time.perf_counter() - started) * 1000
            logging.info("EEF policy plus IK chunk: %.1f ms", elapsed_ms)
        return {**result, "actions": converted}

    def reset(self) -> None:
        self._kinematics.reset()


# 这个主要是为了适配不同的类型eef会ik为joint
def adapt_policy(policy: Policy, metadata: dict, robot_type: str, urdf_path: str | None = None) -> Policy:
    required = {
        "action_space",
        "end_effector",
        "model_dim",
        "robot_type",
        "rotation_format",
    }
    missing = required - metadata.keys()
    if missing:
        raise ValueError(f"Server metadata is missing: {sorted(missing)}")
    if robot_type not in ROBOT_DIMS:
        raise ValueError(f"Unsupported robot type: {robot_type}")
    if metadata["action_space"] == "eef":  # 动作空间如果是EEF则直接执行
        return EEFPolicyAdapter(policy, metadata, robot_type, urdf_path)
    if metadata["action_space"] != "joint":
        raise ValueError(f"Unsupported policy action space: {metadata['action_space']}")
    if int(metadata["model_dim"]) != ROBOT_DIMS[robot_type]:
        raise ValueError(
            f"Joint policy dimension {metadata['model_dim']} does not match {robot_type} ({ROBOT_DIMS[robot_type]})"
        )
    return policy
