"""Unitree G1 dual-arm FK/IK in the policy training frames.

Only the 14 arm joints pass through kinematics. End-effector joints are handled
by ``policy_adapter.py``. Poses use ``torso_link`` as reference and the two palm
links as targets, matching the stable module in the previous OpenPI project.
"""

from __future__ import annotations

import hashlib
import pathlib

import numpy as np

LEFT_ARM_JOINTS = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)
RIGHT_ARM_JOINTS = tuple(name.replace("left_", "right_") for name in LEFT_ARM_JOINTS)


def default_g1_urdf() -> pathlib.Path:
    import unitree_deploy.real_unitree_env as real_unitree_env

    path = pathlib.Path(real_unitree_env.__file__).resolve().parent / "robot_devices/assets/g1/g1_body29_hand14.urdf"
    if not path.is_file():
        raise FileNotFoundError(f"Unitree G1 URDF not found: {path}")
    return path


def rot6d_to_matrix(rot6d: np.ndarray, rotation_format: str) -> np.ndarray:
    value = np.asarray(rot6d, dtype=np.float64)
    if rotation_format == "columns":
        columns = value.reshape(3, 2)
        first = columns[:, 0]
        first = first / max(np.linalg.norm(first), 1e-8)
        second = columns[:, 1] - np.dot(first, columns[:, 1]) * first
        second = second / max(np.linalg.norm(second), 1e-8)
        return np.stack((first, second, np.cross(first, second)), axis=1)
    if rotation_format == "columns_grouped":
        columns = value.reshape(2, 3)
        first = columns[0]
        first = first / max(np.linalg.norm(first), 1e-8)
        second = columns[1] - np.dot(first, columns[1]) * first
        second = second / max(np.linalg.norm(second), 1e-8)
        return np.stack((first, second, np.cross(first, second)), axis=1)
    if rotation_format == "rows":
        rows = value.reshape(2, 3)
        first = rows[0]
        first = first / max(np.linalg.norm(first), 1e-8)
        second = rows[1] - np.dot(first, rows[1]) * first
        second = second / max(np.linalg.norm(second), 1e-8)
        return np.stack((first, second, np.cross(first, second)), axis=0)
    raise ValueError(f"rotation_format must be 'columns', 'columns_grouped', or 'rows', got {rotation_format!r}")


def matrix_to_rot6d(rotation: np.ndarray, rotation_format: str) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation_format == "columns":
        return rotation[:, :2].reshape(6)
    if rotation_format == "columns_grouped":
        return np.concatenate((rotation[:, 0], rotation[:, 1]))
    if rotation_format == "rows":
        return rotation[:2, :].reshape(6)
    raise ValueError(f"rotation_format must be 'columns', 'columns_grouped', or 'rows', got {rotation_format!r}")


class _MovingFilter:
    def __init__(self, weights: tuple[float, ...]) -> None:
        self._weights = np.asarray(weights, dtype=np.float64)
        if not np.isclose(self._weights.sum(), 1.0):
            raise ValueError("IK filter weights must sum to 1")
        self._queue: list[np.ndarray] = []

    def add(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        if self._queue and np.array_equal(value, self._queue[-1]):
            return self._queue[-1].copy()
        self._queue.append(value.copy())
        self._queue = self._queue[-len(self._weights) :]
        if len(self._queue) < len(self._weights):
            return value
        return np.sum(np.asarray(self._queue) * self._weights[:, None], axis=0)

    def clear(self) -> None:
        self._queue.clear()


class G1Kinematics:
    """Pinocchio FK plus regularized damped-least-squares IK."""

    def __init__(
        self,
        urdf_path: str | pathlib.Path | None = None,
        *,
        rotation_format: str = "columns",
        reference_frame: str = "torso_link",
        left_eef_frame: str = "left_hand_palm_link",
        right_eef_frame: str = "right_hand_palm_link",
        max_iterations: int = 50,
        tolerance: float = 1e-6,
        damping: float = 1e-4,
        max_iteration_step: float = 0.2,
        max_solution_delta: float = 0.5,
        nullspace_gain: float = 0.1,
        filter_weights: tuple[float, ...] = (0.4, 0.3, 0.2, 0.1),
    ) -> None:
        try:
            import pinocchio as pin
        except ImportError as exc:
            raise RuntimeError("G1 FK/IK requires Pinocchio in the Unitree client environment") from exc

        if rotation_format not in ("columns", "columns_grouped", "rows"):
            raise ValueError("rotation_format must be 'columns', 'columns_grouped', or 'rows'")
        self._pin = pin
        self.rotation_format = rotation_format
        self._max_iterations = max_iterations
        self._tolerance = tolerance
        self._damping = damping
        self._max_iteration_step = max_iteration_step
        self._max_solution_delta = max_solution_delta
        self._nullspace_gain = nullspace_gain
        self._filter = _MovingFilter(filter_weights)

        self.urdf_path = (default_g1_urdf() if urdf_path is None else pathlib.Path(urdf_path)).expanduser().resolve()
        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"Unitree G1 URDF not found: {self.urdf_path}")
        self.urdf_sha256 = hashlib.sha256(self.urdf_path.read_bytes()).hexdigest()
        self.reference_frame = reference_frame
        self.eef_frames = (left_eef_frame, right_eef_frame)
        self._full_model = pin.buildModelFromUrdf(str(self.urdf_path))
        self._full_data = self._full_model.createData()
        self._neutral = pin.neutral(self._full_model)
        self._left_indices = self._configuration_indices(self._full_model, LEFT_ARM_JOINTS)
        self._right_indices = self._configuration_indices(self._full_model, RIGHT_ARM_JOINTS)
        self._full_reference = self._frame_id(self._full_model, reference_frame)
        self._full_left = self._frame_id(self._full_model, left_eef_frame)
        self._full_right = self._frame_id(self._full_model, right_eef_frame)

        arm_names = set(LEFT_ARM_JOINTS) | set(RIGHT_ARM_JOINTS)
        locked = [
            joint_id
            for joint_id in range(1, self._full_model.njoints)
            if self._full_model.names[joint_id] not in arm_names
        ]
        self._model = pin.buildReducedModel(self._full_model, locked, self._neutral)
        if self._model.nq != 14:
            raise ValueError(f"Reduced G1 arm model must have 14 DoF, got {self._model.nq}")
        self._data = self._model.createData()
        self._reference = self._frame_id(self._model, reference_frame)
        self._left = self._frame_id(self._model, left_eef_frame)
        self._right = self._frame_id(self._model, right_eef_frame)
        self._lower = np.asarray(self._model.lowerPositionLimit, dtype=np.float64)
        self._upper = np.asarray(self._model.upperPositionLimit, dtype=np.float64)

    @staticmethod
    def _configuration_indices(model, names: tuple[str, ...]) -> np.ndarray:
        indices = []
        for name in names:
            joint_id = model.getJointId(name)
            if joint_id == 0 or model.names[joint_id] != name or model.nqs[joint_id] != 1:
                raise ValueError(f"Missing one-DoF joint in G1 URDF: {name}")
            indices.append(model.idx_qs[joint_id])
        return np.asarray(indices, dtype=np.int64)

    @staticmethod
    def _frame_id(model, name: str) -> int:
        frame_id = model.getFrameId(name)
        if frame_id >= model.nframes or model.frames[frame_id].name != name:
            raise ValueError(f"Missing frame in G1 URDF: {name}")
        return frame_id

    def _pose9(self, placement) -> np.ndarray:
        return np.concatenate((placement.translation, matrix_to_rot6d(placement.rotation, self.rotation_format)))

    def forward(self, arm_joints: np.ndarray) -> np.ndarray:
        arm_joints = np.asarray(arm_joints, dtype=np.float64).reshape(-1)
        if arm_joints.size != 14:
            raise ValueError(f"FK expects 14 arm joints, got {arm_joints.size}")
        q = self._neutral.copy()
        q[self._left_indices] = arm_joints[:7]
        q[self._right_indices] = arm_joints[7:]
        self._pin.forwardKinematics(self._full_model, self._full_data, q)
        self._pin.updateFramePlacements(self._full_model, self._full_data)
        reference = self._full_data.oMf[self._full_reference]
        left = reference.inverse() * self._full_data.oMf[self._full_left]
        right = reference.inverse() * self._full_data.oMf[self._full_right]
        return np.concatenate((self._pose9(left), self._pose9(right))).astype(np.float32)

    def _target(self, pose9: np.ndarray, origin) -> object:
        relative = self._pin.SE3(rot6d_to_matrix(pose9[3:9], self.rotation_format), pose9[:3].copy())
        return origin.act(relative)

    def inverse(self, eef_pose: np.ndarray, seed: np.ndarray) -> np.ndarray:
        eef_pose = np.asarray(eef_pose, dtype=np.float64).reshape(-1)
        seed = np.asarray(seed, dtype=np.float64).reshape(-1)
        if eef_pose.size != 18 or seed.size != 14:
            raise ValueError(f"IK expects EEF18 and seed14, got {eef_pose.size} and {seed.size}")
        seed = np.clip(seed, self._lower, self._upper)
        q = seed.copy()
        self._pin.forwardKinematics(self._model, self._data, q)
        self._pin.updateFramePlacements(self._model, self._data)
        origin = self._data.oMf[self._reference]
        targets = (self._target(eef_pose[:9], origin), self._target(eef_pose[9:18], origin))
        damping = self._damping * np.eye(12)

        for _ in range(self._max_iterations):
            self._pin.forwardKinematics(self._model, self._data, q)
            self._pin.updateFramePlacements(self._model, self._data)
            error = np.concatenate(
                (
                    self._pin.log6(self._data.oMf[self._left].actInv(targets[0])).vector,
                    self._pin.log6(self._data.oMf[self._right].actInv(targets[1])).vector,
                )
            )
            if np.linalg.norm(error) < self._tolerance:
                break
            jacobian = np.vstack(
                (
                    self._pin.computeFrameJacobian(self._model, self._data, q, self._left, self._pin.LOCAL),
                    self._pin.computeFrameJacobian(self._model, self._data, q, self._right, self._pin.LOCAL),
                )
            )
            inverse_term = np.linalg.solve(jacobian @ jacobian.T + damping, np.eye(12))
            pseudo_inverse = jacobian.T @ inverse_term
            delta = pseudo_inverse @ error
            delta += (np.eye(14) - pseudo_inverse @ jacobian) @ (self._nullspace_gain * (seed - q))
            norm = np.linalg.norm(delta)
            if norm > self._max_iteration_step:
                delta *= self._max_iteration_step / norm
            q = np.clip(self._pin.integrate(self._model, q, delta), self._lower, self._upper)

        delta = q - seed
        norm = float(np.linalg.norm(delta))
        if norm > self._max_solution_delta:
            q = np.clip(seed + delta * (self._max_solution_delta / norm), self._lower, self._upper)
        return self._filter.add(q).astype(np.float32)

    def reset(self) -> None:
        self._filter.clear()
