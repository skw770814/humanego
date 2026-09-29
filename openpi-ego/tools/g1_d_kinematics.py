"""Dependency-light forward kinematics for the two G1-D arms."""

from __future__ import annotations

import dataclasses
from pathlib import Path
import xml.etree.ElementTree as ET

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
REFERENCE_LINK = "torso_link"
LEFT_EEF_LINK = "left_hand_palm_link"
RIGHT_EEF_LINK = "right_hand_palm_link"
ROTATION_6D_LAYOUT = "columns_grouped:[r00,r10,r20,r01,r11,r21]"


@dataclasses.dataclass(frozen=True)
class UrdfJoint:
    name: str
    joint_type: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray


def _vector(element: ET.Element | None, attribute: str, default: str) -> np.ndarray:
    text = default if element is None else element.attrib.get(attribute, default)
    value = np.fromstring(text, sep=" ", dtype=np.float64)
    if value.shape != (3,):
        raise ValueError(f"Expected three values for {attribute}, got {text!r}")
    return value


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _origin_transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _rpy_matrix(rpy)
    transform[:3, 3] = xyz
    return transform


def _axis_angle(axis: np.ndarray, angles: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(axis)
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"Rotation axis must be finite and non-zero, got {axis}")
    axis = axis / norm
    x, y, z = axis
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    outer = np.outer(axis, axis)
    identity = np.eye(3, dtype=np.float64)
    cosine = np.cos(angles)[:, None, None]
    sine = np.sin(angles)[:, None, None]
    return cosine * identity + (1.0 - cosine) * outer + sine * skew


class G1DForwardKinematics:
    """Compute palm poses relative to torso_link from the 14 arm joints."""

    def __init__(
        self,
        urdf_path: str | Path,
        *,
        reference_link: str = REFERENCE_LINK,
        left_eef_link: str = LEFT_EEF_LINK,
        right_eef_link: str = RIGHT_EEF_LINK,
    ) -> None:
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        if not self.urdf_path.is_file():
            raise FileNotFoundError(self.urdf_path)
        root = ET.parse(self.urdf_path).getroot()
        links = {element.attrib["name"] for element in root.findall("link")}
        for link in (reference_link, left_eef_link, right_eef_link):
            if link not in links:
                raise ValueError(f"URDF does not contain link {link!r}")

        joints = {}
        for element in root.findall("joint"):
            origin = element.find("origin")
            axis = element.find("axis")
            joint = UrdfJoint(
                name=element.attrib["name"],
                joint_type=element.attrib["type"],
                parent=element.find("parent").attrib["link"],  # type: ignore[union-attr]
                child=element.find("child").attrib["link"],  # type: ignore[union-attr]
                xyz=_vector(origin, "xyz", "0 0 0"),
                rpy=_vector(origin, "rpy", "0 0 0"),
                axis=_vector(axis, "xyz", "1 0 0"),
            )
            if joint.child in joints:
                raise ValueError(f"Multiple parent joints for URDF link {joint.child!r}")
            joints[joint.child] = joint

        self._reference_link = reference_link
        self._left_chain = self._chain(joints, reference_link, left_eef_link)
        self._right_chain = self._chain(joints, reference_link, right_eef_link)
        movable = tuple(joint.name for joint in (*self._left_chain, *self._right_chain) if joint.joint_type != "fixed")
        expected = (*LEFT_ARM_JOINTS, *RIGHT_ARM_JOINTS)
        if movable != expected:
            raise ValueError(f"Unexpected G1-D arm chain. Expected {expected}, got {movable}")
        for joint in (*self._left_chain, *self._right_chain):
            if joint.joint_type in {"revolute", "continuous", "prismatic"}:
                norm = np.linalg.norm(joint.axis)
                if not np.isfinite(norm) or norm <= 1e-12:
                    raise ValueError(f"Joint {joint.name!r} has an invalid axis: {joint.axis}")
        self._joint_columns = {name: index for index, name in enumerate(expected)}

    @staticmethod
    def _chain(joints: dict[str, UrdfJoint], reference: str, target: str) -> tuple[UrdfJoint, ...]:
        reverse_chain = []
        current = target
        while current != reference:
            try:
                joint = joints[current]
            except KeyError as exc:
                raise ValueError(f"No URDF chain from {reference!r} to {target!r}") from exc
            reverse_chain.append(joint)
            current = joint.parent
        return tuple(reversed(reverse_chain))

    def _forward_chain(self, arm_joints: np.ndarray, chain: tuple[UrdfJoint, ...]) -> np.ndarray:
        batch_size = arm_joints.shape[0]
        transform = np.broadcast_to(np.eye(4, dtype=np.float64), (batch_size, 4, 4)).copy()
        for joint in chain:
            transform = transform @ _origin_transform(joint.xyz, joint.rpy)
            if joint.joint_type in {"revolute", "continuous"}:
                rotation = np.broadcast_to(np.eye(4), (batch_size, 4, 4)).copy()
                rotation[:, :3, :3] = _axis_angle(joint.axis, arm_joints[:, self._joint_columns[joint.name]])
                transform = transform @ rotation
            elif joint.joint_type == "prismatic":
                translation = np.broadcast_to(np.eye(4), (batch_size, 4, 4)).copy()
                translation[:, :3, 3] = arm_joints[:, self._joint_columns[joint.name], None] * joint.axis[None, :]
                transform = transform @ translation
            elif joint.joint_type != "fixed":
                raise ValueError(f"Unsupported URDF joint type {joint.joint_type!r}")
        return transform

    @staticmethod
    def _pose9(transform: np.ndarray) -> np.ndarray:
        # Canonical Zhou-style 6D rotation: first column followed by second column,
        # [r00,r10,r20,r01,r11,r21]. Do not replace this with R[:, :2].reshape(6),
        # which interleaves the columns and is retained only by legacy evaluation configs.
        rotation6d = np.concatenate((transform[:, :3, 0], transform[:, :3, 1]), axis=-1)
        return np.concatenate((transform[:, :3, 3], rotation6d), axis=-1)

    def forward(self, arm_joints: np.ndarray) -> np.ndarray:
        arm_joints = np.asarray(arm_joints, dtype=np.float64)
        if arm_joints.ndim not in {1, 2}:
            raise ValueError(f"Expected shape (14,) or (N, 14), got {arm_joints.shape}")
        squeeze = arm_joints.ndim == 1
        arm_joints = np.atleast_2d(arm_joints)
        if arm_joints.shape[1] != 14:
            raise ValueError(f"Expected 14 arm joints, got {arm_joints.shape}")
        if arm_joints.shape[0] == 0:
            raise ValueError("At least one arm-joint sample is required")
        if not np.isfinite(arm_joints).all():
            raise ValueError("Arm-joint input contains NaN or infinity")
        result = np.concatenate(
            (
                self._pose9(self._forward_chain(arm_joints, self._left_chain)),
                self._pose9(self._forward_chain(arm_joints, self._right_chain)),
            ),
            axis=-1,
        ).astype(np.float32)
        return result[0] if squeeze else result
