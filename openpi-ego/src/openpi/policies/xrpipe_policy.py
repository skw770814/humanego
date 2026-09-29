"""Transforms for XRPipe Mode1 single-right-hand training.

The stored LeRobot action is an absolute next target in the per-recording
PICO OpenXR world.  Training labels are relative motions of the measured
thumb/index fingertip midpoint.  This module deliberately has no wrist, palm,
flange, robot-base, or dual-arm assumptions.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from openpi import transforms

STATE_BLOCK_DIM = 9
REFERENCE_DIM = 9
ACTION_DIM = 10
UNITY_TO_OPENXR = np.diag(np.asarray([1.0, 1.0, -1.0], dtype=np.float32))


def _validate_vector(value: np.ndarray, expected_dim: int, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.shape[-1] != expected_dim:
        raise ValueError(f"Expected {name} dimension {expected_dim}, got shape {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        scale = 255.0 if image.max(initial=0) <= 1.0 else 1.0
        image = np.clip(image * scale, 0, 255).astype(np.uint8)
    image = np.squeeze(image)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    elif image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.ndim != 3 or image.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected camera0 image with 1, 3, or 4 channels, got {image.shape}")
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[-1] == 4:
        image = image[..., :3]
    return np.ascontiguousarray(image, dtype=np.uint8)


def rot6d_grouped_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Decode ``[R[:,0], R[:,1]]`` and project it back onto SO(3)."""
    columns = np.asarray(rot6d, dtype=np.float32).reshape(*np.asarray(rot6d).shape[:-1], 2, 3)
    column0 = columns[..., 0, :]
    column0 = column0 / np.maximum(np.linalg.norm(column0, axis=-1, keepdims=True), 1e-8)
    column1 = columns[..., 1, :] - np.sum(column0 * columns[..., 1, :], axis=-1, keepdims=True) * column0
    column1 = column1 / np.maximum(np.linalg.norm(column1, axis=-1, keepdims=True), 1e-8)
    return np.stack((column0, column1, np.cross(column0, column1)), axis=-1)


def matrix_to_rot6d_grouped(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float32)
    return np.concatenate((rotation[..., :, 0], rotation[..., :, 1]), axis=-1)


def change_pose9_basis(pose: np.ndarray) -> np.ndarray:
    """Apply ``S @ T @ S`` where ``S=diag(1,1,-1,1)`` to pose9 values."""
    pose = _validate_vector(pose, REFERENCE_DIM, "pose9")
    rotation = rot6d_grouped_to_matrix(pose[..., 3:9])
    converted_translation = np.einsum("ij,...j->...i", UNITY_TO_OPENXR, pose[..., :3])
    converted_rotation = np.einsum("ij,...jk,kl->...il", UNITY_TO_OPENXR, rotation, UNITY_TO_OPENXR)
    return np.concatenate((converted_translation, matrix_to_rot6d_grouped(converted_rotation)), axis=-1).astype(
        np.float32
    )


def relative_pose9(target: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return ``vec9(inv(T_reference) @ T_target)`` in grouped-column Rot6D."""
    target = _validate_vector(target, REFERENCE_DIM, "target pose")
    reference = _validate_vector(reference, REFERENCE_DIM, "reference pose")
    reference_rotation = rot6d_grouped_to_matrix(reference[..., 3:9])
    target_rotation = rot6d_grouped_to_matrix(target[..., 3:9])
    translation = np.einsum("...ji,...j->...i", reference_rotation, target[..., :3] - reference[..., :3])
    rotation = np.einsum("...ji,...jk->...ik", reference_rotation, target_rotation)
    return np.concatenate((translation, matrix_to_rot6d_grouped(rotation)), axis=-1).astype(np.float32)


@dataclasses.dataclass(frozen=True)
class CanonicalizeRelationState(transforms.DataTransformFn):
    """Convert every stored ``T_midpoint_object`` block to XRPipe RH axes."""

    state_dim: int

    def __post_init__(self) -> None:
        if self.state_dim < 10 or (self.state_dim - 1) % STATE_BLOCK_DIM:
            raise ValueError(f"XRPipe state_dim must be 9N+1 with N>=1, got {self.state_dim}")

    def __call__(self, data: dict) -> dict:
        state = _validate_vector(data["state"], self.state_dim, "state")
        converted = state.copy()
        for start in range(0, self.state_dim - 1, STATE_BLOCK_DIM):
            converted[..., start : start + STATE_BLOCK_DIM] = change_pose9_basis(
                state[..., start : start + STATE_BLOCK_DIM]
            )
        # The last value is the current absolute binary gripper state.
        converted[..., -1] = state[..., -1]
        data["state"] = converted
        return data


@dataclasses.dataclass(frozen=True)
class RelativeFingertipActions(transforms.DataTransformFn):
    """Convert absolute OpenXR fingertip targets to current-local deltas."""

    action_dim: int = ACTION_DIM
    reference_dim: int = REFERENCE_DIM

    def __post_init__(self) -> None:
        if self.action_dim != ACTION_DIM or self.reference_dim != REFERENCE_DIM:
            raise ValueError("XRPipe Mode1 requires action_dim=10 and reference_dim=9")

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data
        actions = _validate_vector(data["actions"], self.action_dim, "actions")
        reference = _validate_vector(data["action_reference"], self.reference_dim, "action_reference")
        while reference.ndim < actions.ndim:
            reference = np.expand_dims(reference, axis=-2)
        relative = np.empty_like(actions, dtype=np.float32)
        relative[..., :REFERENCE_DIM] = relative_pose9(actions[..., :REFERENCE_DIM], reference)
        # Gripper supervision is the target frame's absolute binary state.
        relative[..., REFERENCE_DIM:] = actions[..., REFERENCE_DIM:]
        data["actions"] = relative
        return data


@dataclasses.dataclass(frozen=True)
class XRPipeInputs(transforms.DataTransformFn):
    state_dim: int
    action_dim: int = ACTION_DIM

    def __call__(self, data: dict) -> dict:
        images = data.get("images", {})
        if "camera0" not in images:
            raise ValueError(f"XRPipe Mode1 requires camera0, received {tuple(images)}")
        camera0 = _parse_image(images["camera0"])
        result = {
            "image": {
                "base_0_rgb": camera0,
                "left_wrist_0_rgb": np.zeros_like(camera0),
                "right_wrist_0_rgb": np.zeros_like(camera0),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.False_,
            },
            "state": _validate_vector(data["state"], self.state_dim, "state"),
        }
        if "actions" in data:
            result["actions"] = _validate_vector(data["actions"], self.action_dim, "actions")
        if "action_is_pad" in data:
            result["action_pad_mask"] = np.asarray(data["action_is_pad"], dtype=np.bool_)
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class XRPipeOutputs(transforms.DataTransformFn):
    action_dim: int = ACTION_DIM

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., : self.action_dim]}
