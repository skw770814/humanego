from __future__ import annotations

import dataclasses
from typing import Literal

import numpy as np

from openpi import transforms

CameraMode = Literal["single", "three"]
Rotation6DFormat = Literal["columns", "columns_grouped", "rows"]
HumanInputFrame = Literal[
    "native",
    "torso_palm",
    "g1_base_tcp",
    "recording_tcp",
    "pelvis_wrist",
]


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
        raise ValueError(f"Expected an image with 1, 3, or 4 channels, got {image.shape}")
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[-1] == 4:
        image = image[..., :3]
    return np.ascontiguousarray(image, dtype=np.uint8)


def _camera_inputs(images: dict, camera_mode: CameraMode, *, allow_missing: bool = False) -> tuple[dict, dict]:
    if camera_mode not in ("single", "three"):
        raise ValueError(f"camera_mode must be single or three, got {camera_mode!r}")
    if "cam_high" not in images:
        raise ValueError(f"cam_high is required, received {tuple(images)}")
    base = _parse_image(images["cam_high"])
    model_images = {"base_0_rgb": base}
    masks = {"base_0_rgb": np.True_}
    for model_key, source_key in (
        ("left_wrist_0_rgb", "cam_left_wrist"),
        ("right_wrist_0_rgb", "cam_right_wrist"),
    ):
        if camera_mode == "three":
            if source_key not in images:
                if not allow_missing:
                    raise ValueError(f"{source_key} is required for three-camera inference")
                model_images[model_key] = np.zeros_like(base)
                masks[model_key] = np.False_
            else:
                model_images[model_key] = _parse_image(images[source_key])
                masks[model_key] = np.True_
        else:
            model_images[model_key] = np.zeros_like(base)
            masks[model_key] = np.False_
    return model_images, masks


def _validate_vector(value: np.ndarray, expected_dim: int, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.shape[-1] != expected_dim:
        raise ValueError(f"Expected {name} dimension {expected_dim}, got shape {value.shape}")
    return value


@dataclasses.dataclass(frozen=True)
class UnitreeRobotInputs(transforms.DataTransformFn):
    state_dim: int
    camera_mode: CameraMode

    def __call__(self, data: dict) -> dict:
        images, image_masks = _camera_inputs(data["images"], self.camera_mode)
        state = _validate_vector(data["state"], self.state_dim, "state")
        result = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
            # Kept outside normalization/model inputs for exact relative-action
            # decoding even when robust normalized state clipping is enabled.
            "action_reference_state": state.copy(),
        }
        if "actions" in data:
            result["actions"] = _validate_vector(data["actions"], self.state_dim, "actions")
        if "action_is_pad" in data:
            result["action_pad_mask"] = np.asarray(data["action_is_pad"], dtype=np.bool_)
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class UnitreeRobotOutputs(transforms.DataTransformFn):
    action_dim: int

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., : self.action_dim]}


def _rotation_converters(rotation_format: Rotation6DFormat):
    if rotation_format == "columns":
        return _rot6d_columns_to_matrix, _matrix_to_rot6d_columns
    if rotation_format == "columns_grouped":
        return _rot6d_grouped_columns_to_matrix, _matrix_to_rot6d_grouped_columns
    if rotation_format == "rows":
        return _rot6d_rows_to_matrix, _matrix_to_rot6d_rows
    raise ValueError(f"Unsupported rotation format: {rotation_format}")


def _rot6d_columns_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    columns = np.asarray(rot6d).reshape(*np.asarray(rot6d).shape[:-1], 3, 2)
    column0 = columns[..., :, 0]
    column0 = column0 / np.maximum(np.linalg.norm(column0, axis=-1, keepdims=True), 1e-8)
    column1 = columns[..., :, 1] - np.sum(column0 * columns[..., :, 1], axis=-1, keepdims=True) * column0
    column1 = column1 / np.maximum(np.linalg.norm(column1, axis=-1, keepdims=True), 1e-8)
    return np.stack((column0, column1, np.cross(column0, column1)), axis=-1)


def _matrix_to_rot6d_columns(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation)
    return rotation[..., :, :2].reshape(*rotation.shape[:-2], 6)


def _rot6d_grouped_columns_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    columns = np.asarray(rot6d).reshape(*np.asarray(rot6d).shape[:-1], 2, 3)
    column0 = columns[..., 0, :]
    column0 = column0 / np.maximum(np.linalg.norm(column0, axis=-1, keepdims=True), 1e-8)
    column1 = columns[..., 1, :] - np.sum(column0 * columns[..., 1, :], axis=-1, keepdims=True) * column0
    column1 = column1 / np.maximum(np.linalg.norm(column1, axis=-1, keepdims=True), 1e-8)
    return np.stack((column0, column1, np.cross(column0, column1)), axis=-1)


def _matrix_to_rot6d_grouped_columns(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation)
    return np.concatenate((rotation[..., :, 0], rotation[..., :, 1]), axis=-1)


def _transform_grouped_pose(
    pose: np.ndarray,
    torso_in_pelvis: np.ndarray,
    wrist_to_palm: np.ndarray,
    input_frame: HumanInputFrame,
) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    rotation = _rot6d_grouped_columns_to_matrix(pose[..., 3:9])
    if input_frame in ("g1_base_tcp", "pelvis_wrist"):
        palm_position = pose[..., :3] - torso_in_pelvis + np.einsum("...ij,j->...i", rotation, wrist_to_palm)
    elif input_frame == "recording_tcp":
        # The recording/world reference cannot be converted to torso_link
        # without a per-frame placement.  Moving the control point from the
        # wrist-yaw TCP to the palm is nevertheless exact and is sufficient
        # before a reference-frame-invariant relative-action conversion.
        palm_position = pose[..., :3] + np.einsum("...ij,j->...i", rotation, wrist_to_palm)
    elif input_frame in ("native", "torso_palm"):
        palm_position = pose[..., :3]
    else:
        raise ValueError(f"Unsupported Human input_frame: {input_frame!r}")
    return np.concatenate((palm_position, _matrix_to_rot6d_grouped_columns(rotation)), axis=-1)


@dataclasses.dataclass(frozen=True)
class CanonicalizeHumanEEF(transforms.DataTransformFn):
    """Canonicalize Human EEF poses to grouped-column Rot6D.

    ``g1_base_tcp`` is the canonical Mode1 default: it converts the already
    retargeted G1 base / wrist-yaw TCP pose to OpenPI's torso / palm contract.
    ``native`` is retained only as an explicit diagnostic/no-position-change
    profile.
    ``recording_tcp`` moves Mode2's control point from wrist-yaw TCP to palm
    while preserving its recording/world reference; it is valid only before
    relative-action conversion. ``pelvis_wrist`` is retained as an alias for
    the old Mode1 conversion. Every path orthonormalizes and serializes Rot6D
    as grouped columns.
    """

    action_dim: int = 30
    input_frame: HumanInputFrame = "g1_base_tcp"

    def __call__(self, data: dict) -> dict:
        torso_in_pelvis = np.asarray([-0.0039635, 0.0, 0.044], dtype=np.float32)
        offsets = (
            np.asarray([0.0415, 0.003, 0.0], dtype=np.float32),
            np.asarray([0.0415, -0.003, 0.0], dtype=np.float32),
        )
        for key in ("state", "actions"):
            if key not in data:
                continue
            value = _validate_vector(data[key], self.action_dim, key)
            data[key] = np.concatenate(
                (
                    _transform_grouped_pose(value[..., :9], torso_in_pelvis, offsets[0], self.input_frame),
                    _transform_grouped_pose(value[..., 9:18], torso_in_pelvis, offsets[1], self.input_frame),
                    value[..., 18 : self.action_dim],
                ),
                axis=-1,
            )
        return data


def _rot6d_rows_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    rows = np.asarray(rot6d).reshape(*np.asarray(rot6d).shape[:-1], 2, 3)
    row0 = rows[..., 0, :]
    row0 = row0 / np.maximum(np.linalg.norm(row0, axis=-1, keepdims=True), 1e-8)
    row1 = rows[..., 1, :] - np.sum(row0 * rows[..., 1, :], axis=-1, keepdims=True) * row0
    row1 = row1 / np.maximum(np.linalg.norm(row1, axis=-1, keepdims=True), 1e-8)
    return np.stack((row0, row1, np.cross(row0, row1)), axis=-2)


def _matrix_to_rot6d_rows(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation)
    return rotation[..., :2, :].reshape(*rotation.shape[:-2], 6)


def _relative_pose(action_pose: np.ndarray, state_pose: np.ndarray, rotation_format: Rotation6DFormat) -> np.ndarray:
    to_matrix, to_rot6d = _rotation_converters(rotation_format)
    state_rotation = to_matrix(state_pose[..., 3:9])
    action_rotation = to_matrix(action_pose[..., 3:9])
    translation = np.einsum("...ji,...j->...i", state_rotation, action_pose[..., :3] - state_pose[..., :3])
    rotation = np.einsum("...ji,...jk->...ik", state_rotation, action_rotation)
    return np.concatenate((translation, to_rot6d(rotation)), axis=-1)


def _absolute_pose(relative_pose: np.ndarray, state_pose: np.ndarray, rotation_format: Rotation6DFormat) -> np.ndarray:
    to_matrix, to_rot6d = _rotation_converters(rotation_format)
    state_rotation = to_matrix(state_pose[..., 3:9])
    relative_rotation = to_matrix(relative_pose[..., 3:9])
    translation = state_pose[..., :3] + np.einsum("...ij,...j->...i", state_rotation, relative_pose[..., :3])
    rotation = np.einsum("...ij,...jk->...ik", state_rotation, relative_rotation)
    return np.concatenate((translation, to_rot6d(rotation)), axis=-1)


def _state_for_actions(state: np.ndarray, actions: np.ndarray) -> np.ndarray:
    state = np.asarray(state)
    while state.ndim < actions.ndim:
        state = np.expand_dims(state, axis=-2)
    return state


@dataclasses.dataclass(frozen=True)
class RelativeEEFActions(transforms.DataTransformFn):
    action_dim: int
    rotation_format: Rotation6DFormat = "columns"

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data
        actions = _validate_vector(data["actions"], self.action_dim, "actions")
        state = _state_for_actions(_validate_vector(data["state"], self.action_dim, "state"), actions)
        data["actions"] = np.concatenate(
            (
                _relative_pose(actions[..., :9], state[..., :9], self.rotation_format),
                _relative_pose(actions[..., 9:18], state[..., 9:18], self.rotation_format),
                actions[..., 18 : self.action_dim],
            ),
            axis=-1,
        )
        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteEEFActions(transforms.DataTransformFn):
    action_dim: int
    rotation_format: Rotation6DFormat = "columns"

    def __call__(self, data: dict) -> dict:
        actions = _validate_vector(data["actions"][..., : self.action_dim], self.action_dim, "actions")
        reference_state = data.get("action_reference_state", data["state"])
        state = _state_for_actions(
            _validate_vector(reference_state[..., : self.action_dim], self.action_dim, "state"), actions
        )
        data["actions"] = np.concatenate(
            (
                _absolute_pose(actions[..., :9], state[..., :9], self.rotation_format),
                _absolute_pose(actions[..., 9:18], state[..., 9:18], self.rotation_format),
                actions[..., 18 : self.action_dim],
            ),
            axis=-1,
        )
        return data
