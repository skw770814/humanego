from __future__ import annotations

import dataclasses

import numpy as np

from openpi import transforms
from openpi.policies.unitree_robot_policy import CameraMode
from openpi.policies.unitree_robot_policy import _camera_inputs
from openpi.policies.unitree_robot_policy import _validate_vector


@dataclasses.dataclass(frozen=True)
class SelectHumanEEFOnly(transforms.DataTransformFn):
    """Project native Human EEF18 or EEF20-gripper data onto the EEF18 domain."""

    source_dim: int

    def __call__(self, data: dict) -> dict:
        if self.source_dim not in (18, 20):
            raise ValueError(f"Human EEF-only source must be 18D or 20D, got {self.source_dim}D")
        for key in ("state", "actions"):
            if key in data:
                # For EEF20 sources, [18:20] is the recorded left/right gripper
                # tail. It is removed before normalization and never reaches
                # state tokens, action targets, or the action-dimension mask.
                data[key] = _validate_vector(data[key], self.source_dim, key)[..., :18]
        return data


@dataclasses.dataclass(frozen=True)
class HumanInputs(transforms.DataTransformFn):
    state_dim: int
    camera_mode: CameraMode
    model_uses_state: bool
    allow_missing_cameras: bool = False

    def __call__(self, data: dict) -> dict:
        images, image_masks = _camera_inputs(data["images"], self.camera_mode, allow_missing=self.allow_missing_cameras)
        state = data.get("state")
        if state is None:
            if self.model_uses_state:
                raise ValueError("This human policy requires state")
            state = np.zeros(self.state_dim, dtype=np.float32)
        result = {
            "image": images,
            "image_mask": image_masks,
            # no-state pi0.5 configs keep this transport/reference state for
            # normalization and relative-action decoding, but the tokenizer does
            # not expose it to the model when discrete_state_input=False.
            "state": _validate_vector(state, self.state_dim, "state"),
            "action_reference_state": _validate_vector(state, self.state_dim, "state").copy(),
        }
        if "actions" in data:
            result["actions"] = _validate_vector(data["actions"], self.state_dim, "actions")
        if "action_is_pad" in data:
            result["action_pad_mask"] = np.asarray(data["action_is_pad"], dtype=np.bool_)
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class HumanOutputs(transforms.DataTransformFn):
    action_dim: int

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., : self.action_dim]}
