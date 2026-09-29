"""Evaluation-only config loader for new G1-D BrainCo/gripper checkpoints.

The checkpoint manifest is the source of truth.  This prevents a relative EEF
checkpoint from accidentally being served with joint/absolute normalization.
Legacy Unitree evaluation remains in ``unitree_config.py`` and is untouched.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import re

from typing_extensions import override

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.policies.unitree_robot_policy as robot_policy
from openpi.training.config import DataConfig
from openpi.training.config import DataConfigFactory
from openpi.training.config import ModelTransformFactory
from openpi.training.config import TrainConfig
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms


@dataclasses.dataclass(frozen=True)
class UnitreeRobotEvalDataConfig(DataConfigFactory):
    repo_id: str = "unitree_g1d_eval"
    asset_id: str = ""
    robot_mode: int = 2
    action_dimension: int = 30
    action_representation: str = "absolute"
    camera_mode: str = "three"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        del assets_dirs
        dimension = self.action_dimension
        transforms = _transforms.Group(
            inputs=[robot_policy.UnitreeRobotInputs(dimension, self.camera_mode)],
            outputs=[robot_policy.UnitreeRobotOutputs(dimension)],
        )
        if self.action_representation == "relative":
            transforms = transforms.push(
                inputs=[robot_policy.RelativeEEFActions(dimension, "columns_grouped")],
                outputs=[robot_policy.AbsoluteEEFActions(dimension, "columns_grouped")],
            )
        return DataConfig(
            repo_id=self.repo_id,
            asset_id=self.asset_id,
            data_transforms=transforms,
            model_transforms=ModelTransformFactory()(model_config),
            use_quantile_norm=True,
            normalization_clip=5.0,
        )


def load_checkpoint_manifest(checkpoint_dir: pathlib.Path | str) -> dict:
    checkpoint_dir = pathlib.Path(checkpoint_dir).expanduser().resolve()
    path = checkpoint_dir / "assets/runtime_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"New Unitree checkpoint manifest not found: {path}. "
            "Use scripts/serve_unitree_policy.py for legacy checkpoints."
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "rotation_6d": "first_column_then_second_column",
        "rotation_6d_layout": "columns_grouped:[r00,r10,r20,r01,r11,r21]",
        "reference_frame": "torso_link",
        "eef_links": ["left_hand_palm_link", "right_hand_palm_link"],
        "frame_convention": "T_reference_eef",
        "dataset_fps": 30,
        "normalization_clip": 5.0,
    }
    mismatches = {key: (manifest.get(key), value) for key, value in expected.items() if manifest.get(key) != value}
    if mismatches:
        raise ValueError(f"Checkpoint representation is incompatible with G1-D evaluation: {mismatches}")
    return manifest


def build_eval_config(checkpoint_dir: pathlib.Path | str) -> TrainConfig:
    manifest = load_checkpoint_manifest(checkpoint_dir)
    human_components = [component for component in manifest.get("components", []) if component.get("kind") == "human"]
    if human_components:
        human_dataset_mode = manifest.get("human_dataset_mode")
        if human_dataset_mode not in ("mode1", "mode2"):
            raise ValueError(
                "Checkpoint contains Human training data but has no explicit valid human_dataset_mode. "
                "This checkpoint predates the Ego action-contract fix and must not be served."
            )
        human_input_frame = manifest.get("human_input_frame")
        valid_input_frames = {
            "mode1": ("g1_base_tcp", "pelvis_wrist"),
            "mode2": ("recording_tcp",),
        }[human_dataset_mode]
        if human_input_frame not in valid_input_frames:
            raise ValueError(
                "Checkpoint Human frame does not match its Ego dataset mode: "
                f"mode={human_dataset_mode}, frame={human_input_frame}, expected={valid_input_frames}. "
                "This checkpoint must not be served."
            )
        component_frame_mismatches = [
            (component.get("name"), component.get("input_frame"))
            for component in human_components
            if component.get("input_frame") != human_input_frame
        ]
        if component_frame_mismatches:
            raise ValueError(
                f"Human component input_frame does not match the checkpoint manifest: {component_frame_mismatches}"
            )
        invalid_contracts = [
            component.get("name")
            for component in human_components
            if component.get("human_dataset_contract", {}).get("mode") != human_dataset_mode
            or component.get("human_dataset_contract", {}).get("stored_action") != "absolute"
        ]
        if invalid_contracts:
            raise ValueError(
                f"Human components have missing/incompatible Ego dataset contracts: {invalid_contracts}"
            )
    robot_components = [component for component in manifest.get("components", []) if component.get("kind") == "robot"]
    if len(robot_components) != 1:
        raise ValueError(f"Expected exactly one robot component in checkpoint manifest, found {len(robot_components)}")
    robot = robot_components[0]
    mode = int(robot["mode"])
    dimension = int(robot["dimension"])
    representation = str(manifest["action_representation"])
    camera_mode = str(robot.get("camera_mode", manifest.get("robot_camera_mode", "three")))
    if camera_mode not in ("single", "three"):
        raise ValueError(f"Unsupported robot camera mode in checkpoint: {camera_mode!r}")
    manifest_camera_mode = manifest.get("robot_camera_mode")
    if manifest_camera_mode is not None and manifest_camera_mode != camera_mode:
        raise ValueError(
            "Robot component camera mode does not match the checkpoint runtime manifest: "
            f"component={camera_mode!r}, manifest={manifest_camera_mode!r}"
        )
    if mode == 1 and representation != "absolute":
        raise ValueError("Robot mode1 evaluation only supports absolute joint actions")
    if mode not in (1, 2) or representation not in ("absolute", "relative"):
        raise ValueError(f"Unsupported robot evaluation layout: mode={mode}, representation={representation}")
    expected_dimensions = {1: {16, 26}, 2: {20, 30}}
    if dimension not in expected_dimensions[mode]:
        raise ValueError(f"Unsupported robot evaluation dimension: mode={mode}, dimension={dimension}")
    action_domain = manifest.get("action_domain", "gripper" if dimension in (16, 20) else "brainco")
    expected_runtime = {
        "brainco": ({1: 26, 2: 30}, "unitree_g1_brainco", "brainco", "brainco12"),
        "gripper": ({1: 16, 2: 20}, "unitree_g1_dex1", "dex1", "gripper2"),
    }
    if action_domain not in expected_runtime:
        raise ValueError(f"Unsupported action_domain in checkpoint: {action_domain!r}")
    dimensions, robot_type, end_effector, action_tail = expected_runtime[action_domain]
    expected_dimension = dimensions[mode]
    runtime_mismatches = {
        key: (manifest.get(key), expected)
        for key, expected in {
            "robot_type": robot_type,
            "end_effector": end_effector,
            "action_tail": action_tail,
            "action_space": "joint" if mode == 1 else "eef",
        }.items()
        # Old BrainCo manifests predate some runtime fields. Their dimension
        # and any fields that are present are still validated.
        if manifest.get(key, expected) != expected
    }
    if dimension != expected_dimension or runtime_mismatches:
        raise ValueError(
            "Checkpoint action domain does not match its runtime layout: "
            f"domain={action_domain}, mode={mode}, dimension={dimension}, mismatches={runtime_mismatches}"
        )
    if mode == 2:
        urdf_sha256 = manifest.get("urdf_sha256")
        if not isinstance(urdf_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", urdf_sha256) is None:
            raise ValueError(f"Robot mode2 checkpoint has no valid converter URDF SHA256: {urdf_sha256!r}")
        component_kinematics = robot.get("eef_kinematics")
        expected_kinematics = {
            "urdf_sha256": urdf_sha256,
            "reference_frame": manifest["reference_frame"],
            "eef_links": manifest["eef_links"],
            "frame_convention": manifest["frame_convention"],
            "rotation_6d": manifest["rotation_6d_layout"],
        }
        if component_kinematics != expected_kinematics:
            raise ValueError(
                "Robot component FK metadata does not match the checkpoint runtime manifest: "
                f"component={component_kinematics!r}, expected={expected_kinematics!r}"
            )

    asset_id = str(robot["asset_id"])
    asset_path = pathlib.Path(checkpoint_dir) / "assets" / asset_id / "norm_stats.json"
    if not asset_path.is_file():
        raise FileNotFoundError(f"Robot normalization asset missing from checkpoint: {asset_path}")
    return TrainConfig(
        name="unitree_g1d_brainco_eval",
        exp_name="eval_only",
        model=pi0_config.Pi0RTCConfig(pi05=True, discrete_state_input=True),
        data=UnitreeRobotEvalDataConfig(
            asset_id=asset_id,
            robot_mode=mode,
            action_dimension=dimension,
            action_representation=representation,
            camera_mode=camera_mode,
        ),
        weight_loader=weight_loaders.NoOpWeightLoader(),
        policy_metadata={**manifest, "robot_camera_mode": camera_mode, "eval_component": robot},
        wandb_enabled=False,
    )
