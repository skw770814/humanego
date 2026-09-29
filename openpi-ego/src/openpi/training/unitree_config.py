from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import pathlib
from typing import Literal

from typing_extensions import override

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.policies.human_policy as human_policy
import openpi.policies.unitree_robot_policy as robot_policy
from openpi.training.config import AssetsConfig
from openpi.training.config import DataConfig
from openpi.training.config import DataConfigFactory
from openpi.training.config import ModelTransformFactory
from openpi.training.config import TrainConfig
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

PolicyKind = Literal["human", "robot"]
ActionSpace = Literal["eef", "joint"]
EndEffector = Literal["dex1", "brainco", "none"]
CameraMode = Literal["single", "three"]
Rotation6DFormat = Literal["columns", "rows"]

_ROOT = pathlib.Path(__file__).resolve().parents[3]
_CHECKPOINT_ROOT = _ROOT / "checkpoints"
_DATASET_ROOT = _ROOT / "dataset"
_BASE_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_base/params"


@dataclasses.dataclass(frozen=True)
class UnitreePolicySpec:
    name: str
    policy_kind: PolicyKind
    mode: int
    action_space: ActionSpace
    end_effector: EndEffector
    relative: bool
    model_uses_state: bool
    camera_mode: CameraMode
    model_dim: int
    robot_type: str
    checkpoint_dir: pathlib.Path
    dataset_dir: pathlib.Path
    asset_id: str
    head_camera_key: str
    rotation_format: Rotation6DFormat = "columns"

    @property
    def metadata(self) -> dict:
        return {
            "unitree_config": self.name,
            "policy_kind": self.policy_kind,
            "mode": self.mode,
            "action_space": self.action_space,
            "end_effector": self.end_effector,
            "relative": self.relative,
            "model_uses_state": self.model_uses_state,
            "camera_mode": self.camera_mode,
            "model_dim": self.model_dim,
            "robot_type": self.robot_type,
            "rotation_format": self.rotation_format,
            "asset_id": self.asset_id,
        }


@dataclasses.dataclass(frozen=True)
class UnitreeDataConfig(DataConfigFactory):
    spec: UnitreePolicySpec = dataclasses.field(default=None)  # type: ignore[assignment]
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        if self.spec is None:
            raise ValueError("UnitreeDataConfig requires a policy spec")
        image_mapping = {"cam_high": self.spec.head_camera_key}
        if self.spec.camera_mode == "three":
            image_mapping.update(
                {
                    "cam_left_wrist": "observation.images.cam_left_wrist",
                    "cam_right_wrist": "observation.images.cam_right_wrist",
                }
            )
        repack = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": image_mapping,
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
        if self.spec.policy_kind == "human":
            inputs = human_policy.HumanInputs(
                self.spec.model_dim,
                self.spec.camera_mode,
                self.spec.model_uses_state,
            )
            outputs = human_policy.HumanOutputs(self.spec.model_dim)
        else:
            inputs = robot_policy.UnitreeRobotInputs(self.spec.model_dim, self.spec.camera_mode)
            outputs = robot_policy.UnitreeRobotOutputs(self.spec.model_dim)

        data_transforms = _transforms.Group(inputs=[inputs], outputs=[outputs])
        if self.spec.relative:
            if self.spec.action_space == "eef":
                data_transforms = data_transforms.push(
                    inputs=[robot_policy.RelativeEEFActions(self.spec.model_dim, self.spec.rotation_format)],
                    outputs=[robot_policy.AbsoluteEEFActions(self.spec.model_dim, self.spec.rotation_format)],
                )
            else:
                tail_dim = self.spec.model_dim - 14
                mask = _transforms.make_bool_mask(14, -tail_dim) if tail_dim else _transforms.make_bool_mask(14)
                data_transforms = data_transforms.push(
                    inputs=[_transforms.DeltaActions(mask)],
                    outputs=[_transforms.AbsoluteActions(mask)],
                )

        base = self.create_base_config(assets_dirs, model_config)
        return dataclasses.replace(
            base,
            repack_transforms=repack,
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory()(model_config),
            action_sequence_keys=self.action_sequence_keys,
            prompt_from_task=True,
        )


def _mode_layout(policy_kind: PolicyKind, mode: int) -> tuple[ActionSpace, EndEffector, bool, int]:
    if policy_kind == "human":
        layouts = {
            1: ("eef", "dex1", False, 20),
            2: ("eef", "dex1", True, 20),
            3: ("eef", "brainco", False, 30),
            4: ("eef", "brainco", True, 30),
            5: ("eef", "none", False, 18),
            6: ("eef", "none", True, 18),
        }
    else:
        layouts = {
            1: ("eef", "dex1", False, 20),
            2: ("eef", "dex1", True, 20),
            3: ("eef", "brainco", False, 30),
            4: ("eef", "brainco", True, 30),
            5: ("joint", "dex1", False, 16),
            6: ("joint", "dex1", True, 16),
            7: ("joint", "brainco", False, 26),
            8: ("joint", "brainco", True, 26),
        }
    try:
        return layouts[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported {policy_kind} mode: {mode}") from exc


def _default_spec(
    policy_kind: PolicyKind,
    mode: int,
    camera_mode: CameraMode,
    *,
    model_uses_state: bool,
) -> UnitreePolicySpec:
    action_space, end_effector, relative, model_dim = _mode_layout(policy_kind, mode)
    state_suffix = "with_state" if model_uses_state else "no_state"
    name = f"{policy_kind}_mode{mode}_{camera_mode}_{state_suffix}"
    robot_type = "unitree_g1_brainco" if end_effector == "brainco" else "unitree_g1_dex1"
    return UnitreePolicySpec(
        name=name,
        policy_kind=policy_kind,
        mode=mode,
        action_space=action_space,
        end_effector=end_effector,
        relative=relative,
        model_uses_state=model_uses_state,
        camera_mode=camera_mode,
        model_dim=model_dim,
        robot_type=robot_type,
        checkpoint_dir=_CHECKPOINT_ROOT / name,
        dataset_dir=_DATASET_ROOT / name,
        asset_id=name,
        head_camera_key=(
            "observation.images.cam_high" if policy_kind == "human" else "observation.images.cam_left_high"
        ),
    )


# 我们只需要在这里添加即可
def _build_specs() -> dict[str, UnitreePolicySpec]:
    specs = {}
    for mode in range(1, 7):
        for camera_mode in ("single", "three"):
            for model_uses_state in (True, False):
                spec = _default_spec("human", mode, camera_mode, model_uses_state=model_uses_state)
                specs[spec.name] = spec
    for mode in range(1, 9):
        for camera_mode in ("single", "three"):
            spec = _default_spec("robot", mode, camera_mode, model_uses_state=True)
            specs[spec.name] = spec

    # Supplied checkpoints and their corresponding local datasets.
    test_overrides = {
        "human_mode4_single_with_state": (
            _CHECKPOINT_ROOT / "brainco_human_relative_eef_only/25000/25000",
            _DATASET_ROOT / "pick_bottle_put_in_box_have_state",
        ),
        "human_mode4_single_no_state": (
            _CHECKPOINT_ROOT / "brainco_human_relative_eef_no_state_only",
            _DATASET_ROOT / "pick_bottle_put_in_box_no_state",
        ),
        "robot_mode2_three_with_state": (
            _CHECKPOINT_ROOT / "cotrain_eef_relative_ego_no_state_everycase_59999",
            _DATASET_ROOT / "pick_bottle_put_in_box_no_state",
        ),
        "robot_mode7_three_with_state": (
            _CHECKPOINT_ROOT / "base_model/robot_ABC_braincoboth/29999",
            _DATASET_ROOT / "pick_bottle_put_in_box_no_state",
        ),
        "robot_mode4_three_with_state": (
            _CHECKPOINT_ROOT / "base_model/robot_ABC_eef_relative/12000",
            _DATASET_ROOT / "pick_bottle_put_in_box_no_state",
        ),
    }
    for name, (checkpoint_dir, dataset_dir) in test_overrides.items():
        specs[name] = dataclasses.replace(
            specs[name],
            checkpoint_dir=checkpoint_dir,
            dataset_dir=dataset_dir,
            asset_id="WAIC2026_fold_clothes_ABC_robot",
            rotation_format="columns",
        )

    # Temporary compatibility config for the existing mode-4 checkpoint whose
    # 6D rotations are the first two matrix rows. Everything else stays identical.
    rows_name = "robot_mode4_three_with_state_rows"
    specs[rows_name] = dataclasses.replace(
        specs["robot_mode4_three_with_state"],
        name=rows_name,
        rotation_format="rows",
    )
    return specs


UNITREE_SPECS = _build_specs()


def _make_train_config(spec: UnitreePolicySpec) -> TrainConfig:
    return TrainConfig(
        name=spec.name,
        model=pi0_config.Pi0RTCConfig(pi05=True, discrete_state_input=spec.model_uses_state),
        data=UnitreeDataConfig(
            repo_id=str(spec.dataset_dir),
            assets=AssetsConfig(asset_id=spec.asset_id),
            base_config=DataConfig(prompt_from_task=True),
            spec=spec,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(_BASE_CHECKPOINT),
        policy_metadata=spec.metadata,
        num_train_steps=30_000,
        batch_size=32,
        num_workers=8,
    )


UNITREE_TRAIN_CONFIGS = {name: _make_train_config(spec) for name, spec in UNITREE_SPECS.items()}


def get_spec(name: str) -> UnitreePolicySpec:
    try:
        return UNITREE_SPECS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown Unitree config {name!r}. Use --list to show valid configs.") from exc


def get_train_config(name: str) -> TrainConfig:
    get_spec(name)
    return UNITREE_TRAIN_CONFIGS[name]


def checkpoint_norm_stats_path(spec: UnitreePolicySpec, checkpoint_dir: pathlib.Path | None = None) -> pathlib.Path:
    checkpoint = checkpoint_dir or spec.checkpoint_dir
    return checkpoint / "assets" / spec.asset_id / "norm_stats.json"
