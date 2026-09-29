"""Training-only configuration for XRPipe Mode1 fingertip demonstrations."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import hashlib
import json
import pathlib

from typing_extensions import override

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.policies.xrpipe_policy as xrpipe_policy
from openpi.training.config import AssetsConfig
from openpi.training.config import DataConfig
from openpi.training.config import DataConfigFactory
from openpi.training.config import ModelTransformFactory
from openpi.training.config import TrainConfig
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

PRESET = "xrpipe_mode1_rel_shared"
TRAIN_CONFIG_NAME = "xrpipe_mode1_train"
DEFAULT_BASE_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_base/params"
EXPECTED_ROBOT_TYPE = "pico4ultra_right_piper"
EXPECTED_COORDINATE_SYSTEM = "openxr_rh_x_right_y_up_z_back"
EXPECTED_SOURCE_COORDINATE_SYSTEM = "unity_lh_x_right_y_up_z_forward"
EXPECTED_CONTROL_POINT = "right_thumb_index_fingertip_midpoint"
EXPECTED_REFERENCE_FRAME = "pico_world_openxr"
EXPECTED_IMAGE_KEY = "observation.images.camera0"
EXPECTED_FPS = 30
EXPECTED_TIMESTAMP_SEMANTICS = "uniform_frame_index_over_fps"
EXPECTED_STATE_TRANSFORM = (
    "per relation pose T_midpoint_object: S @ T @ S, S=diag(1,1,-1,1); current gripper is unchanged"
)
EXPECTED_ACTION_SEMANTICS = {
    "schema_version": "xrpipe_action_v1",
    "mode": "xrpipe_mode1",
    "stored_action": "absolute_next_target",
    "reference_field": "observation.action_reference_tcp",
    "reference_frame": EXPECTED_REFERENCE_FRAME,
    "coordinate_system": EXPECTED_COORDINATE_SYSTEM,
    "source_coordinate_system": EXPECTED_SOURCE_COORDINATE_SYSTEM,
    "control_point": EXPECTED_CONTROL_POINT,
    "pose_encoding": "xyz_rot6d_columns_grouped",
    "rotation_6d": "first two columns of a 3x3 rotation matrix",
    "action_layout": {"dimension": 10, "pose_slice": [0, 9], "gripper_index": 9},
    "reference_layout": {"dimension": 9, "pose_slice": [0, 9]},
    "relative_formula": "inv(reference[t]) @ action[t+k]",
    "gripper_transform": "none",
}


@dataclasses.dataclass(frozen=True)
class EpisodeSplit:
    train: tuple[int, ...]
    validation: tuple[int, ...]
    digest: str


@dataclasses.dataclass(frozen=True)
class XRPipeDatasetContract:
    dataset: pathlib.Path
    state_dim: int
    reference_dim: int
    action_dim: int
    object_order: tuple[str, ...]
    object_categories: tuple[str, ...]
    semantics: dict
    digest: str

    @property
    def object_count(self) -> int:
        return len(self.object_order)

    def as_dict(self) -> dict:
        return {
            "schema_version": self.semantics["schema_version"],
            "digest": self.digest,
            "state_dim": self.state_dim,
            "reference_dim": self.reference_dim,
            "action_dim": self.action_dim,
            "object_order": list(self.object_order),
            "object_categories": list(self.object_categories),
            "reference_frame": self.semantics["reference_frame"],
            "coordinate_system": self.semantics["coordinate_system"],
            "source_coordinate_system": self.semantics["source_coordinate_system"],
            "control_point": self.semantics["control_point"],
            "stored_action": self.semantics["stored_action"],
            "relative_formula": self.semantics["relative_formula"],
            "training_state_transform": EXPECTED_STATE_TRANSFORM,
        }


def resolve_dataset(path: pathlib.Path | str) -> pathlib.Path:
    root = pathlib.Path(path).expanduser().resolve()
    if (root / "meta/info.json").is_file():
        return root
    if not root.is_dir():
        raise FileNotFoundError(f"XRPipe LeRobot dataset not found: {root}")
    candidates = sorted(child for child in root.iterdir() if (child / "meta/info.json").is_file())
    if len(candidates) != 1:
        raise ValueError(f"Expected one XRPipe dataset below {root}, found {len(candidates)}")
    return candidates[0]


def _feature_dim(info: dict, key: str) -> int:
    feature = info.get("features", {}).get(key)
    if not isinstance(feature, dict):
        raise ValueError(f"XRPipe dataset is missing feature {key!r}")
    shape = feature.get("shape")
    if not isinstance(shape, list) or len(shape) != 1 or not isinstance(shape[0], int):
        raise ValueError(f"XRPipe feature {key!r} must have a one-dimensional shape, got {shape!r}")
    return int(shape[0])


def load_dataset_contract(path: pathlib.Path | str, *, model_action_dim: int = 32) -> XRPipeDatasetContract:
    dataset = resolve_dataset(path)
    info_path = dataset / "meta/info.json"
    semantics_path = dataset / "meta/action_semantics.json"
    extraction_path = dataset / "extraction_meta.json"
    for required in (info_path, semantics_path, extraction_path):
        if not required.is_file():
            raise FileNotFoundError(f"XRPipe dataset contract file is missing: {required}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    semantics = json.loads(semantics_path.read_text(encoding="utf-8"))
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    if semantics != EXPECTED_ACTION_SEMANTICS:
        mismatches = {
            key: (semantics.get(key), expected)
            for key, expected in EXPECTED_ACTION_SEMANTICS.items()
            if semantics.get(key) != expected
        }
        raise ValueError(
            f"Dataset does not declare the XRPipe Mode1 fingertip contract in {semantics_path}: {mismatches}"
        )
    if extraction.get("schema_version") != "xrpipe_v4":
        raise ValueError(
            f"XRPipe Mode1 requires extraction_meta schema xrpipe_v4, got "
            f"{extraction.get('schema_version')!r}; rerun pipeline Step4 only"
        )
    if extraction.get("timestamp_semantics") != EXPECTED_TIMESTAMP_SEMANTICS:
        raise ValueError(
            "XRPipe dataset has legacy/jittered timestamps; rerun Pipeline Step4 with "
            f"timestamp_semantics={EXPECTED_TIMESTAMP_SEMANTICS!r}"
        )
    if info.get("robot_type") != EXPECTED_ROBOT_TYPE:
        raise ValueError(f"Expected robot_type={EXPECTED_ROBOT_TYPE!r}, got {info.get('robot_type')!r}")
    if info.get("fps") != EXPECTED_FPS:
        raise ValueError(f"XRPipe Mode1 requires fps={EXPECTED_FPS}, got {info.get('fps')!r}")
    if EXPECTED_IMAGE_KEY not in info.get("features", {}):
        raise ValueError(f"XRPipe Mode1 requires the single camera feature {EXPECTED_IMAGE_KEY!r}")

    state_dim = _feature_dim(info, "observation.state")
    reference_dim = _feature_dim(info, "observation.action_reference_tcp")
    action_dim = _feature_dim(info, "action")
    if state_dim < 10 or (state_dim - 1) % xrpipe_policy.STATE_BLOCK_DIM:
        raise ValueError(f"XRPipe state must be 9N+1 with N>=1, got {state_dim}D")
    if state_dim > model_action_dim:
        raise ValueError(f"XRPipe state {state_dim}D cannot fit the model's {model_action_dim}D state slot")
    if reference_dim != xrpipe_policy.REFERENCE_DIM or action_dim != xrpipe_policy.ACTION_DIM:
        raise ValueError(f"XRPipe Mode1 requires reference/action dimensions 9/10, got {reference_dim}/{action_dim}")
    if action_dim in (18, 20, 30):
        raise ValueError("Unitree dual-arm EEF layouts are not accepted by XRPipe Mode1")

    relation = info.get("ego_relation", {})
    object_order = tuple(str(v) for v in relation.get("object_order", ()))
    object_categories = tuple(str(v) for v in relation.get("object_categories", ()))
    object_count = (state_dim - 1) // xrpipe_policy.STATE_BLOCK_DIM
    if len(object_order) != object_count or len(object_categories) != object_count:
        raise ValueError(
            f"State encodes {object_count} objects but metadata declares "
            f"object_order={object_order}, object_categories={object_categories}"
        )
    relation_expected = {
        "action_storage": "absolute",
        "action_reference_field": "observation.action_reference_tcp",
        "action_reference_frame": EXPECTED_REFERENCE_FRAME,
        "action_coordinate_system": EXPECTED_COORDINATE_SYSTEM,
        "action_source_coordinate_system": EXPECTED_SOURCE_COORDINATE_SYSTEM,
        "training_state_transform": EXPECTED_STATE_TRANSFORM,
    }
    relation_mismatches = {
        key: (relation.get(key), expected)
        for key, expected in relation_expected.items()
        if relation.get(key) != expected
    }
    if relation_mismatches:
        raise ValueError(f"info.json ego_relation is incompatible with XRPipe Mode1: {relation_mismatches}")

    digest_payload = {
        "semantics": semantics,
        "state_dim": state_dim,
        "reference_dim": reference_dim,
        "action_dim": action_dim,
        "object_order": object_order,
        "object_categories": object_categories,
    }
    digest = hashlib.sha256(json.dumps(digest_payload, sort_keys=True).encode()).hexdigest()[:12]
    return XRPipeDatasetContract(
        dataset=dataset,
        state_dim=state_dim,
        reference_dim=reference_dim,
        action_dim=action_dim,
        object_order=object_order,
        object_categories=object_categories,
        semantics=semantics,
        digest=digest,
    )


def make_episode_split(dataset: pathlib.Path, validation_count: int, seed: int) -> EpisodeSplit:
    rows = [
        json.loads(line)
        for line in (dataset / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    episodes = [int(row["episode_index"]) for row in rows]
    if not episodes:
        raise ValueError(f"XRPipe dataset has no episodes: {dataset}")
    if validation_count < 0:
        raise ValueError("validation_count must be non-negative")
    if validation_count and len(episodes) < 2:
        raise ValueError("Validation requires at least two XRPipe episodes")
    validation_count = min(validation_count, max(0, len(episodes) - 1))
    ranked = sorted(episodes, key=lambda episode: hashlib.sha256(f"{seed}:{episode}".encode()).digest())
    validation = tuple(sorted(ranked[:validation_count]))
    validation_set = set(validation)
    train = tuple(episode for episode in episodes if episode not in validation_set)
    digest = hashlib.sha256(
        json.dumps({"train": train, "validation": validation}, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    return EpisodeSplit(train=train, validation=validation, digest=digest)


def asset_id(contract: XRPipeDatasetContract, split: EpisodeSplit) -> str:
    return f"xrpipe_mode1_fingertip10_rel_shared_contract_{contract.digest}_split_{split.digest}"


def _validate_norm_manifest(
    directory: pathlib.Path,
    *,
    expected_asset_id: str,
    contract: XRPipeDatasetContract,
    split: EpisodeSplit,
    action_horizon: int,
) -> None:
    stats_path = directory / "norm_stats.json"
    if not stats_path.is_file():
        return
    manifest_path = directory / "norm_stats_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Refusing normalization stats without {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "preset": PRESET,
        "asset_id": expected_asset_id,
        "dataset_root": str(contract.dataset),
        "contract_digest": contract.digest,
        "split_digest": split.digest,
        "action_horizon": action_horizon,
        "norm_mode": "shared",
        "complete": True,
        "state_dim": contract.state_dim,
        "action_dim": contract.action_dim,
    }
    mismatches = {key: (manifest.get(key), value) for key, value in expected.items() if manifest.get(key) != value}
    if mismatches:
        raise ValueError(f"Refusing incompatible XRPipe normalization stats in {directory}: {mismatches}")


@dataclasses.dataclass(frozen=True)
class XRPipeModeSpec:
    name: str = "xrpipe_mode1"
    camera_mode: str = "single"
    model_uses_state: bool = True
    action_representation: str = "relative"
    relative_norm: str = "shared"
    control_point: str = EXPECTED_CONTROL_POINT


@dataclasses.dataclass(frozen=True)
class XRPipeMode1DataConfig(DataConfigFactory):
    repo_id: str = "xrpipe_mode1"
    dataset: pathlib.Path = pathlib.Path(".")
    preset: str = PRESET
    validation_episodes: int = 0
    split_seed: int = 2026

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        del assets_dirs
        if self.preset != PRESET:
            raise ValueError(f"Unsupported XRPipe preset {self.preset!r}; expected {PRESET!r}")
        if model_config.action_dim != 32 or model_config.action_horizon != 50:
            raise ValueError(
                f"{PRESET} requires model action_dim/action_horizon 32/50, got "
                f"{model_config.action_dim}/{model_config.action_horizon}"
            )
        contract = load_dataset_contract(self.dataset, model_action_dim=model_config.action_dim)
        split = make_episode_split(contract.dataset, self.validation_episodes, self.split_seed)
        resolved_asset_id = asset_id(contract, split)
        assets_dir = contract.dataset / "meta" / "openpi_assets"
        asset_dir = assets_dir / resolved_asset_id
        _validate_norm_manifest(
            asset_dir,
            expected_asset_id=resolved_asset_id,
            contract=contract,
            split=split,
            action_horizon=model_config.action_horizon,
        )

        repack = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"camera0": EXPECTED_IMAGE_KEY},
                        "state": "observation.state",
                        "action_reference": "observation.action_reference_tcp",
                        "actions": "action",
                        "action_is_pad": "action_is_pad",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[
                xrpipe_policy.CanonicalizeRelationState(contract.state_dim),
                xrpipe_policy.RelativeFingertipActions(contract.action_dim, contract.reference_dim),
                xrpipe_policy.XRPipeInputs(contract.state_dim, contract.action_dim),
            ],
            outputs=[xrpipe_policy.XRPipeOutputs(contract.action_dim)],
        )
        manifest = {
            "schema_version": 1,
            "training_config": TRAIN_CONFIG_NAME,
            "preset": PRESET,
            "mode": dataclasses.asdict(XRPipeModeSpec()),
            "dataset": str(contract.dataset),
            "dataset_fps": EXPECTED_FPS,
            "action_horizon": model_config.action_horizon,
            "model_action_dim": model_config.action_dim,
            "state_dim": contract.state_dim,
            "reference_dim": contract.reference_dim,
            "action_dim": contract.action_dim,
            "normalization": "shared_quantile",
            "normalization_clip": 5.0,
            "mask_padded_action_dims": True,
            "mask_padded_timesteps": True,
            "asset_id": resolved_asset_id,
            "split_digest": split.digest,
            "dataset_contract": contract.as_dict(),
            "output_semantics": (
                "relative SE(3) in the current right thumb/index fingertip-midpoint frame "
                "+ absolute target binary gripper"
            ),
            "deployment_supported": False,
        }
        return DataConfig(
            repo_id=str(contract.dataset),
            asset_id=resolved_asset_id,
            norm_stats=self._load_norm_stats(assets_dir, resolved_asset_id),
            repack_transforms=repack,
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory(track_modalities=True, mask_padded_action_dims=True)(model_config),
            use_quantile_norm=True,
            normalization_clip=5.0,
            action_sequence_keys=("action",),
            prompt_from_task=True,
            train_episodes=split.train,
            validation_episodes=split.validation,
            runtime_manifest=manifest,
        )


def build_train_config(
    *,
    exp_name: str,
    dataset: pathlib.Path,
    preset: str = PRESET,
    validation_episodes: int = 0,
    split_seed: int = 2026,
    batch_size: int = 32,
    fsdp_devices: int = 1,
    num_train_steps: int = 30_000,
    num_workers: int = 8,
    validation_interval: int = 0,
    validation_batches: int = 0,
    modality_diagnostics_interval: int = 0,
    save_interval: int = 1_000,
    keep_period: int | None = 5_000,
    seed: int = 42,
    overwrite: bool = False,
    resume: bool = False,
    wandb_enabled: bool = True,
    base_checkpoint: str = DEFAULT_BASE_CHECKPOINT,
) -> TrainConfig:
    data = XRPipeMode1DataConfig(
        assets=AssetsConfig(),
        dataset=dataset,
        preset=preset,
        validation_episodes=validation_episodes,
        split_seed=split_seed,
    )
    config = TrainConfig(
        name=TRAIN_CONFIG_NAME,
        project_name="openpi-xrpipe",
        exp_name=exp_name,
        model=pi0_config.Pi0RTCConfig(pi05=True, discrete_state_input=True, action_dim=32, action_horizon=50),
        data=data,
        weight_loader=weight_loaders.CheckpointWeightLoader(base_checkpoint),
        batch_size=batch_size,
        fsdp_devices=fsdp_devices,
        num_train_steps=num_train_steps,
        num_workers=num_workers,
        validation_interval=validation_interval if validation_episodes else 0,
        validation_batches=validation_batches if validation_episodes else 0,
        modality_diagnostics_interval=modality_diagnostics_interval if validation_episodes else 0,
        save_interval=save_interval,
        keep_period=keep_period,
        seed=seed,
        overwrite=overwrite,
        resume=resume,
        wandb_enabled=wandb_enabled,
    )
    resolved = data.create(config.assets_dirs, config.model)
    return dataclasses.replace(config, policy_metadata=resolved.runtime_manifest)


def validate_resume_manifest(config: TrainConfig) -> None:
    if not config.resume or not config.checkpoint_dir.is_dir():
        return
    manifests = list(config.checkpoint_dir.glob("*/assets/runtime_manifest.json"))
    if not manifests:
        return
    latest = max(manifests, key=lambda path: int(path.parent.parent.name) if path.parent.parent.name.isdigit() else -1)
    actual = json.loads(latest.read_text(encoding="utf-8"))
    expected = config.policy_metadata or {}
    keys: Sequence[str] = (
        "preset",
        "dataset",
        "action_horizon",
        "model_action_dim",
        "state_dim",
        "reference_dim",
        "action_dim",
        "asset_id",
        "split_digest",
        "dataset_contract",
    )
    mismatches = {key: (actual.get(key), expected.get(key)) for key in keys if actual.get(key) != expected.get(key)}
    if mismatches:
        raise ValueError(f"Cannot resume XRPipe checkpoint with a different numeric/data contract: {mismatches}")
