"""Training-only configuration for G1-D BrainCo/gripper and Human EEF-only experiments.

This module deliberately does not reuse ``unitree_config.py``: that file also
describes legacy evaluation checkpoints whose mode numbers and Rot6D layouts
have different meanings.  The policy/model/training implementation is shared;
only the experiment data graph is isolated here.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import hashlib
import json
import pathlib
import re
from typing import Literal, cast

import numpy as np
from typing_extensions import override

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.policies.human_policy as human_policy
import openpi.policies.unitree_robot_policy as robot_policy
from openpi.training.action_chunk_resampling import required_source_horizon
from openpi.training.action_chunk_resampling import resample_absolute_eef_actions
from openpi.training.config import AssetsConfig
from openpi.training.config import DataConfig
from openpi.training.config import DataConfigFactory
from openpi.training.config import ModelTransformFactory
from openpi.training.config import TrainConfig
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ActionRepresentation = Literal["absolute", "relative"]
RelativeNorm = Literal["shared", "per_step", "hybrid"]
ActionDomain = Literal["brainco", "gripper"]
ComponentActionDomain = Literal["brainco", "gripper", "eef_only"]
RobotCameraMode = Literal["single", "three"]
HumanInputFrame = Literal["native", "torso_palm", "g1_base_tcp", "recording_tcp", "pelvis_wrist"]
HumanDatasetMode = Literal["mode1", "mode2"]

_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_ROBOT_JOINT_DATASET = _ROOT / "dataset/G1_Brainco_Abc_fold_clothes_7_7/G1_Brainco_Abc_fold_clothes_7_7"
DEFAULT_ROBOT_GRIPPER_JOINT_DATASET = _ROOT / "dataset/clothers_3eps"
DEFAULT_ROBOT_EEF_DATASET = _ROOT / "dataset/G1_Brainco_Abc_fold_clothes_7_7_eef30_columns_grouped"
DEFAULT_HUMAN_DATASET = _ROOT / "dataset/pick_bottle_put_in_box_have_state"
DEFAULT_ROBOT_GRIPPER_EEF_DATASET = _ROOT / "dataset/clothers_3eps_eef20_columns_grouped"
DEFAULT_HUMAN_GRIPPER_DATASET = _ROOT / "dataset/pick_bottle_put_in_box_have_state_gripper_eef20"
# EEF-only names the supervised domain, not necessarily the raw storage
# layout. By default it reads the existing EEF20 Human data and drops its
# two recorded gripper values before normalization/training.
DEFAULT_HUMAN_EEF_ONLY_DATASET = DEFAULT_HUMAN_GRIPPER_DATASET
DEFAULT_BASE_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_base/params"
TRAIN_CONFIG_NAME = "unitree_g1d_brainco_train"
G1D_REFERENCE_FRAME = "torso_link"
G1D_EEF_LINKS = ("left_hand_palm_link", "right_hand_palm_link")
G1D_ROTATION_6D_LAYOUT = "columns_grouped:[r00,r10,r20,r01,r11,r21]"
G1D_ARM_JOINT_NAMES = (
    "kLeftShoulderPitch",
    "kLeftShoulderRoll",
    "kLeftShoulderYaw",
    "kLeftElbow",
    "kLeftWristRoll",
    "kLeftWristPitch",
    "kLeftWristYaw",
    "kRightShoulderPitch",
    "kRightShoulderRoll",
    "kRightShoulderYaw",
    "kRightElbow",
    "kRightWristRoll",
    "kRightWristPitch",
    "kRightWristYaw",
)
G1D_BRAINCO_NAMES = (
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
G1D_GRIPPER_NAMES = ("kLeftGripper", "kRightGripper")


@dataclasses.dataclass(frozen=True)
class UnitreeExperimentPreset:
    robot_mode: int
    human_mode: int
    action_representation: ActionRepresentation
    relative_norm: RelativeNorm
    # Kept last with a default so all existing four-argument construction and
    # every legacy preset continue to mean the original BrainCo action domain.
    action_domain: ActionDomain = "brainco"
    # Progress resampling is opt-in and is never enabled by an existing preset.
    task_progress_alignment: bool = False
    # Human EEF-only is opt-in; Robot still uses action_domain for deployment.
    human_eef_only: bool = False
    # Existing preset names remain three-camera. Appending "_single" selects
    # head-only Robot input without changing the numeric action/state domain.
    robot_camera_mode: RobotCameraMode = "three"


def _parse_unitree_preset_base(name: str) -> UnitreeExperimentPreset:
    """Resolve a preset name without the optional Robot camera suffix."""
    if name == "robot1_abs":
        return UnitreeExperimentPreset(1, 0, "absolute", "shared")
    if name == "robot1_gripper_abs":
        return UnitreeExperimentPreset(1, 0, "absolute", "shared", "gripper")
    if name == "robot2_abs":
        return UnitreeExperimentPreset(2, 0, "absolute", "shared")
    if match := re.fullmatch(r"robot2_rel_(shared|per_step|hybrid)", name):
        return UnitreeExperimentPreset(2, 0, "relative", cast(RelativeNorm, match.group(1)))
    if match := re.fullmatch(r"human([3-6])_abs", name):
        return UnitreeExperimentPreset(0, int(match.group(1)), "absolute", "shared")
    if match := re.fullmatch(r"human([3-6])_rel_(shared|per_step|hybrid)", name):
        return UnitreeExperimentPreset(0, int(match.group(1)), "relative", cast(RelativeNorm, match.group(2)))
    if match := re.fullmatch(r"mix_robot2_human([3-6])_abs", name):
        return UnitreeExperimentPreset(2, int(match.group(1)), "absolute", "shared")
    if match := re.fullmatch(r"mix_robot2_human([3-6])_abs_progress", name):
        return UnitreeExperimentPreset(
            2,
            int(match.group(1)),
            "absolute",
            "shared",
            "brainco",
            task_progress_alignment=True,
        )
    if match := re.fullmatch(r"mix_robot2_human([3-6])_rel_(shared|per_step|hybrid)_progress", name):
        return UnitreeExperimentPreset(
            2,
            int(match.group(1)),
            "relative",
            cast(RelativeNorm, match.group(2)),
            "brainco",
            task_progress_alignment=True,
        )
    if match := re.fullmatch(r"mix_robot2_human([3-6])_rel_(shared|per_step|hybrid)", name):
        return UnitreeExperimentPreset(2, int(match.group(1)), "relative", cast(RelativeNorm, match.group(2)))
    if match := re.fullmatch(r"mix_robot2_human([3-6])_eef_only_abs(_progress)?", name):
        return UnitreeExperimentPreset(
            2,
            int(match.group(1)),
            "absolute",
            "shared",
            "brainco",
            task_progress_alignment=match.group(2) is not None,
            human_eef_only=True,
        )
    if match := re.fullmatch(r"mix_robot2_human([3-6])_eef_only_rel_(shared|per_step|hybrid)(_progress)?", name):
        return UnitreeExperimentPreset(
            2,
            int(match.group(1)),
            "relative",
            cast(RelativeNorm, match.group(2)),
            "brainco",
            task_progress_alignment=match.group(3) is not None,
            human_eef_only=True,
        )
    if name == "robot2_gripper_abs":
        return UnitreeExperimentPreset(2, 0, "absolute", "shared", "gripper")
    if match := re.fullmatch(r"robot2_gripper_rel_(shared|per_step|hybrid)", name):
        return UnitreeExperimentPreset(2, 0, "relative", cast(RelativeNorm, match.group(1)), "gripper")
    if match := re.fullmatch(r"human([3-6])_gripper_abs", name):
        return UnitreeExperimentPreset(0, int(match.group(1)), "absolute", "shared", "gripper")
    if match := re.fullmatch(r"human([3-6])_gripper_rel_(shared|per_step|hybrid)", name):
        return UnitreeExperimentPreset(
            0, int(match.group(1)), "relative", cast(RelativeNorm, match.group(2)), "gripper"
        )
    if match := re.fullmatch(r"mix_robot2_human([3-6])_gripper_abs", name):
        return UnitreeExperimentPreset(2, int(match.group(1)), "absolute", "shared", "gripper")
    if match := re.fullmatch(r"mix_robot2_human([3-6])_gripper_abs_progress", name):
        return UnitreeExperimentPreset(
            2,
            int(match.group(1)),
            "absolute",
            "shared",
            "gripper",
            task_progress_alignment=True,
        )
    if match := re.fullmatch(r"mix_robot2_human([3-6])_gripper_rel_(shared|per_step|hybrid)", name):
        return UnitreeExperimentPreset(
            2, int(match.group(1)), "relative", cast(RelativeNorm, match.group(2)), "gripper"
        )
    if match := re.fullmatch(r"mix_robot2_human([3-6])_gripper_rel_(shared|per_step|hybrid)_progress", name):
        return UnitreeExperimentPreset(
            2,
            int(match.group(1)),
            "relative",
            cast(RelativeNorm, match.group(2)),
            "gripper",
            task_progress_alignment=True,
        )
    if match := re.fullmatch(r"mix_robot2_human([3-6])_gripper_eef_only_abs(_progress)?", name):
        return UnitreeExperimentPreset(
            2,
            int(match.group(1)),
            "absolute",
            "shared",
            "gripper",
            task_progress_alignment=match.group(2) is not None,
            human_eef_only=True,
        )
    if match := re.fullmatch(
        r"mix_robot2_human([3-6])_gripper_eef_only_rel_(shared|per_step|hybrid)(_progress)?", name
    ):
        return UnitreeExperimentPreset(
            2,
            int(match.group(1)),
            "relative",
            cast(RelativeNorm, match.group(2)),
            "gripper",
            task_progress_alignment=match.group(3) is not None,
            human_eef_only=True,
        )
    raise ValueError(f"Unknown Unitree preset: {name}")


def parse_unitree_preset(name: str) -> UnitreeExperimentPreset:
    """Resolve a user-facing preset, including the optional ``_single`` suffix.

    Existing unsuffixed names intentionally retain their original three-camera
    behavior. The suffix is valid for every Robot-only or Robot+Human preset,
    so joint, EEF, BrainCo, gripper, absolute, relative, and mixed training all
    share one camera-selection rule.
    """
    single_camera = name.endswith("_single")
    base_name = name.removesuffix("_single") if single_camera else name
    preset = _parse_unitree_preset_base(base_name)
    if single_camera:
        if preset.robot_mode == 0:
            raise ValueError(f"The _single suffix requires a Robot component: {name}")
        preset = dataclasses.replace(preset, robot_camera_mode="single")
    return preset


def parse_unitree_norm_preset(name: str) -> UnitreeExperimentPreset:
    """Resolve a single-domain normalization preset.

    Human modes 3/4/5/6 share the same numeric state/action domain, split, and
    asset id. Mode3 is used only as the representative ComponentSpec here.
    """
    if name == "human_abs":
        return UnitreeExperimentPreset(0, 3, "absolute", "shared")
    if name == "human_abs_progress":
        # This single Human component uses the Robot EEF dataset only as the
        # duration reference needed to reproduce mixed-training action stats.
        return UnitreeExperimentPreset(
            0,
            3,
            "absolute",
            "shared",
            "brainco",
            task_progress_alignment=True,
        )
    if match := re.fullmatch(r"human_rel_(shared|per_step|hybrid)", name):
        return UnitreeExperimentPreset(0, 3, "relative", cast(RelativeNorm, match.group(1)))
    if match := re.fullmatch(r"human_rel_(shared|per_step|hybrid)_progress", name):
        return UnitreeExperimentPreset(
            0,
            3,
            "relative",
            cast(RelativeNorm, match.group(1)),
            "brainco",
            task_progress_alignment=True,
        )
    if name == "human_gripper_abs":
        return UnitreeExperimentPreset(0, 3, "absolute", "shared", "gripper")
    if name == "human_gripper_abs_progress":
        return UnitreeExperimentPreset(
            0,
            3,
            "absolute",
            "shared",
            "gripper",
            task_progress_alignment=True,
        )
    if match := re.fullmatch(r"human_gripper_rel_(shared|per_step|hybrid)", name):
        return UnitreeExperimentPreset(0, 3, "relative", cast(RelativeNorm, match.group(1)), "gripper")
    if match := re.fullmatch(r"human_gripper_rel_(shared|per_step|hybrid)_progress", name):
        return UnitreeExperimentPreset(
            0,
            3,
            "relative",
            cast(RelativeNorm, match.group(1)),
            "gripper",
            task_progress_alignment=True,
        )
    if name == "human_eef_only_abs":
        return UnitreeExperimentPreset(0, 3, "absolute", "shared", human_eef_only=True)
    if match := re.fullmatch(r"human_eef_only_rel_(shared|per_step|hybrid)", name):
        return UnitreeExperimentPreset(
            0,
            3,
            "relative",
            cast(RelativeNorm, match.group(1)),
            human_eef_only=True,
        )
    if name == "human_eef_only_abs_progress":
        return UnitreeExperimentPreset(
            0,
            3,
            "absolute",
            "shared",
            task_progress_alignment=True,
            human_eef_only=True,
        )
    if match := re.fullmatch(r"human_eef_only_rel_(shared|per_step|hybrid)_progress", name):
        return UnitreeExperimentPreset(
            0,
            3,
            "relative",
            cast(RelativeNorm, match.group(1)),
            task_progress_alignment=True,
            human_eef_only=True,
        )
    if match := re.fullmatch(
        r"human_eef_only_robot_(brainco|gripper)_(abs|rel_(?:shared|per_step|hybrid))_progress", name
    ):
        representation = "absolute" if match.group(2) == "abs" else "relative"
        relative_norm = "shared" if representation == "absolute" else match.group(2).removeprefix("rel_")
        return UnitreeExperimentPreset(
            0,
            3,
            cast(ActionRepresentation, representation),
            cast(RelativeNorm, relative_norm),
            cast(ActionDomain, match.group(1)),
            task_progress_alignment=True,
            human_eef_only=True,
        )
    preset = parse_unitree_preset(name)
    if preset.robot_mode == 0 or preset.human_mode != 0:
        raise ValueError(f"Normalization preset must select exactly one Robot or Human domain: {name}")
    return preset


@dataclasses.dataclass(frozen=True)
class EpisodeSplit:
    train: tuple[int, ...]
    validation: tuple[int, ...]
    digest: str


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _resolve_dataset(path: pathlib.Path | str) -> pathlib.Path:
    root = pathlib.Path(path).expanduser().resolve()
    if (root / "meta/info.json").is_file():
        return root
    if not root.is_dir():
        raise FileNotFoundError(f"LeRobot dataset not found: {root}")
    candidates = sorted(child for child in root.iterdir() if (child / "meta/info.json").is_file())
    if len(candidates) != 1:
        raise ValueError(f"Expected one LeRobot dataset below {root}, found {len(candidates)}")
    return candidates[0]


def _read_optional_json(path: pathlib.Path) -> dict | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _resolve_human_dataset_contract(dataset: pathlib.Path, requested_mode: HumanDatasetMode) -> dict:
    """Resolve and validate the stored Ego LeRobot action semantics.

    The visibility modes used by OpenPI (Human 3/4/5/6) do not describe how
    the Ego converter wrote state/action.  That separate source contract is
    intentionally mandatory in every Human asset/checkpoint namespace.
    """
    semantics_path = dataset / "meta" / "action_semantics.json"
    extraction_path = dataset / "extraction_meta.json"
    semantics = _read_optional_json(semantics_path)
    extraction = _read_optional_json(extraction_path)

    detected_mode: HumanDatasetMode | None = None
    stored_action = None
    state_action_alignment = None
    reference_frame = None
    eef_frame = None
    provenance = "explicit_cli"
    if semantics is not None:
        if semantics.get("schema_version") == "xrpipe_action_v1":
            raise ValueError(
                f"XRPipe fingertip dataset {dataset} cannot use the Unitree Human pipeline: "
                "it is single-right-hand 10D data and must be trained with "
                "scripts/train_xrpipe.py --preset xrpipe_mode1_rel_shared."
            )
        provenance = "meta/action_semantics.json"
        semantic_mode = semantics.get("mode")
        tcp_semantics = semantics.get("tcp_semantics")
        rotation = semantics.get("tcp_pose9", {}).get("rotation_6d")
        if rotation != "first two columns of a 3x3 rotation matrix":
            raise ValueError(
                f"Human dataset has unsupported Rot6D semantics in {semantics_path}: {rotation!r}"
            )
        if semantic_mode == "mode1_ideal" and tcp_semantics == "absolute_target_in_g1_base":
            detected_mode = "mode1"
            stored_action = "absolute"
            state_action_alignment = "same_tick_absolute_target"
            reference_frame = "g1_base"
            eef_frame = "wrist_yaw_tcp"
        elif semantic_mode == "mode2_state" and tcp_semantics == "absolute_next_target_in_recording_frame":
            detected_mode = "mode2"
            stored_action = "absolute"
            state_action_alignment = "action[t]=absolute_state[t+1]"
            reference_frame = "recording_frame"
            eef_frame = "wrist_yaw_tcp"
        else:
            raise ValueError(
                f"Unsupported Human Ego action contract in {semantics_path}: "
                f"mode={semantic_mode!r}, tcp_semantics={tcp_semantics!r}"
            )
    elif extraction is not None:
        pipeline_mode = extraction.get("config", {}).get("pipeline_mode")
        cache_version = extraction.get("config", {}).get("cache_version")
        if pipeline_mode == "mode2_state":
            # Old Mode2 products wrote relative TCP deltas into parquet.  Shape
            # and feature names cannot distinguish them from the fixed absolute
            # product, so accepting one without action_semantics is unsafe.
            raise ValueError(
                f"Human Mode2 dataset {dataset} has no {semantics_path.name}; "
                f"extraction cache_version={cache_version!r}. Legacy Mode2 parquet may contain relative actions. "
                "Re-export it with the current Ego converter so meta/action_semantics.json proves "
                "absolute_next_target_in_recording_frame."
            )
        if pipeline_mode == "mode1_ideal":
            detected_mode = "mode1"
            stored_action = "absolute"
            state_action_alignment = "same_tick_absolute_target"
            reference_frame = "g1_base"
            eef_frame = "wrist_yaw_tcp"
            provenance = "legacy extraction_meta.json mode1"

    if detected_mode is not None and detected_mode != requested_mode:
        raise ValueError(
            f"Human dataset mode mismatch for {dataset}: requested {requested_mode}, "
            f"but {provenance} declares {detected_mode}"
        )
    if detected_mode is None:
        raise ValueError(
            f"Cannot prove the Human Ego action contract for {dataset}: neither "
            f"{semantics_path} nor a usable Mode1 {extraction_path} declares how pose actions were stored. "
            "Re-export with the current Ego converter, or preserve its extraction_meta.json. "
            "A CLI mode flag alone is not accepted as provenance."
        )

    digest_payload = {
        "mode": detected_mode,
        "stored_action": stored_action,
        "state_action_alignment": state_action_alignment,
        "reference_frame": reference_frame,
        "eef_frame": eef_frame,
        "rotation_6d": G1D_ROTATION_6D_LAYOUT,
        "provenance": provenance,
    }
    digest = hashlib.sha256(json.dumps(digest_payload, sort_keys=True).encode()).hexdigest()[:12]
    return {**digest_payload, "digest": digest}


def make_episode_split(
    dataset: pathlib.Path | str,
    *,
    validation_count: int,
    seed: int,
    task_contains: str | None = None,
    max_train_episodes: int | None = None,
    minimum_episode_frames: int = 10,
) -> EpisodeSplit:
    """Create a deterministic episode-level split, optionally filtered by task."""
    dataset = _resolve_dataset(dataset)
    episodes = _read_jsonl(dataset / "meta/episodes.jsonl")
    task_rows = _read_jsonl(dataset / "meta/tasks.jsonl")
    task_by_index = {int(row["task_index"]): str(row["task"]) for row in task_rows}

    selected = []
    for row in episodes:
        task_texts = row.get("tasks")
        if task_texts is None:
            task_index = row.get("task_index")
            task_texts = [task_by_index[int(task_index)]] if task_index is not None else []
        task_matches = task_contains is None or any(task_contains.lower() in str(text).lower() for text in task_texts)
        if task_matches and int(row["length"]) >= minimum_episode_frames:
            selected.append(int(row["episode_index"]))
    minimum_selected = 1 if validation_count == 0 else 2
    if len(selected) < minimum_selected:
        raise ValueError(f"Need at least {minimum_selected} matching episode(s) in {dataset}, found {len(selected)}")
    if validation_count < 0:
        raise ValueError("validation_count must be non-negative")
    if minimum_episode_frames <= 0:
        raise ValueError("minimum_episode_frames must be positive")

    # NumPy is intentionally avoided here so config construction is stable
    # across NumPy RNG implementation changes.
    ranked = sorted(
        selected,
        key=lambda episode: hashlib.sha256(f"{seed}:{episode}".encode()).digest(),
    )
    validation_count = min(validation_count, len(ranked) - 1)
    validation = tuple(sorted(ranked[:validation_count]))
    validation_set = set(validation)
    train = [episode for episode in selected if episode not in validation_set]
    if max_train_episodes is not None:
        if max_train_episodes <= 0:
            raise ValueError("max_train_episodes must be positive")
        train = train[:max_train_episodes]
    digest = hashlib.sha256(
        json.dumps({"train": train, "validation": validation}, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    return EpisodeSplit(tuple(train), validation, digest)


@dataclasses.dataclass(frozen=True)
class ComponentSpec:
    name: str
    kind: Literal["robot", "human"]
    mode: int
    dataset: pathlib.Path
    dimension: int
    camera_mode: Literal["single", "three"]
    model_uses_state: bool
    action_space: Literal["joint", "eef"]
    asset_id: str
    split: EpisodeSplit
    action_domain: ComponentActionDomain = "brainco"
    eef_kinematics: dict | None = None
    task_progress_alignment: dict | None = None
    # Kept last so existing positional ComponentSpec construction remains
    # compatible. Set only for a source projected to a smaller training domain.
    source_dimension: int | None = None
    input_frame: HumanInputFrame = "native"
    human_dataset_contract: dict | None = None

    @property
    def metadata(self) -> dict:
        action_tail = {
            "brainco": "brainco12",
            "gripper": "gripper2",
            "eef_only": "none",
        }[self.action_domain]
        metadata = {
            "name": self.name,
            "kind": self.kind,
            "mode": self.mode,
            "dataset": str(self.dataset),
            "dimension": self.dimension,
            "camera_mode": self.camera_mode,
            "model_uses_state": self.model_uses_state,
            "action_space": self.action_space,
            "action_domain": self.action_domain,
            "action_tail": action_tail,
            "input_frame": self.input_frame,
            "canonical_frame": (
                "torso_palm"
                if self.kind == "robot" or self.input_frame in ("torso_palm", "g1_base_tcp", "pelvis_wrist")
                else ("recording_palm" if self.input_frame == "recording_tcp" else "dataset_native")
            ),
            "frame_conversion_applied": self.kind == "human"
            and self.input_frame in ("g1_base_tcp", "recording_tcp", "pelvis_wrist"),
            "asset_id": self.asset_id,
            "train_episodes": len(self.split.train),
            "validation_episodes": len(self.split.validation),
            "split_digest": self.split.digest,
        }
        if self.eef_kinematics is not None:
            metadata["eef_kinematics"] = self.eef_kinematics
        if self.human_dataset_contract is not None:
            metadata["human_dataset_contract"] = self.human_dataset_contract
        if self.action_domain == "eef_only":
            metadata["supervised_action_dimensions"] = self.dimension
        if self.source_dimension is not None:
            metadata["source_projection"] = {
                "input_dimension": self.source_dimension,
                "output_dimension": self.dimension,
                "kept_slice": [0, self.dimension],
                "ignored_slice": [self.dimension, self.source_dimension],
                "ignored_tail": "gripper2",
            }
        if self.task_progress_alignment is not None:
            metadata["task_progress_alignment"] = self.task_progress_alignment
        return metadata


def _task_texts(dataset: pathlib.Path) -> set[str]:
    rows = _read_jsonl(dataset / "meta/tasks.jsonl")
    return {re.sub(r"[^a-z0-9]+", " ", str(row["task"]).lower()).strip() for row in rows}


def _episode_duration_seconds(dataset: pathlib.Path, episodes: Sequence[int]) -> np.ndarray:
    info = json.loads((dataset / "meta/info.json").read_text(encoding="utf-8"))
    fps = float(info["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Dataset FPS must be finite and positive in {dataset / 'meta/info.json'}")
    selected = {int(index) for index in episodes}
    lengths = [
        int(row["length"])
        for row in _read_jsonl(dataset / "meta/episodes.jsonl")
        if int(row["episode_index"]) in selected
    ]
    if len(lengths) != len(selected):
        raise ValueError(f"Could not resolve every selected episode duration in {dataset}")
    durations = (np.asarray(lengths, dtype=np.float64) - 1.0) / fps
    if np.any(durations <= 0):
        raise ValueError(f"Task-progress alignment requires episodes with at least two frames in {dataset}")
    return durations


def estimate_task_progress_alignment(
    robot_dataset: pathlib.Path,
    robot_episodes: Sequence[int],
    human_dataset: pathlib.Path,
    human_episodes: Sequence[int],
) -> dict:
    """Estimate one robust task-level speed ratio from complete episode durations."""
    robot_tasks = _task_texts(robot_dataset)
    human_tasks = _task_texts(human_dataset)
    if robot_tasks != human_tasks:
        raise ValueError(
            "Task-progress alignment requires identical Robot/Human task text; "
            f"robot={sorted(robot_tasks)}, human={sorted(human_tasks)}"
        )
    robot_durations = _episode_duration_seconds(robot_dataset, robot_episodes)
    human_durations = _episode_duration_seconds(human_dataset, human_episodes)
    robot_median = float(np.median(robot_durations))
    human_median = float(np.median(human_durations))
    gamma = robot_median / human_median
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError(f"Invalid task-progress duration ratio: {gamma}")
    payload = {
        "enabled": True,
        "estimator": "ratio_of_median_complete_episode_durations",
        "task_texts": sorted(robot_tasks),
        "robot_episode_count": len(robot_durations),
        "human_episode_count": len(human_durations),
        "robot_duration_seconds_sha256": hashlib.sha256(robot_durations.tobytes()).hexdigest(),
        "human_duration_seconds_sha256": hashlib.sha256(human_durations.tobytes()).hexdigest(),
        "robot_median_duration_seconds": robot_median,
        "human_median_duration_seconds": human_median,
        "robot_duration_q10_q90_seconds": np.quantile(robot_durations, (0.1, 0.9)).tolist(),
        "human_duration_q10_q90_seconds": np.quantile(human_durations, (0.1, 0.9)).tolist(),
        "gamma_robot_over_human": gamma,
        "human_source_step_scale": 1.0 / gamma,
        "position_and_tail_interpolation": "linear",
        "rotation_interpolation": "quaternion_slerp",
        "observation_alignment": "image_and_state_at_action_chunk_start",
    }
    payload["digest"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[
        :12
    ]
    return payload


@dataclasses.dataclass(frozen=True)
class ResampleAbsoluteEEFActions(_transforms.DataTransformFn):
    """Turn a short Human physical-time window into the model's full horizon."""

    output_horizon: int
    source_step_scale: float

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data
        actions = np.asarray(data["actions"])
        source_is_pad = np.asarray(
            data.get("action_is_pad", np.zeros(actions.shape[:-1], dtype=np.bool_)),
            dtype=np.bool_,
        )
        resampled, output_is_pad = resample_absolute_eef_actions(
            actions,
            source_is_pad,
            output_horizon=self.output_horizon,
            source_step_scale=self.source_step_scale,
        )
        return {**data, "actions": resampled, "action_is_pad": output_is_pad}


def _feature_keys(dataset: pathlib.Path) -> set[str]:
    info = json.loads((dataset / "meta/info.json").read_text(encoding="utf-8"))
    return set(info["features"])


def _validate_eef_vector_layout(dataset: pathlib.Path, action_domain: ComponentActionDomain) -> int:
    """Validate the native numeric layout before any training transform runs.

    EEF-only is a supervised 18D domain. Its source may already be EEF18, or it
    may be native EEF20-gripper data whose exact two-value gripper tail is
    discarded before normalization/training. BrainCo values are never reduced
    or reinterpreted as gripper commands.
    """
    info_path = dataset / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    expected_dim = {"gripper": 20, "brainco": 30}.get(action_domain)
    resolved_dim: int | None = None
    for key in ("observation.state", "action"):
        feature = info.get("features", {}).get(key, {})
        shape = feature.get("shape")
        source_dim = shape[0] if isinstance(shape, list) and len(shape) == 1 else None
        names = feature.get("names")
        flat_names = names[0] if isinstance(names, list) and len(names) == 1 and isinstance(names[0], list) else None
        if action_domain == "eef_only":
            expected_tail = [] if source_dim == 18 else list(G1D_GRIPPER_NAMES)
            valid_dim = source_dim in (18, 20)
            layout = "EEF18 or EEF18+[kLeftGripper,kRightGripper]"
        else:
            expected_tail = list(G1D_GRIPPER_NAMES) if action_domain == "gripper" else list(G1D_BRAINCO_NAMES)
            valid_dim = source_dim == expected_dim
            layout = f"EEF18,{','.join(expected_tail)}"
        valid_names = (
            valid_dim and flat_names is not None and len(flat_names) == source_dim and flat_names[18:] == expected_tail
        )
        if not valid_names:
            raise ValueError(
                f"{action_domain} EEF dataset requires {key}=[{layout}] "
                f"in {info_path}; got shape={shape}, names={names}"
            )
        if resolved_dim is not None and source_dim != resolved_dim:
            raise ValueError(
                f"observation.state/action dimensions must match in {info_path}; got {resolved_dim}D and {source_dim}D"
            )
        resolved_dim = source_dim
    assert resolved_dim is not None
    return resolved_dim


def _validate_joint_vector_layout(dataset: pathlib.Path, action_domain: ActionDomain) -> None:
    """Require the exact deploy joint order for direct absolute control."""
    info_path = dataset / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    tail = G1D_GRIPPER_NAMES if action_domain == "gripper" else G1D_BRAINCO_NAMES
    expected_names = [*G1D_ARM_JOINT_NAMES, *tail]
    for key in ("observation.state", "action"):
        feature = info.get("features", {}).get(key, {})
        if feature.get("shape") != [len(expected_names)] or feature.get("names") != [expected_names]:
            raise ValueError(
                f"{action_domain} joint dataset requires exact {key} order {expected_names} in {info_path}; "
                f"got shape={feature.get('shape')}, names={feature.get('names')}"
            )


def _validate_robot_eef_metadata(dataset: pathlib.Path, action_domain: ActionDomain = "brainco") -> dict:
    """Reject EEF datasets whose FK convention differs from real-robot inference."""
    info_path = dataset / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    kinematics = info.get("eef_forward_kinematics")
    if not isinstance(kinematics, dict):
        raise ValueError(
            f"Robot mode2 dataset is missing eef_forward_kinematics in {info_path}. "
            "Create it with tools/convert_g1_joint_to_eef.py; an unlabelled EEF frame is unsafe to train."
        )
    _validate_eef_vector_layout(dataset, action_domain)
    expected = {
        "reference_link": G1D_REFERENCE_FRAME,
        "left_eef_link": G1D_EEF_LINKS[0],
        "right_eef_link": G1D_EEF_LINKS[1],
        "frame_convention": "T_reference_eef",
        "rotation_6d": G1D_ROTATION_6D_LAYOUT,
        action_domain: "copied_without_transformation",
    }
    mismatches = {key: (kinematics.get(key), value) for key, value in expected.items() if kinematics.get(key) != value}
    if mismatches:
        raise ValueError(f"Robot EEF dataset convention is incompatible with G1-D inference: {mismatches}")
    urdf_sha256 = kinematics.get("urdf_sha256")
    if not isinstance(urdf_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", urdf_sha256) is None:
        raise ValueError(f"Invalid or missing G1-D URDF SHA256 in {info_path}: {urdf_sha256!r}")
    return {
        "urdf_sha256": urdf_sha256,
        "reference_frame": G1D_REFERENCE_FRAME,
        "eef_links": list(G1D_EEF_LINKS),
        "frame_convention": "T_reference_eef",
        "rotation_6d": G1D_ROTATION_6D_LAYOUT,
    }


def _asset_id(spec_prefix: str, representation: ActionRepresentation, norm: RelativeNorm, split: EpisodeSplit) -> str:
    suffix = "abs" if representation == "absolute" else f"rel_{norm}"
    return f"{spec_prefix}_{suffix}_split_{split.digest}"


def _validate_human_norm_manifest(
    spec: ComponentSpec,
    *,
    representation: ActionRepresentation,
    relative_norm: RelativeNorm,
    action_horizon: int,
) -> None:
    asset_dir = spec.dataset / "meta" / "openpi_assets" / spec.asset_id
    stats_path = asset_dir / "norm_stats.json"
    if not stats_path.is_file():
        return
    manifest_path = asset_dir / "norm_stats_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(
            f"Refusing unproven Human normalization stats {stats_path}: {manifest_path.name} is missing. "
            "Delete/recompute this asset with compute_unitree_norm_stats.sh."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_norm_mode = "absolute" if representation == "absolute" else relative_norm
    expected = {
        "asset_id": spec.asset_id,
        "dataset_root": str(spec.dataset),
        "norm_mode": expected_norm_mode,
        "action_horizon": action_horizon,
        "input_frame": spec.input_frame,
        "dataset_contract": spec.human_dataset_contract,
    }
    mismatches = {key: (manifest.get(key), value) for key, value in expected.items() if manifest.get(key) != value}
    if mismatches:
        raise ValueError(
            f"Refusing incompatible Human normalization stats {asset_dir}: {mismatches}. "
            "Delete/recompute this asset with the same dataset-mode/frame/training preset."
        )


def _make_component_data_config(
    spec: ComponentSpec,
    *,
    model_config: _model.BaseModelConfig,
    representation: ActionRepresentation,
    load_norm_stats,
    relative_norm: RelativeNorm = "shared",
) -> DataConfig:
    features = _feature_keys(spec.dataset)
    if spec.kind == "robot":
        image_mapping = {"cam_high": "observation.images.cam_left_high"}
        if spec.camera_mode == "three":
            image_mapping.update(
                {
                    "cam_left_wrist": "observation.images.cam_left_wrist",
                    "cam_right_wrist": "observation.images.cam_right_wrist",
                }
            )
    else:
        image_mapping = {"cam_high": "observation.images.cam_high"}
        # The supplied human dataset is genuinely single-camera.  If a later
        # dataset contains real wrist streams, consume them automatically;
        # otherwise HumanInputs emits zero images with mask=false.
        for short_name in ("cam_left_wrist", "cam_right_wrist"):
            full_name = f"observation.images.{short_name}"
            if full_name in features:
                image_mapping[short_name] = full_name

    repack = _transforms.Group(
        inputs=[
            _transforms.RepackTransform(
                {
                    "images": image_mapping,
                    "state": "observation.state",
                    "actions": "action",
                    "action_is_pad": "action_is_pad",
                    "prompt": "prompt",
                }
            )
        ]
    )

    inputs: list[_transforms.DataTransformFn] = []
    outputs: list[_transforms.DataTransformFn] = []
    if spec.kind == "human":
        if spec.action_domain == "eef_only":
            inputs.append(human_policy.SelectHumanEEFOnly(spec.source_dimension or spec.dimension))
        # Every supported Ego dataset mode uses one explicit source profile:
        # Mode1 aligns g1-base/wrist TCP to Robot torso/palm; Mode2 preserves
        # its recording reference but moves the TCP control point to the palm.
        # This happens before the optional relative-action transform, and the
        # same grouped-column Rot6D path is used by norm-stat preparation.
        inputs.append(
            robot_policy.CanonicalizeHumanEEF(
                action_dim=spec.dimension,
                input_frame=spec.input_frame,
            )
        )
        if spec.task_progress_alignment is not None:
            inputs.append(
                ResampleAbsoluteEEFActions(
                    output_horizon=model_config.action_horizon,
                    source_step_scale=spec.task_progress_alignment["human_source_step_scale"],
                )
            )
    if representation == "relative":
        if spec.action_space != "eef":
            raise ValueError("Relative representation is only defined for EEF modes in this experiment")
        inputs.append(robot_policy.RelativeEEFActions(spec.dimension, "columns_grouped"))
        if spec.kind == "robot":
            outputs.append(robot_policy.AbsoluteEEFActions(spec.dimension, "columns_grouped"))

    if spec.kind == "robot":
        inputs.append(robot_policy.UnitreeRobotInputs(spec.dimension, spec.camera_mode))
        outputs.append(robot_policy.UnitreeRobotOutputs(spec.dimension))
    else:
        inputs.append(
            human_policy.HumanInputs(
                spec.dimension,
                spec.camera_mode,
                spec.model_uses_state,
                allow_missing_cameras=spec.camera_mode == "three",
            )
        )
        outputs.append(human_policy.HumanOutputs(spec.dimension))

    component_model = dataclasses.replace(model_config, discrete_state_input=spec.model_uses_state)
    dataset_assets_dir = spec.dataset / "meta" / "openpi_assets"
    if spec.kind == "human":
        _validate_human_norm_manifest(
            spec,
            representation=representation,
            relative_norm=relative_norm,
            action_horizon=model_config.action_horizon,
        )
    return DataConfig(
        repo_id=str(spec.dataset),
        asset_id=spec.asset_id,
        norm_stats=load_norm_stats(dataset_assets_dir, spec.asset_id),
        repack_transforms=repack,
        data_transforms=_transforms.Group(inputs=inputs, outputs=outputs),
        model_transforms=ModelTransformFactory(track_modalities=True, mask_padded_action_dims=True)(component_model),
        use_quantile_norm=True,
        normalization_clip=5.0,
        action_sequence_keys=("action",),
        action_source_step_scale=None
        if spec.task_progress_alignment is None
        else spec.task_progress_alignment["human_source_step_scale"],
        prompt_from_task=True,
        train_episodes=spec.split.train,
        # None means validation is disabled. An empty tuple would be treated as
        # an explicit empty subset and could still construct a validation loader.
        validation_episodes=spec.split.validation or None,
        runtime_manifest=spec.metadata,
    )


@dataclasses.dataclass(frozen=True)
class UnitreeExperimentDataConfig(DataConfigFactory):
    """Build one source or a fixed-ratio mixture of fully transformed sources."""

    repo_id: str = "unitree_g1d_brainco_mixture"
    robot_mode: int = 2
    human_modes: tuple[int, ...] = ()
    action_representation: ActionRepresentation = "absolute"
    relative_norm: RelativeNorm = "shared"
    action_domain: ActionDomain = "brainco"
    robot_joint_dataset: pathlib.Path = DEFAULT_ROBOT_JOINT_DATASET
    robot_gripper_joint_dataset: pathlib.Path = DEFAULT_ROBOT_GRIPPER_JOINT_DATASET
    robot_eef_dataset: pathlib.Path = DEFAULT_ROBOT_EEF_DATASET
    human_dataset: pathlib.Path = DEFAULT_HUMAN_DATASET
    robot_gripper_eef_dataset: pathlib.Path = DEFAULT_ROBOT_GRIPPER_EEF_DATASET
    human_gripper_dataset: pathlib.Path = DEFAULT_HUMAN_GRIPPER_DATASET
    human_eef_only_dataset: pathlib.Path = DEFAULT_HUMAN_EEF_ONLY_DATASET
    human_eef_only: bool = False
    robot_fraction: float = 0.5
    task_progress_alignment: bool = False
    split_seed: int = 2026
    robot_validation_episodes: int = 0
    human_validation_episodes: int = 0
    max_train_episodes: int | None = None
    minimum_episode_frames: int = 10
    robot_task_contains: str | None = "fold clothes"
    # Kept at the end so positional construction of the existing factory
    # fields remains backward-compatible.
    robot_camera_mode: RobotCameraMode = "three"
    # Ego dataset mode is independent from OpenPI's Human visibility mode
    # 3/4/5/6. Current datasets must carry action_semantics; legacy Mode1 may
    # use extraction_meta provenance, while legacy Mode2 is always rejected.
    human_dataset_mode: HumanDatasetMode = "mode1"
    # Resolve the Ego source mode to the canonical Robot control convention by
    # default: Mode1 g1-base/wrist TCP -> torso/palm; Mode2 recording/wrist TCP
    # -> recording/palm. None means this mode-aware default.
    human_input_frame: HumanInputFrame | None = None

    def specs(self) -> tuple[ComponentSpec, ...]:
        if self.robot_mode not in (0, 1, 2):
            raise ValueError("robot_mode must be 0 (off), 1 (joint26/joint16 absolute), or 2 (EEF30/EEF20)")
        if self.action_domain not in ("brainco", "gripper"):
            raise ValueError(f"action_domain must be brainco or gripper, got {self.action_domain!r}")
        if self.robot_camera_mode not in ("single", "three"):
            raise ValueError(f"robot_camera_mode must be single or three, got {self.robot_camera_mode!r}")
        if self.human_dataset_mode not in ("mode1", "mode2"):
            raise ValueError(f"human_dataset_mode must be mode1 or mode2, got {self.human_dataset_mode!r}")
        if self.human_input_frame not in (
            None,
            "native",
            "torso_palm",
            "g1_base_tcp",
            "recording_tcp",
            "pelvis_wrist",
        ):
            raise ValueError(f"Unsupported human_input_frame: {self.human_input_frame!r}")
        human_input_frame: HumanInputFrame = self.human_input_frame or (
            "g1_base_tcp" if self.human_dataset_mode == "mode1" else "recording_tcp"
        )
        unique_human_modes = tuple(dict.fromkeys(self.human_modes))
        invalid_human = sorted(set(unique_human_modes) - {3, 4, 5, 6})
        if invalid_human:
            raise ValueError(f"human_modes only supports 3,4,5,6; got {invalid_human}")
        if len(unique_human_modes) > 1:
            raise ValueError(
                "Select at most one Human mode. Mixed training is robot mode2 + exactly one matching Human mode."
            )
        if self.robot_mode == 0 and not self.human_modes:
            raise ValueError("Enable at least one robot or human mode")
        if self.robot_mode == 1 and self.action_representation != "absolute":
            raise ValueError("robot mode1 only supports absolute joint actions")
        if self.robot_mode == 1 and self.human_modes:
            raise ValueError("Joint-space robot mode1 cannot be mixed with EEF human data; use robot mode2")
        if self.human_eef_only and not self.human_modes:
            raise ValueError("human_eef_only requires one Human mode")
        if self.task_progress_alignment and not self.human_modes:
            raise ValueError("Task-progress alignment requires one Human EEF component")
        if self.task_progress_alignment and self.robot_mode not in (0, 2):
            raise ValueError("Task-progress alignment requires Robot mode2 EEF as its duration reference")
        if not 0.0 < self.robot_fraction < 1.0 and self.robot_mode and self.human_modes:
            raise ValueError("robot_fraction must lie strictly between 0 and 1 for mixed training")

        specs: list[ComponentSpec] = []
        if self.robot_mode:
            if self.robot_mode == 1:
                dataset = _resolve_dataset(
                    self.robot_gripper_joint_dataset if self.action_domain == "gripper" else self.robot_joint_dataset
                )
                _validate_joint_vector_layout(dataset, self.action_domain)
            elif self.action_domain == "gripper":
                dataset = _resolve_dataset(self.robot_gripper_eef_dataset)
            else:
                dataset = _resolve_dataset(self.robot_eef_dataset)
            eef_kinematics = _validate_robot_eef_metadata(dataset, self.action_domain) if self.robot_mode == 2 else None
            split = make_episode_split(
                dataset,
                validation_count=self.robot_validation_episodes,
                seed=self.split_seed,
                task_contains=self.robot_task_contains,
                max_train_episodes=self.max_train_episodes,
                minimum_episode_frames=self.minimum_episode_frames,
            )
            if self.robot_mode == 1:
                prefix = "g1d_robot_joint16_gripper2" if self.action_domain == "gripper" else "g1d_robot_joint26"
            elif self.action_domain == "gripper":
                prefix = "g1d_robot_eef20_gripper2_torso_palm_colgroup"
            else:
                prefix = "g1d_robot_eef30_torso_palm_colgroup"
            specs.append(
                ComponentSpec(
                    name=f"robot_mode{self.robot_mode}",
                    kind="robot",
                    mode=self.robot_mode,
                    dataset=dataset,
                    dimension=(16 if self.action_domain == "gripper" else 26)
                    if self.robot_mode == 1
                    else (20 if self.action_domain == "gripper" else 30),
                    camera_mode=self.robot_camera_mode,
                    model_uses_state=True,
                    action_space="joint" if self.robot_mode == 1 else "eef",
                    asset_id=_asset_id(prefix, self.action_representation, self.relative_norm, split),
                    split=split,
                    action_domain=self.action_domain,
                    eef_kinematics=eef_kinematics,
                    input_frame="torso_palm",
                )
            )

        human_dataset = None
        human_split = None
        human_source_dimension = None
        human_dataset_contract = None
        if self.human_modes:
            if self.human_eef_only:
                human_dataset = _resolve_dataset(self.human_eef_only_dataset)
                human_action_domain: ComponentActionDomain = "eef_only"
                human_dimension = 18
            else:
                human_dataset = _resolve_dataset(
                    self.human_gripper_dataset if self.action_domain == "gripper" else self.human_dataset
                )
                human_action_domain = self.action_domain
                human_dimension = 20 if self.action_domain == "gripper" else 30
            human_source_dimension = _validate_eef_vector_layout(human_dataset, human_action_domain)
            human_dataset_contract = _resolve_human_dataset_contract(human_dataset, self.human_dataset_mode)
            if human_input_frame in ("g1_base_tcp", "pelvis_wrist") and self.human_dataset_mode != "mode1":
                raise ValueError(
                    f"human_input_frame={human_input_frame} is only valid for Ego Mode1 g1-base TCP data"
                )
            if human_input_frame == "recording_tcp" and self.human_dataset_mode != "mode2":
                raise ValueError("human_input_frame=recording_tcp is only valid for Ego Mode2 data")
            if self.human_dataset_mode == "mode1" and human_input_frame not in ("g1_base_tcp", "pelvis_wrist"):
                raise ValueError(
                    "Ego Mode1 Human data must be canonicalized from G1 pelvis/wrist-yaw TCP to "
                    "OpenPI torso/palm for both Human-only and mixed training; use "
                    "human_input_frame=g1_base_tcp."
                )
            if self.human_dataset_mode == "mode2" and human_input_frame != "recording_tcp":
                raise ValueError(
                    "Ego Mode2 Human data must preserve its recording reference while moving the control point "
                    "from wrist-yaw TCP to palm; use human_input_frame=recording_tcp."
                )
            human_split = make_episode_split(
                human_dataset,
                validation_count=self.human_validation_episodes,
                seed=self.split_seed,
                max_train_episodes=self.max_train_episodes,
                minimum_episode_frames=self.minimum_episode_frames,
            )
        for mode in unique_human_modes:
            assert human_dataset is not None
            assert human_split is not None
            assert human_dataset_contract is not None
            uses_state = mode in (3, 5)
            camera_mode = "single" if mode in (3, 4) else "three"
            # The source dataset is not necessarily torso->palm. Keep only the
            # numeric layout in the prefix; the exact source/control-point
            # contract is encoded below and verified by its digest.
            human_asset_prefix = (
                "g1d_human_eef18_only_colgroup"
                if self.human_eef_only
                else (
                    "g1d_human_eef20_gripper2_colgroup"
                    if self.action_domain == "gripper"
                    else "g1d_human_eef30_colgroup"
                )
            )
            # Source-frame identity is part of the numeric contract. Never
            # reuse the old unqualified namespace: it cannot prove whether a
            # frame conversion was applied once, twice, or not at all.
            human_asset_prefix = (
                f"{human_asset_prefix}_ego_{self.human_dataset_mode}"
                f"_source_{human_input_frame}_contract_{human_dataset_contract['digest']}"
            )
            specs.append(
                ComponentSpec(
                    name=f"human_mode{mode}",
                    kind="human",
                    mode=mode,
                    dataset=human_dataset,
                    dimension=human_dimension,
                    camera_mode=camera_mode,
                    model_uses_state=uses_state,
                    action_space="eef",
                    # Camera/state visibility does not alter raw numeric stats,
                    # so all selected human modes intentionally share this asset.
                    asset_id=_asset_id(
                        human_asset_prefix,
                        self.action_representation,
                        self.relative_norm,
                        human_split,
                    ),
                    split=human_split,
                    action_domain=human_action_domain,
                    source_dimension=(human_source_dimension if human_source_dimension != human_dimension else None),
                    input_frame=human_input_frame,
                    human_dataset_contract=human_dataset_contract,
                )
            )
        if self.robot_mode and self.human_modes:
            if human_input_frame == "native":
                raise ValueError(
                    "Mixed Robot/Human training cannot use human_input_frame=native: Robot EEF is "
                    "torso_link→hand_palm_link, while Ego Mode1 is g1_base→wrist_yaw_tcp and Ego Mode2 is "
                    "recording_frame→wrist_yaw_tcp. Use g1_base_tcp for Mode1; use recording_tcp with a "
                    "relative preset for Mode2."
                )
            if self.human_dataset_mode == "mode2" and self.action_representation == "absolute":
                raise ValueError(
                    "Absolute Robot/Human mixing is unsafe for Ego Mode2: its absolute TCP is in the recording "
                    "frame and no per-frame recording→robot placement exists. Use a relative preset with "
                    "human_input_frame=recording_tcp, or use Mode1 for absolute mixing."
                )
            if self.human_dataset_mode == "mode1" and human_input_frame not in (
                "g1_base_tcp",
                "pelvis_wrist",
            ):
                raise ValueError(
                    "Ego Mode1 mixed training requires human_input_frame=g1_base_tcp so both domains become "
                    "torso_link→hand_palm_link."
                )
            if self.human_dataset_mode == "mode2" and human_input_frame != "recording_tcp":
                raise ValueError(
                    "Ego Mode2 relative mixed training requires human_input_frame=recording_tcp so the Human "
                    "wrist-yaw TCP is moved to the same palm control point before relative conversion."
                )
        if self.task_progress_alignment:
            human_spec = next(spec for spec in specs if spec.kind == "human")
            robot_spec = next((spec for spec in specs if spec.kind == "robot"), None)
            if robot_spec is None:
                # Normalization is intentionally computed one domain at a time.
                # A Human progress asset still uses the selected Robot dataset
                # and split as the immutable duration reference for gamma.
                reference_dataset = _resolve_dataset(
                    self.robot_gripper_eef_dataset if self.action_domain == "gripper" else self.robot_eef_dataset
                )
                _validate_robot_eef_metadata(reference_dataset, self.action_domain)
                reference_split = make_episode_split(
                    reference_dataset,
                    validation_count=self.robot_validation_episodes,
                    seed=self.split_seed,
                    task_contains=self.robot_task_contains,
                    max_train_episodes=self.max_train_episodes,
                    minimum_episode_frames=self.minimum_episode_frames,
                )
            else:
                reference_dataset = robot_spec.dataset
                reference_split = robot_spec.split
            alignment = estimate_task_progress_alignment(
                reference_dataset,
                reference_split.train,
                human_spec.dataset,
                human_spec.split.train,
            )
            for index, spec in enumerate(specs):
                if spec.kind == "human":
                    specs[index] = dataclasses.replace(
                        spec,
                        asset_id=f"{spec.asset_id}_progress_{alignment['digest']}",
                        task_progress_alignment=alignment,
                    )
        return tuple(specs)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        del assets_dirs
        specs = self.specs()
        eval_spec = next((spec for spec in specs if spec.kind == "robot"), None)
        components = tuple(
            _make_component_data_config(
                spec,
                model_config=model_config,
                representation=self.action_representation,
                relative_norm=self.relative_norm,
                load_norm_stats=self._load_norm_stats,
            )
            for spec in specs
        )
        progress_alignment = next(
            (spec.task_progress_alignment for spec in specs if spec.task_progress_alignment is not None),
            None,
        )
        if progress_alignment is not None:
            source_scale = progress_alignment["human_source_step_scale"]
            progress_alignment = {
                **progress_alignment,
                "output_horizon": model_config.action_horizon,
                "human_source_horizon": required_source_horizon(model_config.action_horizon, source_scale),
                "human_source_span_frames": (model_config.action_horizon - 1) * source_scale,
            }
        manifest = {
            "schema_version": 2,
            "training_config": TRAIN_CONFIG_NAME,
            "action_representation": self.action_representation,
            "action_domain": self.action_domain,
            "action_tail": "gripper2" if self.action_domain == "gripper" else "brainco12",
            "relative_norm": self.relative_norm,
            "rotation_6d": "first_column_then_second_column",
            "rotation_6d_layout": G1D_ROTATION_6D_LAYOUT,
            # Top-level frame/link fields are a real-robot deployment
            # contract, not a claim about native Human-only coordinates.
            "reference_frame": None if eval_spec is None else G1D_REFERENCE_FRAME,
            "eef_links": None if eval_spec is None else list(G1D_EEF_LINKS),
            "frame_convention": None if eval_spec is None else "T_reference_eef",
            "urdf_sha256": None
            if eval_spec is None or eval_spec.eef_kinematics is None
            else eval_spec.eef_kinematics["urdf_sha256"],
            "dataset_fps": 30,
            "action_horizon": model_config.action_horizon,
            "normalization_clip": 5.0,
            "mask_padded_action_dims": True,
            "mask_padded_timesteps": True,
            "task_progress_alignment": progress_alignment,
            "robot_camera_mode": None if eval_spec is None else eval_spec.camera_mode,
            "robot_task_contains": self.robot_task_contains,
            "human_dataset_mode": self.human_dataset_mode if any(spec.kind == "human" for spec in specs) else None,
            "human_input_frame": next((spec.input_frame for spec in specs if spec.kind == "human"), None),
            # Metadata consumed by examples/unitree_inference/policy_adapter.py.
            "robot_type": (
                None
                if eval_spec is None
                else ("unitree_g1_dex1" if self.action_domain == "gripper" else "unitree_g1_brainco")
            ),
            "action_space": None if eval_spec is None else eval_spec.action_space,
            "end_effector": (
                None if eval_spec is None else ("dex1" if self.action_domain == "gripper" else "brainco")
            ),
            "model_dim": None if eval_spec is None else eval_spec.dimension,
            "rotation_format": "columns_grouped",
            "relative": self.action_representation == "relative",
            "components": [spec.metadata for spec in specs],
            "eval_asset_id": None if eval_spec is None else eval_spec.asset_id,
        }
        if self.human_eef_only:
            manifest["human_action_domain"] = "eef_only"
        if len(components) == 1:
            return dataclasses.replace(components[0], runtime_manifest=manifest)

        robot_count = sum(spec.kind == "robot" for spec in specs)
        human_count = len(specs) - robot_count
        if robot_count and human_count:
            weights = tuple(
                self.robot_fraction / robot_count if spec.kind == "robot" else (1.0 - self.robot_fraction) / human_count
                for spec in specs
            )
        else:
            weights = tuple(1.0 / len(specs) for _ in specs)
        return DataConfig(
            mixture_components=components,
            mixture_weights=weights,
            mixture_names=tuple(spec.name for spec in specs),
            runtime_manifest=manifest,
        )


def build_train_config(
    *,
    exp_name: str,
    robot_mode: int = 2,
    human_modes: Sequence[int] = (),
    action_representation: ActionRepresentation = "absolute",
    relative_norm: RelativeNorm = "shared",
    action_domain: ActionDomain = "brainco",
    robot_camera_mode: RobotCameraMode = "three",
    human_dataset_mode: HumanDatasetMode = "mode1",
    human_input_frame: HumanInputFrame | None = None,
    robot_joint_dataset: pathlib.Path = DEFAULT_ROBOT_JOINT_DATASET,
    robot_gripper_joint_dataset: pathlib.Path = DEFAULT_ROBOT_GRIPPER_JOINT_DATASET,
    robot_eef_dataset: pathlib.Path = DEFAULT_ROBOT_EEF_DATASET,
    human_dataset: pathlib.Path = DEFAULT_HUMAN_DATASET,
    robot_gripper_eef_dataset: pathlib.Path = DEFAULT_ROBOT_GRIPPER_EEF_DATASET,
    human_gripper_dataset: pathlib.Path = DEFAULT_HUMAN_GRIPPER_DATASET,
    human_eef_only_dataset: pathlib.Path = DEFAULT_HUMAN_EEF_ONLY_DATASET,
    human_eef_only: bool = False,
    robot_fraction: float = 0.5,
    task_progress_alignment: bool = False,
    split_seed: int = 2026,
    robot_validation_episodes: int = 0,
    human_validation_episodes: int = 0,
    max_train_episodes: int | None = None,
    minimum_episode_frames: int = 10,
    robot_task_contains: str | None = "fold clothes",
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
    data = UnitreeExperimentDataConfig(
        assets=AssetsConfig(),
        robot_mode=robot_mode,
        human_modes=tuple(human_modes),
        action_representation=action_representation,
        relative_norm=relative_norm,
        action_domain=action_domain,
        robot_camera_mode=robot_camera_mode,
        human_dataset_mode=human_dataset_mode,
        human_input_frame=human_input_frame,
        robot_joint_dataset=robot_joint_dataset,
        robot_gripper_joint_dataset=robot_gripper_joint_dataset,
        robot_eef_dataset=robot_eef_dataset,
        human_dataset=human_dataset,
        robot_gripper_eef_dataset=robot_gripper_eef_dataset,
        human_gripper_dataset=human_gripper_dataset,
        human_eef_only_dataset=human_eef_only_dataset,
        human_eef_only=human_eef_only,
        robot_fraction=robot_fraction,
        task_progress_alignment=task_progress_alignment,
        split_seed=split_seed,
        robot_validation_episodes=robot_validation_episodes,
        human_validation_episodes=human_validation_episodes,
        max_train_episodes=max_train_episodes,
        minimum_episode_frames=minimum_episode_frames,
        robot_task_contains=robot_task_contains,
    )
    validation_enabled = (robot_mode != 0 and robot_validation_episodes > 0) or (
        bool(human_modes) and human_validation_episodes > 0
    )
    config = TrainConfig(
        name=TRAIN_CONFIG_NAME,
        project_name="openpi-unitree-g1d",
        exp_name=exp_name,
        model=pi0_config.Pi0RTCConfig(pi05=True, discrete_state_input=True),
        data=data,
        weight_loader=weight_loaders.CheckpointWeightLoader(base_checkpoint),
        batch_size=batch_size,
        fsdp_devices=fsdp_devices,
        num_train_steps=num_train_steps,
        num_workers=num_workers,
        # With no held-out episodes, disable the complete validation path:
        # loader construction, validation forward passes, and modality ablation.
        validation_interval=validation_interval if validation_enabled else 0,
        validation_batches=validation_batches if validation_enabled else 0,
        modality_diagnostics_interval=modality_diagnostics_interval if validation_enabled else 0,
        save_interval=save_interval,
        keep_period=keep_period,
        seed=seed,
        overwrite=overwrite,
        resume=resume,
        wandb_enabled=wandb_enabled,
    )
    # Fail fast on invalid combinations and attach a serializable manifest for
    # W&B and checkpoint compatibility checks.
    resolved = data.create(config.assets_dirs, config.model)
    return dataclasses.replace(config, policy_metadata=resolved.runtime_manifest)


def component_configs(config: TrainConfig) -> tuple[DataConfig, ...]:
    """Return resolved numeric domains for norm/validation tooling."""
    resolved = config.data.create(config.assets_dirs, config.model)
    return tuple(resolved.mixture_components) or (resolved,)


def validate_resume_manifest(config: TrainConfig) -> None:
    """Reject resuming a checkpoint whose numeric/data contract has changed."""
    if not config.resume:
        return
    checkpoint_root = config.checkpoint_dir
    if not checkpoint_root.is_dir():
        return
    manifests = list(checkpoint_root.glob("*/assets/runtime_manifest.json"))
    if not manifests:
        return

    def step_key(path: pathlib.Path) -> tuple[int, str]:
        step = path.parent.parent.name
        return (int(step) if step.isdigit() else -1, step)

    latest_path = max(manifests, key=step_key)
    actual = json.loads(latest_path.read_text(encoding="utf-8"))
    expected = config.policy_metadata or {}
    contract_keys = (
        "action_representation",
        "action_domain",
        "relative_norm",
        "robot_camera_mode",
        "human_dataset_mode",
        "human_input_frame",
        "task_progress_alignment",
        "eval_asset_id",
    )
    mismatches = {
        key: (actual.get(key), expected.get(key)) for key in contract_keys if actual.get(key) != expected.get(key)
    }
    component_keys = (
        "name",
        "kind",
        "mode",
        "dataset",
        "dimension",
        "camera_mode",
        "model_uses_state",
        "action_space",
        "action_domain",
        "asset_id",
        "split_digest",
        "input_frame",
        "human_dataset_contract",
    )
    actual_components = [
        {key: component.get(key) for key in component_keys} for component in actual.get("components", [])
    ]
    expected_components = [
        {key: component.get(key) for key in component_keys} for component in expected.get("components", [])
    ]
    # Old Robot-only manifests predate the explicit input_frame field but are
    # unambiguously torso→palm. Human manifests may never use that fallback.
    if not any(component["kind"] == "human" for component in actual_components):
        for component in actual_components:
            if component["kind"] == "robot" and component["input_frame"] is None:
                component["input_frame"] = "torso_palm"
    if actual_components != expected_components:
        mismatches["components"] = (actual_components, expected_components)
    if mismatches:
        raise ValueError(
            f"Refusing to resume incompatible Unitree checkpoint {latest_path.parent.parent}: {mismatches}. "
            "Use TRAIN_LIFECYCLE=new with a new EXP_NAME after recomputing normalization stats."
        )
