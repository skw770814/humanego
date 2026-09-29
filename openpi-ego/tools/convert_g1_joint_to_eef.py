#!/usr/bin/env python3
"""Convert G1-D joint data to EEF30-BrainCo or EEF20-gripper data.

The source schema is the action-domain selector. A 26D source must be arm14 +
BrainCo12, while a 16D source must be arm14 + left/right gripper. Tail values
are copied exactly; this converter never derives gripper commands from fingers.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
from datetime import UTC
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

from lerobot.common.datasets.compute_stats import aggregate_stats
from lerobot.common.datasets.utils import load_episodes_stats
from lerobot.common.datasets.utils import write_stats
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from .g1_d_kinematics import LEFT_EEF_LINK
    from .g1_d_kinematics import REFERENCE_LINK
    from .g1_d_kinematics import RIGHT_EEF_LINK
    from .g1_d_kinematics import ROTATION_6D_LAYOUT
    from .g1_d_kinematics import G1DForwardKinematics
except ImportError:  # Direct execution: python tools/convert_g1_joint_to_eef.py ...
    from g1_d_kinematics import LEFT_EEF_LINK
    from g1_d_kinematics import REFERENCE_LINK
    from g1_d_kinematics import RIGHT_EEF_LINK
    from g1_d_kinematics import ROTATION_6D_LAYOUT
    from g1_d_kinematics import G1DForwardKinematics


ARM_JOINT_NAMES = (
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
BRAINCO_NAMES = (
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
GRIPPER_NAMES = ("kLeftGripper", "kRightGripper")

EEF_NAMES = [
    "left_eef_x",
    "left_eef_y",
    "left_eef_z",
    "left_rot_r00",
    "left_rot_r10",
    "left_rot_r20",
    "left_rot_r01",
    "left_rot_r11",
    "left_rot_r21",
    "right_eef_x",
    "right_eef_y",
    "right_eef_z",
    "right_rot_r00",
    "right_rot_r10",
    "right_rot_r20",
    "right_rot_r01",
    "right_rot_r11",
    "right_rot_r21",
]


@dataclasses.dataclass(frozen=True)
class ActionLayout:
    action_domain: str
    tail_names: tuple[str, ...]
    robot_type: str

    @property
    def source_names(self) -> tuple[str, ...]:
        return (*ARM_JOINT_NAMES, *self.tail_names)

    @property
    def output_names(self) -> tuple[str, ...]:
        return (*EEF_NAMES, *self.tail_names)

    @property
    def output_dim(self) -> int:
        return 18 + len(self.tail_names)

    @property
    def modality(self) -> dict:
        result = {
            "wrist_left_pose": {"start": 0, "end": 9},
            "wrist_right_pose": {"start": 9, "end": 18},
        }
        if self.action_domain == "brainco":
            result.update(
                {
                    "fingers_left_qpos": {"start": 18, "end": 24},
                    "fingers_right_qpos": {"start": 24, "end": 30},
                }
            )
        else:
            result.update(
                {
                    "gripper_left_qpos": {"start": 18, "end": 19},
                    "gripper_right_qpos": {"start": 19, "end": 20},
                }
            )
        return result


ACTION_LAYOUTS = (
    ActionLayout("brainco", BRAINCO_NAMES, "Unitree_G1_D_Brainco_EEF_columns"),
    ActionLayout("gripper", GRIPPER_NAMES, "Unitree_G1_Dex1_EEF_columns"),
)
DEFAULT_URDF = Path(__file__).resolve().parents[1] / "third_party/g1_d_description/g1_d.urdf"


def _dataset_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "meta" / "info.json").is_file():
        return path
    candidates = [child for child in path.iterdir() if (child / "meta" / "info.json").is_file()]
    if len(candidates) != 1:
        raise ValueError(f"Expected one LeRobot dataset below {path}, found {len(candidates)}")
    return candidates[0]


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)
        file.write("\n")


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for value in values:
            file.write(json.dumps(value, ensure_ascii=False) + "\n")


def _parse_episodes(value: str, available: set[int]) -> list[int]:
    selected = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", maxsplit=1)
            start, end = int(start_text), int(end_text)
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    unknown = selected - available
    if unknown:
        raise ValueError(f"Unknown episode indices: {sorted(unknown)[:10]}")
    return sorted(selected)


def _episode_task_index(episode: dict, task_ids_by_name: dict[str, int]) -> int:
    """Resolve both LeRobot v2.1 episode task encodings.

    Some datasets store task_index directly, while others only keep a one-item
    tasks string list. Parquet still uses the numeric task_index in both cases.
    """
    if "task_index" in episode:
        return int(episode["task_index"])
    task_names = episode.get("tasks")
    if not isinstance(task_names, list) or len(task_names) != 1 or task_names[0] not in task_ids_by_name:
        raise ValueError(f"Episode {episode.get('episode_index')} has no unambiguous task mapping: {task_names!r}")
    return task_ids_by_name[task_names[0]]


def _as_matrix(column: pa.ChunkedArray, name: str, expected_rows: int, vector_dim: int) -> np.ndarray:
    try:
        values = np.asarray(column.to_pylist(), dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Parquet column {name!r} is not a rectangular numeric array") from exc
    if values.shape != (expected_rows, vector_dim):
        raise ValueError(
            f"Parquet column {name!r} must have shape ({expected_rows}, {vector_dim}), got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"Parquet column {name!r} contains NaN or infinity")
    return values


def _fixed_float_list(values: np.ndarray) -> pa.Array:
    values = np.asarray(values, dtype=np.float32)
    return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1), type=pa.float32()), values.shape[-1])


def _replace_column(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        raise KeyError(f"Missing parquet column {name!r}")
    return table.set_column(index, name, values)


def _replace_int_column(table: pa.Table, name: str, values) -> pa.Table:
    return _replace_column(table, name, pa.array(values, type=pa.int64()))


def _validate_source_schema(info: dict) -> ActionLayout:
    features = info.get("features", {})
    matched_layout: ActionLayout | None = None
    for feature_key in ("observation.state", "action"):
        feature = features.get(feature_key)
        if feature is None:
            raise ValueError(f"Source metadata is missing feature {feature_key!r}")
        matches = [
            layout
            for layout in ACTION_LAYOUTS
            if feature.get("shape") == [len(layout.source_names)]
            and feature.get("names") == [list(layout.source_names)]
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Source {feature_key!r} must be exactly arm14+BrainCo12 or arm14+gripper2; "
                f"got shape={feature.get('shape')}, names={feature.get('names')}. Conversion is refused because "
                "a silent joint permutation would corrupt FK or end-effector commands."
            )
        if matched_layout is not None and matches[0] != matched_layout:
            raise ValueError("observation.state and action use different end-effector layouts")
        matched_layout = matches[0]
    assert matched_layout is not None
    return matched_layout


def _validate_int_column(table: pa.Table, name: str, expected: np.ndarray, source_data: Path) -> None:
    if name not in table.column_names:
        raise ValueError(f"Missing parquet column {name!r} in {source_data}")
    actual = np.asarray(table[name].to_pylist(), dtype=np.int64)
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        raise ValueError(f"Unexpected {name!r} values in {source_data}")


def _rotation_orthogonality_error(values: np.ndarray) -> float:
    errors = []
    for offset in (3, 12):
        first = values[:, offset : offset + 3]
        second = values[:, offset + 3 : offset + 6]
        errors.extend(
            (
                np.abs(np.linalg.norm(first, axis=1) - 1.0),
                np.abs(np.linalg.norm(second, axis=1) - 1.0),
                np.abs(np.sum(first * second, axis=1)),
            )
        )
    return float(max(np.max(error) for error in errors))


def _load_and_convert_episode(
    source_data: Path,
    *,
    length: int,
    old_index: int,
    old_task: int,
    kinematics: G1DForwardKinematics,
    layout: ActionLayout,
) -> tuple[pa.Table, np.ndarray, np.ndarray, float]:
    table = pq.read_table(source_data)
    if table.num_rows != length:
        raise ValueError(f"Frame count mismatch in {source_data}: metadata={length}, parquet={table.num_rows}")
    required = {"observation.state", "action", "frame_index", "episode_index", "index", "task_index"}
    missing = required - set(table.column_names)
    if missing:
        raise ValueError(f"Missing parquet columns in {source_data}: {sorted(missing)}")
    _validate_int_column(table, "frame_index", np.arange(length, dtype=np.int64), source_data)
    _validate_int_column(table, "episode_index", np.full(length, old_index, dtype=np.int64), source_data)
    _validate_int_column(table, "task_index", np.full(length, old_task, dtype=np.int64), source_data)

    source_dim = len(layout.source_names)
    joint_states = _as_matrix(table["observation.state"], "observation.state", length, source_dim)
    joint_actions = _as_matrix(table["action"], "action", length, source_dim)
    eef_states = np.concatenate((kinematics.forward(joint_states[:, :14]), joint_states[:, 14:]), axis=-1)
    eef_actions = np.concatenate((kinematics.forward(joint_actions[:, :14]), joint_actions[:, 14:]), axis=-1)
    if eef_states.shape != (length, layout.output_dim) or eef_actions.shape != (length, layout.output_dim):
        raise AssertionError(f"Internal error: converted EEF arrays do not have shape (frames, {layout.output_dim})")
    if not np.array_equal(eef_states[:, 18:], joint_states[:, 14:]):
        raise AssertionError(f"Internal error: {layout.action_domain} state values changed during conversion")
    if not np.array_equal(eef_actions[:, 18:], joint_actions[:, 14:]):
        raise AssertionError(f"Internal error: {layout.action_domain} action values changed during conversion")
    rotation_error = max(_rotation_orthogonality_error(eef_states), _rotation_orthogonality_error(eef_actions))
    if rotation_error > 2e-5:
        raise ValueError(f"Invalid 6D rotation generated for {source_data}: error={rotation_error:.3g}")
    return table, eef_states, eef_actions, rotation_error


def _feature_stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [len(values)],
    }


def _constant_stats(value: int, count: int) -> dict:
    return {"min": [value], "max": [value], "mean": [float(value)], "std": [0.0], "count": [count]}


def _sequence_stats(start: int, count: int) -> dict:
    end = start + count - 1
    return {
        "min": [start],
        "max": [end],
        "mean": [(start + end) / 2],
        "std": [math.sqrt((count**2 - 1) / 12) if count > 1 else 0.0],
        "count": [count],
    }


def _link_or_copy(source: Path, destination: Path, *, copy_files: bool) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy_files:
        shutil.copy2(source, destination)
        return "copy"
    try:
        os.link(source, destination)
        return "link"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _output_modality(source: Path, video_keys: list[str], layout: ActionLayout) -> dict:
    modality_path = source / "meta" / "modality.json"
    modality = copy.deepcopy(_read_json(modality_path)) if modality_path.is_file() else {}
    modality["state"] = copy.deepcopy(layout.modality)
    modality["action"] = copy.deepcopy(layout.modality)
    modality["video"] = {key: {"original_key": key} for key in video_keys}
    modality.setdefault("annotation", {"human.task_description": {"original_key": "task_index"}})
    return modality


def convert_dataset(
    source: Path,
    output: Path,
    urdf: Path,
    selected: list[int] | None,
    *,
    copy_videos: bool,
    dry_run: bool = False,
    output_task: str | None = None,
) -> dict:
    source = _dataset_root(source)
    output = output.expanduser().resolve()
    urdf = urdf.expanduser().resolve()
    if output.exists() and not dry_run:
        raise FileExistsError(output)
    info = _read_json(source / "meta" / "info.json")
    layout = _validate_source_schema(info)
    episodes = _read_jsonl(source / "meta" / "episodes.jsonl")
    task_rows = _read_jsonl(source / "meta" / "tasks.jsonl")
    task_ids_by_name = {row["task"]: int(row["task_index"]) for row in task_rows}
    episodes = [{**row, "task_index": _episode_task_index(row, task_ids_by_name)} for row in episodes]
    episode_by_index = {row["episode_index"]: row for row in episodes}
    if len(episode_by_index) != len(episodes):
        raise ValueError("Duplicate episode indices in source metadata")
    tasks = {int(row["task_index"]): row["task"] for row in task_rows}
    selected = sorted(episode_by_index) if selected is None else selected
    if not selected:
        raise ValueError("At least one episode must be selected")
    unknown_episodes = set(selected) - set(episode_by_index)
    if unknown_episodes:
        raise ValueError(f"Unknown episode indices: {sorted(unknown_episodes)[:10]}")
    selected = sorted(set(selected))

    for feature_key in ("observation.state", "action"):
        info["features"][feature_key] = {
            "dtype": "float32",
            "shape": [layout.output_dim],
            "names": [list(layout.output_names)],
        }

    used_tasks = sorted({episode_by_index[index]["task_index"] for index in selected})
    missing_tasks = set(used_tasks) - set(tasks)
    if missing_tasks:
        raise ValueError(f"Episodes refer to missing task indices: {sorted(missing_tasks)}")
    if output_task is not None:
        output_task = output_task.strip()
        if not output_task:
            raise ValueError("output_task must not be empty")
        task_remap = dict.fromkeys(used_tasks, 0)
        output_tasks = [{"task_index": 0, "task": output_task}]
    else:
        task_remap = {old: new for new, old in enumerate(used_tasks)}
        output_tasks = [{"task_index": task_remap[index], "task": tasks[index]} for index in used_tasks]
    info["robot_type"] = layout.robot_type
    info["total_episodes"] = len(selected)
    info["total_frames"] = sum(episode_by_index[index]["length"] for index in selected)
    info["total_tasks"] = len(output_tasks)
    info["total_chunks"] = math.ceil(len(selected) / info["chunks_size"])
    info["splits"] = {"train": f"0:{len(selected)}"}
    video_keys = [key for key, value in info["features"].items() if value.get("dtype") == "video"]
    info["total_videos"] = len(selected) * len(video_keys)
    urdf_sha256 = hashlib.sha256(urdf.read_bytes()).hexdigest()
    info["eef_forward_kinematics"] = {
        "urdf_sha256": urdf_sha256,
        "reference_link": REFERENCE_LINK,
        "left_eef_link": LEFT_EEF_LINK,
        "right_eef_link": RIGHT_EEF_LINK,
        "frame_convention": "T_reference_eef",
        "rotation_6d": ROTATION_6D_LAYOUT,
        layout.action_domain: "copied_without_transformation",
    }

    kinematics = G1DForwardKinematics(urdf)
    data_path_template = info["data_path"]
    if dry_run:
        frame_count = 0
        max_rotation_error = 0.0
        tail_min = np.full(len(layout.tail_names), np.inf, dtype=np.float64)
        tail_max = np.full(len(layout.tail_names), -np.inf, dtype=np.float64)
        for old_index in selected:
            episode = episode_by_index[old_index]
            old_chunk = old_index // info["chunks_size"]
            source_data = source / data_path_template.format(
                episode_chunk=old_chunk,
                episode_index=old_index,
            )
            _, eef_states, eef_actions, rotation_error = _load_and_convert_episode(
                source_data,
                length=episode["length"],
                old_index=old_index,
                old_task=episode["task_index"],
                kinematics=kinematics,
                layout=layout,
            )
            tail_values = np.concatenate((eef_states[:, 18:], eef_actions[:, 18:]), axis=0)
            tail_min = np.minimum(tail_min, tail_values.min(axis=0))
            tail_max = np.maximum(tail_max, tail_values.max(axis=0))
            max_rotation_error = max(max_rotation_error, rotation_error)
            frame_count += episode["length"]
        result = {
            "dry_run": True,
            "output": str(output),
            "urdf": str(urdf),
            "urdf_sha256": urdf_sha256,
            "reference_link": REFERENCE_LINK,
            "eef_links": [LEFT_EEF_LINK, RIGHT_EEF_LINK],
            "frame_convention": "T_reference_eef",
            "rotation_6d": ROTATION_6D_LAYOUT,
            "episodes": len(selected),
            "source_episode_indices": selected,
            "frames": frame_count,
            "source_tasks": [tasks[index] for index in used_tasks],
            "output_tasks": [row["task"] for row in output_tasks],
            "rotation_orthogonality_max_error": max_rotation_error,
            "action_domain": layout.action_domain,
            "tail_passthrough": True,
            "tail_min": tail_min.tolist(),
            "tail_max": tail_max.tolist(),
        }
        # Preserve the original dry-run report keys for callers that parse the
        # legacy BrainCo converter output.
        if layout.action_domain == "brainco":
            result.update(
                {
                    "brainco_passthrough": True,
                    "brainco_min": tail_min.tolist(),
                    "brainco_max": tail_max.tolist(),
                }
            )
        return result

    stats_rows = _read_jsonl(source / "meta" / "episodes_stats.jsonl")
    stats_by_index = {row["episode_index"]: row for row in stats_rows}
    if len(stats_by_index) != len(stats_rows):
        raise ValueError("Duplicate episode indices in source statistics")
    missing_stats = set(selected) - set(stats_by_index)
    if missing_stats:
        raise ValueError(f"Missing episode statistics for indices: {sorted(missing_stats)[:10]}")

    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    converted_episodes = []
    converted_stats = []
    provenance = []
    global_index = 0
    linked_videos = 0
    copied_videos = 0
    try:
        for new_index, old_index in enumerate(selected):
            episode = copy.deepcopy(episode_by_index[old_index])
            length = episode["length"]
            old_task = episode["task_index"]
            episode.update({"episode_index": new_index, "task_index": task_remap[old_task]})
            if output_task is not None and "tasks" in episode:
                episode["tasks"] = [output_task]
            converted_episodes.append(episode)
            old_chunk = old_index // info["chunks_size"]
            new_chunk = new_index // info["chunks_size"]
            source_data = source / info["data_path"].format(
                episode_chunk=old_chunk,
                episode_index=old_index,
            )
            destination_data = temporary / info["data_path"].format(
                episode_chunk=new_chunk,
                episode_index=new_index,
            )
            table, eef_states, eef_actions, _ = _load_and_convert_episode(
                source_data,
                length=length,
                old_index=old_index,
                old_task=old_task,
                kinematics=kinematics,
                layout=layout,
            )
            table = _replace_column(table, "observation.state", _fixed_float_list(eef_states))
            table = _replace_column(table, "action", _fixed_float_list(eef_actions))
            table = _replace_int_column(table, "episode_index", [new_index] * length)
            table = _replace_int_column(table, "index", range(global_index, global_index + length))
            table = _replace_int_column(table, "task_index", [task_remap[old_task]] * length)
            destination_data.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, destination_data, compression="snappy")

            episode_stats = copy.deepcopy(stats_by_index[old_index])
            episode_stats["episode_index"] = new_index
            episode_stats["stats"]["observation.state"] = _feature_stats(eef_states)
            episode_stats["stats"]["action"] = _feature_stats(eef_actions)
            episode_stats["stats"]["episode_index"] = _constant_stats(new_index, length)
            episode_stats["stats"]["index"] = _sequence_stats(global_index, length)
            episode_stats["stats"]["task_index"] = _constant_stats(task_remap[old_task], length)
            converted_stats.append(episode_stats)

            for video_key in video_keys:
                source_video = source / info["video_path"].format(
                    episode_chunk=old_chunk,
                    episode_index=old_index,
                    video_key=video_key,
                )
                destination_video = temporary / info["video_path"].format(
                    episode_chunk=new_chunk,
                    episode_index=new_index,
                    video_key=video_key,
                )
                mode = _link_or_copy(source_video, destination_video, copy_files=copy_videos)
                linked_videos += mode == "link"
                copied_videos += mode == "copy"
            provenance.append({"episode_index": new_index, "source_episode_index": old_index})
            global_index += length

        _write_json(temporary / "meta" / "info.json", info)
        _write_jsonl(temporary / "meta" / "episodes.jsonl", converted_episodes)
        _write_jsonl(temporary / "meta" / "episodes_stats.jsonl", converted_stats)
        _write_jsonl(temporary / "meta" / "tasks.jsonl", output_tasks)
        _write_json(temporary / "meta" / "modality.json", _output_modality(source, video_keys, layout))
        write_stats(aggregate_stats(list(load_episodes_stats(temporary).values())), temporary)
        _write_json(
            temporary / "meta" / "joint_to_eef_conversion.json",
            {
                "version": 1,
                "created_at": datetime.now(tz=UTC).isoformat(),
                "source_dataset": str(source),
                "urdf": str(urdf),
                "urdf_sha256": urdf_sha256,
                "reference_link": REFERENCE_LINK,
                "left_eef_link": LEFT_EEF_LINK,
                "right_eef_link": RIGHT_EEF_LINK,
                "frame_convention": "T_reference_eef",
                "rotation_6d": ROTATION_6D_LAYOUT,
                "action_domain": layout.action_domain,
                layout.action_domain: "copied_without_transformation",
                "output_task_override": output_task,
                "episodes": provenance,
            },
        )
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "output": str(output),
        "urdf": str(urdf),
        "urdf_sha256": urdf_sha256,
        "reference_link": REFERENCE_LINK,
        "eef_links": [LEFT_EEF_LINK, RIGHT_EEF_LINK],
        "frame_convention": "T_reference_eef",
        "rotation_6d": ROTATION_6D_LAYOUT,
        "action_domain": layout.action_domain,
        "output_dimension": layout.output_dim,
        "output_tasks": [row["task"] for row in output_tasks],
        "episodes": len(selected),
        "frames": global_index,
        "linked_videos": linked_videos,
        "copied_videos": copied_videos,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument(
        "--task",
        action="append",
        help="Keep an exact task string from tasks.jsonl; repeat this option to keep multiple tasks",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--episodes", help="Episode IDs, for example 0,2,5-8")
    selection.add_argument("--max-episodes", type=int, help="Convert only the first N episodes after task filtering")
    parser.add_argument("--copy-videos", action="store_true", help="Copy videos instead of using hard links")
    parser.add_argument(
        "--output-task",
        help="Assign every converted episode to this single output task without modifying the source dataset",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and report without writing the output dataset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = _dataset_root(args.source)
    episodes = _read_jsonl(source / "meta" / "episodes.jsonl")
    available = {row["episode_index"] for row in episodes}
    candidates = set(available)
    if args.task:
        task_rows = _read_jsonl(source / "meta" / "tasks.jsonl")
        task_ids_by_name = {row["task"]: row["task_index"] for row in task_rows}
        unknown_tasks = set(args.task) - set(task_ids_by_name)
        if unknown_tasks:
            raise ValueError(
                f"Unknown task strings: {sorted(unknown_tasks)}. Available tasks: {sorted(task_ids_by_name)}"
            )
        selected_task_ids = {task_ids_by_name[name] for name in args.task}
        candidates = {
            row["episode_index"]
            for row in episodes
            if _episode_task_index(row, task_ids_by_name) in selected_task_ids
        }
    if args.episodes:
        selected = _parse_episodes(args.episodes, available)
        filtered_out = set(selected) - candidates
        if filtered_out:
            raise ValueError(f"Episode indices excluded by --task: {sorted(filtered_out)[:10]}")
    elif args.max_episodes is not None:
        if args.max_episodes <= 0:
            raise ValueError("max-episodes must be positive")
        selected = sorted(candidates)[: args.max_episodes]
    else:
        selected = sorted(candidates)
    result = convert_dataset(
        source,
        args.output,
        args.urdf,
        selected,
        copy_videos=args.copy_videos,
        dry_run=args.dry_run,
        output_task=args.output_task,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
