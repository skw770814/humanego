#!/usr/bin/env python3
"""Compute robust Unitree EEF/BrainCo or EEF/gripper statistics from Parquet.

This utility deliberately does not instantiate ``LeRobotDataset`` and therefore
does not decode videos.  It reads one episode parquet at a time, constructs the
same future action chunks used by training, and excludes every action that would
be padding beyond the end of an episode.

Supported EEF layouts are::

    left EEF xyz (3), left rotation-6D columns (6),
    right EEF xyz (3), right rotation-6D columns (6),
    left BrainCo (6), right BrainCo (6)

or the 20-D gripper domain::

    left EEF (9), right EEF (9), left gripper (1), right gripper (1)

Human EEF-only data uses the first 18 dimensions with no action tail.

Absolute joint layouts are also supported: arm14 + BrainCo12 (26D) and
arm14 + left/right gripper (16D).

Four normalization modes are supported:

``absolute``
    Absolute actions and one shared statistic across the horizon.
``shared``
    Relative EEF18 plus an optional absolute tail, with one statistic across the horizon.
``per_step``
    Relative EEF18 plus an optional absolute tail, with independent statistics at every
    horizon offset.
``hybrid``
    Relative EEF18 is normalized per horizon offset; an absolute tail uses one
    shared statistic. Without a tail, this is EEF18 per-step normalization.

The resulting ``norm_stats.json`` uses OpenPI's standard schema.  A companion
``norm_stats_manifest.json`` records the exact split, representation, estimator,
padding counts, and every low-scale range that was stabilized.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import dataclasses
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Literal

import numpy as np
import pyarrow.parquet as pq

from openpi.shared import normalize
from openpi.training.action_chunk_resampling import required_source_horizon
from openpi.training.action_chunk_resampling import resample_absolute_eef_actions

NormMode = Literal["absolute", "shared", "per_step", "hybrid"]
RotationFormat = Literal["columns_grouped", "columns_interleaved", "rows"]
InputFrame = Literal["native", "torso_palm", "g1_base_tcp", "recording_tcp", "pelvis_wrist"]

STATE_KEY = "observation.state"
ACTION_KEY = "action"
CANONICAL_DIM = 30
GRIPPER_DIM = 20
JOINT_GRIPPER_DIM = 16
EEF_DIM = 18
GRIPPER_NAMES = ("kLeftGripper", "kRightGripper")
NORMAL_1_TO_99_WIDTH = 4.6526957480816815


@dataclasses.dataclass(frozen=True)
class SemanticGroup:
    name: str
    start: int
    stop: int
    absolute_floor: float


def _semantic_groups(
    vector_dim: int,
    position_floor: float,
    rotation_floor: float,
    joint_floor: float,
    hand_floor: float,
) -> tuple[SemanticGroup, ...]:
    if vector_dim == JOINT_GRIPPER_DIM:
        return (
            SemanticGroup("left_arm_joints", 0, 7, joint_floor),
            SemanticGroup("right_arm_joints", 7, 14, joint_floor),
            SemanticGroup("left_gripper", 14, 15, hand_floor),
            SemanticGroup("right_gripper", 15, 16, hand_floor),
        )
    if vector_dim == 26:
        return (
            SemanticGroup("left_arm_joints", 0, 7, joint_floor),
            SemanticGroup("right_arm_joints", 7, 14, joint_floor),
            SemanticGroup("left_brainco", 14, 20, hand_floor),
            SemanticGroup("right_brainco", 20, 26, hand_floor),
        )
    if vector_dim in (EEF_DIM, GRIPPER_DIM):
        groups = (
            SemanticGroup("left_xyz", 0, 3, position_floor),
            SemanticGroup("left_rotation6d", 3, 9, rotation_floor),
            SemanticGroup("right_xyz", 9, 12, position_floor),
            SemanticGroup("right_rotation6d", 12, 18, rotation_floor),
        )
        if vector_dim == EEF_DIM:
            return groups
        return (
            *groups,
            SemanticGroup("left_gripper", 18, 19, hand_floor),
            SemanticGroup("right_gripper", 19, 20, hand_floor),
        )
    if vector_dim != CANONICAL_DIM:
        raise ValueError(
            f"Only EEF18-only, joint16-gripper, joint26-BrainCo, EEF20-gripper, and "
            f"EEF30-BrainCo layouts are supported, "
            f"got {vector_dim}D"
        )
    return (
        SemanticGroup("left_xyz", 0, 3, position_floor),
        SemanticGroup("left_rotation6d", 3, 9, rotation_floor),
        SemanticGroup("right_xyz", 9, 12, position_floor),
        SemanticGroup("right_rotation6d", 12, 18, rotation_floor),
        SemanticGroup("left_brainco", 18, 24, hand_floor),
        SemanticGroup("right_brainco", 24, 30, hand_floor),
    )


@dataclasses.dataclass(frozen=True)
class EpisodeRecord:
    episode_index: int
    length: int
    task_index: int | None
    parquet_path: Path


@dataclasses.dataclass(frozen=True)
class ResolvedInputs:
    config_name: str | None
    component_name: str | None
    dataset_root: Path
    output_dir: Path
    asset_id: str
    action_horizon: int
    norm_mode: NormMode
    input_rotation_format: RotationFormat
    input_frame: InputFrame
    split_episodes: Sequence[int] | None
    split_source: str
    runtime_manifest: dict[str, Any] | None
    dataset_contract: dict[str, Any] | None


class PriorityReservoir:
    """A deterministic, mergeable uniform sample based on random priorities."""

    def __init__(self, capacity: int, vector_dim: int, seed: int) -> None:
        self.capacity = capacity
        self.vector_dim = vector_dim
        self._rng = np.random.default_rng(seed)
        self._priorities = np.empty((0,), dtype=np.float64)
        self._values = np.empty((0, vector_dim), dtype=np.float32)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.vector_dim:
            raise ValueError(f"Reservoir expected [N, {self.vector_dim}], got {values.shape}")
        if len(values) == 0:
            return
        priorities = self._rng.random(len(values))
        # Once full, only priorities below the current worst retained priority
        # can enter the reservoir.  Filtering first avoids copying/partitioning
        # the full reservoir for every episode and horizon offset.
        if len(self._values) == self.capacity:
            candidates = priorities < np.max(self._priorities)
            if not np.any(candidates):
                return
            priorities = priorities[candidates]
            values = values[candidates]
        priorities = np.concatenate((self._priorities, priorities))
        values = np.concatenate((self._values, values), axis=0)
        if len(values) > self.capacity:
            keep = np.argpartition(priorities, self.capacity - 1)[: self.capacity]
            priorities = priorities[keep]
            values = values[keep]
        self._priorities = priorities
        self._values = values

    @property
    def size(self) -> int:
        return len(self._values)

    def quantiles(self) -> tuple[np.ndarray, np.ndarray]:
        if self.size < 2:
            raise ValueError(f"At least two samples are required for quantiles, got {self.size}")
        return (
            np.quantile(self._values, 0.01, axis=0).astype(np.float64),
            np.quantile(self._values, 0.99, axis=0).astype(np.float64),
        )


class VectorAccumulator:
    """Float64 Chan/Welford moments plus a bounded quantile reservoir."""

    def __init__(self, vector_dim: int, reservoir_size: int, seed: int) -> None:
        self.vector_dim = vector_dim
        self.count = 0
        self.mean = np.zeros(vector_dim, dtype=np.float64)
        self.m2 = np.zeros(vector_dim, dtype=np.float64)
        self.minimum = np.full(vector_dim, np.inf, dtype=np.float64)
        self.maximum = np.full(vector_dim, -np.inf, dtype=np.float64)
        self.reservoir = PriorityReservoir(reservoir_size, vector_dim, seed)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values)
        if values.ndim != 2 or values.shape[1] != self.vector_dim:
            raise ValueError(f"Accumulator expected [N, {self.vector_dim}], got {values.shape}")
        if len(values) == 0:
            return
        if not np.isfinite(values).all():
            location = np.argwhere(~np.isfinite(values))[0].tolist()
            raise ValueError(f"Non-finite value at local row/dimension {location}")

        batch = values.astype(np.float64, copy=False)
        batch_count = len(batch)
        batch_mean = np.mean(batch, axis=0, dtype=np.float64)
        centered = batch - batch_mean
        batch_m2 = np.sum(centered * centered, axis=0, dtype=np.float64)

        if self.count == 0:
            self.mean = batch_mean
            self.m2 = batch_m2
        else:
            total = self.count + batch_count
            delta = batch_mean - self.mean
            self.mean += delta * (batch_count / total)
            self.m2 += batch_m2 + delta * delta * (self.count * batch_count / total)
        self.count += batch_count
        self.minimum = np.minimum(self.minimum, np.min(batch, axis=0))
        self.maximum = np.maximum(self.maximum, np.max(batch, axis=0))
        self.reservoir.update(values)

    def finalize(self) -> tuple[normalize.NormStats, np.ndarray, np.ndarray, int]:
        if self.count < 2:
            raise ValueError(f"At least two values are required, got {self.count}")
        q01, q99 = self.reservoir.quantiles()
        std = np.sqrt(np.maximum(self.m2 / self.count, 0.0))
        stats = normalize.NormStats(mean=self.mean.copy(), std=std, q01=q01, q99=q99)
        return stats, self.minimum.copy(), self.maximum.copy(), self.reservoir.size


class ActionAccumulator:
    def __init__(self, mode: NormMode, horizon: int, vector_dim: int, reservoir_size: int, seed: int) -> None:
        self.mode = mode
        self.horizon = horizon
        self.vector_dim = vector_dim
        self.tail_dim = vector_dim - EEF_DIM
        self.valid_counts = np.zeros(horizon, dtype=np.int64)
        if mode in ("absolute", "shared"):
            if mode == "shared" and vector_dim not in (EEF_DIM, GRIPPER_DIM, CANONICAL_DIM):
                raise ValueError("Relative shared normalization requires EEF18, EEF20, or EEF30 input")
            self.shared = VectorAccumulator(vector_dim, reservoir_size, seed)
            self.steps: list[VectorAccumulator] = []
            self.hands = None
        elif mode == "per_step":
            self.shared = None
            self.steps = [
                VectorAccumulator(vector_dim, reservoir_size, seed + 10_000 + step) for step in range(horizon)
            ]
            self.hands = None
        elif mode == "hybrid":
            self.shared = None
            self.steps = [VectorAccumulator(EEF_DIM, reservoir_size, seed + 10_000 + step) for step in range(horizon)]
            # EEF-only has no absolute tail, so hybrid is exactly EEF per-step.
            self.hands = None if self.tail_dim == 0 else VectorAccumulator(self.tail_dim, reservoir_size, seed + 20_000)
        else:
            raise ValueError(f"Unsupported norm mode: {mode}")

    def update(self, chunks: np.ndarray, valid: np.ndarray) -> None:
        if chunks.ndim != 3 or chunks.shape[1:] != (self.horizon, self.vector_dim):
            raise ValueError(f"Expected chunks [N, {self.horizon}, {self.vector_dim}], got {chunks.shape}")
        if valid.shape != chunks.shape[:2]:
            raise ValueError(f"Expected valid mask {chunks.shape[:2]}, got {valid.shape}")
        self.valid_counts += np.sum(valid, axis=0, dtype=np.int64)

        if self.shared is not None:
            self.shared.update(chunks[valid])
            return

        for step, accumulator in enumerate(self.steps):
            values = chunks[valid[:, step], step]
            if self.mode == "hybrid":
                values = values[:, :EEF_DIM]
            accumulator.update(values)
        if self.hands is not None:
            self.hands.update(chunks[..., EEF_DIM:][valid])

    def finalize(self) -> tuple[normalize.NormStats, np.ndarray, np.ndarray, dict[str, Any]]:
        if self.shared is not None:
            stats, minimum, maximum, sample_size = self.shared.finalize()
            return stats, minimum, maximum, {"shared_reservoir_samples": sample_size}

        missing_steps = np.flatnonzero(self.valid_counts < 2)
        if len(missing_steps):
            preview = missing_steps[:10].tolist()
            raise ValueError(
                "Per-step normalization requires at least two valid action samples at every horizon index; "
                f"insufficient indices={preview}{'...' if len(missing_steps) > len(preview) else ''}. "
                "Use longer episodes, more episodes, or shared normalization."
            )
        step_results = [accumulator.finalize() for accumulator in self.steps]
        step_stats = [result[0] for result in step_results]
        minimum = np.stack([result[1] for result in step_results], axis=0)
        maximum = np.stack([result[2] for result in step_results], axis=0)
        sample_sizes = [result[3] for result in step_results]

        if self.hands is None:
            stats = normalize.NormStats(
                mean=np.stack([item.mean for item in step_stats], axis=0),
                std=np.stack([item.std for item in step_stats], axis=0),
                q01=np.stack([item.q01 for item in step_stats], axis=0),
                q99=np.stack([item.q99 for item in step_stats], axis=0),
            )
            details = {"per_step_reservoir_samples": sample_sizes}
            if self.mode == "hybrid" and self.tail_dim == 0:
                details["hybrid_eef_only"] = True
            return stats, minimum, maximum, details

        hand_stats, hand_minimum, hand_maximum, hand_sample_size = self.hands.finalize()
        hand_mean = np.broadcast_to(hand_stats.mean, (self.horizon, self.tail_dim))
        hand_std = np.broadcast_to(hand_stats.std, (self.horizon, self.tail_dim))
        hand_q01 = np.broadcast_to(hand_stats.q01, (self.horizon, self.tail_dim))
        hand_q99 = np.broadcast_to(hand_stats.q99, (self.horizon, self.tail_dim))
        stats = normalize.NormStats(
            mean=np.concatenate((np.stack([item.mean for item in step_stats]), hand_mean), axis=-1),
            std=np.concatenate((np.stack([item.std for item in step_stats]), hand_std), axis=-1),
            q01=np.concatenate((np.stack([item.q01 for item in step_stats]), hand_q01), axis=-1),
            q99=np.concatenate((np.stack([item.q99 for item in step_stats]), hand_q99), axis=-1),
        )
        minimum = np.concatenate((minimum, np.broadcast_to(hand_minimum, (self.horizon, self.tail_dim))), axis=-1)
        maximum = np.concatenate((maximum, np.broadcast_to(hand_maximum, (self.horizon, self.tail_dim))), axis=-1)
        tail_name = "gripper" if self.tail_dim == 2 else "brainco"
        details = {
            "eef_per_step_reservoir_samples": sample_sizes,
            "tail_shared_reservoir_samples": hand_sample_size,
            "tail_dimension": self.tail_dim,
            f"{tail_name}_shared_reservoir_samples": hand_sample_size,
        }
        return stats, minimum, maximum, details


def _rotation6d_to_matrix(value: np.ndarray, rotation_format: RotationFormat) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.shape[-1] != 6:
        raise ValueError(f"Rotation-6D must have six values, got {value.shape}")
    if rotation_format == "columns_interleaved":
        vectors = value.reshape(*value.shape[:-1], 3, 2)
        first = vectors[..., :, 0]
        second_raw = vectors[..., :, 1]
        stack_axis = -1
    elif rotation_format == "columns_grouped":
        vectors = value.reshape(*value.shape[:-1], 2, 3)
        first = vectors[..., 0, :]
        second_raw = vectors[..., 1, :]
        stack_axis = -1
    elif rotation_format == "rows":
        vectors = value.reshape(*value.shape[:-1], 2, 3)
        first = vectors[..., 0, :]
        second_raw = vectors[..., 1, :]
        stack_axis = -2
    else:
        raise ValueError(f"Unsupported rotation format: {rotation_format}")

    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    first = first / np.maximum(first_norm, 1e-12)
    second = second_raw - np.sum(first * second_raw, axis=-1, keepdims=True) * first
    second_norm = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(first_norm < 1e-8) or np.any(second_norm < 1e-8):
        raise ValueError("Degenerate rotation-6D vector cannot be orthonormalized")
    second = second / np.maximum(second_norm, 1e-12)
    return np.stack((first, second, np.cross(first, second)), axis=stack_axis)


def _matrix_to_rotation6d_columns_grouped(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation)
    return np.concatenate((rotation[..., :, 0], rotation[..., :, 1]), axis=-1)


def _rotation_format_score(values: np.ndarray, rotation_format: RotationFormat) -> float:
    values = np.asarray(values, dtype=np.float64)
    if rotation_format == "columns_interleaved":
        vectors = values.reshape(-1, 3, 2)
        first, second = vectors[..., :, 0], vectors[..., :, 1]
    else:
        vectors = values.reshape(-1, 2, 3)
        first, second = vectors[..., 0, :], vectors[..., 1, :]
    error = np.abs(np.linalg.norm(first, axis=-1) - 1.0)
    error += np.abs(np.linalg.norm(second, axis=-1) - 1.0)
    error += np.abs(np.sum(first * second, axis=-1))
    return float(np.median(error))


def _validate_rotation_format(values: np.ndarray, selected: RotationFormat, label: str) -> dict[str, float]:
    rotations = np.concatenate((values[:, 3:9], values[:, 12:18]), axis=0)
    if len(rotations) > 4096:
        rotations = rotations[:4096]
    scores = {
        "columns_grouped": _rotation_format_score(rotations, "columns_grouped"),
        "columns_interleaved": _rotation_format_score(rotations, "columns_interleaved"),
        "rows": _rotation_format_score(rotations, "rows"),
    }
    best = min(scores, key=scores.get)
    if scores[selected] > 0.05 and scores[best] < scores[selected] * 0.25:
        raise ValueError(
            f"{label} looks like rotation_format={best!r}, not {selected!r}: scores={scores}. "
            f"Pass --input-rotation-format {best}."
        )
    return scores


def _canonicalize(
    values: np.ndarray,
    input_rotation_format: RotationFormat,
    input_frame: InputFrame,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] not in (
        EEF_DIM,
        JOINT_GRIPPER_DIM,
        GRIPPER_DIM,
        26,
        CANONICAL_DIM,
    ):
        raise ValueError(
            f"Expected EEF18-only, joint16-gripper, joint26-BrainCo, EEF20-gripper, "
            f"or EEF30-BrainCo values, got {values.shape}"
        )
    if values.shape[1] in (JOINT_GRIPPER_DIM, 26):
        if input_frame not in ("native", "torso_palm"):
            raise ValueError("Human EEF frame conversion is only defined for EEF input")
        return values.copy()

    output = values.copy()
    torso_in_pelvis = np.asarray([-0.0039635, 0.0, 0.044], dtype=np.float64)
    wrist_to_palm = (
        np.asarray([0.0415, 0.003, 0.0], dtype=np.float64),
        np.asarray([0.0415, -0.003, 0.0], dtype=np.float64),
    )
    for pose_index, pose_start in enumerate((0, 9)):
        rotation = _rotation6d_to_matrix(values[:, pose_start + 3 : pose_start + 9], input_rotation_format)
        if input_frame in ("g1_base_tcp", "pelvis_wrist"):
            output[:, pose_start : pose_start + 3] = (
                values[:, pose_start : pose_start + 3]
                - torso_in_pelvis
                + np.einsum("...ij,j->...i", rotation, wrist_to_palm[pose_index])
            ).astype(np.float32)
        elif input_frame == "recording_tcp":
            output[:, pose_start : pose_start + 3] = (
                values[:, pose_start : pose_start + 3]
                + np.einsum("...ij,j->...i", rotation, wrist_to_palm[pose_index])
            ).astype(np.float32)
        elif input_frame not in ("native", "torso_palm"):
            raise ValueError(f"Unsupported Human input frame: {input_frame!r}")
        output[:, pose_start + 3 : pose_start + 9] = _matrix_to_rotation6d_columns_grouped(rotation).astype(np.float32)
    return output


def _select_eef_only(values: np.ndarray, label: str) -> np.ndarray:
    """Return the EEF18 prefix and reject any source other than EEF18/EEF20."""
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[-1] not in (EEF_DIM, GRIPPER_DIM):
        raise ValueError(f"Expected {label} EEF18 or EEF20-gripper values, got {values.shape}")
    return values[..., :EEF_DIM]


def _relative_pose(action_pose: np.ndarray, state_pose: np.ndarray) -> np.ndarray:
    state_rotation = _rotation6d_to_matrix(state_pose[..., 3:9], "columns_grouped")
    action_rotation = _rotation6d_to_matrix(action_pose[..., 3:9], "columns_grouped")
    translation = np.einsum("...ji,...j->...i", state_rotation, action_pose[..., :3] - state_pose[..., :3])
    rotation = np.einsum("...ji,...jk->...ik", state_rotation, action_rotation)
    return np.concatenate((translation, _matrix_to_rotation6d_columns_grouped(rotation)), axis=-1).astype(np.float32)


def _make_action_chunks(
    states: np.ndarray,
    actions: np.ndarray,
    anchor_count: int,
    horizon: int,
    mode: NormMode,
    source_step_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    episode_length = len(actions)
    source_horizon = required_source_horizon(horizon, source_step_scale)
    indices = np.arange(anchor_count)[:, None] + np.arange(source_horizon)[None, :]
    source_valid = indices < episode_length
    safe_indices = np.minimum(indices, episode_length - 1)
    chunks = actions[safe_indices]
    if source_step_scale != 1.0:
        chunks, output_is_pad = resample_absolute_eef_actions(
            chunks,
            ~source_valid,
            output_horizon=horizon,
            source_step_scale=source_step_scale,
        )
        valid = ~output_is_pad
    else:
        valid = source_valid
    if mode == "absolute":
        return chunks, valid

    if states.shape[-1] not in (EEF_DIM, GRIPPER_DIM, CANONICAL_DIM):
        raise ValueError("Relative EEF actions require the EEF18, EEF20, or EEF30 layout")

    reference = states[:anchor_count, None, :]
    relative = np.empty_like(chunks)
    relative[..., :9] = _relative_pose(chunks[..., :9], reference[..., :9])
    relative[..., 9:18] = _relative_pose(chunks[..., 9:18], reference[..., 9:18])
    relative[..., 18:] = chunks[..., 18:]
    return relative, valid


def _as_2d(column: Any, key: str) -> np.ndarray:
    values = column.to_pylist()
    if not values:
        return np.empty((0, 0), dtype=np.float32)
    try:
        result = np.asarray(values, dtype=np.float32)
    except ValueError:
        result = np.stack([np.asarray(value, dtype=np.float32).reshape(-1) for value in values])
    if result.ndim != 2:
        raise ValueError(f"Parquet column {key!r} must be a vector column, got {result.shape}")
    return result


def _resolve_dataset_root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    if (root / "meta" / "info.json").is_file():
        return root
    candidates = sorted(info.parent.parent for info in root.glob("*/meta/info.json")) if root.is_dir() else []
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(
        f"Expected a LeRobot root at {root} or exactly one child root; found {len(candidates)} candidates"
    )


def _episode_records(root: Path) -> tuple[dict[str, Any], list[EpisodeRecord]]:
    info_path = root / "meta" / "info.json"
    episodes_path = root / "meta" / "episodes.jsonl"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if not episodes_path.is_file():
        raise FileNotFoundError(f"Episode metadata not found: {episodes_path}")
    data_pattern = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    chunks_size = int(info.get("chunks_size", 1000))
    records: list[EpisodeRecord] = []
    for line in episodes_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        episode_index = int(item["episode_index"])
        relative_path = data_pattern.format(
            episode_chunk=episode_index // chunks_size,
            episode_index=episode_index,
        )
        parquet_path = root / relative_path
        if not parquet_path.is_file():
            raise FileNotFoundError(f"Episode parquet not found: {parquet_path}")
        task_index = item.get("task_index")
        records.append(
            EpisodeRecord(
                episode_index=episode_index,
                length=int(item["length"]),
                task_index=None if task_index is None else int(task_index),
                parquet_path=parquet_path,
            )
        )
    if not records:
        raise ValueError(f"No episodes listed in {episodes_path}")
    return info, records


def _parse_episode_expression(expression: str | None) -> list[int] | None:
    if expression is None:
        return None
    result: list[int] = []
    for raw_part in expression.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            first, last = part.split("-", 1)
            start, stop = int(first), int(last)
            if stop < start:
                raise ValueError(f"Invalid descending episode range: {part}")
            result.extend(range(start, stop + 1))
        else:
            result.append(int(part))
    if not result:
        raise ValueError("--episodes did not contain any episode indices")
    return list(dict.fromkeys(result))


def _episodes_from_file(path: str | None, split: str) -> list[int] | None:
    if path is None:
        return None
    source = Path(path).expanduser()
    text = source.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, list):
        return [int(item["episode_index"] if isinstance(item, dict) else item) for item in value]
    if isinstance(value, dict):
        for key in (split, f"{split}_episodes", "episode_indices", "episodes"):
            if key in value:
                items = value[key]
                return [int(item["episode_index"] if isinstance(item, dict) else item) for item in items]

    result = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            result.append(int(line.strip()))
        else:
            result.append(int(item["episode_index"] if isinstance(item, dict) else item))
    if not result:
        raise ValueError(f"Could not read episode indices from {source}")
    return result


def _choose_component(data_config: Any, component: str | None) -> tuple[Any, str | None]:
    components = tuple(getattr(data_config, "mixture_components", ()) or ())
    if not components:
        if component is not None:
            raise ValueError("--component was provided, but this config is not a mixture")
        return data_config, None
    names = tuple(getattr(data_config, "mixture_names", ()) or ())
    if not names:
        names = tuple(str(index) for index in range(len(components)))
    if component is None:
        raise ValueError(f"Mixture config requires --component; available components: {', '.join(names)}")
    if component in names:
        index = names.index(component)
    else:
        try:
            index = int(component)
        except ValueError as exc:
            raise ValueError(f"Unknown mixture component {component!r}; available: {names}") from exc
        if not 0 <= index < len(components):
            raise ValueError(f"Mixture component index {index} is outside [0, {len(components)})")
    return components[index], names[index]


def _resolve_inputs(args: argparse.Namespace) -> ResolvedInputs:
    config = None
    data_config = None
    component_name = None
    metadata: dict[str, Any] = {}
    if args.config_name is not None:
        from openpi.training import config as training_config

        config = training_config.get_config(args.config_name)
        data_config = config.data.create(config.assets_dirs, config.model)
        data_config, component_name = _choose_component(data_config, args.component)
        metadata.update(config.policy_metadata or {})
        metadata.update(getattr(data_config, "runtime_manifest", None) or {})

    dataset_value = args.dataset or (None if data_config is None else data_config.repo_id)
    if dataset_value is None:
        raise ValueError("Pass --dataset, or use --config-name with a concrete LeRobot repo_id")
    root = _resolve_dataset_root(dataset_value)

    asset_id = args.asset_id or (None if data_config is None else data_config.asset_id)
    if not asset_id:
        asset_id = f"unitree_{args.norm_mode or 'norm'}"
    asset_path = Path(str(asset_id))
    if asset_path.is_absolute() and args.output_dir is None:
        raise ValueError("Config asset_id is absolute; pass --output-dir and use a stable non-path --asset-id")

    norm_mode: NormMode
    if args.norm_mode is not None:
        norm_mode = args.norm_mode
    elif metadata.get("action_representation") == "relative" and metadata.get("relative_norm") in {
        "shared",
        "per_step",
        "hybrid",
    }:
        norm_mode = metadata["relative_norm"]
    elif metadata.get("relative_norm_mode") in {"shared", "per_step", "hybrid"}:
        norm_mode = metadata["relative_norm_mode"]
    elif bool(metadata.get("relative", False)):
        norm_mode = "shared"
    else:
        norm_mode = "absolute"

    if args.action_horizon is not None:
        horizon = args.action_horizon
    elif config is not None:
        horizon = int(config.model.action_horizon)
    else:
        horizon = 50

    rotation_format = args.input_rotation_format
    if rotation_format is None:
        rotation_format = metadata.get("input_rotation_format", metadata.get("rotation_format"))
        if rotation_format is None and metadata.get("rotation_6d") == "first_column_then_second_column":
            rotation_format = "columns_grouped"
        rotation_format = rotation_format or "columns_grouped"
    if rotation_format == "columns":
        # Legacy Unitree configs use this name for R[:, :2].reshape(6), i.e.
        # interleaved columns. Keep the alias local to config resolution.
        rotation_format = "columns_interleaved"
    if rotation_format not in ("columns_grouped", "columns_interleaved", "rows"):
        raise ValueError(f"Invalid input rotation format from config/CLI: {rotation_format!r}")

    input_frame = args.input_frame
    if input_frame is None:
        policy_kind = metadata.get("kind", metadata.get("policy_kind"))
        input_frame = metadata.get("input_frame")
        if input_frame is None and policy_kind == "human":
            raise ValueError(
                "Human normalization requires an explicit source input_frame. "
                "Pass --input-frame=native to preserve an already-retargeted Ego dataset, "
                "or an explicit cross-domain conversion profile."
            )
        input_frame = input_frame or "native"

    if args.output_dir is not None:
        output_dir = Path(args.output_dir).expanduser().resolve()
    elif config is not None:
        output_dir = (Path(config.assets_dirs) / str(asset_id)).resolve()
    else:
        output_dir = root / "meta" / "openpi_assets" / str(asset_id)

    split_episodes = None
    split_source = "all_episodes"
    if data_config is not None and args.split != "all":
        field = "train_episodes" if args.split == "train" else "validation_episodes"
        value = getattr(data_config, field, None)
        if value is not None:
            split_episodes = [int(index) for index in value]
            split_source = f"config.{field}"

    runtime_manifest = None if data_config is None else getattr(data_config, "runtime_manifest", None)
    dataset_contract = None
    if args.dataset_contract_json is not None:
        dataset_contract = json.loads(args.dataset_contract_json)
        if not isinstance(dataset_contract, dict):
            raise ValueError("--dataset-contract-json must decode to a JSON object")
    return ResolvedInputs(
        config_name=args.config_name,
        component_name=component_name,
        dataset_root=root,
        output_dir=output_dir,
        asset_id=str(asset_id),
        action_horizon=horizon,
        norm_mode=norm_mode,
        input_rotation_format=rotation_format,
        input_frame=input_frame,
        split_episodes=split_episodes,
        split_source=split_source,
        runtime_manifest=runtime_manifest,
        dataset_contract=dataset_contract,
    )


def _filter_records(
    records: Sequence[EpisodeRecord],
    requested: Sequence[int] | None,
    task_index: int | None,
    max_episodes: int | None,
) -> list[EpisodeRecord]:
    by_index = {record.episode_index: record for record in records}
    if requested is None:
        selected = list(records)
    else:
        missing = sorted(set(requested) - by_index.keys())
        if missing:
            raise ValueError(f"Requested episodes are not in the dataset: {missing[:20]}")
        selected = [by_index[index] for index in requested]

    if task_index is not None:
        filtered = []
        for record in selected:
            episode_task = record.task_index
            if episode_task is None:
                table = pq.read_table(record.parquet_path, columns=["task_index"])
                task_value = table.column("task_index")[0].as_py()
                if isinstance(task_value, list):
                    task_value = task_value[0]
                episode_task = int(task_value)
            if episode_task == task_index:
                filtered.append(record)
        selected = filtered
    if max_episodes is not None:
        selected = selected[:max_episodes]
    if not selected:
        raise ValueError("The episode selection is empty")
    return selected


def _stabilize_stats(
    stats: normalize.NormStats,
    minimum: np.ndarray,
    maximum: np.ndarray,
    groups: Sequence[SemanticGroup],
    group_floor_ratio: float,
    scope: str,
) -> tuple[normalize.NormStats, list[dict[str, Any]]]:
    mean = np.asarray(stats.mean, dtype=np.float64).copy()
    std = np.asarray(stats.std, dtype=np.float64).copy()
    q01 = np.asarray(stats.q01, dtype=np.float64).copy()
    q99 = np.asarray(stats.q99, dtype=np.float64).copy()
    minimum = np.asarray(minimum, dtype=np.float64)
    maximum = np.asarray(maximum, dtype=np.float64)
    vector_dim = groups[-1].stop
    if mean.shape[-1] != vector_dim:
        raise ValueError(f"Expected final statistic dimension {vector_dim}, got {mean.shape}")

    leading_shape = mean.shape[:-1]
    flat_mean = mean.reshape(-1, vector_dim)
    flat_std = std.reshape(-1, vector_dim)
    flat_q01 = q01.reshape(-1, vector_dim)
    flat_q99 = q99.reshape(-1, vector_dim)
    flat_minimum = minimum.reshape(-1, vector_dim)
    flat_maximum = maximum.reshape(-1, vector_dim)
    report: list[dict[str, Any]] = []

    for leading_index in range(len(flat_mean)):
        for group in groups:
            slc = slice(group.start, group.stop)
            raw_ranges = flat_q99[leading_index, slc] - flat_q01[leading_index, slc]
            positive = raw_ranges[raw_ranges > np.finfo(np.float64).eps]
            group_reference = float(np.median(positive)) if len(positive) else 0.0
            floor = max(group.absolute_floor, group_floor_ratio * group_reference)
            low = raw_ranges < floor

            flat_std[leading_index, slc] = np.maximum(flat_std[leading_index, slc], floor / NORMAL_1_TO_99_WIDTH)
            if not np.any(low):
                continue

            local_q01 = flat_q01[leading_index, slc]
            local_q99 = flat_q99[leading_index, slc]
            center = (local_q01 + local_q99) / 2.0
            local_q01[low] = center[low] - floor / 2.0
            local_q99[low] = center[low] + floor / 2.0
            flat_q01[leading_index, slc] = local_q01
            flat_q99[leading_index, slc] = local_q99

            local_dims = np.flatnonzero(low)
            absolute_dims = (local_dims + group.start).tolist()
            support_ranges = flat_maximum[leading_index, slc][low] - flat_minimum[leading_index, slc][low]
            sparse_tail_dims = [
                absolute_dims[index] for index, support_range in enumerate(support_ranges) if support_range > floor
            ]
            item: dict[str, Any] = {
                "scope": scope,
                "group": group.name,
                "floor": floor,
                "dimensions": absolute_dims,
                "raw_q01_q99_ranges": raw_ranges[low].tolist(),
                "observed_min_max_ranges": support_ranges.tolist(),
                "sparse_tail_dimensions": sparse_tail_dims,
            }
            if leading_shape:
                item["horizon_index"] = int(np.unravel_index(leading_index, leading_shape)[0])
            report.append(item)

    stabilized = normalize.NormStats(
        mean=flat_mean.reshape(mean.shape),
        std=flat_std.reshape(std.shape),
        q01=flat_q01.reshape(q01.shape),
        q99=flat_q99.reshape(q99.shape),
    )
    return stabilized, report


def _selection_hash(dataset_root: Path, episode_indices: Sequence[int], args: argparse.Namespace) -> str:
    digest = hashlib.sha256()
    digest.update((dataset_root / "meta" / "info.json").read_bytes())
    digest.update(json.dumps(list(episode_indices), separators=(",", ":")).encode())
    digest.update(
        json.dumps(
            {
                "task_index": args.task_index,
                "max_episodes": args.max_episodes,
                "max_frames": args.max_frames,
                "select_eef_only": args.select_eef_only,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    return digest.hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-name", help="OpenPI config used to resolve dataset, horizon, split, and output asset")
    parser.add_argument("--component", help="Mixture component name or zero-based index")
    parser.add_argument("--dataset", help="LeRobot dataset root; overrides config repo_id")
    parser.add_argument("--output-dir", help="Directory in which norm_stats.json and its manifest are written")
    parser.add_argument("--asset-id", help="Asset identifier recorded in the manifest")
    parser.add_argument(
        "--dataset-contract-json",
        help="Resolved Human source contract recorded verbatim in the normalization manifest",
    )
    parser.add_argument(
        "--norm-mode",
        choices=("absolute", "shared", "per_step", "hybrid"),
        help="Absolute actions, or one of the three relative EEF normalization modes",
    )
    parser.add_argument("--action-horizon", type=int, help="Action chunk length; defaults to config or 50")
    parser.add_argument(
        "--action-source-step-scale",
        type=float,
        default=1.0,
        help="Fractional source-frame spacing for task-progress resampling; 1 preserves the original chunk",
    )
    parser.add_argument(
        "--select-eef-only",
        action="store_true",
        help="Keep EEF[0:18] from a Human EEF18 or EEF20-gripper source before computing statistics",
    )
    parser.add_argument(
        "--input-rotation-format",
        choices=("columns_grouped", "columns_interleaved", "rows"),
        help="Rot6D layout in parquet; canonical output is grouped first-column then second-column",
    )
    parser.add_argument(
        "--input-frame",
        choices=("native", "torso_palm", "g1_base_tcp", "recording_tcp", "pelvis_wrist"),
        help="Human pose transform profile; native preserves parquet positions",
    )
    parser.add_argument("--split", choices=("all", "train", "validation"), default="train")
    parser.add_argument("--episodes", help="Comma-separated episode IDs/ranges, e.g. 0-9,20,22")
    parser.add_argument("--episodes-file", help="JSON/JSONL/text file containing episode indices")
    parser.add_argument("--task-index", type=int, help="Only include episodes with this LeRobot task index")
    parser.add_argument("--max-episodes", type=int, help="Process only the first N selected episodes (smoke tests)")
    parser.add_argument("--max-frames", type=int, help="Maximum number of anchor frames across selected episodes")
    parser.add_argument("--reservoir-size", type=int, default=8192, help="Per-statistic quantile reservoir size")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic quantile reservoir seed")
    parser.add_argument("--position-floor", type=float, default=1e-3, help="Minimum q01-q99 range for XYZ")
    parser.add_argument("--rotation-floor", type=float, default=1e-2, help="Minimum q01-q99 range for rotation-6D")
    parser.add_argument("--joint-floor", type=float, default=1e-2, help="Minimum q01-q99 range for arm joints")
    parser.add_argument(
        "--hand-floor", type=float, default=1e-2, help="Minimum q01-q99 range for BrainCo or gripper values"
    )
    parser.add_argument(
        "--group-floor-ratio",
        type=float,
        default=1e-2,
        help="Also floor each dimension at this fraction of its semantic group's median active range",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.action_horizon is not None and args.action_horizon <= 0:
        raise ValueError("--action-horizon must be positive")
    if not math.isfinite(args.action_source_step_scale) or args.action_source_step_scale <= 0:
        raise ValueError("--action-source-step-scale must be finite and positive")
    if args.max_episodes is not None and args.max_episodes <= 0:
        raise ValueError("--max-episodes must be positive")
    if args.max_frames is not None and args.max_frames <= 1:
        raise ValueError("--max-frames must be at least 2")
    if args.reservoir_size < 128:
        raise ValueError("--reservoir-size must be at least 128")
    for name in ("position_floor", "rotation_floor", "joint_floor", "hand_floor", "group_floor_ratio"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    if args.episodes is not None and args.episodes_file is not None:
        raise ValueError("Use only one of --episodes and --episodes-file")


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_args(args)
    resolved = _resolve_inputs(args)
    if resolved.action_horizon <= 0:
        raise ValueError("Resolved action horizon must be positive")

    info, all_records = _episode_records(resolved.dataset_root)
    explicit_episodes = _parse_episode_expression(args.episodes)
    if explicit_episodes is None:
        explicit_episodes = _episodes_from_file(args.episodes_file, args.split)
    if explicit_episodes is not None:
        requested = explicit_episodes
        split_source = "explicit_cli_or_file"
    else:
        requested = resolved.split_episodes
        split_source = resolved.split_source
    records = _filter_records(all_records, requested, args.task_index, args.max_episodes)

    features = info.get("features", {})
    state_shape = features.get(STATE_KEY, {}).get("shape")
    action_shape = features.get(ACTION_KEY, {}).get("shape")
    source_vector_dim = int(state_shape[-1]) if state_shape else CANONICAL_DIM
    action_source_dim = int(action_shape[-1]) if action_shape else source_vector_dim
    if action_source_dim != source_vector_dim:
        raise ValueError(f"State/action metadata dimensions must match, got state={state_shape}, action={action_shape}")
    if args.select_eef_only:
        if source_vector_dim not in (EEF_DIM, GRIPPER_DIM):
            raise ValueError(
                f"--select-eef-only requires Human EEF18 or EEF20-gripper source data, got state shape {state_shape}"
            )
        for key in (STATE_KEY, ACTION_KEY):
            names = features.get(key, {}).get("names")
            flat_names = (
                names[0] if isinstance(names, list) and len(names) == 1 and isinstance(names[0], list) else None
            )
            expected_tail = [] if source_vector_dim == EEF_DIM else list(GRIPPER_NAMES)
            if flat_names is None or len(flat_names) != source_vector_dim or flat_names[EEF_DIM:] != expected_tail:
                raise ValueError(
                    f"--select-eef-only requires {key}=EEF18 or EEF18+{list(GRIPPER_NAMES)}; got names={names}"
                )
        vector_dim = EEF_DIM
    else:
        vector_dim = source_vector_dim
    if vector_dim not in (EEF_DIM, JOINT_GRIPPER_DIM, GRIPPER_DIM, 26, CANONICAL_DIM):
        raise ValueError(
            "Only EEF18-only, joint16-gripper, joint26-BrainCo, EEF20-gripper, and "
            "EEF30-BrainCo datasets are supported, "
            f"got state shape {state_shape}"
        )
    if resolved.norm_mode != "absolute" and vector_dim not in (EEF_DIM, GRIPPER_DIM, CANONICAL_DIM):
        raise ValueError("Relative normalization requires EEF18, EEF20, or EEF30 data")
    groups = _semantic_groups(
        vector_dim,
        args.position_floor,
        args.rotation_floor,
        args.joint_floor,
        args.hand_floor,
    )
    state_accumulator = VectorAccumulator(vector_dim, args.reservoir_size, args.seed + 1)
    action_accumulator = ActionAccumulator(
        resolved.norm_mode, resolved.action_horizon, vector_dim, args.reservoir_size, args.seed + 2
    )

    processed_episodes: list[int] = []
    processed_frames = 0
    rotation_scores: dict[str, dict[str, float]] | None = None
    for record_index, record in enumerate(records):
        remaining = None if args.max_frames is None else args.max_frames - processed_frames
        if remaining is not None and remaining <= 0:
            break
        table = pq.read_table(record.parquet_path, columns=[STATE_KEY, ACTION_KEY])
        states_raw = _as_2d(table.column(STATE_KEY), STATE_KEY)
        actions_raw = _as_2d(table.column(ACTION_KEY), ACTION_KEY)
        if len(states_raw) != len(actions_raw):
            raise ValueError(
                f"State/action length mismatch in {record.parquet_path}: {len(states_raw)} != {len(actions_raw)}"
            )
        if len(states_raw) != record.length:
            raise ValueError(
                f"Metadata/parquet length mismatch for episode {record.episode_index}: "
                f"{record.length} != {len(states_raw)}"
            )
        if states_raw.shape[-1] != source_vector_dim or actions_raw.shape[-1] != source_vector_dim:
            raise ValueError(
                f"Metadata/parquet vector dimension mismatch in {record.parquet_path}: "
                f"metadata={source_vector_dim}, state={states_raw.shape[-1]}, action={actions_raw.shape[-1]}"
            )
        if args.select_eef_only:
            # EEF-only is the supervised domain. A recorded gripper tail is
            # removed before canonicalization, relative conversion, progress
            # resampling, and every normalization accumulator.
            states_raw = _select_eef_only(states_raw, STATE_KEY)
            actions_raw = _select_eef_only(actions_raw, ACTION_KEY)
        if rotation_scores is None and vector_dim in (EEF_DIM, GRIPPER_DIM, CANONICAL_DIM):
            rotation_scores = {
                "state": _validate_rotation_format(
                    states_raw, resolved.input_rotation_format, f"{record.parquet_path}:{STATE_KEY}"
                ),
                "action": _validate_rotation_format(
                    actions_raw, resolved.input_rotation_format, f"{record.parquet_path}:{ACTION_KEY}"
                ),
            }

        states = _canonicalize(states_raw, resolved.input_rotation_format, resolved.input_frame)
        actions = _canonicalize(actions_raw, resolved.input_rotation_format, resolved.input_frame)
        anchor_count = len(states) if remaining is None else min(len(states), remaining)
        state_accumulator.update(states[:anchor_count])
        chunks, valid = _make_action_chunks(
            states,
            actions,
            anchor_count,
            resolved.action_horizon,
            resolved.norm_mode,
            args.action_source_step_scale,
        )
        try:
            action_accumulator.update(chunks, valid)
        except ValueError as exc:
            raise ValueError(f"Failed while processing {record.parquet_path}: {exc}") from exc
        processed_frames += anchor_count
        processed_episodes.append(record.episode_index)
        print(
            f"[{record_index + 1}/{len(records)}] episode={record.episode_index} "
            f"anchors={anchor_count} total_anchors={processed_frames}",
            file=sys.stderr,
        )

    if not processed_episodes:
        raise ValueError("No episode frames were processed")

    state_stats, state_minimum, state_maximum, state_sample_size = state_accumulator.finalize()
    action_stats, action_minimum, action_maximum, action_details = action_accumulator.finalize()
    state_stats, state_stabilization = _stabilize_stats(
        state_stats, state_minimum, state_maximum, groups, args.group_floor_ratio, "state"
    )
    action_stats, action_stabilization = _stabilize_stats(
        action_stats, action_minimum, action_maximum, groups, args.group_floor_ratio, "actions"
    )
    norm_stats = {"state": state_stats, "actions": action_stats}

    valid_counts = action_accumulator.valid_counts
    possible_counts = np.full(resolved.action_horizon, processed_frames, dtype=np.int64)
    padding_counts = possible_counts - valid_counts
    output_dir = resolved.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    normalize.save(output_dir, norm_stats)

    manifest = {
        "schema_version": 2,
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "config_name": resolved.config_name,
        "component_name": resolved.component_name,
        "asset_id": resolved.asset_id,
        "dataset_root": str(resolved.dataset_root),
        "dataset_robot_type": info.get("robot_type"),
        "dataset_fps": info.get("fps"),
        "split": args.split,
        "split_source": split_source,
        "task_index": args.task_index,
        "episode_indices": processed_episodes,
        "episode_count": len(processed_episodes),
        "anchor_frame_count": processed_frames,
        "selection_sha256": _selection_hash(resolved.dataset_root, processed_episodes, args),
        "norm_mode": resolved.norm_mode,
        "action_representation": "absolute" if resolved.norm_mode == "absolute" else "relative_eef",
        "action_horizon": resolved.action_horizon,
        "action_chunk_resampling": {
            "enabled": args.action_source_step_scale != 1.0,
            "source_step_scale": args.action_source_step_scale,
            "source_horizon": required_source_horizon(resolved.action_horizon, args.action_source_step_scale),
            "position_and_tail_interpolation": "linear",
            "rotation_interpolation": "quaternion_slerp",
        },
        "canonical_layout": {
            "dimension": vector_dim,
            "left_eef": [0, 9] if vector_dim in (EEF_DIM, GRIPPER_DIM, CANONICAL_DIM) else None,
            "right_eef": [9, 18] if vector_dim in (EEF_DIM, GRIPPER_DIM, CANONICAL_DIM) else None,
            "left_brainco": [18, 24] if vector_dim == CANONICAL_DIM else ([14, 20] if vector_dim == 26 else None),
            "right_brainco": [24, 30] if vector_dim == CANONICAL_DIM else ([20, 26] if vector_dim == 26 else None),
            "left_gripper": [18, 19]
            if vector_dim == GRIPPER_DIM
            else ([14, 15] if vector_dim == JOINT_GRIPPER_DIM else None),
            "right_gripper": [19, 20]
            if vector_dim == GRIPPER_DIM
            else ([15, 16] if vector_dim == JOINT_GRIPPER_DIM else None),
            "rotation6d": "first_column_then_second_column"
            if vector_dim in (EEF_DIM, GRIPPER_DIM, CANONICAL_DIM)
            else None,
            "relative_eef": "R_state.T@(p_action-p_state), R_state.T@R_action",
            "tail_action": "none" if vector_dim == EEF_DIM else "absolute",
            "action_domain": "eef_only"
            if vector_dim == EEF_DIM
            else ("gripper" if vector_dim in (JOINT_GRIPPER_DIM, GRIPPER_DIM) else "brainco"),
        },
        "input_rotation_format": resolved.input_rotation_format,
        "input_frame": resolved.input_frame,
        "dataset_contract": resolved.dataset_contract,
        "rotation_format_scores_first_episode": rotation_scores,
        "padding": {
            "policy": "excluded_from_all_action_statistics",
            "valid_samples_per_horizon": valid_counts.tolist(),
            "excluded_samples_per_horizon": padding_counts.tolist(),
        },
        "estimator": {
            "moments": "float64_chan_welford_population_std",
            "quantiles": "deterministic_uniform_priority_reservoir",
            "reservoir_capacity": args.reservoir_size,
            "seed": args.seed,
            "state_reservoir_samples": state_sample_size,
            **action_details,
        },
        "scale_floor": {
            "position": args.position_floor,
            "rotation6d": args.rotation_floor,
            "arm_joints": args.joint_floor,
            "brainco_or_gripper": args.hand_floor,
            "group_median_ratio": args.group_floor_ratio,
            "method": "symmetric_q01_q99_expansion_around_raw_quantile_midpoint",
        },
        "stabilized_ranges": [*state_stabilization, *action_stabilization],
        "stats_shapes": {
            "state": list(np.asarray(state_stats.mean).shape),
            "actions": list(np.asarray(action_stats.mean).shape),
        },
        "runtime_manifest_from_config": resolved.runtime_manifest,
    }
    if args.select_eef_only:
        manifest["source_projection"] = {
            "input_dimension": source_vector_dim,
            "output_dimension": EEF_DIM,
            "kept_slice": [0, EEF_DIM],
            "ignored_slice": [EEF_DIM, source_vector_dim],
            "ignored_tail": "gripper2" if source_vector_dim == GRIPPER_DIM else "none",
            "applied_before": [
                "frame_canonicalization",
                "relative_action_conversion",
                "task_progress_resampling",
                "normalization_accumulation",
            ],
        }
    manifest_path = output_dir / "norm_stats_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote {output_dir / 'norm_stats.json'}")
    print(f"Wrote {manifest_path}")
    print(
        f"mode={resolved.norm_mode} state_shape={np.asarray(state_stats.mean).shape} "
        f"action_shape={np.asarray(action_stats.mean).shape} episodes={len(processed_episodes)} "
        f"anchors={processed_frames} stabilized_groups={len(manifest['stabilized_ranges'])}"
    )


if __name__ == "__main__":
    main()
