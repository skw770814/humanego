#!/usr/bin/env python3
"""Compute XRPipe Mode1 state and relative-action normalization statistics."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys

import numpy as np
import pyarrow.parquet as pq

import openpi.policies.xrpipe_policy as xrpipe_policy
from openpi.shared import normalize
from openpi.training.xrpipe_train_config import PRESET
from openpi.training.xrpipe_train_config import asset_id
from openpi.training.xrpipe_train_config import load_dataset_contract
from openpi.training.xrpipe_train_config import make_episode_split


def _as_2d(table, key: str, expected_dim: int) -> np.ndarray:
    values = np.asarray(table.column(key).to_pylist(), dtype=np.float32)
    if values.ndim != 2 or values.shape[-1] != expected_dim:
        raise ValueError(f"Expected {key} shape [N,{expected_dim}], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{key} contains non-finite values")
    return values


def _episode_rows(dataset: pathlib.Path) -> dict[int, dict]:
    rows = [
        json.loads(line)
        for line in (dataset / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {int(row["episode_index"]): row for row in rows}


def _parquet_path(dataset: pathlib.Path, info: dict, episode_index: int) -> pathlib.Path:
    chunks_size = int(info["chunks_size"])
    return dataset / info["data_path"].format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
    )


def compute(
    dataset: pathlib.Path,
    *,
    preset: str = PRESET,
    validation_episodes: int = 0,
    split_seed: int = 2026,
    action_horizon: int = 50,
    max_frames: int | None = None,
    force: bool = False,
) -> pathlib.Path:
    if preset != PRESET:
        raise ValueError(f"Unsupported XRPipe preset {preset!r}; expected {PRESET!r}")
    if action_horizon != 50:
        raise ValueError(f"{PRESET} fixes action_horizon=50, got {action_horizon}")
    contract = load_dataset_contract(dataset)
    split = make_episode_split(contract.dataset, validation_episodes, split_seed)
    resolved_asset_id = asset_id(contract, split)
    output_dir = contract.dataset / "meta" / "openpi_assets" / resolved_asset_id
    stats_path = output_dir / "norm_stats.json"
    if stats_path.exists() and not force:
        raise FileExistsError(f"Normalization stats already exist: {stats_path} (pass --force to replace)")

    info = json.loads((contract.dataset / "meta/info.json").read_text(encoding="utf-8"))
    episode_rows = _episode_rows(contract.dataset)
    state_stats = normalize.RunningStats()
    action_stats = normalize.RunningStats()
    state_transform = xrpipe_policy.CanonicalizeRelationState(contract.state_dim)
    action_transform = xrpipe_policy.RelativeFingertipActions(contract.action_dim, contract.reference_dim)

    processed_frames = 0
    valid_action_targets = 0
    padded_action_targets = 0
    processed_episodes: list[int] = []
    for row_index, episode_index in enumerate(split.train):
        if max_frames is not None and processed_frames >= max_frames:
            break
        parquet_path = _parquet_path(contract.dataset, info, episode_index)
        table = pq.read_table(
            parquet_path,
            columns=["observation.state", "observation.action_reference_tcp", "action"],
        )
        states = _as_2d(table, "observation.state", contract.state_dim)
        references = _as_2d(table, "observation.action_reference_tcp", contract.reference_dim)
        actions = _as_2d(table, "action", contract.action_dim)
        expected_length = int(episode_rows[episode_index]["length"])
        if not (len(states) == len(references) == len(actions) == expected_length):
            raise ValueError(
                f"Episode {episode_index} metadata/parquet lengths differ: "
                f"expected={expected_length}, state={len(states)}, reference={len(references)}, action={len(actions)}"
            )
        anchor_count = len(states)
        if max_frames is not None:
            anchor_count = min(anchor_count, max_frames - processed_frames)
        canonical_state = state_transform({"state": states[:anchor_count].copy()})["state"]
        state_stats.update(canonical_state)

        # action[j] is already the absolute target for j+1.  The loader's
        # horizon therefore starts at action[t], while reference remains t.
        anchor_batch = 2048
        for start in range(0, anchor_count, anchor_batch):
            stop = min(anchor_count, start + anchor_batch)
            indices = np.arange(start, stop)[:, None] + np.arange(action_horizon)[None, :]
            valid = indices < len(actions)
            safe = np.minimum(indices, len(actions) - 1)
            chunks = actions[safe]
            relative = action_transform(
                {
                    "action_reference": references[start:stop].copy(),
                    "actions": chunks.copy(),
                }
            )["actions"]
            action_stats.update(relative[valid])
            valid_action_targets += int(valid.sum())
            padded_action_targets += int((~valid).sum())

        processed_frames += anchor_count
        processed_episodes.append(episode_index)
        print(
            f"[{row_index + 1}/{len(split.train)}] episode={episode_index} "
            f"anchors={anchor_count} total={processed_frames}",
            file=sys.stderr,
        )

    if processed_frames < 2 or valid_action_targets < 2:
        raise ValueError("At least two XRPipe frames/action targets are required for normalization")
    expected_frames = sum(int(episode_rows[index]["length"]) for index in split.train)
    complete = processed_episodes == list(split.train) and processed_frames == expected_frames
    norm_stats = {"state": state_stats.get_statistics(), "actions": action_stats.get_statistics()}
    output_dir.mkdir(parents=True, exist_ok=True)
    normalize.save(output_dir, norm_stats)
    manifest = {
        "schema_version": 1,
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "preset": PRESET,
        "asset_id": resolved_asset_id,
        "dataset_root": str(contract.dataset),
        "contract_digest": contract.digest,
        "dataset_contract": contract.as_dict(),
        "split_digest": split.digest,
        "train_episodes": list(split.train),
        "processed_episodes": processed_episodes,
        "action_horizon": action_horizon,
        "norm_mode": "shared",
        "complete": complete,
        "state_dim": contract.state_dim,
        "reference_dim": contract.reference_dim,
        "action_dim": contract.action_dim,
        "processed_frames": processed_frames,
        "valid_action_targets": valid_action_targets,
        "padded_action_targets_excluded": padded_action_targets,
        "state_transform": "per pose9 block: S @ T @ S, S=diag(1,1,-1,1)",
        "action_transform": "inv(reference[t]) @ absolute_action[t+k]; gripper unchanged",
    }
    (output_dir / "norm_stats_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {stats_path}")
    return output_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default=PRESET, choices=(PRESET,))
    parser.add_argument("--dataset", required=True, type=pathlib.Path)
    parser.add_argument("--validation-episodes", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    compute(**vars(_parser().parse_args()))


if __name__ == "__main__":
    main()
