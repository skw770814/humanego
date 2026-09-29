#!/usr/bin/env python3
"""Attach recomputed Mode1 Human stats and upgrade one legacy checkpoint contract.

This migration is intentionally Mode1-only.  It cannot be used to bless a
legacy Mode2 checkpoint, whose historical recording-frame semantics were not
encoded strongly enough to recover safely.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil

MODE1_ASSET = re.compile(
    r"^g1d_human_eef(?P<dimension>18_only|20_gripper2)_colgroup_"
    r"ego_mode1_source_g1_base_tcp_contract_(?P<contract>[0-9a-f]{12})_"
    r"abs_split_(?P<split>[0-9a-f]{12})_progress_(?P<progress>[0-9a-f]{12})$"
)


def _read_json(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: pathlib.Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _stats_dimension(stats: dict, *, path: pathlib.Path) -> int:
    try:
        state = stats["norm_stats"]["state"]["mean"]
        actions = stats["norm_stats"]["actions"]["mean"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Malformed norm_stats.json: {path}") from exc
    if not isinstance(state, list) or not isinstance(actions, list) or len(state) != len(actions):
        raise ValueError(f"State/action statistic dimensions disagree: {path}")
    return len(state)


def validate_inputs(checkpoint: pathlib.Path, stats_dir: pathlib.Path) -> tuple[dict, dict, dict]:
    runtime_path = checkpoint / "assets/runtime_manifest.json"
    stats_path = stats_dir / "norm_stats.json"
    stats_manifest_path = stats_dir / "norm_stats_manifest.json"
    for path in (runtime_path, stats_path, stats_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    runtime = _read_json(runtime_path)
    stats = _read_json(stats_path)
    stats_manifest = _read_json(stats_manifest_path)
    if runtime.get("action_representation") != "absolute":
        raise ValueError("Only absolute-action Mode1 checkpoints can be migrated")

    humans = [component for component in runtime.get("components", []) if component.get("kind") == "human"]
    if len(humans) != 1:
        raise ValueError(f"Expected exactly one Human component, found {len(humans)}")
    human = humans[0]
    if "mode1" not in pathlib.Path(str(human.get("dataset", ""))).name.lower():
        raise ValueError("Legacy Human component does not identify a Mode1 dataset")

    asset_id = str(stats_manifest.get("asset_id", ""))
    match = MODE1_ASSET.fullmatch(asset_id)
    if match is None or stats_dir.name != asset_id:
        raise ValueError(f"Stats asset is not an explicit g1_base_tcp Mode1 asset: {stats_dir}")
    contract = stats_manifest.get("dataset_contract")
    if (
        stats_manifest.get("schema_version") != 2
        or stats_manifest.get("input_frame") != "g1_base_tcp"
        or not isinstance(contract, dict)
        or contract.get("mode") != "mode1"
        or contract.get("stored_action") != "absolute"
        or contract.get("reference_frame") != "g1_base"
        or contract.get("eef_frame") != "wrist_yaw_tcp"
        or contract.get("digest") != match["contract"]
    ):
        raise ValueError("Stats manifest does not carry the required Mode1 g1_base_tcp contract")

    expected_dimension = int(str(match["dimension"]).split("_", maxsplit=1)[0])
    if int(human.get("dimension", -1)) != expected_dimension:
        raise ValueError(
            f"Human checkpoint dimension {human.get('dimension')} != stats dimension {expected_dimension}"
        )
    if _stats_dimension(stats, path=stats_path) != expected_dimension:
        raise ValueError("norm_stats.json dimension does not match its asset id")
    if str(human.get("split_digest")) != match["split"]:
        raise ValueError("Checkpoint split digest does not match the recomputed stats")

    progress = human.get("task_progress_alignment")
    if not isinstance(progress, dict) or progress.get("digest") != match["progress"]:
        raise ValueError("Checkpoint task-progress digest does not match the recomputed stats")
    resampling = stats_manifest.get("action_chunk_resampling")
    if (
        not isinstance(resampling, dict)
        or not resampling.get("enabled")
        or float(resampling.get("source_step_scale", -1.0))
        != float(progress.get("human_source_step_scale", -2.0))
    ):
        raise ValueError("Checkpoint task-progress scale does not match the recomputed stats")

    expected_domain = "eef_only" if expected_dimension == 18 else "gripper"
    if human.get("action_domain") != expected_domain:
        raise ValueError(f"Checkpoint Human action domain is not {expected_domain!r}")
    return runtime, stats, stats_manifest


def migrate(checkpoint: pathlib.Path, stats_dir: pathlib.Path, *, apply: bool) -> pathlib.Path:
    checkpoint = checkpoint.expanduser().resolve()
    stats_dir = stats_dir.expanduser().resolve()
    runtime, _stats, stats_manifest = validate_inputs(checkpoint, stats_dir)
    human = next(component for component in runtime["components"] if component.get("kind") == "human")
    destination = checkpoint / "assets" / str(stats_manifest["asset_id"])

    if not apply:
        print(f"WOULD COPY {stats_dir} -> {destination}")
        print(f"WOULD UPGRADE {checkpoint / 'assets/runtime_manifest.json'}")
        return destination
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing stats asset: {destination}")

    # Keep the checkpoint's original dataset identity while recording exactly
    # where this one-off reconstruction read its samples.
    migrated_stats_manifest = dict(stats_manifest)
    migrated_stats_manifest["config_name"] = runtime.get("training_config")
    migrated_stats_manifest["component_name"] = human.get("name")
    migrated_stats_manifest["reconstruction_source_dataset_root"] = stats_manifest.get("dataset_root")
    migrated_stats_manifest["dataset_root"] = human["dataset"]
    migrated_stats_manifest["checkpoint_contract_migration"] = {
        "kind": "legacy_mode1_g1_base_tcp_restore",
        "checkpoint": str(checkpoint),
    }

    shutil.copytree(stats_dir, destination)
    _write_json(destination / "norm_stats_manifest.json", migrated_stats_manifest)

    human.update(
        {
            "asset_id": stats_manifest["asset_id"],
            "input_frame": "g1_base_tcp",
            "canonical_frame": "torso_palm",
            "frame_conversion_applied": True,
            "human_dataset_contract": stats_manifest["dataset_contract"],
        }
    )
    for component in runtime["components"]:
        if component.get("kind") == "robot":
            component.update(
                {
                    "input_frame": "torso_palm",
                    "canonical_frame": "torso_palm",
                    "frame_conversion_applied": False,
                }
            )
    robot = next((component for component in runtime["components"] if component.get("kind") == "robot"), None)
    top_progress = runtime.get("task_progress_alignment") or {}
    task_texts = top_progress.get("task_texts") or []
    runtime.update(
        {
            "schema_version": 2,
            "robot_camera_mode": None if robot is None else robot.get("camera_mode"),
            "robot_task_contains": task_texts[0] if task_texts else None,
            "human_dataset_mode": "mode1",
            "human_input_frame": "g1_base_tcp",
        }
    )
    _write_json(checkpoint / "assets/runtime_manifest.json", runtime)

    component_runtime = {
        "name": human.get("name"),
        "asset_id": human["asset_id"],
        "repo_id": human["dataset"],
        **human,
    }
    _write_json(destination / "runtime_manifest.json", component_runtime)
    if robot is not None:
        robot_runtime_path = checkpoint / "assets" / str(robot["asset_id"]) / "runtime_manifest.json"
        if robot_runtime_path.is_file():
            robot_runtime = _read_json(robot_runtime_path)
            robot_runtime.update(
                {
                    "input_frame": "torso_palm",
                    "canonical_frame": "torso_palm",
                    "frame_conversion_applied": False,
                }
            )
            _write_json(robot_runtime_path, robot_runtime)
    print(f"MIGRATED {checkpoint}")
    print(f"ATTACHED {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=pathlib.Path)
    parser.add_argument("stats_dir", type=pathlib.Path)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Copy stats and update manifests; default is validation-only dry-run",
    )
    args = parser.parse_args()
    migrate(args.checkpoint, args.stats_dir, apply=args.apply)


if __name__ == "__main__":
    main()
