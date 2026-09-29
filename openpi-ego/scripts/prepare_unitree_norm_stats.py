#!/usr/bin/env python3
"""Compute one Robot or Human normalization asset inside its LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import pathlib

import compute_unitree_norm_stats as core

from openpi.training.unitree_train_config import DEFAULT_HUMAN_DATASET
from openpi.training.unitree_train_config import DEFAULT_HUMAN_EEF_ONLY_DATASET
from openpi.training.unitree_train_config import DEFAULT_HUMAN_GRIPPER_DATASET
from openpi.training.unitree_train_config import DEFAULT_ROBOT_EEF_DATASET
from openpi.training.unitree_train_config import DEFAULT_ROBOT_GRIPPER_EEF_DATASET
from openpi.training.unitree_train_config import DEFAULT_ROBOT_GRIPPER_JOINT_DATASET
from openpi.training.unitree_train_config import DEFAULT_ROBOT_JOINT_DATASET
from openpi.training.unitree_train_config import UnitreeExperimentDataConfig
from openpi.training.unitree_train_config import parse_unitree_norm_preset


def _human_mode(value: str) -> int:
    try:
        mode = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("human mode must be one value: 0, 3, 4, 5, or 6") from exc
    if mode not in (0, 3, 4, 5, 6):
        raise argparse.ArgumentTypeError("human mode must be one value: 0, 3, 4, 5, or 6")
    return mode


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--norm-preset", help="Single-domain preset listed in compute_unitree_norm_stats.sh")
    parser.add_argument("--robot-mode", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument(
        "--human-mode",
        "--human-modes",
        dest="human_mode",
        type=_human_mode,
        default=0,
        help="One Human mode (3/4/5/6), or 0 to disable Human data",
    )
    parser.add_argument("--action-representation", choices=("absolute", "relative"), default="absolute")
    parser.add_argument("--relative-norm", choices=("shared", "per_step", "hybrid"), default="shared")
    parser.add_argument("--action-domain", choices=("brainco", "gripper"), default="brainco")
    parser.add_argument(
        "--robot-camera-mode",
        choices=("single", "three"),
        default="three",
        help="Recorded in the resolved Robot component; numeric normalization is camera-independent",
    )
    parser.add_argument(
        "--human-dataset-mode",
        choices=("mode1", "mode2"),
        default="mode1",
        help="Ego converter source mode stored in the selected Human LeRobot dataset",
    )
    parser.add_argument(
        "--human-input-frame",
        choices=("native", "torso_palm", "g1_base_tcp", "recording_tcp", "pelvis_wrist"),
        default=None,
        help="Human pose profile; default is g1_base_tcp for Mode1 and recording_tcp for Mode2",
    )
    parser.add_argument(
        "--task-progress-alignment",
        action="store_true",
        help="Build Human EEF stats with automatic Robot/Human median-duration progress alignment",
    )
    parser.add_argument("--robot-joint-dataset", type=pathlib.Path, default=DEFAULT_ROBOT_JOINT_DATASET)
    parser.add_argument("--robot-gripper-joint-dataset", type=pathlib.Path, default=DEFAULT_ROBOT_GRIPPER_JOINT_DATASET)
    parser.add_argument("--robot-eef-dataset", type=pathlib.Path, default=DEFAULT_ROBOT_EEF_DATASET)
    parser.add_argument("--human-dataset", type=pathlib.Path, default=DEFAULT_HUMAN_DATASET)
    parser.add_argument(
        "--human-eef-only-dataset",
        type=pathlib.Path,
        default=DEFAULT_HUMAN_EEF_ONLY_DATASET,
        help="Human EEF18 or EEF20-gripper source; EEF-only presets ignore a recorded gripper tail",
    )
    parser.add_argument(
        "--human-eef-only",
        action="store_true",
        help="Supervise only Human EEF18; discard a validated EEF20 source's final gripper2 tail",
    )
    parser.add_argument("--robot-gripper-eef-dataset", type=pathlib.Path, default=DEFAULT_ROBOT_GRIPPER_EEF_DATASET)
    parser.add_argument("--human-gripper-dataset", type=pathlib.Path, default=DEFAULT_HUMAN_GRIPPER_DATASET)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--robot-validation-episodes", type=int, default=0)
    parser.add_argument("--human-validation-episodes", type=int, default=0)
    parser.add_argument("--max-train-episodes", type=int, help="Must match the training experiment when set")
    parser.add_argument("--minimum-episode-frames", type=int, default=10)
    parser.add_argument(
        "--robot-task-contains",
        default="fold clothes",
        help="Case-insensitive Robot task substring; pass an empty string to include every Robot task",
    )
    parser.add_argument(
        "--output-root",
        type=pathlib.Path,
        help="Override dataset/meta/openpi_assets; intended only for smoke tests",
    )
    parser.add_argument("--max-episodes", type=int, help="Smoke test only; requires --output-root")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--reservoir-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if (args.max_episodes is not None or args.max_frames is not None) and args.output_root is None:
        parser.error("--max-episodes/--max-frames require a temporary --output-root to protect production stats")

    try:
        preset = None if args.norm_preset is None else parse_unitree_norm_preset(args.norm_preset)
    except ValueError as exc:
        parser.error(str(exc))
    robot_mode = args.robot_mode if preset is None else preset.robot_mode
    human_mode = args.human_mode if preset is None else preset.human_mode
    action_representation = args.action_representation if preset is None else preset.action_representation
    relative_norm = args.relative_norm if preset is None else preset.relative_norm
    action_domain = args.action_domain if preset is None else preset.action_domain
    robot_camera_mode = args.robot_camera_mode if preset is None else preset.robot_camera_mode
    task_progress_alignment = args.task_progress_alignment if preset is None else preset.task_progress_alignment
    human_eef_only = args.human_eef_only if preset is None else preset.human_eef_only
    factory = UnitreeExperimentDataConfig(
        robot_mode=robot_mode,
        human_modes=() if human_mode == 0 else (human_mode,),
        action_representation=action_representation,
        relative_norm=relative_norm,
        action_domain=action_domain,
        robot_camera_mode=robot_camera_mode,
        human_dataset_mode=args.human_dataset_mode,
        human_input_frame=args.human_input_frame,
        task_progress_alignment=task_progress_alignment,
        robot_joint_dataset=args.robot_joint_dataset,
        robot_gripper_joint_dataset=args.robot_gripper_joint_dataset,
        robot_eef_dataset=args.robot_eef_dataset,
        human_dataset=args.human_dataset,
        human_eef_only_dataset=args.human_eef_only_dataset,
        robot_gripper_eef_dataset=args.robot_gripper_eef_dataset,
        human_gripper_dataset=args.human_gripper_dataset,
        human_eef_only=human_eef_only,
        split_seed=args.split_seed,
        robot_validation_episodes=args.robot_validation_episodes,
        human_validation_episodes=args.human_validation_episodes,
        max_train_episodes=args.max_train_episodes,
        minimum_episode_frames=args.minimum_episode_frames,
        robot_task_contains=args.robot_task_contains or None,
    )
    specs = factory.specs()  # The factory is the single source of truth for split/asset ids.
    if len(specs) != 1:
        parser.error("Normalization must select exactly one domain; compute Robot and Human in separate runs")

    spec = specs[0]
    norm_mode = "absolute" if action_representation == "absolute" else relative_norm
    output_base = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else spec.dataset / "meta" / "openpi_assets"
    )
    output_dir = output_base / spec.asset_id
    core_args = [
        "--dataset",
        str(spec.dataset),
        "--output-dir",
        str(output_dir),
        "--asset-id",
        spec.asset_id,
        "--norm-mode",
        norm_mode,
        "--input-rotation-format",
        "columns_grouped",
        "--input-frame",
        spec.input_frame,
        "--episodes",
        ",".join(map(str, spec.split.train)),
        "--reservoir-size",
        str(args.reservoir_size),
        "--seed",
        str(args.seed),
    ]
    if spec.task_progress_alignment is not None:
        core_args += [
            "--action-source-step-scale",
            str(spec.task_progress_alignment["human_source_step_scale"]),
        ]
    if spec.human_dataset_contract is not None:
        core_args += [
            "--dataset-contract-json",
            json.dumps(spec.human_dataset_contract, sort_keys=True),
        ]
    if spec.action_domain == "eef_only":
        # EEF-only is the supervised domain. The source may be native EEF18
        # or EEF20-gripper; the core removes a recorded gripper tail before
        # every normalization operation.
        core_args.append("--select-eef-only")
    if args.max_episodes is not None:
        core_args += ["--max-episodes", str(args.max_episodes)]
    if args.max_frames is not None:
        core_args += ["--max-frames", str(args.max_frames)]
    print(f"[{spec.name}] {spec.dataset} -> {output_dir}")
    if not args.dry_run:
        core.main(core_args)


if __name__ == "__main__":
    main()
