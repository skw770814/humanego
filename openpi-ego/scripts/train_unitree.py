#!/usr/bin/env python3
"""Train an isolated G1-D BrainCo, EEF20-gripper, or Human EEF18 mixture."""

from __future__ import annotations

import argparse
import json
import pathlib

from openpi.training.unitree_train_config import build_train_config
from openpi.training.unitree_train_config import parse_unitree_preset
from openpi.training.unitree_train_config import validate_resume_manifest


def _human_mode(value: str) -> int:
    try:
        mode = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("human mode must be one value: 0, 3, 4, 5, or 6") from exc
    if mode not in (0, 3, 4, 5, 6):
        raise argparse.ArgumentTypeError("human mode must be one value: 0, 3, 4, 5, or 6")
    return mode


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--preset", help="Preset name listed in train_unitree.sh; overrides individual mode flags")
    parser.add_argument("--robot-mode", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument(
        "--human-mode",
        "--human-modes",
        dest="human_mode",
        type=_human_mode,
        default=0,
        help="One Human mode (3/4/5/6), or 0 to disable Human data; numeric mode is shared by both action domains",
    )
    parser.add_argument("--action-representation", choices=("absolute", "relative"), default="absolute")
    parser.add_argument("--relative-norm", choices=("shared", "per_step", "hybrid"), default="shared")
    parser.add_argument(
        "--robot-camera-mode",
        choices=("single", "three"),
        default="three",
        help="Robot image input: head only, or head plus both wrist cameras",
    )
    parser.add_argument(
        "--human-dataset-mode",
        choices=("mode1", "mode2"),
        default="mode1",
        help="Ego converter mode stored in the selected Human LeRobot dataset; independent of Human mode 3/4/5/6",
    )
    parser.add_argument(
        "--human-input-frame",
        choices=("native", "torso_palm", "g1_base_tcp", "recording_tcp", "pelvis_wrist"),
        default=None,
        help="Human pose profile; default is g1_base_tcp for Mode1 and recording_tcp for Mode2",
    )
    parser.add_argument(
        "--action-domain",
        choices=("brainco", "gripper"),
        default="brainco",
        help="EEF tail layout: BrainCo12 (30D total) or left/right gripper2 (20D total)",
    )
    parser.add_argument("--robot-fraction", type=float, default=0.5)
    parser.add_argument(
        "--task-progress-alignment",
        action="store_true",
        help="Resample only the Human EEF chunk using the automatic Robot/Human median episode-duration ratio",
    )
    parser.add_argument("--robot-joint-dataset", type=pathlib.Path)
    parser.add_argument("--robot-gripper-joint-dataset", type=pathlib.Path)
    parser.add_argument("--robot-eef-dataset", type=pathlib.Path)
    parser.add_argument("--human-dataset", type=pathlib.Path)
    parser.add_argument(
        "--human-eef-only-dataset",
        type=pathlib.Path,
        help="Human EEF18 or EEF20-gripper source; EEF-only presets ignore a recorded gripper tail",
    )
    parser.add_argument(
        "--human-eef-only",
        action="store_true",
        help="Supervise only Human EEF18; discard a validated EEF20 source's final gripper2 tail",
    )
    parser.add_argument("--robot-gripper-eef-dataset", type=pathlib.Path)
    parser.add_argument("--human-gripper-dataset", type=pathlib.Path)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--robot-validation-episodes", type=int, default=0)
    parser.add_argument("--human-validation-episodes", type=int, default=0)
    parser.add_argument("--max-train-episodes", type=int)
    parser.add_argument("--minimum-episode-frames", type=int, default=10)
    parser.add_argument(
        "--robot-task-contains",
        default="fold clothes",
        help="Case-insensitive Robot task substring; pass an empty string to include every Robot task",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--fsdp-devices",
        type=_positive_int,
        default=1,
        help="Number of visible devices used to shard model and optimizer state",
    )
    parser.add_argument("--num-train-steps", type=int, default=30_000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--validation-interval", type=int, default=0)
    parser.add_argument("--validation-batches", type=int, default=0)
    parser.add_argument("--modality-diagnostics-interval", type=int, default=0)
    parser.add_argument("--save-interval", type=int, default=1_000)
    parser.add_argument(
        "--keep-period",
        type=_positive_int,
        default=5_000,
        help="Permanently retain checkpoints whose step is a multiple of this value",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-checkpoint", default="gs://openpi-assets/checkpoints/pi05_base/params")
    lifecycle = parser.add_mutually_exclusive_group()
    lifecycle.add_argument("--overwrite", action="store_true")
    lifecycle.add_argument("--resume", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--print-config", action="store_true", help="Validate and print the resolved experiment only")
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    kwargs = vars(args).copy()
    print_config = kwargs.pop("print_config")
    preset_name = kwargs.pop("preset")
    human_mode = kwargs.pop("human_mode")
    if preset_name is None:
        kwargs["human_modes"] = () if human_mode == 0 else (human_mode,)
    else:
        try:
            preset = parse_unitree_preset(preset_name)
        except ValueError as exc:
            parser.error(str(exc))
        kwargs["robot_mode"] = preset.robot_mode
        kwargs["human_modes"] = () if preset.human_mode == 0 else (preset.human_mode,)
        kwargs["action_representation"] = preset.action_representation
        kwargs["relative_norm"] = preset.relative_norm
        kwargs["action_domain"] = preset.action_domain
        kwargs["robot_camera_mode"] = preset.robot_camera_mode
        kwargs["task_progress_alignment"] = preset.task_progress_alignment
        kwargs["human_eef_only"] = preset.human_eef_only
    kwargs["robot_task_contains"] = kwargs["robot_task_contains"] or None
    kwargs["wandb_enabled"] = not kwargs.pop("no_wandb")
    for key in (
        "robot_joint_dataset",
        "robot_gripper_joint_dataset",
        "robot_eef_dataset",
        "human_dataset",
        "human_eef_only_dataset",
        "robot_gripper_eef_dataset",
        "human_gripper_dataset",
    ):
        if kwargs[key] is None:
            kwargs.pop(key)
    config = build_train_config(**kwargs)
    validate_resume_manifest(config)
    if print_config:
        resolved = config.data.create(config.assets_dirs, config.model)
        print(json.dumps(resolved.runtime_manifest, indent=2, ensure_ascii=False))
        return

    # Imported lazily so --print-config does not initialize JAX accelerators.
    from train import main as train_main

    train_main(config)


if __name__ == "__main__":
    main()
