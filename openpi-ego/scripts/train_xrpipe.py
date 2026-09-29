#!/usr/bin/env python3
"""Train the isolated XRPipe Mode1 single-right-fingertip policy."""

from __future__ import annotations

import argparse
import json
import pathlib

from openpi.training.xrpipe_train_config import PRESET
from openpi.training.xrpipe_train_config import build_train_config
from openpi.training.xrpipe_train_config import validate_resume_manifest


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default=PRESET, choices=(PRESET,))
    parser.add_argument("--dataset", required=True, type=pathlib.Path)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--validation-episodes", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=_positive_int, default=32)
    parser.add_argument("--fsdp-devices", type=_positive_int, default=1)
    parser.add_argument("--num-train-steps", type=_positive_int, default=30_000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--validation-interval", type=int, default=0)
    parser.add_argument("--validation-batches", type=int, default=0)
    parser.add_argument("--modality-diagnostics-interval", type=int, default=0)
    parser.add_argument("--save-interval", type=_positive_int, default=1_000)
    parser.add_argument("--keep-period", type=_positive_int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-checkpoint", default="gs://openpi-assets/checkpoints/pi05_base/params")
    lifecycle = parser.add_mutually_exclusive_group()
    lifecycle.add_argument("--overwrite", action="store_true")
    lifecycle.add_argument("--resume", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    kwargs = vars(args).copy()
    print_config = kwargs.pop("print_config")
    kwargs["wandb_enabled"] = not kwargs.pop("no_wandb")
    config = build_train_config(**kwargs)
    validate_resume_manifest(config)
    if print_config:
        print(json.dumps(config.policy_metadata, indent=2, ensure_ascii=False))
        return

    # Avoid initializing JAX devices for contract validation / --print-config.
    from train import main as train_main  # noqa: PLC0415

    train_main(config)


if __name__ == "__main__":
    main()
