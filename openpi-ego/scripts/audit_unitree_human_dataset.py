#!/usr/bin/env python3
"""Validate and print the Ego pose/action contract of a Human LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import pathlib

from openpi.training import unitree_train_config


def _recommendations(mode: str) -> dict:
    if mode == "mode1":
        return {
            "human_only": {
                "absolute_or_relative": "HUMAN_INPUT_FRAME=g1_base_tcp",
            },
            "robot_human_mixed": {
                "absolute_or_relative": "HUMAN_INPUT_FRAME=g1_base_tcp",
            },
        }
    return {
        "human_only": {
            "absolute_or_relative": "HUMAN_INPUT_FRAME=recording_tcp",
        },
        "robot_human_mixed": {
            "relative_only": "HUMAN_INPUT_FRAME=recording_tcp",
            "absolute": "unsupported: recording_frame has no robot placement",
        },
    }


def audit_dataset(dataset: pathlib.Path, requested_mode: str) -> dict:
    dataset = unitree_train_config._resolve_dataset(dataset)  # noqa: SLF001
    contract = unitree_train_config._resolve_human_dataset_contract(  # noqa: SLF001
        dataset,
        requested_mode,
    )
    info = json.loads((dataset / "meta/info.json").read_text(encoding="utf-8"))
    features = info.get("features", {})
    state = features.get("observation.state", {})
    action = features.get("action", {})
    return {
        "dataset": str(dataset),
        "requested_mode": requested_mode,
        "contract": contract,
        "vector_layout": {
            "state_shape": state.get("shape"),
            "action_shape": action.get("shape"),
            "state_names": state.get("names"),
            "action_names": action.get("names"),
        },
        "recommendations": _recommendations(contract["mode"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=pathlib.Path)
    parser.add_argument("--mode", choices=("mode1", "mode2"), required=True)
    args = parser.parse_args()
    print(json.dumps(audit_dataset(args.dataset, args.mode), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
