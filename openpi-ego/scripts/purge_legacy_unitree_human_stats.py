#!/usr/bin/env python3
"""List or delete obsolete Human norm assets with no Ego action/frame contract."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil

LEGACY_PREFIXES = (
    "g1d_human_eef18_only_colgroup_",
    "g1d_human_eef20_gripper2_colgroup_",
    "g1d_human_eef30_colgroup_",
    "g1d_human_eef18_only_torso_palm_colgroup_",
    "g1d_human_eef20_gripper2_torso_palm_colgroup_",
    "g1d_human_eef30_torso_palm_colgroup_",
)
VALID_CONTRACT = re.compile(
    r"(?:_ego_mode1_source_(?:g1_base_tcp|pelvis_wrist)"
    r"|_ego_mode2_source_recording_tcp)_contract_[0-9a-f]{12}_"
)


def find_legacy_assets(dataset: pathlib.Path) -> list[pathlib.Path]:
    dataset = dataset.expanduser().resolve()
    if not (dataset / "meta/info.json").is_file():
        raise FileNotFoundError(f"LeRobot dataset meta/info.json not found: {dataset}")
    assets_root = dataset / "meta/openpi_assets"
    if not assets_root.is_dir():
        return []
    return sorted(
        child
        for child in assets_root.iterdir()
        if child.is_dir()
        and child.name.startswith(LEGACY_PREFIXES)
        and VALID_CONTRACT.search(child.name) is None
    )


def find_legacy_checkpoint_assets(root: pathlib.Path) -> list[pathlib.Path]:
    """Find only bad Human stats copied into checkpoints, preserving weights/Robot assets."""
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint root not found: {root}")
    matches: set[pathlib.Path] = set()
    for manifest_path in root.glob("**/assets/runtime_manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for component in manifest.get("components", []):
            if component.get("kind") != "human":
                continue
            asset_id = str(component.get("asset_id", ""))
            asset_dir = manifest_path.parent / asset_id
            if VALID_CONTRACT.search(asset_id) is None and asset_dir.is_dir():
                matches.add(asset_dir)
    return sorted(matches)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", type=pathlib.Path, nargs="*")
    parser.add_argument(
        "--checkpoint-root",
        type=pathlib.Path,
        action="append",
        default=[],
        help="Also scan copied Human norm assets inside this checkpoint tree; may be repeated",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Permanently delete matched legacy asset directories; default is dry-run",
    )
    args = parser.parse_args()
    if not args.datasets and not args.checkpoint_root:
        parser.error("provide at least one LeRobot dataset or --checkpoint-root")

    matches = [path for dataset in args.datasets for path in find_legacy_assets(dataset)]
    matches.extend(
        path
        for checkpoint_root in args.checkpoint_root
        for path in find_legacy_checkpoint_assets(checkpoint_root)
    )
    matches = sorted(set(matches))
    if not matches:
        print("No legacy unqualified Human norm assets found in datasets or checkpoints.")
        return
    for path in matches:
        print(("DELETE " if args.apply else "STALE  ") + str(path))
    if not args.apply:
        print("Dry-run only. Re-run with --apply after checking every path.")
        return
    for path in matches:
        shutil.rmtree(path)
    print(f"Deleted {len(matches)} legacy Human norm asset directorie(s).")


if __name__ == "__main__":
    main()
