#!/usr/bin/env python3
"""List or delete Unitree checkpoints with an ambiguous Human Ego contract."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil

VALID_ASSET = re.compile(
    r"(?:_ego_mode1_source_(?:g1_base_tcp|pelvis_wrist)"
    r"|_ego_mode2_source_recording_tcp)_contract_[0-9a-f]{12}_"
)


def find_legacy_checkpoints(root: pathlib.Path) -> list[pathlib.Path]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint root not found: {root}")
    invalid: set[pathlib.Path] = set()
    for manifest_path in root.glob("**/assets/runtime_manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        human_components = [
            component for component in manifest.get("components", []) if component.get("kind") == "human"
        ]
        if not human_components:
            continue
        dataset_mode = manifest.get("human_dataset_mode")
        source_frame = manifest.get("human_input_frame")
        valid_frames = {
            "mode1": ("g1_base_tcp", "pelvis_wrist"),
            "mode2": ("recording_tcp",),
        }.get(dataset_mode, ())
        valid = source_frame in valid_frames and all(
            component.get("input_frame") == source_frame
            and component.get("human_dataset_contract", {}).get("mode") == dataset_mode
            and component.get("human_dataset_contract", {}).get("stored_action") == "absolute"
            and VALID_ASSET.search(str(component.get("asset_id", ""))) is not None
            for component in human_components
        )
        if not valid:
            invalid.add(manifest_path.parent.parent)
    return sorted(invalid)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_root", type=pathlib.Path)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Permanently delete matched checkpoint step directories; default is dry-run",
    )
    args = parser.parse_args()

    matches = find_legacy_checkpoints(args.checkpoint_root)
    if not matches:
        print("No ambiguous legacy Human checkpoints found.")
        return
    for path in matches:
        print(("DELETE " if args.apply else "INVALID ") + str(path))
    if not args.apply:
        print("Dry-run only. Re-run with --apply after checking every path.")
        return
    for path in matches:
        shutil.rmtree(path)
    print(f"Deleted {len(matches)} invalid Human checkpoint directorie(s).")


if __name__ == "__main__":
    main()
