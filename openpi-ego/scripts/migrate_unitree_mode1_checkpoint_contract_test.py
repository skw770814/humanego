import copy
import json
import pathlib

import pytest

from scripts import migrate_unitree_mode1_checkpoint_contract as migrate_mode1


def _write_fixture(root: pathlib.Path, *, dimension: int = 20) -> tuple[pathlib.Path, pathlib.Path]:
    checkpoint = root / "checkpoint"
    assets = checkpoint / "assets"
    robot_id = "g1d_robot_eef20_gripper2_torso_palm_colgroup_abs_split_robot"
    (assets / robot_id).mkdir(parents=True)
    (assets / robot_id / "runtime_manifest.json").write_text(json.dumps({"kind": "robot"}))
    domain = "eef_only" if dimension == 18 else "gripper"
    runtime = {
        "schema_version": 1,
        "training_config": "unitree_g1d_brainco_train",
        "action_representation": "absolute",
        "task_progress_alignment": {
            "task_texts": ["fold clothes"],
            "human_source_step_scale": 0.25,
        },
        "components": [
            {
                "name": "robot_mode2",
                "kind": "robot",
                "mode": 2,
                "dataset": "/data/robot",
                "dimension": 20,
                "camera_mode": "three",
                "asset_id": robot_id,
            },
            {
                "name": "human_mode3",
                "kind": "human",
                "mode": 3,
                "dataset": "/data/gripper_mode1",
                "dimension": dimension,
                "action_domain": domain,
                "split_digest": "0123456789ab",
                "task_progress_alignment": {
                    "digest": "abcdef012345",
                    "human_source_step_scale": 0.25,
                },
                "asset_id": "legacy",
            },
        ],
    }
    (assets / "runtime_manifest.json").write_text(json.dumps(runtime))

    layout = "18_only" if dimension == 18 else "20_gripper2"
    stats_id = (
        f"g1d_human_eef{layout}_colgroup_ego_mode1_source_g1_base_tcp_"
        "contract_111111111111_abs_split_0123456789ab_progress_abcdef012345"
    )
    stats_dir = root / stats_id
    stats_dir.mkdir()
    (stats_dir / "norm_stats.json").write_text(
        json.dumps(
            {
                "norm_stats": {
                    "state": {"mean": [0.0] * dimension},
                    "actions": {"mean": [0.0] * dimension},
                }
            }
        )
    )
    (stats_dir / "norm_stats_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "asset_id": stats_id,
                "dataset_root": "/reconstruction/source",
                "input_frame": "g1_base_tcp",
                "dataset_contract": {
                    "mode": "mode1",
                    "stored_action": "absolute",
                    "reference_frame": "g1_base",
                    "eef_frame": "wrist_yaw_tcp",
                    "digest": "111111111111",
                },
                "action_chunk_resampling": {
                    "enabled": True,
                    "source_step_scale": 0.25,
                },
            }
        )
    )
    return checkpoint, stats_dir


@pytest.mark.parametrize("dimension", [18, 20])
def test_migrate_attaches_only_contract_qualified_mode1_stats(tmp_path: pathlib.Path, dimension: int):
    checkpoint, stats_dir = _write_fixture(tmp_path, dimension=dimension)
    destination = migrate_mode1.migrate(checkpoint, stats_dir, apply=True)

    runtime = json.loads((checkpoint / "assets/runtime_manifest.json").read_text())
    human = next(component for component in runtime["components"] if component["kind"] == "human")
    assert runtime["schema_version"] == 2
    assert runtime["human_dataset_mode"] == "mode1"
    assert runtime["human_input_frame"] == "g1_base_tcp"
    assert human["input_frame"] == "g1_base_tcp"
    assert human["canonical_frame"] == "torso_palm"
    assert human["frame_conversion_applied"] is True
    assert destination.is_dir()
    stats_manifest = json.loads((destination / "norm_stats_manifest.json").read_text())
    assert stats_manifest["dataset_root"] == "/data/gripper_mode1"
    assert stats_manifest["reconstruction_source_dataset_root"] == "/reconstruction/source"


def test_migrate_rejects_mode2_even_with_mode1_stats(tmp_path: pathlib.Path):
    checkpoint, stats_dir = _write_fixture(tmp_path)
    path = checkpoint / "assets/runtime_manifest.json"
    runtime = json.loads(path.read_text())
    runtime["components"][1]["dataset"] = "/data/gripper_mode2"
    path.write_text(json.dumps(runtime))

    with pytest.raises(ValueError, match="Mode1"):
        migrate_mode1.migrate(checkpoint, stats_dir, apply=False)


def test_migrate_dry_run_does_not_write(tmp_path: pathlib.Path):
    checkpoint, stats_dir = _write_fixture(tmp_path)
    before = copy.deepcopy(json.loads((checkpoint / "assets/runtime_manifest.json").read_text()))

    destination = migrate_mode1.migrate(checkpoint, stats_dir, apply=False)

    assert not destination.exists()
    assert json.loads((checkpoint / "assets/runtime_manifest.json").read_text()) == before
