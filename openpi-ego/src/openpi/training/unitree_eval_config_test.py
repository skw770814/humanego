import json
import pathlib

import numpy as np
import pytest

from openpi.training import unitree_eval_config


def _manifest() -> dict:
    return {
        "schema_version": 2,
        "rotation_6d": "first_column_then_second_column",
        "rotation_6d_layout": "columns_grouped:[r00,r10,r20,r01,r11,r21]",
        "reference_frame": "torso_link",
        "eef_links": ["left_hand_palm_link", "right_hand_palm_link"],
        "frame_convention": "T_reference_eef",
        "urdf_sha256": "a" * 64,
        "dataset_fps": 30,
        "normalization_clip": 5.0,
        "action_representation": "relative",
        "human_dataset_mode": "mode1",
        "human_input_frame": "g1_base_tcp",
        "components": [
            {
                "name": "robot_mode2",
                "kind": "robot",
                "mode": 2,
                "dimension": 30,
                "asset_id": "robot_eef_relative",
                "eef_kinematics": {
                    "urdf_sha256": "a" * 64,
                    "reference_frame": "torso_link",
                    "eef_links": ["left_hand_palm_link", "right_hand_palm_link"],
                    "frame_convention": "T_reference_eef",
                    "rotation_6d": "columns_grouped:[r00,r10,r20,r01,r11,r21]",
                },
            },
            {
                "name": "human_mode4",
                "kind": "human",
                "mode": 4,
                "dimension": 30,
                "asset_id": "human_eef_relative",
                "input_frame": "g1_base_tcp",
                "human_dataset_contract": {
                    "mode": "mode1",
                    "stored_action": "absolute",
                },
            },
        ],
    }


def test_eval_config_selects_robot_asset_from_mixture(tmp_path: pathlib.Path):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "runtime_manifest.json").write_text(json.dumps(_manifest()))
    robot_asset = assets / "robot_eef_relative"
    robot_asset.mkdir()
    (robot_asset / "norm_stats.json").write_text("{}")

    config = unitree_eval_config.build_eval_config(tmp_path)

    assert config.data.asset_id == "robot_eef_relative"
    assert config.data.robot_mode == 2
    assert config.data.action_representation == "relative"
    assert config.data.camera_mode == "three"
    assert config.policy_metadata["robot_camera_mode"] == "three"


def test_eval_config_restores_single_camera_mode_from_manifest(tmp_path: pathlib.Path):
    assets = tmp_path / "assets"
    assets.mkdir()
    manifest = _manifest()
    manifest["robot_camera_mode"] = "single"
    manifest["components"][0]["camera_mode"] = "single"
    (assets / "runtime_manifest.json").write_text(json.dumps(manifest))
    robot_asset = assets / "robot_eef_relative"
    robot_asset.mkdir()
    (robot_asset / "norm_stats.json").write_text("{}")

    config = unitree_eval_config.build_eval_config(tmp_path)
    resolved = config.data.create(config.assets_dirs, config.model)
    transform = resolved.data_transforms.inputs[0]
    image = np.ones((8, 10, 3), dtype=np.uint8)
    transformed = transform(
        {
            "images": {"cam_high": image},
            "state": np.zeros(30, dtype=np.float32),
        }
    )

    assert config.data.camera_mode == "single"
    assert config.policy_metadata["robot_camera_mode"] == "single"
    assert transformed["image_mask"]["base_0_rgb"]
    assert not transformed["image_mask"]["left_wrist_0_rgb"]
    assert not transformed["image_mask"]["right_wrist_0_rgb"]


def test_eval_config_rejects_manifest_component_camera_mismatch(tmp_path: pathlib.Path):
    assets = tmp_path / "assets"
    assets.mkdir()
    manifest = _manifest()
    manifest["robot_camera_mode"] = "single"
    manifest["components"][0]["camera_mode"] = "three"
    (assets / "runtime_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="camera mode does not match"):
        unitree_eval_config.build_eval_config(tmp_path)


def test_eval_config_rejects_old_human_checkpoint_without_source_frame(tmp_path: pathlib.Path):
    assets = tmp_path / "assets"
    assets.mkdir()
    manifest = _manifest()
    manifest.pop("human_input_frame")
    manifest["components"][1].pop("input_frame")
    (assets / "runtime_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="does not match its Ego dataset mode"):
        unitree_eval_config.build_eval_config(tmp_path)


def test_eval_config_ignores_eef_only_human_tail_and_keeps_robot_contract(tmp_path: pathlib.Path):
    assets = tmp_path / "assets"
    assets.mkdir()
    manifest = _manifest()
    manifest["human_action_domain"] = "eef_only"
    manifest["components"][1].update(
        {
            "dimension": 18,
            "action_domain": "eef_only",
            "action_tail": "none",
            "supervised_action_dimensions": 18,
            "source_projection": {
                "input_dimension": 20,
                "output_dimension": 18,
                "kept_slice": [0, 18],
                "ignored_slice": [18, 20],
                "ignored_tail": "gripper2",
            },
        }
    )
    (assets / "runtime_manifest.json").write_text(json.dumps(manifest))
    robot_asset = assets / "robot_eef_relative"
    robot_asset.mkdir()
    (robot_asset / "norm_stats.json").write_text("{}")

    config = unitree_eval_config.build_eval_config(tmp_path)

    assert config.data.action_dimension == 30
    assert config.data.asset_id == "robot_eef_relative"
    assert config.policy_metadata["human_action_domain"] == "eef_only"
    assert config.policy_metadata["components"][1]["source_projection"]["ignored_slice"] == [18, 20]


def test_eval_config_rejects_rotation_layout_mismatch(tmp_path: pathlib.Path):
    assets = tmp_path / "assets"
    assets.mkdir()
    manifest = _manifest()
    manifest["rotation_6d"] = "legacy_interleaved_columns"
    (assets / "runtime_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="incompatible"):
        unitree_eval_config.load_checkpoint_manifest(tmp_path)


def test_eval_config_rejects_component_urdf_mismatch(tmp_path: pathlib.Path):
    assets = tmp_path / "assets"
    assets.mkdir()
    manifest = _manifest()
    manifest["components"][0]["eef_kinematics"]["urdf_sha256"] = "b" * 64
    (assets / "runtime_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="component FK metadata"):
        unitree_eval_config.build_eval_config(tmp_path)


def test_eval_config_accepts_eef20_gripper_runtime_contract(tmp_path: pathlib.Path):
    assets = tmp_path / "assets"
    assets.mkdir()
    manifest = _manifest()
    manifest.update(
        {
            "action_domain": "gripper",
            "action_tail": "gripper2",
            "robot_type": "unitree_g1_dex1",
            "end_effector": "dex1",
        }
    )
    robot = manifest["components"][0]
    robot.update({"dimension": 20, "action_domain": "gripper", "action_tail": "gripper2"})
    manifest["components"] = [robot]
    (assets / "runtime_manifest.json").write_text(json.dumps(manifest))
    robot_asset = assets / "robot_eef_relative"
    robot_asset.mkdir()
    (robot_asset / "norm_stats.json").write_text("{}")

    config = unitree_eval_config.build_eval_config(tmp_path)

    assert config.data.action_dimension == 20
    assert config.policy_metadata["robot_type"] == "unitree_g1_dex1"


def test_eval_config_accepts_joint16_gripper_runtime_contract(tmp_path: pathlib.Path):
    assets = tmp_path / "assets"
    assets.mkdir()
    manifest = _manifest()
    manifest.update(
        {
            "action_representation": "absolute",
            "action_domain": "gripper",
            "action_tail": "gripper2",
            "robot_type": "unitree_g1_dex1",
            "end_effector": "dex1",
            "action_space": "joint",
        }
    )
    robot = manifest["components"][0]
    robot.update(
        {
            "mode": 1,
            "dimension": 16,
            "action_domain": "gripper",
            "action_tail": "gripper2",
            "action_space": "joint",
            "asset_id": "robot_joint16_gripper_absolute",
        }
    )
    robot.pop("eef_kinematics")
    manifest["components"] = [robot]
    (assets / "runtime_manifest.json").write_text(json.dumps(manifest))
    robot_asset = assets / "robot_joint16_gripper_absolute"
    robot_asset.mkdir()
    (robot_asset / "norm_stats.json").write_text("{}")

    config = unitree_eval_config.build_eval_config(tmp_path)

    assert config.data.robot_mode == 1
    assert config.data.action_dimension == 16
    assert config.data.action_representation == "absolute"
