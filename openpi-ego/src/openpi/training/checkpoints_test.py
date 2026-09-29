import json
import pathlib

import numpy as np
import pytest

import openpi.shared.normalize as normalize
import openpi.training.checkpoints as checkpoints
import openpi.training.config as config


def _stats(value: float) -> dict[str, normalize.NormStats]:
    array = np.asarray([value], dtype=np.float32)
    return {
        "state": normalize.NormStats(mean=array, std=np.ones_like(array), q01=array, q99=array),
        "actions": normalize.NormStats(mean=array, std=np.ones_like(array), q01=array, q99=array),
    }


def test_save_data_assets_preserves_single_dataset_layout(tmp_path: pathlib.Path):
    data_config = config.DataConfig(asset_id="single", norm_stats=_stats(1.0))

    checkpoints.save_data_assets(tmp_path / "assets", data_config)

    loaded = normalize.load(tmp_path / "assets" / "single")
    np.testing.assert_array_equal(loaded["state"].mean, np.asarray([1.0], dtype=np.float32))
    assert not (tmp_path / "assets" / "runtime_manifest.json").exists()
    assert not (tmp_path / "assets" / "single" / "runtime_manifest.json").exists()


def test_save_data_assets_writes_every_mixture_component_and_manifests(tmp_path: pathlib.Path):
    robot = config.DataConfig(
        repo_id="robot/data",
        asset_id="robot",
        norm_stats=_stats(1.0),
        runtime_manifest={"policy_kind": "robot", "model_dim": np.int64(30)},
    )
    human = config.DataConfig(
        repo_id="human/data",
        asset_id="human",
        norm_stats=_stats(2.0),
        runtime_manifest={"policy_kind": "human", "config_path": pathlib.Path("human.json")},
    )
    mixture = config.DataConfig(
        mixture_components=(robot, human),
        mixture_weights=(0.5, 0.5),
        mixture_names=("robot_mode2", "human_mode3"),
        runtime_manifest={"policy_kind": "mixed", "mix_ratio": [1, 1]},
    )

    checkpoints.save_data_assets(tmp_path / "assets", mixture)

    np.testing.assert_array_equal(
        normalize.load(tmp_path / "assets" / "robot")["actions"].mean,
        np.asarray([1.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        normalize.load(tmp_path / "assets" / "human")["actions"].mean,
        np.asarray([2.0], dtype=np.float32),
    )

    aggregate = json.loads((tmp_path / "assets" / "runtime_manifest.json").read_text())
    assert aggregate["policy_kind"] == "mixed"
    assert [component["name"] for component in aggregate["components"]] == ["robot_mode2", "human_mode3"]
    assert [component["asset_id"] for component in aggregate["components"]] == ["robot", "human"]

    robot_manifest = json.loads((tmp_path / "assets" / "robot" / "runtime_manifest.json").read_text())
    assert robot_manifest == {
        "name": "robot_mode2",
        "asset_id": "robot",
        "repo_id": "robot/data",
        "policy_kind": "robot",
        "model_dim": 30,
    }
    human_manifest = json.loads((tmp_path / "assets" / "human" / "runtime_manifest.json").read_text())
    assert human_manifest["config_path"] == "human.json"


def test_save_data_assets_rejects_conflicting_duplicate_asset_ids(tmp_path: pathlib.Path):
    mixture = config.DataConfig(
        mixture_components=(
            config.DataConfig(asset_id="shared", norm_stats=_stats(1.0)),
            config.DataConfig(asset_id="shared", norm_stats=_stats(2.0)),
        )
    )

    with pytest.raises(ValueError, match="share asset_id 'shared'"):
        checkpoints.save_data_assets(tmp_path / "assets", mixture)
