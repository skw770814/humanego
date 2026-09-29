import json
import pathlib

import pytest

import openpi.models.pi0_config as pi0_config
from openpi.training import unitree_train_config
from openpi.training import xrpipe_train_config as config
import openpi.transforms as transforms


def _write_dataset(root: pathlib.Path, *, schema: str = "xrpipe_v4", semantics_overrides=None) -> pathlib.Path:
    meta = root / "meta"
    meta.mkdir(parents=True)
    semantics = dict(config.EXPECTED_ACTION_SEMANTICS)
    if semantics_overrides:
        semantics.update(semantics_overrides)
    (meta / "action_semantics.json").write_text(json.dumps(semantics))
    (root / "extraction_meta.json").write_text(
        json.dumps(
            {
                "schema_version": schema,
                "timestamp_semantics": config.EXPECTED_TIMESTAMP_SEMANTICS,
            }
        )
    )
    (meta / "episodes.jsonl").write_text(
        '{"episode_index":0,"tasks":["pick object one"],"length":20}\n'
        '{"episode_index":1,"tasks":["pick object one"],"length":24}\n'
    )
    (meta / "tasks.jsonl").write_text('{"task_index":0,"task":"pick object one"}\n')
    (meta / "info.json").write_text(
        json.dumps(
            {
                "robot_type": config.EXPECTED_ROBOT_TYPE,
                "fps": config.EXPECTED_FPS,
                "chunks_size": 1000,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "features": {
                    "observation.state": {"shape": [19], "names": [[f"state_{i}" for i in range(19)]]},
                    "observation.action_reference_tcp": {
                        "shape": [9],
                        "names": [[f"reference_{i}" for i in range(9)]],
                    },
                    "action": {"shape": [10], "names": [[f"action_{i}" for i in range(10)]]},
                    config.EXPECTED_IMAGE_KEY: {"shape": [480, 640, 3]},
                },
                "ego_relation": {
                    "object_order": ["obj1", "obj2"],
                    "object_categories": ["case", "earphone"],
                    "action_storage": "absolute",
                    "action_reference_field": "observation.action_reference_tcp",
                    "action_reference_frame": config.EXPECTED_REFERENCE_FRAME,
                    "action_coordinate_system": config.EXPECTED_COORDINATE_SYSTEM,
                    "action_source_coordinate_system": config.EXPECTED_SOURCE_COORDINATE_SYSTEM,
                    "training_state_transform": config.EXPECTED_STATE_TRANSFORM,
                },
            }
        )
    )
    return root


def test_contract_resolves_two_object_fingertip_dataset(tmp_path: pathlib.Path):
    dataset = _write_dataset(tmp_path)

    contract = config.load_dataset_contract(dataset)

    assert contract.state_dim == 19
    assert contract.reference_dim == 9
    assert contract.action_dim == 10
    assert contract.object_order == ("obj1", "obj2")
    assert contract.semantics["control_point"] == "right_thumb_index_fingertip_midpoint"


def test_contract_rejects_old_schema_and_wrong_control_point(tmp_path: pathlib.Path):
    old = _write_dataset(tmp_path / "old", schema="xrpipe_v3")
    with pytest.raises(ValueError, match="rerun pipeline Step4"):
        config.load_dataset_contract(old)

    wrong = _write_dataset(
        tmp_path / "wrong",
        semantics_overrides={"control_point": "right_hand_palm_link"},
    )
    with pytest.raises(ValueError, match="fingertip contract"):
        config.load_dataset_contract(wrong)


def test_contract_rejects_legacy_jittered_timestamp_semantics(tmp_path: pathlib.Path):
    dataset = _write_dataset(tmp_path)
    (dataset / "extraction_meta.json").write_text(json.dumps({"schema_version": "xrpipe_v4"}))

    with pytest.raises(ValueError, match="legacy/jittered timestamps"):
        config.load_dataset_contract(dataset)


def test_mode1_data_config_has_separate_dimensions_and_manifest(tmp_path: pathlib.Path, monkeypatch):
    dataset = _write_dataset(tmp_path)
    monkeypatch.setattr(config.ModelTransformFactory, "__call__", lambda self, model: transforms.Group())
    monkeypatch.setattr(config.XRPipeMode1DataConfig, "_load_norm_stats", lambda self, path, asset: None)
    factory = config.XRPipeMode1DataConfig(dataset=dataset)
    model = pi0_config.Pi0RTCConfig(pi05=True, action_dim=32, action_horizon=50)

    resolved = factory.create(tmp_path / "unused", model)

    assert resolved.action_sequence_keys == ("action",)
    assert resolved.prompt_from_task
    assert resolved.runtime_manifest["state_dim"] == 19
    assert resolved.runtime_manifest["reference_dim"] == 9
    assert resolved.runtime_manifest["action_dim"] == 10
    assert resolved.runtime_manifest["deployment_supported"] is False
    assert "fingertip10_rel_shared" in resolved.asset_id


def test_unitree_mode2_parser_does_not_accept_xrpipe_contract(tmp_path: pathlib.Path):
    dataset = _write_dataset(tmp_path)

    with pytest.raises(ValueError, match="must be trained with scripts/train_xrpipe.py"):
        unitree_train_config._resolve_human_dataset_contract(dataset, "mode2")  # noqa: SLF001
