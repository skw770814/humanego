import json
import pathlib

from scripts import purge_legacy_unitree_human_checkpoints


def _write_manifest(step: pathlib.Path, manifest: dict) -> pathlib.Path:
    assets = step / "assets"
    assets.mkdir(parents=True)
    (assets / "runtime_manifest.json").write_text(json.dumps(manifest))
    return step


def test_find_legacy_checkpoints_requires_full_ego_contract(tmp_path: pathlib.Path):
    valid_contract = {
        "mode": "mode1",
        "stored_action": "absolute",
    }
    valid = _write_manifest(
        tmp_path / "valid" / "100",
        {
            "human_dataset_mode": "mode1",
            "human_input_frame": "g1_base_tcp",
            "components": [
                {
                    "kind": "human",
                    "input_frame": "g1_base_tcp",
                    "asset_id": (
                        "g1d_human_eef30_colgroup_ego_mode1_source_g1_base_tcp_"
                        "contract_0123456789ab_abs_split_test"
                    ),
                    "human_dataset_contract": valid_contract,
                }
            ],
        },
    )
    legacy = _write_manifest(
        tmp_path / "legacy" / "200",
        {
            "human_input_frame": "torso_palm",
            "components": [
                {
                    "kind": "human",
                    "input_frame": "torso_palm",
                    "asset_id": "g1d_human_eef30_torso_palm_colgroup_source_torso_palm_abs_split_old",
                }
            ],
        },
    )

    assert purge_legacy_unitree_human_checkpoints.find_legacy_checkpoints(tmp_path) == [legacy]
    assert valid not in purge_legacy_unitree_human_checkpoints.find_legacy_checkpoints(tmp_path)
