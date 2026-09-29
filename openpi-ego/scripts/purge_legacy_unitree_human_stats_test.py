import json
import pathlib

from scripts import purge_legacy_unitree_human_stats


def test_find_legacy_assets_excludes_explicit_source_frame_assets(tmp_path: pathlib.Path):
    (tmp_path / "meta/openpi_assets").mkdir(parents=True)
    (tmp_path / "meta/info.json").write_text("{}")
    legacy = tmp_path / "meta/openpi_assets/g1d_human_eef30_torso_palm_colgroup_abs_split_old"
    corrected = (
        tmp_path
        / "meta/openpi_assets/"
        "g1d_human_eef30_colgroup_ego_mode1_source_g1_base_tcp_contract_0123456789ab_abs_split_new"
    )
    legacy.mkdir()
    corrected.mkdir()

    assert purge_legacy_unitree_human_stats.find_legacy_assets(tmp_path) == [legacy]


def test_find_legacy_checkpoint_assets_returns_only_human_stats(tmp_path: pathlib.Path):
    assets = tmp_path / "run/100/assets"
    legacy_id = "g1d_human_eef20_gripper2_torso_palm_colgroup_abs_split_old"
    current_id = (
        "g1d_human_eef20_gripper2_colgroup_ego_mode2_source_recording_tcp_"
        "contract_0123456789ab_abs_split_new"
    )
    robot_id = "g1d_robot_eef20_gripper2_torso_palm_colgroup_abs_split_robot"
    for asset_id in (legacy_id, current_id, robot_id):
        (assets / asset_id).mkdir(parents=True)
    (assets / "runtime_manifest.json").write_text(
        json.dumps(
            {
                "components": [
                    {"kind": "human", "asset_id": legacy_id},
                    {"kind": "human", "asset_id": current_id},
                    {"kind": "robot", "asset_id": robot_id},
                ]
            }
        )
    )

    assert purge_legacy_unitree_human_stats.find_legacy_checkpoint_assets(tmp_path) == [
        assets / legacy_id
    ]
