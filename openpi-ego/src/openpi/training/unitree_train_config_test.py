import json
import pathlib

import pytest

from openpi.training import unitree_train_config


def _write_info(dataset: pathlib.Path, **overrides) -> None:
    metadata = {
        "urdf_sha256": "a" * 64,
        "reference_link": "torso_link",
        "left_eef_link": "left_hand_palm_link",
        "right_eef_link": "right_hand_palm_link",
        "frame_convention": "T_reference_eef",
        "rotation_6d": "columns_grouped:[r00,r10,r20,r01,r11,r21]",
        "brainco": "copied_without_transformation",
    }
    metadata.update(overrides)
    (dataset / "meta").mkdir(parents=True)
    eef_names = [f"eef_{index}" for index in range(18)]
    brainco_names = [
        "kLeftHandThumb",
        "kLeftHandThumbAux",
        "kLeftHandIndex",
        "kLeftHandMiddle",
        "kLeftHandRing",
        "kLeftHandPinky",
        "kRightHandThumb",
        "kRightHandThumbAux",
        "kRightHandIndex",
        "kRightHandMiddle",
        "kRightHandRing",
        "kRightHandPinky",
    ]
    feature = {"shape": [30], "names": [[*eef_names, *brainco_names]]}
    (dataset / "meta/info.json").write_text(
        json.dumps(
            {
                "eef_forward_kinematics": metadata,
                "features": {"observation.state": feature, "action": feature},
            }
        )
    )


def _write_human_dataset(
    dataset: pathlib.Path,
    *,
    semantic_mode: str | None = None,
    tcp_semantics: str | None = None,
    extraction_mode: str | None = None,
) -> None:
    (dataset / "meta").mkdir(parents=True)
    feature = {
        "shape": [30],
        "names": [[*[f"eef_{index}" for index in range(18)], *unitree_train_config.G1D_BRAINCO_NAMES]],
    }
    (dataset / "meta/info.json").write_text(
        json.dumps({"features": {"observation.state": feature, "action": feature}})
    )
    (dataset / "meta/tasks.jsonl").write_text('{"task_index":0,"task":"fold clothes."}\n')
    (dataset / "meta/episodes.jsonl").write_text(
        '{"episode_index":0,"tasks":["fold clothes."],"length":100}\n'
    )
    if semantic_mode is not None:
        (dataset / "meta/action_semantics.json").write_text(
            json.dumps(
                {
                    "mode": semantic_mode,
                    "tcp_semantics": tcp_semantics,
                    "tcp_pose9": {
                        "rotation_6d": "first two columns of a 3x3 rotation matrix",
                    },
                }
            )
        )
    if extraction_mode is not None:
        (dataset / "extraction_meta.json").write_text(
            json.dumps({"config": {"pipeline_mode": extraction_mode, "cache_version": 19}})
        )


def _write_mode1_semantics(dataset: pathlib.Path) -> None:
    (dataset / "meta/action_semantics.json").write_text(
        json.dumps(
            {
                "mode": "mode1_ideal",
                "tcp_semantics": "absolute_target_in_g1_base",
                "tcp_pose9": {
                    "rotation_6d": "first two columns of a 3x3 rotation matrix",
                },
            }
        )
    )


def test_robot_eef_metadata_preserves_urdf_and_frame_contract(tmp_path: pathlib.Path):
    _write_info(tmp_path)

    actual = unitree_train_config._validate_robot_eef_metadata(tmp_path)  # noqa: SLF001

    assert actual == {
        "urdf_sha256": "a" * 64,
        "reference_frame": "torso_link",
        "eef_links": ["left_hand_palm_link", "right_hand_palm_link"],
        "frame_convention": "T_reference_eef",
        "rotation_6d": "columns_grouped:[r00,r10,r20,r01,r11,r21]",
    }


def test_robot_eef_metadata_rejects_a_different_reference_frame(tmp_path: pathlib.Path):
    _write_info(tmp_path, reference_link="pelvis")

    with pytest.raises(ValueError, match="incompatible"):
        unitree_train_config._validate_robot_eef_metadata(tmp_path)  # noqa: SLF001


def test_gripper_eef_metadata_requires_native_gripper_tail(tmp_path: pathlib.Path):
    _write_info(tmp_path, brainco=None, gripper="copied_without_transformation")
    info_path = tmp_path / "meta/info.json"
    info = json.loads(info_path.read_text())
    gripper_feature = {
        "shape": [20],
        "names": [[*[f"eef_{index}" for index in range(18)], "kLeftGripper", "kRightGripper"]],
    }
    info["features"] = {"observation.state": gripper_feature, "action": gripper_feature}
    info_path.write_text(json.dumps(info))

    actual = unitree_train_config._validate_robot_eef_metadata(tmp_path, "gripper")  # noqa: SLF001

    assert actual["reference_frame"] == "torso_link"


def test_gripper_domain_rejects_brainco30_layout(tmp_path: pathlib.Path):
    _write_info(tmp_path)

    with pytest.raises(ValueError, match="gripper EEF dataset requires"):
        unitree_train_config._validate_robot_eef_metadata(tmp_path, "gripper")  # noqa: SLF001


def test_eef_only_human_layout_accepts_eef18_or_eef20_gripper_source(tmp_path: pathlib.Path):
    (tmp_path / "meta").mkdir()
    names = [f"eef_{index}" for index in range(18)]
    feature = {"shape": [18], "names": [names]}
    info_path = tmp_path / "meta/info.json"
    info_path.write_text(json.dumps({"features": {"observation.state": feature, "action": feature}}))

    assert unitree_train_config._validate_eef_vector_layout(tmp_path, "eef_only") == 18  # noqa: SLF001

    gripper_feature = {
        "shape": [20],
        "names": [[*names, "kLeftGripper", "kRightGripper"]],
    }
    info_path.write_text(json.dumps({"features": {"observation.state": gripper_feature, "action": gripper_feature}}))
    assert unitree_train_config._validate_eef_vector_layout(tmp_path, "eef_only") == 20  # noqa: SLF001

    gripper_feature["names"][0][-2:] = ["kRightGripper", "kLeftGripper"]
    info_path.write_text(json.dumps({"features": {"observation.state": gripper_feature, "action": gripper_feature}}))
    with pytest.raises(ValueError, match="EEF18 or EEF18"):
        unitree_train_config._validate_eef_vector_layout(tmp_path, "eef_only")  # noqa: SLF001


def test_eef_only_human_component_uses_18d_asset_and_no_action_tail(tmp_path: pathlib.Path):
    (tmp_path / "meta").mkdir()
    names = [*[f"eef_{index}" for index in range(18)], "kLeftGripper", "kRightGripper"]
    feature = {"shape": [20], "names": [names]}
    (tmp_path / "meta/info.json").write_text(
        json.dumps({"features": {"observation.state": feature, "action": feature}})
    )
    (tmp_path / "meta/tasks.jsonl").write_text('{"task_index":0,"task":"fold clothes."}\n')
    (tmp_path / "meta/episodes.jsonl").write_text('{"episode_index":0,"tasks":["fold clothes."],"length":100}\n')
    _write_mode1_semantics(tmp_path)
    factory = unitree_train_config.UnitreeExperimentDataConfig(
        robot_mode=0,
        human_modes=(3,),
        human_eef_only=True,
        human_eef_only_dataset=tmp_path,
    )

    (spec,) = factory.specs()

    assert spec.dimension == 18
    assert spec.action_domain == "eef_only"
    assert spec.metadata["action_tail"] == "none"
    assert spec.metadata["supervised_action_dimensions"] == 18
    assert spec.source_dimension == 20
    assert spec.metadata["source_projection"] == {
        "input_dimension": 20,
        "output_dimension": 18,
        "kept_slice": [0, 18],
        "ignored_slice": [18, 20],
        "ignored_tail": "gripper2",
    }
    assert "g1d_human_eef18_only" in spec.asset_id
    assert spec.input_frame == "g1_base_tcp"
    assert "_ego_mode1_source_g1_base_tcp_contract_" in spec.asset_id
    assert spec.metadata["frame_conversion_applied"]


def test_legacy_pelvis_wrist_human_uses_a_separate_explicit_asset_namespace(tmp_path: pathlib.Path):
    (tmp_path / "meta").mkdir()
    names = [*[f"eef_{index}" for index in range(18)], *unitree_train_config.G1D_BRAINCO_NAMES]
    feature = {"shape": [30], "names": [names]}
    (tmp_path / "meta/info.json").write_text(
        json.dumps({"features": {"observation.state": feature, "action": feature}})
    )
    (tmp_path / "meta/tasks.jsonl").write_text('{"task_index":0,"task":"fold clothes."}\n')
    (tmp_path / "meta/episodes.jsonl").write_text('{"episode_index":0,"tasks":["fold clothes."],"length":100}\n')
    _write_mode1_semantics(tmp_path)
    factory = unitree_train_config.UnitreeExperimentDataConfig(
        robot_mode=0,
        human_modes=(3,),
        human_dataset=tmp_path,
        human_input_frame="pelvis_wrist",
    )

    (spec,) = factory.specs()

    assert spec.input_frame == "pelvis_wrist"
    assert "_ego_mode1_source_pelvis_wrist_contract_" in spec.asset_id
    assert spec.metadata["frame_conversion_applied"]


def test_current_mode2_absolute_contract_is_accepted_and_namespaced(tmp_path: pathlib.Path):
    _write_human_dataset(
        tmp_path,
        semantic_mode="mode2_state",
        tcp_semantics="absolute_next_target_in_recording_frame",
    )
    factory = unitree_train_config.UnitreeExperimentDataConfig(
        robot_mode=0,
        human_modes=(3,),
        human_dataset=tmp_path,
        human_dataset_mode="mode2",
    )

    (spec,) = factory.specs()

    assert spec.human_dataset_contract["stored_action"] == "absolute"
    assert spec.human_dataset_contract["state_action_alignment"] == "action[t]=absolute_state[t+1]"
    assert spec.input_frame == "recording_tcp"
    assert "_ego_mode2_source_recording_tcp_contract_" in spec.asset_id


def test_human_only_manifest_does_not_claim_robot_deployment_frame(tmp_path: pathlib.Path):
    _write_human_dataset(
        tmp_path,
        semantic_mode="mode1_ideal",
        tcp_semantics="absolute_target_in_g1_base",
    )
    factory = unitree_train_config.UnitreeExperimentDataConfig(
        robot_mode=0,
        human_modes=(3,),
        human_dataset=tmp_path,
    )

    config = factory.create(tmp_path, unitree_train_config.pi0_config.Pi0RTCConfig(pi05=True))
    manifest = config.runtime_manifest

    assert manifest["reference_frame"] is None
    assert manifest["eef_links"] is None
    assert manifest["frame_convention"] is None
    assert manifest["robot_type"] is None
    assert manifest["end_effector"] is None
    assert manifest["human_input_frame"] == "g1_base_tcp"
    assert manifest["components"][0]["canonical_frame"] == "torso_palm"


def test_legacy_mode2_without_action_semantics_is_rejected(tmp_path: pathlib.Path):
    _write_human_dataset(tmp_path, extraction_mode="mode2_state")
    factory = unitree_train_config.UnitreeExperimentDataConfig(
        robot_mode=0,
        human_modes=(3,),
        human_dataset=tmp_path,
        human_dataset_mode="mode2",
    )

    with pytest.raises(ValueError, match="Legacy Mode2 parquet may contain relative actions"):
        factory.specs()


def test_unlabelled_human_dataset_is_rejected_even_when_cli_says_mode1(tmp_path: pathlib.Path):
    _write_human_dataset(tmp_path)
    factory = unitree_train_config.UnitreeExperimentDataConfig(
        robot_mode=0,
        human_modes=(3,),
        human_dataset=tmp_path,
        human_dataset_mode="mode1",
    )

    with pytest.raises(ValueError, match="CLI mode flag alone is not accepted"):
        factory.specs()


def test_mixed_training_rejects_native_human_pose_contract(tmp_path: pathlib.Path):
    robot = tmp_path / "robot"
    human = tmp_path / "human"
    _write_info(robot)
    (robot / "meta/tasks.jsonl").write_text('{"task_index":0,"task":"fold clothes."}\n')
    (robot / "meta/episodes.jsonl").write_text(
        '{"episode_index":0,"tasks":["fold clothes."],"length":100}\n'
    )
    _write_human_dataset(
        human,
        semantic_mode="mode1_ideal",
        tcp_semantics="absolute_target_in_g1_base",
    )
    factory = unitree_train_config.UnitreeExperimentDataConfig(
        robot_mode=2,
        human_modes=(3,),
        robot_eef_dataset=robot,
        human_dataset=human,
        human_dataset_mode="mode1",
        human_input_frame="native",
    )

    with pytest.raises(ValueError, match="must be canonicalized"):
        factory.specs()


def test_mode2_absolute_mixing_is_rejected_even_with_tcp_to_palm_profile(tmp_path: pathlib.Path):
    robot = tmp_path / "robot"
    human = tmp_path / "human"
    _write_info(robot)
    (robot / "meta/tasks.jsonl").write_text('{"task_index":0,"task":"fold clothes."}\n')
    (robot / "meta/episodes.jsonl").write_text(
        '{"episode_index":0,"tasks":["fold clothes."],"length":100}\n'
    )
    _write_human_dataset(
        human,
        semantic_mode="mode2_state",
        tcp_semantics="absolute_next_target_in_recording_frame",
    )
    factory = unitree_train_config.UnitreeExperimentDataConfig(
        robot_mode=2,
        human_modes=(3,),
        action_representation="absolute",
        robot_eef_dataset=robot,
        human_dataset=human,
        human_dataset_mode="mode2",
        human_input_frame="recording_tcp",
    )

    with pytest.raises(ValueError, match="Absolute Robot/Human mixing is unsafe for Ego Mode2"):
        factory.specs()


@pytest.mark.parametrize(
    ("dataset_mode", "input_frame"),
    [("mode1", "g1_base_tcp"), ("mode2", "recording_tcp")],
)
def test_supported_mixed_pose_profiles_resolve_to_robot_palm(
    tmp_path: pathlib.Path,
    dataset_mode: str,
    input_frame: str,
):
    robot = tmp_path / "robot"
    human = tmp_path / "human"
    _write_info(robot)
    (robot / "meta/tasks.jsonl").write_text('{"task_index":0,"task":"fold clothes."}\n')
    (robot / "meta/episodes.jsonl").write_text(
        '{"episode_index":0,"tasks":["fold clothes."],"length":100}\n'
    )
    _write_human_dataset(
        human,
        semantic_mode="mode1_ideal" if dataset_mode == "mode1" else "mode2_state",
        tcp_semantics=(
            "absolute_target_in_g1_base"
            if dataset_mode == "mode1"
            else "absolute_next_target_in_recording_frame"
        ),
    )
    factory = unitree_train_config.UnitreeExperimentDataConfig(
        robot_mode=2,
        human_modes=(3,),
        action_representation="relative",
        robot_eef_dataset=robot,
        human_dataset=human,
        human_dataset_mode=dataset_mode,
        human_input_frame=input_frame,
    )

    robot_spec, human_spec = factory.specs()

    assert robot_spec.metadata["canonical_frame"] == "torso_palm"
    assert human_spec.metadata["canonical_frame"] in ("torso_palm", "recording_palm")
    assert human_spec.metadata["frame_conversion_applied"]


def test_existing_human_gripper_preset_keeps_all_20_dimensions(tmp_path: pathlib.Path):
    (tmp_path / "meta").mkdir()
    names = [*[f"eef_{index}" for index in range(18)], "kLeftGripper", "kRightGripper"]
    feature = {"shape": [20], "names": [names]}
    (tmp_path / "meta/info.json").write_text(
        json.dumps({"features": {"observation.state": feature, "action": feature}})
    )
    (tmp_path / "meta/tasks.jsonl").write_text('{"task_index":0,"task":"fold clothes."}\n')
    (tmp_path / "meta/episodes.jsonl").write_text('{"episode_index":0,"tasks":["fold clothes."],"length":100}\n')
    _write_mode1_semantics(tmp_path)
    factory = unitree_train_config.UnitreeExperimentDataConfig(
        robot_mode=0,
        human_modes=(3,),
        action_domain="gripper",
        human_gripper_dataset=tmp_path,
    )

    (spec,) = factory.specs()

    assert spec.dimension == 20
    assert spec.action_domain == "gripper"
    assert spec.metadata["action_tail"] == "gripper2"
    assert spec.source_dimension is None
    assert "source_projection" not in spec.metadata


def test_eef_only_flag_requires_a_human_component():
    factory = unitree_train_config.UnitreeExperimentDataConfig(
        robot_mode=2,
        human_modes=(),
        human_eef_only=True,
    )

    with pytest.raises(ValueError, match="requires one Human mode"):
        factory.specs()


def test_single_camera_robot_repack_reads_only_the_head_camera(tmp_path: pathlib.Path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text(
        json.dumps(
            {
                "features": {
                    "observation.images.cam_left_high": {"dtype": "video"},
                    "observation.state": {"shape": [26]},
                    "action": {"shape": [26]},
                }
            }
        )
    )
    spec = unitree_train_config.ComponentSpec(
        name="robot_mode1",
        kind="robot",
        mode=1,
        dataset=tmp_path,
        dimension=26,
        camera_mode="single",
        model_uses_state=True,
        action_space="joint",
        asset_id="joint26",
        split=unitree_train_config.EpisodeSplit((0,), (), "test"),
    )

    config = unitree_train_config._make_component_data_config(  # noqa: SLF001
        spec,
        model_config=unitree_train_config.pi0_config.Pi0RTCConfig(pi05=True),
        representation="absolute",
        load_norm_stats=lambda *_: None,
    )
    structure = config.repack_transforms.inputs[0].structure

    assert structure["images"] == {"cam_high": "observation.images.cam_left_high"}


def test_three_camera_robot_repack_remains_unchanged(tmp_path: pathlib.Path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text(
        json.dumps(
            {
                "features": {
                    "observation.images.cam_left_high": {"dtype": "video"},
                    "observation.images.cam_left_wrist": {"dtype": "video"},
                    "observation.images.cam_right_wrist": {"dtype": "video"},
                    "observation.state": {"shape": [26]},
                    "action": {"shape": [26]},
                }
            }
        )
    )
    spec = unitree_train_config.ComponentSpec(
        name="robot_mode1",
        kind="robot",
        mode=1,
        dataset=tmp_path,
        dimension=26,
        camera_mode="three",
        model_uses_state=True,
        action_space="joint",
        asset_id="joint26",
        split=unitree_train_config.EpisodeSplit((0,), (), "test"),
    )

    config = unitree_train_config._make_component_data_config(  # noqa: SLF001
        spec,
        model_config=unitree_train_config.pi0_config.Pi0RTCConfig(pi05=True),
        representation="absolute",
        load_norm_stats=lambda *_: None,
    )
    structure = config.repack_transforms.inputs[0].structure

    assert structure["images"] == {
        "cam_high": "observation.images.cam_left_high",
        "cam_left_wrist": "observation.images.cam_left_wrist",
        "cam_right_wrist": "observation.images.cam_right_wrist",
    }


def test_zero_validation_uses_every_episode_for_training(tmp_path: pathlib.Path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text("{}")
    (tmp_path / "meta/tasks.jsonl").write_text('{"task_index":0,"task":"fold clothes"}\n')
    (tmp_path / "meta/episodes.jsonl").write_text(
        "".join(f'{{"episode_index":{index},"tasks":["fold clothes"],"length":100}}\n' for index in range(3))
    )

    split = unitree_train_config.make_episode_split(
        tmp_path,
        validation_count=0,
        seed=2026,
        task_contains="fold clothes",
    )

    assert split.train == (0, 1, 2)
    assert split.validation == ()


def test_joint16_gripper_layout_requires_exact_deploy_order(tmp_path: pathlib.Path):
    names = [*unitree_train_config.G1D_ARM_JOINT_NAMES, *unitree_train_config.G1D_GRIPPER_NAMES]
    feature = {"shape": [16], "names": [names]}
    (tmp_path / "meta").mkdir()
    info_path = tmp_path / "meta/info.json"
    info_path.write_text(json.dumps({"features": {"observation.state": feature, "action": feature}}))

    unitree_train_config._validate_joint_vector_layout(tmp_path, "gripper")  # noqa: SLF001

    feature["names"][0][-2:] = ["kRightGripper", "kLeftGripper"]
    info_path.write_text(json.dumps({"features": {"observation.state": feature, "action": feature}}))
    with pytest.raises(ValueError, match="exact observation.state order"):
        unitree_train_config._validate_joint_vector_layout(tmp_path, "gripper")  # noqa: SLF001


def test_resume_rejects_old_human_checkpoint_without_explicit_source_frame(tmp_path: pathlib.Path):
    expected = {
        "action_representation": "absolute",
        "action_domain": "brainco",
        "relative_norm": "shared",
        "robot_camera_mode": "three",
        "human_input_frame": "torso_palm",
        "task_progress_alignment": None,
        "eval_asset_id": "robot",
        "components": [
            {
                "name": "robot_mode2",
                "kind": "robot",
                "mode": 2,
                "dimension": 30,
                "camera_mode": "three",
                "model_uses_state": True,
                "action_space": "eef",
                "action_domain": "brainco",
                "asset_id": "robot",
                "split_digest": "robot-split",
                "input_frame": "torso_palm",
            },
            {
                "name": "human_mode3",
                "kind": "human",
                "mode": 3,
                "dimension": 30,
                "camera_mode": "single",
                "model_uses_state": True,
                "action_space": "eef",
                "action_domain": "brainco",
                "asset_id": "human_source_torso_palm",
                "split_digest": "human-split",
                "input_frame": "torso_palm",
            },
        ],
    }
    actual = json.loads(json.dumps(expected))
    actual.pop("human_input_frame")
    actual["components"][1].pop("input_frame")
    actual["components"][1]["asset_id"] = "human_legacy_unqualified"
    checkpoint = tmp_path / "unitree" / "run" / "100" / "assets"
    checkpoint.mkdir(parents=True)
    (checkpoint / "runtime_manifest.json").write_text(json.dumps(actual))
    config = unitree_train_config.TrainConfig(
        name="unitree",
        exp_name="run",
        checkpoint_base_dir=str(tmp_path),
        resume=True,
        policy_metadata=expected,
    )

    with pytest.raises(ValueError, match="Refusing to resume incompatible"):
        unitree_train_config.validate_resume_manifest(config)
