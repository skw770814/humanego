from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from xrpipe import (
    ACTION_COORDINATE_SYSTEM,
    ACTION_REFERENCE_FRAME,
    ACTION_SOURCE_COORDINATE_SYSTEM,
)
from xrpipe.episode import (
    UNITY_TO_OPENXR,
    build_episode_arrays,
    check_action_coordinate_conversion,
    unity_to_openxr_transforms,
)


def _pose(translation, rotation) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.asarray(rotation, dtype=np.float64)
    result[:3, 3] = np.asarray(translation, dtype=np.float64)
    return result


def test_unity_to_openxr_identity_and_translation() -> None:
    source = np.stack([
        np.eye(4),
        _pose([1.0, 2.0, 3.0], np.eye(3)),
    ])
    converted = unity_to_openxr_transforms(source)
    np.testing.assert_array_equal(converted[0], np.eye(4))
    np.testing.assert_array_equal(converted[1, :3, 3], [1.0, 2.0, -3.0])
    np.testing.assert_array_equal(converted[1, :3, :3], np.eye(3))
    assert check_action_coordinate_conversion(source, converted) == 0.0


def test_unity_to_openxr_random_rigid_poses_are_involutive() -> None:
    rng = np.random.default_rng(20260925)
    rotations = Rotation.random(32, random_state=rng).as_matrix()
    translations = rng.normal(size=(32, 3))
    source = np.stack([_pose(t, r) for t, r in zip(translations, rotations, strict=True)])
    converted = unity_to_openxr_transforms(source)
    restored = unity_to_openxr_transforms(converted)
    np.testing.assert_allclose(restored, source, atol=1e-12)
    np.testing.assert_allclose(
        np.swapaxes(converted[:, :3, :3], 1, 2) @ converted[:, :3, :3],
        np.broadcast_to(np.eye(3), (len(converted), 3, 3)),
        atol=1e-12,
    )
    np.testing.assert_allclose(np.linalg.det(converted[:, :3, :3]), 1.0, atol=1e-12)


def test_relative_motion_is_conjugated_and_magnitudes_are_preserved() -> None:
    a = _pose([0.1, -0.2, 0.3], Rotation.from_euler("xyz", [10, -20, 30], degrees=True).as_matrix())
    b = _pose([-0.4, 0.5, 0.6], Rotation.from_euler("xyz", [-5, 15, 40], degrees=True).as_matrix())
    a_rh, b_rh = unity_to_openxr_transforms(np.stack([a, b]))
    relative_lh = np.linalg.inv(a) @ b
    relative_rh = np.linalg.inv(a_rh) @ b_rh
    change = np.eye(4)
    change[:3, :3] = UNITY_TO_OPENXR
    np.testing.assert_allclose(relative_rh, change @ relative_lh @ change, atol=1e-12)
    np.testing.assert_allclose(
        np.linalg.norm(relative_rh[:3, 3]), np.linalg.norm(relative_lh[:3, 3]), atol=1e-12
    )


def test_episode_action_uses_openxr_pose_but_state_is_unchanged() -> None:
    hand_camera = np.repeat(np.eye(4)[None], 3, axis=0)
    objects_camera = np.repeat(np.eye(4)[None, None], 3, axis=0)
    objects_camera[:, 0, :3, 3] = [[0.1, 0.2, 0.3], [0.2, 0.2, 0.3], [0.3, 0.2, 0.3]]
    closed = np.array([False, False, True])
    world_lh = np.stack([
        _pose([0.0, 0.0, 1.0], np.eye(3)),
        _pose([0.1, 0.0, 2.0], Rotation.from_euler("y", 10, degrees=True).as_matrix()),
        _pose([0.2, 0.0, 3.0], Rotation.from_euler("y", 20, degrees=True).as_matrix()),
    ])
    world_rh = unity_to_openxr_transforms(world_lh)
    index = np.array([0, 2])
    converted = build_episode_arrays(hand_camera, objects_camera, closed, index, world_rh)
    raw = build_episode_arrays(hand_camera, objects_camera, closed, index, world_lh)

    np.testing.assert_array_equal(converted["state"], raw["state"])
    np.testing.assert_array_equal(converted["action"][0, :9], converted["reference"][1])
    np.testing.assert_array_equal(converted["action"][-1, :9], converted["reference"][-1])
    np.testing.assert_array_equal(converted["reference"][:, 2], [-1.0, -3.0])
    np.testing.assert_array_equal(converted["action"][:, 9], [1.0, 1.0])


def test_coordinate_contract_and_pipeline_source_is_self_contained() -> None:
    assert ACTION_REFERENCE_FRAME == "pico_world_openxr"
    assert ACTION_COORDINATE_SYSTEM == "openxr_rh_x_right_y_up_z_back"
    assert ACTION_SOURCE_COORDINATE_SYSTEM == "unity_lh_x_right_y_up_z_forward"
    root = Path(__file__).resolve().parents[1]
    for path in root.rglob("*"):
        if path.suffix not in {".py", ".sh", ".md"} or not path.is_file():
            continue
        assert ("test" + "/work") not in path.read_text(encoding="utf-8"), path


def test_export_info_declares_openxr_action_coordinates() -> None:
    import pytest
    pytest.importorskip("cv2", reason="export module imports the media writer")
    from xrpipe.export import info_dict

    info = info_dict(
        features={},
        n_episodes=1,
        total_frames=2,
        tasks=["task"],
        robot_type="test",
        fps=30.0,
        codec="h264",
        variant="binary",
        object_order=["obj1"],
        object_categories=["object"],
        latched_frames=0,
        max_invalid_gap=30,
        visual_source="pipeline_step2_raw",
    )
    contract = info["ego_relation"]
    assert contract["action_reference_frame"] == "pico_world_openxr"
    assert contract["action_coordinate_system"] == "openxr_rh_x_right_y_up_z_back"
    assert contract["action_source_coordinate_system"] == "unity_lh_x_right_y_up_z_forward"
    assert contract["action_coordinate_transform"]["matrix"] == [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, -1],
    ]
