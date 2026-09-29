import numpy as np
import pytest

from openpi import transforms
from openpi.policies import human_policy
from openpi.policies import unitree_robot_policy as policy


def _pose(position: list[float]) -> np.ndarray:
    # Identity rotation in grouped-column layout.
    return np.asarray([*position, 1, 0, 0, 0, 1, 0], dtype=np.float32)


def test_human_frame_canonicalization_and_relative_round_trip():
    state = np.concatenate((_pose([0.2, 0.3, 0.4]), _pose([-0.1, 0.2, 0.5]), np.linspace(0, 1, 12))).astype(np.float32)
    actions = np.stack([state + index * 0.001 for index in range(4)])
    # Keep Rot6D valid and change only position/BrainCo values.
    actions[:, 3:9] = state[3:9]
    actions[:, 12:18] = state[12:18]

    canonical = policy.CanonicalizeHumanEEF(
        input_frame="g1_base_tcp"
    )({"state": state.copy(), "actions": actions.copy()})
    np.testing.assert_allclose(
        canonical["state"][:3], state[:3] - [-0.0039635, 0, 0.044] + [0.0415, 0.003, 0], atol=1e-7
    )
    np.testing.assert_allclose(
        canonical["state"][9:12], state[9:12] - [-0.0039635, 0, 0.044] + [0.0415, -0.003, 0], atol=1e-7
    )

    relative = policy.RelativeEEFActions(30, "columns_grouped")(
        {"state": canonical["state"].copy(), "actions": canonical["actions"].copy()}
    )
    restored = policy.AbsoluteEEFActions(30, "columns_grouped")(
        {"state": canonical["state"].copy(), "actions": relative["actions"].copy()}
    )
    np.testing.assert_allclose(restored["actions"], canonical["actions"], atol=2e-6)
    np.testing.assert_array_equal(relative["actions"][:, 18:], canonical["actions"][:, 18:])


def test_native_human_positions_are_not_translated_again():
    state = np.concatenate((_pose([0.2, 0.3, 0.4]), _pose([-0.1, 0.2, 0.5]), np.linspace(0, 1, 12))).astype(np.float32)
    actions = np.stack((state, state))

    canonical = policy.CanonicalizeHumanEEF(
        action_dim=30,
        input_frame="native",
    )({"state": state.copy(), "actions": actions.copy()})

    np.testing.assert_allclose(canonical["state"][:3], state[:3], atol=1e-7)
    np.testing.assert_allclose(canonical["state"][9:12], state[9:12], atol=1e-7)
    np.testing.assert_allclose(canonical["actions"][..., :3], actions[..., :3], atol=1e-7)
    np.testing.assert_allclose(canonical["actions"][..., 9:12], actions[..., 9:12], atol=1e-7)
    np.testing.assert_array_equal(canonical["state"][18:], state[18:])


def test_default_human_profile_restores_mode1_robot_link_canonicalization():
    state = np.concatenate((_pose([0.2, 0.3, 0.4]), _pose([-0.1, 0.2, 0.5]))).astype(np.float32)

    default = policy.CanonicalizeHumanEEF(action_dim=18)({"state": state.copy()})
    explicit = policy.CanonicalizeHumanEEF(
        action_dim=18,
        input_frame="g1_base_tcp",
    )({"state": state.copy()})

    np.testing.assert_allclose(default["state"], explicit["state"], atol=1e-7)


def test_recording_tcp_only_moves_the_control_point_to_palm():
    state = np.concatenate((_pose([0.2, 0.3, 0.4]), _pose([-0.1, 0.2, 0.5]))).astype(np.float32)

    canonical = policy.CanonicalizeHumanEEF(
        action_dim=18,
        input_frame="recording_tcp",
    )({"state": state.copy()})

    np.testing.assert_allclose(canonical["state"][:3], state[:3] + [0.0415, 0.003, 0], atol=1e-7)
    np.testing.assert_allclose(canonical["state"][9:12], state[9:12] + [0.0415, -0.003, 0], atol=1e-7)


def test_eef20_gripper_relative_round_trip_preserves_absolute_grippers():
    state = np.concatenate((_pose([0.2, 0.3, 0.4]), _pose([-0.1, 0.2, 0.5]), [0.25, 0.75])).astype(np.float32)
    actions = np.stack([state.copy() for _ in range(4)])
    actions[:, :3] += np.arange(4, dtype=np.float32)[:, None] * 0.01
    actions[:, 9:12] -= np.arange(4, dtype=np.float32)[:, None] * 0.02
    actions[:, 18:] = [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6]]

    relative = policy.RelativeEEFActions(20, "columns_grouped")({"state": state, "actions": actions})
    restored = policy.AbsoluteEEFActions(20, "columns_grouped")(
        {
            "state": np.zeros(32, dtype=np.float32),
            "action_reference_state": np.pad(state, (0, 12)),
            "actions": np.pad(relative["actions"], ((0, 0), (0, 12))),
        }
    )

    np.testing.assert_allclose(restored["actions"], actions, atol=2e-6)
    np.testing.assert_array_equal(relative["actions"][:, 18:20], actions[:, 18:20])


def test_eef18_only_human_canonicalization_and_relative_round_trip():
    state = np.concatenate((_pose([0.2, 0.3, 0.4]), _pose([-0.1, 0.2, 0.5]))).astype(np.float32)
    actions = np.stack([state.copy() for _ in range(4)])
    actions[:, :3] += np.arange(4, dtype=np.float32)[:, None] * 0.01
    actions[:, 9:12] -= np.arange(4, dtype=np.float32)[:, None] * 0.02

    canonical = policy.CanonicalizeHumanEEF(18)({"state": state, "actions": actions})
    relative = policy.RelativeEEFActions(18, "columns_grouped")(canonical.copy())
    restored = policy.AbsoluteEEFActions(18, "columns_grouped")(
        {"state": canonical["state"], "actions": relative["actions"]}
    )

    assert canonical["state"].shape == (18,)
    assert canonical["actions"].shape == (4, 18)
    np.testing.assert_allclose(restored["actions"], canonical["actions"], atol=2e-6)


def test_eef18_only_human_actions_pad_to_model_width_with_tail_loss_masked():
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    human = human_policy.HumanInputs(18, "single", model_uses_state=True)(
        {
            "images": {"cam_high": image},
            "state": np.zeros(18, dtype=np.float32),
            "actions": np.zeros((5, 18), dtype=np.float32),
        }
    )
    padded = transforms.PadStatesAndActions(32, mask_padded_action_dims=True)(human)

    assert padded["state"].shape == (32,)
    assert padded["actions"].shape == (5, 32)
    np.testing.assert_array_equal(
        padded["action_dim_mask"],
        np.concatenate((np.ones(18, dtype=np.bool_), np.zeros(14, dtype=np.bool_))),
    )


def test_eef_only_human_projection_discards_recorded_gripper_state_and_actions():
    source_state = np.arange(20, dtype=np.float32)
    source_actions = np.stack((source_state, source_state + 100))

    projected = human_policy.SelectHumanEEFOnly(20)({"state": source_state.copy(), "actions": source_actions.copy()})

    np.testing.assert_array_equal(projected["state"], source_state[:18])
    np.testing.assert_array_equal(projected["actions"], source_actions[:, :18])
    assert projected["state"].shape == (18,)
    assert projected["actions"].shape == (2, 18)


def test_missing_human_wrist_cameras_are_masked_not_duplicated():
    image = np.full((8, 10, 3), 127, dtype=np.uint8)
    transform = human_policy.HumanInputs(
        state_dim=30,
        camera_mode="three",
        model_uses_state=True,
        allow_missing_cameras=True,
    )
    result = transform(
        {
            "images": {"cam_high": image},
            "state": np.zeros(30, dtype=np.float32),
            "actions": np.zeros((5, 30), dtype=np.float32),
            "action_is_pad": np.asarray([False, False, False, True, True]),
        }
    )
    assert result["image_mask"] == {
        "base_0_rgb": np.True_,
        "left_wrist_0_rgb": np.False_,
        "right_wrist_0_rgb": np.False_,
    }
    assert np.count_nonzero(result["image"]["left_wrist_0_rgb"]) == 0
    np.testing.assert_array_equal(result["action_pad_mask"], [False, False, False, True, True])


def test_single_camera_robot_masks_zero_wrist_slots():
    image = np.full((8, 10, 3), 127, dtype=np.uint8)
    transform = policy.UnitreeRobotInputs(state_dim=26, camera_mode="single")

    result = transform(
        {
            "images": {"cam_high": image},
            "state": np.zeros(26, dtype=np.float32),
            "actions": np.zeros((5, 26), dtype=np.float32),
        }
    )

    assert result["image_mask"] == {
        "base_0_rgb": np.True_,
        "left_wrist_0_rgb": np.False_,
        "right_wrist_0_rgb": np.False_,
    }
    np.testing.assert_array_equal(result["image"]["base_0_rgb"], image)
    assert np.count_nonzero(result["image"]["left_wrist_0_rgb"]) == 0
    assert np.count_nonzero(result["image"]["right_wrist_0_rgb"]) == 0


def test_robot_camera_transform_rejects_unknown_mode():
    with pytest.raises(ValueError, match="camera_mode must be single or three"):
        policy._camera_inputs(  # type: ignore[arg-type]  # noqa: SLF001
            {"cam_high": np.zeros((8, 10, 3), dtype=np.uint8)},
            "typo",
        )
