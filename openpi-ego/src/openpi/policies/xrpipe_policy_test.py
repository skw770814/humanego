import numpy as np

from openpi import transforms
from openpi.policies import xrpipe_policy as policy


def _pose(position=(0.0, 0.0, 0.0), rotation=None) -> np.ndarray:
    rotation = np.eye(3, dtype=np.float32) if rotation is None else np.asarray(rotation, dtype=np.float32)
    return np.concatenate((np.asarray(position, dtype=np.float32), rotation[:, 0], rotation[:, 1]))


def _rotation_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1
    return rotation.astype(np.float32)


def test_relation_state_basis_change_is_involutive_and_preserves_gripper():
    first = _pose((0.1, -0.2, 0.3), _rotation_z(0.4))
    second = _pose((-0.4, 0.5, -0.6), _rotation_z(-0.2))
    state = np.concatenate((first, second, [1.0])).astype(np.float32)
    transform = policy.CanonicalizeRelationState(19)

    converted = transform({"state": state.copy()})["state"]
    restored = transform({"state": converted.copy()})["state"]

    np.testing.assert_allclose(converted[:3], [0.1, -0.2, -0.3], atol=1e-6)
    np.testing.assert_allclose(converted[9:12], [-0.4, 0.5, 0.6], atol=1e-6)
    np.testing.assert_allclose(restored, state, atol=1e-6)
    assert converted[-1] == 1.0
    for start in (0, 9):
        rotation = policy.rot6d_grouped_to_matrix(converted[start + 3 : start + 9])
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-6)


def test_relative_action_uses_separate_reference_and_preserves_gripper():
    reference = _pose((1.0, 2.0, 3.0), _rotation_z(np.pi / 2))
    actions = np.stack(
        (
            np.concatenate((_pose((1.0, 3.0, 3.0), _rotation_z(np.pi / 2)), [0.0])),
            np.concatenate((_pose((0.0, 2.0, 3.0), _rotation_z(np.pi)), [1.0])),
        )
    ).astype(np.float32)

    result = policy.RelativeFingertipActions()(
        {"state": np.zeros(19), "action_reference": reference, "actions": actions.copy()}
    )["actions"]

    np.testing.assert_allclose(result[0, :3], [1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(result[1, :3], [0.0, 1.0, 0.0], atol=1e-6)
    np.testing.assert_array_equal(result[:, 9], actions[:, 9])
    expected_rotation = _rotation_z(np.pi / 2).T @ _rotation_z(np.pi)
    np.testing.assert_allclose(policy.rot6d_grouped_to_matrix(result[1, 3:9]), expected_rotation, atol=1e-6)


def test_identity_terminal_action_is_not_encoded_as_six_zeros():
    reference = _pose((0.2, -0.1, 0.7), _rotation_z(0.3))
    action = np.concatenate((reference, [1.0]))[None].astype(np.float32)

    result = policy.RelativeFingertipActions()({"action_reference": reference, "actions": action})["actions"][0]

    np.testing.assert_allclose(result[:3], 0.0, atol=1e-6)
    np.testing.assert_allclose(result[3:9], [1, 0, 0, 0, 1, 0], atol=1e-6)
    assert result[9] == 1.0


def test_random_basis_and_relative_transforms_match_matrix_algebra():
    rng = np.random.default_rng(2026)
    reflection = np.diag([1.0, 1.0, -1.0])
    for _ in range(32):
        reference_rotation = _random_rotation(rng)
        target_rotation = _random_rotation(rng)
        reference = _pose(rng.normal(size=3), reference_rotation)
        target = _pose(rng.normal(size=3), target_rotation)

        converted = policy.change_pose9_basis(target)
        np.testing.assert_allclose(converted[:3], reflection @ target[:3], atol=2e-6)
        np.testing.assert_allclose(
            policy.rot6d_grouped_to_matrix(converted[3:9]),
            reflection @ target_rotation @ reflection,
            atol=2e-6,
        )
        np.testing.assert_allclose(policy.change_pose9_basis(converted), target, atol=2e-6)

        relative = policy.relative_pose9(target, reference)
        np.testing.assert_allclose(relative[:3], reference_rotation.T @ (target[:3] - reference[:3]), atol=2e-6)
        np.testing.assert_allclose(
            policy.rot6d_grouped_to_matrix(relative[3:9]),
            reference_rotation.T @ target_rotation,
            atol=2e-6,
        )


def test_action_chunk_starts_at_next_frame_and_terminal_row_is_identity():
    references = np.stack([_pose((float(index), 0.0, 0.0)) for index in range(4)])
    actions = np.concatenate(
        (
            np.concatenate((references[1:], references[-1:]), axis=0),
            np.asarray([[0.0], [1.0], [1.0], [1.0]], dtype=np.float32),
        ),
        axis=-1,
    )
    transform = policy.RelativeFingertipActions()

    middle = transform({"action_reference": references[1], "actions": actions[1:].copy()})["actions"]
    terminal = transform({"action_reference": references[-1], "actions": actions[-1:].copy()})["actions"]

    np.testing.assert_allclose(middle[:, 0], [1.0, 2.0, 2.0], atol=1e-6)
    np.testing.assert_allclose(terminal[0, :3], 0.0, atol=1e-6)
    np.testing.assert_allclose(terminal[0, 3:9], [1, 0, 0, 0, 1, 0], atol=1e-6)
    assert terminal[0, 9] == 1.0


def test_xrpipe_inputs_and_model_padding_keep_only_ten_action_dimensions():
    raw = {
        "images": {"camera0": np.zeros((12, 16, 3), dtype=np.uint8)},
        "state": np.arange(19, dtype=np.float32),
        "actions": np.zeros((4, 10), dtype=np.float32),
        "action_is_pad": np.asarray([False, False, False, True]),
        "prompt": "pick up object one",
    }
    inputs = policy.XRPipeInputs(state_dim=19)(raw)
    padded = transforms.PadStatesAndActions(32, mask_padded_action_dims=True)(inputs)

    assert padded["state"].shape == (32,)
    assert padded["actions"].shape == (4, 32)
    assert np.all(padded["action_dim_mask"][:10])
    assert not np.any(padded["action_dim_mask"][10:])
    np.testing.assert_array_equal(padded["action_pad_mask"], raw["action_is_pad"])
    assert bool(padded["image_mask"]["base_0_rgb"])
    assert not bool(padded["image_mask"]["left_wrist_0_rgb"])
