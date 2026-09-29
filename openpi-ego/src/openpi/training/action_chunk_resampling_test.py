import numpy as np

from openpi.training.action_chunk_resampling import required_source_horizon
from openpi.training.action_chunk_resampling import resample_absolute_eef_actions


def _eef30(left_rotation: np.ndarray, value: float) -> np.ndarray:
    action = np.full(30, value, dtype=np.float32)
    action[3:9] = left_rotation
    action[12:18] = left_rotation
    return action


def test_resample_absolute_eef_actions_uses_linear_and_slerp_interpolation():
    identity = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)
    z_90 = np.asarray([0, 1, 0, -1, 0, 0], dtype=np.float32)
    source = np.stack((_eef30(identity, 0.0), _eef30(z_90, 2.0)))
    source[:, 18:] = np.asarray((0.0, 1.0))[:, None]

    output, is_pad = resample_absolute_eef_actions(
        source,
        np.zeros(2, dtype=np.bool_),
        output_horizon=3,
        source_step_scale=0.5,
    )

    assert required_source_horizon(3, 0.5) == 2
    np.testing.assert_allclose(output[:, 0], [0.0, 1.0, 2.0], atol=1e-6)
    np.testing.assert_allclose(output[:, 18], [0.0, 0.5, 1.0], atol=1e-6)
    root_half = np.sqrt(0.5)
    np.testing.assert_allclose(
        output[1, 3:9],
        [root_half, root_half, 0.0, -root_half, root_half, 0.0],
        atol=1e-6,
    )
    np.testing.assert_array_equal(is_pad, [False, False, False])


def test_resample_absolute_eef_actions_recomputes_padding_at_fractional_offsets():
    identity = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)
    source = np.stack((_eef30(identity, 0.0), _eef30(identity, 0.0)))

    _, is_pad = resample_absolute_eef_actions(
        source,
        np.asarray([False, True]),
        output_horizon=3,
        source_step_scale=0.5,
    )

    np.testing.assert_array_equal(is_pad, [False, True, True])


def test_resample_absolute_eef_actions_accepts_eef18_without_a_tail():
    identity = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)
    source = np.stack((_eef30(identity, 0.0)[:18], _eef30(identity, 2.0)[:18]))

    output, is_pad = resample_absolute_eef_actions(
        source,
        np.zeros(2, dtype=np.bool_),
        output_horizon=3,
        source_step_scale=0.5,
    )

    assert output.shape == (3, 18)
    np.testing.assert_allclose(output[:, 0], [0.0, 1.0, 2.0], atol=1e-6)
    np.testing.assert_array_equal(is_pad, [False, False, False])


def test_resample_absolute_eef_actions_supports_source_step_scale_above_one():
    identity = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)
    source = np.stack([_eef30(identity, float(index))[:18] for index in range(4)])

    output, is_pad = resample_absolute_eef_actions(
        source,
        np.zeros(4, dtype=np.bool_),
        output_horizon=3,
        source_step_scale=1.5,
    )

    assert required_source_horizon(3, 1.5) == 4
    np.testing.assert_allclose(output[:, 0], [0.0, 1.5, 3.0], atol=1e-6)
    np.testing.assert_array_equal(is_pad, [False, False, False])
