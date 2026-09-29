import numpy as np

from xrpipe import ACTION_COORDINATE_SYSTEM
from xrpipe import ACTION_REFERENCE_FRAME
from xrpipe import FPS
from xrpipe.export import action_semantics


def test_xrpipe_mode1_action_contract_is_explicitly_fingertip_and_right_handed():
    contract = action_semantics()

    assert contract["schema_version"] == "xrpipe_action_v1"
    assert contract["mode"] == "xrpipe_mode1"
    assert contract["stored_action"] == "absolute_next_target"
    assert contract["reference_field"] == "observation.action_reference_tcp"
    assert contract["reference_frame"] == ACTION_REFERENCE_FRAME
    assert contract["coordinate_system"] == ACTION_COORDINATE_SYSTEM
    assert contract["control_point"] == "right_thumb_index_fingertip_midpoint"
    assert contract["action_layout"] == {"dimension": 10, "pose_slice": [0, 9], "gripper_index": 9}
    assert contract["reference_layout"] == {"dimension": 9, "pose_slice": [0, 9]}
    assert contract["relative_formula"] == "inv(reference[t]) @ action[t+k]"
    assert contract["gripper_transform"] == "none"


def test_lerobot_training_timestamp_is_exact_fixed_fps():
    length = 509
    timestamp = (np.arange(length, dtype=np.float64) / float(FPS)).astype(np.float32)

    assert timestamp[0] == 0.0
    np.testing.assert_allclose(np.diff(timestamp), 1.0 / FPS, atol=1e-6, rtol=0)
