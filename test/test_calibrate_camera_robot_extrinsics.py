from __future__ import annotations

import math

import numpy as np
import pytest

import calibrate_camera_robot_extrinsics as calibration


def _axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)


def _transform(translation, axis=(0, 0, 1), angle=0.0):
    result = np.eye(4)
    result[:3, :3] = _axis_angle(axis, angle)
    result[:3, 3] = translation
    return result


def test_pose_matrix_round_trip_uses_xyzw():
    pose = np.array([0.1, -0.2, 0.3, 0.2, -0.1, 0.3, 0.9])
    pose[3:] /= np.linalg.norm(pose[3:])
    restored = calibration.matrix_to_pose_xyzw(calibration.pose_xyzw_to_matrix(pose))
    np.testing.assert_allclose(restored[:3], pose[:3], atol=1e-12)
    assert abs(float(np.dot(restored[3:], pose[3:]))) > 1 - 1e-12


def test_midpoint_pose_interpolates_translation_and_rotation():
    first = calibration.matrix_to_pose_xyzw(_transform([0, 0, 0], angle=0))
    second = calibration.matrix_to_pose_xyzw(_transform([2, 4, 6], angle=math.pi / 2))
    midpoint = calibration.pose_xyzw_to_matrix(calibration.midpoint_pose(first, second))
    np.testing.assert_allclose(midpoint[:3, 3], [1, 2, 3])
    assert calibration.rotation_distance_deg(np.eye(4), midpoint) == pytest.approx(45.0)


def test_board_spec_is_12_by_9_squares_and_11_by_8_corners():
    board = calibration.BoardSpec()
    assert board.pattern_size == (11, 8)
    points = calibration.board_object_points(board)
    assert points.shape == (88, 3)
    np.testing.assert_allclose(points[-1], [0.2, 0.14, 0.0])


def test_outlier_detection_rejects_large_mount_jump():
    transforms = [_transform([index * 1e-4, 0, 0]) for index in range(12)]
    transforms.append(_transform([0.08, 0, 0], axis=(1, 0, 0), angle=math.radians(12)))
    mask, limits = calibration.identify_inliers(transforms)
    assert mask[:-1].all()
    assert not mask[-1]
    assert limits["translation_m"] >= 0.005
    assert limits["rotation_deg"] >= 1.0


def test_jsonable_converts_nonfinite_metrics_to_json_null():
    assert calibration._jsonable(float("nan")) is None
    assert calibration._jsonable(np.float64("inf")) is None


def test_real_rpc_payload_shape_extracts_rgb_depth_and_timestamp():
    stamp = {"sec": 1790485581, "nanosec": 158631836}
    payload = {
        "format": "bgr8; jpeg compressed bgr8",
        "frame_id": "/hdas/camera_wrist_right_color_optical_frame",
        "stamp": stamp,
        "seq": 41581,
        "data": b"jpeg bytes",
        "depth": {
            "encoding": "16UC1",
            "frame_id": "/hdas/camera_wrist_right_color_optical_frame",
            "stamp": stamp,
            "seq": 32141,
            "width": 2,
            "height": 2,
            "step": 4,
            "is_bigendian": False,
            "data": np.array([1000, 2000, 0, 3000], dtype="<u2").tobytes(),
        },
    }
    assert calibration._packet_data(calibration._find_rgb_packet(payload)) == b"jpeg bytes"
    depth_m, metadata = calibration.decode_depth_packet(calibration._find_depth_packet(payload))
    np.testing.assert_allclose(depth_m, [[1.0, 2.0], [0.0, 3.0]])
    assert metadata["encoding"] == "16UC1"
    assert calibration._timestamp_from_payload(payload) == 1790485581158631836


def test_read_only_client_calls_only_required_rpc_methods(monkeypatch):
    class Server:
        def __init__(self):
            self.calls = []

        def get_right_ee_pose(self):
            self.calls.append("get_right_ee_pose")
            return [0, 0, 0, 0, 0, 0, 1]

        def get_right_wrist_rgbd(self):
            self.calls.append("get_right_wrist_rgbd")
            return object()

    server = Server()
    client = calibration.ReadOnlyRobotClient(server=server)
    np.testing.assert_array_equal(client.get_ee_pose_right(), [0, 0, 0, 0, 0, 0, 1])
    assert client.get_right_wrist_rgbd() is not None
    assert server.calls == ["get_right_ee_pose", "get_right_wrist_rgbd"]
    assert not hasattr(client, "move_ee_right")


def test_synthetic_eye_to_hand_recovers_control_camera():
    cv2 = pytest.importorskip("cv2")
    if not hasattr(cv2, "calibrateHandEye"):
        pytest.skip("OpenCV build lacks calibrateHandEye")
    rng = np.random.default_rng(2026)
    T_control_camera = _transform([0.42, -0.18, 0.63], axis=(0.3, 0.8, -0.2), angle=0.7)
    T_tcp_board = _transform([0.03, 0.01, 0.12], axis=(0.7, -0.1, 0.5), angle=-0.4)
    samples = []
    for index in range(30):
        axis = rng.normal(size=3)
        T_control_tcp = _transform(rng.uniform([-0.2, -0.3, 0.1], [0.5, 0.3, 0.8]), axis, rng.uniform(-1.1, 1.1))
        T_camera_board = calibration.invert_transform(T_control_camera) @ T_control_tcp @ T_tcp_board
        samples.append(
            {
                "sample_id": f"{index:04d}",
                "T_control_tcp": T_control_tcp.tolist(),
                "T_camera_board": T_camera_board.tolist(),
            }
        )
    solved = calibration.solve_hand_eye(samples, cv2.CALIB_HAND_EYE_PARK)
    translation_m, rotation_deg = calibration.transform_distance(T_control_camera, solved)
    assert translation_m < 1e-6
    assert rotation_deg < 1e-5
