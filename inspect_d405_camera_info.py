#!/usr/bin/env python3
"""Print one live D405 CameraInfo message and compare it with calibration defaults.

Run this with the robot's ROS 2 Python, not necessarily the OpenPI ``uv``
environment::

    source ~/galaxea/install/setup.bash
    python3 inspect_d405_camera_info.py

Use ``ros2 topic list | grep camera_info`` and pass ``--topic`` if the default
topic is not present.
"""

from __future__ import annotations

import argparse
import json
import math
import sys


EXPECTED_WIDTH = 640
EXPECTED_HEIGHT = 480
EXPECTED_K = [
    432.781433,
    0.0,
    322.661163,
    0.0,
    432.279083,
    239.838882,
    0.0,
    0.0,
    1.0,
]
EXPECTED_D = [-0.0533524305, 0.0595001988, -0.0002590606, 0.0002516542, -0.0193552151]


def differences(actual: list[float], expected: list[float]) -> dict[str, object]:
    if len(actual) != len(expected):
        return {"same_length": False, "actual_length": len(actual), "expected_length": len(expected)}
    delta = [float(a) - float(b) for a, b in zip(actual, expected, strict=True)]
    return {
        "same_length": True,
        "max_abs": max((abs(value) for value in delta), default=0.0),
        "delta": delta,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topic",
        default="/hdas/camera_wrist_right/color/camera_info",
        help="sensor_msgs/msg/CameraInfo topic",
    )
    parser.add_argument("--timeout", type=float, default=10.0, help="seconds to wait for one message")
    args = parser.parse_args()

    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import CameraInfo
    except ImportError as exc:
        print(
            "ERROR: ROS 2 Python packages are unavailable. Source the robot ROS setup before running this script.",
            file=sys.stderr,
        )
        print(f"DETAIL: {exc}", file=sys.stderr)
        return 2

    result = None

    class CameraInfoReader(Node):
        def __init__(self) -> None:
            super().__init__("inspect_d405_camera_info")
            qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
            )
            self.subscription = self.create_subscription(CameraInfo, args.topic, self.callback, qos)

        def callback(self, message: CameraInfo) -> None:
            nonlocal result
            result = message

    rclpy.init()
    node = CameraInfoReader()
    deadline = node.get_clock().now().nanoseconds + int(args.timeout * 1e9)
    try:
        while rclpy.ok() and result is None and node.get_clock().now().nanoseconds < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if result is None:
        print(f"ERROR: no CameraInfo received from {args.topic!r} within {args.timeout:g}s", file=sys.stderr)
        print("Run: ros2 topic list | grep -E 'camera_wrist_right.*camera_info'", file=sys.stderr)
        return 1

    message = result
    actual_k = [float(value) for value in message.k]
    actual_d = [float(value) for value in message.d]
    k_diff = differences(actual_k, EXPECTED_K)
    d_diff = differences(actual_d, EXPECTED_D)
    payload = {
        "topic": args.topic,
        "header": {
            "frame_id": message.header.frame_id,
            "stamp": {"sec": message.header.stamp.sec, "nanosec": message.header.stamp.nanosec},
        },
        "width": int(message.width),
        "height": int(message.height),
        "distortion_model": message.distortion_model,
        "D": actual_d,
        "K": actual_k,
        "R": [float(value) for value in message.r],
        "P": [float(value) for value in message.p],
        "binning_x": int(message.binning_x),
        "binning_y": int(message.binning_y),
        "roi": {
            "x_offset": int(message.roi.x_offset),
            "y_offset": int(message.roi.y_offset),
            "height": int(message.roi.height),
            "width": int(message.roi.width),
            "do_rectify": bool(message.roi.do_rectify),
        },
        "comparison_with_calibration_script": {
            "expected_width": EXPECTED_WIDTH,
            "expected_height": EXPECTED_HEIGHT,
            "resolution_matches": message.width == EXPECTED_WIDTH and message.height == EXPECTED_HEIGHT,
            "K": k_diff,
            "D": d_diff,
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    warnings = []
    if not payload["comparison_with_calibration_script"]["resolution_matches"]:
        warnings.append("image resolution differs from the calibration script")
    if not k_diff.get("same_length") or not math.isclose(float(k_diff.get("max_abs", math.inf)), 0.0, abs_tol=1e-5):
        warnings.append("K differs from the calibration script")
    if not d_diff.get("same_length") or not math.isclose(float(d_diff.get("max_abs", math.inf)), 0.0, abs_tol=1e-6):
        warnings.append("D differs from the calibration script")
    if message.distortion_model not in ("plumb_bob", "rational_polynomial"):
        warnings.append(f"unexpected distortion model: {message.distortion_model!r}")

    if warnings:
        print("\nCHECK RESULT: MISMATCH", file=sys.stderr)
        for warning in warnings:
            print(f"  - {warning}", file=sys.stderr)
        return 1
    print("\nCHECK RESULT: camera_info matches the calibration script defaults")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
