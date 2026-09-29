#!/usr/bin/env python3
"""Eye-to-hand calibration for a fixed D405 and the XinHai right arm.

The script is deliberately independent from the policy client.  It only reads
``get_right_wrist_rgbd`` and ``get_right_ee_pose`` from the robot RPC server.

Examples::

    python calibrate_camera_robot_extrinsics.py collect --output calibration_runs/d405_01
    python calibrate_camera_robot_extrinsics.py solve --run-dir calibration_runs/d405_01
    python calibrate_camera_robot_extrinsics.py validate --run-dir calibration_runs/d405_01

Transform notation follows ``T_A_B``: it maps coordinates in B into A.
The solved equation is::

    T_B_E[i] @ T_E_P == T_B_C @ T_C_P[i]

where B=torso_link4, E=right gripper TCP, C=RGB optical frame, and
P=checkerboard.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


SCHEMA_VERSION = "xinhai_eye_to_hand_v1"
DEFAULT_K = np.array(
    [[432.781433, 0.0, 322.661163], [0.0, 432.279083, 239.838882], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
DEFAULT_D = np.array(
    [-0.0533524305, 0.0595001988, -0.0002590606, 0.0002516542, -0.0193552151],
    dtype=np.float64,
)
DEFAULT_IMAGE_SIZE = (640, 480)  # width, height


@dataclass(frozen=True)
class BoardSpec:
    squares_x: int = 12
    squares_y: int = 9
    square_size_m: float = 0.020

    @property
    def corners_x(self) -> int:
        return self.squares_x - 1

    @property
    def corners_y(self) -> int:
        return self.squares_y - 1

    @property
    def pattern_size(self) -> tuple[int, int]:
        return self.corners_x, self.corners_y

    def validate(self) -> None:
        if self.squares_x < 3 or self.squares_y < 3:
            raise ValueError("checkerboard must have at least 3x3 squares")
        if not math.isfinite(self.square_size_m) or self.square_size_m <= 0:
            raise ValueError("square_size_m must be a positive finite value")


@dataclass(frozen=True)
class CameraModel:
    K: np.ndarray
    D: np.ndarray
    image_size: tuple[int, int]

    def validate(self) -> None:
        if self.K.shape != (3, 3) or not np.isfinite(self.K).all():
            raise ValueError("K must be a finite 3x3 matrix")
        if self.D.ndim != 1 or len(self.D) not in (4, 5, 8, 12, 14) or not np.isfinite(self.D).all():
            raise ValueError("D must contain 4, 5, 8, 12, or 14 finite coefficients")
        if self.image_size[0] <= 0 or self.image_size[1] <= 0:
            raise ValueError("image_size must be positive")


@dataclass
class Detection:
    corners: np.ndarray
    T_camera_board: np.ndarray
    reprojection_rms_px: float
    corner_margin_px: float
    reversed_order: bool


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on deployment environment
        raise RuntimeError(
            "OpenCV is required. Run this script in the robot/OpenPI environment "
            "with opencv-python>=4.10 installed."
        ) from exc
    required = ("findChessboardCornersSB", "calibrateHandEye", "solvePnPGeneric")
    missing = [name for name in required if not hasattr(cv2, name)]
    if missing:
        raise RuntimeError(f"OpenCV {cv2.__version__} is missing required APIs: {', '.join(missing)}")
    return cv2


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return data


def normalize_quaternion_xyzw(quaternion: Sequence[float]) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64).reshape(4)
    if not np.isfinite(q).all():
        raise ValueError("quaternion contains a non-finite value")
    norm = float(np.linalg.norm(q))
    if norm < 1e-9:
        raise ValueError("quaternion norm is zero")
    return q / norm


def quaternion_xyzw_to_rotation(quaternion: Sequence[float]) -> np.ndarray:
    x, y, z, w = normalize_quaternion_xyzw(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_to_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(R))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.array([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s])
    else:
        index = int(np.argmax(np.diag(R)))
        if index == 0:
            s = math.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) * 2.0
            q = np.array([0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s])
        elif index == 1:
            s = math.sqrt(max(0.0, 1.0 + R[1, 1] - R[0, 0] - R[2, 2])) * 2.0
            q = np.array([(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s])
        else:
            s = math.sqrt(max(0.0, 1.0 + R[2, 2] - R[0, 0] - R[1, 1])) * 2.0
            q = np.array([(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s])
    return normalize_quaternion_xyzw(q)


def quaternion_slerp_xyzw(first: Sequence[float], second: Sequence[float], fraction: float) -> np.ndarray:
    q0 = normalize_quaternion_xyzw(first)
    q1 = normalize_quaternion_xyzw(second)
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return normalize_quaternion_xyzw(q0 + fraction * (q1 - q0))
    theta = math.acos(dot)
    return normalize_quaternion_xyzw(
        math.sin((1.0 - fraction) * theta) / math.sin(theta) * q0
        + math.sin(fraction * theta) / math.sin(theta) * q1
    )


def pose_xyzw_to_matrix(pose: Sequence[float]) -> np.ndarray:
    values = np.asarray(pose, dtype=np.float64).reshape(-1)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError("TCP pose must contain 7 finite values [x,y,z,qx,qy,qz,qw]")
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quaternion_xyzw_to_rotation(values[3:7])
    T[:3, 3] = values[:3]
    return T


def matrix_to_pose_xyzw(transform: np.ndarray) -> np.ndarray:
    T = validate_transform(transform)
    return np.concatenate([T[:3, 3], rotation_to_quaternion_xyzw(T[:3, :3])])


def validate_transform(transform: np.ndarray, *, name: str = "transform") -> np.ndarray:
    T = np.asarray(transform, dtype=np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=2e-5) or not np.isclose(np.linalg.det(R), 1.0, atol=2e-5):
        raise ValueError(f"{name} has an invalid rotation")
    return T


def invert_transform(transform: np.ndarray) -> np.ndarray:
    T = validate_transform(transform)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = T[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ T[:3, 3]
    return result


def midpoint_pose(first: Sequence[float], second: Sequence[float]) -> np.ndarray:
    a = np.asarray(first, dtype=np.float64).reshape(7)
    b = np.asarray(second, dtype=np.float64).reshape(7)
    return np.concatenate([(a[:3] + b[:3]) * 0.5, quaternion_slerp_xyzw(a[3:], b[3:], 0.5)])


def rotation_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first)[:3, :3].T @ np.asarray(second)[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def transform_distance(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    translation_m = float(np.linalg.norm(np.asarray(first)[:3, 3] - np.asarray(second)[:3, 3]))
    return translation_m, rotation_distance_deg(first, second)


def board_object_points(board: BoardSpec) -> np.ndarray:
    board.validate()
    grid = np.zeros((board.corners_x * board.corners_y, 3), dtype=np.float64)
    grid[:, :2] = np.mgrid[0 : board.corners_x, 0 : board.corners_y].T.reshape(-1, 2)
    grid[:, :2] *= board.square_size_m
    return grid


def _matrix_from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    cv2 = _require_cv2()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3))[0]
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return validate_transform(T)


def _rvec_tvec_from_matrix(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cv2 = _require_cv2()
    T = validate_transform(transform)
    return cv2.Rodrigues(T[:3, :3])[0].reshape(3, 1), T[:3, 3].reshape(3, 1)


def detect_checkerboard(
    rgb: np.ndarray,
    board: BoardSpec,
    camera: CameraModel,
    *,
    reverse_order: bool = False,
) -> Detection | None:
    cv2 = _require_cv2()
    camera.validate()
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("RGB image must be HxWx3 uint8")
    height, width = image.shape[:2]
    if (width, height) != camera.image_size:
        raise ValueError(f"image is {width}x{height}, expected {camera.image_size[0]}x{camera.image_size[1]}")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    found, corners = cv2.findChessboardCornersSB(gray, board.pattern_size, flags=flags)
    if not found or corners is None or len(corners) != board.corners_x * board.corners_y:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    corners = cv2.cornerSubPix(gray, corners.astype(np.float32), (5, 5), (-1, -1), criteria).reshape(-1, 2)
    if reverse_order:
        corners = corners[::-1].copy()
    object_points = board_object_points(board)
    result = cv2.solvePnPGeneric(
        object_points,
        corners,
        camera.K,
        camera.D,
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not result[0]:
        return None
    rvecs, tvecs = result[1], result[2]
    candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
    for rvec, tvec in zip(rvecs, tvecs, strict=True):
        T = _matrix_from_rvec_tvec(rvec, tvec)
        points_camera = (T[:3, :3] @ object_points.T).T + T[:3, 3]
        if np.any(points_camera[:, 2] <= 0):
            continue
        projected = cv2.projectPoints(object_points, rvec, tvec, camera.K, camera.D)[0].reshape(-1, 2)
        rms = float(np.sqrt(np.mean(np.sum((projected - corners) ** 2, axis=1))))
        candidates.append((rms, np.asarray(rvec), np.asarray(tvec)))
    if not candidates:
        return None
    _, rvec, tvec = min(candidates, key=lambda item: item[0])
    if hasattr(cv2, "solvePnPRefineLM"):
        rvec, tvec = cv2.solvePnPRefineLM(object_points, corners, camera.K, camera.D, rvec, tvec)
    T_camera_board = _matrix_from_rvec_tvec(rvec, tvec)
    projected = cv2.projectPoints(object_points, rvec, tvec, camera.K, camera.D)[0].reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum((projected - corners) ** 2, axis=1))))
    margin = float(
        min(np.min(corners[:, 0]), np.min(corners[:, 1]), width - 1 - np.max(corners[:, 0]), height - 1 - np.max(corners[:, 1]))
    )
    return Detection(corners, T_camera_board, rms, margin, reverse_order)


def draw_detection(rgb: np.ndarray, detection: Detection | None, board: BoardSpec, camera: CameraModel) -> np.ndarray:
    cv2 = _require_cv2()
    canvas = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    if detection is None:
        cv2.putText(canvas, "NO 11x8 CHECKERBOARD", (18, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        return canvas
    cv2.drawChessboardCorners(canvas, board.pattern_size, detection.corners.reshape(-1, 1, 2), True)
    first = tuple(np.rint(detection.corners[0]).astype(int))
    cv2.circle(canvas, first, 9, (0, 0, 255), 3)
    cv2.putText(canvas, "0", (first[0] + 8, first[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    rvec, tvec = _rvec_tvec_from_matrix(detection.T_camera_board)
    cv2.drawFrameAxes(canvas, camera.K, camera.D, rvec, tvec, board.square_size_m * 3.0, 2)
    status = f"RMS {detection.reprojection_rms_px:.3f}px | reverse={detection.reversed_order}"
    cv2.putText(canvas, status, (18, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    return canvas


def _timestamp_from_payload(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return None
    for key in ("timestamp_ns", "timestamp", "time_ns", "stamp"):
        if key in payload:
            value = payload[key]
            if isinstance(value, dict) and "sec" in value:
                return int(value["sec"]) * 1_000_000_000 + int(value.get("nanosec", value.get("nsec", 0)))
            return value
    header = payload.get("header")
    if isinstance(header, dict):
        return _timestamp_from_payload(header)
    return None


def _packet_data(packet: Any) -> bytes | np.ndarray:
    if isinstance(packet, np.ndarray):
        return packet
    data = packet.get("data") if isinstance(packet, dict) else packet
    if data is None:
        keys = sorted(str(key) for key in packet) if isinstance(packet, dict) else []
        raise ValueError(f"image packet has no data field; type={type(packet).__name__}, keys={keys}")
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, list):
        return np.asarray(data, dtype=np.uint8)
    # ZeroRPC may return a bytes-compatible wrapper rather than the exact
    # built-in bytes class.  Accept the buffer protocol instead of relying on
    # a narrow isinstance check.
    try:
        return memoryview(data).tobytes()
    except TypeError as exc:
        raise ValueError(
            f"image data does not support the buffer protocol; "
            f"packet_type={type(packet).__name__}, data_type={type(data).__name__}"
        ) from exc


def _find_rgb_packet(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    for key in ("rgb", "color", "image", "rgb_image", "color_image"):
        if key in payload:
            return payload[key]
    if "data" in payload:
        return payload
    raise ValueError("RGBD payload does not contain an RGB/color packet")


def _find_depth_packet(payload: Any) -> Any | None:
    if not isinstance(payload, dict):
        return None
    for key in ("depth", "aligned_depth", "depth_image"):
        if key in payload:
            return payload[key]
    return None


def decode_rgb_packet(packet: Any) -> tuple[np.ndarray, dict[str, Any]]:
    cv2 = _require_cv2()
    if isinstance(packet, dict):
        encoding = str(packet.get("encoding", packet.get("format", ""))).lower()
        width = int(packet.get("width", 0) or 0)
        height = int(packet.get("height", 0) or 0)
        step = int(packet.get("step", 0) or 0)
    else:
        encoding, width, height, step = "", 0, 0, 0
    data = _packet_data(packet)
    array = np.asarray(data)
    if array.ndim == 3:
        image = array.astype(np.uint8, copy=False)
        if encoding.startswith("bgr") or not encoding:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif encoding.startswith("rgb"):
            rgb = image[..., :3].copy()
        else:
            raise ValueError(f"unsupported RGB array encoding: {encoding}")
    elif width and height and encoding in ("rgb8", "bgr8", "rgba8", "bgra8"):
        channels = 4 if "a8" in encoding else 3
        row_bytes = step or width * channels
        raw = np.frombuffer(array.tobytes(), dtype=np.uint8)
        if raw.size < row_bytes * height:
            raise ValueError("RGB payload is shorter than height*step")
        image = raw[: row_bytes * height].reshape(height, row_bytes)[:, : width * channels].reshape(height, width, channels)
        conversion = {
            "rgb8": None,
            "bgr8": cv2.COLOR_BGR2RGB,
            "rgba8": cv2.COLOR_RGBA2RGB,
            "bgra8": cv2.COLOR_BGRA2RGB,
        }[encoding]
        rgb = image[..., :3].copy() if conversion is None else cv2.cvtColor(image, conversion)
    else:
        encoded = np.frombuffer(array.tobytes(), dtype=np.uint8)
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"cannot decode compressed RGB payload (encoding={encoding!r})")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    metadata = {
        "encoding": encoding or "compressed/unknown",
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        "step": step or None,
        "timestamp": _timestamp_from_payload(packet),
    }
    return np.ascontiguousarray(rgb), metadata


def decode_depth_packet(packet: Any) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    if packet is None:
        return None, None
    encoding = str(packet.get("encoding", "")) if isinstance(packet, dict) else ""
    normalized = encoding.upper()
    width = int(packet.get("width", 0) or 0) if isinstance(packet, dict) else 0
    height = int(packet.get("height", 0) or 0) if isinstance(packet, dict) else 0
    step = int(packet.get("step", 0) or 0) if isinstance(packet, dict) else 0
    is_bigendian = bool(packet.get("is_bigendian", False)) if isinstance(packet, dict) else False
    data = _packet_data(packet)
    array = np.asarray(data)
    if array.ndim == 2:
        raw = array
    elif width and height and normalized in ("16UC1", "MONO16", "32FC1"):
        dtype = np.dtype(">u2" if is_bigendian else "<u2") if normalized != "32FC1" else np.dtype(">f4" if is_bigendian else "<f4")
        row_bytes = step or width * dtype.itemsize
        blob = np.frombuffer(array.tobytes(), dtype=np.uint8)
        if blob.size < row_bytes * height:
            raise ValueError("depth payload is shorter than height*step")
        rows = blob[: row_bytes * height].reshape(height, row_bytes)[:, : width * dtype.itemsize]
        raw = np.frombuffer(rows.copy().tobytes(), dtype=dtype).reshape(height, width)
    else:
        raise ValueError(f"unsupported depth packet encoding={encoding!r} shape={array.shape}")
    if normalized in ("16UC1", "MONO16") or raw.dtype.kind in "ui":
        depth_m = raw.astype(np.float32) * 0.001
        scale = 0.001
    elif normalized == "32FC1" or raw.dtype.kind == "f":
        depth_m = raw.astype(np.float32)
        scale = 1.0
    else:
        raise ValueError(f"unsupported depth dtype/encoding: {raw.dtype}, {encoding!r}")
    depth_m[~np.isfinite(depth_m) | (depth_m <= 0)] = 0
    metadata = {
        "encoding": encoding or str(raw.dtype),
        "width": int(depth_m.shape[1]),
        "height": int(depth_m.shape[0]),
        "step": step or None,
        "is_bigendian": is_bigendian,
        "scale_to_m": scale,
        "stored_unit": "m",
        "timestamp": _timestamp_from_payload(packet),
    }
    return depth_m, metadata


def decode_rgbd_payload(payload: Any) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    rgb, rgb_metadata = decode_rgb_packet(_find_rgb_packet(payload))
    depth, depth_metadata = decode_depth_packet(_find_depth_packet(payload))
    timestamp = _timestamp_from_payload(payload)
    if timestamp is None:
        timestamp = rgb_metadata.get("timestamp")
    return rgb, depth, {"timestamp": timestamp, "rgb": rgb_metadata, "depth": depth_metadata}


class ReadOnlyRobotClient:
    """Minimal read-only RPC facade; intentionally exposes no motion methods."""

    def __init__(self, ip: str = "172.16.0.30", port: int = 4242, *, server: Any | None = None):
        if server is None:
            try:
                import zerorpc  # type: ignore
            except ImportError as exc:  # pragma: no cover - hardware environment only
                raise RuntimeError("collect requires zerorpc in the robot environment") from exc
            server = zerorpc.Client(heartbeat=20)
            server.connect(f"tcp://{ip}:{port}")
        self._server = server

    def close(self) -> None:
        close = getattr(self._server, "close", None)
        if callable(close):
            close()

    def get_ee_pose_right(self) -> np.ndarray:
        return np.asarray(self._server.get_right_ee_pose(), dtype=np.float64).reshape(-1)

    def get_right_wrist_rgbd(self) -> Any:
        return self._server.get_right_wrist_rgbd()


def capture_observation(robot: ReadOnlyRobotClient) -> dict[str, Any]:
    before_time = time.monotonic_ns()
    pose_before = robot.get_ee_pose_right()
    payload = robot.get_right_wrist_rgbd()
    image_time = time.monotonic_ns()
    pose_after = robot.get_ee_pose_right()
    after_time = time.monotonic_ns()
    pose_xyzw_to_matrix(pose_before)
    pose_xyzw_to_matrix(pose_after)
    rgb, depth_m, metadata = decode_rgbd_payload(payload)
    return {
        "pose_before": pose_before,
        "pose_after": pose_after,
        "pose_midpoint": midpoint_pose(pose_before, pose_after),
        "rgb": rgb,
        "depth_m": depth_m,
        "rgbd_metadata": metadata,
        "host_monotonic_ns": {"before": before_time, "image": image_time, "after": after_time},
    }


def _load_camera(path: Path | None) -> CameraModel:
    if path is None:
        camera = CameraModel(DEFAULT_K.copy(), DEFAULT_D.copy(), DEFAULT_IMAGE_SIZE)
    else:
        payload = read_json(path)
        camera = CameraModel(
            np.asarray(payload["K"], dtype=np.float64),
            np.asarray(payload["D"], dtype=np.float64).reshape(-1),
            tuple(int(value) for value in payload["image_size"]),
        )
    camera.validate()
    return camera


def _manifest(run_dir: Path, board: BoardSpec, camera: CameraModel) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    if path.exists():
        payload = read_json(path)
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported manifest schema in {path}")
        saved_board, saved_camera = _config_from_manifest(payload)
        if saved_board != board:
            raise ValueError(
                f"board settings do not match existing run: requested {board}, saved {saved_board}"
            )
        if saved_camera.image_size != camera.image_size or not np.allclose(saved_camera.K, camera.K) or not np.allclose(
            saved_camera.D, camera.D
        ):
            raise ValueError("camera intrinsics do not match the existing run manifest")
        return payload
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "control_frame": "torso_link4",
        "tcp_frame": "right_gripper_link",
        "camera_frame": "hdas/camera_wrist_right_color_optical_frame",
        "translation_unit": "m",
        "quaternion_order": "xyzw",
        "board": asdict(board),
        "camera": {"K": camera.K, "D": camera.D, "image_size": camera.image_size},
        "samples": [],
    }
    write_json(path, payload)
    return payload


def _sample_transform(sample: dict[str, Any], key: str) -> np.ndarray:
    return validate_transform(np.asarray(sample[key], dtype=np.float64), name=key)


def _accepted_samples(run_dir: Path) -> list[dict[str, Any]]:
    manifest = read_json(run_dir / "manifest.json")
    samples = []
    for entry in manifest.get("samples", []):
        if entry.get("status", "accepted") != "accepted":
            continue
        sample = read_json(run_dir / entry["sample_json"])
        sample["_entry"] = entry
        samples.append(sample)
    return samples


def _quality_reasons(
    capture: dict[str, Any],
    detection: Detection | None,
    existing: Iterable[dict[str, Any]],
    camera: CameraModel,
) -> list[str]:
    reasons: list[str] = []
    rgb = capture["rgb"]
    if rgb.dtype != np.uint8 or rgb.shape != (camera.image_size[1], camera.image_size[0], 3):
        reasons.append(f"RGB shape/dtype is {rgb.shape}/{rgb.dtype}, expected 480x640x3 uint8")
    if capture["rgbd_metadata"].get("timestamp") is None:
        reasons.append("RGBD payload has no camera timestamp")
    before = pose_xyzw_to_matrix(capture["pose_before"])
    after = pose_xyzw_to_matrix(capture["pose_after"])
    motion_m, motion_deg = transform_distance(before, after)
    if motion_m > 0.001 or motion_deg > 0.2:
        reasons.append(f"robot moved during capture: {motion_m * 1000:.2f} mm, {motion_deg:.3f} deg")
    if detection is None:
        reasons.append("complete 11x8 inner-corner grid was not detected")
        return reasons
    if detection.reprojection_rms_px > 1.0:
        reasons.append(f"PnP reprojection RMS {detection.reprojection_rms_px:.3f}px exceeds 1px")
    if detection.corner_margin_px < 10.0:
        reasons.append(f"checkerboard corner margin {detection.corner_margin_px:.1f}px is below 10px")
    current = pose_xyzw_to_matrix(capture["pose_midpoint"])
    for sample in existing:
        previous = _sample_transform(sample, "T_control_tcp")
        distance_m, distance_deg = transform_distance(previous, current)
        if distance_m < 0.020 and distance_deg < 8.0:
            reasons.append(
                f"near-duplicate of sample {sample['sample_id']}: {distance_m * 1000:.1f} mm, {distance_deg:.2f} deg"
            )
            break
    return reasons


def _save_sample(
    run_dir: Path,
    manifest: dict[str, Any],
    capture: dict[str, Any],
    detection: Detection,
    board: BoardSpec,
    camera: CameraModel,
) -> dict[str, Any]:
    cv2 = _require_cv2()
    sample_number = max([int(item["sample_id"]) for item in manifest.get("samples", [])] + [0]) + 1
    sample_id = f"{sample_number:04d}"
    sample_dir = run_dir / "samples" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=False)
    cv2.imwrite(str(sample_dir / "rgb.png"), cv2.cvtColor(capture["rgb"], cv2.COLOR_RGB2BGR))
    if capture["depth_m"] is not None:
        np.save(sample_dir / "depth.npy", capture["depth_m"].astype(np.float32))
    cv2.imwrite(str(sample_dir / "detection.png"), draw_detection(capture["rgb"], detection, board, camera))
    before = pose_xyzw_to_matrix(capture["pose_before"])
    after = pose_xyzw_to_matrix(capture["pose_after"])
    motion_m, motion_deg = transform_distance(before, after)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_id,
        "split": "validation" if sample_number % 5 == 0 else "calibration",
        "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "camera_timestamp": capture["rgbd_metadata"].get("timestamp"),
        "host_monotonic_ns": capture["host_monotonic_ns"],
        "tcp_pose_before_xyzw": capture["pose_before"],
        "tcp_pose_after_xyzw": capture["pose_after"],
        "tcp_pose_midpoint_xyzw": capture["pose_midpoint"],
        "T_control_tcp": pose_xyzw_to_matrix(capture["pose_midpoint"]),
        "motion_during_capture": {"translation_m": motion_m, "rotation_deg": motion_deg},
        "corners_px": detection.corners,
        "corner_order_reversed": detection.reversed_order,
        "T_camera_board": detection.T_camera_board,
        "pnp_reprojection_rms_px": detection.reprojection_rms_px,
        "corner_margin_px": detection.corner_margin_px,
        "camera": {"K": camera.K, "D": camera.D, "image_size": camera.image_size},
        "rgbd": capture["rgbd_metadata"],
        "files": {
            "rgb": "rgb.png",
            "depth": "depth.npy" if capture["depth_m"] is not None else None,
            "detection": "detection.png",
        },
    }
    write_json(sample_dir / "sample.json", payload)
    entry = {
        "sample_id": sample_id,
        "status": "accepted",
        "split": payload["split"],
        "sample_json": str((Path("samples") / sample_id / "sample.json").as_posix()),
    }
    manifest.setdefault("samples", []).append(entry)
    write_json(run_dir / "manifest.json", manifest)
    return payload


def collect_command(args: argparse.Namespace) -> int:
    cv2 = _require_cv2()
    board = BoardSpec(args.squares_x, args.squares_y, args.square_size_m)
    board.validate()
    camera = _load_camera(args.intrinsics)
    run_dir = args.output or Path("calibration_runs") / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(run_dir, board, camera)
    print("\nREAD-ONLY CALIBRATION COLLECTION")
    print("  TCP feedback is assumed to be: torso_link4 -> right_gripper_link")
    print("  pose format: [x,y,z,qx,qy,qz,qw], translation in meters")
    print("  checkerboard: 12x9 squares, 11x8 inner corners, 20 mm pitch")
    print("  SPACE=capture, R=reverse corner order, Q/ESC=finish")
    if not args.yes:
        response = input("Confirm that camera/base/torso are fixed and the board is rigidly mounted [y/N]: ").strip().lower()
        if response not in ("y", "yes"):
            print("Cancelled without connecting to the robot.")
            return 2
    robot = ReadOnlyRobotClient(args.robot_ip, args.robot_port)
    reverse_order = False
    window = "D405 eye-to-hand calibration"
    try:
        existing = _accepted_samples(run_dir)
        while True:
            preview_payload = robot.get_right_wrist_rgbd()
            preview_rgb, _, _ = decode_rgbd_payload(preview_payload)
            detection = detect_checkerboard(preview_rgb, board, camera, reverse_order=reverse_order)
            canvas = draw_detection(preview_rgb, detection, board, camera)
            cv2.putText(
                canvas,
                f"accepted={len(existing)}/30 | SPACE capture | R reverse | Q quit",
                (18, canvas.shape[0] - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            cv2.imshow(window, canvas)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("r"), ord("R")):
                reverse_order = not reverse_order
                print(f"Corner order reversed: {reverse_order}. Ensure corner 0 is nearest the physical marker.")
                continue
            if key != 32:
                continue
            try:
                capture = capture_observation(robot)
                detection = detect_checkerboard(capture["rgb"], board, camera, reverse_order=reverse_order)
                reasons = _quality_reasons(capture, detection, existing, camera)
                if reasons:
                    print("REJECTED: " + "; ".join(reasons))
                    continue
                assert detection is not None
                saved = _save_sample(run_dir, manifest, capture, detection, board, camera)
                existing.append(saved)
                print(
                    f"ACCEPTED {saved['sample_id']} ({saved['split']}), "
                    f"RMS={saved['pnp_reprojection_rms_px']:.3f}px"
                )
            except Exception as exc:  # keep the live collector usable after a malformed frame
                print(f"CAPTURE ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        robot.close()
        cv2.destroyAllWindows()
    print(f"Saved {len(_accepted_samples(run_dir))} accepted samples in {run_dir}")
    return 0


def average_transforms(transforms: Sequence[np.ndarray]) -> np.ndarray:
    if not transforms:
        raise ValueError("cannot average an empty transform sequence")
    checked = [validate_transform(item) for item in transforms]
    translations = np.stack([item[:3, 3] for item in checked])
    quaternions = np.stack([rotation_to_quaternion_xyzw(item[:3, :3]) for item in checked])
    reference = quaternions[0]
    quaternions[np.sum(quaternions * reference, axis=1) < 0] *= -1
    accumulator = sum(np.outer(q, q) for q in quaternions)
    quaternion = np.linalg.eigh(accumulator)[1][:, -1]
    if np.dot(quaternion, reference) < 0:
        quaternion *= -1
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = quaternion_xyzw_to_rotation(quaternion)
    result[:3, 3] = np.median(translations, axis=0)
    return result


def _percentile(values: Sequence[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile)) if values else float("nan")


def transform_residuals(transforms: Sequence[np.ndarray], center: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = average_transforms(transforms) if center is None else validate_transform(center)
    translations, rotations = [], []
    for transform in transforms:
        translation_m, rotation_deg = transform_distance(center, transform)
        translations.append(translation_m)
        rotations.append(rotation_deg)
    return center, np.asarray(translations), np.asarray(rotations)


def identify_inliers(transforms: Sequence[np.ndarray]) -> tuple[np.ndarray, dict[str, float]]:
    _, translation, rotation = transform_residuals(transforms)

    def threshold(values: np.ndarray, floor: float) -> float:
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        return max(floor, median + 3.0 * 1.4826 * mad)

    translation_limit = threshold(translation, 0.005)
    rotation_limit = threshold(rotation, 1.0)
    mask = (translation <= translation_limit) & (rotation <= rotation_limit)
    return mask, {"translation_m": translation_limit, "rotation_deg": rotation_limit}


def _hand_eye_methods(cv2: Any) -> dict[str, int]:
    return {
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }


def solve_hand_eye(samples: Sequence[dict[str, Any]], method: int) -> np.ndarray:
    cv2 = _require_cv2()
    if len(samples) < 3:
        raise ValueError("OpenCV hand-eye calibration requires at least 3 poses")
    T_tcp_control = [invert_transform(_sample_transform(sample, "T_control_tcp")) for sample in samples]
    T_camera_board = [_sample_transform(sample, "T_camera_board") for sample in samples]
    rotation, translation = cv2.calibrateHandEye(
        [item[:3, :3] for item in T_tcp_control],
        [item[:3, 3].reshape(3, 1) for item in T_tcp_control],
        [item[:3, :3] for item in T_camera_board],
        [item[:3, 3].reshape(3, 1) for item in T_camera_board],
        method=method,
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.asarray(rotation, dtype=np.float64)
    result[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return validate_transform(result, name="T_control_camera")


def board_mount_transforms(samples: Sequence[dict[str, Any]], T_control_camera: np.ndarray) -> list[np.ndarray]:
    result = []
    for sample in samples:
        T_control_tcp = _sample_transform(sample, "T_control_tcp")
        T_camera_board = _sample_transform(sample, "T_camera_board")
        result.append(invert_transform(T_control_tcp) @ T_control_camera @ T_camera_board)
    return result


def _project_board(T_camera_board: np.ndarray, board: BoardSpec, camera: CameraModel) -> np.ndarray:
    cv2 = _require_cv2()
    rvec, tvec = _rvec_tvec_from_matrix(T_camera_board)
    return cv2.projectPoints(board_object_points(board), rvec, tvec, camera.K, camera.D)[0].reshape(-1, 2)


def _validation_metrics(
    samples: Sequence[dict[str, Any]],
    T_control_camera: np.ndarray,
    T_tcp_board: np.ndarray,
    board: BoardSpec,
    camera: CameraModel,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        T_control_tcp = _sample_transform(sample, "T_control_tcp")
        measured = _sample_transform(sample, "T_camera_board")
        mount = invert_transform(T_control_tcp) @ T_control_camera @ measured
        mount_translation, mount_rotation = transform_distance(T_tcp_board, mount)
        predicted = invert_transform(T_control_camera) @ T_control_tcp @ T_tcp_board
        predicted_pixels = _project_board(predicted, board, camera)
        observed_pixels = np.asarray(sample["corners_px"], dtype=np.float64).reshape(-1, 2)
        reprojection = float(np.sqrt(np.mean(np.sum((predicted_pixels - observed_pixels) ** 2, axis=1))))
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "mount_translation_m": mount_translation,
                "mount_rotation_deg": mount_rotation,
                "predicted_reprojection_rms_px": reprojection,
                "T_camera_board_predicted": predicted,
                "predicted_corners_px": predicted_pixels,
            }
        )
    translations = [row["mount_translation_m"] for row in rows]
    rotations = [row["mount_rotation_deg"] for row in rows]
    reprojections = [row["predicted_reprojection_rms_px"] for row in rows]
    metrics = {
        "sample_count": len(rows),
        "mount_translation_median_m": float(np.median(translations)) if rows else float("nan"),
        "mount_translation_p95_m": _percentile(translations, 95),
        "mount_rotation_median_deg": float(np.median(rotations)) if rows else float("nan"),
        "mount_rotation_p95_deg": _percentile(rotations, 95),
        "predicted_reprojection_rms_median_px": float(np.median(reprojections)) if rows else float("nan"),
        "predicted_reprojection_rms_max_px": max(reprojections, default=float("nan")),
    }
    return metrics, rows


def _draw_validation(
    run_dir: Path,
    sample: dict[str, Any],
    row: dict[str, Any],
    board: BoardSpec,
    camera: CameraModel,
) -> None:
    cv2 = _require_cv2()
    source = run_dir / sample["_entry"]["sample_json"]
    image = cv2.imread(str(source.parent / sample["files"]["rgb"]), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read {source.parent / sample['files']['rgb']}")
    observed = np.rint(np.asarray(sample["corners_px"])).astype(int)
    predicted = np.rint(np.asarray(row["predicted_corners_px"])).astype(int)
    for point in observed:
        cv2.circle(image, tuple(point), 2, (0, 255, 0), -1)
    for point in predicted:
        cv2.drawMarker(image, tuple(point), (0, 0, 255), cv2.MARKER_CROSS, 6, 1)
    rvec, tvec = _rvec_tvec_from_matrix(np.asarray(row["T_camera_board_predicted"]))
    cv2.drawFrameAxes(image, camera.K, camera.D, rvec, tvec, board.square_size_m * 3.0, 2)
    cv2.putText(
        image,
        f"green=observed red=predicted RMS={row['predicted_reprojection_rms_px']:.2f}px",
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )
    output = run_dir / "validation" / f"{sample['sample_id']}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), image)


def _config_from_manifest(manifest: dict[str, Any]) -> tuple[BoardSpec, CameraModel]:
    board_payload = manifest["board"]
    board = BoardSpec(
        int(board_payload["squares_x"]), int(board_payload["squares_y"]), float(board_payload["square_size_m"])
    )
    camera_payload = manifest["camera"]
    camera = CameraModel(
        np.asarray(camera_payload["K"], dtype=np.float64),
        np.asarray(camera_payload["D"], dtype=np.float64).reshape(-1),
        tuple(int(value) for value in camera_payload["image_size"]),
    )
    board.validate()
    camera.validate()
    return board, camera


def _method_solutions(samples: Sequence[dict[str, Any]]) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    cv2 = _require_cv2()
    solutions, failures = {}, {}
    for name, method in _hand_eye_methods(cv2).items():
        try:
            solutions[name] = solve_hand_eye(samples, method)
        except Exception as exc:
            failures[name] = f"{type(exc).__name__}: {exc}"
    if "PARK" not in solutions:
        raise RuntimeError(f"PARK hand-eye calibration failed: {failures.get('PARK', 'unknown error')}")
    return solutions, failures


def _solve(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = read_json(run_dir / "manifest.json")
    board, camera = _config_from_manifest(manifest)
    samples = _accepted_samples(run_dir)
    training = [sample for sample in samples if sample.get("split") == "calibration"]
    validation = [sample for sample in samples if sample.get("split") == "validation"]
    if len(training) < 8:
        raise ValueError(f"need at least 8 calibration samples for a guarded solve; found {len(training)}")
    initial_solutions, initial_failures = _method_solutions(training)
    initial_mounts = board_mount_transforms(training, initial_solutions["PARK"])
    inlier_mask, outlier_thresholds = identify_inliers(initial_mounts)
    if int(np.count_nonzero(inlier_mask)) < 8:
        raise ValueError("outlier rejection left fewer than 8 calibration samples")
    inliers = [sample for sample, keep in zip(training, inlier_mask, strict=True) if keep]
    outliers = [sample["sample_id"] for sample, keep in zip(training, inlier_mask, strict=True) if not keep]
    solutions, failures = _method_solutions(inliers)
    mounts = board_mount_transforms(inliers, solutions["PARK"])
    T_tcp_board, training_translation, training_rotation = transform_residuals(mounts)
    validation_metrics, validation_rows = _validation_metrics(
        validation, solutions["PARK"], T_tcp_board, board, camera
    )
    method_rows: dict[str, Any] = {}
    stable_names = []
    park_holdout = validation_metrics["predicted_reprojection_rms_median_px"]
    for name, transform in solutions.items():
        mount = average_transforms(board_mount_transforms(inliers, transform))
        metrics, _ = _validation_metrics(validation, transform, mount, board, camera)
        delta_m, delta_deg = transform_distance(solutions["PARK"], transform)
        stable = bool(
            name == "PARK"
            or (
                validation
                and np.isfinite(metrics["predicted_reprojection_rms_median_px"])
                and metrics["predicted_reprojection_rms_median_px"] <= max(3.0, 1.5 * park_holdout)
            )
        )
        if stable:
            stable_names.append(name)
        method_rows[name] = {
            "T_control_camera": transform,
            "delta_from_park_translation_m": delta_m,
            "delta_from_park_rotation_deg": delta_deg,
            "validation": metrics,
            "stable": stable,
        }
    stable_deltas_m = [method_rows[name]["delta_from_park_translation_m"] for name in stable_names if name != "PARK"]
    stable_deltas_deg = [method_rows[name]["delta_from_park_rotation_deg"] for name in stable_names if name != "PARK"]
    method_spread = {
        "stable_methods": stable_names,
        "max_translation_m": max(stable_deltas_m, default=float("nan")),
        "max_rotation_deg": max(stable_deltas_deg, default=float("nan")),
    }
    failures_qa: list[str] = []
    if len(inliers) < 20:
        failures_qa.append(f"only {len(inliers)} inlier calibration samples; require at least 20")
    if len(validation) < 5:
        failures_qa.append(f"only {len(validation)} validation samples; require at least 5")
    if validation:
        checks = (
            (validation_metrics["predicted_reprojection_rms_max_px"] <= 2.0, "validation reprojection max exceeds 2px"),
            (validation_metrics["mount_translation_median_m"] <= 0.003, "mount translation median exceeds 3mm"),
            (validation_metrics["mount_translation_p95_m"] <= 0.005, "mount translation p95 exceeds 5mm"),
            (validation_metrics["mount_rotation_median_deg"] <= 0.5, "mount rotation median exceeds 0.5deg"),
            (validation_metrics["mount_rotation_p95_deg"] <= 1.0, "mount rotation p95 exceeds 1deg"),
        )
        failures_qa.extend(message for passed, message in checks if not passed)
    if len(stable_names) < 2:
        failures_qa.append("no second stable hand-eye method agrees with PARK")
    elif method_spread["max_translation_m"] > 0.005 or method_spread["max_rotation_deg"] > 1.0:
        failures_qa.append("stable solver methods disagree by more than 5mm or 1deg")
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "equation": "T_control_tcp @ T_tcp_board = T_control_camera @ T_camera_board",
        "sample_counts": {
            "accepted": len(samples),
            "calibration": len(training),
            "calibration_inliers": len(inliers),
            "validation": len(validation),
        },
        "inlier_sample_ids": [sample["sample_id"] for sample in inliers],
        "outlier_sample_ids": outliers,
        "outlier_thresholds": outlier_thresholds,
        "training_mount_residuals": {
            "translation_median_m": float(np.median(training_translation)),
            "translation_p95_m": _percentile(training_translation.tolist(), 95),
            "rotation_median_deg": float(np.median(training_rotation)),
            "rotation_p95_deg": _percentile(training_rotation.tolist(), 95),
        },
        "validation_metrics": validation_metrics,
        "validation_samples": validation_rows,
        "solver_methods": method_rows,
        "solver_failures": {**initial_failures, **failures},
        "method_spread": method_spread,
        "qa_passed": not failures_qa,
        "qa_failures": failures_qa,
    }
    extrinsics = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": report["generated_at"],
        "T_control_camera": solutions["PARK"],
        "T_camera_control": invert_transform(solutions["PARK"]),
        "T_tcp_board": T_tcp_board,
        "control_frame": manifest["control_frame"],
        "camera_frame": manifest["camera_frame"],
        "tcp_frame": manifest["tcp_frame"],
        "translation_unit": "m",
        "quaternion_order": "xyzw",
        "board": asdict(board),
        "K": camera.K,
        "D": camera.D,
        "image_size": camera.image_size,
        "primary_solver": "PARK",
        "inlier_sample_ids": report["inlier_sample_ids"],
        "solver_methods": list(solutions),
        "validation_metrics": validation_metrics,
        "qa_passed": report["qa_passed"],
        "qa_failures": failures_qa,
    }
    for sample, row in zip(validation, validation_rows, strict=True):
        _draw_validation(run_dir, sample, row, board, camera)
    return extrinsics, report


def solve_command(args: argparse.Namespace) -> int:
    _require_cv2()
    run_dir = args.run_dir.resolve()
    extrinsics, report = _solve(run_dir)
    write_json(run_dir / "extrinsics.json", extrinsics)
    write_json(run_dir / "report.json", report)
    print(f"Wrote {run_dir / 'extrinsics.json'}")
    print(f"QA passed: {extrinsics['qa_passed']}")
    if not extrinsics["qa_passed"]:
        for failure in extrinsics["qa_failures"]:
            print(f"  - {failure}")
        return 1
    return 0


def validate_command(args: argparse.Namespace) -> int:
    _require_cv2()
    run_dir = args.run_dir.resolve()
    extrinsics_path = run_dir / "extrinsics.json"
    if not extrinsics_path.exists():
        raise FileNotFoundError(f"run solve first; missing {extrinsics_path}")
    manifest = read_json(run_dir / "manifest.json")
    board, camera = _config_from_manifest(manifest)
    extrinsics = read_json(extrinsics_path)
    T_control_camera = validate_transform(np.asarray(extrinsics["T_control_camera"], dtype=np.float64))
    T_tcp_board = validate_transform(np.asarray(extrinsics["T_tcp_board"], dtype=np.float64))
    validation = [sample for sample in _accepted_samples(run_dir) if sample.get("split") == "validation"]
    metrics, rows = _validation_metrics(validation, T_control_camera, T_tcp_board, board, camera)
    for sample, row in zip(validation, rows, strict=True):
        _draw_validation(run_dir, sample, row, board, camera)
    report_path = run_dir / "validation_report.json"
    write_json(report_path, {"schema_version": SCHEMA_VERSION, "metrics": metrics, "samples": rows})
    print(json.dumps(_jsonable(metrics), ensure_ascii=False, indent=2))
    print(f"Wrote {report_path} and overlays under {run_dir / 'validation'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="interactively collect paired RGBD/TCP samples")
    collect.add_argument("--output", type=Path, help="run directory; defaults to calibration_runs/<timestamp>")
    collect.add_argument("--robot-ip", default="172.16.0.30")
    collect.add_argument("--robot-port", type=int, default=4242)
    collect.add_argument("--intrinsics", type=Path, help="JSON with K, D, and image_size")
    collect.add_argument("--squares-x", type=int, default=12)
    collect.add_argument("--squares-y", type=int, default=9)
    collect.add_argument("--square-size-m", type=float, default=0.020)
    collect.add_argument("--yes", action="store_true", help="skip the physical setup confirmation prompt")
    collect.set_defaults(handler=collect_command)

    solve = subparsers.add_parser("solve", help="solve extrinsics from a completed run")
    solve.add_argument("--run-dir", type=Path, required=True)
    solve.set_defaults(handler=solve_command)

    validate = subparsers.add_parser("validate", help="regenerate holdout metrics and overlay images")
    validate.add_argument("--run-dir", type=Path, required=True)
    validate.set_defaults(handler=validate_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
