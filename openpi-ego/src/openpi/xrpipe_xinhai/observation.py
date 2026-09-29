from __future__ import annotations

import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .config import load_geometry, read_json
from .geometry import pose7, relation_state
from .perception import ObjectTracker


def stamp_ns(value):
    return int(value["sec"]) * 1_000_000_000 + int(value["nanosec"])


def decode_rgbd(packet, camera):
    if not isinstance(packet, dict) or "depth" not in packet:
        raise ValueError("RPC did not return a paired RGBD dict")
    depth = packet["depth"]
    stamp = stamp_ns(packet["stamp"])
    if stamp != stamp_ns(depth["stamp"]):
        raise ValueError("RGB/depth timestamps differ")
    expected = camera["camera_frame"].lstrip("/")
    if packet["frame_id"].lstrip("/") != expected:
        raise ValueError("RGB optical frame differs from camera calibration")
    if depth.get("frame_id", expected).lstrip("/") != expected:
        raise ValueError("Depth is not aligned to the configured RGB optical frame")
    if depth["encoding"] != "16UC1":
        raise ValueError("Depth encoding must be 16UC1")
    h, w, step = int(depth["height"]), int(depth["width"]), int(depth["step"])
    if (w, h) != (camera["width"], camera["height"]) or step < 2 * w or step % 2:
        raise ValueError("Depth dimensions or stride do not match calibration")
    blob = bytes(depth["data"])
    if len(blob) != h * step:
        raise ValueError("Depth buffer length differs from height*step")
    dtype = ">u2" if depth["is_bigendian"] else "<u2"
    raw = np.frombuffer(blob, dtype=dtype).reshape(h, step // 2)[:, :w].astype(np.uint16)
    bgr = cv2.imdecode(np.frombuffer(bytes(packet["data"]), dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None or bgr.shape != (h, w, 3):
        raise ValueError("Invalid RGB JPEG or RGB/depth size mismatch")
    if "bgr8" not in packet.get("format", "").lower() and "rgb8" not in packet.get("format", "").lower():
        raise ValueError("Unsupported compressed RGB format")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    metres = raw.astype(np.float32) * float(camera["depth_scale"])
    metres[raw == 0] = np.nan
    return rgb, metres, stamp


class GripperFilter:
    """Feedback-only causal hysteresis; close .70/open .60, confirm 5/30 s, dwell 12/30 s."""

    def __init__(self):
        self.closed = None
        self.candidate = None
        self.since = None
        self.changed = -float("inf")
        self.last = -float("inf")
        self.samples = 0

    def update(self, width, now):
        if not np.isfinite(width) or now < self.last:
            raise ValueError("Nonfinite gripper feedback or nonmonotonic feedback time")
        if now - self.last > 1.:
            self.candidate = None
        self.last = now
        fraction = 1 - np.clip(float(width) / 80., 0, 1)
        candidate = True if fraction >= .70 else False if fraction <= .60 else self.closed
        if candidate is None:
            raise ValueError("Initial gripper feedback lies inside hysteresis band; establish open/closed first")
        if candidate != self.candidate:
            self.candidate, self.since, self.samples = candidate, now, 0
        self.samples += 1
        if self.samples >= 5 and now - self.since >= 5 / 30. and now - self.changed >= 12 / 30.:
            if self.closed != candidate:
                self.closed, self.changed = candidate, now
        return self.closed


class RobotReader:
    """Read-only facade. Reuses legacy right TCP/gripper readers, exposes no command method."""

    def __init__(self, args):
        from robot_client_xinhai_ws import R1ProInterface
        self._robot = R1ProInterface(args.robot_ip, args.robot_port)

    def read_tcp(self):
        return self._robot.get_ee_pose_right()

    def read_gripper(self):
        return self._robot.get_gripper_state_right()

    def read_rgbd(self):
        return self._robot._server.get_right_wrist_rgbd()

    def close(self):
        self._robot.close()


class RobotController(RobotReader):
    def move(self, pose, gripper):
        # These are the SAME legacy methods; no IK, left arm, joint commands or frame remapping.
        self._robot.move_ee_right(pose)
        self._robot.open_gripper_right(float(gripper))


def lift_mask(depth, mask, camera):
    from .config import pipeline_imports
    pipeline_imports()
    from xrrel.lift import DEFAULT_ERODE_PX, DEFAULT_STRIDE
    support = np.asarray(mask, np.uint8)
    if DEFAULT_ERODE_PX:
        eroded = cv2.erode(support, np.ones((2 * DEFAULT_ERODE_PX + 1,) * 2, np.uint8))
        if eroded.sum() >= 16:
            support = eroded
    v, u = np.where(support.astype(bool) & np.isfinite(depth) & (depth > 0))
    # Match pipeline regular image-grid sampling, not sampling every Nth masked pixel.
    selected = (u % DEFAULT_STRIDE == 0) & (v % DEFAULT_STRIDE == 0)
    u, v = u[selected], v[selected]
    uv = np.column_stack([u, v]).astype(np.float64)
    if not len(uv):
        return np.empty((0, 3))
    rays = cv2.undistortPoints(uv[:, None], np.asarray(camera["K"], float), np.asarray(camera["D"], float))[:, 0]
    return np.c_[rays, np.ones(len(rays))] * depth[v, u, None]


class ObservationPipeline:
    def __init__(self, args, perception, tracker=None):
        self.args, self.perception = args, perception
        self.camera = read_json(args.camera_config)
        if self.camera.get("distortion_model") != "plumb_bob":
            raise ValueError("Only D405 plumb_bob calibration supported")
        self.geometry, self.tem, self.extrinsics = load_geometry(args)
        self.tbc = np.asarray(self.extrinsics["T_B_C"], float) if self.extrinsics else None
        if self.extrinsics and self.extrinsics["camera_frame"].lstrip("/") != self.camera["camera_frame"].lstrip("/"):
            raise ValueError("Extrinsics and intrinsics refer to different camera frames")
        self.tracker = tracker if tracker is not None else ObjectTracker()
        self.gripper = GripperFilter()
        self.last_stamp = None

    def poll_feedback(self, reader):
        tcp, width, now = reader.read_tcp(), reader.read_gripper(), time.monotonic()
        te = pose7(tcp)
        closed = self.gripper.update(width, now)
        if closed is not None and self.tbc is not None:
            self.tracker.feedback(np.linalg.inv(self.tbc) @ te @ self.tem, closed, now)
        return tcp, width, closed

    def replay_feedback(self, execution_file):
        """Replay recorded actual feedback between two image observations, never target commands."""
        if self.tbc is None or not Path(execution_file).is_file():
            return
        for event in read_json(execution_file).get("feedback", []):
            if event["closed"] is not None:
                hand = np.linalg.inv(self.tbc) @ pose7(event["tcp"]) @ self.tem
                self.tracker.feedback(hand, event["closed"], event["time_ns"] / 1e9)

    def capture_observation(self, reader=None, replay=None):
        start = time.monotonic()
        capture_epoch_ns = time.time_ns()
        if replay is not None:
            with np.load(replay, allow_pickle=False) as z:
                rgb, depth, tcp = z["rgb"], z["depth"], z["tcp"]
                meta = json.loads(str(z["metadata"]))
            stamp, width, closed = meta["stamp_ns"], meta["gripper_width"], meta["closed"]
            tcp_before = np.asarray(meta["tcp_before"])
            bracket_s = meta["tcp_bracket_seconds"]
            sample_time = stamp / 1e9
            packet_meta = meta.get("rgbd", {})
            # Replay must explicitly retain the calibration under which the snapshot was captured.
            if meta["camera"] != self.camera or meta["geometry"] != self.geometry or meta["extrinsics"] != self.extrinsics:
                raise ValueError("Replay calibration mismatch; pass original camera/TCP/extrinsics files")
            received = time.monotonic()
        else:
            if reader is None:
                raise ValueError("reader required for live capture")
            while self.gripper.closed is None or time.monotonic() - start < .2:
                self.poll_feedback(reader)
                if time.monotonic() - start > self.args.capture_timeout:
                    raise TimeoutError("Gripper feedback never established a confirmed state")
                time.sleep(.02)
            while True:
                begin = time.monotonic()
                tcp_before = reader.read_tcp()
                packet = reader.read_rgbd()
                tcp, width, closed = self.poll_feedback(reader)
                received = time.monotonic()
                bracket_s = received - begin
                if packet is not None:
                    rgb, depth, stamp = decode_rgbd(packet, self.camera)
                    if stamp >= capture_epoch_ns and (self.last_stamp is None or stamp > self.last_stamp):
                        break
                if received - start > self.args.capture_timeout:
                    raise TimeoutError("No new exactly-paired RGBD before timeout")
                time.sleep(.02)
            # Robot and client clocks must be synchronized; do not silently accept stale cached images.
            age = time.time() - stamp / 1e9
            if abs(age) > self.args.max_observation_age:
                raise RuntimeError("RGBD stale or host clocks unsynchronized; check NTP/ROS clock")
            sample_time = received
            received -= max(0., age)
            packet_meta = {k: v for k, v in packet.items() if k not in ("data", "depth")}
            packet_meta["depth"] = {k: v for k, v in packet["depth"].items() if k != "data"}
        if self.last_stamp is not None and stamp <= self.last_stamp:
            raise ValueError("Snapshot timestamp must strictly increase")
        before, te = pose7(tcp_before), pose7(tcp)
        if (np.linalg.norm(before[:3, 3] - te[:3, 3]) > self.args.max_tcp_bracket_translation
                or np.rad2deg(Rotation.from_matrix(before[:3, :3].T @ te[:3, :3]).magnitude()) > self.args.max_tcp_bracket_rotation_deg):
            raise RuntimeError("TCP moved while reading RGBD; snapshot is not coherent")
        self.last_stamp = stamp
        hand = np.linalg.inv(self.tbc) @ te @ self.tem if self.tbc is not None else None
        allowed_missing = {self.tracker.lock[0]} if closed and self.tracker.lock is not None else set()
        result = self.perception.process(rgb, stamp, allowed_missing=allowed_missing)
        pose_start = time.perf_counter()
        clouds = [lift_mask(depth, mask, self.camera) for mask in result["masks"]]
        centres = [np.argwhere(mask).mean(axis=0)[::-1] if mask.any() else np.full(2, np.nan)
                   for mask in result["masks"]]
        objects, quality = self.tracker.update(clouds, centres, hand, closed, sample_time)
        raw, state = relation_state(hand, objects, closed) if hand is not None else (None, None)
        result["timing"]["pose_estimation"] = time.perf_counter() - pose_start
        result["timing"]["capture_total"] = time.monotonic() - start
        metadata = dict(stamp_ns=int(stamp), observation_id=str(stamp), gripper_width=float(width),
                        closed=bool(closed), tcp_before=np.asarray(tcp_before).tolist(),
                        tcp_bracket_seconds=bracket_s, camera=self.camera, geometry=self.geometry,
                        extrinsics=self.extrinsics, rgbd=packet_meta, timing=result["timing"], quality=quality)
        return dict(rgb=rgb, depth=depth, tcp=np.asarray(tcp), metadata=metadata,
                    received_monotonic=received, T_C_M=hand, T_C_O=objects, clouds=clouds,
                    raw_state=raw, state=state, perception=result)


def save_snapshot(observation, directory):
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    obs = observation
    metadata = json.dumps(obs["metadata"], ensure_ascii=False, allow_nan=False)
    np.savez_compressed(path / "snapshot.npz", rgb=obs["rgb"], depth=obs["depth"], tcp=obs["tcp"], metadata=metadata)
    arrays = {"T_C_O": np.stack(obs["T_C_O"]), "masks": obs["perception"]["masks"]}
    for key in ("T_C_M", "raw_state", "state"):
        if obs[key] is not None:
            arrays[key] = obs[key]
    for i, cloud in enumerate(obs["clouds"], 1):
        arrays[f"cloud_obj{i}"] = cloud
    np.savez_compressed(path / "derived.npz", **arrays)
    return path
