from __future__ import annotations

import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ego_relation.config import ProjectConfig
from ego_relation.contracts.manifest import StageManifest, file_sha256
from ego_relation.contracts.se3 import (
    UNITY_TO_OPENXR,
    compose,
    invert,
    matrix_from_pose7,
    nearest_indices,
    normalize_rotation,
    rotation_angle_deg,
    unity_pose7_to_openxr,
)
from ego_relation.s1_pico_mode2.contracts import write_step1_contracts
from ego_relation.s1_pico_mode2.tcp import palm_pose_to_tcp

XR_WRIST = 1
XR_THUMB_TIP = 5
XR_INDEX_TIP = 10
XR_INDEX_KNUCKLE = 7
XR_MIDDLE_KNUCKLE = 12
XR_PINKY_KNUCKLE = 22


def _decode_jpeg(value) -> np.ndarray:
    buffer = np.frombuffer(bytes(value), dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("JPEG 解码失败")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def project_camera_keypoints(
    points_camera: np.ndarray,
    K: np.ndarray,
    image_size: tuple[int, int],
    frame_valid: np.ndarray,
) -> dict[str, np.ndarray]:
    """Project camera-frame keypoints with the PICO PINHOLE contract."""
    points = np.asarray(points_camera, dtype=np.float64)
    intrinsics = np.asarray(K, dtype=np.float64)
    frame_valid = np.asarray(frame_valid, dtype=bool)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError(f"camera keypoints must be (T,J,3), got {points.shape}")
    if intrinsics.shape != (3, 3):
        raise ValueError(f"K must be (3,3), got {intrinsics.shape}")
    if frame_valid.shape != (len(points),):
        raise ValueError(f"frame_valid must be {(len(points),)}, got {frame_valid.shape}")
    width, height = (int(value) for value in image_size)
    depth = points[..., 2]
    in_front = np.isfinite(points).all(axis=2) & (depth > 1e-6)
    uv = np.full((*points.shape[:2], 2), np.nan, dtype=np.float64)
    uv[..., 0][in_front] = (
        intrinsics[0, 0] * points[..., 0][in_front] / depth[in_front]
        + intrinsics[0, 2]
    )
    uv[..., 1][in_front] = (
        intrinsics[1, 1] * points[..., 1][in_front] / depth[in_front]
        + intrinsics[1, 2]
    )
    in_image = (
        in_front
        & (uv[..., 0] >= 0)
        & (uv[..., 0] < width)
        & (uv[..., 1] >= 0)
        & (uv[..., 1] < height)
    )
    tracking_valid = np.broadcast_to(frame_valid[:, None], in_image.shape)
    pixel_valid = tracking_valid & in_image
    uv_int = np.full(uv.shape, -1, dtype=np.int32)
    rounded = np.rint(uv[pixel_valid]).astype(np.int32)
    rounded[:, 0] = np.clip(rounded[:, 0], 0, width - 1)
    rounded[:, 1] = np.clip(rounded[:, 1], 0, height - 1)
    uv_int[pixel_valid] = rounded
    return {
        "uv": uv.astype(np.float32),
        "uv_int": uv_int,
        "depth_m": depth.astype(np.float32),
        "in_front": in_front,
        "in_image": in_image,
        "pixel_valid": pixel_valid,
    }


def write_hand_pixel_keypoints(
    camera_dir: Path,
    qa_dir: Path,
    timestamps_ns: np.ndarray,
    K: np.ndarray,
    image_size: tuple[int, int],
    hands: dict[str, np.ndarray],
) -> tuple[Path, Path, dict]:
    output: dict[str, np.ndarray] = {
        "camera_index": np.arange(len(timestamps_ns), dtype=np.int64),
        "timestamp_ns": np.asarray(timestamps_ns, dtype=np.int64),
        "K": np.asarray(K, dtype=np.float64),
        "image_size_wh": np.asarray(image_size, dtype=np.int32),
        "coordinate_system": np.asarray("opencv_rh_x_right_y_down_z_forward"),
        "projection_model": np.asarray("PICO_XR_CAMERA_MODEL_PINHOLE_PICO"),
    }
    report = {
        "frames": int(len(timestamps_ns)),
        "joints_per_hand": 26,
        "coordinate_system": "opencv_rh_x_right_y_down_z_forward",
        "projection_model": "PICO_XR_CAMERA_MODEL_PINHOLE_PICO",
        "pixel_coordinates": "u_right_v_down_zero_based",
        "hands": {},
    }
    for side in ("left", "right"):
        frame_valid = np.asarray(hands[f"{side}_valid"], dtype=bool)
        projected = project_camera_keypoints(
            hands[f"{side}_joints_camera"],
            K,
            image_size,
            frame_valid,
        )
        output[f"{side}_tracking_valid"] = frame_valid
        for name, value in projected.items():
            output[f"{side}_{name}"] = value
        tracked = np.broadcast_to(frame_valid[:, None], projected["in_image"].shape)
        tracked_count = int(tracked.sum())
        report["hands"][side] = {
            "tracking_valid_frame_ratio": float(frame_valid.mean()),
            "positive_depth_keypoint_ratio": float(
                projected["in_front"][tracked].mean()
            ) if tracked_count else 0.0,
            "in_image_keypoint_ratio": float(
                projected["in_image"][tracked].mean()
            ) if tracked_count else 0.0,
            "frames_with_any_pixel": float(projected["pixel_valid"].any(axis=1).mean()),
            "frames_with_wrist_pixel": float(projected["pixel_valid"][:, XR_WRIST].mean()),
        }
    output_path = camera_dir / "hand_keypoints_pixels.npz"
    report_path = qa_dir / "hand_keypoints_pixels_report.json"
    np.savez_compressed(output_path, **output)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path, report_path, report


def write_wrist_tcp_poses(
    camera_dir: Path,
    qa_dir: Path,
    timestamps_ns: np.ndarray,
    camera_eye: str,
    hands: dict[str, np.ndarray],
) -> tuple[Path, Path, dict]:
    output: dict[str, np.ndarray] = {
        "camera_index": np.arange(len(timestamps_ns), dtype=np.int64),
        "timestamp_ns": np.asarray(timestamps_ns, dtype=np.int64),
        "camera0_eye": np.asarray(camera_eye),
        "pose_convention": np.asarray("T_reference_entity"),
        "camera_coordinate_system": np.asarray("opencv_rh_x_right_y_down_z_forward"),
        "tcp_origin": np.asarray("XR wrist position; zero wrist-to-TCP translation offset"),
        "tcp_rotation": np.asarray("geometric palm rotation times fixed side TCP axis alignment"),
    }
    report = {
        "frames": int(len(timestamps_ns)),
        "camera0_eye": camera_eye,
        "pose_convention": "T_reference_entity",
        "step2_reference_frame": "camera0_static_left_camera",
        "tcp_definition": {
            "origin": "XR wrist position",
            "wrist_to_tcp_translation_m": [0.0, 0.0, 0.0],
            "rotation": "R_geometric_palm @ R_tcp_to_inward_palm(side).T",
        },
        "hands": {},
    }
    for side in ("left", "right"):
        valid = np.asarray(hands[f"{side}_wrist_valid"], dtype=bool)
        camera0_wrist = np.asarray(hands[f"T_camera0_{side}_wrist"], dtype=np.float64)
        camera0_tcp = np.asarray(hands[f"T_camera0_{side}_tcp"], dtype=np.float64)
        output[f"{side}_valid"] = valid
        output[f"T_camera0_{side}_wrist"] = camera0_wrist
        output[f"T_camera0_{side}_tcp"] = camera0_tcp
        report["hands"][side] = {
            "valid_frame_ratio": float(valid.mean()),
            "wrist_tcp_translation_delta_m_max": float(
                np.linalg.norm(camera0_wrist[:, :3, 3] - camera0_tcp[:, :3, 3], axis=1).max()
            ),
        }
    output_path = camera_dir / "wrist_tcp_poses_camera0.npz"
    legacy_path = camera_dir / "wrist_tcp_poses_left_camera.npz"
    if legacy_path.is_file():
        legacy_path.unlink()
    report_path = qa_dir / "wrist_tcp_poses_report.json"
    np.savez_compressed(output_path, **output)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path, report_path, report


def _palm_transform(joints_cv: np.ndarray, side: str) -> np.ndarray:
    wrist = joints_cv[XR_WRIST]
    forward = joints_cv[XR_MIDDLE_KNUCKLE] - wrist
    forward /= max(np.linalg.norm(forward), 1e-12)
    across = (
        joints_cv[XR_INDEX_KNUCKLE] - joints_cv[XR_PINKY_KNUCKLE]
        if side == "right"
        else joints_cv[XR_PINKY_KNUCKLE] - joints_cv[XR_INDEX_KNUCKLE]
    )
    normal = np.cross(forward, across)
    normal /= max(np.linalg.norm(normal), 1e-12)
    forward -= np.dot(forward, normal) * normal
    forward /= max(np.linalg.norm(forward), 1e-12)
    lateral = np.cross(normal, forward)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = normalize_rotation(np.stack([forward, lateral, normal], axis=1))
    result[:3, 3] = wrist
    return result


def _wrist_transform_camera0(
    wrist_pose_head_unity: np.ndarray,
    T_camera_head: np.ndarray,
    T_camera0_camera: np.ndarray,
) -> np.ndarray:
    """Convert the HDF wrist pose7 from head coordinates into static camera0."""
    return compose(
        T_camera0_camera,
        _wrist_transform_camera(wrist_pose_head_unity, T_camera_head),
    )


def _wrist_transform_camera(
    wrist_pose_head_unity: np.ndarray,
    T_camera_head: np.ndarray,
) -> np.ndarray:
    """Convert the HDF wrist pose7 from head coordinates into the current camera."""
    wrist_pose = np.asarray(wrist_pose_head_unity, dtype=np.float64)
    if wrist_pose.shape != (7,):
        raise ValueError(f"wrist pose7 应为 (7,)，实际 {wrist_pose.shape}")
    return compose(
        T_camera_head,
        unity_pose7_to_openxr(wrist_pose),
    )


class PicoEpisode:
    """只读 PICO schema 2.1 适配器，所有几何统一成 CV 右手系。"""

    def __init__(self, path: str | Path, cfg: ProjectConfig):
        self.path = Path(path).resolve()
        self.cfg = cfg
        self.file = h5py.File(self.path, "r")
        schema = str(self.file.attrs.get("pico_hdf5_schema_version", ""))
        if not schema.startswith("2."):
            raise ValueError(f"需要 PICO HDF5 schema 2.x，实际 {schema!r}")
        self.eye = cfg.camera.eye
        if self.eye not in {"left", "right"}:
            raise ValueError("camera.eye 只能为 left 或 right")

    def close(self) -> None:
        self.file.close()

    def __enter__(self) -> "PicoEpisode":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    @property
    def camera_timestamps(self) -> np.ndarray:
        return self.file["camera/timestamps_ns"][:].astype(np.int64)

    @property
    def camera_xr_timestamps(self) -> np.ndarray:
        return self.file["camera/pose_xr_timestamps_ns"][:].astype(np.int64)

    @property
    def tracking_timestamps(self) -> np.ndarray:
        return self.file["tracking/xr_timestamps_ns"][:].astype(np.int64)

    @property
    def K(self) -> np.ndarray:
        return self.file[f"camera/K_{self.eye}"][:].astype(np.float64)

    @property
    def image_size(self) -> tuple[int, int]:
        width, height = self.file[f"camera/image_size_{self.eye}"][:]
        return int(width), int(height)

    def image(self, index: int) -> np.ndarray:
        return _decode_jpeg(self.file[f"camera/images_{self.eye}_jpeg"][index])

    def T_head_camera(self) -> np.ndarray:
        dataset = self.file[f"camera/extrinsics_{self.eye}"]
        source = str(dataset.attrs.get("source", ""))
        if source != "XR_PICO_camera_image":
            raise ValueError(
                f"只支持已记录原生 XR_PICO_camera_image 契约的外参，实际 source={source!r}"
            )
        # XrCameraExtrinsics is T_xr_device_camera: its translation/rotation
        # live in OpenXR device coordinates while the camera local axes are CV
        # optical axes.  Do not run this mixed-frame pose through a generic
        # Unity<->CV conjugation.
        transform = matrix_from_pose7(dataset[:])
        if self.cfg.camera.extrinsic_direction == "camera_to_head":
            return invert(transform)
        if self.cfg.camera.extrinsic_direction != "head_to_camera":
            raise ValueError("extrinsic_direction 必须为 head_to_camera 或 camera_to_head")
        return transform

    def camera_poses(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        source = self.cfg.camera.head_pose_source
        if source not in {"tracking", "global"}:
            raise ValueError("head_pose_source 只能为 tracking 或 global")
        head = self.file[f"camera/poses_head_{source}"][:]
        valid = self.file[f"camera/poses_head_{source}_valid"][:].astype(bool)
        T_head_camera = self.T_head_camera()
        T_global_head = np.stack([unity_pose7_to_openxr(pose) for pose in head])
        T_global_camera = np.stack([compose(pose, T_head_camera) for pose in T_global_head])
        T_camera0_camera = np.stack([compose(invert(T_global_camera[0]), pose) for pose in T_global_camera])
        return T_global_camera, T_camera0_camera, valid

    def hands_at_camera(self) -> dict[str, np.ndarray]:
        # 手势与曝光时刻头姿使用 OpenXR runtime epoch；不能和 Unix/device
        # timestamps_ns 混用。后者只用于和 Mode2 动作时间线对齐。
        camera_ts = self.camera_xr_timestamps
        output: dict[str, np.ndarray] = {}
        _, T_camera0_camera, _ = self.camera_poses()
        T_camera_head = invert(self.T_head_camera())
        for side in ("left", "right"):
            timestamps = self.file[f"tracking/hands/{side}/xr_timestamps_ns"][:].astype(np.int64)
            indices, gaps = nearest_indices(timestamps, camera_ts)
            # joints_head 与手势自己的 XrTime 同步，先直接换到当前左相机，再用曝光时刻
            # camera pose 换到 cam0；避免 tracking/global 原点和 Unix/XrTime 混用。
            joint_poses_head = self.file[f"tracking/hands/{side}/joints_head"][:][indices].astype(np.float64)
            joints = joint_poses_head[..., :3]
            joints_head_cv = joints @ UNITY_TO_OPENXR.T
            joints_camera = np.stack(
                [joints_head_cv[i] @ T_camera_head[:3, :3].T + T_camera_head[:3, 3] for i in range(len(joints))]
            )
            joints_camera0 = np.stack(
                [
                    joints_camera[i] @ T_camera0_camera[i, :3, :3].T + T_camera0_camera[i, :3, 3]
                    for i in range(len(joints))
                ]
            )
            camera0_palm_poses = np.stack(
                [_palm_transform(row, side) for row in joints_camera0]
            )
            camera0_wrist_poses = np.stack(
                [
                    _wrist_transform_camera0(
                        joint_poses_head[i, XR_WRIST],
                        T_camera_head,
                        T_camera0_camera[i],
                    )
                    for i in range(len(joint_poses_head))
                ]
            )
            active = self.file[f"tracking/hands/{side}/active"][:][indices].astype(bool)
            valid = self.file[f"tracking/hands/{side}/joints_head_valid"][:][indices].astype(bool)
            hand_valid = active & valid & (gaps <= self.cfg.timeline.max_tracking_gap_ms)
            # Keep the historical geometric palm frame for Mode2/visual compatibility.
            output[f"T_camera0_{side}_hand"] = camera0_palm_poses
            output[f"T_camera0_{side}_wrist"] = camera0_wrist_poses
            output[f"T_camera0_{side}_tcp"] = palm_pose_to_tcp(camera0_palm_poses, side)
            output[f"{side}_joints_camera0"] = joints_camera0
            output[f"{side}_joints_camera"] = joints_camera
            output[f"{side}_valid"] = hand_valid
            output[f"{side}_wrist_valid"] = hand_valid
            output[f"{side}_gap_ms"] = gaps.astype(np.float32)
            output[f"{side}_pinch_m"] = np.linalg.norm(
                joints_head_cv[:, XR_THUMB_TIP] - joints_head_cv[:, XR_INDEX_TIP], axis=1
            ).astype(np.float32)
        return output

    def extrinsic_self_qa(self, hands: dict[str, np.ndarray]) -> dict:
        left_dataset = self.file["camera/extrinsics_left"]
        right_dataset = self.file["camera/extrinsics_right"]
        left = matrix_from_pose7(left_dataset[:])
        right = matrix_from_pose7(right_dataset[:])
        relative = compose(invert(left), right)
        baseline = float(np.linalg.norm(relative[:3, 3]))
        ratios = {}
        positive_depth_ratios = {}
        width, height = self.image_size
        for side in ("left", "right"):
            points = hands[f"{side}_joints_camera"]
            z = points[..., 2]
            uv = np.zeros((*z.shape, 2), dtype=np.float64)
            uv[..., 0] = self.K[0, 0] * points[..., 0] / np.maximum(z, 1e-9) + self.K[0, 2]
            uv[..., 1] = self.K[1, 1] * points[..., 1] / np.maximum(z, 1e-9) + self.K[1, 2]
            inside = (
                (z > 0)
                & (uv[..., 0] >= 0)
                & (uv[..., 0] < width)
                & (uv[..., 1] >= 0)
                & (uv[..., 1] < height)
            )
            valid_frames = hands[f"{side}_valid"][:, None]
            valid_points = valid_frames.repeat(inside.shape[1], axis=1)
            ratios[side] = float(inside[valid_points].mean())
            positive_depth_ratios[side] = float((z > 0)[valid_points].mean())
        attrs_ok = all(
            str(dataset.attrs.get("source", "")) == "XR_PICO_camera_image"
            and str(dataset.attrs.get("direction", "")).startswith("T_xr_device_camera")
            for dataset in (left_dataset, right_dataset)
        )
        rig_ok = bool(
            0.04 <= baseline <= 0.09
            and left[0, 3] < right[0, 3]
            and rotation_angle_deg(relative[:3, :3]) <= 1.0
        )
        projection_ok = bool(
            min(ratios.values()) >= self.cfg.camera.minimum_hand_projection_ratio
            and min(positive_depth_ratios.values()) >= self.cfg.camera.minimum_hand_positive_depth_ratio
        )
        return {
            "sdk_attrs_ok": attrs_ok,
            "rig_geometry_ok": rig_ok,
            "hand_projection_ok": projection_ok,
            "hand_projection_ratio": ratios,
            "hand_positive_depth_ratio": positive_depth_ratios,
            "baseline_m": baseline,
            "relative_rotation_deg": rotation_angle_deg(relative[:3, :3]),
            "left_camera_x_in_head_m": float(left[0, 3]),
            "right_camera_x_in_head_m": float(right[0, 3]),
            "sdk_contract_verified": bool(attrs_ok and rig_ok and projection_ok),
        }

    @property
    def metadata(self) -> dict:
        def convert(value):
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")
            if isinstance(value, np.generic):
                return value.item()
            return value

        return {key: convert(value) for key, value in self.file.attrs.items()}


def prepare_pico_episode(cfg: ProjectConfig, source: str | Path, episode_dir: str | Path) -> StageManifest:
    source = Path(source).resolve()
    episode_dir = Path(episode_dir).resolve()
    contracts = write_step1_contracts(cfg, source, episode_dir)
    camera_dir = episode_dir / "camera"
    sync_dir = episode_dir / "sync"
    qa_dir = episode_dir / "qa"
    for directory in (camera_dir, sync_dir, qa_dir):
        directory.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    with PicoEpisode(source, cfg) as episode:
        timestamps = episode.camera_timestamps
        T_global_camera, T_camera0_camera, camera_valid = episode.camera_poses()
        hands = episode.hands_at_camera()
        extrinsic_self_qa = episode.extrinsic_self_qa(hands)
        np.save(camera_dir / "K.npy", episode.K)
        np.save(camera_dir / "T_global_camera.npy", T_global_camera)
        np.save(camera_dir / "T_camera_to_camera0.npy", T_camera0_camera)
        np.savez_compressed(camera_dir / "hands_camera0.npz", **hands)
        hand_pixels_path, hand_pixels_report_path, hand_pixels_report = write_hand_pixel_keypoints(
            camera_dir,
            qa_dir,
            timestamps,
            episode.K,
            episode.image_size,
            hands,
        )
        wrist_tcp_path, wrist_tcp_report_path, wrist_tcp_report = write_wrist_tcp_poses(
            camera_dir,
            qa_dir,
            timestamps,
            episode.eye,
            hands,
        )

        table = pa.table(
            {
                "camera_index": np.arange(len(timestamps), dtype=np.int64),
                "timestamp_ns": timestamps,
                "xr_timestamp_ns": episode.camera_xr_timestamps,
                "camera_valid": camera_valid,
                "left_hand_valid": hands["left_valid"],
                "right_hand_valid": hands["right_valid"],
                "left_hand_gap_ms": hands["left_gap_ms"],
                "right_hand_gap_ms": hands["right_gap_ms"],
            }
        )
        pq.write_table(table, sync_dir / "camera_frames.parquet")

        relative_translation = np.linalg.norm(T_camera0_camera[:, :3, 3] - T_camera0_camera[0, :3, 3], axis=1)
        relative_rotation = np.array([rotation_angle_deg(pose[:3, :3]) for pose in T_camera0_camera])
        stereo_delta = (
            np.abs(
                episode.file["camera/left_capture_timestamps_xr_ns"][:].astype(np.int64)
                - episode.file["camera/right_capture_timestamps_xr_ns"][:].astype(np.int64)
            )
            / 1e6
        )
        calibration_id = str(episode.metadata.get("calibration_id", ""))
        target_calibration_verified = bool(cfg.camera.calibration_verified and calibration_id)
        factory_calibration_verified = bool(
            cfg.camera.accept_sdk_factory_extrinsics and extrinsic_self_qa["sdk_contract_verified"]
        )
        extrinsic_verified = bool(target_calibration_verified or factory_calibration_verified)
        if cfg.camera.require_verified_extrinsics and not extrinsic_verified:
            warnings.append("PICO SDK 外参未通过标定板或 rig/手投影自检；禁止把 3D 结果用于训练或真机。")
        elif factory_calibration_verified and not target_calibration_verified:
            warnings.append("采用通过 rig/手投影自检的 PICO 工厂外参；正式真机部署仍建议补录标定板 calibration_id。")
        if float(relative_translation.max()) < cfg.camera.minimum_scan_translation_m:
            warnings.append("头部扫视平移基线不足，单目三角化会退化。")
        if float(relative_rotation.max()) < cfg.camera.minimum_scan_rotation_deg:
            warnings.append("头部扫视旋转范围不足，建议每条示范开始先扫视 3--5 秒。")
        if float(np.percentile(stereo_delta, 95)) > cfg.camera.maximum_stereo_delta_ms:
            warnings.append("左右目曝光时间差超过阈值，不应直接当同步双目。")

        qa = {
            "schema": str(episode.metadata.get("pico_hdf5_schema_version")),
            "frames": int(len(timestamps)),
            "image_size": list(episode.image_size),
            "camera_valid_ratio": float(camera_valid.mean()),
            "left_hand_valid_ratio": float(hands["left_valid"].mean()),
            "right_hand_valid_ratio": float(hands["right_valid"].mean()),
            "scan_translation_m": float(relative_translation.max()),
            "scan_rotation_deg": float(relative_rotation.max()),
            "stereo_delta_p95_ms": float(np.percentile(stereo_delta, 95)),
            "extrinsic_direction": cfg.camera.extrinsic_direction,
            "calibration_id": calibration_id,
            "target_calibration_verified": target_calibration_verified,
            "factory_calibration_verified": factory_calibration_verified,
            "extrinsic_verified": extrinsic_verified,
            "extrinsic_self_qa": extrinsic_self_qa,
            "hand_pixel_projection": hand_pixels_report,
            "wrist_tcp_poses": wrist_tcp_report,
            "geometry_deployable": bool(extrinsic_verified or not cfg.camera.require_verified_extrinsics),
            "warnings": warnings,
        }
        (qa_dir / "pico_report.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = StageManifest(
        schema_version=cfg.schema_version,
        stage="pico_ingest",
        episode=source.stem,
        source_path=str(source),
        source_sha256=file_sha256(source),
        config={"timeline": cfg.to_dict()["timeline"], "camera": cfg.to_dict()["camera"]},
        outputs={
            "K": str(camera_dir / "K.npy"),
            "camera_poses": str(camera_dir / "T_camera_to_camera0.npy"),
            "hands": str(camera_dir / "hands_camera0.npz"),
            "hand_pixels": str(hand_pixels_path),
            "wrist_tcp_poses": str(wrist_tcp_path),
            "frame_table": str(sync_dir / "camera_frames.parquet"),
            "qa": str(qa_dir / "pico_report.json"),
            "hand_pixels_qa": str(hand_pixels_report_path),
            "wrist_tcp_poses_qa": str(wrist_tcp_report_path),
            "task_semantics": str(contracts["task_semantics"]),
            "object_instances": str(contracts["object_instances"]),
        },
        metrics=qa,
        warnings=tuple(warnings),
    )
    manifest.write(episode_dir / "pico_ingest.manifest.json")
    return manifest
