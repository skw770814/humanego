from __future__ import annotations

import base64
import html
import json
import os
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import pyarrow.parquet as pq

from ego_relation.config import ProjectConfig
from ego_relation.contracts.se3 import compose, invert, nearest_indices, rotation_angle_deg, vec9_to_transform
from ego_relation.common.pipeline import episode_work_dir
from ego_relation.visualization.html import RawHtml
from ego_relation.visualization.html import comparison
from ego_relation.visualization.html import heatmap
from ego_relation.visualization.html import line_chart
from ego_relation.visualization.html import metrics
from ego_relation.visualization.html import panel
from ego_relation.visualization.html import scene3d
from ego_relation.visualization.html import skeleton3d
from ego_relation.visualization.html import status_badge
from ego_relation.visualization.html import table
from ego_relation.visualization.html import timeline_player
from ego_relation.visualization.html import vector_view
from ego_relation.visualization.html import write_page


_COLORS = ("#42d9d0", "#f5c451", "#ff7185", "#76a9ff", "#75e6a4", "#d697ff", "#ff9f68")

_HAND_BONES = (
    (0, 1),
    (1, 2), (2, 3), (3, 4), (4, 5),
    (1, 6), (6, 7), (7, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13), (13, 14), (14, 15),
    (1, 16), (16, 17), (17, 18), (18, 19), (19, 20),
    (1, 21), (21, 22), (22, 23), (23, 24), (24, 25),
    (7, 12), (12, 17), (17, 22),
)

# PICO BODY_WITHOUT_ARM 24-role order.  This view deliberately stays in the
# SDK local frame; it is a capture QA view, never a robot-base visualization.
_BODY_BONES = (
    (0, 1), (1, 4), (4, 7), (7, 10),
    (0, 2), (2, 5), (5, 8), (8, 11),
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    (12, 13), (13, 16), (16, 18), (18, 20), (20, 22),
    (12, 14), (14, 17), (17, 19), (19, 21), (21, 23),
)


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _clean(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, list | tuple):
        return [_clean(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    return value


def _indices(length: int, count: int) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=np.int64)
    return np.unique(np.linspace(0, length - 1, min(length, max(1, count)), dtype=np.int64))


def _downsample(length: int, maximum: int) -> np.ndarray:
    return _indices(length, maximum) if length > maximum else np.arange(length, dtype=np.int64)


def _image_data_url(image_bgr: np.ndarray, quality: int) -> str:
    ok, encoded = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("HTML preview JPEG encoding failed")
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")


def _relative_url(target: Path, page: Path) -> str:
    return os.path.relpath(target.resolve(), page.parent.resolve()).replace(os.sep, "/")


def _episode_nav(episode_dir: Path, current: str) -> list[tuple[str, str]]:
    page = episode_dir / "reports" / f"{current}.html"
    links = [("总览", _relative_url(episode_dir.parent / "reports" / "index.html", page))]
    for step in ("step1", "step2", "step3"):
        if step == current or (episode_dir / "reports" / f"{step}.html").is_file():
            links.append((step.upper(), f"{step}.html"))
    return links


def _format_float(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(number) else f"{number:.{digits}f}"


def _eef_speed(state: np.ndarray, ticks_ns: np.ndarray, offset: int) -> tuple[np.ndarray, np.ndarray]:
    dt = np.diff(ticks_ns.astype(np.int64)) / 1e9
    translation = np.zeros(len(state), dtype=np.float64)
    rotation = np.zeros(len(state), dtype=np.float64)
    for index in range(1, len(state)):
        delta = compose(
            invert(vec9_to_transform(state[index - 1, offset : offset + 9])),
            vec9_to_transform(state[index, offset : offset + 9]),
        )
        translation[index] = np.linalg.norm(delta[:3, 3]) / max(dt[index - 1], 1e-9)
        rotation[index] = rotation_angle_deg(delta[:3, :3]) / max(dt[index - 1], 1e-9)
    return translation, rotation


def _pose9_scene_entity(
    vectors: np.ndarray,
    *,
    offset: int,
    name: str,
    color: str,
    kind: str = "hand",
    alpha: float = 1.0,
    trail_alpha: float = 0.38,
    dash: list[int] | None = None,
) -> dict[str, Any]:
    """Convert the 9D Mode2 pose block into a browser scene entity.

    Mode2 stores TCP pose as xyz + the first two rotation columns.  For the
    report we expand it back to SE(3), then draw both the translation trail and
    the local TCP axes in the robot base frame.
    """

    positions: list[list[float]] = []
    axes: list[list[list[float]]] = []
    for row in np.asarray(vectors):
        try:
            transform = vec9_to_transform(row[offset : offset + 9])
        except Exception:
            transform = np.full((4, 4), np.nan, dtype=np.float64)
        positions.append(transform[:3, 3].astype(float).tolist())
        axes.append(
            [
                transform[:3, 0].astype(float).tolist(),
                transform[:3, 1].astype(float).tolist(),
                transform[:3, 2].astype(float).tolist(),
            ]
        )
    entity: dict[str, Any] = {
        "name": name,
        "color": color,
        "kind": kind,
        "positions": _clean(positions),
        "axes": _clean(axes),
        "alpha": alpha,
        "trailAlpha": trail_alpha,
    }
    if dash:
        entity["dash"] = dash
    return entity


def _trajectory_length(positions: np.ndarray) -> float:
    positions = np.asarray(positions, dtype=np.float64)
    if len(positions) < 2:
        return 0.0
    valid = np.isfinite(positions).all(axis=1)
    length = 0.0
    for index in range(1, len(positions)):
        if valid[index - 1] and valid[index]:
            length += float(np.linalg.norm(positions[index] - positions[index - 1]))
    return length


def _workspace_extent(positions: np.ndarray) -> tuple[float, float, float]:
    positions = np.asarray(positions, dtype=np.float64)
    valid = np.isfinite(positions).all(axis=1)
    if not np.any(valid):
        return 0.0, 0.0, 0.0
    span = np.nanmax(positions[valid], axis=0) - np.nanmin(positions[valid], axis=0)
    return tuple(float(value) for value in span)


def _chart(
    title: str,
    x: np.ndarray,
    series: list[tuple[str, np.ndarray]],
    *,
    y_label: str,
    threshold: float | None = None,
) -> dict:
    result: dict[str, Any] = {
        "title": title,
        "x": _clean(np.asarray(x, dtype=np.float64)),
        "xLabel": "time (s)",
        "yLabel": y_label,
        "series": [
            {"name": name, "y": _clean(np.asarray(values, dtype=np.float64)), "color": _COLORS[index % len(_COLORS)]}
            for index, (name, values) in enumerate(series)
        ],
    }
    if threshold is not None:
        result["threshold"] = float(threshold)
    return result


def _missing_report(
    destination: Path,
    *,
    title: str,
    eyebrow: str,
    message: str,
    required: list[Path],
    nav: list[tuple[str, str]],
) -> Path:
    rows = [
        [
            path.name,
            RawHtml(status_badge("存在", "ok") if path.exists() else status_badge("缺失", "bad")),
            str(path),
        ]
        for path in required
    ]
    body = panel("阶段尚未生成", f'<div class="notice">{html.escape(message)}</div>' + table(["文件", "状态", "路径"], rows))
    return write_page(
        destination,
        title=title,
        eyebrow=eyebrow,
        subtitle="就绪检查页面；上游文件出现后重新运行 report 即可刷新。",
        body=body,
        nav=nav,
    )


def _projection_gallery(cfg: ProjectConfig, source: Path, episode_dir: Path) -> str:
    archive_path = episode_dir / "camera" / "hands_camera0.npz"
    if not archive_path.is_file():
        return '<div class="notice">缺少 hands_camera0.npz，无法画手投影。</div>'
    K = np.load(episode_dir / "camera" / "K.npy")
    frames = []
    with np.load(archive_path, allow_pickle=False) as archive, h5py.File(source, "r") as file:
        left = archive["left_joints_camera"]
        right = archive["right_joints_camera"]
        left_valid = archive["left_valid"]
        right_valid = archive["right_valid"]
        images = file["camera/images_left_jpeg"]
        for index in _indices(len(images), cfg.visualization.preview_frames):
            image = cv2.imdecode(np.frombuffer(bytes(images[int(index)]), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                continue
            for points, valid, color, name in (
                (left[index], bool(left_valid[index]), (80, 230, 120), "L"),
                (right[index], bool(right_valid[index]), (80, 180, 255), "R"),
            ):
                for point in points:
                    if point[2] <= 0:
                        continue
                    u = int(round(K[0, 0] * point[0] / point[2] + K[0, 2]))
                    v = int(round(K[1, 1] * point[1] / point[2] + K[1, 2]))
                    if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
                        cv2.circle(image, (u, v), 2, color, -1, cv2.LINE_AA)
                cv2.putText(
                    image,
                    f"{name}:{'valid' if valid else 'invalid'}",
                    (12, 26 if name == "L" else 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            frames.append(
                '<figure class="frame">'
                f'<img src="{_image_data_url(image, cfg.visualization.jpeg_quality)}" alt="frame {index}">'
                f"<figcaption>camera frame {int(index)} · 绿=左手，黄=右手</figcaption></figure>"
            )
    return '<div class="gallery">' + "".join(frames) + "</div>"


def _step1_frame_assets(cfg: ProjectConfig, source: Path, destination: Path) -> tuple[list[str], int, int]:
    """Create browser-friendly frame proxies once; raw HDF5 remains untouched."""
    frame_dir = destination.parent / "assets" / "step1" / "camera"
    frame_dir.mkdir(parents=True, exist_ok=True)
    urls: list[str] = []
    width = height = 0
    with h5py.File(source, "r") as file:
        images = file["camera/images_left_jpeg"]
        for index in range(len(images)):
            output = frame_dir / f"{index:05d}.jpg"
            if not output.is_file():
                image = cv2.imdecode(np.frombuffer(bytes(images[index]), dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f"无法解码 {source.name} camera frame {index}")
                height, width = image.shape[:2]
                if not cv2.imwrite(
                    str(output), image, [cv2.IMWRITE_JPEG_QUALITY, int(cfg.visualization.jpeg_quality)]
                ):
                    raise RuntimeError(f"无法写入调试帧 {output}")
            elif width == 0:
                sample = cv2.imread(str(output))
                if sample is not None:
                    height, width = sample.shape[:2]
            urls.append(_relative_url(output, destination))
    return urls, width, height


def _project_skeleton(points_camera: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    points = np.asarray(points_camera, dtype=np.float64)
    projected = np.full((*points.shape[:-1], 2), np.nan, dtype=np.float64)
    z = points[..., 2]
    valid = np.isfinite(points).all(axis=-1) & (z > 1e-6)
    projected[..., 0][valid] = intrinsic[0, 0] * points[..., 0][valid] / z[valid] + intrinsic[0, 2]
    projected[..., 1][valid] = intrinsic[1, 1] * points[..., 1][valid] / z[valid] + intrinsic[1, 2]
    return projected


def _step1_interactive_data(
    cfg: ProjectConfig,
    source: Path,
    episode_dir: Path,
    destination: Path,
    state: np.ndarray,
    ticks: np.ndarray,
) -> dict[str, Any]:
    frames, width, height = _step1_frame_assets(cfg, source, destination)
    camera_table_path = episode_dir / "sync/camera_frames.parquet"
    if camera_table_path.is_file():
        camera_table = pq.read_table(camera_table_path).to_pydict()
        camera_timestamps = np.asarray(camera_table["timestamp_ns"], dtype=np.int64)
    else:
        camera_table = {}
        with h5py.File(source, "r") as file:
            if "camera/timestamps_ns" in file:
                camera_timestamps = file["camera/timestamps_ns"][:].astype(np.int64)
            else:
                camera_timestamps = np.linspace(ticks[0], ticks[-1], len(frames), dtype=np.int64)
    camera_time_s = (camera_timestamps - camera_timestamps[0]) / 1e9
    control_index, control_gap_ms = nearest_indices(ticks, camera_timestamps)
    action_path = episode_dir / "mode2/action_abs.npy"
    action = np.load(action_path) if action_path.is_file() else np.concatenate([state[1:], state[-1:]], axis=0)
    aligned_state = state[control_index]
    aligned_action = action[control_index]
    intrinsic = np.load(episode_dir / "camera/K.npy")
    with np.load(episode_dir / "camera/hands_camera0.npz", allow_pickle=False) as archive:
        left_camera = archive["left_joints_camera"].astype(np.float64)
        right_camera = archive["right_joints_camera"].astype(np.float64)
        left_valid = archive["left_valid"].astype(bool)
        right_valid = archive["right_valid"].astype(bool)
        left_cam0 = archive["left_joints_camera0"].astype(np.float64) if "left_joints_camera0" in archive else left_camera
        right_cam0 = archive["right_joints_camera0"].astype(np.float64) if "right_joints_camera0" in archive else right_camera
        left_gap = archive["left_gap_ms"].astype(float) if "left_gap_ms" in archive else np.zeros(len(frames))
        right_gap = archive["right_gap_ms"].astype(float) if "right_gap_ms" in archive else np.zeros(len(frames))
        left_pinch = archive["left_pinch_m"].astype(float) if "left_pinch_m" in archive else np.full(len(frames), np.nan)
        right_pinch = archive["right_pinch_m"].astype(float) if "right_pinch_m" in archive else np.full(len(frames), np.nan)
    left_uv = _project_skeleton(left_camera, intrinsic)
    right_uv = _project_skeleton(right_camera, intrinsic)
    body_points: np.ndarray | None = None
    body_valid: np.ndarray | None = None
    with h5py.File(source, "r") as file:
        if "body_pose" in file:
            body_ts_key = "body_timestamps_ns" if "body_timestamps_ns" in file else "timestamps_ns"
            body_ts = file[body_ts_key][:].astype(np.int64)
            body_index, _ = nearest_indices(body_ts, camera_timestamps)
            body_points = file["body_pose"][body_index, :, :3].astype(np.float64)
            if "body_pose_valid" in file:
                body_valid = file["body_pose_valid"][:][body_index].astype(bool)
    stereo_delta = np.asarray(camera_table.get("stereo_pair_delta_ms", np.zeros(len(frames))), dtype=float)
    if not np.any(stereo_delta):
        with h5py.File(source, "r") as file:
            if "camera/stereo_pair_delta_ns" in file:
                stereo_delta = np.abs(file["camera/stereo_pair_delta_ns"][:].astype(float)) / 1e6
    labels = (
        [f"L-pos-{axis}" for axis in "xyz"] + [f"L-rot6-{i}" for i in range(6)]
        + [f"R-pos-{axis}" for axis in "xyz"] + [f"R-rot6-{i}" for i in range(6)]
        + [f"L-Revo2-{name}" for name in ("thumb", "thumbAux", "index", "middle", "ring", "pinky")]
        + [f"R-Revo2-{name}" for name in ("thumb", "thumbAux", "index", "middle", "ring", "pinky")]
    )
    base_axis_length = max(
        0.08,
        0.45
        * max(
            np.nanmax(aligned_state[:, 0]) - np.nanmin(aligned_state[:, 0]) if len(aligned_state) else 0.0,
            np.nanmax(aligned_state[:, 1]) - np.nanmin(aligned_state[:, 1]) if len(aligned_state) else 0.0,
            np.nanmax(aligned_state[:, 2]) - np.nanmin(aligned_state[:, 2]) if len(aligned_state) else 0.0,
            np.nanmax(aligned_state[:, 9]) - np.nanmin(aligned_state[:, 9]) if len(aligned_state) else 0.0,
            np.nanmax(aligned_state[:, 10]) - np.nanmin(aligned_state[:, 10]) if len(aligned_state) else 0.0,
            np.nanmax(aligned_state[:, 11]) - np.nanmin(aligned_state[:, 11]) if len(aligned_state) else 0.0,
        ),
    )
    if not np.isfinite(base_axis_length):
        base_axis_length = 0.12
    data: dict[str, Any] = {
        "timelinePlayers": {
            "step1_capture": {
                "frames": frames,
                "width": width,
                "height": height,
                "fps": float(cfg.realtime.camera_hz),
                "times": _clean(camera_time_s),
                "syncGroup": "step1",
                "layers": [
                    {
                        "name": "左手 26 关节",
                        "color": "#75e6a4",
                        "points": _clean(left_uv),
                        "bones": _HAND_BONES,
                        "valid": _clean(left_valid),
                    },
                    {
                        "name": "右手 26 关节",
                        "color": "#f5c451",
                        "points": _clean(right_uv),
                        "bones": _HAND_BONES,
                        "valid": _clean(right_valid),
                    },
                ],
                "telemetry": [
                    {"label": "camera timestamp", "values": _clean(camera_timestamps), "unit": "ns"},
                    {"label": "control index", "values": _clean(control_index)},
                    {"label": "camera→control gap", "values": _clean(control_gap_ms), "unit": "ms"},
                    {"label": "stereo Δt", "values": _clean(stereo_delta), "unit": "ms"},
                    {"label": "left hand gap", "values": _clean(left_gap), "unit": "ms"},
                    {"label": "right hand gap", "values": _clean(right_gap), "unit": "ms"},
                    {"label": "left pinch", "values": _clean(left_pinch), "unit": "m"},
                    {"label": "right pinch", "values": _clean(right_pinch), "unit": "m"},
                ],
            }
        },
        "scenes": {
            "step1_robot_base": {
                "frames": len(frames),
                "times": _clean(camera_time_s),
                "syncGroup": "step1",
                "baseFrame": {
                    "name": "robot base",
                    "origin": [0.0, 0.0, 0.0],
                    "axisLength": float(base_axis_length),
                },
                "referencePoints": [[0.0, 0.0, 0.0]],
                "note": "G1 robot base 系；实线=state[t] TCP 轨迹，虚线=action[t] 下一时刻绝对目标；RGB 小轴为 TCP 姿态",
                "entities": [
                    _pose9_scene_entity(aligned_state, offset=0, name="L TCP state", color="#42d9d0"),
                    _pose9_scene_entity(aligned_state, offset=9, name="R TCP state", color="#f5c451"),
                    _pose9_scene_entity(
                        aligned_action,
                        offset=0,
                        name="L action target",
                        color="#42d9d0",
                        alpha=0.45,
                        trail_alpha=0.16,
                        dash=[7, 5],
                    ),
                    _pose9_scene_entity(
                        aligned_action,
                        offset=9,
                        name="R action target",
                        color="#f5c451",
                        alpha=0.45,
                        trail_alpha=0.16,
                        dash=[7, 5],
                    ),
                ],
            }
        },
        "skeletons3d": {
            "step1_hands_cam0": {
                "frames": len(frames),
                "times": _clean(camera_time_s),
                "syncGroup": "step1",
                "note": "cam0 坐标系；轨迹单位 m；拖拽旋转；与左侧图像同帧",
                "skeletons": [
                    {"name": "left hand", "color": "#75e6a4", "points": _clean(left_cam0), "bones": _HAND_BONES, "valid": _clean(left_valid), "trailJoint": 1},
                    {"name": "right hand", "color": "#f5c451", "points": _clean(right_cam0), "bones": _HAND_BONES, "valid": _clean(right_valid), "trailJoint": 1},
                ],
            }
        },
        "vectorViews": {
            "step1_mode2": {
                "title": "Mode2 state[t] 与 action[t]（下一时刻绝对目标）",
                "current": _clean(aligned_state),
                "target": _clean(aligned_action),
                "labels": labels,
                "times": _clean(camera_time_s),
                "syncGroup": "step1",
                "groups": [
                    {"name": "left EEF 9D", "start": 0, "end": 9, "color": "#42d9d0"},
                    {"name": "right EEF 9D", "start": 9, "end": 18, "color": "#76a9ff"},
                    {"name": "left Revo2", "start": 18, "end": 24, "color": "#75e6a4"},
                    {"name": "right Revo2", "start": 24, "end": 30, "color": "#f5c451"},
                ],
            }
        },
    }
    if body_points is not None:
        data["skeletons3d"]["step1_body"] = {
            "frames": len(frames),
            "times": _clean(camera_time_s),
            "syncGroup": "step1",
            "note": "PICO BodyTracking SDK local 原始坐标；只用于采集 QA，不是 robot base",
            "skeletons": [
                {
                    "name": "PICO body 24",
                    "color": "#d697ff",
                    "points": _clean(body_points),
                    "bones": _BODY_BONES,
                    "valid": _clean(body_valid) if body_valid is not None else [True] * len(frames),
                    "trailJoint": 0,
                }
            ],
        }
    return data


def _step1_mujoco_player(
    cfg: ProjectConfig,
    episode_dir: Path,
    destination: Path,
    state: np.ndarray,
    action: np.ndarray,
    ticks: np.ndarray,
) -> tuple[dict[str, Any] | None, str | None]:
    """Render the real Revo2 MJCF at every Mode2 control tick."""
    try:
        from ego_relation.visualization.mujoco_hands import render_revo2_replay

        paths, metadata = render_revo2_replay(
            state,
            action,
            cfg.paths.assets_dir / "revo2",
            destination.parent / "assets/step1/mujoco_revo2_mode2_v4",
            jpeg_quality=cfg.visualization.jpeg_quality,
        )
    except Exception as error:  # The report must still expose why rendering is unavailable.
        return None, f"{type(error).__name__}: {error}"
    time_s = (ticks - ticks[0]) / 1e9
    player = {
        "frames": [_relative_url(path, destination) for path in paths],
        "width": 960,
        "height": 540,
        "fps": float(cfg.timeline.control_hz),
        "frameName": "control",
        "times": _clean(time_s),
        "syncGroup": "step1",
        "layers": [],
        "telemetry": [
            {"label": "control frame", "values": list(range(len(paths)))},
            {"label": "world frame", "values": [metadata["world_frame"]] * len(paths)},
            {"label": "action left TCP x", "values": _clean(action[:, 0]), "unit": "m"},
            {"label": "action left TCP y", "values": _clean(action[:, 1]), "unit": "m"},
            {"label": "action left TCP z", "values": _clean(action[:, 2]), "unit": "m"},
            {"label": "action right TCP x", "values": _clean(action[:, 9]), "unit": "m"},
            {"label": "action right TCP y", "values": _clean(action[:, 10]), "unit": "m"},
            {"label": "action right TCP z", "values": _clean(action[:, 11]), "unit": "m"},
            {"label": "state left finger mean", "values": _clean(state[:, 18:24].mean(axis=1))},
            {"label": "state right finger mean", "values": _clean(state[:, 24:30].mean(axis=1))},
        ],
    }
    return player, None


def generate_step1_report(cfg: ProjectConfig, source: Path, episode_dir: Path | None = None) -> Path:
    episode_dir = episode_dir or episode_work_dir(cfg, source)
    destination = episode_dir / "reports" / "step1.html"
    required = [
        episode_dir / "pico_ingest.manifest.json",
        episode_dir / "mode2.manifest.json",
        episode_dir / "mode2/state_abs.npy",
        episode_dir / "mode2/action_abs.npy",
        episode_dir / "camera/K.npy",
    ]
    if not all(path.is_file() for path in required):
        return _missing_report(
            destination,
            title=f"{source.stem} · Step1",
            eyebrow="PICO / MODE2",
            message="先执行 Step1，页面随后会自动刷新。",
            required=required,
            nav=_episode_nav(episode_dir, "step1"),
        )
    pico = _read_json(episode_dir / "qa/pico_report.json", {})
    mode2 = _read_json(episode_dir / "mode2.manifest.json", {})
    task_contract = _read_json(episode_dir / "step1/task_semantics.json", {})
    object_contract = _read_json(episode_dir / "step1/object_instances.json", {})
    action_contract = _read_json(episode_dir / "step1/action_contract.json", {})
    timeline_qa = _read_json(episode_dir / "qa/step1_timeline_qa.json", {})
    step1_quality = _read_json(episode_dir / "qa/step1_quality_report.json", {})
    metrics_mode2 = mode2.get("metrics", {})
    state = np.load(episode_dir / "mode2/state_abs.npy")
    action = np.load(episode_dir / "mode2/action_abs.npy")
    ticks = np.load(episode_dir / "mode2/ticks_ns.npy")
    time_s = (ticks - ticks[0]) / 1e9
    left_speed, left_rot_speed = _eef_speed(state, ticks, 0)
    right_speed, right_rot_speed = _eef_speed(state, ticks, 9)
    camera_pose = np.load(episode_dir / "camera/T_camera_to_camera0.npy")
    camera_time = np.linspace(0, time_s[-1] if len(time_s) else 0, len(camera_pose))
    index = _downsample(len(state), cfg.visualization.max_chart_points)
    camera_index = _downsample(len(camera_pose), cfg.visualization.max_chart_points)
    interactive = _step1_interactive_data(cfg, source, episode_dir, destination, state, ticks)
    mujoco_player, mujoco_error = _step1_mujoco_player(cfg, episode_dir, destination, state, action, ticks)
    if mujoco_player is not None:
        interactive["timelinePlayers"]["step1_mujoco_revo2"] = mujoco_player

    charts = {
        "eef_position": _chart(
            "左右 TCP 平移轨迹",
            time_s[index],
            [("L-x", state[index, 0]), ("L-y", state[index, 1]), ("L-z", state[index, 2]),
             ("R-x", state[index, 9]), ("R-y", state[index, 10]), ("R-z", state[index, 11])],
            y_label="m",
        ),
        "eef_speed": _chart(
            "TCP 平移速度（跳变定位）",
            time_s[index],
            [("left", left_speed[index]), ("right", right_speed[index])],
            y_label="m/s",
            threshold=cfg.mode2.maximum_eef_speed_m_s,
        ),
        "eef_rot_speed": _chart(
            "TCP 旋转速度",
            time_s[index],
            [("left", left_rot_speed[index]), ("right", right_rot_speed[index])],
            y_label="deg/s",
            threshold=cfg.mode2.maximum_eef_rotation_deg_s,
        ),
        "brainco": _chart(
            "BrainCo Revo2 目标关节",
            time_s[index],
            [(f"L{joint}", state[index, 18 + joint]) for joint in range(6)]
            + [(f"R{joint}", state[index, 24 + joint]) for joint in range(6)],
            y_label="normalized joint",
        ),
        "camera": _chart(
            "头显相机相对 cam0 轨迹",
            camera_time[camera_index],
            [("x", camera_pose[camera_index, 0, 3]), ("y", camera_pose[camera_index, 1, 3]),
             ("z", camera_pose[camera_index, 2, 3])],
            y_label="m",
        ),
    }
    for chart in charts.values():
        chart["syncGroup"] = "step1"
    geometry_ok = bool(pico.get("geometry_deployable"))
    motion_ok = bool(metrics_mode2.get("motion_deployable"))
    stereo = _read_json(episode_dir / "qa/stereo_depth_report.json", {})
    left_extent = _workspace_extent(state[:, 0:3])
    right_extent = _workspace_extent(state[:, 9:12])
    cards = [
        ("Mode2 帧数", str(len(state)), f"{metrics_mode2.get('fps', cfg.timeline.control_hz)} Hz"),
        (
            "30 Hz 对齐",
            status_badge("通过", "ok")
            if timeline_qa.get("target_30hz_verified")
            else status_badge("未通过", "bad"),
            f"有效率 {_format_float(timeline_qa.get('combined_valid_ratio', 0) * 100, 2)}%",
        ),
        (
            "Step1 总质检",
            status_badge("通过", "ok")
            if step1_quality.get("accepted")
            else status_badge("未通过", "bad"),
            "时间 / 几何 / 动作 / 连续性",
        ),
        ("动作连续性", status_badge("通过", "ok") if motion_ok else status_badge("隔离", "bad"), "完整 episode 粒度"),
        ("相机几何", status_badge("通过", "ok") if geometry_ok else status_badge("阻止", "bad"), "K / 外参 / 手投影"),
        ("双目 baseline", f"{_format_float(pico.get('extrinsic_self_qa', {}).get('baseline_m', 0) * 100, 2)} cm", "PICO 左右目"),
        ("相机同步 p95", f"{_format_float(pico.get('stereo_delta_p95_ms'))} ms", "左右目曝光"),
        ("双目深度", status_badge("通过", "ok") if stereo.get("stereo_depth_deployable") else status_badge("未运行", "warn"), "Step2 前置"),
        ("左 TCP 轨迹长度", f"{_trajectory_length(state[:, 0:3]):.3f} m", f"范围 x/y/z={left_extent[0]:.2f}/{left_extent[1]:.2f}/{left_extent[2]:.2f}"),
        ("右 TCP 轨迹长度", f"{_trajectory_length(state[:, 9:12]):.3f} m", f"范围 x/y/z={right_extent[0]:.2f}/{right_extent[1]:.2f}/{right_extent[2]:.2f}"),
    ]
    warnings = list(pico.get("warnings", [])) + list(mode2.get("warnings", []))
    violation_rows = [[index + 1, value] for index, value in enumerate(metrics_mode2.get("motion_violations", []))]
    body = metrics(cards)
    if timeline_qa:
        source_rates = timeline_qa.get("source_rates_hz", {})
        stream_rows = []
        for name, stream in timeline_qa.get("streams", {}).items():
            stream_rows.append(
                [
                    name,
                    _format_float(source_rates.get(f"{name}_hz")),
                    _format_float(stream.get("p95_ms")),
                    _format_float(stream.get("max_ms")),
                    f"{_format_float(stream.get('valid_ratio', 0) * 100, 2)}%",
                    stream.get("repeated_control_frames", 0),
                ]
            )
        body += panel(
            "30 Hz 多流时间对齐质检",
            table(
                ["stream", "source Hz", "gap p95 ms", "gap max ms", "valid", "repeated"],
                stream_rows,
            ),
        )
    if task_contract or object_contract or action_contract:
        contract_rows = [
            ["任务语义", task_contract.get("instruction", "—")],
            ["动作坐标", action_contract.get("pose_transform", "—")],
            ["动作监督", action_contract.get("action_semantics", "—")],
            ["30 维布局", "left TCP 9 + right TCP 9 + left Revo2 6 + right Revo2 6"],
        ]
        body += panel("Step1 任务与动作契约", table(["项目", "定义"], contract_rows))
        instances = object_contract.get("instances", [])
        body += panel(
            "Step1 对象实例",
            table(
                ["instance_id", "category", "prompt", "anchor"],
                [
                    [row.get("instance_id"), row.get("category"), row.get("prompt"), row.get("is_anchor")]
                    for row in instances
                ],
            ),
        )
    if mujoco_player is not None:
        body += panel(
            "固定 robot base 系 · BrainCo Revo2 MuJoCo 回放",
            timeline_player("step1_mujoco_revo2")
            + '<p class="legend-note">参考 egodata_targeting_project 的 Mode2 逻辑：左侧为 action[t] 的 robot-base TCP 目标，经固定 TCP→Revo2 palm 安装变换驱动；右上/右下为固定视角的 state[t] 左右手指近景。场景只包含 BrainCo Revo2，不包含 G1 或人体。</p>',
        )
    else:
        body += panel(
            "固定 robot base 系 · BrainCo Revo2 MuJoCo 回放",
            '<div class="notice bad">MuJoCo 真实手模型回放生成失败，下面的轨迹图不能替代它。'
            + html.escape(mujoco_error or "unknown error")
            + "</div>",
        )
    body += panel(
        "逐帧 PICO 数据播放器",
        timeline_player("step1_capture")
        + '<p class="legend-note">这是全部相机帧，不是抽样图。左右手 26 关节由同一曝光时刻的相机系坐标重投影；右侧同时显示同步误差、捏合距离和控制帧索引。</p>',
    )
    body += panel(
        "机器人 base 系下左右 TCP 轨迹",
        scene3d("step1_robot_base")
        + '<p class="legend-note">这里把 Mode2 的左右 TCP 9D pose 还原成 SE(3)：青色/黄色实线是 state[t]，虚线是 action[t]（下一时刻绝对目标）。原点为 G1 robot base，红/绿/蓝轴分别表示 x/y/z。若左右镜像、尺度漂移或 action 延迟错位，这个面板会非常明显。</p>',
    )
    skeleton_panels = panel("cam0 双手三维骨架", skeleton3d("step1_hands_cam0"))
    if "step1_body" in interactive["skeletons3d"]:
        skeleton_panels += panel("PICO 24 关节人体骨架", skeleton3d("step1_body"))
    body += '<div class="grid2">' + skeleton_panels + "</div>"
    body += panel(
        "当前 Mode2 30 维标签",
        vector_view("step1_mode2")
        + '<p class="legend-note">实心柱是 state[t]，白框是 action[t]。拖动任意一个 Step1 滑块，图像、三维骨架、标签和曲线游标会联动。</p>',
    )
    body += '<div class="grid2">' + panel("EEF 位置", line_chart("eef_position")) + panel("EEF 速度", line_chart("eef_speed")) + "</div>"
    body += '<div class="grid2">' + panel("旋转速度", line_chart("eef_rot_speed")) + panel("BrainCo", line_chart("brainco")) + "</div>"
    body += panel("相机运动", line_chart("camera"))
    body += panel("手势重投影抽检（导出快照）", _projection_gallery(cfg, source, episode_dir))
    if warnings or violation_rows:
        rows = [["warning", warning] for warning in warnings] + [["violation", row[1]] for row in violation_rows]
        body += panel("QA 警告与隔离原因", table(["类型", "内容"], rows))
    return write_page(
        destination,
        title=f"{source.stem} · Step1",
        eyebrow="PICO / MODE2 / CAMERA QA",
        subtitle="检查动作标签、头显/手势坐标、相机外参与时间同步；红色虚线是部署阈值。",
        body=body,
        data={**interactive, "charts": charts},
        nav=_episode_nav(episode_dir, "step1"),
    )


def _relation_overlay_gallery(cfg: ProjectConfig, episode_dir: Path) -> str:
    preprocess = episode_dir / "perception/humanego_session/preprocess"
    image_paths = sorted((preprocess / "all_data").glob("*/rgb.png"))
    tracks = _read_json(preprocess / "cotracker_results.json", {})
    catalog = _read_json(preprocess / "object_catalog.json", [])
    if not image_paths:
        return '<div class="notice">没有 HumanEgo staged RGB，无法生成 2D 轨迹叠加图。</div>'
    rows = []
    for frame in _indices(len(image_paths), cfg.visualization.preview_frames):
        image = cv2.imread(str(image_paths[int(frame)]))
        if image is None:
            continue
        for object_index, item in enumerate(catalog):
            instance = str(item["instance_id"])
            color = tuple(int(value) for value in np.array([70, 210, 240]) * (0.75 + 0.1 * (object_index % 3)))
            entry = tracks.get(instance, {})
            track = np.asarray(entry.get("tracks", []), dtype=np.float64)
            visible = np.asarray(entry.get("visibility", []), dtype=np.float64)
            if track.ndim == 3 and int(frame) < len(track):
                for point_index, point in enumerate(track[int(frame)]):
                    if visible.ndim == 2 and visible[int(frame), point_index] < cfg.perception.visibility_threshold:
                        continue
                    cv2.circle(image, tuple(np.rint(point).astype(int)), 3, color, -1, cv2.LINE_AA)
            if int(frame) == 0:
                mask = cv2.imread(str(image_paths[0].parent / f"mask_{instance}.png"), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(image, contours, -1, color, 2, cv2.LINE_AA)
            cv2.putText(image, instance, (12, 24 + object_index * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        rows.append(
            '<figure class="frame">'
            f'<img src="{_image_data_url(image, cfg.visualization.jpeg_quality)}" alt="track {frame}">'
            f"<figcaption>HumanEgo frame {int(frame)} · frame0 同时显示 SAM2 轮廓</figcaption></figure>"
        )
    return '<div class="gallery">' + "".join(rows) + "</div>"


def _caption_image(image: np.ndarray, label: str, color: tuple[int, int, int] = (230, 230, 230)) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (min(output.shape[1], 310), 35), (5, 12, 18), -1)
    cv2.putText(output, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 1, cv2.LINE_AA)
    return output


def _step2_depth_player(cfg: ProjectConfig, episode_dir: Path, destination: Path) -> tuple[dict[str, Any], dict[str, Any]] | None:
    depth_dir = episode_dir / "depth/stereo_mm"
    depth_paths = sorted(depth_dir.glob("*.png"))
    preprocess = episode_dir / "perception/humanego_session/preprocess"
    rgb_paths = sorted((preprocess / "all_data").glob("*/rgb.png"))
    if not depth_paths or not rgb_paths:
        return None
    output_dir = destination.parent / "assets/step2/stereo_depth"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_urls: list[str] = []
    frame_indices: list[int] = []
    coverage: list[float] = []
    median_depth: list[float] = []
    near_depth: list[float] = []
    far_depth: list[float] = []
    output_width = output_height = 0
    for depth_path in depth_paths:
        frame_index = int(depth_path.stem)
        if frame_index >= len(rgb_paths):
            continue
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        rgb = cv2.imread(str(rgb_paths[frame_index]))
        if depth is None or rgb is None:
            continue
        valid = depth > 0
        values_m = depth[valid].astype(np.float64) / 1000.0
        coverage.append(float(valid.mean()))
        median_depth.append(float(np.median(values_m)) if len(values_m) else np.nan)
        near_depth.append(float(np.percentile(values_m, 5)) if len(values_m) else np.nan)
        far_depth.append(float(np.percentile(values_m, 95)) if len(values_m) else np.nan)
        normalized = np.clip(
            (depth.astype(np.float32) / 1000.0 - cfg.depth.min_depth_m)
            / max(cfg.depth.max_depth_m - cfg.depth.min_depth_m, 1e-6),
            0,
            1,
        )
        depth_color = cv2.applyColorMap(np.uint8((1.0 - normalized) * 255), cv2.COLORMAP_TURBO)
        depth_color[~valid] = 0
        if depth_color.shape[:2] != rgb.shape[:2]:
            depth_color = cv2.resize(depth_color, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        joined = cv2.hconcat(
            [
                _caption_image(rgb, "PICO left RGB"),
                _caption_image(depth_color, f"Stereo depth {cfg.depth.min_depth_m:.2f}-{cfg.depth.max_depth_m:.2f} m"),
            ]
        )
        output = output_dir / f"{frame_index:05d}.jpg"
        if not output.is_file():
            cv2.imwrite(str(output), joined, [cv2.IMWRITE_JPEG_QUALITY, int(cfg.visualization.jpeg_quality)])
        output_height, output_width = joined.shape[:2]
        frame_urls.append(_relative_url(output, destination))
        frame_indices.append(frame_index)
    if not frame_urls:
        return None
    times = np.asarray(frame_indices, dtype=np.float64) / max(cfg.realtime.camera_hz, 1e-6)
    player = {
        "frames": frame_urls,
        "width": output_width,
        "height": output_height,
        "fps": cfg.realtime.camera_hz,
        "times": _clean(times),
        "syncGroup": "step2_camera",
        "layers": [],
        "telemetry": [
            {"label": "camera frame", "values": frame_indices},
            {"label": "valid depth coverage", "values": _clean(np.asarray(coverage) * 100), "unit": "%"},
            {"label": "depth p05", "values": _clean(near_depth), "unit": "m"},
            {"label": "depth median", "values": _clean(median_depth), "unit": "m"},
            {"label": "depth p95", "values": _clean(far_depth), "unit": "m"},
        ],
    }
    chart = _chart(
        "双目深度有效覆盖率",
        times,
        [("coverage", np.asarray(coverage) * 100), ("median depth", np.asarray(median_depth))],
        y_label="% / m",
    )
    chart["syncGroup"] = "step2_camera"
    return player, chart


def _step2_reference_gallery(cfg: ProjectConfig, episode_dir: Path) -> str:
    preprocess = episode_dir / "perception/humanego_session/preprocess"
    reference = preprocess / "all_data/00000/rgb.png"
    if not reference.is_file():
        return '<div class="notice">尚未生成 HumanEgo reference frame。</div>'
    raw = cv2.imread(str(reference))
    if raw is None:
        return '<div class="notice">reference frame 无法读取。</div>'
    catalog = _read_json(preprocess / "object_catalog.json", [])
    mask_overlay = raw.copy()
    mask_found = False
    for object_index, row in enumerate(catalog):
        instance = str(row.get("instance_id", f"obj{object_index + 1}"))
        mask_path = reference.parent / f"mask_{instance}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.is_file() else None
        if mask is None:
            continue
        mask_found = True
        color = np.asarray(
            ((117, 230, 164), (245, 196, 81), (118, 169, 255), (214, 151, 255))[object_index % 4],
            dtype=np.float32,
        )
        alpha = (mask.astype(np.float32) / 255.0 * 0.46)[..., None]
        mask_overlay = np.uint8(mask_overlay.astype(np.float32) * (1 - alpha) + color[None, None] * alpha)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask_overlay, contours, -1, tuple(int(v) for v in color), 2, cv2.LINE_AA)
        cv2.putText(mask_overlay, instance, (12, 26 + object_index * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, tuple(int(v) for v in color), 1, cv2.LINE_AA)
    keypoint_overlay = mask_overlay.copy()
    keypoints = _read_json(preprocess / "kptsselector_results.json", {}).get("objects", {})
    for object_index, (instance, points) in enumerate(keypoints.items()):
        color = ((117, 230, 164), (245, 196, 81), (118, 169, 255), (214, 151, 255))[object_index % 4]
        for point_index, point in enumerate(points):
            center = tuple(np.rint(point).astype(int))
            cv2.circle(keypoint_overlay, center, 4, color, -1, cv2.LINE_AA)
            cv2.putText(keypoint_overlay, str(point_index), (center[0] + 4, center[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1)
    rows = [
        '<figure class="frame">'
        f'<img src="{_image_data_url(raw, cfg.visualization.jpeg_quality)}" alt="reference RGB">'
        '<figcaption>Reference RGB</figcaption></figure>'
    ]
    dino = preprocess / "dinosam_reference.jpg"
    if dino.is_file():
        dino_image = cv2.imread(str(dino))
        if dino_image is not None:
            rows.append(
                '<figure class="frame">'
                f'<img src="{_image_data_url(dino_image, cfg.visualization.jpeg_quality)}" alt="Grounding DINO">'
                '<figcaption>Grounding DINO boxes + SAM2 preview</figcaption></figure>'
            )
    if mask_found:
        rows.append(
            '<figure class="frame">'
            f'<img src="{_image_data_url(mask_overlay, cfg.visualization.jpeg_quality)}" alt="SAM2 masks">'
            '<figcaption>SAM2 instance masks</figcaption></figure>'
        )
    if keypoints:
        rows.append(
            '<figure class="frame">'
            f'<img src="{_image_data_url(keypoint_overlay, cfg.visualization.jpeg_quality)}" alt="sampled keypoints">'
            '<figcaption>轮廓采样关键点与编号</figcaption></figure>'
        )
    return '<div class="gallery">' + "".join(rows) + "</div>"


def _step2_tracker_player(cfg: ProjectConfig, episode_dir: Path, destination: Path) -> dict[str, Any] | None:
    preprocess = episode_dir / "perception/humanego_session/preprocess"
    image_paths = sorted((preprocess / "all_data").glob("*/rgb.png"))
    tracks = _read_json(preprocess / "cotracker_results.json", {})
    catalog = _read_json(preprocess / "object_catalog.json", [])
    if not image_paths or not any(str(key).startswith("obj") for key in tracks):
        return None
    sample = cv2.imread(str(image_paths[0]))
    if sample is None:
        return None
    layers = []
    telemetry = [{"label": "tracking frame", "values": list(range(len(image_paths)))}]
    for object_index, row in enumerate(catalog):
        instance = str(row.get("instance_id", f"obj{object_index + 1}"))
        entry = tracks.get(instance, {})
        points = np.asarray(entry.get("tracks", []), dtype=np.float64)
        visibility = np.asarray(entry.get("visibility", []), dtype=np.float64)
        if points.ndim != 3:
            continue
        points = points[: len(image_paths)].copy()
        if visibility.shape == points.shape[:2]:
            points[visibility[: len(points)] < cfg.perception.visibility_threshold] = np.nan
            visible_count = np.sum(visibility[: len(points)] >= cfg.perception.visibility_threshold, axis=1)
        else:
            visible_count = np.isfinite(points).all(axis=2).sum(axis=1)
        color = ("#75e6a4", "#f5c451", "#76a9ff", "#d697ff", "#ff7185")[object_index % 5]
        layers.append({"name": f"{instance} tracks", "color": color, "points": _clean(points), "bones": []})
        telemetry.append({"label": f"{instance} visible", "values": _clean(visible_count), "unit": "pts"})
    if not layers:
        return None
    return {
        "frames": [_relative_url(path, destination) for path in image_paths],
        "width": int(sample.shape[1]),
        "height": int(sample.shape[0]),
        "fps": cfg.realtime.camera_hz,
        "times": _clean(np.arange(len(image_paths)) / max(cfg.realtime.camera_hz, 1e-6)),
        "syncGroup": "step2_camera",
        "layers": layers,
        "telemetry": telemetry,
    }


def _draw_instance_boxes(image: np.ndarray, catalog: list[dict], reference_dir: Path) -> np.ndarray:
    output = image.copy()
    colors = ((117, 230, 164), (245, 196, 81), (118, 169, 255), (214, 151, 255))
    for object_index, row in enumerate(catalog):
        instance = str(row.get("instance_id", f"obj{object_index + 1}"))
        mask = cv2.imread(str(reference_dir / f"mask_{instance}.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        points = cv2.findNonZero(mask)
        if points is None:
            continue
        x, y, width, height = cv2.boundingRect(points)
        color = colors[object_index % len(colors)]
        cv2.rectangle(output, (x, y), (x + width, y + height), color, 2, cv2.LINE_AA)
        prompt = str(row.get("prompt", row.get("category", instance))).strip()
        cv2.putText(output, f"{instance}: {prompt}", (x, max(20, y - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    return output


def _draw_reference_masks(image: np.ndarray, catalog: list[dict], reference_dir: Path) -> np.ndarray:
    output = image.copy()
    colors = ((117, 230, 164), (245, 196, 81), (118, 169, 255), (214, 151, 255))
    for object_index, row in enumerate(catalog):
        instance = str(row.get("instance_id", f"obj{object_index + 1}"))
        mask = cv2.imread(str(reference_dir / f"mask_{instance}.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        color = np.asarray(colors[object_index % len(colors)], dtype=np.float32)
        alpha = (mask.astype(np.float32) / 255.0 * 0.52)[..., None]
        output = np.uint8(output.astype(np.float32) * (1 - alpha) + color[None, None] * alpha)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, tuple(int(value) for value in color), 2, cv2.LINE_AA)
    return output


def _draw_points(image: np.ndarray, tracks: dict, catalog: list[dict], frame: int, *, trail: int = 0) -> tuple[np.ndarray, int]:
    output = image.copy()
    colors = ((117, 230, 164), (245, 196, 81), (118, 169, 255), (214, 151, 255))
    visible_total = 0
    for object_index, row in enumerate(catalog):
        instance = str(row.get("instance_id", f"obj{object_index + 1}"))
        entry = tracks.get(instance, {})
        points = np.asarray(entry.get("tracks", []), dtype=np.float64)
        visibility = np.asarray(entry.get("visibility", []), dtype=np.float64)
        if points.ndim != 3 or frame >= len(points):
            continue
        color = colors[object_index % len(colors)]
        for point_index, point in enumerate(points[frame]):
            visible = not (visibility.ndim == 2 and frame < len(visibility) and point_index < visibility.shape[1] and visibility[frame, point_index] < 0.5)
            if not visible or not np.isfinite(point).all():
                continue
            visible_total += 1
            center = tuple(np.rint(point).astype(int))
            if trail:
                history = points[max(0, frame - trail): frame + 1, point_index]
                history = history[np.isfinite(history).all(axis=1)]
                if len(history) > 1:
                    cv2.polylines(output, [np.rint(history).astype(np.int32)], False, color, 1, cv2.LINE_AA)
            cv2.circle(output, center, 3, color, -1, cv2.LINE_AA)
        cv2.putText(output, instance, (12, 52 + object_index * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    return output, visible_total


def _draw_object_pose_axes(
    image: np.ndarray,
    intrinsic: np.ndarray,
    T_camera0_camera: np.ndarray,
    T_camera0_object: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, int]:
    output = image.copy()
    valid_count = 0
    T_camera_camera0 = np.linalg.inv(T_camera0_camera)
    axis_colors = ((50, 50, 245), (50, 220, 80), (245, 120, 40))
    for object_index, pose in enumerate(T_camera0_object):
        if object_index >= len(valid) or not valid[object_index] or not np.isfinite(pose).all():
            continue
        current = T_camera_camera0 @ pose
        scale = 0.055
        points = np.stack((current[:3, 3], *[current[:3, 3] + current[:3, axis] * scale for axis in range(3)]))
        if np.any(points[:, 2] <= 1e-5):
            continue
        uv = np.column_stack((
            intrinsic[0, 0] * points[:, 0] / points[:, 2] + intrinsic[0, 2],
            intrinsic[1, 1] * points[:, 1] / points[:, 2] + intrinsic[1, 2],
        ))
        origin = tuple(np.rint(uv[0]).astype(int))
        cv2.circle(output, origin, 5, (255, 255, 255), -1, cv2.LINE_AA)
        for axis, color in enumerate(axis_colors):
            cv2.arrowedLine(output, origin, tuple(np.rint(uv[axis + 1]).astype(int)), color, 2, cv2.LINE_AA, tipLength=0.25)
        cv2.putText(output, f"obj{object_index + 1}", (origin[0] + 7, origin[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 245, 245), 1, cv2.LINE_AA)
        valid_count += 1
    return output, valid_count


def _inactive_stage(image: np.ndarray, message: str) -> np.ndarray:
    output = np.uint8(image.astype(np.float32) * 0.30)
    lines = ("REFERENCE FRAME ONLY", message)
    for line_index, line in enumerate(lines):
        cv2.putText(output, line, (18, output.shape[0] // 2 + line_index * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (180, 200, 215), 1, cv2.LINE_AA)
    return output


def _step2_full_pipeline_player(cfg: ProjectConfig, episode_dir: Path, destination: Path) -> tuple[dict[str, Any] | None, bool]:
    """Build one all-frame view of every actual Step2 intermediate stage."""
    preprocess = episode_dir / "perception/humanego_session/preprocess"
    image_paths = sorted((preprocess / "all_data").glob("*/rgb.png"))
    if not image_paths:
        return None, False
    catalog = _read_json(preprocess / "object_catalog.json", [])
    tracks = _read_json(preprocess / "cotracker_results.json", {})
    keypoints = _read_json(preprocess / "kptsselector_results.json", {}).get("objects", {})
    manifest = _read_json(episode_dir / "humanego_perception.manifest.json", {})
    fallback = bool(manifest.get("metrics", {}).get("fallback", False))
    method = str(manifest.get("config", {}).get("method", "unknown"))
    dino_path = preprocess / "dinosam_reference.jpg"
    dino_reference = cv2.imread(str(dino_path)) if dino_path.is_file() else None
    reference_dir = image_paths[0].parent
    depth_paths = {int(path.stem): path for path in (episode_dir / "depth/stereo_mm").glob("*.png")}
    object_track_path = episode_dir / "entities/objects_stereo_track.npz"
    if object_track_path.is_file():
        with np.load(object_track_path, allow_pickle=False) as archive:
            object_poses = archive["T_camera0_object"].astype(np.float64)
            object_valid = archive["valid"].astype(bool)
    else:
        object_poses = np.zeros((len(image_paths), 0, 4, 4), dtype=np.float64)
        object_valid = np.zeros((len(image_paths), 0), dtype=bool)
    intrinsic = np.load(episode_dir / "camera/K.npy")
    camera_poses = np.load(episode_dir / "camera/T_camera_to_camera0.npy")

    output_dir = destination.parent / "assets/step2" / f"full_pipeline_{method}"
    output_dir.mkdir(parents=True, exist_ok=True)
    urls: list[str] = []
    visible_counts: list[int] = []
    pose_counts: list[int] = []
    depth_coverage: list[float] = []
    tile_size = (320, 240)
    for frame, image_path in enumerate(image_paths):
        raw = cv2.imread(str(image_path))
        if raw is None:
            continue
        if frame == 0:
            detection = dino_reference.copy() if dino_reference is not None and not fallback else _draw_instance_boxes(raw, catalog, reference_dir)
            segmentation = _draw_reference_masks(raw, catalog, reference_dir)
            sampled = segmentation.copy()
            for object_index, row in enumerate(catalog):
                instance = str(row.get("instance_id", f"obj{object_index + 1}"))
                color = ((117, 230, 164), (245, 196, 81), (118, 169, 255), (214, 151, 255))[object_index % 4]
                for point_index, point in enumerate(keypoints.get(instance, [])):
                    center = tuple(np.rint(point).astype(int))
                    cv2.circle(sampled, center, 3, color, -1, cv2.LINE_AA)
                    if point_index % 4 == 0:
                        cv2.putText(sampled, str(point_index), (center[0] + 3, center[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        else:
            detection = _inactive_stage(raw, "Grounding DINO not invoked")
            segmentation = _inactive_stage(raw, "SAM2 instance mask not invoked")
            sampled = _inactive_stage(raw, "contour sampler not invoked")
        tracked, visible_count = _draw_points(raw, tracks, catalog, frame, trail=14)
        visible_counts.append(visible_count)

        depth = cv2.imread(str(depth_paths.get(frame)), cv2.IMREAD_UNCHANGED) if frame in depth_paths else None
        if depth is None:
            depth_view = np.zeros_like(raw)
            coverage = 0.0
        else:
            valid_depth = depth > 0
            coverage = float(valid_depth.mean()) * 100
            normalized = np.clip((depth.astype(np.float32) / 1000.0 - cfg.depth.min_depth_m) / max(cfg.depth.max_depth_m - cfg.depth.min_depth_m, 1e-6), 0, 1)
            depth_view = cv2.applyColorMap(np.uint8((1.0 - normalized) * 255), cv2.COLORMAP_TURBO)
            depth_view[~valid_depth] = 0
        depth_coverage.append(coverage)

        if frame < len(object_poses) and frame < len(camera_poses):
            pose_view, pose_count = _draw_object_pose_axes(raw, intrinsic, camera_poses[frame], object_poses[frame], object_valid[frame])
        else:
            pose_view, pose_count = raw.copy(), 0
        pose_counts.append(pose_count)
        authenticity = "DEBUG FALLBACK - MODELS NOT RUN" if fallback else "REAL MODEL OUTPUT"
        cv2.putText(pose_view, authenticity, (12, pose_view.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (45, 70, 245) if fallback else (80, 225, 130), 2, cv2.LINE_AA)
        status_view = np.full_like(raw, (18, 25, 31))
        status_lines = (
            f"frame: {frame:05d} / {len(image_paths) - 1:05d}",
            f"method: {method}",
            f"tracked points: {visible_count}",
            f"valid object poses: {pose_count}",
            f"depth coverage: {coverage:.1f}%",
            authenticity,
        )
        for line_index, line in enumerate(status_lines):
            cv2.putText(status_view, line, (20, 50 + line_index * 35), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                        (60, 90, 245) if fallback and line_index == 5 else (220, 232, 238), 1, cv2.LINE_AA)
        grid = cv2.vconcat((
            cv2.hconcat((_tile(raw, "2.0 current RGB", tile_size), _tile(detection, "2.1 DINO prompt + boxes", tile_size), _tile(segmentation, "2.2 SAM2 mask", tile_size), _tile(sampled, "2.3 contour keypoints", tile_size))),
            cv2.hconcat((_tile(tracked, "2.4 CoTracker3 + trail", tile_size), _tile(depth_view, "2.5 stereo depth", tile_size), _tile(pose_view, "2.6 3D centroid + orientation", tile_size), _tile(status_view, "frame diagnostics", tile_size))),
        ))
        output = output_dir / f"{frame:05d}.jpg"
        if not output.is_file():
            cv2.imwrite(str(output), grid, [cv2.IMWRITE_JPEG_QUALITY, int(cfg.visualization.jpeg_quality)])
        urls.append(_relative_url(output, destination))
    if not urls:
        return None, fallback
    return {
        "frames": urls,
        "width": tile_size[0] * 4,
        "height": tile_size[1] * 2,
        "fps": cfg.realtime.camera_hz,
        "times": _clean(np.arange(len(urls), dtype=np.float64) / max(cfg.realtime.camera_hz, 1e-6)),
        "syncGroup": "step2_camera",
        "layers": [],
        "telemetry": [
            {"label": "camera frame", "values": list(range(len(urls)))},
            {"label": "pipeline method", "values": [method] * len(urls)},
            {"label": "authentic model output", "values": [not fallback] * len(urls)},
            {"label": "CoTracker visible points", "values": visible_counts, "unit": "pts"},
            {"label": "valid object poses", "values": pose_counts},
            {"label": "depth coverage", "values": _clean(depth_coverage), "unit": "%"},
        ],
    }, fallback


def _step2_stage_rows(episode_dir: Path) -> list[list[Any]]:
    preprocess = episode_dir / "perception/humanego_session/preprocess"
    perception_manifest = _read_json(episode_dir / "humanego_perception.manifest.json", {})
    fallback = bool(perception_manifest.get("metrics", {}).get("fallback", False))
    stages = [
        ("2.0", "双目 SGBM 深度", episode_dir / "stereo_depth.manifest.json"),
        ("2.1", "Grounding DINO + SAM2", preprocess / "all_data/00000/mask_obj1.png"),
        ("2.2", "轮廓关键点采样", preprocess / "kptsselector_results.json"),
        ("2.3", "CoTracker3 时序跟踪", preprocess / "cotracker_results.json"),
        ("2.4", "3D 质心 + Orient-Anything", preprocess / "camtriangulator_results.json"),
        ("2.5", "双手—物体关系编码", episode_dir / "relations.manifest.json"),
    ]
    rows = []
    for number, name, path in stages:
        if fallback and number in {"2.1", "2.2", "2.3", "2.4"}:
            badge = status_badge("未运行 · DEBUG fallback", "bad")
        else:
            badge = status_badge("已有真实数据", "ok") if path.is_file() else status_badge("等待上游", "warn")
        rows.append([number, name, RawHtml(badge), str(path)])
    return rows


def generate_step2_report(cfg: ProjectConfig, source: Path, episode_dir: Path | None = None) -> Path:
    episode_dir = episode_dir or episode_work_dir(cfg, source)
    destination = episode_dir / "reports/step2.html"
    required = [
        episode_dir / "relations.manifest.json",
        episode_dir / "entities/poses.npz",
        episode_dir / "entities/relation_tokens.npy",
        episode_dir / "entities/relation_mask.npy",
    ]
    debug_data: dict[str, Any] = {"timelinePlayers": {}, "charts": {}}
    debug_body = panel(
        "Step2 子阶段状态",
        table(["编号", "算法节点", "状态", "主产物"], _step2_stage_rows(episode_dir)),
    )
    full_pipeline, fallback = _step2_full_pipeline_player(cfg, episode_dir, destination)
    if full_pipeline is not None:
        debug_data["timelinePlayers"]["step2_full_pipeline"] = full_pipeline
        authenticity = (
            '<div class="notice bad"><strong>当前不是模型结果：</strong>这个回合使用 simple_stereo_fallback；页面用红字标出，DINO、SAM2、CoTracker3、Orient-Anything 均未真实执行。只能检查数据契约和页面交互。</div>'
            if fallback
            else '<div class="notice"><strong>真实模型产物：</strong>六宫格只读取本回合已落盘的 DINO/SAM2/关键点/CoTracker/深度/姿态结果。</div>'
        )
        debug_body += panel(
            "Step2 全部相机帧 · 全中间链路",
            authenticity + timeline_player("step2_full_pipeline")
            + '<p class="legend-note">每一格对应同一个相机帧。DINO、SAM2 与轮廓采样按 HumanEgo 设计只在 reference frame=0 执行；后续帧会明确显示 not invoked。CoTracker、双目深度和 3D 位姿逐帧更新。</p>',
        )
    depth_debug = _step2_depth_player(cfg, episode_dir, destination)
    if depth_debug is not None:
        depth_player, depth_chart = depth_debug
        debug_data["timelinePlayers"]["step2_depth"] = depth_player
        debug_data["charts"]["step2_depth_coverage"] = depth_chart
        debug_body += panel(
            "2.0 双目深度逐帧检查",
            timeline_player("step2_depth")
            + '<p class="legend-note">左边是真实 PICO RGB，右边是同帧 uint16 毫米深度伪彩色；右侧实时显示覆盖率和深度分位数。</p>',
        )
        debug_body += panel("2.0 深度时序 QA", line_chart("step2_depth_coverage"))
    debug_body += panel("2.1 / 2.2 检测、分割与轮廓点", _step2_reference_gallery(cfg, episode_dir))
    tracker_player = _step2_tracker_player(cfg, episode_dir, destination)
    if tracker_player is not None:
        debug_data["timelinePlayers"]["step2_tracker"] = tracker_player
        debug_body += panel(
            "2.3 CoTracker3 全序列播放器",
            timeline_player("step2_tracker")
            + '<p class="legend-note">逐实例开关跟踪点；被 visibility 阈值判无效的点不会绘制。与深度播放器使用同一帧时间轴。</p>',
        )
    if not all(path.is_file() for path in required):
        missing_rows = [
            [path.name, RawHtml(status_badge("存在", "ok") if path.is_file() else status_badge("缺失", "bad")), str(path)]
            for path in required
        ]
        debug_body += panel(
            "最终关系产物尚未齐全",
            '<div class="notice">页面上方仍展示已经真实生成的子阶段数据；不是空白占位页。完成其余 HumanEgo 节点后重新执行 report。</div>'
            + table(["文件", "状态", "路径"], missing_rows),
        )
        return write_page(
            destination,
            title=f"{source.stem} · Step2",
            eyebrow="HUMANEGO / RELATIONS",
            subtitle="按真实产物渐进展示：双目深度 → DINO/SAM2 → 轮廓点 → CoTracker3 → 3D/姿态 → ICT。",
            body=debug_body,
            data=debug_data,
            nav=_episode_nav(episode_dir, "step2"),
        )
    with np.load(episode_dir / "entities/poses.npz", allow_pickle=False) as archive:
        categories = archive["categories"].astype(str)
        instance_ids = archive["instance_ids"].astype(str)
        left_key = "T_camera0_left_wrist" if "T_camera0_left_wrist" in archive.files else "T_camera0_left_hand"
        right_key = "T_camera0_right_wrist" if "T_camera0_right_wrist" in archive.files else "T_camera0_right_hand"
        left = archive[left_key].astype(np.float64)
        right = archive[right_key].astype(np.float64)
        layout_class = str(archive["layout_class"].item()) if "layout_class" in archive.files else "unknown"
        objects = archive["T_camera0_object"].astype(np.float64)
        left_grasp = archive["left_grasp"].astype(bool)
        right_grasp = archive["right_grasp"].astype(bool)
        dynamic = archive["object_dynamic"].astype(bool)
    tokens = np.load(episode_dir / "entities/relation_tokens.npy").astype(np.float32)
    masks = np.load(episode_dir / "entities/relation_mask.npy").astype(bool)
    ticks = np.load(episode_dir / "mode2/ticks_ns.npy")
    time_s = (ticks - ticks[0]) / 1e9
    index = _downsample(len(time_s), cfg.visualization.max_chart_points)
    distance_series = []
    for object_index, category in enumerate(categories):
        distance_series.append((f"{category}/{instance_ids[object_index]}→L", np.linalg.norm(objects[:, object_index, :3, 3] - left[:, :3, 3], axis=1)))
        distance_series.append((f"{category}/{instance_ids[object_index]}→R", np.linalg.norm(objects[:, object_index, :3, 3] - right[:, :3, 3], axis=1)))
    charts = {
        "distance": _chart(
            "物体—双腕距离",
            time_s[index],
            [(name, values[index]) for name, values in distance_series],
            y_label="m",
            threshold=cfg.relations.contact_distance_m,
        ),
        "grasp": _chart(
            "抓持与物体 latch 状态",
            time_s[index],
            [("left grasp", left_grasp[index].astype(float)), ("right grasp", right_grasp[index].astype(float))]
            + [(f"{instance_ids[i]} dynamic", dynamic[index, i].astype(float)) for i in range(len(instance_ids))],
            y_label="boolean",
        ),
    }
    for chart in charts.values():
        chart["syncGroup"] = "step2_relation"
    entities = [
        {"name": "left_wrist", "kind": "hand", "color": "#75e6a4", "positions": _clean(left[:, :3, 3]), "axes": _clean(np.swapaxes(left[:, :3, :3], 1, 2))},
        {"name": "right_wrist", "kind": "hand", "color": "#f5c451", "positions": _clean(right[:, :3, 3]), "axes": _clean(np.swapaxes(right[:, :3, :3], 1, 2))},
    ]
    for object_index, instance in enumerate(instance_ids):
        pose = objects[:, object_index]
        entities.append(
            {
                "name": f"{categories[object_index]}/{instance}",
                "kind": "object",
                "color": _COLORS[(object_index + 3) % len(_COLORS)],
                "positions": _clean(pose[:, :3, 3]),
                "axes": _clean(np.swapaxes(pose[:, :3, :3], 1, 2)),
            }
        )
    row_labels = ["left_wrist", "right_wrist"] + [f"{categories[i]}/{instance_ids[i]}" for i in range(len(instance_ids))]
    row_labels += [f"pad{i}" for i in range(len(row_labels), tokens.shape[1])]
    col_labels = ["type"] + [f"entity{i}" for i in range(9)] + [f"left{i}" for i in range(9)] + [f"right{i}" for i in range(9)] + ["flag"]
    text_rows = _jsonl(episode_dir / "entities/relation_text.jsonl")
    text_sample = [text_rows[int(i)] for i in _indices(len(text_rows), min(8, cfg.visualization.preview_frames))]
    relation_manifest = _read_json(episode_dir / "relations.manifest.json", {})
    stereo_report = _read_json(episode_dir / "qa/stereo_object_track_report.json", {})
    valid_ratios = stereo_report.get("valid_ratio_per_object", {})
    cards = [
        ("初始布局", layout_class, "camera0 x-axis"),
        ("对象实例", str(len(instance_ids)), "不含左右手 token"),
        ("关系张量", f"{tokens.shape[1]} × {tokens.shape[2]}", "max_entities × ICT"),
        ("有效 token", f"{masks.sum(axis=1).mean():.1f}", "每帧平均"),
        ("左手抓持", f"{left_grasp.mean() * 100:.1f}%", "hysteresis"),
        ("右手抓持", f"{right_grasp.mean() * 100:.1f}%", "hysteresis"),
        ("动态物体", f"{dynamic.mean() * 100:.1f}%", "FK latch 帧占比"),
    ]
    body = debug_body + metrics(cards)
    body += panel("DINO / SAM2 / CoTracker3 叠加", _relation_overlay_gallery(cfg, episode_dir))
    body += panel("cam0 三维交互轨迹", scene3d("relations"))
    body += '<div class="grid2">' + panel("物体—腕部距离", line_chart("distance")) + panel("抓持状态", line_chart("grasp")) + "</div>"
    body += panel("ICT token 热力图", heatmap("tokens") + '<p class="legend-note">滑块逐帧检查 padding、type、三组 9D pose 和 latch flag。</p>')
    body += panel(
        "实例与 3D 跟踪 QA",
        table(
            ["instance", "category", "stereo valid ratio"],
            [[instance_ids[i], categories[i], _format_float(valid_ratios.get(str(instance_ids[i])))] for i in range(len(instance_ids))],
        ),
    )
    if text_sample:
        body += panel("方案一语言提示抽样", table(["frame", "prompt"], [[row["frame_index"], row["prompt"]] for row in text_sample]))
    matrix_direction = relation_manifest.get("config", {}).get("matrix_direction", "T_object_wrist")
    if isinstance(matrix_direction, list):
        matrix_direction = "; ".join(str(value) for value in matrix_direction)
    return write_page(
        destination,
        title=f"{source.stem} · Step2",
        eyebrow="HUMANEGO / 3D / DUAL-WRIST ICT",
        subtitle=f"{matrix_direction}；2D、3D、关系 token 在同页对照。",
        body=body,
        data={
            **debug_data,
            "charts": {**debug_data["charts"], **charts},
            "scenes": {
                "relations": {
                    "frames": len(left),
                    "times": _clean(time_s),
                    "syncGroup": "step2_relation",
                    "entities": entities,
                }
            },
            "heatmaps": {
                "tokens": {
                    "frames": _clean(tokens),
                    "times": _clean(time_s),
                    "syncGroup": "step2_relation",
                    "rowLabels": row_labels,
                    "colLabels": col_labels,
                }
            },
        },
        nav=_episode_nav(episode_dir, "step2"),
    )


def _read_video_frames(path: Path, count: int) -> tuple[list[np.ndarray], int, float]:
    capture = cv2.VideoCapture(str(path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    for index in _indices(total, count):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if ok:
            frames.append(frame)
    capture.release()
    return frames, total, fps


def _video_frame_assets(
    cfg: ProjectConfig,
    video: Path,
    destination: Path,
    asset_key: str,
    maximum: int | None = None,
) -> tuple[list[str], int, int, float]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        return [], 0, 0, 0.0
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 0.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    count = min(total, maximum) if maximum is not None else total
    output_dir = destination.parent / "assets" / asset_key
    output_dir.mkdir(parents=True, exist_ok=True)
    urls: list[str] = []
    width = height = 0
    for index in range(count):
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        height, width = frame.shape[:2]
        output = output_dir / f"{index:05d}.jpg"
        if not output.is_file():
            cv2.imwrite(str(output), frame, [cv2.IMWRITE_JPEG_QUALITY, int(cfg.visualization.jpeg_quality)])
        urls.append(_relative_url(output, destination))
    capture.release()
    return urls, width, height, fps


def _open_capture(path: Path) -> cv2.VideoCapture | None:
    if not path.is_file():
        return None
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        return None
    return capture


def _capture_frame(capture: cv2.VideoCapture | None, fallback: np.ndarray) -> np.ndarray:
    if capture is None:
        return fallback.copy()
    ok, frame = capture.read()
    if not ok or frame is None:
        return fallback.copy()
    if frame.shape[:2] != fallback.shape[:2]:
        frame = cv2.resize(frame, (fallback.shape[1], fallback.shape[0]))
    return frame


def _tile(image: np.ndarray, label: str, size: tuple[int, int]) -> np.ndarray:
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return _caption_image(resized, label)


def _step3_stage_rows(manifest: dict[str, Any] | None, episode_dir: Path) -> tuple[list[list[Any]], Path | None]:
    composite = Path(manifest["outputs"]["composite"]) if manifest else None
    work = composite.parent if composite else episode_dir / "visual/hand_swap_work"
    stages = [
        ("3.0", "PICO 2D 投影", work / "landmarks2d.npz"),
        ("3.1", "PICO bbox + SAM2 提示", work / "video_bboxes.mkv"),
        ("3.2", "SAM2 手臂 mask", work / "masks_arm.npy"),
        ("3.3", "E2FGVI 去手背景", work / "bg.mkv"),
        ("3.4", "SAPIEN Revo2 渲染/合成", work / "composite.mp4"),
        ("3.5", "Step3 manifest", episode_dir / "hand_swap.manifest.json"),
    ]
    rows = [
        [number, name, RawHtml(status_badge("已有数据", "ok") if path.is_file() else status_badge("等待上游", "warn")), str(path)]
        for number, name, path in stages
    ]
    return rows, composite


def _step3_pipeline_player(
    cfg: ProjectConfig,
    source: Path,
    destination: Path,
    composite: Path,
) -> dict[str, Any] | None:
    work = composite.parent
    bbox_capture = _open_capture(work / "video_bboxes.mkv")
    background_capture = _open_capture(work / "bg.mkv")
    composite_capture = _open_capture(composite)
    if composite_capture is None:
        for capture in (bbox_capture, background_capture):
            if capture is not None:
                capture.release()
        return None
    masks_path = work / "masks_arm.npy"
    masks = np.load(masks_path).astype(bool) if masks_path.is_file() else None
    output_dir = destination.parent / "assets/step3/pipeline"
    output_dir.mkdir(parents=True, exist_ok=True)
    urls: list[str] = []
    coverage: list[float] = []
    composite_change: list[float] = []
    render_area: list[float] = []
    tile_size = (480, 270)
    with h5py.File(source, "r") as file:
        images = file["camera/images_left_jpeg"]
        composite_total = int(composite_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        count = min(len(images), composite_total) if composite_total > 0 else len(images)
        for index in range(count):
            raw = cv2.imdecode(np.frombuffer(bytes(images[index]), dtype=np.uint8), cv2.IMREAD_COLOR)
            if raw is None:
                continue
            bbox = _capture_frame(bbox_capture, raw)
            background = _capture_frame(background_capture, raw)
            final = _capture_frame(composite_capture, background)
            if masks is not None and index < len(masks):
                mask = masks[index]
                if mask.shape != raw.shape[:2]:
                    mask = cv2.resize(mask.astype(np.uint8), (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            else:
                mask = np.zeros(raw.shape[:2], dtype=bool)
            mask_view = raw.copy()
            mask_view[mask] = np.uint8(mask_view[mask].astype(np.float32) * 0.34 + np.asarray([50, 60, 255]) * 0.66)
            render_delta = cv2.absdiff(final, background)
            render_mask = np.max(render_delta, axis=2) > 8
            render_only = np.zeros_like(final)
            render_only[render_mask] = final[render_mask]
            coverage.append(float(mask.mean()) * 100)
            composite_change.append(float(np.mean(cv2.absdiff(raw, final))) / 255.0 * 100)
            render_area.append(float(render_mask.mean()) * 100)
            grid = cv2.vconcat(
                [
                    cv2.hconcat(
                        [
                            _tile(raw, "3.0 PICO raw", tile_size),
                            _tile(bbox, "3.1 PICO bbox / SAM2 seed", tile_size),
                            _tile(mask_view, "3.2 SAM2 arm mask", tile_size),
                        ]
                    ),
                    cv2.hconcat(
                        [
                            _tile(background, "3.3 E2FGVI background", tile_size),
                            _tile(render_only, "3.4 SAPIEN Revo2 RGBA", tile_size),
                            _tile(final, "3.5 final composite", tile_size),
                        ]
                    ),
                ]
            )
            output = output_dir / f"{index:05d}.jpg"
            if not output.is_file():
                cv2.imwrite(str(output), grid, [cv2.IMWRITE_JPEG_QUALITY, int(cfg.visualization.jpeg_quality)])
            urls.append(_relative_url(output, destination))
    for capture in (bbox_capture, background_capture, composite_capture):
        if capture is not None:
            capture.release()
    if not urls:
        return None
    times = np.arange(len(urls), dtype=np.float64) / max(cfg.realtime.camera_hz, 1e-6)
    return {
        "frames": urls,
        "width": tile_size[0] * 3,
        "height": tile_size[1] * 2,
        "fps": cfg.realtime.camera_hz,
        "times": _clean(times),
        "syncGroup": "step3",
        "layers": [],
        "telemetry": [
            {"label": "frame", "values": list(range(len(urls)))},
            {"label": "SAM2 mask coverage", "values": _clean(coverage), "unit": "%"},
            {"label": "rendered hand area", "values": _clean(render_area), "unit": "%"},
            {"label": "raw→final pixel Δ", "values": _clean(composite_change), "unit": "%"},
        ],
    }


def _step3_result_player(
    cfg: ProjectConfig,
    source: Path,
    destination: Path,
    composite: Path,
) -> dict[str, Any] | None:
    """All-frame raw/final/difference view focused on replacement quality."""
    capture = _open_capture(composite)
    if capture is None:
        return None
    output_dir = destination.parent / "assets/step3/result_compare"
    output_dir.mkdir(parents=True, exist_ok=True)
    urls: list[str] = []
    changes: list[float] = []
    tile_size = (420, 315)
    with h5py.File(source, "r") as file:
        images = file["camera/images_left_jpeg"]
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        count = min(len(images), total) if total > 0 else len(images)
        for frame in range(count):
            raw = cv2.imdecode(np.frombuffer(bytes(images[frame]), dtype=np.uint8), cv2.IMREAD_COLOR)
            if raw is None:
                continue
            final = _capture_frame(capture, raw)
            if final.shape[:2] != raw.shape[:2]:
                final = cv2.resize(final, (raw.shape[1], raw.shape[0]))
            delta = cv2.absdiff(raw, final)
            changes.append(float(delta.mean()) / 255.0 * 100)
            amplified = cv2.applyColorMap(np.clip(delta.max(axis=2) * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            grid = cv2.hconcat((
                _tile(raw, "PICO original", tile_size),
                _tile(final, "BrainCo Revo2 composite", tile_size),
                _tile(amplified, "absolute difference x4", tile_size),
            ))
            output = output_dir / f"{frame:05d}.jpg"
            if not output.is_file():
                cv2.imwrite(str(output), grid, [cv2.IMWRITE_JPEG_QUALITY, int(cfg.visualization.jpeg_quality)])
            urls.append(_relative_url(output, destination))
    capture.release()
    if not urls:
        return None
    return {
        "frames": urls,
        "width": tile_size[0] * 3,
        "height": tile_size[1],
        "fps": cfg.realtime.camera_hz,
        "times": _clean(np.arange(len(urls), dtype=np.float64) / max(cfg.realtime.camera_hz, 1e-6)),
        "syncGroup": "step3",
        "layers": [],
        "telemetry": [
            {"label": "camera frame", "values": list(range(len(urls)))},
            {"label": "raw to final pixel change", "values": _clean(changes), "unit": "%"},
        ],
    }


def generate_step3_report(cfg: ProjectConfig, source: Path, episode_dir: Path | None = None) -> Path:
    episode_dir = episode_dir or episode_work_dir(cfg, source)
    destination = episode_dir / "reports/step3.html"
    manifest_path = episode_dir / "hand_swap.manifest.json"
    manifest = _read_json(manifest_path, None)
    stage_rows, stage_composite = _step3_stage_rows(manifest, episode_dir)
    stage_body = panel("Step3 子阶段状态", table(["编号", "视觉节点", "状态", "主产物"], stage_rows))
    if not manifest_path.is_file():
        stage_body += panel(
            "视觉支路尚未运行",
            '<div class="notice">Step3 运行后，这里会按同一个帧号同时显示 raw、PICO bbox、SAM2 mask、E2FGVI、SAPIEN RGBA 和最终合成，不再只抽几张对比图。</div>',
        )
        return write_page(
            destination,
            title=f"{source.stem} · Step3",
            eyebrow="BRAINCO VISUAL SWAP",
            subtitle="按真实中间产物渐进展示：PICO 投影 → SAM2 → E2FGVI → SAPIEN → composite。",
            body=stage_body,
            nav=_episode_nav(episode_dir, "step3"),
        )
    composite = stage_composite or Path(manifest["outputs"]["composite"])
    pipeline_player = _step3_pipeline_player(cfg, source, destination, composite)
    result_player = _step3_result_player(cfg, source, destination, composite)
    swapped, total, fps = _read_video_frames(composite, cfg.visualization.preview_frames)
    if not swapped:
        raise RuntimeError(f"BrainCo composite has no readable frames: {composite}")
    comparisons = []
    diffs = []
    with h5py.File(source, "r") as file:
        images = file["camera/images_left_jpeg"]
        raw_indices = _indices(len(images), len(swapped))
        for order, swapped_frame in enumerate(swapped):
            raw_index = int(raw_indices[min(order, len(raw_indices) - 1)])
            raw = cv2.imdecode(np.frombuffer(bytes(images[raw_index]), dtype=np.uint8), cv2.IMREAD_COLOR)
            if raw is None:
                continue
            if swapped_frame.shape[:2] != raw.shape[:2]:
                swapped_frame = cv2.resize(swapped_frame, (raw.shape[1], raw.shape[0]))
            diffs.append(float(np.mean(cv2.absdiff(raw, swapped_frame))) / 255.0)
            comparisons.append(
                {
                    "left": _image_data_url(raw, cfg.visualization.jpeg_quality),
                    "right": _image_data_url(swapped_frame, cfg.visualization.jpeg_quality),
                    "label": f"sample {order} · raw camera frame {raw_index}",
                }
            )
    cards = [
        ("Composite 帧数", str(total), f"{fps:.2f} fps"),
        ("抽检帧", str(len(comparisons)), "均匀采样"),
        ("平均像素变化", f"{np.mean(diffs) * 100:.1f}%" if diffs else "—", "仅作视觉 QA，不是质量指标"),
        ("导出视觉源", cfg.visual.source, "raw / brainco_swap"),
    ]
    body = stage_body + metrics(cards)
    data: dict[str, Any] = {"comparisons": {"swap": {"frames": comparisons}}, "timelinePlayers": {}}
    if result_player is not None:
        data["timelinePlayers"]["step3_result"] = result_player
        body += panel(
            "全部帧 · 原图 / BrainCo 替换 / 像素差异",
            timeline_player("step3_result")
            + '<p class="legend-note">三栏严格使用同一帧号；差异图放大 4 倍，用来定位残留人手、边缘接缝、错位和闪烁。这个播放器直接回答最终替换效果，不是抽样图。</p>',
        )
    if pipeline_player is not None:
        data["timelinePlayers"]["step3_pipeline"] = pipeline_player
        body += panel(
            "Step3 全链路逐帧六宫格",
            timeline_player("step3_pipeline")
            + '<p class="legend-note">六个格子严格使用同一 HDF5 帧号；右侧给出 mask 覆盖率、渲染手面积和 raw→final 像素变化。</p>',
        )
    body += panel("原始图与 BrainCo 替换对齐", comparison("swap", "PICO 原始图", "BrainCo Revo2 替换图"))
    if composite.is_file():
        body += panel(
            "Composite 视频",
            f'<video controls preload="metadata" src="{html.escape(_relative_url(composite, destination), quote=True)}"></video>'
            '<p class="legend-note">若浏览器不支持该 MP4 codec，仍可使用上面的逐帧滑块完成检查。</p>',
        )
    return write_page(
        destination,
        title=f"{source.stem} · Step3",
        eyebrow="OPTIONAL VISUAL CANONICALIZATION",
        subtitle="同一采样位置并排检查去手、修复和 BrainCo 渲染；此支路不影响关系/动作数据。",
        body=body,
        data=data,
        nav=_episode_nav(episode_dir, "step3"),
    )


def _stack_column(table_data, name: str) -> np.ndarray:
    return np.asarray(table_data[name].to_pylist())


def generate_step4_report(cfg: ProjectConfig, dataset: Path) -> Path:
    dataset = dataset.resolve()
    destination = dataset / "reports/index.html"
    info_path = dataset / "meta/info.json"
    parquet_paths = sorted(dataset.glob("data/chunk-*/episode_*.parquet"))
    if not info_path.is_file() or not parquet_paths:
        return _missing_report(
            destination,
            title=f"{dataset.name} · Step4",
            eyebrow="LEROBOT V2",
            message="数据集尚未完整导出。",
            required=[info_path, dataset / "meta/tasks.jsonl", dataset / "meta/episodes.jsonl"],
            nav=[],
        )
    info = _read_json(info_path, {})
    relation = info.get("ego_relation", {})
    columns = [
        "observation.state",
        "observation.action_reference_tcp",
        "action",
        "timestamp",
        "task_index",
    ]
    first = pq.read_table(parquet_paths[0], columns=columns)
    state = _stack_column(first, "observation.state").astype(np.float64)
    reference = _stack_column(first, "observation.action_reference_tcp").astype(np.float64)
    action = _stack_column(first, "action").astype(np.float64)
    time_s = _stack_column(first, "timestamp").reshape(-1).astype(np.float64)
    variant = str(relation.get("variant", "continuous"))
    gripper_dim = 12 if variant == "continuous" else 2
    if state.shape[1] != 54 + gripper_dim or action.shape[1] != 18 + gripper_dim:
        raise ValueError(f"Step4 {variant} 维度契约无效: state={state.shape}, action={action.shape}")
    index = _downsample(len(state), cfg.visualization.max_chart_points)
    expected_reference = np.concatenate([reference[1:], reference[-1:]], axis=0)
    current_gripper = state[:, 54:]
    expected_gripper = np.concatenate([current_gripper[1:], current_gripper[-1:]], axis=0)
    expected = np.concatenate([expected_reference, expected_gripper], axis=1)
    shift_error = float(np.max(np.abs(action - expected)))
    charts = {
        "action_delta": _chart(
            "绝对目标相对当前 TCP 的平移差（仅 QA 展示）",
            time_s[index],
            [("L-dx", action[index, 0] - reference[index, 0]), ("L-dy", action[index, 1] - reference[index, 1]),
             ("L-dz", action[index, 2] - reference[index, 2]), ("R-dx", action[index, 9] - reference[index, 9]),
             ("R-dy", action[index, 10] - reference[index, 10]), ("R-dz", action[index, 11] - reference[index, 11])],
            y_label="m",
        ),
        "brainco_action": _chart(
            "BrainCo action 标签",
            time_s[index],
            [(f"gripper-{joint}", action[index, 18 + joint]) for joint in range(gripper_dim)],
            y_label="binary" if variant == "binary" else "normalized joint",
        ),
    }
    for chart in charts.values():
        chart["syncGroup"] = "step4"
    relation_tokens = state[:, :54].reshape(len(state), 6, 9)
    heatmaps = {
        "tokens": {
            "frames": _clean(relation_tokens),
            "rowLabels": ["L-holder", "L-red", "L-yellow", "R-holder", "R-red", "R-yellow"],
            "colLabels": ["x", "y", "z", "r1x", "r1y", "r1z", "r2x", "r2y", "r2z"],
            "times": _clean(time_s),
            "syncGroup": "step4",
        }
    }
    tasks = _jsonl(dataset / "meta/tasks.jsonl")
    videos = sorted(dataset.glob("videos/chunk-*/*/episode_*.mp4"))
    global_index = cfg.paths.work_dir / "reports/index.html"
    nav = [("工程总览", _relative_url(global_index, destination))] if global_index.is_file() else []
    cards = [
        ("Variant", variant, str(relation.get("action_semantics", ""))),
        ("Episodes", str(info.get("total_episodes", "—")), f"{info.get('total_frames', '—')} frames"),
        ("FPS", str(info.get("fps", "—")), "LeRobot timebase"),
        ("Action shift", status_badge("通过", "ok") if shift_error <= 1e-6 else status_badge("失败", "bad"), f"max error {shift_error:.2e}"),
        ("Compact relation", "6 × 9D", "左右 TCP 到 holder/red/yellow"),
        ("Visual source", str(relation.get("visual_source", "—")), "训练图像支路"),
    ]
    body = metrics(cards)
    timeline_players: dict[str, Any] = {}
    if videos:
        urls, video_width, video_height, video_fps = _video_frame_assets(
            cfg, videos[0], destination, "step4/dataset", maximum=len(state)
        )
        if urls:
            usable = len(urls)
            task_index = _stack_column(first, "task_index").reshape(-1)[:usable]
            timeline_players["step4_dataset"] = {
                "frames": urls,
                "width": video_width,
                "height": video_height,
                "fps": video_fps or float(info.get("fps", 20)),
                "times": _clean(time_s[:usable]),
                "syncGroup": "step4",
                "layers": [],
                "telemetry": [
                    {"label": "dataset frame", "values": list(range(usable))},
                    {"label": "timestamp", "values": _clean(time_s[:usable]), "unit": "s"},
                    {"label": "task index", "values": _clean(task_index)},
                    {"label": "relation transforms", "values": [6] * usable},
                ],
            }
            body += panel(
                "LeRobot 图像—标签联动播放器",
                timeline_player("step4_dataset")
                + '<p class="legend-note">拖动图像会同步更新下面的 state/action 柱、关系热力图和动作曲线游标。</p>',
            )
    vector_labels = (
        [f"L-pos-{axis}" for axis in "xyz"] + [f"L-rot6-{i}" for i in range(6)]
        + [f"R-pos-{axis}" for axis in "xyz"] + [f"R-rot6-{i}" for i in range(6)]
        + ([f"L-Revo2-{i}" for i in range(6)] + [f"R-Revo2-{i}" for i in range(6)]
           if variant == "continuous" else ["left-grasp", "right-grasp"])
    )
    current_action_space = np.concatenate([reference, current_gripper], axis=1)
    vector_views = {
        "step4_action": {
            "title": "当前绝对 TCP/加爪 → 下一帧绝对目标",
            "current": _clean(current_action_space),
            "target": _clean(action),
            "labels": vector_labels,
            "times": _clean(time_s),
            "syncGroup": "step4",
            "groups": [
                {"name": "left EEF 9D", "start": 0, "end": 9, "color": "#42d9d0"},
                {"name": "right EEF 9D", "start": 9, "end": 18, "color": "#76a9ff"},
                {"name": "gripper", "start": 18, "end": 18 + gripper_dim, "color": "#75e6a4"},
            ],
        }
    }
    body += panel("逐帧 state / action 标签", vector_view("step4_action"))
    body += '<div class="grid2">' + panel("相对动作标签", line_chart("action_delta")) + panel("BrainCo 标签", line_chart("brainco_action")) + "</div>"
    body += panel("结构化关系样本", heatmap("tokens"))
    body += panel("任务提示", table(["task_index", "task"], [[row.get("task_index"), row.get("task")] for row in tasks[:100]]))
    if videos:
        body += panel("首个 LeRobot 视频", f'<video controls preload="metadata" src="{html.escape(_relative_url(videos[0], destination), quote=True)}"></video>')
    feature_rows = [[name, value.get("dtype"), value.get("shape")] for name, value in info.get("features", {}).items()]
    body += panel("LeRobot features contract", table(["feature", "dtype", "shape"], feature_rows))
    return write_page(
        destination,
        title=f"{dataset.name} · Step4",
        eyebrow="LEROBOT V2 / EXPORT QA",
        subtitle="检查紧凑手物关系、绝对 action 延迟一帧、辅助 TCP reference、任务映射和视频对齐。",
        body=body,
        data={"charts": charts, "heatmaps": heatmaps, "timelinePlayers": timeline_players, "vectorViews": vector_views},
        nav=nav,
    )


def generate_index(cfg: ProjectConfig, sources: list[Path]) -> Path:
    destination = cfg.paths.work_dir / "reports/index.html"
    episode_sections = []
    for source in sources:
        episode_dir = episode_work_dir(cfg, source)
        pico = _read_json(episode_dir / "qa/pico_report.json", {})
        mode2 = _read_json(episode_dir / "mode2.manifest.json", {})
        motion = mode2.get("metrics", {}).get("motion_deployable")
        stage_cards = []
        step1_preview = episode_dir / "reports/assets/step1/mujoco_revo2_mode2_v4/00000.jpg"
        if not step1_preview.is_file():
            step1_preview = episode_dir / "reports/assets/step1/mujoco_revo2_mode2_v3/00000.jpg"
        if not step1_preview.is_file():
            step1_preview = episode_dir / "reports/assets/step1/mujoco_revo2/00000.jpg"
        if not step1_preview.is_file():
            step1_preview = episode_dir / "reports/assets/step1/camera/00000.jpg"
        step2_previews = sorted((episode_dir / "reports/assets/step2").glob("full_pipeline_*/00000.jpg"))
        step2_preview = step2_previews[0] if step2_previews else episode_dir / "reports/assets/step2/stereo_depth/00000.jpg"
        step3_preview = episode_dir / "reports/assets/step3/result_compare/00000.jpg"
        if not step3_preview.is_file():
            step3_preview = episode_dir / "reports/assets/step3/pipeline/00000.jpg"
        for number, step, title, detail, manifest, partial, preview in (
            ("01", "Step1", "Revo2 动作回放", "固定 robot_base；逐帧检查双腕轨迹、手势关节与动作标签。", episode_dir / "mode2.manifest.json", None, step1_preview),
            ("02", "Step2", "物体—双手关系", "逐帧检查 DINO、SAM2、CoTracker、深度、3D 位姿和关系编码。", episode_dir / "relations.manifest.json", episode_dir / "stereo_depth.manifest.json", step2_preview),
            (
                "03",
                "Step3",
                "BrainCo 视觉替换",
                "逐帧检查原图、去手、Revo2 渲染、合成结果及差异图。",
                episode_dir / "hand_swap.manifest.json",
                episode_dir / "visual/hand_swap_work",
                step3_preview,
            ),
        ):
            page = episode_dir / "reports" / f"{step.lower()}.html"
            if manifest.is_file():
                label = status_badge("完成", "ok")
            elif partial is not None and partial.exists():
                label = status_badge("有中间数据", "warn")
            else:
                label = status_badge("待运行", "warn")
            preview_html = (
                f'<img src="{html.escape(_relative_url(preview, destination), quote=True)}" alt="{html.escape(title)} 首帧预览">'
                if preview.is_file()
                else f'<div class="stage-placeholder">{html.escape(step.upper())}</div>'
            )
            stage_cards.append(
                f'<a class="stage-card" href="{html.escape(_relative_url(page, destination), quote=True)}">'
                f'<div class="stage-preview">{preview_html}<span class="stage-number">{number}</span></div>'
                f'<div class="stage-content"><div class="stage-title"><strong>{html.escape(title)}</strong>{label}</div>'
                f'<p>{html.escape(detail)}</p><span class="stage-open">打开 {html.escape(step)} 可视化 →</span></div></a>'
            )
        geometry_badge = status_badge("相机几何通过", "ok") if pico.get("geometry_deployable") else status_badge("相机几何未通过", "bad")
        motion_badge = status_badge("Mode2 通过", "ok") if motion else status_badge("Mode2 隔离", "bad")
        episode_sections.append(
            '<section class="episode-block">'
            f'<div class="episode-head"><div><div class="eyebrow">EPISODE</div><h2>{html.escape(source.stem)}</h2></div>'
            f'<div class="episode-qa">{geometry_badge}{motion_badge}</div></div>'
            f'<div class="stage-grid">{"".join(stage_cards)}</div></section>'
        )
    datasets = []
    for info_path in sorted(cfg.paths.output_dir.glob("*/meta/info.json")):
        dataset = info_path.parent.parent
        info = _read_json(info_path, {})
        report = dataset / "reports/index.html"
        datasets.append(
            [
                RawHtml(f'<a href="{html.escape(_relative_url(report, destination), quote=True)}">{html.escape(dataset.name)}</a>'),
                info.get("ego_relation", {}).get("variant"),
                info.get("total_episodes"),
                info.get("total_frames"),
            ]
        )
    body = (
        '<div class="notice"><strong>直接点击下面任意一张大卡片。</strong> '
        'Step1 看 Revo2 MuJoCo 动作；Step2 看逐帧感知链路；Step3 看视觉替换。</div>'
        + "".join(episode_sections)
    )
    body += panel(
        "LeRobot v2 数据集",
        table(["dataset", "variant", "episodes", "frames"], datasets)
        if datasets
        else '<div class="notice">尚未生成正式 LeRobot 数据集；Step4 完成后这里会出现入口。</div>',
    )
    body += panel(
        "调试顺序",
        "<p>先在 Step1 找坐标/同步/动作跳变，再在 Step2 对齐 2D mask、CoTracker、3D 轨迹和 ICT；"
        "Step3 仅检查视觉替换；Step4 最后确认训练数据契约。页面不修改任何原始数据。</p>",
    )
    return write_page(
        destination,
        title="Ego Relation Pipeline · 可视化总览",
        eyebrow="DEBUG REPORT INDEX",
        subtitle="每个阶段执行后自动更新；也可随时用 report 命令从现有产物重建。",
        body=body,
    )


def generate_available_reports(
    cfg: ProjectConfig,
    sources: list[Path],
    steps: tuple[str, ...] = ("step1", "step2", "step3", "step4"),
) -> dict[str, list[Path] | Path]:
    selected = set(steps)
    unknown = selected - {"step1", "step2", "step3", "step4"}
    if unknown:
        raise ValueError(f"Unknown report steps: {sorted(unknown)}")
    output: dict[str, list[Path] | Path] = {step: [] for step in selected}
    for source in sources:
        episode_dir = episode_work_dir(cfg, source)
        if "step1" in selected:
            output["step1"].append(generate_step1_report(cfg, source, episode_dir))
        if "step2" in selected:
            output["step2"].append(generate_step2_report(cfg, source, episode_dir))
        if "step3" in selected:
            output["step3"].append(generate_step3_report(cfg, source, episode_dir))
    if "step4" in selected:
        for info_path in sorted(cfg.paths.output_dir.glob("*/meta/info.json")):
            output["step4"].append(generate_step4_report(cfg, info_path.parent.parent))
    output["index"] = generate_index(cfg, sources)
    return output
