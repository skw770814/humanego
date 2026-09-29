"""LeRobot v2.1 写出 —— 逐项镜像 `ego_relation.s4_lerobot_export.lerobot`。

镜像的是**契约**, 不是它的数据布局: 列名/列序/features/info/meta 的行式写法/统计口径
与 s4 一致 (`lerobot.py:266-345`、`:348-452`), 但维度按本数据缩小 (右手 + 单物体),
视频来源换成本 pipeline 的 step3 观测帧。

`_stats` / `_pose_names` / `VIDEO_KEY` / `transform_to_vec9` **直接从 s4 import 复用**,
不另抄一份 —— 名字或统计口径漂移是 openpi 侧最难查的一类错。

不引入 `lerobot` 库 (没装, s4 也没装): pyarrow 直写 parquet + cv2 写 mp4 + json 写 meta。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from . import (ACTION_COORDINATE_SYSTEM, ACTION_REFERENCE_FRAME, ACTION_SOURCE_COORDINATE_SYSTEM,
               ACTION_STORAGE, FPS, bootstrap)
from .names import ACTION_NAMES, REFERENCE_NAMES, STATE_NAMES

CODEC_PIX_FMT = "yuv420p"
CHUNKS_SIZE = 1000
DEFAULT_ROBOT_TYPE = "pico4ultra_right_piper"
# 默认 h264 (libx264 + yuv420p + faststart, 走 xrhand.video): 常见播放器直接打得开,
# 体积也比 mp4v 小。LEGACY_CODEC 是 s4 `configs/default.yaml:183` 的值, 只有明确要
# 「与 s4 逐字一致」时才用 —— mp4v 是 MPEG-4 Part 2, 浏览器 / QuickTime 打不开。
DEFAULT_CODEC = "h264"
LEGACY_CODEC = "mp4v"

# 进数据集的那条 mp4 的分辨率, 与 `ego_relation_policy` 的训练图像口径对齐 (480 高 x 640 宽)。
# 左眼原图是 1080x810, 同一个 4:3, 所以缩放不变形。缩放在**编码这一处**做, 原图/观测 PNG
# /可视化都保持 1080x810 —— 只有 mp4 与 features 里声明的 shape 用这个尺寸。
OBSERVATION_WIDTH, OBSERVATION_HEIGHT = 640, 480
OBSERVATION_SIZE = (OBSERVATION_WIDTH, OBSERVATION_HEIGHT)


def _s4():
    bootstrap()
    from ego_relation.s4_lerobot_export import lerobot

    return lerobot


def stats(values: np.ndarray) -> dict[str, Any]:
    """s4 `_stats` 的直接复用 (population std, count 是 `[len]`)。"""
    return _s4()._stats(np.asarray(values))


def validate_transforms(name: str, transforms: np.ndarray) -> None:
    """s4 `_validate_transforms`: 齐次末行 / 正交 / det=+1, atol 2e-3。"""
    _s4()._validate_transforms(name, np.asarray(transforms, dtype=np.float64))


# ---------------------------------------------------------------- 视频


class BgrSink:
    """收 **BGR** 帧的编码出口 —— 「用哪个 writer」只在这一处决定。

    cv2 读写的都是 BGR, 而 `xrhand.video.VideoWriter` (libx264 + yuv420p + faststart,
    仓库里其它视频都走它) 吃的是 RGB, 所以这个转换只在 `write()` 里做一次; 调用方继续
    待在 cv2 的世界里, 不需要各自记得翻转。

    `codec == LEGACY_CODEC` 时走原来那段 cv2 代码 (逐字保留), 留给「要与 s4 的
    `cfg.export.video_codec` 逐字一致」的场景; 其余一律走 xrhand.video。
    """

    def __init__(self, destination: Path, *, codec: str, size: tuple[int, int], fps: float):
        self.destination = destination
        self.codec = codec
        self.width, self.height = int(size[0]), int(size[1])
        self.fps = float(fps)
        if codec == LEGACY_CODEC:
            import cv2

            writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*codec),
                                     self.fps, (self.width, self.height))
            if not writer.isOpened():
                raise RuntimeError(f"无法创建视频 {destination}, codec={codec}")
            self._writer, self._legacy = writer, True
        else:
            bootstrap()
            from xrhand.video import VideoWriter

            self._writer = VideoWriter(str(destination), self.width, self.height, self.fps)
            self._legacy = False

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"帧尺寸 {frame.shape} != 期望 {(self.height, self.width, 3)}")
        if self._legacy:
            self._writer.write(frame)
        else:
            # 唯一的 BGR -> RGB 转换点。转错的表现是画面红蓝互换, 而 step4 的
            # `_check_video_frames` (解码帧 vs cv2.imread 的 PNG) 会直接把它抓出来。
            self._writer.write(frame[:, :, ::-1])

    def close(self) -> None:
        """收尾。ffmpeg 非零退出 (编码失败) 会在这里抛。"""
        if self._legacy:
            self._writer.release()
        else:
            self._writer.close()


def write_stream(frames, destination: Path, expected: int, *, codec: str = DEFAULT_CODEC,
                 size: tuple[int, int] = OBSERVATION_SIZE, fps: float = FPS) -> int:
    """把一串 **BGR** 帧编码成一条 mp4 (唯一一次有损编码)。

    `write_video` 与 `--observation step2` 的源帧生成器都走这里 —— 「谁决定编码参数」
    只有这一处。尺寸不符的帧按 `fit_frame` 缩放 (进数据集固定 640x480)。
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    sink = BgrSink(destination, codec=codec, size=size, fps=fps)
    written = 0
    try:
        for frame in frames:
            sink.write(fit_frame(frame, size))
            written += 1
    finally:
        sink.close()
    if written != expected:
        destination.unlink(missing_ok=True)
        raise ValueError(f"编码 {written} 帧 != 期望 {expected} 帧")
    return written


def fit_frame(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """缩到目标尺寸 (W,H)。尺寸已对就原样返回。

    下采样固定 INTER_AREA (缩小的正解, 不会像 INTER_LINEAR 那样丢像素点)。
    """
    if (frame.shape[1], frame.shape[0]) == tuple(size):
        return frame
    import cv2

    return cv2.resize(frame, tuple(size), interpolation=cv2.INTER_AREA)


def write_video(frame_paths: list[Path], destination: Path, *, codec: str = DEFAULT_CODEC,
                size: tuple[int, int] = OBSERVATION_SIZE, fps: float = FPS) -> int:
    """把观测 PNG 按给定顺序编码成一条 mp4。

    帧用 cv2 读进来 (BGR) 交给 `write_stream`, 由它决定怎么编码 —— 默认 h264, 于是颜色与
    s4 的 `_write_video` 语义一致, 用 cv2 读回来看也是正确的 BGR。
    """
    import cv2

    if not frame_paths:
        raise ValueError("没有可编码的帧")

    def frames():
        for path in frame_paths:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise FileNotFoundError(f"读不到观测帧 {path}")
            yield frame

    return write_stream(frames(), destination, len(frame_paths), codec=codec, size=size, fps=fps)


def probe_video(path: Path) -> dict:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"opencv 打不开 {path}")
    try:
        return {
            "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        }
    finally:
        capture.release()


# ---------------------------------------------------------------- features / info


def features(state_dim: int, action_dim: int, reference_dim: int, *, height: int, width: int,
             codec: str, fps: float, state_names: list[str] | None = None,
             reference_names: list[str] | None = None,
             action_names: list[str] | None = None) -> dict:
    """与 s4 `:348-386` 同一套 key; `observation.action_reference_tcp` 的 shape 由 18 缩到 9。

    三份 names 不给就用模块常量 (单物体那一份); 多物体时按 9N+1 现算着传进来。
    reference/action 的 names 与 s4 **逐字同一份** (`right_tcp_absolute_current` /
    `right_tcp_absolute_target` + 爪) —— 存的是绝对位姿, 名字里没有任何物体 id。
    """
    video_key = _s4().VIDEO_KEY
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": [int(state_dim)],
            "names": [list(state_names or STATE_NAMES)],
        },
        "observation.action_reference_tcp": {
            "dtype": "float32",
            "shape": [int(reference_dim)],
            "names": [list(reference_names or REFERENCE_NAMES)],
        },
        "action": {
            "dtype": "float32",
            "shape": [int(action_dim)],
            "names": [list(action_names or ACTION_NAMES)],
        },
        video_key: {
            "dtype": "video",
            "shape": [int(height), int(width), 3],
            "names": ["height", "width", "channel"],
            "info": {
                "video.height": int(height),
                "video.width": int(width),
                "video.codec": codec,
                "video.pix_fmt": CODEC_PIX_FMT,
                "video.is_depth_map": False,
                "video.fps": float(fps),
                "video.channels": 3,
                "has_audio": False,
            },
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }


def info_dict(*, features: dict, n_episodes: int, total_frames: int, tasks: list[str],
              robot_type: str, fps: float, codec: str, variant: str,
              object_order: list[str], object_categories: list[str], latched_frames: int,
              max_invalid_gap: int, visual_source: str,
              max_simultaneous_latched: int = 1) -> dict:
    """与 s4 `:387-420` 同一套顶层字段; `ego_relation` 块按本数据改写。

    `object_order` / `object_categories` 是**同长的有序列表** (s4 `lerobot.py:404-405`
    是一对同长常量), 长度 = 物体个数。

    action/reference 的参考系是 PICO OpenXR 右手世界系 —— **每段采集各自一个** (原点由 runtime 定),
    所以这里写的是全局声明 `ACTION_REFERENCE_FRAME`; 每段的**锚点**在
    `extraction_meta.json` 里, 缺了它绝对量就没有落点。
    """
    if len(object_order) != len(object_categories):
        raise ValueError(
            f"object_order {object_order} 与 object_categories {object_categories} 长度不等"
        )
    return {
        "codebase_version": "v2.1",
        "robot_type": robot_type,
        "total_episodes": n_episodes,
        "total_frames": int(total_frames),
        "total_tasks": len(tasks),
        "total_chunks": (n_episodes - 1) // CHUNKS_SIZE + 1,
        "total_videos": n_episodes,
        "chunks_size": CHUNKS_SIZE,
        "fps": float(fps),
        "splits": {"train": f"0:{n_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
        "ego_relation": {
            "schema": "compact_hand_object_state_v1",
            "variant": variant,
            "hand": "right",
            "object_order": list(object_order),
            "object_categories": list(object_categories),
            "relation_direction": "T_tcp_object",
            "relation_pose_encoding": "[tx,ty,tz,R[:,0],R[:,1]]",
            "relation_reference_frame": "TCP-local relation derived from camera0 (left eye) poses",
            # action/reference 存的是**绝对位姿** (与 s4 同构), 相对动作不进数据集 —— 由
            # openpi 训练期按 training_action_transform 现算。别把"绝对"改回"相对"。
            "action_storage": ACTION_STORAGE,
            "action_semantics": "action[t] contains T_pico_world_openxr_tcp target at min(t+1,T-1)",
            "action_pose_encoding": "[tx,ty,tz,R[:,0],R[:,1]]",
            "action_reference_field": "observation.action_reference_tcp",
            "action_reference_is_model_input": False,
            "action_reference_frame": ACTION_REFERENCE_FRAME,
            "action_coordinate_system": ACTION_COORDINATE_SYSTEM,
            "action_source_coordinate_system": ACTION_SOURCE_COORDINATE_SYSTEM,
            "action_coordinate_transform": {
                "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
                "translation": "t_openxr = M @ t_unity",
                "rotation": "R_openxr = M @ R_unity @ M",
            },
            "action_reference_frame_definition": (
                "runtime-reported PICO tracking space; fixed for the whole recording; each "
                "recording has its own world frame (origin set by the runtime, no re-zeroing in "
                "this pipeline) ⇒ absolute values are NOT comparable across episodes"
            ),
            "training_action_transform": "deferred: inv(T_current) @ T_absolute_target",
            "gripper_action_transform": "none",
            "grasp_binary_semantics": (
                "state[t]=closed[t]; action[t]=closed[episode-local t+1] (last frame repeats itself)"
            ),
            "hand_frame": (
                "right thumb/index fingertip midpoint; this control point is preserved by "
                "OpenPI XRPipe Mode1 and MUST NOT be reinterpreted as wrist, palm, or flange"
            ),
            "training_state_transform": (
                "per relation pose T_midpoint_object: S @ T @ S, "
                "S=diag(1,1,-1,1); current gripper is unchanged"
            ),
            "label_space": "camera0 (left eye, eye0) for observation.state; "
                           "pico_world_openxr for reference/action",
            "single_latch_rule": (
                "at most one object is latched (hand-pushed) at any instant; asserted at step2 "
                "assembly and again before writing"
            ),
            "max_simultaneous_latched": int(max_simultaneous_latched),
            # 观测是 640x480 —— 与 ego_relation_policy 的训练图像口径一致 (左眼 1080x810 等比缩放)。
            "visual_source": visual_source,
            "visual_size": [OBSERVATION_HEIGHT, OBSERVATION_WIDTH],
            "visual_alignment": (
                "frame k == source video frame keep_index[k] (30 Hz grid, 1:1)"
            ),
            "valid_cropping": (
                f"one episode per recording; invalid runs shorter than {max_invalid_gap} frames "
                f"are bridged in place, longer ones are excised and the remainder spliced into "
                f"the same episode"
            ),
            "latched_frames_total": int(latched_frames),
            "video_codec": codec,
        },
    }


def action_semantics() -> dict:
    """Machine-readable XRPipe Mode1 action contract.

    The old three-key Mode2 declaration caused OpenPI to infer a wrist-yaw TCP
    and apply a wrist-to-palm offset. This dataset's control point is instead
    the measured thumb/index fingertip midpoint, so every relevant frame,
    layout, and deferred transform is explicit here.
    """
    bootstrap()
    from ego_relation.contracts.se3 import transform_to_vec9

    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = (1.0, 2.0, 3.0)
    encoded = np.asarray(transform_to_vec9(pose), dtype=np.float64)
    expected = np.concatenate([pose[:3, 3], rotation[:, 0], rotation[:, 1]])
    if not np.allclose(encoded, expected, atol=1e-12):
        raise AssertionError(
            f"vec9 的编码 {encoded.tolist()} 与 [tx,ty,tz,R[:,0],R[:,1]] "
            f"{expected.tolist()} 不一致 —— action_semantics.json 里那句 rotation_6d 的描述要改"
        )
    return {
        "schema_version": "xrpipe_action_v1",
        "mode": "xrpipe_mode1",
        "stored_action": "absolute_next_target",
        "reference_field": "observation.action_reference_tcp",
        "reference_frame": ACTION_REFERENCE_FRAME,
        "coordinate_system": ACTION_COORDINATE_SYSTEM,
        "source_coordinate_system": ACTION_SOURCE_COORDINATE_SYSTEM,
        "control_point": "right_thumb_index_fingertip_midpoint",
        "pose_encoding": "xyz_rot6d_columns_grouped",
        "rotation_6d": "first two columns of a 3x3 rotation matrix",
        "action_layout": {"dimension": 10, "pose_slice": [0, 9], "gripper_index": 9},
        "reference_layout": {"dimension": 9, "pose_slice": [0, 9]},
        "relative_formula": "inv(reference[t]) @ action[t+k]",
        "gripper_transform": "none",
    }
