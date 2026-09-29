from __future__ import annotations

import json
from itertools import chain
from pathlib import Path
import shutil
from typing import Any

import cv2
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ego_relation.config import ProjectConfig
from ego_relation.contracts.se3 import transform_to_vec9


VIDEO_KEY = "observation.images.camera0"
VALID_VARIANTS = {"continuous", "binary"}
EXPECTED_INSTANCE_IDS = ("obj1", "obj2", "obj3")
EXPECTED_CATEGORIES = ("black, metal pen holder", "red cube", "yellow cube")


def _stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(len(values))],
    }


def _decode_jpeg(value: Any) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(bytes(value), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("JPEG 解码失败")
    return image


def _brainco_video_path(episode_dir: Path) -> Path:
    manifest_path = episode_dir / "hand_swap.manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"缺少 Step3 manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = Path(manifest["outputs"]["composite"]).expanduser()
    if not path.is_absolute():
        path = (manifest_path.parent / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"缺少 BrainCo composite 视频: {path}")
    return path


def _video_frames(source: Path, episode_dir: Path, camera_match: np.ndarray, visual_source: str):
    if visual_source == "raw":
        with h5py.File(source, "r") as file:
            images = file["camera/images_left_jpeg"]
            if len(camera_match) == 0 or np.any(camera_match < 0) or np.any(camera_match >= len(images)):
                raise ValueError(f"{source.stem} camera_match 越界")
            for index in camera_match:
                yield _decode_jpeg(images[int(index)])
        return
    if visual_source != "brainco_swap":
        raise ValueError(f"visual.source 只能为 raw 或 brainco_swap，实际 {visual_source!r}")

    capture = cv2.VideoCapture(str(_brainco_video_path(episode_dir)))
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    if (
        frame_count <= 0
        or len(camera_match) == 0
        or np.any(camera_match < 0)
        or np.any(camera_match >= frame_count)
        or np.any(np.diff(camera_match) < 0)
    ):
        capture.release()
        raise ValueError(
            f"{episode_dir.name} BrainCo camera_match 无效: "
            f"video_frames={frame_count}, labels={len(camera_match)}, "
            f"range={int(camera_match.min()) if len(camera_match) else None}.."
            f"{int(camera_match.max()) if len(camera_match) else None}"
        )
    current_index = -1
    current_frame = None
    for target_index in camera_match:
        while current_index < int(target_index):
            ok, current_frame = capture.read()
            current_index += 1
            if not ok or current_frame is None:
                capture.release()
                raise RuntimeError(
                    f"{episode_dir.name} BrainCo 视频在第 {current_index} 帧提前结束"
                )
        yield current_frame
    capture.release()


def _write_video(
    source: Path,
    episode_dir: Path,
    camera_match: np.ndarray,
    destination: Path,
    fps: float,
    codec: str,
    visual_source: str,
) -> tuple[int, int, int]:
    iterator = iter(_video_frames(source, episode_dir, camera_match, visual_source))
    try:
        first = next(iterator)
    except StopIteration as error:
        raise RuntimeError(f"{source.stem} 没有可导出的视频帧") from error
    height, width = first.shape[:2]
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频 {destination}，codec={codec}")
    written = 0
    for frame in chain((first,), iterator):
        if frame.shape[:2] != (height, width):
            writer.release()
            raise ValueError(f"{source.stem} 视频分辨率在第 {written} 帧发生变化")
        writer.write(frame)
        written += 1
    writer.release()
    if written != len(camera_match):
        destination.unlink(missing_ok=True)
        raise ValueError(f"{source.stem} 导出视频 {written} 帧 != 标签 {len(camera_match)} 帧")
    return height, width, written


def _pose_names(prefix: str) -> list[str]:
    return [f"{prefix}_{name}" for name in ("x", "y", "z", "r1x", "r1y", "r1z", "r2x", "r2y", "r2z")]


def _relation_names() -> list[str]:
    names = []
    for hand in ("left_tcp", "right_tcp"):
        for object_name in ("holder", "red", "yellow"):
            names.extend(_pose_names(f"{hand}_to_{object_name}"))
    return names


def _continuous_names() -> list[str]:
    return [f"left_brainco_{index}" for index in range(6)] + [f"right_brainco_{index}" for index in range(6)]


def _validate_transforms(name: str, transforms: np.ndarray) -> None:
    if not np.isfinite(transforms).all():
        raise ValueError(f"{name} 包含 NaN/Inf")
    if not np.allclose(transforms[..., 3, :], np.asarray([0, 0, 0, 1]), atol=1e-5):
        raise ValueError(f"{name} 齐次矩阵末行无效")
    rotations = transforms[..., :3, :3]
    identity = np.eye(3)
    if not np.allclose(np.swapaxes(rotations, -1, -2) @ rotations, identity, atol=2e-3):
        raise ValueError(f"{name} 旋转矩阵不正交")
    if not np.allclose(np.linalg.det(rotations), 1.0, atol=2e-3):
        raise ValueError(f"{name} 旋转矩阵行列式不是 +1")


def _encode_relations(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    frames = len(left)
    encoded = np.empty((frames, 54), dtype=np.float32)
    offset = 0
    for transforms in (left, right):
        for object_index in range(3):
            encoded[:, offset : offset + 9] = np.stack(
                [transform_to_vec9(transform) for transform in transforms[:, object_index]], axis=0
            )
            offset += 9
    return encoded


def _load_task(episode_dir: Path) -> str:
    path = episode_dir / "step1/task_semantics.json"
    if not path.is_file():
        raise FileNotFoundError(f"缺少 Step1 任务语义: {path}")
    task = str(json.loads(path.read_text(encoding="utf-8")).get("instruction", "")).strip()
    if not task:
        raise ValueError(f"{path} instruction 为空")
    return task


def _load_episode(episode_dir: Path, variant: str) -> dict[str, Any]:
    if variant not in VALID_VARIANTS:
        raise ValueError(f"未知导出方案 {variant!r}")
    mode2 = episode_dir / "mode2"
    state_abs = np.load(mode2 / "state_abs_smoothed.npy").astype(np.float32)
    action_abs = np.load(mode2 / "action_abs_smoothed.npy").astype(np.float32)
    ticks = np.load(mode2 / "ticks_ns.npy").astype(np.int64)
    camera_match = np.load(mode2 / "camera_match.npy").astype(np.int64)
    if state_abs.ndim != 2 or state_abs.shape[1] != 30 or action_abs.shape != state_abs.shape:
        raise ValueError(f"{episode_dir.name} Step1 state/action 应为 (T,30)")
    expected_action = np.concatenate([state_abs[1:], state_abs[-1:]], axis=0)
    if not np.allclose(action_abs, expected_action, atol=1e-6):
        raise ValueError(f"{episode_dir.name} 平滑 action 的下一帧约定被破坏")

    with np.load(episode_dir / "entities/poses.npz", allow_pickle=False) as poses:
        instance_ids = tuple(str(value) for value in poses["instance_ids"])
        categories = tuple(str(value) for value in poses["categories"])
        if instance_ids != EXPECTED_INSTANCE_IDS or categories != EXPECTED_CATEGORIES:
            raise ValueError(
                f"{episode_dir.name} 对象顺序必须为 {EXPECTED_INSTANCE_IDS}/{EXPECTED_CATEGORIES}，"
                f"实际 {instance_ids}/{categories}"
            )
        left = poses["T_left_tcp_object"].astype(np.float64)
        right = poses["T_right_tcp_object"].astype(np.float64)
        relation_left_grasp = poses["left_grasp"].astype(bool)
        relation_right_grasp = poses["right_grasp"].astype(bool)
    if left.shape != (len(state_abs), 3, 4, 4) or right.shape != left.shape:
        raise ValueError(f"{episode_dir.name} Step2 TCP-object 矩阵形状无效")
    _validate_transforms("T_left_tcp_object", left)
    _validate_transforms("T_right_tcp_object", right)
    relations = _encode_relations(left, right)

    with np.load(mode2 / "brainco_grasp_binary.npz", allow_pickle=False) as grasp:
        closed_left = grasp["closed_left"].astype(bool)
        closed_right = grasp["closed_right"].astype(bool)
    lengths = {
        "state": len(state_abs),
        "ticks": len(ticks),
        "camera_match": len(camera_match),
        "closed_left": len(closed_left),
        "closed_right": len(closed_right),
        "relation_left_grasp": len(relation_left_grasp),
        "relation_right_grasp": len(relation_right_grasp),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{episode_dir.name} 各模态帧数不一致: {lengths}")
    if not (
        np.array_equal(closed_left, relation_left_grasp)
        and np.array_equal(closed_right, relation_right_grasp)
    ):
        raise ValueError(f"{episode_dir.name} Step1/Step2 二值加爪状态不一致")

    action_reference_tcp = state_abs[:, :18].copy()
    if variant == "continuous":
        observation_state = np.concatenate([relations, state_abs[:, 18:30]], axis=1)
        action = action_abs.copy()
        grasp_names = _continuous_names()
    else:
        current_binary = np.column_stack([closed_left, closed_right]).astype(np.float32)
        target_binary = np.concatenate([current_binary[1:], current_binary[-1:]], axis=0)
        observation_state = np.concatenate([relations, current_binary], axis=1)
        action = np.concatenate([action_abs[:, :18], target_binary], axis=1)
        grasp_names = ["left_grasp_binary", "right_grasp_binary"]
        if not np.isin(observation_state[:, 54:], (0.0, 1.0)).all() or not np.isin(action[:, 18:], (0.0, 1.0)).all():
            raise ValueError(f"{episode_dir.name} 二值加爪字段不是 0/1")

    return {
        "observation_state": observation_state.astype(np.float32),
        "action_reference_tcp": action_reference_tcp.astype(np.float32),
        "action": action.astype(np.float32),
        "ticks": ticks,
        "camera_match": camera_match,
        "task": _load_task(episode_dir),
        "state_names": _relation_names() + grasp_names,
        "action_names": _pose_names("left_tcp_absolute_target")
        + _pose_names("right_tcp_absolute_target")
        + grasp_names,
    }


def export_one(
    cfg: ProjectConfig,
    sources: list[Path],
    episode_dirs: list[Path],
    variant: str,
    destination: Path,
) -> dict[str, Any]:
    if variant not in VALID_VARIANTS:
        raise ValueError(f"未知导出方案 {variant!r}")
    if destination.exists():
        raise FileExistsError(f"输出已存在，避免覆盖: {destination}")
    (destination / "meta").mkdir(parents=True)

    task_to_index: dict[str, int] = {}
    episode_data = []
    for source, episode_dir in zip(sources, episode_dirs, strict=True):
        data = _load_episode(episode_dir, variant)
        task = data["task"]
        if task not in task_to_index:
            task_to_index[task] = len(task_to_index)
        data["task_index"] = task_to_index[task]
        episode_data.append(data)

    total_frames = 0
    episode_lines = []
    episode_stats = []
    all_states, all_references, all_actions = [], [], []
    height, width = 480, 640
    for episode_index, (source, episode_dir, data) in enumerate(
        zip(sources, episode_dirs, episode_data, strict=True)
    ):
        state = data["observation_state"]
        reference = data["action_reference_tcp"]
        action = data["action"]
        ticks = data["ticks"]
        camera_match = data["camera_match"]
        length = len(state)
        chunk = episode_index // cfg.export.chunks_size
        video_path = destination / "videos" / f"chunk-{chunk:03d}" / VIDEO_KEY / f"episode_{episode_index:06d}.mp4"
        height, width, video_frames = _write_video(
            source,
            episode_dir,
            camera_match,
            video_path,
            cfg.timeline.control_hz,
            cfg.export.video_codec,
            cfg.visual.source,
        )
        if video_frames != length:
            raise AssertionError("视频与标签帧数检查未生效")
        timestamp = ((ticks - ticks[0]) / 1e9).astype(np.float32)
        parquet_path = destination / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {
                    "observation.state": list(state),
                    "observation.action_reference_tcp": list(reference),
                    "action": list(action),
                    "timestamp": timestamp,
                    "frame_index": np.arange(length, dtype=np.int64),
                    "episode_index": np.full(length, episode_index, dtype=np.int64),
                    "index": np.arange(total_frames, total_frames + length, dtype=np.int64),
                    "task_index": np.full(length, data["task_index"], dtype=np.int64),
                }
            ),
            parquet_path,
        )
        stats = {
            "observation.state": _stats(state),
            "observation.action_reference_tcp": _stats(reference),
            "action": _stats(action),
        }
        episode_lines.append({"episode_index": episode_index, "tasks": [data["task"]], "length": length})
        episode_stats.append({"episode_index": episode_index, "stats": stats})
        all_states.append(state)
        all_references.append(reference)
        all_actions.append(action)
        total_frames += length

    state_dim = int(episode_data[0]["observation_state"].shape[1])
    action_dim = int(episode_data[0]["action"].shape[1])
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [state_dim],
            "names": [episode_data[0]["state_names"]],
        },
        "observation.action_reference_tcp": {
            "dtype": "float32",
            "shape": [18],
            "names": [
                _pose_names("left_tcp_absolute_current") + _pose_names("right_tcp_absolute_current")
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": [action_dim],
            "names": [episode_data[0]["action_names"]],
        },
        VIDEO_KEY: {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channel"],
            "info": {
                "video.height": height,
                "video.width": width,
                "video.codec": cfg.export.video_codec,
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": cfg.timeline.control_hz,
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
    info = {
        "codebase_version": "v2.1",
        "robot_type": cfg.export.robot_type,
        "total_episodes": len(sources),
        "total_frames": total_frames,
        "total_tasks": len(task_to_index),
        "total_chunks": (len(sources) - 1) // cfg.export.chunks_size + 1,
        "total_videos": len(sources),
        "chunks_size": cfg.export.chunks_size,
        "fps": cfg.timeline.control_hz,
        "splits": {"train": f"0:{len(sources)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
        "ego_relation": {
            "schema": "compact_hand_object_state_v1",
            "variant": variant,
            "object_order": list(EXPECTED_INSTANCE_IDS),
            "object_categories": list(EXPECTED_CATEGORIES),
            "relation_direction": "T_tcp_object",
            "relation_pose_encoding": "[tx,ty,tz,R[:,0],R[:,1]]",
            "relation_reference_frame": "TCP-local relation derived from camera0_static_frame poses",
            "action_storage": "absolute",
            "action_semantics": "action[t] contains T_g1_base_tcp target at min(t+1,T-1)",
            "action_pose_encoding": "[tx,ty,tz,R[:,0],R[:,1]]",
            "action_reference_field": "observation.action_reference_tcp",
            "action_reference_is_model_input": False,
            "training_action_transform": "deferred: inv(T_current) @ T_absolute_target for each TCP",
            "gripper_action_transform": "none",
            "visual_source": cfg.visual.source,
            "visual_alignment": "mode2/camera_match.npy maps the 30Hz control grid to source/BrainCo camera frames",
        },
    }
    meta = destination / "meta"
    (meta / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tasks_by_index = sorted(task_to_index.items(), key=lambda item: item[1])
    (meta / "tasks.jsonl").write_text(
        "".join(json.dumps({"task_index": index, "task": task}, ensure_ascii=False) + "\n" for task, index in tasks_by_index),
        encoding="utf-8",
    )
    (meta / "episodes.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in episode_lines), encoding="utf-8"
    )
    (meta / "episodes_stats.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in episode_stats), encoding="utf-8"
    )
    global_stats = {
        "observation.state": _stats(np.concatenate(all_states)),
        "observation.action_reference_tcp": _stats(np.concatenate(all_references)),
        "action": _stats(np.concatenate(all_actions)),
    }
    (meta / "stats.json").write_text(json.dumps(global_stats, indent=2) + "\n", encoding="utf-8")
    (destination / "extraction_meta.json").write_text(
        json.dumps(
            {
                "schema_version": cfg.schema_version,
                "variant": variant,
                "config": cfg.to_dict(),
                "sources": [str(path) for path in sources],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return info


def export_variants(
    cfg: ProjectConfig,
    sources: list[str | Path],
    episode_dirs: list[str | Path],
    variants: tuple[str, ...] | None = None,
    *,
    force: bool = False,
) -> dict[str, dict[str, Any]]:
    sources = [Path(path).resolve() for path in sources]
    episode_dirs = [Path(path).resolve() for path in episode_dirs]
    if len(sources) != len(episode_dirs):
        raise ValueError("sources 与 episode_dirs 数量不同")
    if not sources:
        raise ValueError("没有可导出的 episode")
    variants = variants or cfg.export.variants
    results = {}
    safe_prefix = cfg.export.repo_id_prefix.replace("/", "_")
    for variant in variants:
        destination = cfg.paths.output_dir / f"{safe_prefix}_{variant}"
        if destination.exists() and not force:
            info_path = destination / "meta/info.json"
            if not info_path.is_file():
                raise FileExistsError(f"输出目录已存在但不完整，请检查或使用 --force: {destination}")
            print(f"[Step4] 跳过已有数据集: {destination}", flush=True)
            results[variant] = json.loads(info_path.read_text(encoding="utf-8"))
            continue
        if not destination.exists():
            results[variant] = export_one(cfg, sources, episode_dirs, variant, destination)
            continue
        rebuilding = destination.with_name(destination.name + ".rebuilding")
        previous = destination.with_name(destination.name + ".previous")
        if rebuilding.exists():
            shutil.rmtree(rebuilding)
        if previous.exists():
            raise FileExistsError(f"发现未清理的备份，拒绝覆盖: {previous}")
        try:
            results[variant] = export_one(cfg, sources, episode_dirs, variant, rebuilding)
            destination.rename(previous)
            rebuilding.rename(destination)
        except Exception:
            if rebuilding.exists():
                shutil.rmtree(rebuilding)
            if previous.exists() and not destination.exists():
                previous.rename(destination)
            raise
        else:
            shutil.rmtree(previous)
    return results
