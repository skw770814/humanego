from __future__ import annotations

from pathlib import Path
import os
import subprocess

import h5py
import numpy as np
import pyarrow.parquet as pq

from ego_relation.config import ProjectConfig
from ego_relation.contracts.manifest import StageManifest, file_sha256
from ego_relation.contracts.se3 import compose, invert, nearest_indices, transform_to_pose6
from ego_relation.s1_pico_mode2.pico import PicoEpisode

XR26_TO_21 = [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25]
REVO2_DENORM = np.array([0.94, 0.70, 0.60, 1.47, 1.47, 1.47], dtype=np.float32)


def build_hand_swap_bridge(cfg: ProjectConfig, source: str | Path, episode_dir: str | Path) -> Path:
    """把 schema-2.1 原始 PICO + Mode2 输出桥接成现有 hand_swap_pipeline 的只读契约。"""
    source = Path(source).resolve()
    episode_dir = Path(episode_dir).resolve()
    episode_number = int(source.stem.rsplit("_", 1)[-1])
    bridge_dir = episode_dir / "visual" / "hand_swap_input"
    bridge_dir.mkdir(parents=True, exist_ok=True)
    output = bridge_dir / f"episode_{episode_number}.hdf5"
    camera_poses = np.load(episode_dir / "camera" / "T_camera_to_camera0.npy")
    with np.load(episode_dir / "camera" / "hands_camera0.npz", allow_pickle=False) as archive:
        left_pose = archive["T_camera0_left_hand"]
        right_pose = archive["T_camera0_right_hand"]
        left_landmarks = archive["left_joints_camera0"][:, XR26_TO_21]
        right_landmarks = archive["right_joints_camera0"][:, XR26_TO_21]
    ticks = np.load(episode_dir / "mode2" / "ticks_ns.npy")
    state = np.load(episode_dir / "mode2" / "state_abs.npy")
    frame_table = pq.read_table(episode_dir / "sync" / "camera_frames.parquet").to_pydict()
    camera_timestamps = np.asarray(frame_table["timestamp_ns"], dtype=np.int64)
    mode2_indices, _ = nearest_indices(ticks, camera_timestamps)
    left_qpos = state[mode2_indices, 18:24]
    right_qpos = state[mode2_indices, 24:30]
    left_wrist = np.stack(
        [transform_to_pose6(compose(invert(camera_poses[index]), left_pose[index])) for index in range(len(left_pose))]
    )
    right_wrist = np.stack(
        [
            transform_to_pose6(compose(invert(camera_poses[index]), right_pose[index]))
            for index in range(len(right_pose))
        ]
    )

    with PicoEpisode(source, cfg) as episode, h5py.File(output, "w") as file:
        images = file.create_dataset(
            "images",
            shape=(len(camera_timestamps), episode.image_size[1], episode.image_size[0], 3),
            dtype=np.uint8,
            chunks=(1, episode.image_size[1], episode.image_size[0], 3),
            compression="gzip",
            compression_opts=1,
        )
        for index in range(len(camera_timestamps)):
            images[index] = episode.image(index)
        file.create_dataset("camera_pose", data=camera_poses)
        file.create_dataset("camera_intrinsic", data=episode.K)
        file.create_dataset("timestamps_ns", data=camera_timestamps)
        file.create_dataset("left_landmarks_world", data=left_landmarks)
        file.create_dataset("right_landmarks_world", data=right_landmarks)
        file.create_dataset("wrist_left_pose", data=left_wrist)
        file.create_dataset("wrist_right_pose", data=right_wrist)
        file.create_dataset("fingers_left_qpos", data=left_qpos)
        file.create_dataset("fingers_right_qpos", data=right_qpos)
        file.attrs["coordinate_mode"] = "native"
        file.attrs["finger_unit"] = "brainco_normalized"
        file.attrs["finger_denormalize_scale_rad"] = REVO2_DENORM
        file.attrs["bridge_source"] = str(source)
        file.attrs["camera_frame"] = "cam0; wrist pose6 is current-camera relative"
    return output


def run_hand_swap(cfg: ProjectConfig, source: str | Path, episode_dir: str | Path) -> StageManifest:
    source = Path(source).resolve()
    episode_dir = Path(episode_dir).resolve()
    bridge = build_hand_swap_bridge(cfg, source, episode_dir)
    episode_number = int(source.stem.rsplit("_", 1)[-1])
    work_dir = episode_dir / "visual" / "hand_swap_work"
    environment = {
        **os.environ,
        "HDF5_DIR": str(bridge.parent),
        "WORK": str(work_dir),
        "OUT_DIR": str(episode_dir / "visual" / "hand_swap_output"),
    }
    object_mask_dir = episode_dir / "perception" / "humanego_session" / "preprocess" / "sam2_video_masks"
    if object_mask_dir.is_dir():
        environment["OBJECT_MASK_DIR"] = str(object_mask_dir)
    unified_python = str(cfg.visual.python)
    environment.setdefault("TOOL_PY", unified_python)
    environment.setdefault("EGODEX_PY", unified_python)
    environment.setdefault("EGODEX_DIR", str(cfg.paths.egodex_project))
    process = subprocess.run(
        ["bash", "run.sh", str(episode_number), "0,1,3"],
        cwd=cfg.paths.hand_swap_project,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log_path = episode_dir / "visual" / "hand_swap.log"
    log_path.write_text(process.stdout, encoding="utf-8")
    if process.returncode != 0:
        tail = "\n".join(process.stdout.splitlines()[-50:])
        raise RuntimeError(f"hand_swap_pipeline 失败（exit={process.returncode}）:\n{tail}")
    composite = work_dir / f"episode_{episode_number}" / "composite.mp4"
    if not composite.is_file():
        raise FileNotFoundError(composite)
    outputs = {"bridge": str(bridge), "composite": str(composite), "log": str(log_path)}
    comparison = composite.with_name("composite_sbs.mp4")
    if comparison.is_file():
        outputs["comparison"] = str(comparison)
    manifest = StageManifest(
        schema_version=cfg.schema_version,
        stage="hand_swap",
        episode=source.stem,
        source_path=str(source),
        source_sha256=file_sha256(source),
        config={"enabled": cfg.visual.enabled, "source": cfg.visual.source},
        outputs=outputs,
    )
    manifest.write(episode_dir / "hand_swap.manifest.json")
    return manifest
