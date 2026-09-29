from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import cv2
import h5py
import numpy as np
import yaml

from ego_relation.config import ProjectConfig
from ego_relation.contracts.manifest import StageManifest, file_sha256
from ego_relation.s1_pico_mode2.pico import PicoEpisode


def _episode_prompts(source: Path, cfg: ProjectConfig, episode_dir: Path | None = None) -> tuple[list[dict], str]:
    contract_path = episode_dir / "step1" / "object_instances.json" if episode_dir is not None else None
    if contract_path is not None and contract_path.is_file():
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        catalog = [dict(row) for row in contract.get("instances", [])]
        if not catalog:
            raise ValueError(f"Step1 对象实例契约为空: {contract_path}")
        return catalog, ""
    with h5py.File(source, "r") as file:
        raw = dict(cfg.perception.object_prompts)
        if not raw:
            raw = json.loads(str(file.attrs.get("object_prompts_json", "{}")))
        anchor_category = cfg.perception.anchor_category.strip() or str(file.attrs.get("anchor_object", ""))
    if not raw:
        raise ValueError("未配置 perception.object_prompts，HDF5 也没有 object_prompts_json")
    catalog = []
    for index, (category, prompt) in enumerate(raw.items(), start=1):
        prompt = str(prompt).strip()
        if not prompt.endswith("."):
            prompt += " ."
        catalog.append(
            {
                "instance_id": f"obj{index}",
                "category": str(category).strip(" ,"),
                "prompt": prompt,
                # Main HumanEgo ICT experiments use a static camera reference frame.
                # Keep the field for backward-compatible artifacts, but do not
                # promote any object to a coordinate anchor by default.
                "is_anchor": False,
                "expected_instances": int(cfg.perception.instance_counts.get(str(category).strip(" ,"), 0)),
            }
        )
    return catalog, anchor_category


def _write_merged_config(base: Path, destination: Path, updates: dict) -> None:
    raw = yaml.safe_load(base.read_text(encoding="utf-8")) if base.is_file() else {}
    if raw is None:
        raw = {}
    raw.update(updates)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _fallback_instances(catalog: list[dict], maximum: int) -> list[dict]:
    rows: list[dict] = []
    for row in catalog:
        count = int(row.get("expected_instances", 0)) or 1
        for instance_index in range(count):
            if len(rows) >= maximum:
                raise RuntimeError(f"Object instances exceed maximum {maximum}; increase perception.max_entities")
            rows.append({**row, "instance_index": instance_index})
    rows.sort(key=lambda item: (str(item.get("category")), int(item["instance_index"])))
    for index, row in enumerate(rows, start=1):
        row["is_anchor"] = False
        row["instance_id"] = f"obj{index}"
    return rows


def _nearest_valid_uv(valid_uv: np.ndarray, target: np.ndarray) -> np.ndarray:
    if not len(valid_uv):
        return target
    index = int(np.argmin(np.sum((valid_uv - target[None]) ** 2, axis=1)))
    return valid_uv[index]


def _circle_keypoints(center: np.ndarray, radius: float, count: int, width: int, height: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, max(count, 4), endpoint=False)
    points = np.stack([center[0] + np.cos(angles) * radius, center[1] + np.sin(angles) * radius], axis=1)
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    return points[:count]


def run_simple_stereo_fallback(
    cfg: ProjectConfig,
    source: Path,
    episode_dir: Path,
    paths: dict[str, Path],
    *,
    reason: str,
) -> StageManifest:
    """Debug-only fallback that lets the full pipeline run without HumanEgo source.

    It creates deterministic pseudo object instances from frame-0 stereo depth.
    The output contract is identical to the HumanEgo import path, but the
    manifest warning makes it explicit that these are not detector/tracker
    results and should not be used for final training conclusions.
    """

    preprocess_dir = paths["preprocess"]
    all_data = preprocess_dir / "all_data"
    catalog = _fallback_instances(
        json.loads((preprocess_dir / "object_catalog.json").read_text(encoding="utf-8")),
        cfg.perception.max_entities - 2,
    )
    (preprocess_dir / "object_catalog.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    image_path = all_data / "00000" / "rgb.png"
    image = cv2.imread(str(image_path))
    depth = cv2.imread(str(episode_dir / "depth" / "stereo_mm" / "00000.png"), cv2.IMREAD_UNCHANGED)
    if image is None or depth is None:
        raise RuntimeError("simple_stereo_fallback 需要 staged frame0 RGB 和 depth/stereo_mm/00000.png")
    height, width = image.shape[:2]
    K = np.load(episode_dir / "camera" / "K.npy")
    camera_poses = np.load(episode_dir / "camera" / "T_camera_to_camera0.npy")
    frame_count = len(camera_poses)
    valid_y, valid_x = np.nonzero(depth > 0)
    valid_uv = np.stack([valid_x, valid_y], axis=1).astype(np.float64) if len(valid_x) else np.zeros((0, 2))

    objects = {}
    tracks: dict[str, dict] = {}
    poses = []
    categories = []
    ids = []
    anchors = []
    confidences = []
    keypoints_result = {"objects": {}}
    count = max(len(catalog), 1)
    for object_index, row in enumerate(catalog):
        target = np.asarray([(object_index + 1) * width / (count + 1), height * 0.55], dtype=np.float64)
        center_uv = _nearest_valid_uv(valid_uv, target)
        u, v = float(center_uv[0]), float(center_uv[1])
        z = float(depth[int(round(v)), int(round(u))]) / 1000.0 if len(valid_uv) else 1.0
        if z <= 0:
            z = 1.0
        center_camera = np.asarray([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])
        center_cam0 = camera_poses[0, :3, :3] @ center_camera + camera_poses[0, :3, 3]
        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = center_cam0
        instance_id = str(row["instance_id"])
        objects[instance_id] = {
            "object_to_cam0_matrix": transform.tolist(),
            "info": {
                "confidence": 0.1,
                "method": "simple_stereo_fallback",
                "reason": reason,
                "center_uv": [u, v],
            },
        }
        radius = max(14.0, min(width, height) * 0.045)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, (int(round(u)), int(round(v))), int(round(radius)), 255, -1, cv2.LINE_AA)
        cv2.imwrite(str(image_path.parent / f"mask_{instance_id}.png"), mask)
        keypoints = _circle_keypoints(center_uv, radius, cfg.perception.keypoints_per_object, width, height)
        keypoints_result["objects"][instance_id] = keypoints.tolist()
        repeated = np.repeat(keypoints[None], frame_count, axis=0)
        tracks[instance_id] = {
            "tracks": repeated.tolist(),
            "visibility": np.ones((frame_count, len(keypoints)), dtype=np.float32).tolist(),
            "method": "simple_stereo_fallback_constant_uv",
        }
        ids.append(instance_id)
        categories.append(str(row["category"]))
        anchors.append(bool(row["is_anchor"]))
        poses.append(transform)
        confidences.append(0.1)

    (preprocess_dir / "kptsselector_results.json").write_text(
        json.dumps(keypoints_result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (preprocess_dir / "cotracker_results.json").write_text(
        json.dumps(tracks, ensure_ascii=False), encoding="utf-8"
    )
    (preprocess_dir / "camtriangulator_results.json").write_text(
        json.dumps(
            {
                "objects": objects,
                "initial_translation_source": "simple_stereo_fallback_frame0_depth",
                "warning": "debug fallback only; no DINO/SAM/CoTracker/Orient-Anything was run",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    entities_dir = episode_dir / "entities"
    entities_dir.mkdir(parents=True, exist_ok=True)
    initial = np.stack(poses).astype(np.float64)
    np.savez_compressed(
        entities_dir / "objects_initial.npz",
        instance_ids=np.asarray(ids),
        categories=np.asarray(categories),
        is_anchor=np.asarray(anchors, dtype=bool),
        T_camera0_object=initial,
        confidence=np.asarray(confidences, dtype=np.float32),
    )
    np.savez_compressed(
        entities_dir / "objects_stereo_track.npz",
        instance_ids=np.asarray(ids),
        T_camera0_object=np.repeat(initial[None], frame_count, axis=0),
        valid=np.ones((frame_count, len(ids)), dtype=bool),
        confidence=np.full((frame_count, len(ids)), 0.1, dtype=np.float32),
        residual_m=np.full((frame_count, len(ids)), np.nan, dtype=np.float32),
    )
    (episode_dir / "qa" / "stereo_object_track_report.json").write_text(
        json.dumps(
            {
                "frames": frame_count,
                "objects": len(ids),
                "valid_ratio_per_object": {instance_id: 1.0 for instance_id in ids},
                "median_residual_m_per_object": {instance_id: None for instance_id in ids},
                "method": "simple_stereo_fallback_constant_pose",
                "warning": reason,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log_path = preprocess_dir / "humanego.log"
    log_path.write_text(f"simple_stereo_fallback used: {reason}\n", encoding="utf-8")
    manifest = StageManifest(
        schema_version=cfg.schema_version,
        stage="humanego_perception",
        episode=source.stem,
        source_path=str(source),
        source_sha256=file_sha256(source),
        config={
            "method": "simple_stereo_fallback",
            "objects": len(ids),
            "reason": reason,
        },
        outputs={
            "session": str(paths["root"]),
            "objects_initial": str(entities_dir / "objects_initial.npz"),
            "objects_stereo_track": str(entities_dir / "objects_stereo_track.npz"),
            "log": str(log_path),
        },
        metrics={"fallback": True, "objects": len(ids), "frames": frame_count},
        warnings=(
            "DEBUG FALLBACK: 未运行 GroundingDINO/SAM2/CoTracker/Orient-Anything；"
            "仅用于打通 Step2->Step4 数据契约，不可作为正式训练感知结果。",
            reason,
        ),
    )
    manifest.write(episode_dir / "humanego_perception.manifest.json")
    return manifest


def stage_humanego_input(cfg: ProjectConfig, source: Path, episode_dir: Path) -> dict[str, Path]:
    root = episode_dir / "perception" / "humanego_session"
    if root.exists():
        # This is a derived stage directory. Starting clean prevents masks or
        # tracks from a previous prompt/instance-count configuration leaking
        # into a rerun.
        shutil.rmtree(root)
    preprocess_dir = root / "preprocess"
    all_data = preprocess_dir / "all_data"
    config_dir = preprocess_dir / "cfg"
    all_data.mkdir(parents=True, exist_ok=True)
    catalog, _ = _episode_prompts(source, cfg, episode_dir)
    (preprocess_dir / "object_catalog.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    camera_poses = np.load(episode_dir / "camera" / "T_camera_to_camera0.npy")
    K = np.load(episode_dir / "camera" / "K.npy")
    with np.load(episode_dir / "camera" / "hands_camera0.npz", allow_pickle=False) as hand_archive:
        pinch = np.minimum(hand_archive["left_pinch_m"], hand_archive["right_pinch_m"])
    grasp_indices = np.flatnonzero(pinch < cfg.perception.grasp_distance_m)
    scan_frame_count = int(grasp_indices[0]) if len(grasp_indices) else len(camera_poses)
    scan_frame_count = max(min(scan_frame_count, len(camera_poses)), min(10, len(camera_poses)))
    (preprocess_dir / "perception_window.json").write_text(
        json.dumps(
            {
                "tracking_frame_count": int(len(camera_poses)),
                "triangulation_frame_count": scan_frame_count,
                "rule": "frames before first PICO thumb-index pinch; minimum 10",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with PicoEpisode(source, cfg) as episode:
        timestamps = episode.camera_timestamps
        for index in range(len(timestamps)):
            frame_dir = all_data / f"{index:05d}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            image = episode.image(index)
            cv2.imwrite(str(frame_dir / "rgb.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            camera_json = {
                "idx": index,
                "ts": int(timestamps[index]),
                "k": K.tolist(),
                # HumanEgo CamTriangulator 契约是 camera-to-world；这里 world=cam0。
                "c2w": camera_poses[index].tolist(),
                "coordinate_system": "opencv_rh_x_right_y_down_z_forward",
                "world_frame": "pico_left_camera_frame_0",
            }
            (frame_dir / "aria_cam_rgb.json").write_text(json.dumps(camera_json, indent=2), encoding="utf-8")

    base = cfg.paths.humanego_project / "cfg" / "preprocess" / "base"
    prompts = {row["instance_id"]: row["prompt"] for row in catalog}
    _write_merged_config(
        base / "DINOSAM.yaml",
        config_dir / "DINOSAM.yaml",
        {
            "box_threshold": cfg.perception.dino_box_threshold,
            "dinosam_prompt": prompts,
        },
    )
    bands = max(2, int(np.ceil(cfg.perception.keypoints_per_object / 2)))
    _write_merged_config(base / "KptsSelector.yaml", config_dir / "KptsSelector.yaml", {"kpts_n_bands": bands})
    _write_merged_config(base / "CoTracker.yaml", config_dir / "CoTracker.yaml", {"ref_idx": 0})
    pose_method = {row["instance_id"]: cfg.perception.pose_method for row in catalog}
    pose_method["default"] = cfg.perception.pose_method
    _write_merged_config(
        base / "CamTriangulator.yaml",
        config_dir / "CamTriangulator.yaml",
        {"step": cfg.perception.triangulation_step, "pose_method": pose_method},
    )
    return {
        "root": root,
        "preprocess": preprocess_dir,
        "dinosam": config_dir / "DINOSAM.yaml",
        "keypoints": config_dir / "KptsSelector.yaml",
        "cotracker": config_dir / "CoTracker.yaml",
        "triangulator": config_dir / "CamTriangulator.yaml",
    }


def run_humanego(
    cfg: ProjectConfig,
    source: str | Path,
    episode_dir: str | Path,
    *,
    allow_unverified_calibration: bool = False,
) -> StageManifest:
    source = Path(source).resolve()
    episode_dir = Path(episode_dir).resolve()
    qa = json.loads((episode_dir / "qa" / "pico_report.json").read_text(encoding="utf-8"))
    if not qa["geometry_deployable"] and not allow_unverified_calibration:
        raise RuntimeError(
            "相机外参未通过标定 QA，已阻止三角化。完成标定并设置 calibration_verified=true，"
            "或仅为调试显式传 --allow-unverified-calibration。"
        )
    paths = stage_humanego_input(cfg, source, episode_dir)
    required_humanego_files = [
        cfg.paths.humanego_project / "preprocess" / "DINOSAM.py",
        cfg.paths.humanego_project / "preprocess" / "KptsSelector.py",
        cfg.paths.humanego_project / "preprocess" / "CoTrackerOffline.py",
        cfg.paths.humanego_project / "preprocess" / "CamTriangulator.py",
    ]
    missing_humanego_files = [path for path in required_humanego_files if not path.is_file()]
    if missing_humanego_files:
        reason = "HumanEgo internal runtime is incomplete: " + ", ".join(
            str(path) for path in missing_humanego_files[:4]
        )
        if not cfg.perception.allow_debug_fallback:
            raise FileNotFoundError(reason)
        return run_simple_stereo_fallback(
            cfg,
            source,
            episode_dir,
            paths,
            reason=reason,
        )
    worker = Path(__file__).with_name("humanego_worker.py")
    command = [
        cfg.perception.python,
        str(worker),
        "--humanego-root",
        str(cfg.paths.humanego_project),
        "--session-root",
        str(paths["root"]),
        "--dinosam-config",
        str(paths["dinosam"]),
        "--keypoints-config",
        str(paths["keypoints"]),
        "--cotracker-config",
        str(paths["cotracker"]),
        "--triangulator-config",
        str(paths["triangulator"]),
        "--depth-dir",
        str(episode_dir / "depth" / "stereo_mm"),
        "--minimum-instance-area-px",
        str(cfg.perception.minimum_instance_area_px),
        "--maximum-object-instances",
        str(cfg.perception.max_entities - 2),
        "--minimum-keypoint-depths",
        str(cfg.depth.minimum_keypoint_depths),
        "--depth-patch-radius-px",
        str(cfg.depth.patch_radius_px),
        "--pose-method",
        str(cfg.perception.pose_method),
    ]
    environment = {**os.environ, "PYTHONPATH": str(cfg.paths.humanego_project)}
    environment.setdefault("HF_HOME", str(cfg.paths.models_dir / "huggingface"))
    environment.setdefault("HUGGINGFACE_HUB_CACHE", str(cfg.paths.models_dir / "huggingface/hub"))
    environment.setdefault("TORCH_HOME", str(cfg.paths.models_dir / "torch"))
    process = subprocess.run(
        command,
        cwd=cfg.paths.humanego_project,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log_path = paths["preprocess"] / "humanego.log"
    log_path.write_text(process.stdout, encoding="utf-8")
    if process.returncode != 0:
        tail = "\n".join(process.stdout.splitlines()[-50:])
        raise RuntimeError(f"HumanEgo 感知失败（exit={process.returncode}）:\n{tail}")
    objects_path = import_humanego_results(episode_dir)
    from ego_relation.s2_object_relations.stereo_fusion import fuse_stereo_tracks

    stereo_tracks_path = fuse_stereo_tracks(cfg, episode_dir)
    warnings = () if qa["geometry_deployable"] else ("本次在未验证外参下强制运行，结果只可调试。",)
    manifest = StageManifest(
        schema_version=cfg.schema_version,
        stage="humanego_perception",
        episode=source.stem,
        source_path=str(source),
        source_sha256=file_sha256(source),
        config={
            "pose_method": cfg.perception.pose_method,
            "triangulation_step": cfg.perception.triangulation_step,
            "keypoints_per_object": cfg.perception.keypoints_per_object,
        },
        outputs={
            "session": str(paths["root"]),
            "objects_initial": str(objects_path),
            "objects_stereo_track": str(stereo_tracks_path),
            "log": str(log_path),
        },
        metrics={"geometry_deployable": qa["geometry_deployable"]},
        warnings=warnings,
    )
    manifest.write(episode_dir / "humanego_perception.manifest.json")
    return manifest


def import_humanego_results(episode_dir: str | Path) -> Path:
    episode_dir = Path(episode_dir).resolve()
    preprocess_dir = episode_dir / "perception" / "humanego_session" / "preprocess"
    result = json.loads((preprocess_dir / "camtriangulator_results.json").read_text(encoding="utf-8"))
    catalog = json.loads((preprocess_dir / "object_catalog.json").read_text(encoding="utf-8"))
    by_id = {row["instance_id"]: row for row in catalog}
    ids, categories, anchors, poses, confidences = [], [], [], [], []
    for instance_id, value in sorted(result["objects"].items()):
        ids.append(instance_id)
        categories.append(by_id[instance_id]["category"])
        anchors.append(bool(by_id[instance_id]["is_anchor"]))
        poses.append(np.asarray(value["object_to_cam0_matrix"], dtype=np.float64))
        info = value.get("info", {})
        confidences.append(float(info.get("confidence", 1.0)))
    destination = episode_dir / "entities" / "objects_initial.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        instance_ids=np.asarray(ids),
        categories=np.asarray(categories),
        is_anchor=np.asarray(anchors, dtype=bool),
        T_camera0_object=np.stack(poses),
        confidence=np.asarray(confidences, dtype=np.float32),
    )
    return destination
