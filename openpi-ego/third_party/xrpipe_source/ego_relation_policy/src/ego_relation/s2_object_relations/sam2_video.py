from __future__ import annotations

from contextlib import nullcontext
import gc
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from ego_relation.config import ProjectConfig


_COLORS = {
    "obj1": (60, 190, 80),
    "obj2": (40, 40, 235),
    "obj3": (20, 220, 245),
}


def _image_paths(preprocess_dir: Path) -> list[Path]:
    paths = sorted((preprocess_dir / "all_data").glob("*/rgb.png"))
    if not paths:
        raise FileNotFoundError(f"没有 staged RGB frames: {preprocess_dir / 'all_data'}")
    return paths


def _prepare_jpeg_frames(image_paths: list[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    expected_names = {f"{index:05d}.jpg" for index in range(len(image_paths))}
    for stale in destination.glob("*.jpg"):
        if stale.name not in expected_names:
            stale.unlink()
    for index, source in enumerate(image_paths):
        output = destination / f"{index:05d}.jpg"
        if output.is_file():
            continue
        image = cv2.imread(str(source))
        if image is None:
            raise FileNotFoundError(source)
        if not cv2.imwrite(str(output), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"无法写入 SAM2 video frame: {output}")


def _mask_sequence_metrics(areas: np.ndarray, centroids: np.ndarray) -> dict[str, Any]:
    areas = np.asarray(areas, dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64)
    initial = max(float(areas[0]), 1.0)
    ratios = areas / initial
    valid_centers = np.isfinite(centroids).all(axis=1)
    jumps = np.linalg.norm(np.diff(centroids[valid_centers], axis=0), axis=1)
    return {
        "initial_area_px": int(areas[0]),
        "nonempty_ratio": float(np.mean(areas > 0)),
        "area_ratio_median": float(np.median(ratios)),
        "area_ratio_p05": float(np.percentile(ratios, 5)),
        "area_ratio_p95": float(np.percentile(ratios, 95)),
        "tiny_mask_ratio": float(np.mean(ratios < 0.10)),
        "large_mask_ratio": float(np.mean(ratios > 4.0)),
        "centroid_jump_p95_px": float(np.percentile(jumps, 95)) if len(jumps) else None,
    }


def _true_runs(mask: np.ndarray) -> list[list[int]]:
    padded = np.r_[False, np.asarray(mask, dtype=bool), False]
    changes = np.diff(padded.astype(np.int8))
    return [
        [int(start), int(end - 1)]
        for start, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1), strict=True)
    ]


def _render_metrics_chart(
    catalog: list[dict],
    areas: np.ndarray,
    cotracker_visibility: dict[str, np.ndarray],
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    frames = np.arange(len(areas))
    figure, axes = plt.subplots(len(catalog), 1, figsize=(14, 3.1 * len(catalog)), sharex=True, squeeze=False)
    for object_index, row in enumerate(catalog):
        axis = axes[object_index, 0]
        instance_id = str(row["instance_id"])
        area_ratio = areas[:, object_index] / max(float(areas[0, object_index]), 1.0)
        axis.plot(frames, area_ratio, label="SAM2 mask area / frame-0 area", color="#1976d2", linewidth=1.2)
        visibility = cotracker_visibility.get(instance_id)
        if visibility is not None:
            visible_ratio = visibility.sum(axis=1) / max(float(visibility.shape[1]), 1.0)
            axis.plot(frames, visible_ratio, label="CoTracker visible points ratio", color="#d32f2f", linewidth=1.0)
            rescued = (visible_ratio < (3.0 / visibility.shape[1])) & (area_ratio >= 0.10)
            for start, end in _true_runs(rescued):
                axis.axvspan(start, end, color="#2e7d32", alpha=0.18)
        axis.axhline(0.1, color="#555555", linestyle="--", linewidth=0.8, alpha=0.7)
        axis.set_title(f"{instance_id}: {row['category']}")
        axis.set_ylabel("normalized signal")
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right", fontsize=8)
    axes[-1, 0].set_xlabel("camera frame")
    figure.suptitle("SAM2 Video vs CoTracker | green = mask survives while fewer than 3 points are visible")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _render_video(
    image_paths: list[Path],
    catalog: list[dict],
    mask_root: Path,
    areas: np.ndarray,
    scores: np.ndarray,
    output: Path,
    *,
    fps: float,
    cotracker_visibility: dict[str, np.ndarray],
) -> None:
    writer = None
    try:
        for frame, image_path in enumerate(image_paths):
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(image_path)
            overlay = image.copy()
            lines = []
            for object_index, row in enumerate(catalog):
                instance_id = str(row["instance_id"])
                color = _COLORS.get(instance_id, (220, 220, 220))
                mask = cv2.imread(
                    str(mask_root / instance_id / f"{frame:05d}.png"),
                    cv2.IMREAD_GRAYSCALE,
                )
                if mask is None:
                    raise FileNotFoundError(mask_root / instance_id / f"{frame:05d}.png")
                support = mask > 127
                overlay[support] = color
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image, contours, -1, color, 2, cv2.LINE_AA)
                visibility = cotracker_visibility.get(instance_id)
                visible_points = int(visibility[frame].sum()) if visibility is not None else -1
                visible_text = f" | CoTracker {visible_points:02d}" if visible_points >= 0 else ""
                lines.append(
                    f"{instance_id} {row['category']} | mask {int(areas[frame, object_index]):5d}px "
                    f"| score {scores[frame, object_index]:.3f}{visible_text}"
                )
            image = cv2.addWeighted(image, 0.66, overlay, 0.34, 0.0)
            header = np.full((52, image.shape[1], 3), 24, dtype=np.uint8)
            cv2.putText(
                header,
                f"SAM2 Video masks | frame {frame:03d}/{len(image_paths) - 1:03d} | {frame / fps:05.2f}s",
                (14, 31),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )
            footer = np.full((28 + 25 * len(lines), image.shape[1], 3), 24, dtype=np.uint8)
            for line_index, line in enumerate(lines):
                cv2.putText(
                    footer,
                    line,
                    (14, 24 + line_index * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (235, 235, 235),
                    1,
                    cv2.LINE_AA,
                )
            composite = np.vstack((header, image, footer))
            if writer is None:
                output.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(
                    str(output),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (composite.shape[1], composite.shape[0]),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"无法创建视频: {output}")
            writer.write(composite)
    finally:
        if writer is not None:
            writer.release()


def _project_point(K: np.ndarray, point: np.ndarray) -> tuple[int, int] | None:
    if not np.isfinite(point).all() or point[2] <= 1e-5:
        return None
    pixel = K @ point
    return int(round(pixel[0] / pixel[2])), int(round(pixel[1] / pixel[2]))


def render_sam2_pose_video(cfg: ProjectConfig, episode_dir: str | Path) -> Path:
    """Overlay SAM2 identities and the stereo/CoTracker 6DoF pose estimate."""
    episode_dir = Path(episode_dir).resolve()
    preprocess = episode_dir / "perception" / "humanego_session" / "preprocess"
    all_data = preprocess / "all_data"
    mask_root = preprocess / "sam2_video_masks"
    catalog_path = preprocess / "object_catalog.json"
    pose_path = episode_dir / "entities" / "objects_stereo_track.npz"
    metrics_path = episode_dir / "qa" / "sam2_video_metrics.npz"
    required = (catalog_path, pose_path, metrics_path)
    missing = [str(path) for path in required if not path.is_file()]
    if not mask_root.is_dir():
        missing.append(str(mask_root))
    if missing:
        raise FileNotFoundError(f"SAM2 pose video 缺少输入: {missing}")

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog_by_id = {str(row["instance_id"]): row for row in catalog}
    K = np.load(episode_dir / "camera" / "K.npy")
    camera_to_camera0 = np.load(episode_dir / "camera" / "T_camera_to_camera0.npy")
    with np.load(pose_path, allow_pickle=False) as archive:
        instance_ids = archive["instance_ids"]
        poses_camera0 = archive["T_camera0_object"]
        pose_valid = archive["valid"]
        confidence = archive["confidence"]
    with np.load(metrics_path, allow_pickle=False) as archive:
        metric_ids = [str(value) for value in archive["instance_ids"]]
        areas = archive["areas"]
        scores = archive["scores"]
    metric_index = {instance_id: index for index, instance_id in enumerate(metric_ids)}

    frame_count = min(len(camera_to_camera0), len(poses_camera0), len(areas))
    frame_output = episode_dir / "qa" / "sam2_pose_frames"
    frame_output.mkdir(parents=True, exist_ok=True)
    for stale in frame_output.glob("*.jpg"):
        stale.unlink()
    video_output = episode_dir / "qa" / f"{episode_dir.name}_sam2_pose.mp4"
    axis_colors = ((40, 40, 240), (40, 210, 40), (240, 100, 30))
    writer = None
    fps = float(cfg.timeline.control_hz)

    try:
        for frame in range(frame_count):
            rgb_path = all_data / f"{frame:05d}" / "rgb.png"
            rgb = cv2.imread(str(rgb_path))
            if rgb is None:
                raise FileNotFoundError(rgb_path)
            overlay = rgb.copy()
            masks: dict[str, np.ndarray] = {}
            for instance_id_value in instance_ids:
                instance_id = str(instance_id_value)
                mask_path = mask_root / instance_id / f"{frame:05d}.png"
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise FileNotFoundError(mask_path)
                masks[instance_id] = mask
                support = mask > 127
                overlay[support] = _COLORS.get(instance_id, (220, 220, 220))
            rgb = cv2.addWeighted(rgb, 0.68, overlay, 0.32, 0.0)

            camera0_to_camera = np.linalg.inv(camera_to_camera0[frame])
            status_lines = []
            for object_index, instance_id_value in enumerate(instance_ids):
                instance_id = str(instance_id_value)
                row = catalog_by_id[instance_id]
                color = _COLORS.get(instance_id, (220, 220, 220))
                mask = masks[instance_id]
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(rgb, contours, -1, color, 2, cv2.LINE_AA)

                pose_camera = camera0_to_camera @ poses_camera0[frame, object_index]
                origin = _project_point(K, pose_camera[:3, 3])
                is_valid = bool(pose_valid[frame, object_index])
                thickness = 2 if is_valid else 1
                if origin is not None:
                    cv2.circle(rgb, origin, 6, color, thickness, cv2.LINE_AA)
                    for axis, axis_color in enumerate(axis_colors):
                        endpoint = _project_point(
                            K,
                            pose_camera[:3, 3] + 0.04 * pose_camera[:3, axis],
                        )
                        if endpoint is not None:
                            cv2.arrowedLine(
                                rgb,
                                origin,
                                endpoint,
                                axis_color,
                                thickness,
                                cv2.LINE_AA,
                                0,
                                0.25,
                            )
                    cv2.putText(
                        rgb,
                        f"{instance_id} {'POSE' if is_valid else 'HELD'}",
                        (origin[0] + 8, int(np.clip(origin[1] - 8, 14, rgb.shape[0] - 10))),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.43,
                        color,
                        1,
                        cv2.LINE_AA,
                    )

                mask_index = metric_index[instance_id]
                mask_present = int(areas[frame, mask_index]) > 0
                xyz = poses_camera0[frame, object_index, :3, 3]
                status_lines.append(
                    f"{instance_id} {row['category']} | mask {'visible' if mask_present else 'lost'} "
                    f"{int(areas[frame, mask_index])}px {scores[frame, mask_index]:.2f} | "
                    f"pose {'valid' if is_valid else 'held'} {confidence[frame, object_index]:.2f} | "
                    f"cam0 xyz [{xyz[0]:+.3f}, {xyz[1]:+.3f}, {xyz[2]:+.3f}]m"
                )

            header = np.full((52, rgb.shape[1], 3), 24, dtype=np.uint8)
            cv2.putText(
                header,
                f"SAM2 masks + stereo 6DoF pose | frame {frame:03d}/{frame_count - 1:03d} | {frame / fps:05.2f}s",
                (14, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                header,
                "pose axes: X=red Y=green Z=blue | HELD = last valid pose",
                (14, 44),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (190, 190, 190),
                1,
                cv2.LINE_AA,
            )
            footer = np.full((88, rgb.shape[1], 3), 24, dtype=np.uint8)
            for line_index, line in enumerate(status_lines):
                cv2.putText(
                    footer,
                    line,
                    (12, 20 + line_index * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.36,
                    (232, 232, 232),
                    1,
                    cv2.LINE_AA,
                )
            composite = np.vstack((header, rgb, footer))
            cv2.imwrite(
                str(frame_output / f"{frame:05d}.jpg"),
                composite,
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )
            if writer is None:
                writer = cv2.VideoWriter(
                    str(video_output),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (composite.shape[1], composite.shape[0]),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"无法创建视频: {video_output}")
            writer.write(composite)
    finally:
        if writer is not None:
            writer.release()
    return video_output


def run_sam2_video_masks(
    cfg: ProjectConfig,
    episode_dir: str | Path,
    *,
    render_pose: bool = True,
) -> dict[str, Path]:
    """Propagate frame-0 masks as an auxiliary identity and region signal."""
    import torch
    from huggingface_hub import hf_hub_download
    from sam2.build_sam import build_sam2_video_predictor

    episode_dir = Path(episode_dir).resolve()
    preprocess = episode_dir / "perception" / "humanego_session" / "preprocess"
    catalog_path = preprocess / "object_catalog.json"
    config_path = preprocess / "cfg" / "DINOSAM.yaml"
    if not catalog_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("请先运行 dinosam 和 expand_instances")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    image_paths = _image_paths(preprocess)
    first_frame = image_paths[0].parent
    for row in catalog:
        initial_mask = first_frame / f"mask_{row['instance_id']}.png"
        if not initial_mask.is_file():
            raise FileNotFoundError(initial_mask)

    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    checkpoint = hf_hub_download(
        repo_id=str(raw_config["sam2_repo_id"]),
        filename=str(raw_config["sam2_checkpoint_name"]),
        cache_dir=str(cfg.paths.models_dir / "huggingface" / "hub"),
        local_files_only=True,
    )
    frame_cache = preprocess / "sam2_video_frames"
    mask_root = preprocess / "sam2_video_masks"
    _prepare_jpeg_frames(image_paths, frame_cache)
    mask_root.mkdir(parents=True, exist_ok=True)
    for row in catalog:
        object_dir = mask_root / str(row["instance_id"])
        object_dir.mkdir(parents=True, exist_ok=True)
        for stale in object_dir.glob("*.png"):
            stale.unlink()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = build_sam2_video_predictor(
        str(raw_config["sam2_config"]),
        checkpoint,
        device=device,
    )
    state = predictor.init_state(
        video_path=str(frame_cache),
        offload_video_to_cpu=True,
        offload_state_to_cpu=device == "cuda",
        async_loading_frames=False,
    )
    object_id_map: dict[int, str] = {}
    for numeric_id, row in enumerate(catalog, start=1):
        instance_id = str(row["instance_id"])
        initial_mask = cv2.imread(str(first_frame / f"mask_{instance_id}.png"), cv2.IMREAD_GRAYSCALE)
        if initial_mask is None:
            raise FileNotFoundError(first_frame / f"mask_{instance_id}.png")
        predictor.add_new_mask(state, frame_idx=0, obj_id=numeric_id, mask=initial_mask > 127)
        object_id_map[numeric_id] = instance_id

    frame_count = len(image_paths)
    object_count = len(catalog)
    areas = np.zeros((frame_count, object_count), dtype=np.int32)
    scores = np.zeros((frame_count, object_count), dtype=np.float32)
    centroids = np.full((frame_count, object_count, 2), np.nan, dtype=np.float32)
    index_by_instance = {str(row["instance_id"]): index for index, row in enumerate(catalog)}
    context = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
    try:
        with context:
            for frame, output_ids, mask_logits in predictor.propagate_in_video(state):
                for output_index, numeric_id in enumerate(output_ids):
                    instance_id = object_id_map[int(numeric_id)]
                    object_index = index_by_instance[instance_id]
                    logits = mask_logits[output_index, 0]
                    support_tensor = logits > 0
                    support = support_tensor.detach().cpu().numpy()
                    mask = support.astype(np.uint8) * 255
                    cv2.imwrite(str(mask_root / instance_id / f"{int(frame):05d}.png"), mask)
                    area = int(support.sum())
                    areas[int(frame), object_index] = area
                    if area:
                        ys, xs = np.nonzero(support)
                        centroids[int(frame), object_index] = [float(xs.mean()), float(ys.mean())]
                        scores[int(frame), object_index] = float(torch.sigmoid(logits[support_tensor]).mean().item())
    finally:
        del state
        del predictor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    missing = [
        str(mask_root / str(row["instance_id"]) / f"{frame:05d}.png")
        for frame in range(frame_count)
        for row in catalog
        if not (mask_root / str(row["instance_id"]) / f"{frame:05d}.png").is_file()
    ]
    if missing:
        raise RuntimeError(f"SAM2 Video 缺少 {len(missing)} 个输出，首个: {missing[0]}")

    overlap_pixels = np.zeros(frame_count, dtype=np.int32)
    for frame in range(frame_count):
        occupancy = None
        for row in catalog:
            mask = cv2.imread(
                str(mask_root / str(row["instance_id"]) / f"{frame:05d}.png"),
                cv2.IMREAD_GRAYSCALE,
            )
            support = mask > 127
            occupancy = support.astype(np.uint8) if occupancy is None else occupancy + support
        overlap_pixels[frame] = int(np.count_nonzero(occupancy > 1))

    cotracker_visibility: dict[str, np.ndarray] = {}
    cotracker_path = preprocess / "cotracker_results.json"
    if cotracker_path.is_file():
        cotracker = json.loads(cotracker_path.read_text(encoding="utf-8"))
        for row in catalog:
            instance_id = str(row["instance_id"])
            if instance_id in cotracker:
                cotracker_visibility[instance_id] = (
                    np.asarray(cotracker[instance_id]["visibility"], dtype=np.float32)[:frame_count] >= 0.5
                )

    object_metrics = {}
    for index, row in enumerate(catalog):
        instance_id = str(row["instance_id"])
        initial_area = max(int(areas[0, index]), 1)
        mask_present = areas[:, index] >= 0.10 * initial_area
        metrics = {
            "category": str(row["category"]),
            **_mask_sequence_metrics(areas[:, index], centroids[:, index]),
            "score_median": float(np.median(scores[:, index])),
            "empty_mask_runs": _true_runs(areas[:, index] == 0),
        }
        visibility = cotracker_visibility.get(instance_id)
        if visibility is not None:
            cotracker_lost = visibility.sum(axis=1) < 3
            rescued = cotracker_lost & mask_present
            metrics.update(
                {
                    "cotracker_fewer_than_3_runs": _true_runs(cotracker_lost),
                    "sam2_mask_survives_cotracker_loss_runs": _true_runs(rescued),
                    "sam2_rescued_frames": int(rescued.sum()),
                }
            )
        object_metrics[instance_id] = metrics

    report = {
        "method": "SAM2VideoPredictor frame-0 mask propagation",
        "device": device,
        "frames": frame_count,
        "objects": object_metrics,
        "overlap_pixels_p95": float(np.percentile(overlap_pixels, 95)),
        "outputs_are_diagnostic_only": True,
    }
    report_path = episode_dir / "qa" / "sam2_video_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(
        episode_dir / "qa" / "sam2_video_metrics.npz",
        instance_ids=np.asarray([str(row["instance_id"]) for row in catalog]),
        areas=areas,
        scores=scores,
        centroids=centroids,
        overlap_pixels=overlap_pixels,
    )

    chart_path = episode_dir / "qa" / "sam2_video_quality.png"
    _render_metrics_chart(catalog, areas, cotracker_visibility, chart_path)
    video_path = episode_dir / "qa" / f"{episode_dir.name}_sam2_video_masks.mp4"
    _render_video(
        image_paths,
        catalog,
        mask_root,
        areas,
        scores,
        video_path,
        fps=float(cfg.timeline.control_hz),
        cotracker_visibility=cotracker_visibility,
    )
    outputs = {"masks": mask_root, "report": report_path, "chart": chart_path, "video": video_path}
    if render_pose and (episode_dir / "entities" / "objects_stereo_track.npz").is_file():
        outputs["pose_video"] = render_sam2_pose_video(cfg, episode_dir)
    return outputs
