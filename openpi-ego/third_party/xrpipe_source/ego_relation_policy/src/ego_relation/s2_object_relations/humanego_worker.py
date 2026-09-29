from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np


WORKER_STEPS = (
    "check_inputs",
    "dinosam",
    "expand_instances",
    "keypoints",
    "cotracker_reset",
    "cotracker_frame",
    "cotracker_all",
    "stereo_init",
    "camtriangulator",
    "all",
)


_COLOR_HSV_RANGES = {
    "red": (((0, 100, 50), (12, 255, 255)), ((168, 100, 50), (179, 255, 255))),
    "yellow": (((18, 100, 80), (40, 255, 255)),),
}
_MINIMUM_MASK_COLOR_FRACTION = 0.70


def _category_color(category: str) -> str | None:
    first_word = str(category).strip().lower().split(maxsplit=1)[0]
    return first_word if first_word in _COLOR_HSV_RANGES else None


def _color_support(hsv: np.ndarray, color: str) -> np.ndarray:
    support = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in _COLOR_HSV_RANGES[color]:
        support |= cv2.inRange(hsv, np.asarray(lower), np.asarray(upper))
    return cv2.morphologyEx(
        support,
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8),
    )


def _color_candidate_box(
    image: np.ndarray,
    color: str,
    minimum_area: int,
) -> tuple[int, int, int, int] | None:
    support = _color_support(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), color)
    count, _, stats, _ = cv2.connectedComponentsWithStats(support, connectivity=8)
    candidates = []
    for index in range(1, count):
        x, y, width, height, area = map(int, stats[index])
        aspect = width / max(height, 1)
        fill = area / max(width * height, 1)
        if area >= minimum_area and 0.45 <= aspect <= 2.2 and fill >= 0.40:
            candidates.append((area * fill, (x, y, x + width - 1, y + height - 1)))
    return max(candidates, default=(0.0, None), key=lambda item: item[0])[1]


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    first_support = first > 127
    second_support = second > 127
    union = np.count_nonzero(first_support | second_support)
    return 0.0 if not union else float(np.count_nonzero(first_support & second_support) / union)


def _mask_color_fraction(mask: np.ndarray, hsv: np.ndarray, color: str) -> float:
    mask_support = mask > 127
    area = np.count_nonzero(mask_support)
    if not area:
        return 0.0
    color_support = _color_support(hsv, color) > 0
    return float(np.count_nonzero(mask_support & color_support) / area)


def _repair_color_instance_masks(
    preprocess_dir: Path,
    catalog: list[dict],
    dino,
    *,
    minimum_area: int,
) -> dict:
    reference_dir = preprocess_dir / "all_data" / "00000"
    image = cv2.imread(str(reference_dir / "rgb.png"))
    if image is None:
        raise FileNotFoundError(reference_dir / "rgb.png")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    masks = {
        str(row["instance_id"]): cv2.imread(
            str(reference_dir / f"mask_{row['instance_id']}.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        for row in catalog
    }
    if any(mask is None for mask in masks.values()):
        raise RuntimeError("DINO-SAM did not write every configured object mask")

    duplicated = set()
    for first_index, first in enumerate(catalog):
        for second in catalog[first_index + 1 :]:
            if first["category"] == second["category"]:
                continue
            if _mask_iou(masks[str(first["instance_id"])], masks[str(second["instance_id"])]) >= 0.80:
                duplicated.update((str(first["instance_id"]), str(second["instance_id"])))

    report = {"duplicate_instance_ids_before_repair": sorted(duplicated), "objects": {}}
    for row in catalog:
        instance_id = str(row["instance_id"])
        color = _category_color(str(row["category"]))
        mask = masks[instance_id]
        before_fraction = _mask_color_fraction(mask, hsv, color) if color else None
        repaired = False
        candidate_box = None
        if color and (
            instance_id in duplicated
            or before_fraction < _MINIMUM_MASK_COLOR_FRACTION
        ):
            candidate_box = _color_candidate_box(image, color, minimum_area)
            if candidate_box is None:
                raise RuntimeError(
                    f"{instance_id} {row['category']} mask is incorrect and no {color} candidate was found"
                )
            x1, y1, x2, y2 = candidate_box
            padding = 4
            box = np.asarray(
                [[
                    max(x1 - padding, 0),
                    max(y1 - padding, 0),
                    min(x2 + padding, image.shape[1] - 1),
                    min(y2 + padding, image.shape[0] - 1),
                ]],
                dtype=np.float32,
            )
            sam_masks, _, _ = dino.engine.predictor.predict(
                box=box,
                multimask_output=False,
            )
            repaired_mask = np.asarray(sam_masks).squeeze().astype(np.uint8) * 255
            after_fraction = _mask_color_fraction(repaired_mask, hsv, color)
            if (
                np.count_nonzero(repaired_mask) < minimum_area
                or after_fraction < _MINIMUM_MASK_COLOR_FRACTION
            ):
                raise RuntimeError(
                    f"SAM2 color repair failed for {instance_id} {row['category']}: "
                    f"area={np.count_nonzero(repaired_mask)}, color_fraction={after_fraction:.3f}"
                )
            cv2.imwrite(str(reference_dir / f"mask_{instance_id}.png"), repaired_mask)
            masks[instance_id] = repaired_mask
            repaired = True
        after_fraction = _mask_color_fraction(masks[instance_id], hsv, color) if color else None
        report["objects"][instance_id] = {
            "category": str(row["category"]),
            "expected_color": color,
            "color_fraction_before": before_fraction,
            "color_fraction_after": after_fraction,
            "candidate_box": list(candidate_box) if candidate_box else None,
            "repaired": repaired,
        }

    remaining_duplicates = []
    for first_index, first in enumerate(catalog):
        for second in catalog[first_index + 1 :]:
            if first["category"] == second["category"]:
                continue
            iou = _mask_iou(masks[str(first["instance_id"])], masks[str(second["instance_id"])])
            if iou >= 0.80:
                remaining_duplicates.append(
                    {"first": first["instance_id"], "second": second["instance_id"], "iou": iou}
                )
    report["duplicate_pairs_after_repair"] = remaining_duplicates
    if remaining_duplicates:
        raise RuntimeError(f"DINO-SAM returned duplicate category masks: {remaining_duplicates}")
    report_path = preprocess_dir / "dinosam_mask_qa.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _expand_instances(preprocess_dir: Path, catalog: list[dict], minimum_area: int, maximum: int) -> list[dict]:
    """Split category masks into deterministic per-instance masks on frame 0."""
    reference_dir = preprocess_dir / "all_data" / "00000"
    candidates: list[tuple[dict, np.ndarray]] = []
    for row in catalog:
        mask = cv2.imread(str(reference_dir / f"mask_{row['instance_id']}.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(reference_dir / f"mask_{row['instance_id']}.png")
        count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 127).astype(np.uint8), connectivity=8)
        components = [
            (int(stats[index, cv2.CC_STAT_AREA]), index)
            for index in range(1, count)
            if int(stats[index, cv2.CC_STAT_AREA]) >= minimum_area
        ]
        components.sort(reverse=True)
        expected = int(row.get("expected_instances", 0))
        take = expected if expected > 0 else len(components)
        if len(components) < take:
            raise RuntimeError(
                f"{row['category']} expected {take} instances, but only {len(components)} masks exceed {minimum_area}px"
            )
        for instance_index, (area, label) in enumerate(components[:take]):
            if len(candidates) >= maximum:
                raise RuntimeError(f"Object instances exceed maximum {maximum}; increase max_entities")
            instance_mask = np.where(labels == label, 255, 0).astype(np.uint8)
            candidates.append(
                (
                    {
                    **row,
                    "instance_index": instance_index,
                    "mask_area_px": area,
                    },
                    instance_mask,
                )
            )
    if not candidates:
        raise RuntimeError("Grounding DINO + SAM2 produced no usable object instance")
    for first_index, (first_row, first_mask) in enumerate(candidates):
        for second_row, second_mask in candidates[first_index + 1 :]:
            if first_row["category"] == second_row["category"]:
                continue
            iou = _mask_iou(first_mask, second_mask)
            if iou >= 0.80:
                raise RuntimeError(
                    "DINO-SAM category masks overlap as one object: "
                    f"{first_row['category']} vs {second_row['category']}, IoU={iou:.3f}"
                )
    candidates.sort(key=lambda item: (item[0]["category"], item[0]["instance_index"]))
    expanded: list[dict] = []
    for index, (row, mask) in enumerate(candidates, start=1):
        row = dict(row)
        row["is_anchor"] = False
        row["instance_id"] = f"obj{index}"
        cv2.imwrite(str(reference_dir / f"mask_{row['instance_id']}.png"), mask)
        expanded.append(row)
    return expanded


def _sample_depth(depth: np.ndarray, u: float, v: float, radius: int) -> float | None:
    x, y = int(round(u)), int(round(v))
    y0, y1 = max(0, y - radius), min(depth.shape[0], y + radius + 1)
    x0, x1 = max(0, x - radius), min(depth.shape[1], x + radius + 1)
    values = depth[y0:y1, x0:x1]
    values = values[values > 0]
    return None if not len(values) else float(np.median(values)) / 1000.0


def _mask_stereo_center(
    depth: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    *,
    erosion_radius: int,
    minimum_depths: int,
) -> tuple[np.ndarray, dict]:
    """Estimate translation without letting sparse depth pull the 2D center sideways."""
    if depth.shape != mask.shape:
        raise ValueError(f"Depth/mask shape mismatch: {depth.shape} != {mask.shape}")
    support = mask > 127
    ys, xs = np.nonzero(support)
    if not len(xs):
        raise RuntimeError("Object mask is empty")

    inner = support.astype(np.uint8)
    if erosion_radius > 0:
        size = 2 * erosion_radius + 1
        eroded = cv2.erode(inner, np.ones((size, size), dtype=np.uint8)) > 0
        if np.count_nonzero(depth[eroded]) >= minimum_depths:
            inner = eroded
        else:
            inner = support
    else:
        inner = support

    depth_mm = depth[inner]
    depth_mm = depth_mm[depth_mm > 0].astype(np.float64)
    if len(depth_mm) < minimum_depths:
        raise RuntimeError(f"Object mask has only {len(depth_mm)} stereo depths; need {minimum_depths}")

    u = float(np.mean(xs))
    v = float(np.mean(ys))
    z = float(np.median(depth_mm)) / 1000.0
    center = np.array(
        [(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z],
        dtype=np.float64,
    )
    depth_mad_m = float(np.median(np.abs(depth_mm - np.median(depth_mm)))) / 1000.0
    info = {
        "translation_source": "pico_stereo_sgbm_mask_robust",
        "translation_center_pixel": [u, v],
        "stereo_mask_pixels": int(np.count_nonzero(support)),
        "stereo_mask_depths": int(len(depth_mm)),
        "stereo_mask_depth_coverage": float(len(depth_mm) / max(np.count_nonzero(inner), 1)),
        "stereo_mask_depth_median_m": z,
        "stereo_mask_depth_mad_m": depth_mad_m,
    }
    return center, info


def _stereo_initial_poses(
    preprocess_dir: Path,
    depth_dir: Path,
    catalog: list[dict],
    keypoints: dict,
    *,
    pose_method: str,
    patch_radius: int,
    minimum_depths: int,
) -> None:
    from preprocess.OrientAnything import estimate_frame_pca1, estimate_frame_pca2, estimate_frame_vlm
    from preprocess.OrientAnything import get_crop_from_2d_kpts

    image_path = preprocess_dir / "all_data" / "00000" / "rgb.png"
    image = cv2.imread(str(image_path))
    depth = cv2.imread(str(depth_dir / "00000.png"), cv2.IMREAD_UNCHANGED)
    if image is None or depth is None or depth.dtype != np.uint16:
        raise RuntimeError("Stereo initialization requires frame 0 RGB and uint16 millimeter depth")
    camera = json.loads((image_path.parent / "aria_cam_rgb.json").read_text(encoding="utf-8"))
    K = np.asarray(camera["k"], dtype=np.float64)
    objects = {}
    for row in catalog:
        points_2d = np.asarray(keypoints["objects"][row["instance_id"]], dtype=np.float64)
        points_3d = []
        for u, v in points_2d:
            z = _sample_depth(depth, u, v, patch_radius)
            if z is None:
                continue
            points_3d.append([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])
        points_3d = np.asarray(points_3d, dtype=np.float64)
        if len(points_3d) < minimum_depths:
            raise RuntimeError(
                f"{row['instance_id']} has only {len(points_3d)} stereo keypoint depths; need {minimum_depths}"
            )
        mask = cv2.imread(str(image_path.parent / f"mask_{row['instance_id']}.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(image_path.parent / f"mask_{row['instance_id']}.png")
        center, center_info = _mask_stereo_center(
            depth,
            mask,
            K,
            erosion_radius=patch_radius,
            minimum_depths=minimum_depths,
        )
        method = pose_method.lower()
        if method == "vlm":
            crop = get_crop_from_2d_kpts(image, points_2d)
            pose, info = estimate_frame_vlm(
                crop,
                center,
                is_anchor=True,
                anchor_center_cam=None,
                do_rm_bkg=True,
            )
            if "error" in info:
                raise RuntimeError(f"Orient-Anything failed for {row['instance_id']}: {info['error']}")
        elif method == "pca1":
            pose, info = estimate_frame_pca1(
                points_3d,
                is_anchor=True,
                anchor_center_cam=None,
            )
        elif method == "pca2":
            pose, info = estimate_frame_pca2(
                points_3d,
                is_anchor=True,
                anchor_center_cam=None,
            )
        else:
            raise ValueError(f"Unsupported pose method: {pose_method}")
        pose[:3, 3] = center
        info = {
            **info,
            **center_info,
            "runtime_orientation_method": info.get("method"),
            "method": f"{method}_camera_absolute",
            "orientation_reference": "camera_frame",
            "catalog_is_anchor": bool(row["is_anchor"]),
            "stereo_keypoint_depths": len(points_3d),
        }
        info["confidence"] = float(len(points_3d) / max(len(points_2d), 1))
        objects[row["instance_id"]] = {"object_to_cam0_matrix": pose.tolist(), "info": info}
    (preprocess_dir / "camtriangulator_results.json").write_text(
        json.dumps(
            {
                "objects": objects,
                "initial_translation_source": "pico_stereo_sgbm_mask_robust",
                "initial_orientation_reference": "camera_frame",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _preprocess_dir(session_root: Path) -> Path:
    return session_root / "preprocess"


def _image_paths(session_root: Path) -> list[Path]:
    image_paths = sorted((_preprocess_dir(session_root) / "all_data").glob("*/rgb.png"))
    if not image_paths:
        raise RuntimeError("没有 staged RGB frames")
    return image_paths


def _read_catalog(preprocess_dir: Path) -> list[dict]:
    return json.loads((preprocess_dir / "object_catalog.json").read_text(encoding="utf-8"))


def run_check_inputs(args: argparse.Namespace) -> None:
    preprocess_dir = _preprocess_dir(args.session_root)
    image_paths = _image_paths(args.session_root)
    required = {
        "session_root": args.session_root,
        "preprocess_dir": preprocess_dir,
        "object_catalog": preprocess_dir / "object_catalog.json",
        "dinosam_config": args.dinosam_config,
        "keypoints_config": args.keypoints_config,
        "cotracker_config": args.cotracker_config,
        "triangulator_config": args.triangulator_config,
        "depth_dir": args.depth_dir,
    }
    missing = [str(path) for path in required.values() if not Path(path).exists()]
    first_frame = image_paths[0].parent
    first_frame_required = [
        first_frame / "rgb.png",
        first_frame / "aria_cam_rgb.json",
    ]
    missing.extend(str(path) for path in first_frame_required if not path.exists())
    result = {
        "ok": not missing,
        "frames": len(image_paths),
        "first_frame": str(image_paths[0]),
        "last_frame": str(image_paths[-1]),
        "missing": missing,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if missing:
        raise FileNotFoundError("HumanEgo staged input 缺文件: " + ", ".join(missing[:8]))


def run_dinosam_step(args: argparse.Namespace, image_paths: list[Path] | None = None) -> None:
    from preprocess.DINOSAM import DINOSAM
    from utils.utils_io import load_cfg

    preprocess_dir = _preprocess_dir(args.session_root)
    image_paths = image_paths or _image_paths(args.session_root)
    dino_cfg = load_cfg(str(args.dinosam_config))
    dino = DINOSAM(str(args.dinosam_config))
    try:
        # DINO+SAM initializes instances once. CoTracker3 carries them through
        # the sequence; running the detector on every frame only adds latency
        # and does not enter HumanEgo's offline tracker.
        dino_visualization, _ = dino.process_and_save(str(image_paths[0]), dino_cfg.dinosam_prompt)
        _repair_color_instance_masks(
            preprocess_dir,
            _read_catalog(preprocess_dir),
            dino,
            minimum_area=args.minimum_instance_area_px,
        )
        if dino_visualization is not None:
            cv2.imwrite(str(preprocess_dir / "dinosam_reference.jpg"), dino_visualization)
    finally:
        dino.cleanup()


def run_expand_instances_step(args: argparse.Namespace) -> list[dict]:
    preprocess_dir = _preprocess_dir(args.session_root)
    catalog = _expand_instances(
        preprocess_dir,
        _read_catalog(preprocess_dir),
        args.minimum_instance_area_px,
        args.maximum_object_instances,
    )
    (preprocess_dir / "object_catalog.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    return catalog


def run_keypoints_step(args: argparse.Namespace, image_paths: list[Path] | None = None) -> dict:
    from preprocess.KptsSelector import run_kptsselector

    preprocess_dir = _preprocess_dir(args.session_root)
    image_paths = image_paths or _image_paths(args.session_root)
    catalog = _read_catalog(preprocess_dir)
    reference = image_paths[0]
    keypoints = {"objects": {}}
    for row in catalog:
        instance_id = row["instance_id"]
        points = run_kptsselector(
            str(args.keypoints_config),
            str(reference.parent / f"mask_{instance_id}.png"),
            str(preprocess_dir / f"kptsselector_vis_{instance_id}.png"),
            rgb_path=str(reference),
        )
        if not points:
            raise RuntimeError(f"{instance_id} 没有可跟踪关键点")
        keypoints["objects"][instance_id] = points
    (preprocess_dir / "kptsselector_results.json").write_text(json.dumps(keypoints, indent=2), encoding="utf-8")
    return keypoints


def run_cotracker_reset_step() -> None:
    from preprocess.CoTrackerOffline import reset_cotracker_offline

    reset_cotracker_offline()


def run_cotracker_frame_step(args: argparse.Namespace, image_paths: list[Path] | None = None) -> None:
    from preprocess.CoTrackerOffline import run_cotracker_offline

    image_paths = image_paths or _image_paths(args.session_root)
    if args.frame_index < 0 or args.frame_index >= len(image_paths):
        raise IndexError(f"--frame-index 越界: {args.frame_index}, frames={len(image_paths)}")
    run_cotracker_offline(
        str(image_paths[args.frame_index]),
        str(args.cotracker_config),
        args.frame_index,
        all_image_paths=[str(path) for path in image_paths],
        mps_path=str(args.session_root),
    )


def run_cotracker_all_step(args: argparse.Namespace, image_paths: list[Path] | None = None) -> None:
    image_paths = image_paths or _image_paths(args.session_root)
    run_cotracker_reset_step()
    for index in range(len(image_paths)):
        frame_args = argparse.Namespace(**{**vars(args), "frame_index": index})
        run_cotracker_frame_step(frame_args, image_paths)


def run_stereo_init_step(args: argparse.Namespace) -> None:
    preprocess_dir = _preprocess_dir(args.session_root)
    _stereo_initial_poses(
        preprocess_dir,
        args.depth_dir,
        _read_catalog(preprocess_dir),
        json.loads((preprocess_dir / "kptsselector_results.json").read_text(encoding="utf-8")),
        pose_method=args.pose_method,
        patch_radius=args.depth_patch_radius_px,
        minimum_depths=args.minimum_keypoint_depths,
    )


def run_camtriangulator_step(args: argparse.Namespace, image_paths: list[Path] | None = None) -> None:
    from preprocess.CamTriangulator import run_camtriagulator

    preprocess_dir = _preprocess_dir(args.session_root)
    image_paths = image_paths or _image_paths(args.session_root)
    window = json.loads((preprocess_dir / "perception_window.json").read_text(encoding="utf-8"))
    triangulation_paths = image_paths[: int(window["triangulation_frame_count"])]
    for index, image_path in enumerate(triangulation_paths):
        visualization = run_camtriagulator(
            str(image_path),
            str(args.triangulator_config),
            index,
            all_image_paths=[str(path) for path in triangulation_paths],
            mps_path=str(args.session_root),
        )
        if visualization is not None and index == len(triangulation_paths) - 1:
            cv2.imwrite(str(preprocess_dir / "camtriangulator_last_frame.png"), visualization)


def run_all_steps(args: argparse.Namespace) -> None:
    preprocess_dir = _preprocess_dir(args.session_root)
    image_paths = _image_paths(args.session_root)

    run_dinosam_step(args, image_paths)
    catalog = run_expand_instances_step(args)
    keypoints = run_keypoints_step(args, image_paths)
    run_cotracker_all_step(args, image_paths)

    try:
        _stereo_initial_poses(
            preprocess_dir,
            args.depth_dir,
            catalog,
            keypoints,
            pose_method=args.pose_method,
            patch_radius=args.depth_patch_radius_px,
            minimum_depths=args.minimum_keypoint_depths,
        )
    except RuntimeError as stereo_error:
        print(f"[stereo initialization failed, falling back to multiview] {stereo_error}")
        run_camtriangulator_step(args, image_paths)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--humanego-root", type=Path, required=True)
    parser.add_argument("--session-root", type=Path, required=True)
    parser.add_argument("--dinosam-config", type=Path, required=True)
    parser.add_argument("--keypoints-config", type=Path, required=True)
    parser.add_argument("--cotracker-config", type=Path, required=True)
    parser.add_argument("--triangulator-config", type=Path, required=True)
    parser.add_argument("--depth-dir", type=Path, required=True)
    parser.add_argument("--minimum-instance-area-px", type=int, required=True)
    parser.add_argument("--maximum-object-instances", type=int, required=True)
    parser.add_argument("--minimum-keypoint-depths", type=int, required=True)
    parser.add_argument("--depth-patch-radius-px", type=int, required=True)
    parser.add_argument("--pose-method", choices=("vlm", "pca1", "pca2"), required=True)
    parser.add_argument("--step", choices=WORKER_STEPS, default="all")
    parser.add_argument("--frame-index", type=int, default=0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    sys.path.insert(0, str(args.humanego_root))
    if args.step == "check_inputs":
        run_check_inputs(args)
    elif args.step == "dinosam":
        run_dinosam_step(args)
    elif args.step == "expand_instances":
        run_expand_instances_step(args)
    elif args.step == "keypoints":
        run_keypoints_step(args)
    elif args.step == "cotracker_reset":
        run_cotracker_reset_step()
    elif args.step == "cotracker_frame":
        run_cotracker_frame_step(args)
    elif args.step == "cotracker_all":
        run_cotracker_all_step(args)
    elif args.step == "stereo_init":
        run_stereo_init_step(args)
    elif args.step == "camtriangulator":
        run_camtriangulator_step(args)
    elif args.step == "all":
        run_all_steps(args)


if __name__ == "__main__":
    main()
