from __future__ import annotations

import json
from pathlib import Path

import cv2
import h5py
import numpy as np

from ego_relation.config import ProjectConfig
from ego_relation.contracts.manifest import StageManifest, file_sha256
from ego_relation.contracts.se3 import compose, invert, matrix_from_pose7, rotation_angle_deg
from ego_relation.s1_pico_mode2.pico import PicoEpisode


def stereo_geometry(file: h5py.File) -> dict:
    # Native XrCameraExtrinsics poses share one OpenXR device frame and their
    # local axes are camera optical axes; direct relative composition is valid.
    left = matrix_from_pose7(file["camera/extrinsics_left"][:])
    right = matrix_from_pose7(file["camera/extrinsics_right"][:])
    T_left_right = compose(invert(left), right)
    return {
        "T_left_right": T_left_right,
        "baseline_m": float(np.linalg.norm(T_left_right[:3, 3])),
        "relative_rotation_deg": rotation_angle_deg(T_left_right[:3, :3]),
        "horizontal_baseline_ratio": float(
            abs(T_left_right[0, 3]) / max(np.linalg.norm(T_left_right[:3, 3]), 1e-12)
        ),
    }


def _matcher(cfg: ProjectConfig, *, right: bool = False):
    disparities = int(np.ceil(cfg.depth.num_disparities / 16.0) * 16)
    block_size = int(cfg.depth.block_size)
    if block_size < 3 or block_size % 2 == 0:
        raise ValueError("depth.block_size 必须是大于等于 3 的奇数")
    return cv2.StereoSGBM_create(
        minDisparity=-disparities if right else 0,
        numDisparities=disparities,
        blockSize=block_size,
        P1=8 * block_size * block_size,
        P2=32 * block_size * block_size,
        disp12MaxDiff=1,
        uniquenessRatio=cfg.depth.uniqueness_ratio,
        speckleWindowSize=cfg.depth.speckle_window_size,
        speckleRange=cfg.depth.speckle_range,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def _rectification(file: h5py.File, image_size: tuple[int, int]) -> dict:
    """Build zero-distortion rectification for PICO PINHOLE stereo images."""
    K_left = np.asarray(file["camera/K_left"][:], dtype=np.float64)
    K_right = np.asarray(file["camera/K_right"][:], dtype=np.float64)
    geometry = stereo_geometry(file)
    # T_left_right is the pose of the right camera in the left camera frame.
    # OpenCV expects X_right = R * X_left + T.
    T_right_left = invert(geometry["T_left_right"])
    zero_distortion = np.zeros((5, 1), dtype=np.float64)
    R_left, R_right, P_left, P_right, Q, roi_left, roi_right = cv2.stereoRectify(
        K_left,
        zero_distortion,
        K_right,
        zero_distortion,
        image_size,
        np.asarray(T_right_left[:3, :3], dtype=np.float64),
        np.asarray(T_right_left[:3, 3], dtype=np.float64).reshape(3, 1),
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )
    left_x, left_y = cv2.initUndistortRectifyMap(
        K_left, zero_distortion, R_left, P_left, image_size, cv2.CV_32FC1
    )
    right_x, right_y = cv2.initUndistortRectifyMap(
        K_right, zero_distortion, R_right, P_right, image_size, cv2.CV_32FC1
    )

    # Build the inverse lookup used to return rectified Z to original left-image
    # pixels. This preserves alignment with HumanEgo masks and keypoints.
    width, height = image_size
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64)
    )
    original_pixels = np.stack([grid_x, grid_y, np.ones_like(grid_x)], axis=-1)
    original_rays = original_pixels @ np.linalg.inv(K_left).T
    rectified_rays = original_rays @ R_left.T
    ray_z = rectified_rays[..., 2]
    K_rectified = P_left[:3, :3]
    original_to_rectified_x = (
        K_rectified[0, 0] * rectified_rays[..., 0] / ray_z + K_rectified[0, 2]
    ).astype(np.float32)
    original_to_rectified_y = (
        K_rectified[1, 1] * rectified_rays[..., 1] / ray_z + K_rectified[1, 2]
    ).astype(np.float32)
    original_depth_scale = np.divide(
        1.0,
        ray_z,
        out=np.zeros_like(ray_z),
        where=np.abs(ray_z) > 1e-12,
    ).astype(np.float32)
    baseline_rectified = abs(float(P_right[0, 3] / P_right[0, 0]))
    return {
        **geometry,
        "K_rectified": K_rectified,
        "baseline_rectified_m": baseline_rectified,
        "R_left": R_left,
        "R_right": R_right,
        "left_maps": (left_x, left_y),
        "right_maps": (right_x, right_y),
        "original_lookup": (
            original_to_rectified_x,
            original_to_rectified_y,
            original_depth_scale,
        ),
        "Q": Q,
        "roi_left": tuple(int(value) for value in roi_left),
        "roi_right": tuple(int(value) for value in roi_right),
    }


def _left_right_consistency(
    disparity_left: np.ndarray,
    disparity_right: np.ndarray,
    maximum_difference_px: float,
) -> np.ndarray:
    height, width = disparity_left.shape
    grid_y, grid_x = np.indices((height, width))
    right_x = np.rint(grid_x - disparity_left).astype(np.int32)
    inside = (right_x >= 0) & (right_x < width) & (disparity_left > 0.5)
    sampled_right = np.zeros_like(disparity_left)
    sampled_right[inside] = disparity_right[grid_y[inside], right_x[inside]]
    return (
        inside
        & (sampled_right < -0.5)
        & (np.abs(disparity_left + sampled_right) <= maximum_difference_px)
    )


def compute_stereo_depth(
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    K: np.ndarray,
    baseline_m: float,
    cfg: ProjectConfig,
    matcher=None,
    right_matcher=None,
) -> tuple[np.ndarray, np.ndarray]:
    if left_rgb.shape != right_rgb.shape:
        raise ValueError(f"左右图尺寸不同: {left_rgb.shape} vs {right_rgb.shape}")
    if baseline_m <= 0:
        raise ValueError("双目 baseline 必须大于 0")
    gray_left = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2GRAY)
    gray_right = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2GRAY)
    matcher = matcher or _matcher(cfg)
    disparity_raw = matcher.compute(gray_left, gray_right)
    # OpenCV 的固定点 disparity 是真实像素的 16 倍。先在原始整数域去散斑。
    cv2.filterSpeckles(
        disparity_raw,
        0,
        cfg.depth.speckle_window_size,
        int(cfg.depth.speckle_range * 16),
    )
    disparity = disparity_raw.astype(np.float32) / 16.0
    depth = np.zeros_like(disparity, dtype=np.float32)
    valid = disparity > 0.5
    if cfg.depth.left_right_consistency:
        if right_matcher is None:
            right_matcher = _matcher(cfg, right=True)
        disparity_right = right_matcher.compute(gray_right, gray_left).astype(np.float32) / 16.0
        valid &= _left_right_consistency(
            disparity,
            disparity_right,
            cfg.depth.left_right_max_diff_px,
        )
    depth[valid] = float(K[0, 0]) * baseline_m / disparity[valid]
    valid &= np.isfinite(depth)
    valid &= depth >= cfg.depth.min_depth_m
    valid &= depth <= cfg.depth.max_depth_m
    depth[~valid] = 0.0
    return depth, disparity


def _epipolar_report(
    left: np.ndarray,
    right: np.ndarray,
    disparity: np.ndarray,
    valid_depth: np.ndarray,
) -> dict:
    empty = {
        "matches": 0,
        "positive_epipolar_matches": 0,
        "vertical_abs_px": [],
        "disparity_px": [],
        "sgbm_disparity_error_px": [],
        "left_points": [],
        "right_points": [],
        "sgbm_disparity_px": [],
    }
    points = cv2.goodFeaturesToTrack(left, maxCorners=1000, qualityLevel=0.01, minDistance=7, blockSize=7)
    if points is None:
        return empty
    left_points = points.reshape(-1, 2)
    x = np.rint(left_points[:, 0]).astype(np.int32)
    y = np.rint(left_points[:, 1]).astype(np.int32)
    initial_disparity = disparity[y, x]
    keep = (
        (initial_disparity > 0.5)
        & (left_points[:, 0] - initial_disparity >= 0)
        & valid_depth[y, x]
    )
    left_points = left_points[keep]
    initial_disparity = initial_disparity[keep]
    if not len(left_points):
        return empty
    right_initial = np.column_stack(
        [left_points[:, 0] - initial_disparity, left_points[:, 1]]
    ).astype(np.float32)
    flow_args = {
        "winSize": (15, 15),
        "maxLevel": 2,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.001),
        "flags": cv2.OPTFLOW_USE_INITIAL_FLOW,
    }
    right_points, status_forward, error_forward = cv2.calcOpticalFlowPyrLK(
        left, right, left_points.astype(np.float32), right_initial, **flow_args
    )
    if right_points is None or status_forward is None or error_forward is None:
        return empty
    left_returned, status_backward, _ = cv2.calcOpticalFlowPyrLK(
        right, left, right_points, left_points.astype(np.float32), **flow_args
    )
    if left_returned is None or status_backward is None:
        return empty
    forward_backward_error = np.linalg.norm(left_returned - left_points, axis=1)
    horizontal = left_points[:, 0] - right_points[:, 0]
    reliable = (
        status_forward.ravel().astype(bool)
        & status_backward.ravel().astype(bool)
        & (forward_backward_error < 0.1)
        & (error_forward.ravel() < 5.0)
        & (horizontal > 0.5)
    )
    vertical = left_points[:, 1] - right_points[:, 1]
    return {
        "matches": int(len(left_points)),
        "positive_epipolar_matches": int(reliable.sum()),
        "vertical_abs_px": np.abs(vertical[reliable]).tolist(),
        "disparity_px": horizontal[reliable].tolist(),
        "sgbm_disparity_error_px": np.abs(horizontal[reliable] - initial_disparity[reliable]).tolist(),
        "left_points": left_points[reliable].tolist(),
        "right_points": right_points[reliable].tolist(),
        "sgbm_disparity_px": initial_disparity[reliable].tolist(),
    }


def _fundamental_ransac_inliers(
    left_points: np.ndarray,
    right_points: np.ndarray,
    *,
    threshold_px: float = 1.0,
) -> np.ndarray:
    """Reject sparse stereo correspondence outliers without changing image geometry."""
    left_points = np.asarray(left_points, dtype=np.float32).reshape(-1, 2)
    right_points = np.asarray(right_points, dtype=np.float32).reshape(-1, 2)
    if len(left_points) != len(right_points):
        raise ValueError("left/right epipolar points must have equal length")
    if len(left_points) < 8:
        return np.zeros(len(left_points), dtype=bool)
    cv2.setRNGSeed(0)
    fundamental, mask = cv2.findFundamentalMat(
        left_points,
        right_points,
        cv2.FM_RANSAC,
        threshold_px,
        0.999,
        10_000,
    )
    if fundamental is None or mask is None or fundamental.shape != (3, 3):
        return np.zeros(len(left_points), dtype=bool)
    return mask.reshape(-1).astype(bool)


def _restore_original_left_depth(depth_rectified: np.ndarray, lookup: tuple[np.ndarray, ...]) -> np.ndarray:
    map_x, map_y, depth_scale = lookup
    restored = cv2.remap(
        depth_rectified,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return restored * depth_scale


def _stereo_qa_status(qa: dict, cfg: ProjectConfig) -> tuple[bool, bool]:
    common = bool(
        qa["stereo_valid_ratio"] >= 0.95
        and qa["pair_delta_max_ms"] <= cfg.camera.maximum_stereo_delta_ms
        and qa["dense_depth_coverage_median"] >= cfg.depth.minimum_dense_coverage
    )
    deployable = bool(
        common
        and qa["epipolar_matches"] >= cfg.depth.minimum_epipolar_matches
        and qa["epipolar_vertical_p95_px"] <= cfg.depth.maximum_epipolar_p95_px
    )
    vertical_median = qa.get("epipolar_vertical_median_px")
    vertical_p90 = qa.get("epipolar_vertical_p90_px")
    disparity_median = qa.get("sgbm_lk_disparity_delta_median_px")
    ransac_inlier_ratio = qa.get("epipolar_ransac_inlier_ratio")
    debug_usable = bool(
        common
        and qa["epipolar_matches"] >= cfg.depth.debug_minimum_epipolar_matches
        and ransac_inlier_ratio is not None
        and ransac_inlier_ratio >= cfg.depth.debug_minimum_epipolar_ransac_inlier_ratio
        and vertical_median is not None
        and vertical_median <= cfg.depth.debug_maximum_epipolar_median_px
        and vertical_p90 is not None
        and vertical_p90 <= cfg.depth.debug_maximum_epipolar_p90_px
        and disparity_median is not None
        and disparity_median <= cfg.depth.debug_maximum_sgbm_lk_disparity_median_px
    )
    return deployable, debug_usable


def run_stereo_depth(
    cfg: ProjectConfig,
    source: str | Path,
    episode_dir: str | Path,
    *,
    allow_debug_qa: bool = False,
) -> StageManifest:
    if cfg.depth.method != "stereo_sgbm":
        raise ValueError(f"暂不支持 depth.method={cfg.depth.method!r}")
    source = Path(source).resolve()
    episode_dir = Path(episode_dir).resolve()
    output_dir = episode_dir / "depth" / "stereo_mm"
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage = []
    epipolar_left_points: list[list[float]] = []
    epipolar_right_points: list[list[float]] = []
    epipolar_sgbm_disparity: list[float] = []
    coverage_before_consistency = []
    consistency_ratio = []
    with PicoEpisode(source, cfg) as episode:
        geometry = stereo_geometry(episode.file)
        rectification = _rectification(episode.file, episode.image_size)
        matcher = _matcher(cfg)
        right_matcher = _matcher(cfg, right=True) if cfg.depth.left_right_consistency else None
        count = len(episode.camera_timestamps)
        sample_indices = set(np.linspace(0, count - 1, min(8, count), dtype=int).tolist())
        for index in range(count):
            left = episode.image(index)
            right = cv2.cvtColor(
                cv2.imdecode(
                    np.frombuffer(bytes(episode.file["camera/images_right_jpeg"][index]), dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                ),
                cv2.COLOR_BGR2RGB,
            )
            if cfg.depth.rectify_pinhole_stereo:
                left_rectified = cv2.remap(
                    left, *rectification["left_maps"], interpolation=cv2.INTER_LINEAR
                )
                right_rectified = cv2.remap(
                    right, *rectification["right_maps"], interpolation=cv2.INTER_LINEAR
                )
                depth_rectified, disparity = compute_stereo_depth(
                    left_rectified,
                    right_rectified,
                    rectification["K_rectified"],
                    rectification["baseline_rectified_m"],
                    cfg,
                    matcher,
                    right_matcher,
                )
                depth = _restore_original_left_depth(
                    depth_rectified, rectification["original_lookup"]
                )
            else:
                left_rectified, right_rectified = left, right
                depth, disparity = compute_stereo_depth(
                    left, right, episode.K, geometry["baseline_m"], cfg, matcher, right_matcher
                )
            depth_K = (
                rectification["K_rectified"] if cfg.depth.rectify_pinhole_stereo else episode.K
            )
            depth_baseline = (
                rectification["baseline_rectified_m"]
                if cfg.depth.rectify_pinhole_stereo
                else geometry["baseline_m"]
            )
            provisional_depth = np.zeros_like(disparity)
            provisional_valid = disparity > 0.5
            provisional_depth[provisional_valid] = (
                float(depth_K[0, 0]) * depth_baseline / disparity[provisional_valid]
            )
            provisional_valid &= provisional_depth >= cfg.depth.min_depth_m
            provisional_valid &= provisional_depth <= cfg.depth.max_depth_m
            coverage_before_consistency.append(float(provisional_valid.mean()))
            coverage.append(float(np.mean(depth > 0)))
            consistency_ratio.append(
                float(np.mean(depth > 0) / max(float(provisional_valid.mean()), 1e-12))
            )
            millimeters = np.clip(np.rint(depth * 1000.0), 0, np.iinfo(np.uint16).max).astype(np.uint16)
            if not cv2.imwrite(str(output_dir / f"{index:05d}.png"), millimeters):
                raise RuntimeError(f"深度图写入失败: {index}")
            if index in sample_indices:
                report = _epipolar_report(
                    cv2.cvtColor(left_rectified, cv2.COLOR_RGB2GRAY),
                    cv2.cvtColor(right_rectified, cv2.COLOR_RGB2GRAY),
                    disparity,
                    depth_rectified > 0 if cfg.depth.rectify_pinhole_stereo else depth > 0,
                )
                epipolar_left_points.extend(report["left_points"])
                epipolar_right_points.extend(report["right_points"])
                epipolar_sgbm_disparity.extend(report["sgbm_disparity_px"])

        pair_delta_ms = np.abs(episode.file["camera/stereo_pair_delta_ns"][:].astype(np.int64)) / 1e6
        left_points = np.asarray(epipolar_left_points, dtype=np.float32).reshape(-1, 2)
        right_points = np.asarray(epipolar_right_points, dtype=np.float32).reshape(-1, 2)
        sgbm_disparity = np.asarray(epipolar_sgbm_disparity, dtype=np.float32)
        raw_epipolar_matches = len(left_points)
        ransac_inliers = _fundamental_ransac_inliers(left_points, right_points)
        left_inliers = left_points[ransac_inliers]
        right_inliers = right_points[ransac_inliers]
        sgbm_inliers = sgbm_disparity[ransac_inliers]
        epipolar_matches = int(ransac_inliers.sum())
        epipolar_vertical = np.abs(left_inliers[:, 1] - right_inliers[:, 1])
        epipolar_disparity = left_inliers[:, 0] - right_inliers[:, 0]
        sgbm_disparity_error = np.abs(epipolar_disparity - sgbm_inliers)
        p95_vertical = (
            float(np.percentile(epipolar_vertical, 95)) if len(epipolar_vertical) else float("inf")
        )
        median_coverage = float(np.median(coverage))
        qa = {
            **{key: value for key, value in geometry.items() if key != "T_left_right"},
            "frames": count,
            "stereo_valid_ratio": float(episode.file["camera/stereo_valid"][:].mean()),
            "pair_delta_max_ms": float(pair_delta_ms.max()),
            "epipolar_filter": "pooled_8_frame_fundamental_ransac",
            "epipolar_ransac_threshold_px": 1.0,
            "epipolar_raw_matches": raw_epipolar_matches,
            "epipolar_matches": epipolar_matches,
            "epipolar_ransac_inlier_ratio": float(
                epipolar_matches / max(raw_epipolar_matches, 1)
            ),
            "epipolar_vertical_median_px": (
                float(np.median(epipolar_vertical)) if len(epipolar_vertical) else None
            ),
            "epipolar_vertical_p90_px": (
                float(np.percentile(epipolar_vertical, 90)) if len(epipolar_vertical) else None
            ),
            "epipolar_vertical_p95_px": p95_vertical,
            "epipolar_disparity_median_px": (
                float(np.median(epipolar_disparity)) if len(epipolar_disparity) else None
            ),
            "sgbm_lk_disparity_delta_median_px": (
                float(np.median(sgbm_disparity_error)) if len(sgbm_disparity_error) else None
            ),
            "sgbm_lk_disparity_delta_p90_px": (
                float(np.percentile(sgbm_disparity_error, 90))
                if len(sgbm_disparity_error)
                else None
            ),
            "sgbm_lk_disparity_delta_p95_px": (
                float(np.percentile(sgbm_disparity_error, 95))
                if len(sgbm_disparity_error)
                else None
            ),
            "sgbm_lk_large_tail_warning": bool(
                len(sgbm_disparity_error)
                and np.percentile(sgbm_disparity_error, 90)
                > cfg.depth.debug_maximum_sgbm_lk_disparity_p95_px
            ),
            "rectification_applied": bool(cfg.depth.rectify_pinhole_stereo),
            "rectification_input_model": "PICO XR_CAMERA_MODEL_PINHOLE_PICO",
            "depth_output_pixel_frame": "original_left_pinhole_image",
            "metric_depth_accuracy_verified": bool(cfg.camera.calibration_verified),
            "rectification_left_rotation_deg": rotation_angle_deg(rectification["R_left"]),
            "rectification_right_rotation_deg": rotation_angle_deg(rectification["R_right"]),
            "rectified_baseline_m": rectification["baseline_rectified_m"],
            "dense_depth_coverage_before_consistency_median": float(
                np.median(coverage_before_consistency)
            ),
            "left_right_consistency_ratio_median": float(np.median(consistency_ratio)),
            "dense_depth_coverage_median": median_coverage,
        }
        deployable, debug_usable = _stereo_qa_status(qa, cfg)
        qa["stereo_depth_deployable"] = deployable
        qa["stereo_depth_debug_usable"] = debug_usable
        qa["stereo_depth_acceptance"] = (
            "deployable" if deployable else "debug_only" if allow_debug_qa and debug_usable else "rejected"
        )
    qa_path = episode_dir / "qa" / "stereo_depth_report.json"
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    if not qa["stereo_depth_deployable"] and not (allow_debug_qa and qa["stereo_depth_debug_usable"]):
        raise RuntimeError(f"双目深度 QA 未通过: {qa}")
    if not qa["stereo_depth_deployable"]:
        print(
            "[warning] stereo depth passed debug-only QA; it may be used for visualization, "
            "but remains non-deployable",
            flush=True,
        )
    warnings = []
    if not qa["metric_depth_accuracy_verified"]:
        warnings.append(
            "PICO factory pinhole calibration passed stereo QA, but metric depth accuracy has not "
            "been verified against a physical calibration target."
        )
    if not qa["stereo_depth_deployable"]:
        warnings.append(
            "DEBUG-ONLY STEREO QA: strict epipolar deployment thresholds failed; outputs are retained "
            "only for diagnosis and visualization."
        )
    manifest = StageManifest(
        schema_version=cfg.schema_version,
        stage="stereo_depth",
        episode=source.stem,
        source_path=str(source),
        source_sha256=file_sha256(source),
        config=cfg.to_dict()["depth"],
        outputs={"depth_dir": str(output_dir), "qa": str(qa_path)},
        metrics=qa,
        warnings=tuple(warnings),
    )
    manifest.write(episode_dir / "stereo_depth.manifest.json")
    return manifest
