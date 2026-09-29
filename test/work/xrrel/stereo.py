"""双目深度 —— **直接调用** `ego_relation_policy` 的 `stereo_depth.py` 骨架。

这个文件里**没有一行深度的数学**。深度算法 100% 在他模块里:

    stereo_geometry(file)                    两眼 pose7 -> T_left_right / baseline / 相对旋转角
    _rectification(file, image_size)         cv2.stereoRectify(CALIB_ZERO_DISPARITY, alpha=0) +
                                             逆映射查找表 (把校正深度映回原左图像素)
    _matcher(cfg, right=False)               cv2.StereoSGBM, 参数全取他的 yaml
    compute_stereo_depth(...)                SGBM -> filterSpeckles -> /16 -> f*b/d -> 左右一致
    _restore_original_left_depth(...)        校正深度 -> 原左图像素
    _epipolar_report / _fundamental_ransac_inliers / _stereo_qa_status   QA

这里只负责他没做的那部分: **驱动**。他的 `run_stereo_depth` 是 HDF5 driver
(`PicoEpisode`、`camera/images_right_jpeg`、`camera/stereo_pair_delta_ns`), 我们没有那些
字段, 所以照他 349-424 行的循环体写一遍, 把取帧换成我们的 mp4 两半 —— 顺序、参数、
判据、输出格式 (uint16 毫米 PNG, 无效 0) 全部照旧。

参数一个不改 (包括 `num_disparities=128`)。`f*B = 686.868 * 0.064068 = 44.01`, 128 视差
覆盖 Z >= 0.344 m; 物体在手上约 0.5 m (视差 88 px) 够用。要用更近的量程得显式
`--num-disparities` 覆盖, 报告里会标成 DEVIATION。
"""

from __future__ import annotations

import json
import time

import cv2
import numpy as np

from . import EYE_H, EYE_W, RelPaths
from .adapter import (
    EYE_K,
    QA_NO_SOURCE_NOTE,
    QA_NO_SOURCE_VALUE,
    eye_pair_iter,
    focal_baseline_mm,
    load_ego_config,
    make_shim,
    nearest_covered_depth_m,
    stereo_source_hash,
)
from ego_relation.contracts.se3 import rotation_angle_deg
from ego_relation.s2_object_relations import stereo_depth as sd

SAMPLE_FRAMES = 8  # 与他一样: 均匀抽 8 帧算极线报告 (goodFeaturesToTrack + LK 不便宜)


def prepare(cfg, header) -> tuple[dict, dict]:
    """建成他 driver 里那两样: geometry + rectification (都是他的函数)。

    `image_size` 传**单眼**尺寸 (1080, 810): 他那边两眼是两张独立的图, 校正也按单眼做。
    """
    shim = make_shim(header)
    geometry = sd.stereo_geometry(shim)
    rectification = sd._rectification(shim, (EYE_W, EYE_H))
    return geometry, rectification


def _run_frames(paths: RelPaths, cfg, geometry, rectification, *, frames, force, verbose):
    """他 run_stereo_depth 的循环体, 换成我们的逐帧取帧。

    返回 (depth 写盘帧号列表, 他 QA 用的累加器)。累加器里多一个 `"reused"`: 本次**复用**
    已有 PNG、没有重算的帧号 —— 它们不进任何统计 (没有样本), 由 `run` 区分"测了"与"没测"。
    """
    matcher = sd._matcher(cfg)
    right_matcher = sd._matcher(cfg, right=True) if cfg.depth.left_right_consistency else None

    coverage: list[float] = []
    coverage_before: list[float] = []
    consistency: list[float] = []
    epi_left: list[list[float]] = []
    epi_right: list[list[float]] = []
    epi_disp: list[float] = []

    wanted = set(int(f) for f in frames)
    sample = set(np.linspace(min(wanted), max(wanted), SAMPLE_FRAMES, dtype=int).tolist())
    written: list[int] = []
    # 复用已有 PNG 的帧单独记一支: 它们**没有**进下面任何一个累加器 (没重算就没有样本),
    # 而 `run` 要能区分"这次测了 N 帧"和"这次一帧都没测"(见 run 里的 [skip] 分支)。
    reused: list[int] = []
    started = time.time()

    for index, (left, right) in enumerate(eye_pair_iter(paths.mp4)):
        if index not in wanted:
            continue
        out_path = paths.depth(index)
        if out_path.is_file() and not force:
            written.append(index)
            reused.append(index)
            continue

        # ---- 以下五步与 stereo_depth.py:363-410 逐步对应 ----
        if cfg.depth.rectify_pinhole_stereo:
            left_rectified = cv2.remap(
                left, *rectification["left_maps"], interpolation=cv2.INTER_LINEAR
            )
            right_rectified = cv2.remap(
                right, *rectification["right_maps"], interpolation=cv2.INTER_LINEAR
            )
            depth_rectified, disparity = sd.compute_stereo_depth(
                left_rectified,
                right_rectified,
                rectification["K_rectified"],
                rectification["baseline_rectified_m"],
                cfg,
                matcher,
                right_matcher,
            )
            depth = sd._restore_original_left_depth(depth_rectified, rectification["original_lookup"])
        else:
            left_rectified, right_rectified = left, right
            depth, disparity = sd.compute_stereo_depth(
                left, right, EYE_K, geometry["baseline_m"], cfg, matcher, right_matcher
            )
            depth_rectified = depth

        depth_K = (
            rectification["K_rectified"] if cfg.depth.rectify_pinhole_stereo else EYE_K
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
        coverage_before.append(float(provisional_valid.mean()))
        coverage.append(float(np.mean(depth > 0)))
        consistency.append(
            float(np.mean(depth > 0) / max(float(provisional_valid.mean()), 1e-12))
        )

        millimeters = np.clip(np.rint(depth * 1000.0), 0, np.iinfo(np.uint16).max).astype(
            np.uint16
        )
        if not cv2.imwrite(str(out_path), millimeters):
            raise RuntimeError(f"深度图写入失败: {out_path}")
        written.append(index)

        if index in sample:
            report = sd._epipolar_report(
                cv2.cvtColor(left_rectified, cv2.COLOR_RGB2GRAY),
                cv2.cvtColor(right_rectified, cv2.COLOR_RGB2GRAY),
                disparity,
                depth_rectified > 0 if cfg.depth.rectify_pinhole_stereo else depth > 0,
            )
            epi_left.extend(report["left_points"])
            epi_right.extend(report["right_points"])
            epi_disp.extend(report["sgbm_disparity_px"])

        if verbose and len(written) % 25 == 0:
            rate = len(written) / max(time.time() - started, 1e-9)
            print(f"      深度 {len(written)}/{len(wanted)} 帧  ({rate:.1f} 帧/s)", flush=True)

    return written, {
        "coverage": coverage,
        "coverage_before": coverage_before,
        "consistency": consistency,
        "epi_left": epi_left,
        "epi_right": epi_right,
        "epi_disp": epi_disp,
        "reused": reused,
    }


def build_qa(cfg, geometry, rectification, acc: dict, written: list[int]) -> dict:
    """照他 stereo_depth.py:417-491 汇总同一个 qa dict (键名逐字相同)。

    多出来的键 (`xrrel_*` / `qa_notes`) 前缀隔开, 不污染他的键名; 他那两栏没有数据源的
    量按 QA_NO_SOURCE_VALUE 填并记明 (见 adapter.QA_NO_SOURCE_NOTE)。
    """
    left_points = np.asarray(acc["epi_left"], dtype=np.float32).reshape(-1, 2)
    right_points = np.asarray(acc["epi_right"], dtype=np.float32).reshape(-1, 2)
    sgbm_disparity = np.asarray(acc["epi_disp"], dtype=np.float32)
    raw_matches = len(left_points)
    inliers = sd._fundamental_ransac_inliers(left_points, right_points)
    left_in = left_points[inliers]
    right_in = right_points[inliers]
    sgbm_in = sgbm_disparity[inliers]
    matches = int(inliers.sum())
    vertical = np.abs(left_in[:, 1] - right_in[:, 1]) if matches else np.zeros(0)
    disparity_in = left_in[:, 0] - right_in[:, 0] if matches else np.zeros(0)
    sgbm_error = np.abs(disparity_in - sgbm_in) if matches else np.zeros(0)
    p95_vertical = (
        float(np.percentile(vertical, 95)) if len(vertical) else float("inf")
    )
    coverage = acc["coverage"]

    def _pct(values, q):
        return float(np.percentile(values, q)) if len(values) else None

    qa = {
        **{key: value for key, value in geometry.items() if key != "T_left_right"},
        "frames": len(written),
        **QA_NO_SOURCE_VALUE,
        "epipolar_filter": "pooled_8_frame_fundamental_ransac",
        "epipolar_ransac_threshold_px": 1.0,
        "epipolar_raw_matches": raw_matches,
        "epipolar_matches": matches,
        "epipolar_ransac_inlier_ratio": float(matches / max(raw_matches, 1)),
        "epipolar_vertical_median_px": _pct(vertical, 50),
        "epipolar_vertical_p90_px": _pct(vertical, 90),
        "epipolar_vertical_p95_px": p95_vertical,
        "epipolar_disparity_median_px": _pct(disparity_in, 50),
        "sgbm_lk_disparity_delta_median_px": _pct(sgbm_error, 50),
        "sgbm_lk_disparity_delta_p90_px": _pct(sgbm_error, 90),
        "sgbm_lk_disparity_delta_p95_px": _pct(sgbm_error, 95),
        "sgbm_lk_large_tail_warning": bool(
            len(sgbm_error)
            and np.percentile(sgbm_error, 90)
            > cfg.depth.debug_maximum_sgbm_lk_disparity_p95_px
        ),
        "rectification_applied": bool(cfg.depth.rectify_pinhole_stereo),
        "rectification_input_model": "PICO XR_CAMERA_MODEL_PINHOLE_PICO",
        "depth_output_pixel_frame": "original_left_pinhole_image",
        "metric_depth_accuracy_verified": bool(cfg.camera.calibration_verified),
        "rectification_left_rotation_deg": rotation_angle_deg(rectification["R_left"]),
        "rectification_right_rotation_deg": rotation_angle_deg(rectification["R_right"]),
        "rectified_baseline_m": rectification["baseline_rectified_m"],
        # 这三条走 `_pct` (空 -> None) 而不是裸 np.median: 空累加器会出
        # `RuntimeWarning: Mean of empty slice` + 把 nan/inf 写进 JSON (那不是合法 JSON)。
        # 正常路径上 `run` 已经保证至少重算过一帧 (累加器非空), 这里是防手滑的护栏。
        "dense_depth_coverage_before_consistency_median": _pct(acc["coverage_before"], 50),
        "left_right_consistency_ratio_median": _pct(acc["consistency"], 50),
        "dense_depth_coverage_median": _pct(coverage, 50),
    }
    deployable, debug_usable = sd._stereo_qa_status(qa, cfg)
    qa["stereo_depth_deployable"] = deployable
    qa["stereo_depth_debug_usable"] = debug_usable
    qa["stereo_depth_acceptance"] = (
        "deployable" if deployable else ("debug_only" if debug_usable else "rejected")
    )
    return qa


def run(paths: RelPaths, header, *, frames, num_disparities=None, force=False, verbose=True,
        strict=False) -> dict:
    """跑深度并写 out/rel_<stem>/depth/*.png + stereo_qa.json。"""
    cfg, notes = load_ego_config(num_disparities)
    geometry, rectification = prepare(cfg, header)
    covered = nearest_covered_depth_m(cfg, geometry["baseline_m"])
    source_before = stereo_source_hash()

    print(
        f"    双目骨架: 他的 stereo_depth.py (sha256 {source_before['sha256'][:12]}…)  "
        f"参数取他的 configs/default.yaml"
    )
    print(
        f"    f*B = {focal_baseline_mm(cfg, geometry['baseline_m']):.2f} px*m  |  "
        f"num_disparities={cfg.depth.num_disparities} -> 覆盖 Z >= {covered:.3f} m  "
        f"(最近可测距离)"
    )
    print(
        f"    baseline {geometry['baseline_m'] * 1000:.3f} mm  "
        f"相对旋转 {geometry['relative_rotation_deg']:.6f} deg  "
        f"水平基线占比 {geometry['horizontal_baseline_ratio']:.6f}"
    )
    for note in notes:
        print(f"    [warn] {note}")

    written, acc = _run_frames(
        paths, cfg, geometry, rectification, frames=frames, force=force, verbose=verbose
    )
    if not written:
        raise RuntimeError("没有任何帧被算出深度 —— 检查 --frames / 起始帧")
    if len(acc["reused"]) == len(written):
        # 一帧都没重算 (全是复用已有的 depth/*.png): 覆盖率/极线统计**没有样本**。
        # 上一轮 `--stage all` 就是拿这组空累加器算出了 覆盖率 0 / 极线 0 / p95 inf, 判成
        # rejected (+[ALERT] +RuntimeWarning), 还把上一次真实测得的 stereo_qa.json **覆盖**了
        # —— 那不是一次测量, 不该出判定, 更不该改写记录。
        old_qa = None
        if paths.qa_json.is_file():
            try:
                old_qa = json.loads(paths.qa_json.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                old_qa = None
        print(f"    [skip] 本次 0 帧重算 ({len(written)} 帧复用已有 depth/*.png; 要重算加 --force)")
        if old_qa is None:
            print("           覆盖率/极线统计这次**没有样本**, 不做 QA 判定; 也没有历史 qa 可沿用")
        else:
            print(
                f"           覆盖率/极线统计这次**没有样本**, 不做 QA 判定; 沿用 "
                f"{paths.qa_json.name} 里已有的判定 acceptance="
                f"{old_qa.get('stereo_depth_acceptance')}"
                f" (覆盖率中位 {old_qa.get('dense_depth_coverage_median')})"
            )
            print(f"           -> {paths.depth_dir} ({len(written)} 帧, 本次未重算)")
        return {
            "written": written,
            "qa": old_qa,
            "geometry": geometry,
            "notes": notes,
            "recomputed": 0,
        }

    qa = build_qa(cfg, geometry, rectification, acc, written)
    qa["xrrel_frames"] = [int(min(written)), int(max(written))]
    qa["xrrel_num_disparities"] = int(cfg.depth.num_disparities)
    qa["xrrel_nearest_covered_depth_m"] = covered
    qa["xrrel_focal_baseline_px_m"] = focal_baseline_mm(cfg, geometry["baseline_m"])
    # 这份统计只覆盖**本次重算**的帧: 复用来的 PNG 没有样本。补跑的场合要说清楚是子集,
    # 否则看 qa_json 的人会以为这组数字是全片的。
    qa["xrrel_frames_computed"] = int(len(written) - len(acc["reused"]))
    qa["xrrel_frames_reused"] = int(len(acc["reused"]))
    qa["qa_notes"] = [QA_NO_SOURCE_NOTE, *notes]
    if acc["reused"]:
        qa["qa_notes"].append(
            f"本次只重算 {qa['xrrel_frames_computed']} 帧, 另 {qa['xrrel_frames_reused']} 帧复用"
            f"已有 depth/*.png —— 覆盖率/极线统计**只覆盖重算的那些帧**"
        )
    qa["xrrel_source"] = source_before
    # plan 验证 2: 他的文件跑前跑后同 hash —— 证明是只读调用而非改写
    source_after = stereo_source_hash()
    qa["xrrel_source_unchanged"] = source_after["sha256"] == source_before["sha256"]
    if not qa["xrrel_source_unchanged"]:
        raise RuntimeError(
            f"他的 stereo_depth.py 在本次运行中变了: {source_before['sha256']} -> "
            f"{source_after['sha256']} —— 我们只 import 调用, 不该改写它"
        )
    paths.qa_json.write_text(json.dumps(qa, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(
        f"    密集覆盖率中位 {qa['dense_depth_coverage_median']:.4f} "
        f"(门槛 {cfg.depth.minimum_dense_coverage})  "
        f"极线匹配 {qa['epipolar_matches']} (门槛 {cfg.depth.minimum_epipolar_matches})  "
        f"极线 p95 {qa['epipolar_vertical_p95_px']:.3f} px "
        f"(门槛 {cfg.depth.maximum_epipolar_p95_px})"
    )
    if not qa["stereo_depth_deployable"]:
        print(
            f"    [ALERT] 双目深度 QA 未过严格门槛 -> stereo_depth_acceptance="
            f"{qa['stereo_depth_acceptance']} (debug_usable={qa['stereo_depth_debug_usable']})。"
            f"这一轮**不是**可部署的深度, 只用于诊断与可视化。详见 {paths.qa_json}"
        )
        if strict:
            raise RuntimeError(f"双目深度 QA 未通过 (--strict-qa): {qa}")
    print(f"    -> {paths.depth_dir} ({len(written)} 帧)  +  {paths.qa_json}")
    return {"written": written, "qa": qa, "geometry": geometry, "notes": notes}
