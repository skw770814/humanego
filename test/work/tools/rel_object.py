#!/usr/bin/env python3
"""右手末端相对 SAM2 分割物体的位姿 (step3)。

链路 (每一步的出处写在对应模块的文件头):

    mp4 两半 -> 双目深度 (调 ego_relation_policy 的 stereo_depth.py)
             -> 物体点云 (mask + 深度反投影)
             -> 物体位姿 (帧50 PCA 定向 + 逐帧 ICP 鲁棒刚体拟合 + 门控/平滑/抓取锁存)
             -> 相对位姿 (inv(T_cam0_obj) @ T_cam0_midpoint)
             -> 可视化 (step1 5 关键点 + step2 分割 + 逐帧实时相对位姿)

分阶段跑 (产物都落盘, 可续; `--stage all` = stereo+lift+pose+relation+render):

    check      全部链条断言: 他的模块能 import / pose7 往返无损 / 双目几何 / 内参两倍
               陷阱 / 视频两半确有视差 / 视差量程 / filterSpeckles 就地性 /
               T_cam0_world 与 Projector 逐点一致 / step2 的无观测帧段。不下任何东西。
    stereo     SGBM 深度 -> out/rel_<stem>/depth/%05d.png (uint16 毫米, 无效 0, 他的格式)
               + stereo_qa.json (键名照他 run_stereo_depth 的 qa dict)
    lift       mask + 深度 -> 点云, 落 pose/clouds.npz
    pose       物体位姿, 落 pose/T_camera0_object.npz + pose/object_pose_meta.json
    relation   相对位姿, 落 relation/rel_<stem>.npz + report.json (终端打表)
    render     render/rel_<stem>.mp4 (2160x1156) 与静帧

**必须用 work/.venv 的解释器** (cv2 / h5py 只在里面): tools/rel_run.sh。

    ./tools/rel_run.sh 20260920_111342 --stage check
    ./tools/rel_run.sh 20260920_111342 --stage all
    ./tools/rel_run.sh 20260920_111342 --stage render --frames 50,120,260
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

WORK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORK))

import numpy as np  # noqa: E402

try:
    from xrrel import (  # noqa: E402
        EGO_CONFIG,
        EGO_REPO,
        EYE_CX,
        EYE_F,
        FULL_H,
        FULL_W,
        RelPaths,
        adapter,
        lift,
        objectpose,
        relation,
        render,
        stereo,
        step3,
    )
except ImportError as exc:  # cv2 / h5py / pyarrow 只在 .venv 里 —— 给人话而不是 traceback
    print(
        f"\n✗ import xrrel 失败: {exc}\n"
        f"  本链路要 cv2 / h5py / pyarrow / scipy, 它们只装在 {WORK / '.venv'}。\n"
        f"  用 {WORK / 'tools/rel_run.sh'} <stem> --stage ... 跑, 或 "
        f"{WORK / '.venv/bin/python'} {__file__} ...\n"
        f"  环境没装: cd {WORK} && .venv/bin/pip install h5py pyarrow\n",
        file=sys.stderr,
    )
    raise SystemExit(2)

STAGES = ("check", "stereo", "lift", "pose", "relation", "render", "step3", "all", "all_step3")


def fail(message: str) -> None:
    print(f"\n✗ {message}\n", file=sys.stderr)
    raise SystemExit(2)


def _has(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def require_venv() -> None:
    missing = [
        name for name in ("cv2", "h5py", "pyarrow", "scipy", "yaml", "PIL") if not _has(name)
    ]
    if missing:
        fail(
            f"当前解释器 {sys.executable} 缺 {'/'.join(missing)}。\n"
            f"  用 {WORK / '.venv/bin/python'} 跑 (或 ./tools/rel_run.sh)。\n"
            f"  环境没装: cd {WORK} && .venv/bin/pip install h5py pyarrow"
        )


def parse_frames(value: str | None) -> list[int] | None:
    if not value:
        return None
    out: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part[1:]:
            start, end = part.split("-", 1)
            out.extend(range(int(start), int(end) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def runs_of(frames) -> list[list[int]]:
    frames = sorted(int(f) for f in frames)
    if not frames:
        return []
    out = [[frames[0], frames[0]]]
    for frame in frames[1:]:
        if frame == out[-1][1] + 1:
            out[-1][1] = frame
        else:
            out.append([frame, frame])
    return out


def video_frames(paths: RelPaths) -> int:
    from xrhand.video import count_frames

    return int(count_frames(str(paths.mp4)))


def seg_empty_runs(paths: RelPaths) -> list[list[int]]:
    """step2 的**权威**无观测判据: `metrics.npz` 的 `areas == 0`。

    与 `report.json` 的 `stage_video.objects.obj1.empty_mask_runs` 口径不同 (后者是 seg
    阶段另一套定义), 两个都抄进报告并注明差异, 见 plan 风险 3。
    """
    if not paths.seg_metrics.is_file():
        return []
    with np.load(paths.seg_metrics) as archive:
        areas = archive["areas"]
    # 多物体时 areas 是 (帧, 物体): 任一物体有面积就算"这一帧有观测", 全零才算无观测
    areas = areas.any(axis=1) if areas.ndim > 1 else areas
    return runs_of(np.nonzero(areas == 0)[0])


# ------------------------------------------------------------------ check


def stage_check(paths: RelPaths, args) -> dict:
    """把所有链条断言掉, 不产出数据。"""
    print("[1/8] 他的模块能 import 且参数是他 yaml 里的")
    cfg, notes = adapter.load_ego_config(args.num_disparities)
    import cv2

    from ego_relation.s2_object_relations import stereo_depth as sd

    matcher = sd._matcher(cfg)
    matcher_right = sd._matcher(cfg, right=True)
    depth = cfg.depth
    print(f"    cv2 {cv2.__version__}  |  ego_relation_policy = {EGO_REPO}")
    print(f"    config = {EGO_CONFIG}")
    print(
        f"    depth: method={depth.method} num_disparities={depth.num_disparities} "
        f"block_size={depth.block_size} uniqueness={depth.uniqueness_ratio} "
        f"speckle={depth.speckle_window_size}/{depth.speckle_range} "
        f"range=[{depth.min_depth_m}, {depth.max_depth_m}] m "
        f"rectify={depth.rectify_pinhole_stereo} lr_consistency={depth.left_right_consistency}"
    )
    print(f"    matcher: {type(matcher).__name__} (right 也建得出来: {type(matcher_right).__name__})")
    for note in notes:
        print(f"    [warn] {note}")
    if int(depth.num_disparities) % 16 != 0:
        fail(f"num_disparities={depth.num_disparities} 不是 16 的倍数")
    perception = cfg.perception
    print(
        f"    perception 门槛: inlier>={perception.minimum_pose_inlier_ratio} "
        f"平移步长<={perception.maximum_object_translation_step_m} m "
        f"旋转步长<={perception.maximum_object_rotation_step_deg} deg "
        f"平滑 t={perception.object_translation_smoothing} r={perception.object_rotation_smoothing} "
        f"median_window={perception.object_translation_median_window}"
    )

    print("[2/8] 视频与 step2 产物")
    from xrhand.video import probe

    info = probe(str(paths.mp4))
    n_frames = video_frames(paths)
    print(f"    {paths.mp4.name}: {info.get('width')}x{info.get('height')} "
          f"{info.get('fps')} fps, {n_frames} 帧")
    if (info.get("width"), info.get("height")) != (FULL_W, FULL_H):
        fail(f"视频不是 {FULL_W}x{FULL_H} 左右并排: {info.get('width')}x{info.get('height')}")
    if not paths.seg_dir.is_dir():
        fail(f"缺 step2 产物目录 {paths.seg_dir} —— 先跑 tools/seg_run.sh --stage all")
    start = paths.seg_start_frame()
    empty_runs = seg_empty_runs(paths)
    report = json.loads(paths.seg_report.read_text(encoding="utf-8")) if paths.seg_report.is_file() else {}
    report_runs = (
        report.get("stage_video", {}).get("objects", {}).get("obj1", {}).get("empty_mask_runs")
    )
    print(f"    SAM2 起始帧 (report.json stage_video.initial_frame) = {start}")
    print(f"    无观测帧段 (metrics.npz areas==0, 权威判据): {empty_runs}")
    print(f"    report.json 的另一套口径 empty_mask_runs = {report_runs} (定义不同, 只记录不判)")
    print(f"    prompt = {paths.seg_category()!r}")
    if start is None:
        fail("拿不到 SAM2 起始帧 (report.json 与 metrics.npz 都没有)")
    halves = _check_eye_halves(paths, int(start))
    print(f"    两半确有视差 (帧 {halves['frame']}): 左右完全相同的像素占比 "
          f"{halves['identical_ratio'] * 100:.2f}% (≈0 才是两只眼, 不是复制)")
    print(f"      逐行最佳水平位移: 中位 {halves['median_shift_px']:.1f} px, "
          f"范围 {halves['min_shift_px']}~{halves['max_shift_px']} px (随行变化 = 真实视差)")
    if halves["identical_ratio"] > 0.9:
        fail(f"两半几乎逐像素相同 (占比 {halves['identical_ratio']:.3f}) —— "
             f"不是左右眼, 视差无从谈起")
    if halves["median_shift_px"] <= 2:
        fail(f"逐行最佳水平位移中位只有 {halves['median_shift_px']} px —— 不像水平双目")

    print("[3/8] 内参: 实际用的是标称 K, header 只做不变量核对 (camera.py:1-24 的反解)")
    header, _records = _load_tracking(paths)
    intr = adapter._assert_header_intrinsics(header)
    print(f"    header cameraIntrinsics [cx,cy,fx,fy] = "
          f"[{intr['header_cx']} {intr['header_cy']} {intr['header_fx']} {intr['header_fy']}]"
          f"  (全幅 {FULL_W} 坐标; 取内参时 width=双眼总宽, height=单眼高)")
    print(f"    (1) fy 就是单眼焦距:        {intr['header_fy']:.10f} - {EYE_F} = "
          f"{intr['fy_minus_nominal_f']:+.3e} px  (标称值是四舍五入到 3 位)")
    print(f"    (2) fx 被算成两倍 (≈2 不是 =2): fx/fy = {intr['fx_over_fy']:.8f}  ->  "
          f"fx/2 - f = {intr['half_fx_minus_nominal_f']:+.3e} px")
    print(f"    (3) cy = FULL_H/2-0.5:      {intr['header_cy']} (差 {intr['cy_minus_expected']:+.1e})")
    print(f"    (4) cx = FULL_W/2-0.5:      {intr['header_cx']} (差 {intr['cx_minus_expected']:+.1e})")
    print(f"    单眼主点**不是** cx/2: cx/2 = {intr['header_cx'] / 2.0}, 而 EYE_CX = EYE_W/2-0.5 = "
          f"{EYE_CX} (差 {intr['eye_cx_minus_half_cx']:+.2f} px)")
    print(f"    容差 {intr['tolerance_px']} px —— header 与标称值核对通过, 四项都对得上")
    print(f"    **喂给他的是标称 K** (与 step1 Projector / lift 反投影 / 画回去同一个像素系):")
    print(f"      K = {intr['fed_K']}")
    print(f"      用它而非 header 原值的代价: cx 差 {abs(intr['eye_cx_minus_half_cx']):.2f} px "
          f"-> 0.5 m 处 {abs(intr['eye_cx_minus_half_cx']) / EYE_F * 0.5 * 1000:.3f} mm; "
          f"f 差 {abs(intr['half_fx_minus_nominal_f']):.3f} px -> 0.5 m 处 "
          f"{abs(intr['half_fx_minus_nominal_f']) / EYE_F * 0.5 * 1000:.3f} mm (可忽略, 且"
          f" _restore_original_left_depth 两侧同一个 K, 自洽)")

    print("[4/8] pose7 往返无损 + 双目几何 (调他的 stereo_geometry / _rectification)")
    for eye in (0, 1):
        error = adapter.pose7_roundtrip_error(header.extrinsics[eye])
        print(f"    eye{eye}: 4x4 -> pose7 -> matrix_from_pose7 最大误差 {error:.3e}")
        if error > 1e-12:
            fail(f"eye{eye} 的 pose7 往返误差 {error:.3e} > 1e-12 —— 外参喂给他会变")
    geometry, rectification = stereo.prepare(cfg, header)
    print(f"    baseline {geometry['baseline_m'] * 1000:.4f} mm  "
          f"相对旋转 {geometry['relative_rotation_deg']:.6f} deg  "
          f"水平基线占比 {geometry['horizontal_baseline_ratio']:.6f}")
    if geometry["relative_rotation_deg"] > 0.5:
        fail(f"两眼相对旋转 {geometry['relative_rotation_deg']:.4f} deg —— 不是平行双目假设")
    if abs(geometry["baseline_m"] - 0.064068) > 1e-4:
        print(f"    [warn] baseline 与实测 64.068 mm 差 "
              f"{(geometry['baseline_m'] - 0.064068) * 1000:+.4f} mm")
    rotation_left = np.rad2deg(np.arccos(np.clip((np.trace(rectification["R_left"]) - 1) / 2, -1, 1)))
    print(f"    校正: R_left {rotation_left:.6f} deg  R_right "
          f"{np.rad2deg(np.arccos(np.clip((np.trace(rectification['R_right']) - 1) / 2, -1, 1))):.6f} deg  "
          f"rectified baseline {rectification['baseline_rectified_m'] * 1000:.4f} mm")
    print(f"    K_rectified = {np.asarray(rectification['K_rectified']).round(4).tolist()}")

    print("[5/8] 视差量程 (f*B)")
    focal_baseline = adapter.focal_baseline_mm(cfg, geometry["baseline_m"])
    covered = adapter.nearest_covered_depth_m(cfg, geometry["baseline_m"])
    print(f"    f*B = {EYE_F} x {geometry['baseline_m']:.6f} = {focal_baseline:.4f} px*m  ->  Z = {focal_baseline:.2f} / d")
    print(f"    num_disparities={depth.num_disparities} -> 覆盖 Z >= {covered:.4f} m "
          f"(更近的会视差饱和 -> 深度 0/无效)")
    for distance in (0.35, 0.5, 1.0, 2.0):
        print(f"      Z={distance:4.2f} m -> 视差 {focal_baseline / distance:6.2f} px"
              + ("  [超出量程, 无效]" if focal_baseline / distance > depth.num_disparities else ""))

    print("[6/8] filterSpeckles 的 dtype 与就地性 (cv2 5.0 行为; 他 compute_stereo_depth 里丢掉了返回值)")
    _check_filter_speckles(paths, cfg, int(start))

    print("[7/8] T_cam0_world 与 Projector 逐点一致 (防第 2 轮那个坐标系坑)")
    _, _, reel = relation.load_reel(paths.stem, calib=args.calib, lag=args.lag)
    check = relation.self_check_camera0(reel)
    print(f"    世界点经 T_cam0_world + cam_to_pixel vs Projector.project: "
          f"最大像素误差 {check['max_pixel_error']:.3e}, 深度误差 {check['max_depth_error']:.3e}")
    print(f"    det(R)={check['det_rotation']:.15f}  |RᵀR-I|max={check['max_orthogonality_error']:.3e}")
    print(f"    extra_t (必须照用) = {check['extra_t_applied_m']}  "
          f"flip_head_quat={check['flip_head_quat']} "
          f"eye0_is_left_half={check['eye0_is_left_half']}")
    print(f"    Reel: {reel.n_frames} 帧, lag={reel.lag} ({reel.lag_note()})")
    gripper = relation.hand_states(reel, reel.n_frames)
    print(f"    手的中点系可用 {int(gripper['valid'].sum())}/{reel.n_frames} 帧, "
          f"二值爪闭合 {int(gripper['closed'].sum())} 帧 "
          f"(标定: q95={gripper['grasp_calibration']['open_baseline_m']:.4f} m, "
          f"q10={gripper['grasp_calibration']['closed_reference_m']:.4f} m)")

    print("[8/8] 深度数值闭环 (在参考帧上真跑一遍他的 compute_stereo_depth, 不落盘)")
    closed_loop = _smoke_depth(paths, cfg, rectification, start, focal_baseline)
    if closed_loop is None:
        print(f"    [warn] 帧 {start} 的 mask 区内没有有效深度 —— 看 stereo 阶段的 QA 与风险 2 "
              f"(暗色低纹理 + SGBM, 覆盖率可能很低)")
    else:
        print(f"    帧 {start}: mask {closed_loop['mask_px']} px, 其中有效深度 "
              f"{closed_loop['valid_px']} px ({closed_loop['valid_ratio'] * 100:.1f}%), "
              f"中位深度 {closed_loop['median_depth_m']:.4f} m")
        print(f"    闭环: 中位深度 x 中位视差 = {closed_loop['median_depth_m']:.4f} x "
              f"{closed_loop['median_disparity_px']:.2f} = {closed_loop['product']:.3f} px*m "
              f"(应 ≈ f*B = {focal_baseline:.3f}, 差 "
              f"{abs(closed_loop['product'] - focal_baseline) / focal_baseline * 100:.2f}%)")
        if not 0.15 <= closed_loop["median_depth_m"] <= 1.5:
            print(f"    [warn] 中位深度 {closed_loop['median_depth_m']:.3f} m 不在 0.15~1.5 m"
                  f" —— 物体在手上应该在这个范围")

    print(f"\n✓ check 全绿 —— 他的 stereo_depth.py sha256 = "
          f"{adapter.stereo_source_hash()['sha256'][:16]}… (只读调用, 未改写)")
    return {
        "geometry": geometry,
        "rectified_baseline_m": rectification["baseline_rectified_m"],
        "empty_runs": empty_runs,
        "seg_start_frame": start,
        "focal_baseline_px_m": focal_baseline,
        "nearest_covered_depth_m": covered,
        "camera0_self_check": check,
    }


def _load_tracking(paths: RelPaths):
    from xrhand.io_tracking import load

    return load(str(paths.tracking))


def _check_eye_halves(paths: RelPaths, frame: int, rows: int = 16, max_shift: int = 160) -> dict:
    """确认左右两半是**两只眼**而不是同一只眼的复制 (plan check 第 5 条)。

    判据两个: (a) 左右完全相同像素的占比; (b) 抽若干行在 0..max_shift 里找使左右灰度
    平均绝对差最小的水平位移 —— 真实视差会随行变化, 复制品则恒为 0。
    """
    import cv2

    left, right = adapter.read_eye_pair(paths.mp4, frame)
    identical = float(np.mean(left == right))
    left_gray = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY).astype(np.float32)
    right_gray = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sample_rows = np.linspace(0, left_gray.shape[0] - 1, rows, dtype=int)
    shifts: list[int] = []
    for row in sample_rows:
        reference = left_gray[row, max_shift:]
        best, best_shift = None, 0
        for shift in range(max_shift + 1):
            candidate = right_gray[row, max_shift - shift : right_gray.shape[1] - shift]
            score = float(np.mean(np.abs(reference - candidate)))
            if best is None or score < best:
                best, best_shift = score, shift
        shifts.append(best_shift)
    shifts_array = np.asarray(shifts)
    return {
        "frame": int(frame),
        "identical_ratio": identical,
        "rows": [int(r) for r in sample_rows],
        "shifts_px": shifts_array.tolist(),
        "median_shift_px": float(np.median(shifts_array)),
        "min_shift_px": int(shifts_array.min()),
        "max_shift_px": int(shifts_array.max()),
    }


def _check_filter_speckles(paths: RelPaths, cfg, frame: int) -> None:
    """cv2 5.0 的 `filterSpeckles` 收什么 dtype、是不是就地改 —— 他的代码靠「int16 + 就地」。

    他 `compute_stereo_depth` 里是 `disparity_raw = matcher.compute(...)` 然后
    `cv2.filterSpeckles(disparity_raw, 0, speckle_window_size, speckle_range*16)` **丢掉返回值**,
    最后才 `astype(float32) / 16.0`。所以这里两件事都要**证**出来, 不能口头断言:

      (1) 拿真帧对**真的 matcher** 跑一次, 看它返回的到底是不是 int16;
      (2) 对他的那个 int16 数组就地跑一遍同一个 filterSpeckles, 看散斑是否真被清掉。

    上一轮这个探针是拿 `float32` 造的, 被本版 cv2 直接拒 (它只收 CV_8UC1 / CV_16SC1),
    于是 `check` 挂在一个**跟他的代码无关**的断言上 —— 探针本身错, 不是他的路径错。

    仍然只报警、不失败: 真不就地的话, 他那一步散斑滤波在本环境里等于空转, 后果是深度图上留
    孤立的椒盐视差。我们的兜底是 mask 腐蚀 + 点云抽稀 + 鲁棒刚体拟合剔外点, 但**要让你知道**。
    """
    import cv2

    from ego_relation.s2_object_relations import stereo_depth as sd

    max_speckle = int(cfg.depth.speckle_window_size)
    max_diff = int(cfg.depth.speckle_range * 16)  # 定点视差域 (x16), 与他一致

    # (1) 真帧 + 真 matcher: 拿到他那一步真正处理的数组
    left, right = adapter.read_eye_pair(paths.mp4, frame)
    matcher = sd._matcher(cfg)
    raw = matcher.compute(
        cv2.cvtColor(left, cv2.COLOR_RGB2GRAY), cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
    )
    print(f"    真帧 {frame}: matcher.compute() -> dtype {raw.dtype}, shape {tuple(raw.shape)} "
          f"(他的 compute_stereo_depth 直接把这个数组交给 filterSpeckles)")
    if raw.dtype != np.int16:
        print(f"    [warn] 它返回的是 {raw.dtype} 而不是 int16 —— 他那一步在本环境可能不兼容, "
              f"去核对 stereo_depth.py:154-160")

    # (2) 就地性: 用他的参数在他自己的数组上跑
    before = raw.copy()
    returned = cv2.filterSpeckles(raw, 0, max_speckle, max_diff)
    in_place = not np.array_equal(raw, before)
    changed = int(np.count_nonzero(raw != before))
    print(f"    就地生效={in_place} (改动 {changed} px; 与他同参 maxSpeckleSize={max_speckle} "
          f"maxDiff={max_diff}; 返回值 {type(returned).__name__} —— 他丢掉了它)")
    if not in_place:
        print(
            "    [warn] cv2.filterSpeckles 在本版本不就地生效, 而他的 compute_stereo_depth "
            "丢掉返回值 -> 他那一步散斑滤波等于空转。\n"
            "           他的文件只读, 我们不替他打补丁; 靠 mask 腐蚀 + 点云抽稀 + 鲁棒拟合"
            "剔外点兜。深度图上的孤立椒盐视差会比他在自己环境里跑出来的多。"
        )

    # (3) 定点探针: 明确「该清的单像素散斑被清 / 该留的有效块保留」
    probe = np.zeros((32, 32), dtype=np.int16)
    probe[10:20, 10:20] = 64 * 16  # 100 px 的"真实"视差 (> maxSpeckleSize=80, 该留)
    probe[5, 5] = 20 * 16  # 单个散斑 (1 px, 该清成 0)
    cv2.filterSpeckles(probe, 0, 80, max_diff)
    cleared = bool(probe[5, 5] == 0)
    kept = bool(probe[15, 15] == 64 * 16)
    print(f"    int16 定点探针 (视差 x16 定标): 单像素散斑被清={cleared}  20x20 有效块保留={kept}")
    if not cleared:
        print("    [warn] 散斑没被清掉, 参数含义可能与预期不同 (maxSpeckleSize/maxDiff 的顺序)")
    try:
        cv2.filterSpeckles(np.zeros((4, 4), dtype=np.float32), 0, 80, max_diff)
        print("    (本版 cv2 也收 float32, 不必再绕)")
    except cv2.error:
        print("    (本版 cv2 **拒绝** float32, 只收 CV_8UC1/CV_16SC1 —— 上一轮我的探针就栽在这; "
              "他的路径是 int16, 不受影响)")



def _smoke_depth(paths, cfg, rectification, frame, focal_baseline):
    """在参考帧上跑一遍深度, 校验 `中位深度 x 中位视差 ≈ f*B` 的闭环。不写任何文件。

    全程用他的函数: `_matcher` -> `compute_stereo_depth` -> `_restore_original_left_depth`
    (与 `xrrel/stereo.py::_run_frames` 同一串调用, 只是不落盘)。
    """
    import cv2

    from ego_relation.s2_object_relations import stereo_depth as sd

    mask = lift.load_mask(paths, frame)
    if mask is None or not (mask > 127).any():
        print(f"    [warn] 帧 {frame} 没有 mask, 跳过深度闭环")
        return None

    matcher = sd._matcher(cfg)
    right_matcher = sd._matcher(cfg, right=True) if cfg.depth.left_right_consistency else None
    for index, (left, right) in enumerate(adapter.eye_pair_iter(paths.mp4)):
        if index != frame:
            continue
        left_rectified = cv2.remap(left, *rectification["left_maps"],
                                   interpolation=cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right, *rectification["right_maps"],
                                    interpolation=cv2.INTER_LINEAR)
        depth_rectified, disparity = sd.compute_stereo_depth(
            left_rectified, right_rectified, rectification["K_rectified"],
            rectification["baseline_rectified_m"], cfg, matcher, right_matcher,
        )
        depth = sd._restore_original_left_depth(depth_rectified, rectification["original_lookup"])
        break
    else:
        fail(f"mp4 里没有第 {frame} 帧")

    # mask 与 depth 都在**原左眼像素**系 (他 _restore_original_left_depth 保证的)。
    # disparity 名义上在**校正后**的网格上, 但本采集实测 R_left = R_right = 0.000000 deg、
    # K_rectified 与 EYE_K 逐位相同 —— 校正就是恒等映射, 两格是同一个格, 所以直接按同一批
    # 像素取中位。若换了别的采集 (两眼不平行) 这一步会偏, 故 check 第 4 步打印校正旋转角。
    #
    # 关于 乘积 ≈ f*B: `Z = f*B/d` 是严格单调的, 所以 median(Z) = f*B/median(d) 是恒等式 ——
    # 这一项**不**独立验证测距精度, 它验证的是「深度与视差是同一套 K、同一条基线、同一个网格
    # 算出来的」。真正说明问题的是 valid_ratio 与 median_depth 本身。
    support = mask > 127
    values = depth[support]
    disparities = disparity[support]
    usable = (values > 0) & (disparities > 0)
    if not usable.any():
        return None
    median_depth = float(np.median(values[usable]))
    median_disparity = float(np.median(disparities[usable]))
    return {
        "frame": int(frame),
        "mask_px": int(support.sum()),
        "valid_px": int(usable.sum()),
        "valid_ratio": float(usable.sum()) / float(max(int(support.sum()), 1)),
        "median_depth_m": median_depth,
        "median_disparity_px": median_disparity,
        "product": median_depth * median_disparity,
        "focal_baseline_px_m": float(focal_baseline),
    }


# ------------------------------------------------------------------ 各 stage


def stage_stereo(paths: RelPaths, args) -> dict:
    header, _ = _load_tracking(paths)
    start = args.start_frame if args.start_frame is not None else paths.seg_start_frame()
    if start is None:
        fail("深度起始帧未定: report.json 里没有 stage_video.initial_frame, "
             "也没给 --start-frame")
    start = int(start)
    n_frames = video_frames(paths)
    end = n_frames - 1 if args.max_frames is None else min(n_frames - 1, start + args.max_frames - 1)
    frames = list(range(start, end + 1))
    print(f"    深度只算帧 {start}..{end} (SAM2 起始帧之前是空 mask, 不需要深度; "
          f"整片 {n_frames} 帧)")
    return stereo.run(
        paths, header, frames=frames, num_disparities=args.num_disparities,
        force=args.force, verbose=True, strict=args.strict_qa,
    )


def _require(path: Path, hint: str) -> None:
    """前置产物不存在就说人话, 别让 np.load 抛 FileNotFoundError。"""
    if not path.exists():
        fail(
            f"缺前置产物 {path}\n"
            f"  {hint}\n"
            f"  (分阶段跑是为了能续跑; 要一次跑完用 --stage all)"
        )


def _require_depth(paths: RelPaths) -> int:
    """深度 PNG 至少要有几帧才算 stereo 跑过 (起始帧之前的那些不需要深度)。"""
    _require(paths.depth_dir, f"先跑 --stage stereo (产物落到 {paths.depth_dir})")
    written = sorted(int(p.stem) for p in paths.depth_dir.glob("*.png"))
    if not written:
        fail(f"{paths.depth_dir} 是空的 —— 先跑 --stage stereo")
    return len(written)


def stage_lift(paths: RelPaths, args) -> dict:
    n_depth = _require_depth(paths)
    print(f"    深度已有 {n_depth} 帧 ({paths.depth_dir})")
    n_frames = video_frames(paths)
    frames = list(range(n_frames))
    if args.max_frames is not None:
        frames = frames[: args.max_frames]
    # 单物体的老入口: 不给 instance_ids -> `build_clouds` 只做 obj1, 产物与以前同一份
    # (`clouds_npz` 就是 `clouds_npz_for('obj1')`)。多物体走 pipeline (xrpipe/step2.py)。
    clouds = lift.build_clouds(
        paths, frames, erode_px=args.erode_px, stride=args.cloud_stride, verbose=True
    )
    print(f"    -> {paths.clouds_npz}")
    return clouds


def _hands(paths: RelPaths, args, n_frames: int):
    _, _, reel = relation.load_reel(paths.stem, calib=args.calib, lag=args.lag)
    return reel, relation.hand_states(reel, n_frames)


def stage_pose(paths: RelPaths, args) -> objectpose.PoseResult:
    _require(paths.clouds_npz, "先跑 --stage lift (从 mask + 深度反投影出点云)")
    clouds = lift.load_clouds(paths)
    n_frames = len(clouds["frame_index"])
    _, hands = _hands(paths, args, n_frames)
    reference_frame = (
        int(args.reference_frame) if args.reference_frame is not None else paths.seg_start_frame()
    )
    if reference_frame is None:
        fail("拿不到参考帧 (report.json 没有 stage_video.initial_frame, 也没给 --reference-frame)")
    print(f"    参考帧 = {reference_frame} "
          f"({'命令行指定' if args.reference_frame is not None else 'SAM2 起始帧 report.json'}); "
          f"不是帧 0")
    result = objectpose.estimate(
        paths,
        clouds,
        hands["T_camera0_midpoint"],
        hands["valid"],
        hands["closed"],
        reference_frame=reference_frame,
        reference_pos=None,
        verbose=True,
    )
    objectpose.save(paths, result)
    print(f"    -> {paths.object_npz}  +  {paths.pose_dir / 'object_pose_meta.json'}")
    _assert_reference(paths, result)
    return result


def _assert_reference(paths: RelPaths, result) -> None:
    expected = paths.seg_start_frame()
    if expected is None:
        return
    if result.reference_info["used_frame"] == int(expected):
        print(f"    ✓ 参考帧 == step2 的初始帧 {expected}")
        return
    if result.reference_pos == 0:
        fail(f"参考帧用了帧 {result.reference_info['used_frame']} (pos 0) 而不是 {expected}")
    print(f"    [warn] 参考帧退到了 {result.reference_info['used_frame']} (初始帧 {expected} 无观测)")


def stage_relation(paths: RelPaths, args):
    _require(paths.object_npz, "先跑 --stage pose (帧50 PCA + 逐帧 ICP 的物体位姿)")
    clouds = lift.load_clouds(paths)
    n_frames = len(clouds["frame_index"])
    _, hands = _hands(paths, args, n_frames)
    frames = parse_frames(args.frames)
    return relation.run(
        paths, hands, verbose=True, frame_print=not args.no_frame_print, frames=frames
    )


def stage_render(paths: RelPaths, args) -> dict:
    _require(
        paths.relation_npz,
        "先跑 --stage relation (相对位姿 npz 是这一层要画的数, 没有它画不出面板)",
    )
    _, _, reel = relation.load_reel(paths.stem, calib=args.calib, lag=args.lag)
    return render.run(
        paths,
        reel=reel,
        frames=parse_frames(args.frames),
        radius=args.radius,
        frame_print=not args.no_frame_print,
        verbose=True,
    )


def stage_step3(paths: RelPaths, args) -> dict:
    _require(paths.video, "先跑 --stage render (Step3 读取 render/rel_<stem>.mp4)")
    _, _, reel = relation.load_reel(paths.stem, calib=args.calib, lag=args.lag)
    return step3.run(
        paths, reel=reel, frames=parse_frames(args.frames), verbose=True,
        allow_opencv_fallback=args.allow_opencv_fallback,
        lama_model=args.lama_model,
        piper_model=args.piper_model,
        piper_assets=args.piper_assets,
        piper_tcp_calibration=args.piper_tcp_calibration,
        gripper_render=args.gripper_render,
        piper_open_joint=args.piper_open_joint,
        piper_closed_joint=args.piper_closed_joint,
    )


# ------------------------------------------------------------------ main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="右手末端 (两指尖中点系) 相对 SAM2 分割物体的位姿",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("stem", nargs="*", help="采集 stem (默认 20260920_111342, 它有 masks)")
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--frames", default=None,
                        help="逗号分隔帧号 (支持 a-b 区间), 用于出静帧 / 限制终端逐帧打印")
    parser.add_argument("--reference-frame", type=int, default=None,
                        help="物体参考帧 (默认取 report.json 的 stage_video.initial_frame = 50)")
    parser.add_argument("--start-frame", type=int, default=None,
                        help="深度从哪帧开始算 (默认 = 参考帧)")
    parser.add_argument("--num-disparities", type=int, default=None,
                        help="覆盖他 yaml 的 num_disparities=128 (逃生口, 报告里会标 DEVIATION)")
    parser.add_argument("--cloud-stride", type=int, default=lift.DEFAULT_STRIDE,
                        help="点云抽稀步长 (像素)")
    parser.add_argument("--erode-px", type=int, default=lift.DEFAULT_ERODE_PX,
                        help="mask 腐蚀半径, 去掉物体边缘的脏视差")
    parser.add_argument("--strict-qa", action="store_true",
                        help="双目 QA 未过严格门槛时直接失败 (默认只报警)")
    parser.add_argument("--no-frame-print", action="store_true",
                        help="关掉逐帧终端打印相对位姿 (画面上仍逐帧刷新)")
    parser.add_argument("--calib", default=None, help="calib.json (默认 out/calib.json)")
    parser.add_argument("--lag", type=int, default=None, help="覆盖 align_*.json 的 lag")
    parser.add_argument("--radius", type=int, default=5, help="5 个关键点的半径")
    parser.add_argument("--max-frames", type=int, default=None, help="只处理前 N 帧 (冒烟用)")
    parser.add_argument("--outdir", default=None, help="产物根目录 (默认 work/out)")
    parser.add_argument(
        "--lama-model", default=os.environ.get("LAMA_MODEL_PATH"),
        help="本地 LaMa ONNX 权重路径 (也可通过 LAMA_MODEL_PATH 设置)",
    )
    parser.add_argument("--piper-model", default=None, help="Piper gripper URDF path")
    parser.add_argument("--piper-assets", default=None, help="Piper asset root containing meshes/")
    parser.add_argument("--piper-tcp-calibration", default=None, help="T_tcp_model JSON path")
    parser.add_argument("--gripper-render", choices=("piper", "wireframe"), default="piper")
    parser.add_argument("--piper-open-joint", type=float, default=None)
    parser.add_argument("--piper-closed-joint", type=float, default=None)
    parser.add_argument("--force", action="store_true", help="重算已有产物 (深度 PNG)")
    parser.add_argument(
        "--allow-opencv-fallback", action="store_true",
        help="仅调试用: DINO/SAM2 或 LaMa 不可用时允许使用旧的关键点+OpenCV 修复",
    )
    return parser


def run_one(stem: str, args) -> int:
    paths = (
        RelPaths.for_stem(stem, args.outdir) if args.outdir else RelPaths.for_stem(stem)
    )
    paths.ensure()
    print(f"\n{'=' * 78}\n=== {stem}  ->  {paths.outdir}\n{'=' * 78}")
    require_venv()

    if paths.stem.endswith("111300") and not paths.seg_metrics.is_file():
        print(
            "    [warn] out/seg_20260920_111300/ 里没有 masks/metrics —— step3 现在只能跑 "
            "111342。111300 要先补 tools/seg_run.sh --stage image,video"
        )
    started = time.time()
    if args.stage == "check":
        stage_check(paths, args)
        return 0

    if not paths.seg_dir.is_dir():
        fail(f"缺 step2 产物目录 {paths.seg_dir} —— 先跑 tools/seg_run.sh <stem> --stage all")
    if paths.seg_start_frame() is None:
        fail("拿不到 SAM2 起始帧: report.json 的 stage_video.initial_frame 与 metrics.npz 都读不到")

    if args.stage in ("stereo", "all", "all_step3"):
        print("[stereo] 双目深度 (调他的 stereo_depth.py)")
        stage_stereo(paths, args)
    if args.stage in ("lift", "all", "all_step3"):
        print("[lift] mask + 深度 -> 点云")
        stage_lift(paths, args)
    if args.stage in ("pose", "all", "all_step3"):
        print("[pose] 物体位姿 (帧50 PCA + 逐帧 ICP + 门控/平滑/抓取锁存; 握持期间照 HumanEgo-main 一律手推)")
        stage_pose(paths, args)
    if args.stage in ("relation", "all", "all_step3"):
        print("[relation] 相对位姿")
        stage_relation(paths, args)
    if args.stage in ("render", "all", "all_step3"):
        print("[render] 可视化: step1 5 关键点 + step2 分割 + 逐帧实时相对位姿")
        stage_render(paths, args)
    if args.stage in ("step3", "all_step3"):
        print("[step3] 手臂修复 + HumanEgo 参数化虚拟夹爪")
        stage_step3(paths, args)
    print(f"\n✓ {stem} 完成 ({time.time() - started:.1f} s)")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    stems = args.stem or ["20260920_111342"]
    for stem in stems:
        try:
            run_one(stem, args)
        except KeyboardInterrupt:
            raise
        except BaseException as exc:
            if len(stems) == 1:
                raise
            traceback.print_exc()
            print(f"✗ {stem} 失败 ({type(exc).__name__}: {exc}), 继续下一条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
