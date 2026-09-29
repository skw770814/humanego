"""Step3 hand/arm removal and virtual gripper compositing.

The module deliberately does not touch the existing relation renderer.  It consumes
that renderer's video and writes a sibling ``step3`` directory.  Geometry follows
HumanEgo's five-point midpoint TCP convention; the visible gripper is the same
parameterised two-finger wireframe used by HumanEgo-main/VisualKpts.py.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from xrhand.video import VideoWriter, iter_frames, probe
from . import EYE_H, EYE_W, FULL_H, FULL_W, RelPaths


class _AllFramesEmptyError(RuntimeError):
    """All DINO frames completed but none contained a candidate box."""


# 与加这个形参之前逐字一致 (手臂 + 手一起 mask)。
DEFAULT_MASK_PROMPT = "human arms . human hands ."
# 只要手 (pipeline 的默认; 手臂是否一起处理由调用方决定)。
HANDS_ONLY_MASK_PROMPT = "human hands ."


def _cv2():
    import cv2
    return cv2


def _empty_detection_runs(records):
    """Return contiguous [start, end] runs of frames with no DINO boxes."""
    runs = []
    start = None
    for i, record in enumerate(records):
        empty = not bool(record.get("detected", False))
        if empty and start is None:
            start = i
        elif not empty and start is not None:
            runs.append([start, i - 1])
            start = None
    if start is not None:
        runs.append([start, len(records) - 1])
    return runs


def _project_points(reel, points, frame, half=0):
    rec = reel.records[reel.ridx[frame]]
    u, v, z, _ = reel.proj.project(np.asarray(points), rec.head_pos, rec.head_quat, half)
    return np.asarray(u), np.asarray(v), np.asarray(z)


def _hand_mask(reel, frame: int, dilation: int = 18) -> np.ndarray:
    """Build a conservative right-hand/forearm mask from the tracked 26 points."""
    cv2 = _cv2()
    rec = reel.records[reel.ridx[frame]]
    pos = np.asarray(rec.right.pos, dtype=np.float64)
    valid = np.asarray(rec.right.valid_mask, dtype=bool)
    u, v, z = _project_points(reel, pos, frame)
    good = valid & (z > 0) & (u >= 0) & (u < EYE_W) & (v >= 0) & (v < EYE_H)
    mask = np.zeros((EYE_H, EYE_W), np.uint8)
    pts = np.column_stack([u[good], v[good]]).astype(np.int32)
    if len(pts) >= 3:
        hull = cv2.convexHull(pts.reshape(-1, 1, 2))
        cv2.fillConvexPoly(mask, hull, 255)
    # Skeleton strokes make the mask robust when the palm is edge-on.
    edges = ((1, 2), (2, 3), (3, 4), (4, 5), (1, 6), (6, 7), (7, 8),
             (8, 9), (9, 10), (1, 6), (1, 11), (11, 12), (12, 13),
             (13, 14), (14, 15), (15, 16), (1, 17), (17, 18), (18, 19),
             (19, 20), (20, 21), (1, 22), (22, 23), (23, 24), (24, 25))
    for a, b in edges:
        if a < len(good) and b < len(good) and good[a] and good[b]:
            cv2.line(mask, (int(u[a]), int(v[a])), (int(u[b]), int(v[b])), 255, 12)
    # Extend from wrist away from the palm to cover the visible forearm.
    if len(good) > 1 and good[1]:
        wrist = np.array([u[1], v[1]], float)
        palm_ids = [i for i in (6, 7, 11, 17, 22) if i < len(good) and good[i]]
        if palm_ids:
            direction = wrist - np.mean(np.column_stack([u[palm_ids], v[palm_ids]]), axis=0)
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                end = wrist + direction / norm * 220.0
                cv2.line(mask, tuple(wrist.astype(int)), tuple(end.astype(int)), 255, 42)
    kernel = np.ones((max(3, dilation), max(3, dilation)), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    return mask


def _object_mask_paths(paths: RelPaths, frame: int) -> list[Path]:
    """这一帧**所有**物体的 step2 mask 路径。

    多物体时 `RelPaths.seg_masks(frame)` 给全部物体; 老路径对象只有单物体的
    `seg_mask(frame)` —— 两种都接, 于是这里对单物体逐字节保持原行为。
    """
    if hasattr(paths, "seg_masks"):
        return list(paths.seg_masks(frame))
    return [paths.seg_mask(frame)]


def _humanego_mask_and_lama(paths: RelPaths, step_dir: Path, reel=None, *, allow_fallback: bool,
                            lama_model=None, fps: float | None = None, frames=None,
                            prompt: str = DEFAULT_MASK_PROMPT, verbose: bool = True,
                            dump_boxes: bool = True):
    """Run HumanEgo-main's DINO+SAM2 mask and LaMa implementation verbatim.

    All new parameters are keyword-only with defaults equal to the previous hard-coded
    behaviour, so existing callers are byte-for-byte unchanged:

      fps     `reel.fps` unless given (the pipeline has no `Reel`).
      frames  source frame indices to process; None = every frame.  The dumped
              `frames_left/%05d.png` are keyed by **source frame index**, so a caller
              that only needs a subset can pass it and still address frames by their
              original index.
      prompt  DINO text prompt.  Default = arms + hands (unchanged); pass
              `HANDS_ONLY_MASK_PROMPT` for hands only.
      verbose per-frame DINO/LaMa timings.
      dump_boxes
              write the per-frame `dino_boxes/%05d.json` (boxes/confidences/timings).
              Purely diagnostic — nothing downstream reads them — so a caller that
              only wants the masks + `bg.mkv` passes False.  Default True = unchanged.
    """
    cv2 = _cv2()
    fps = float(fps if fps is not None else reel.fps)
    wanted = None if frames is None else {int(f) for f in frames}
    frames_dir = step_dir / "frames_left"
    mask_dir = step_dir / "masks_arm"
    box_dir = step_dir / "dino_boxes"
    frames_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    if dump_boxes:
        box_dir.mkdir(parents=True, exist_ok=True)
    raw_paths = []
    raw_frames = []
    for i, frame in enumerate(iter_frames(str(paths.mp4), FULL_W, FULL_H)):
        if wanted is not None and i not in wanted:
            continue
        p = frames_dir / f"{i:05d}.png"
        cv2.imwrite(str(p), cv2.cvtColor(frame[:, :EYE_W], cv2.COLOR_RGB2BGR))
        raw_paths.append(p)
        raw_frames.append(i)
    masks = []
    detection_records = []
    backend = "HumanEgo-main DINO+SAM2 + LaMa"
    try:
        repo = Path(__file__).resolve().parents[3]
        # Reuse the exact Step2 adapter.  It is a line-for-line HumanEgo
        # DINOSAM implementation with this repository's HF cache handling.
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ.setdefault("HF_HOME", str(repo / "test/work/.cache/huggingface"))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
        sys.path.insert(0, str(repo / "test/work"))
        from xrseg.utils.utils_io import load_cfg
        from xrseg.dinosam import DINOSAMEngine, PromptNotFoundError
        dino_cfg_path = repo / "test/work/xrseg/cfg/DINOSAM.yaml"
        dino_cfg = load_cfg(str(dino_cfg_path))
        dino = DINOSAMEngine(dino_cfg)
        for k, p in enumerate(raw_paths):
            i = raw_frames[k]
            frame_t0 = time.perf_counter()
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            dino.predictor.set_image(bgr)
            try:
                mask, avg_conf, boxes, confidences = dino.predict_frame_internal(bgr, prompt)
            except PromptNotFoundError:
                # 兼容显式严格模式或旧版 adapter：单帧空检测仍按
                # HumanEgo-main 的批处理约定继续，不污染后续帧。
                mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
                avg_conf = 0.0
                boxes = np.empty((0, 4), dtype=np.float32)
                confidences = np.empty((0,), dtype=np.float32)
            mask = mask if mask is not None else np.zeros(bgr.shape[:2], np.uint8)
            # 把**所有**物体的分割区从手部 mask 里抠掉, 免得物体被当成手一起被 LaMa 修掉。
            # 多物体时只抠 obj1 会让第二个物体留在这张 mask 里 -> 被 `mask > 0` 判成手 ->
            # 抹掉。`seg_masks` 不存在时 (老路径对象) 退回单物体的 `seg_mask`。
            for obj_path in _object_mask_paths(paths, i):
                if not obj_path.is_file():
                    continue
                obj = cv2.imread(str(obj_path), cv2.IMREAD_GRAYSCALE)
                if obj is not None and obj.shape == mask.shape:
                    mask[obj > 127] = 0
            masks.append(mask.astype(np.uint8))
            boxes_list = [] if boxes is None else np.asarray(boxes).tolist()
            confs_list = [] if confidences is None else np.asarray(confidences).tolist()
            detected = bool(len(boxes_list) > 0)
            record = {
                "prompt": prompt,
                "confidence": float(avg_conf) if detected else 0.0,
                "boxes": boxes_list,
                "box_confidences": confs_list,
                "detected": detected,
                "dino_sam2_ms": (time.perf_counter() - frame_t0) * 1000.0,
            }
            detection_records.append(record)
            if dump_boxes:
                (box_dir / f"{i:05d}.json").write_text(
                    json.dumps(record, indent=2), encoding="utf-8"
                )
            if verbose:
                print(
                    f"    [step3] frame {i:04d} mask DINO+SAM2: "
                    f"{record['dino_sam2_ms']:.1f} ms "
                    f"({'detected' if detected else 'empty'})",
                    flush=True,
                )
        frames_detected = sum(1 for x in detection_records if x["detected"])
        if frames_detected == 0:
            # 已经完成所有帧的检测，并且每帧的零 mask/JSON 都已落盘；
            # 这时才把问题报告为整段视频无检测，便于区分偶发丢帧。
            dino.cleanup()
            raise _AllFramesEmptyError(
                "GroundingDINO 在整段视频中没有检测到 human arms / human hands"
            )
        dino.cleanup()
        # LaMa is the exact HumanEgo-main engine and blending implementation.
        sys.path.insert(0, str(repo / "HumanEgo-main"))
        from preprocess.Lama import LamaEngine
        lama_cfg = repo / "HumanEgo-main/cfg/preprocess/base/Lama.yaml"
        lama = LamaEngine(str(lama_cfg.resolve()), model_path=lama_model)
        bg = step_dir / "bg.mkv"
        writer = VideoWriter(str(bg), EYE_W, EYE_H, fps)
        for k, (p, mask) in enumerate(zip(raw_paths, masks)):
            i = raw_frames[k]
            lama_t0 = time.perf_counter()
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            repaired_bgr = lama.inpaint(bgr, mask)
            writer.write(cv2.cvtColor(repaired_bgr, cv2.COLOR_BGR2RGB))
            lama_ms = (time.perf_counter() - lama_t0) * 1000.0
            detection_records[k]["lama_ms"] = lama_ms
            if dump_boxes:
                (box_dir / f"{i:05d}.json").write_text(
                    json.dumps(detection_records[k], indent=2), encoding="utf-8"
                )
            if verbose:
                print(f"    [step3] frame {i:04d} LaMa repair: {lama_ms:.1f} ms", flush=True)
        writer.close()
        return np.stack(masks), bg, backend, prompt, detection_records
    except _AllFramesEmptyError:
        # 即使显式打开 OpenCV 调试 fallback，也不能把“全视频无检测”
        # 掩盖成一段看似成功的修复结果。
        raise
    except Exception:
        if not allow_fallback:
            raise
        if reel is None:
            raise RuntimeError(
                "OpenCV 修复 fallback 需要 `reel` (逐帧 26 点几何), 但这次没传 —— "
                "pipeline 一律不开 fallback"
            )
        # Explicit debug fallback only; never selected silently.
        masks = np.stack([_hand_mask(reel, i) for i in raw_frames])
        bg = step_dir / "bg.mkv"
        writer = VideoWriter(str(bg), EYE_W, EYE_H, fps)
        for i, p in enumerate(raw_paths):
            rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            writer.write(cv2.inpaint(rgb, masks[i], 5, cv2.INPAINT_TELEA))
        writer.close()
        return masks, bg, "opencv_inpaint_fallback", "fallback", detection_records


def _draw_gripper_humanego(bgr: np.ndarray, T: np.ndarray, closed: bool, K: np.ndarray) -> np.ndarray:
    """Port of HumanEgo-main VisualKpts.process_single_gripper."""
    cv2 = _cv2()
    width = 0.05 if closed else 0.18
    half, finger, root = width / 2.0, 0.08, 0.08
    segments = [
        (np.array([0, 0, -(finger + root)]), np.array([-half, 0, -finger])),
        (np.array([0, 0, -(finger + root)]), np.array([half, 0, -finger])),
        (np.array([-half, 0, -finger]), np.array([-half, 0, 0])),
        (np.array([half, 0, -finger]), np.array([half, 0, 0])),
    ]
    if closed:
        colors = ([255, 255, 255], [255, 255, 255], [255, 0, 255], [255, 0, 255])
    else:
        colors = ([0, 255, 255], [0, 160, 255], [0, 60, 255], [0, 0, 255])
    def interp(a, b, t):
        return [int(a[k] + (b[k] - a[k]) * t) for k in range(3)]
    for sidx, (p0, p1) in enumerate(segments):
        start, end = (colors[3], colors[2]) if sidx < 2 else (colors[1], colors[0])
        for j, p in enumerate(np.linspace(p0, p1, 6)):
            pc = T[:3, :3] @ p + T[:3, 3]
            if pc[2] <= 0.001:
                continue
            uvh = K @ pc
            u, v = float(uvh[0] / uvh[2]), float(uvh[1] / uvh[2])
            if 0 <= u < bgr.shape[1] and 0 <= v < bgr.shape[0]:
                color = interp(start, end, j / 5.0)
                cv2.circle(bgr, (int(u), int(v)), 4, color, -1, cv2.LINE_AA)
                cv2.circle(bgr, (int(u), int(v)), 1, (255, 255, 255), -1, cv2.LINE_AA)
    return bgr


def run(paths: RelPaths, *, reel, frames=None, verbose=True, allow_opencv_fallback=False,
        lama_model=None, piper_model=None, piper_assets=None,
        piper_tcp_calibration=None, gripper_render="piper",
        piper_open_joint=None, piper_closed_joint=None) -> dict:
    if not paths.video.is_file():
        raise FileNotFoundError(f"先运行 --stage render: {paths.video}")
    step_dir = paths.outdir / "step3"
    step_dir.mkdir(parents=True, exist_ok=True)
    cv2 = _cv2(); n = int(reel.n_frames)
    masks, bg, backend, prompt, detection_records = _humanego_mask_and_lama(
        paths, step_dir, reel, allow_fallback=allow_opencv_fallback,
        lama_model=lama_model,
    )
    mask_dir = step_dir / "masks_right_hand"; mask_dir.mkdir(exist_ok=True)
    arm_dir = step_dir / "masks_arm"; arm_dir.mkdir(exist_ok=True)
    for i in range(n):
        cv2.imwrite(str(mask_dir / f"{i:05d}.png"), masks[i])
        cv2.imwrite(str(arm_dir / f"{i:05d}.png"), masks[i])
    mask_video = step_dir / "arm_mask.mp4"
    mask_writer = VideoWriter(str(mask_video), EYE_W, EYE_H, reel.fps)
    for mask in masks:
        mask_writer.write(np.repeat(mask[..., None], 3, axis=2))
    mask_writer.close()
    K = np.array([[reel.proj.p.eff_f, 0, reel.proj.p.eff_cx], [0, reel.proj.p.eff_f, reel.proj.p.eff_cy], [0, 0, 1]], float)
    wanted = set(frames or [])
    out = step_dir / "composite.mp4"
    piper = None
    piper_depth_writer = None
    piper_qa = []
    piper_meta = None
    if gripper_render == "piper":
        from .piper import PiperGripperModel, load_tcp_calibration, load_tcp_offset_model
        Kp = np.array([[reel.proj.p.eff_f, 0, reel.proj.p.eff_cx],
                       [0, reel.proj.p.eff_f, reel.proj.p.eff_cy], [0, 0, 1]], float)
        piper = PiperGripperModel(model_path=piper_model, assets_root=piper_assets,
                                  width=EYE_W, height=EYE_H, fx=Kp[0, 0], fy=Kp[1, 1],
                                  cx=Kp[0, 2], cy=Kp[1, 2])
        if piper_open_joint is not None:
            piper.open_joint = float(piper_open_joint)
        if piper_closed_joint is not None:
            piper.closed_joint = float(piper_closed_joint)
        T_hand_from_piper = load_tcp_calibration(piper_tcp_calibration)
        # Explicit gripper_base -> fingertip offset (Piper's t_flange_tool).
        # None means "take it from the URDF by forward kinematics".
        tcp_offset_model = load_tcp_offset_model(piper_tcp_calibration)
        out = step_dir / "composite_piper.mp4"
        piper_depth_writer = VideoWriter(str(step_dir / "piper_depth.mp4"), EYE_W, EYE_H, reel.fps)
        piper_meta = piper.manifest()
        piper_meta["tcp_calibration"] = str(Path(piper_tcp_calibration).resolve()) if piper_tcp_calibration else "default"
    else:
        T_hand_from_piper = np.eye(4, dtype=np.float64)
        tcp_offset_model = None
    info = probe(str(paths.video)); rendered_h = FULL_H + 178 + 168
    writer = VideoWriter(str(out), FULL_W, rendered_h, reel.fps)
    comparison = step_dir / ("composite_piper_sbs.mp4" if piper is not None else "composite_sbs.mp4")
    sbs_writer = VideoWriter(str(comparison), EYE_W * 3, EYE_H, reel.fps)
    bg_iter = iter(iter_frames(str(bg), EYE_W, EYE_H))
    geom_checks = {"valid_frames": 0, "invalid_frames": 0, "bad_rotation": 0,
                   "tcp_projection_mismatch": 0, "anchor_off_mesh": 0,
                   "anchor_not_distal": 0}
    for i, rendered in enumerate(iter_frames(str(paths.video), FULL_W, FULL_H + 178 + 168)):
        composite_t0 = time.perf_counter()
        # Render video has panels; replace only the left camera half of its top image.
        clean = next(bg_iter)
        canvas = rendered.copy()
        # Keep every existing step1/step2/relation pixel outside the hand mask;
        # only the real hand/forearm region is replaced by the repaired background.
        hand = masks[i][..., None] > 0
        canvas[:EYE_H, :EYE_W] = np.where(hand, clean, rendered[:EYE_H, :EYE_W])
        rec = reel.records[reel.ridx[i]]
        try:
            valid = bool(reel.G_POSE_VALID[i])
            mid = np.asarray(reel.G_MID[i], dtype=np.float64)
            R = np.asarray(reel.G_R[i], dtype=np.float64)
            valid = valid and mid.shape == (3,) and R.shape == (3, 3)
            valid = valid and np.isfinite(mid).all() and np.isfinite(R).all()
            if not valid:
                geom_checks["invalid_frames"] += 1
            else:
                geom_checks["valid_frames"] += 1
                det_r = float(np.linalg.det(R))
                if abs(det_r - 1.0) > 1e-3:
                    geom_checks["bad_rotation"] += 1
                    print(f"    [step3] gripper frame {i} invalid det(R)={det_r:.6f}", flush=True)
                    valid = False
            if valid:
                half = 0
                eye = reel.proj.eye_of_half(half)
                from .relation import camera_from_world
                T_cam_world = camera_from_world(
                    reel.proj.E, reel.proj.p, rec.head_pos, rec.head_quat, eye
                )
                T_world_gripper = np.eye(4, dtype=np.float64)
                # G_R/G_MID are the authoritative, already smoothed Step1
                # five-keypoint TCP pose.  Do not recompute or apply another
                # hand-to-gripper axis transform here; the fixed Piper model
                # relationship belongs in tcp_calibration.json.
                T_world_gripper[:3, :3] = R
                T_world_gripper[:3, 3] = mid
                T = T_cam_world @ T_world_gripper
                # Verify the TCP translation agrees with the same Projector path
                # used by Step1. The tolerance is in camera metres.
                p_head = reel.proj.project(mid, rec.head_pos, rec.head_quat, half)[3]
                p_cam_ref = reel.proj.head_to_cam(p_head, eye)
                if float(np.linalg.norm(T[:3, 3] - p_cam_ref)) > 1e-6:
                    geom_checks["tcp_projection_mismatch"] += 1
                    print(f"    [step3] gripper frame {i} TCP projection mismatch", flush=True)
                if piper is None:
                    left_bgr = cv2.cvtColor(canvas[:EYE_H, :EYE_W], cv2.COLOR_RGB2BGR)
                    left_bgr = _draw_gripper_humanego(left_bgr, T, bool(reel.G_CLOSED[i]), K)
                    canvas[:EYE_H, :EYE_W] = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
                else:
                    ratio = 0.0 if bool(reel.G_CLOSED[i]) else 1.0
                    # Place the official mesh by its measured fingertip
                    # midpoint, not by the gripper root.  This guarantees the
                    # Piper fingertips coincide with the Step1 TCP midpoint.
                    T_cam_model = piper.pose_from_tcp(
                        T, T_hand_from_piper, ratio, tcp_offset_model=tcp_offset_model
                    )
                    rgba, depth = piper.render(T_cam_model, ratio)
                    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
                    mesh_rgb = rgba[..., :3].astype(np.float32)
                    base = canvas[:EYE_H, :EYE_W].astype(np.float32)
                    canvas[:EYE_H, :EYE_W] = np.clip(
                        mesh_rgb * alpha + base * (1.0 - alpha), 0, 255
                    ).astype(np.uint8)
                    if piper_depth_writer is not None:
                        dimg = np.zeros((EYE_H, EYE_W, 3), np.uint8)
                        good = depth > 0
                        if np.any(good):
                            d = np.clip(depth, 0.0, 2.0) / 2.0
                            vals = (255.0 * (1.0 - d[good])).astype(np.uint8)
                            dimg[good] = np.repeat(vals[:, None], 3, axis=1)
                        piper_depth_writer.write(dimg)
                    p_model = np.asarray(piper.last_fingertip_midpoint_model, dtype=np.float64)
                    tip_cam = T_cam_model[:3, :3] @ p_model + T_cam_model[:3, 3]
                    root_cam = T_cam_model[:3, 3]
                    def _uv(point):
                        q = K @ point
                        return [float(q[0] / q[2]), float(q[1] / q[2])] if q[2] > 1e-9 else [None, None]
                    tip_uv = _uv(tip_cam)
                    hand_uv = _uv(T[:3, 3])
                    root_uv = _uv(root_cam)
                    # Real checks.  Do NOT go back to "distance between the anchor
                    # projection and the hand projection": pose_from_tcp solves the
                    # root so that T_cam_model @ p_model == T[:3, 3] exactly, so that
                    # distance is identically 0 and can never catch a wrong anchor.
                    anchor_on_mesh = None
                    if None not in tip_uv:
                        u, v = int(round(tip_uv[0])), int(round(tip_uv[1]))
                        if 0 <= u < rgba.shape[1] and 0 <= v < rgba.shape[0]:
                            anchor_on_mesh = bool(rgba[v, u, 3] > 0)
                    if anchor_on_mesh is False:
                        geom_checks["anchor_off_mesh"] += 1
                    distal = np.asarray(getattr(piper, "last_mesh_distal_extent_model", p_model),
                                       dtype=np.float64)
                    anchor_distal_gap_m = float(p_model[2] - distal[2])
                    if abs(anchor_distal_gap_m) > 1e-3:
                        geom_checks["anchor_not_distal"] += 1
                    piper_qa.append({
                        "frame": i, "open_ratio": ratio,
                        "rendered_pixels": int(np.count_nonzero(rgba[..., 3])),
                        "tcp_valid": True,
                        "det_rotation": float(np.linalg.det(T_cam_model[:3, :3])),
                        "tcp_alignment_error_m": float(getattr(piper, "last_tcp_alignment_error_m", 0.0)),
                        "fingertip_midpoint_model_m": p_model.tolist(),
                        "tcp_anchor_source": getattr(piper, "last_anchor_source", "fk_link_origins"),
                        "mesh_distal_extent_model_m": distal.tolist(),
                        "anchor_minus_mesh_distal_extent_m": anchor_distal_gap_m,
                        "anchor_inside_silhouette": anchor_on_mesh,
                        "pose_contract": "HumanEgo VisualKpts fingertip-midpoint frame",
                        "tip_midpoint_projection": tip_uv,
                        "hand_midpoint_projection": hand_uv,
                        "root_projection": root_uv,
                    })
        except Exception as exc:
            print(f"    [step3] gripper frame {i} failed: {type(exc).__name__}: {exc}", flush=True)
        writer.write(canvas)
        sbs_writer.write(np.concatenate((rendered[:EYE_H, :EYE_W], clean,
                                         canvas[:EYE_H, :EYE_W]), axis=1))
        if i in wanted: Image.fromarray(canvas).save(step_dir / f"step3_f{i:04d}.png")
        composite_ms = (time.perf_counter() - composite_t0) * 1000.0
        detection_records[i]["composite_ms"] = composite_ms
        extra_ms = (detection_records[i].get("dino_sam2_ms", 0.0)
                    + detection_records[i].get("lama_ms", 0.0)
                    + composite_ms)
        detection_records[i]["total_extra_ms"] = extra_ms
        print(
            f"    [step3] frame {i:04d} extra latency: "
            f"mask={detection_records[i].get('dino_sam2_ms', 0.0):.1f} ms, "
            f"lama={detection_records[i].get('lama_ms', 0.0):.1f} ms, "
            f"composite={composite_ms:.1f} ms, total={extra_ms:.1f} ms",
            flush=True,
        )
    writer.close()
    sbs_writer.close()
    if piper_depth_writer is not None:
        piper_depth_writer.close()
    if piper is not None:
        piper.close()
        shutil.copyfile(out, step_dir / "composite.mp4")
        Image.fromarray(canvas[:EYE_H, :EYE_W]).save(step_dir / "piper_preview.png")
        (step_dir / "piper_geometry_qa.json").write_text(
            json.dumps(piper_qa, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # The pre-loop manifest could not know which anchor the frames used.
        piper_meta.update(piper.manifest())
    manifest = {"stage": "step3", "frames": n, "fps": float(info.get("fps", reel.fps)),
                "outputs": {"masks": str(mask_dir), "mask_video": str(mask_video), "background": str(bg), "composite": str(out), "comparison": str(comparison),
                            "piper_depth": str(step_dir / "piper_depth.mp4") if piper_meta else None,
                            "piper_preview": str(step_dir / "piper_preview.png") if piper_meta else None},
                "piper_geometry_qa": str(step_dir / "piper_geometry_qa.json") if piper_meta else None,
                "mask_backend": "HumanEgo-main DINO+SAM2" if backend.startswith("HumanEgo") else "projected_tracking_fallback",
                "mask_prompt": prompt,
                "inpaint_backend": "HumanEgo-main LaMa" if backend.startswith("HumanEgo") else backend,
                "gripper_backend": "official AgileX Piper URDF mesh" if piper is not None else "HumanEgo-main VisualKpts.process_single_gripper",
                "gripper_render": gripper_render,
                "piper": piper_meta,
                "piper_material": "black" if piper is not None else None,
                "opencv_fallback": bool(backend == "opencv_inpaint_fallback"),
                "frames_total": n,
                "frames_detected": sum(1 for x in detection_records if x.get("detected", False)),
                "frames_empty_detection": sum(1 for x in detection_records if not x.get("detected", False)),
                "empty_detection_runs": _empty_detection_runs(detection_records),
                "lama_model_path": (str(Path(lama_model).resolve()) if lama_model else None),
                "lama_model_source": "local" if lama_model else "huggingface",
                "latency_ms": {
                    "per_frame": detection_records,
                    "average_total_extra": float(np.mean([x.get("total_extra_ms", 0.0) for x in detection_records])) if detection_records else 0.0,
                    "max_total_extra": float(np.max([x.get("total_extra_ms", 0.0) for x in detection_records])) if detection_records else 0.0,
                },
                "gripper_geometry_checks": geom_checks,
                "gripper_axis_mapping": "T_hand_from_piper in tcp_calibration.json; no per-frame extra transform",
                "tcp_anchor": (
                    "tcp_offset_model_m in tcp_calibration.json"
                    if piper_meta.get("tcp_anchor_source") == "calibration"
                    else "URDF forward kinematics (finger link origins)"
                ) if piper_meta else None,
                "pose_contract": "HumanEgo VisualKpts fingertip-midpoint frame",
                "pose_source": "Step1 reel.G_MID/reel.G_R",
                "extra_pose_axis_transform": False,
                "backend": backend, "backend_env": {
                    "PHANTOM_PY": os.environ.get("PHANTOM_PY"),
                    "PHANTOM_ROOT": os.environ.get("PHANTOM_ROOT"),
                    "PHANTOM_STAGE1": os.environ.get("PHANTOM_STAGE1"),
                    "PHANTOM_STAGE3_CMD": os.environ.get("PHANTOM_STAGE3_CMD"),
                }, "tcp": "right five-point midpoint frame"}
    (step_dir / "step3.manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
