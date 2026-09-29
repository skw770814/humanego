"""step3: 手 (可选手臂) mask + 修复 + piper 夹爪合成 -> 纯 RGB 观测帧。

和 `xrrel.step3.run` 的区别只有一处, 但很关键: **合成基座是原始左眼帧**, 不是那份
render 叠加视频。后者 (2160x1156) 里已经烤进了 XYZ 轴 / REL 面板 / step1 关键点 /
step2 mask, 拿它当底图就永远去不掉那些辅助可视化。这里改成:

    clean  = bg.mkv 第 k 帧                 # LaMa 修复后的左眼原 RGB
    raw    = 原始左眼第 i 帧                 # CameraRecord 的左半
    canvas = np.where(hand_mask > 0, clean, raw)     # 只在手/手臂区域替换
    + piper 夹爪按 step1 的 T_camera0_midpoint 合成

于是最终帧里只有「原 RGB + 手修复 + 夹爪」, 没有别的叠加。

复用 `xrrel.step3._humanego_mask_and_lama` 做 mask+修复 (HumanEgo-main 的 DINO+SAM2 +
LaMa 本体)。它用到的 `paths` 成员只有 `.mp4` 和 `.seg_mask(i)`, `PipePaths` 两个都有。

夹爪位姿**直接取 step1 的 `T_camera0_midpoint`**, 不再走 `camera_from_world` 重推 ——
step1 与 step3 看到的末端因此是同一个矩阵 (构造上相等, 不靠重新推导对齐)。

帧命名按**源帧号**: `observation/%05d.png`。所以 `--only-keep` 只处理保留帧时, 文件名
仍然是源帧号, step4 按 `keep_index` 取帧两种跑法结果一致。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import (EYE_H, EYE_W, FPS, FULL_H, FULL_W, MASK_PROMPT_HANDS,
               MASK_PROMPT_HANDS_ARMS, bootstrap)
from .paths import PipePaths


def _compositing_frames(paths: PipePaths, only_keep: bool, max_invalid_gap: int) -> np.ndarray:
    """要合成的源帧号。`--only-keep` 时只做 step4 会保留的那些帧 (纯提速, 不改结果)。"""
    from .episode import keep_index
    from .step1 import load

    n = int(load(paths)["n_frames"])
    if not only_keep:
        return np.arange(n, dtype=np.int64)
    if not paths.state_action_npz.is_file():
        raise FileNotFoundError(
            f"--only-keep 需要 step2 的 {paths.state_action_npz} (有效段在那里)"
        )
    with np.load(paths.state_action_npz) as archive:
        valid = archive["state_valid"]
    return keep_index(valid, max_invalid_gap).astype(np.int64)


def run(paths: PipePaths, *, with_arm: bool = False, gripper_render: str = "piper",
        lama_model=None, piper_model=None, piper_assets=None, piper_tcp_calibration=None,
        piper_open_joint=None, piper_closed_joint=None, mask_prompt=None,
        only_keep: bool = False, max_invalid_gap: int = 30, force: bool = False,
        quiet: bool = False, vis: bool = False) -> dict:
    """跑 step3, 写 `step3/observation/%05d.png` + `step3/observation.json`。

    `vis` 在这里只表示「保留诊断产物」(`dino_boxes/*.json`); 可视化视频由模块级的
    `render_vis()` 单独出 —— 那样 step3 已经跑完时不用 --force 重跑整步。
    """
    bootstrap()
    from xrhand.video import iter_frames
    from xrrel.step3 import _humanego_mask_and_lama

    from .step1 import load

    reel = load(paths)
    n = int(reel["n_frames"])
    T_mid = np.asarray(reel["T_camera0_midpoint"], dtype=np.float64)
    closed = np.asarray(reel["grasp_closed"], dtype=bool)
    pose_valid = np.asarray(reel["pose_valid"], dtype=bool)
    eff_f, eff_cx, eff_cy = (float(reel["eff_f"]), float(reel["eff_cx"]), float(reel["eff_cy"]))

    frames = _compositing_frames(paths, only_keep, max_invalid_gap)
    step_dir = paths.step3_dir
    step_dir.mkdir(parents=True, exist_ok=True)
    paths.observation_dir.mkdir(parents=True, exist_ok=True)

    # 这是给 DINO/SAM2 的**检测提示词** (「框出手」), 不是任务指令。
    # 默认只 mask 手; `--with-arm` 等价于选 work 里那条「手 + 手臂」的默认词。
    prompt = mask_prompt or (MASK_PROMPT_HANDS_ARMS if with_arm else MASK_PROMPT_HANDS)
    # allow_fallback=False: 手部 mask 由 DINO+SAM2 负责, 不接受 OpenCV 兜底
    # (兜底出来的 mask 不是「检测到手」的证据, 会静默产出没修干净的画面)。
    masks, bg, backend, prompt_used, records = _humanego_mask_and_lama(
        paths, step_dir, reel=None, allow_fallback=False, lama_model=lama_model,
        fps=FPS, frames=frames, prompt=prompt, verbose=not quiet, dump_boxes=vis,
    )

    K = np.array([[eff_f, 0.0, eff_cx], [0.0, eff_f, eff_cy], [0.0, 0.0, 1.0]], float)
    piper = None
    T_hand_from_piper = np.eye(4, dtype=np.float64)
    tcp_offset_model = None
    if gripper_render == "piper":
        from xrrel.piper import (PiperGripperModel, load_tcp_calibration,
                                 load_tcp_offset_model)

        piper = PiperGripperModel(model_path=piper_model, assets_root=piper_assets,
                                  width=1080, height=810,
                                  fx=eff_f, fy=eff_f, cx=eff_cx, cy=eff_cy)
        if piper_open_joint is not None:
            piper.open_joint = float(piper_open_joint)
        if piper_closed_joint is not None:
            piper.closed_joint = float(piper_closed_joint)
        T_hand_from_piper = load_tcp_calibration(piper_tcp_calibration)
        tcp_offset_model = load_tcp_offset_model(piper_tcp_calibration)
    elif gripper_render != "wireframe":
        raise SystemExit(f"未知的 --gripper-render {gripper_render!r} (piper|wireframe)")

    stats = {"valid_frames": 0, "invalid_frames": 0, "bad_rotation": 0,
             "tcp_alignment_error_m_max": 0.0, "hand_mask_px": 0, "alpha_px": 0}
    bg_iter = iter(iter_frames(str(bg), 1080, 810))
    wanted = set(int(f) for f in frames)
    out_index = 0
    for i, frame in enumerate(iter_frames(str(paths.mp4), 2160, 810)):
        if i not in wanted:
            continue
        clean = next(bg_iter)
        mask = masks[out_index]
        out_index += 1
        raw = frame[:, :1080]
        stats["hand_mask_px"] += int(np.count_nonzero(mask))

        canvas = np.where(mask[..., None] > 0, clean, raw)

        T = T_mid[i]
        rotation = T[:3, :3]
        usable = bool(pose_valid[i]) and np.isfinite(T).all()
        if usable and abs(float(np.linalg.det(rotation)) - 1.0) > 1e-3:
            stats["bad_rotation"] += 1
            usable = False
        if not usable:
            stats["invalid_frames"] += 1
            _save(paths, i, canvas)
            continue
        stats["valid_frames"] += 1

        ratio = 0.0 if bool(closed[i]) else 1.0
        if piper is None:
            _draw_wireframe(canvas, T, bool(closed[i]), K)
        else:
            T_cam_model = piper.pose_from_tcp(T, T_hand_from_piper, ratio,
                                              tcp_offset_model=tcp_offset_model)
            rgba, _depth = piper.render(T_cam_model, ratio)
            alpha = rgba[..., 3:4].astype(np.float32) / 255.0
            base = canvas.astype(np.float32)
            canvas = np.clip(rgba[..., :3].astype(np.float32) * alpha + base * (1.0 - alpha),
                             0, 255).astype(np.uint8)
            stats["alpha_px"] += int(np.count_nonzero(alpha > 0.01))
            stats["tcp_alignment_error_m_max"] = max(
                stats["tcp_alignment_error_m_max"],
                float(getattr(piper, "last_tcp_alignment_error_m", 0.0)),
            )
        _save(paths, i, canvas)

    payload = {
        "stem": paths.stem,
        "n_frames": n,
        "frames_written": len(frames),
        "observation": {"eye": "left (camera0 / eye0)", "width": 1080, "height": 810,
                        "fps": FPS,
                        "intrinsics": {"f": eff_f, "cx": eff_cx, "cy": eff_cy},
                        "naming": "observation/%05d.png 用**源帧号**; step4 按 keep_index 取帧"},
        "mask_prompt": prompt_used,
        "with_arm": bool(with_arm),
        "gripper_render": gripper_render,
        "mask_backend": backend,
        "only_keep": bool(only_keep),
        "dino_empty_frames": [i for i, r in enumerate(records) if not r.get("detected")],
        "geometry": stats,
        "piper": (piper.manifest() if piper is not None else None),
    }
    paths.observation_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    if not quiet:
        print(f"  step3 {paths.stem}: 合成 {len(frames)} 帧 -> {paths.observation_dir}")
        print(f"    mask 提示词 {prompt_used!r} / 夹爪 {gripper_render} / 后端 {backend}")
        print(f"    夹爪有效 {stats['valid_frames']} 帧, 位姿无效 {stats['invalid_frames']} 帧, "
              f"TCP 对齐误差最大 {stats['tcp_alignment_error_m_max']:.2e} m, "
              f"DINO 空检测 {len(payload['dino_empty_frames'])} 帧")
    return payload


def _save(paths: PipePaths, frame: int, rgb: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(rgb).save(paths.observation(frame))


def _draw_wireframe(canvas: np.ndarray, T: np.ndarray, closed: bool, K: np.ndarray) -> None:
    """`--gripper-render wireframe`: 只作对照, 就地画在 canvas 上 (RGB 进 BGR 出再转回)。"""
    import cv2

    from xrrel.step3 import _draw_gripper_humanego

    bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    _draw_gripper_humanego(bgr, T, closed, K)
    canvas[:] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------- 可视化 (--vis)


def render_vis(paths: PipePaths, *, sbs: bool = True) -> dict:
    """`--vis`: 把**已经在盘上的** `observation/*.png` 重编码成视频。

    刻意不这么做: 在合成循环里顺手写一条视频。改用重编码, 换来两件事 ——

      - step4 编码的就是同一批 PNG, 所以「视频与标签同源同序」是构造上的结论;
      - step3 已经跑完时不用 `--force` 重跑整步 (含 DINO+SAM2+LaMa)。

    产物 (落在 `step3/` 里, 与 work 的 `rel_<stem>/step3/` 布局平行):

      `composite_piper.mp4`  合成帧本身 (原 RGB + 手修复 + piper 夹爪)
      `composite_sbs.mp4`    原始左眼帧 || 合成帧 (2160x810), 用来判断手修得干不干净
    """
    from .export import write_video

    files = sorted(paths.observation_dir.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"{paths.observation_dir} 里没有观测帧 —— 先跑 step3")
    out = {"composite_piper": str(paths.step3_dir / "composite_piper.mp4")}
    write_video(files, Path(out["composite_piper"]), size=(EYE_W, EYE_H), fps=FPS)
    if sbs:
        out["composite_sbs"] = str(_write_sbs(paths, files))
    return out


def _write_sbs(paths: PipePaths, files: list[Path]) -> Path:
    """原始左眼帧 || 合成帧。帧按**源帧号**命名, 所以这里把 mp4 顺序与文件名对上。"""
    import cv2
    from xrhand.video import iter_frames

    from .export import BgrSink, DEFAULT_CODEC

    wanted = [int(p.stem) for p in files]
    destination = paths.step3_dir / "composite_sbs.mp4"
    # 可视化固定走 DEFAULT_CODEC (h264), 不跟 --video-codec —— 它的用处就是双击打开看。
    sink = BgrSink(destination, codec=DEFAULT_CODEC, size=(2 * EYE_W, EYE_H), fps=FPS)
    written = 0
    try:
        for i, frame in enumerate(iter_frames(str(paths.mp4), FULL_W, FULL_H)):
            if written >= len(wanted):
                break
            if i != wanted[written]:
                continue
            composited = cv2.imread(str(files[written]), cv2.IMREAD_COLOR)
            if composited is None:
                raise FileNotFoundError(f"读不到观测帧 {files[written]}")
            # iter_frames 给的是 RGB; cv2 读写一律 BGR, 两边都转齐再拼 (sink 再转回 RGB)。
            left = cv2.cvtColor(frame[:, :EYE_W], cv2.COLOR_RGB2BGR)
            sink.write(np.hstack([left, composited]))
            written += 1
    finally:
        sink.close()
    if written != len(files):
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"对照视频只写到 {written}/{len(files)} 帧 —— 观测帧号与 mp4 对不上")
    return destination
