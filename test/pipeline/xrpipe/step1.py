"""step1: MP4 与 txt 对齐到 30Hz -> 每帧指尖中点相对**左眼相机**的位姿。

全部复用 `test/pipeline`, 不新写几何:

  - 帧对齐   `tools/overlay.py::Reel` (内部 `xrhand.align.build_mapping` + `read_lag`,
             读 `work/out/align_<stem>.json` 里相位相关测出来的 lag, 不硬编码)
  - 5 关键点 `xrhand.gripper` (PICO 26 点里的 W/TB/IB/T/IT = 1/3/7/5/10)
             -> `Reel.compute_gripper()` 产出 G_MID / G_R / G_POSE_VALID / G_CLOSED
  - 世界->左眼 `xrrel.relation.camera_from_world` + `hand_states`

启动时跑一次 `xrrel.relation.self_check_camera0`: 它把 `camera_from_world` 与
`xrhand.camera.Projector` 逐点比对并要求 <1e-9。这是防「物体点云一个系、手另一个系」
那类老坑的锁, 必须保留。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import FPS, bootstrap
from .paths import PipePaths

POSE_CONTRACT = (
    "HumanEgo MidpointFrameBuilder: 原点 = 拇指尖/食指尖的中点, "
    "x = 食指根 - 拇指根, y = 由两手根中点与腕的 Gram-Schmidt 得到, z = x cross y; "
    "**不是** TCP / 手腕语义"
)


def ensure_alignment(stem: str, *, force: bool = False) -> Path:
    """确保 step1 所需的 lag 对齐产物存在；缺失或强制时在 pipeline/out 内生成。"""
    from . import DATA, OUT
    from xrhand import align

    destination = OUT / f"align_{stem}.json"
    if destination.is_file() and not force:
        return destination
    tracking = DATA / f"trackingData_{stem}.txt"
    video = DATA / f"CameraRecord_{stem}.mp4"
    if not tracking.is_file() or not video.is_file():
        raise FileNotFoundError(f"自动对齐缺输入: {tracking} / {video}")
    code = align.main([str(tracking), "--video", str(video), "--outdir", str(OUT)])
    if code or not destination.is_file():
        raise RuntimeError(f"自动对齐 {stem} 失败: exit={code}, 未生成 {destination}")
    return destination


def compute(stem: str, *, calib: str | Path | None = None, lag: int | None = None,
            force_alignment: bool = False) -> dict:
    """跑链路, 返回算好的数组 (不落盘)。未指定 lag 时自动完成视频/跟踪对齐。"""
    bootstrap()
    if lag is None:
        ensure_alignment(stem, force=force_alignment)
    from xrrel.relation import hand_states, load_reel, self_check_camera0

    overlay, params, reel = load_reel(stem, calib=calib, lag=lag, gripper=True)
    n = int(reel.n_frames)
    hands = hand_states(reel, n)
    check = self_check_camera0(reel)
    return {
        "overlay": overlay,
        "params": params,
        "reel": reel,
        "hands": hands,
        "camera_check": check,
        "lag_meta": getattr(reel, "lag_meta", None),
        "clipped_frames": int(np.count_nonzero(reel.clipped)),
        "out_of_span_frames": int(np.count_nonzero(~np.asarray(reel.mapping.in_span, dtype=bool))),
    }


def write(paths: PipePaths, result: dict, *, force: bool = False) -> dict:
    """落 `step1/reel.npz` + `step1/reel.json`。"""
    from xrrel import EYE_CX, EYE_CY, EYE_F, INSTANCE_ID

    if paths.reel_npz.is_file() and not force:
        raise FileExistsError(f"{paths.reel_npz} 已存在 (--force 覆盖)")
    paths.ensure("step1")

    reel, hands = result["reel"], result["hands"]
    n = int(reel.n_frames)
    np.savez_compressed(
        paths.reel_npz,
        stem=paths.stem,
        n_frames=np.int32(n),
        fps=np.float32(FPS),
        lag=np.int32(reel.lag),
        frame_index=np.arange(n, dtype=np.int32),
        record_index=np.asarray(reel.ridx, dtype=np.int32),
        in_span=np.asarray(reel.mapping.in_span, dtype=bool),
        clipped=np.asarray(reel.clipped, dtype=bool),
        residual_us=np.asarray(reel.mapping.residual_us, dtype=np.float64),
        timestamp_ns=hands["timestamps_ns"],
        T_camera0_world=hands["T_camera0_world"],
        T_camera0_midpoint=hands["T_camera0_midpoint"],
        T_world_midpoint=hands["T_world_midpoint"],
        pose_valid=hands["valid"],
        key_valid=hands["key_valid"],
        grasp_closed=hands["closed"],
        grasp_closure=hands["closure"].astype(np.float32),
        pinch_m=hands["pinch_m"].astype(np.float32),
        judge=hands["judge"],
        G_MID=np.asarray(reel.G_MID, dtype=np.float64),
        G_R=np.asarray(reel.G_R, dtype=np.float64),
        eff_f=np.float64(reel.proj.p.eff_f),
        eff_cx=np.float64(reel.proj.p.eff_cx),
        eff_cy=np.float64(reel.proj.p.eff_cy),
        extra_R=np.asarray(reel.params.extra_R, dtype=np.float64),
        extra_t=np.asarray(reel.params.extra_t, dtype=np.float64),
        # 跟踪文件头 (双目外参 + cameraIntrinsics)。step2 的 SGBM 深度要 `header.extrinsics`,
        # 顺手把整个 header 存下来, 让 step2 用真正的 `xrhand.io_tracking.Header`,
        # 而不是自己造一个鸭子类型 —— 那样 `make_shim` 里的内参不变量断言会被绕过去。
        header_extrinsics=np.asarray(reel.header.extrinsics, dtype=np.float64),
        header_intrinsics=np.asarray(reel.header.intrinsics, dtype=np.float64),
        header_time_stamp_ns=np.int64(reel.header.time_stamp_ns),
        header_notice=str(reel.header.notice),
    )

    payload = {
        "stem": paths.stem,
        "n_frames": n,
        "fps": FPS,
        "observation": {
            "eye": "left (camera0 / eye0)",
            "width": 1080,
            "height": 810,
            "intrinsics": {"f": EYE_F, "cx": EYE_CX, "cy": EYE_CY,
                           "eff_f": float(reel.proj.p.eff_f),
                           "eff_cx": float(reel.proj.p.eff_cx),
                           "eff_cy": float(reel.proj.p.eff_cy)},
        },
        "eef": "right",
        "pose_contract": POSE_CONTRACT,
        "camera_frame": "camera0 = 左眼 (eye0), 与 SAM2 mask / 双目深度同一个像素系",
        "instance_id": INSTANCE_ID,
        "lag": {
            "frames": int(reel.lag),
            "meta": result["lag_meta"],
        },
        "camera_self_check": result["camera_check"],
        "grasp_calibration": hands["grasp_calibration"],
        "pose_valid_frames": int(hands["valid"].sum()),
        "key_valid_frames": int(hands["key_valid"].all(axis=1).sum()),
        "clipped_frames": result["clipped_frames"],
        "out_of_span_frames": result["out_of_span_frames"],
        "grasp_closed_frames": int(hands["closed"].sum()),
    }
    paths.reel_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return payload


def run(paths: PipePaths, *, calib=None, lag=None, force: bool = False, quiet: bool = False) -> dict:
    result = compute(paths.stem, calib=calib, lag=lag, force_alignment=force)
    payload = write(paths, result, force=force)
    if not quiet:
        check = result["camera_check"]
        print(
            f"  step1 {paths.stem}: {payload['n_frames']} 帧, lag={payload['lag']['frames']}, "
            f"指尖中点有效 {payload['pose_valid_frames']} 帧, 夹爪闭合 "
            f"{payload['grasp_closed_frames']} 帧, 相机自检最大像素误差 "
            f"{check['max_pixel_error']:.2e}"
        )
    return payload


def load(paths: PipePaths) -> dict:
    """读回 step1 的 npz (step3 用)。"""
    if not paths.reel_npz.is_file():
        raise FileNotFoundError(f"缺 {paths.reel_npz} —— 先跑 step1")
    with np.load(paths.reel_npz) as archive:
        return {key: archive[key] for key in archive.files}


def header(paths: PipePaths):
    """从产物里重建 `xrhand.io_tracking.Header` (step2 用)。

    SGBM 深度要 `header.extrinsics` (双目基线/相对位姿); 内参只被 `make_shim` 里的
    四条不变量断言读一次。存的是原始字段, 重建出来的是真的 `Header` 实例。
    """
    from xrhand.io_tracking import Header

    data = load(paths)
    return Header(
        notice=str(data["header_notice"]),
        time_stamp_ns=int(data["header_time_stamp_ns"]),
        extrinsics=data["header_extrinsics"],
        intrinsics=data["header_intrinsics"],
    )


def hands(paths: PipePaths) -> dict:
    """从产物里重建 `xrrel.relation.hand_states` 返回的那个 dict (step2 用)。

    step2 的下游 (`objectpose.estimate` / `relation.run`) 只认
    `T_camera0_midpoint / valid / closed / timestamps_ns / T_camera0_world / grasp_calibration`
    这几个 key, 这里按原名凑齐 —— 名字对不上会让「锁存用一个末端、报出来用另一个」这类
    问题变成静默错误, 不如在这里显式接一次。
    """
    data = load(paths)
    payload = json.loads(paths.reel_json.read_text(encoding="utf-8"))
    return {
        "T_camera0_world": data["T_camera0_world"],
        "T_camera0_midpoint": data["T_camera0_midpoint"],
        "T_world_midpoint": data["T_world_midpoint"],
        "valid": data["pose_valid"],
        "closed": data["grasp_closed"],
        "key_valid": data["key_valid"],
        "closure": data["grasp_closure"],
        "pinch_m": data["pinch_m"],
        "judge": data["judge"],
        "timestamps_ns": data["timestamp_ns"],
        "grasp_calibration": payload["grasp_calibration"],
    }


# ---------------------------------------------------------------- 可视化 (--vis)


VIS_TAG = {"skeleton": "overlay", "gripper": "gripper", "both": "gripper_both"}


def render_vis(paths: PipePaths, *, mode: str = "gripper") -> dict:
    """`--vis`: 用 work 的 `tools/overlay.py` 出一段「手 / 夹爪叠在原片上」的视频。

    这正是 work 那边一直在用的那份可视化, 参数上只有两处**必须显式给**, 否则会踩坑:

      `--calib`  overlay.py:561 是 `cj_path = args.calib or os.path.join(args.outdir, "calib.json")`
                 —— calib 跟着 `--outdir` 走。我们的 outdir 是 `out/<stem>/step1`,
                 那里没有 calib.json, 不给就会 SystemExit。
      `--lag`    `Reel.read_lag` (overlay.py:84) 读的是 overlay.py 模块级的
                 `work/out/align_<stem>.json`, **不跟 outdir**。直接把 step1 已经定下来
                 的 lag 传进去, 把这个隐性依赖掐掉。
    """
    from . import DEFAULT_CALIB, tool_module

    if mode not in VIS_TAG:
        raise ValueError(f"未知的 mode {mode!r} (可选 {sorted(VIS_TAG)})")
    data = load(paths)
    argv = [
        paths.stem, "--mode", mode, "--video",
        "--outdir", str(paths.step1_dir),
        "--calib", str(DEFAULT_CALIB),
        "--lag", str(int(data["lag"])),
    ]
    module = tool_module("overlay")
    try:
        code = module.main(argv)
    except SystemExit as exit_error:  # overlay.py 的失败路径是 raise SystemExit, 不是返回码
        raise RuntimeError(f"overlay.py {' '.join(argv)} 失败: {exit_error}") from exit_error
    if code:
        raise RuntimeError(f"overlay.py {' '.join(argv)} 返回 {code}")
    return {"video": str(paths.step1_dir / f"{VIS_TAG[mode]}_{paths.stem}.mp4")}
