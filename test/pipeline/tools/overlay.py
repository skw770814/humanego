"""Step 2: 手部关键点叠加 (26 点灵巧手 / 5 点二值爪)。

**标定默认就加载**: 不传 `--calib` 时读 `out/calib.json` (与 `xrseg/hand_overlay.py`
同约定); 读不到直接退出, 想用标称参数得显式 `--no-calib`。
以前默认标称参数是静默的坑 —— `extra_t` 全零会让深度差 5.24 cm、骨架整体偏 37~163 px,
画面上就是"关键点没贴合手"。

`--mode` 三选一 (默认 `gripper`):
  - `skeleton`: 26 点灵巧手, 与加 `--mode` 之前逐像素一致;
  - `gripper` : 每只手只画 5 个关键点 (腕/拇指根/食指根/拇指尖/食指尖) —— 由
    `draw_hand` 本体按原配色原半径画, 只是收窄了 `joints`/`edges`; 另叠夹爪那一层
    (两指尖中点 = 夹爪位置, MidpointFrameBuilder 的 6DoF 坐标架并标 X/Y/Z) 与
    二值开合状态 (判据见 `xrhand/gripper.py`);
  - `both`    : 两套都画。

原来的用法 (Step 2 诊断, 只用标称参数看贴合程度) 现在写成 `--no-calib`:

    python tools/overlay.py 20260920_111300 --no-calib        # 自动挑 8 帧静帧
    python tools/overlay.py 20260920_111300 --frames 10,120,200
    python tools/overlay.py 20260920_111300 --video           # 全片 (Step 4)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORK)

from xrhand import gripper as gr  # noqa: E402
from xrhand import render as R  # noqa: E402
from xrhand import skeleton as sk  # noqa: E402
from xrhand.align import build_mapping, head_angular_step  # noqa: E402
from xrhand.camera import EYE_H, EYE_W, CameraParams, make_projector  # noqa: E402
from xrhand.io_tracking import load, quat_to_mat  # noqa: E402
from xrhand.video import VideoWriter, count_frames, iter_frames, probe  # noqa: E402

DATA = os.path.dirname(WORK)
OUT = os.path.join(WORK, "out")


# ---------------------------------------------------------------- 每帧几何


class Reel:
    """一条采集的完整几何: 每帧 -> 记录 -> 26 点在两半画面上的投影。"""

    def __init__(self, stem: str, lag: int | None, params: CameraParams | None):
        self.stem = stem
        self.txt = os.path.join(DATA, f"trackingData_{stem}.txt")
        self.mp4 = os.path.join(DATA, f"CameraRecord_{stem}.mp4")
        self.header, self.records = load(self.txt)
        self.info = probe(self.mp4)
        self.fps = float(self.info["fps"])
        self.n_frames = count_frames(self.mp4)

        self.mapping = build_mapping(
            self.header, self.records, self.n_frames, self.fps, clock="predict"
        )
        self.lag = self.read_lag() if lag is None else int(lag)
        self.params = params or CameraParams()
        self.proj = make_projector(self.header, self.params)

        # 逐帧记录下标 = 标称映射 + lag。clip 到合法范围, 同时记下被 clip 的帧。
        raw = self.mapping.frame_index + self.lag
        n_rec = len(self.records)
        self.clipped = (raw < 0) | (raw >= n_rec)
        self.ridx = np.clip(raw, 0, n_rec - 1)

        # head 角速度 (相邻帧), 用于 plan §3.2 的误差归属
        self.ang_step_deg = head_angular_step(self.records, self.ridx)
        self.ang_vel = np.concatenate([[0.0], self.ang_step_deg]) * self.fps

        self._project_all()

    # ---- lag 从阶段 2 的产物读, 不在代码里写死 ----
    def read_lag(self) -> int:
        p = os.path.join(OUT, f"align_{self.stem}.json")
        if not os.path.exists(p):
            raise SystemExit(f"缺少阶段 2 产物 {p}, 请先跑 python -m xrhand.align")
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        lag = d["lag"]
        self.lag_meta = {
            "path": p,
            "method": lag.get("method"),
            "best_lag_frames": lag.get("best_lag_frames"),
            "best_r": lag.get("best_r"),
            "ci_lag_frames": lag.get("ci_lag_frames"),
            "measure_sign_version": d.get("geometry_convention", {}).get(
                "measure_sign_version"
            ),
        }
        return int(lag["best_lag_frames"])

    def _project_all(self):
        n = self.n_frames
        self.U = np.full((n, 2, sk.NUM_JOINTS), np.nan)
        self.V = np.full((n, 2, sk.NUM_JOINTS), np.nan)
        self.Z = np.full((n, 2, sk.NUM_JOINTS), np.nan)
        self.HH = np.full(n, np.nan)  # 手-头距离中位 (head 系原点到关节的距离)
        self.VALID = np.zeros((n, sk.NUM_JOINTS), dtype=bool)

        for f in range(n):
            r = self.records[self.ridx[f]]
            pos = r.right.pos
            self.VALID[f] = r.right.valid_mask
            self.HH[f] = float(np.median(np.linalg.norm(pos - r.head_pos, axis=1)))
            for half in range(2):
                eye = self.proj.eye_of_half(half)
                u, v, z, p_head = self.proj.project(
                    pos, r.head_pos, r.head_quat, eye
                )
                self.U[f, half] = u
                self.V[f, half] = v
                self.Z[f, half] = z

        self.inside = (
            (self.Z > 0)
            & (self.U >= R.MARGIN)
            & (self.U < EYE_W - R.MARGIN)
            & (self.V >= R.MARGIN)
            & (self.V < EYE_H - R.MARGIN)
        )
        # 可判读的点 = 追踪有效 且 落在画面内
        self.usable = self.VALID[:, None, :] & self.inside
        self.in_frac = self.usable.mean(axis=2)  # (n, 2)

    # ---- 二值爪: 5 点几何 + 夹爪 6DoF + 开/闭二值 ----
    def compute_gripper(
        self,
        *,
        close_hi: float = gr.DEFAULT_CLOSE_HI,
        open_lo: float = gr.DEFAULT_OPEN_LO,
        min_range: float = gr.MIN_RANGE_M,
        median_window: int = gr.DEFAULT_MEDIAN_WINDOW,
        confirm_ticks: int = gr.DEFAULT_CONFIRM_TICKS,
        min_state_ticks: int = gr.DEFAULT_MIN_STATE_TICKS,
        smooth_alpha: float = gr.DEFAULT_SMOOTH_ALPHA,
    ) -> dict:
        """算出夹爪那条链路, 时间轴与现有 U/V/Z/VALID 完全一致 (同一个 ridx/lag)。

        只在 main() 里 mode != skeleton 时调用: `xrseg/hand_overlay.py:77` 是
        `Reel(stem, lag, params)` **位置传参**建 Reel 的, 所以这里不能挂进 __init__,
        也不能新增必填位置参数。

        新数组一律**另起一套** (G* 前缀), 不往 U/V/Z/VALID 里塞: usable / in_frac
        被 auto_frames、xrseg 的凸包诊断、_crop_zoom 共用, 动它们会连带改掉
        `--mode skeleton` 的逐像素输出。
        """
        n = self.n_frames
        n_axis = 4  # 中点 + x/y/z 三个轴端点
        self.GU = np.full((n, 2, n_axis), np.nan)
        self.GV = np.full((n, 2, n_axis), np.nan)
        self.GZ = np.full((n, 2, n_axis), np.nan)
        self.G_MID = np.full((n, 3), np.nan)
        self.G_R = np.full((n, 3, 3), np.nan)
        self.G_PINCH = np.full(n, np.nan)
        self.G_KEY_VALID = np.zeros((n, len(gr.KEYPOINTS)), dtype=bool)
        self.G_POSE_VALID = np.zeros(n, dtype=bool)
        self.G_SMOOTH_ALPHA = float(smooth_alpha)  # HUD/终端要显示, 免得事后不知道跑的哪个值

        idx = list(gr.KEYPOINTS)
        prev_R = None  # 上一帧的 **opt** 朝向 (HumanEgo 的 mid_prev_R = mid_R_opt)
        x_ema = None
        y_ema = None
        for f in range(n):
            r = self.records[self.ridx[f]]
            pos = r.right.pos  # 世界系
            kv = r.right.valid_mask[idx]
            self.G_KEY_VALID[f] = kv
            if not kv.all():
                continue  # 5 点不全 -> 这一帧不产生夹爪 (点仍然照画, 由 draw_gripper 标灰)
            self.G_PINCH[f] = gr.pinch_distance(pos)
            self.G_MID[f] = gr.midpoint(pos)
            R_raw = gr.midpoint_frame(pos, prev_R)
            if R_raw is None:
                # 上游 AriaHands.py:369 的兜底: 先退上一帧, 连上一帧都没有就用腕的朝向
                R_raw = prev_R if prev_R is not None else quat_to_mat(r.right.quat[gr.WRIST])
            # HumanEgo opt: 对 x/y 基向量 EMA 平滑再正交化 (AriaHandsOptimizer.py:267-274)
            R, x_ema, y_ema = gr.smooth_midpoint_frame(
                R_raw, x_ema=x_ema, y_ema=y_ema, alpha=smooth_alpha
            )
            prev_R = R  # 只在几何可用的帧上更新 (存的是 opt 结果, 不是 raw)
            self.G_R[f] = R
            self.G_POSE_VALID[f] = True
            pts = gr.axis_points(self.G_MID[f], R)
            for half in range(2):
                eye = self.proj.eye_of_half(half)
                u, v, z, _ = self.proj.project(pts, r.head_pos, r.head_quat, eye)
                self.GU[f, half] = u
                self.GV[f, half] = v
                self.GZ[f, half] = z

        # 投票的帧: 5 点齐 + 追踪记录没被 clip + 在记录跨度内。lag 会把开头几帧
        # (111300 有 5 帧、111342 有 4 帧) 折到 record 0 上, 那几帧是同一个采样
        # 重复出现的, 不该按 confirm_ticks 的次数参与确认。
        self.G_JUDGE = self.G_POSE_VALID & ~self.clipped & self.mapping.in_span

        self.G_CLOSURE, self.G_CLOSED, self.G_CALIB = gr.grasp_from_pinch(
            self.G_PINCH,
            self.G_JUDGE,
            close_hi=close_hi,
            open_lo=open_lo,
            median_window=median_window,
            confirm_ticks=confirm_ticks,
            min_state_ticks=min_state_ticks,
            min_range=min_range,
        )
        # 只用于终端对照: ego_relation_policy 给 PICO 自己的固定 0.035 m 判据
        self.G_CLOSED_METRIC = gr.grasp_from_pinch_metric(self.G_PINCH, self.G_JUDGE)
        return self.G_CALIB

    # ---- 取一帧的统计 ----
    def stats(self, f: int) -> dict:
        return {
            "frame": int(f),
            "record": int(self.ridx[f]),
            "lag": int(self.lag),
            "in_span": bool(self.mapping.in_span[f]) and not bool(self.clipped[f]),
            "clipped": bool(self.clipped[f]),
            "residual_us": float(self.mapping.residual_us[f]),
            "in_frac_eye0": float(self.in_frac[f, 0]),
            "in_frac_eye1": float(self.in_frac[f, 1]),
            "median_depth_m_eye0": float(np.nanmedian(self.Z[f, 0])),
            "hand_head_m": float(self.HH[f]),
            "head_ang_vel_deg_s": float(self.ang_vel[f]),
            "is_active": int(self.records[self.ridx[f]].right.is_active),
        }

    def lag_note(self) -> str:
        m = getattr(self, "lag_meta", None)
        if not m:
            return f"lag={self.lag} (命令行指定)"
        return (
            f"lag={self.lag} 帧 (来自 {os.path.basename(m['path'])}, "
            f"{m['method']}, r={m['best_r']:.4f}, CI={m['ci_lag_frames']})"
        )


# ---------------------------------------------------------------- 选帧


def auto_frames(reel: Reel, n: int, min_in_frac: float = 0.35) -> list[int]:
    """按**手-头距离**分层抽样: 覆盖不同深度, 才能看出偏移是否随深度变化。"""
    ok = np.nonzero((reel.in_frac[:, 0] >= min_in_frac) & np.isfinite(reel.HH))[0]
    if ok.size == 0:
        # 一个都不够格: 退化成按画面内点数的前若干帧
        ok = np.argsort(-reel.in_frac[:, 0])[: max(n * 3, 20)]
        ok = np.sort(ok)
    d = reel.HH[ok]
    order = np.argsort(d)
    pick = order[np.linspace(0, order.size - 1, min(n, order.size)).round().astype(int)]
    return sorted(int(ok[i]) for i in pick)


# ---------------------------------------------------------------- 画


def draw_frame(
    reel: Reel,
    frame_rgb: np.ndarray,
    f: int,
    radius: int = 5,
    inset: bool = True,
    *,
    mode: str = "skeleton",
) -> Image.Image:
    """一帧的叠加图。

    `mode` 是**仅关键字**参数 (在 inset 之后), 因为原有调用是
    `draw_frame(reel, fr, i, args.radius, inset=False)` 这样位置传参的:
      - skeleton: 26 点灵巧手, 行为与加这个参数之前逐像素一致 (默认值);
      - gripper : 只画 5 点二值爪 + 夹爪坐标架, 放大图按 5 点裁剪;
      - both    : 两套都画, 放大图仍按 26 点裁剪。
    """
    img = Image.fromarray(frame_rgb).convert("RGBA")
    d = ImageDraw.Draw(img)
    st = reel.stats(f)
    r = reel.records[reel.ridx[f]]
    font = R._font(15)

    # 二值爪模式下画的是**同一批关键点**, 只是从 26 个里取 5 个 —— 直接让
    # draw_hand 本体去画 (传 joints/edges 收窄), 不另写一份绘制代码, 这样配色
    # (sk.JOINT_COLOR)、半径规则、画外红三角、无效点空心圈全都与原方法一致。
    # joints 只有 5 个点, edges 是拇指/食指的完整 9 段折线 (gr.BONES, 含掌骨段) ——
    # 折线沿手指穿过 2/4/6/8/9 但不给它们画点, 其余三根手指整根不画。
    # `both` 模式仍画全 26 点, 夹爪那层叠在上面; `grip` 为假 (没跑 compute_gripper)
    # 时也退回全 26 点, 免得画面上空无一物。
    grip = mode in ("gripper", "both") and getattr(reel, "G_CLOSED", None) is not None
    two_fingers = grip and mode == "gripper"   # 只有纯 gripper 模式才收窄成这两根手指

    # --- 右手: 两半各画一遍 (左半用 eye_of_half(0), 右半用 1) ---
    for half in range(2):
        R.draw_hand(
            d,
            reel.U[f, half],
            reel.V[f, half],
            reel.Z[f, half],
            reel.VALID[f],
            half,
            radius=radius,
            alpha_fill=True,
            img_for_poly=img,
            **({"joints": gr.KEYPOINTS, "edges": gr.BONES} if two_fingers else {}),
        )
        R.draw_center_cross(d, half)

    # --- 二值爪: 指尖中点(夹爪位置) + 6DoF 坐标架 + 状态字 ---
    #     几何/判据见 xrhand/gripper.py (HumanEgo 的 5 点 + ego_relation_policy 的判据)。
    #     左手在两条采集里都 isActive=0, 所以这段只对右手算 —— 与 _project_all 一致。
    #     5 个关键点本身已由上面的 draw_hand 画好, 这里只管夹爪独有的那几样。
    if grip:
        idx = list(gr.KEYPOINTS)
        for half in range(2):
            R.draw_gripper(
                d,
                half,
                key_u=reel.U[f, half][idx],
                key_v=reel.V[f, half][idx],
                key_z=reel.Z[f, half][idx],
                key_valid=reel.G_KEY_VALID[f],
                mid_u=reel.GU[f, half, 0],
                mid_v=reel.GV[f, half, 0],
                mid_z=reel.GZ[f, half, 0],
                mid_valid=bool(reel.G_POSE_VALID[f]),
                axis_u=reel.GU[f, half, 1:],
                axis_v=reel.GV[f, half, 1:],
                axis_z=reel.GZ[f, half, 1:],
                axis_valid=bool(reel.G_POSE_VALID[f]),
                closed=bool(reel.G_CLOSED[f]),
                pinch_m=float(reel.G_PINCH[f]),
                closure=float(reel.G_CLOSURE[f]),
            )

    # --- 左手: 实测 isActive=0 / s 全 0, 整体按无效渲染 (plan §1.5) ---
    #     画出来是为了让"没识别到"这件事在画面上可见, 而不是缺一块。
    lv = r.left.valid_mask
    if not lv.any():
        for half in range(2):
            x0 = half * EYE_W
            txt = "L-hand: isActive=%d  not tracked" % r.left.is_active
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                d.text((x0 + 12 + dx, EYE_H - 34 + dy), txt, font=font, fill=(0, 0, 0))
            d.text((x0 + 12, EYE_H - 34), txt, font=font, fill=R.COLOR_INVALID)

    # --- HUD ---
    lines = [
        f"frame {f}/{reel.n_frames - 1}   record {st['record']}   lag {reel.lag}",
        f"t resid {st['residual_us']:+.1f} us" + ("  [OUT OF SPAN]" if not st["in_span"] else ""),
        f"R isActive={st['is_active']}  pts {int(reel.VALID[f].sum())}/26",
        f"in-frame L/R {st['in_frac_eye0'] * 100:.0f}% / {st['in_frac_eye1'] * 100:.0f}%",
        f"depth(eye0) med {st['median_depth_m_eye0']:.3f} m",
        f"|hand-head| {st['hand_head_m']:.3f} m",
        f"head ang vel {st['head_ang_vel_deg_s']:.1f} deg/s",
        R.gauge(min(st["head_ang_vel_deg_s"] / 120.0, 1.0), 24),
    ]
    if grip:
        # 加在原有 8 行**之后**: 前面几行的内容和行号都不动, skeleton 模式逐像素不变
        cal = reel.G_CALIB
        # 把 5 个点的**实际关节下标**写出来: 指根是近节(Proximal), 不是名字更像的
        # 掌骨(Metacarpal) —— 这两者差一节, 写清楚免得以后再对错。
        lines += [
            f"GRIPPER 5pt W(1)/TB:prox(3)/IB:prox(7)/T(5)/I(10)  "
            f"{'2-finger polyline' if two_fingers else 'overlay'}  "
            f"pinch {reel.G_PINCH[f]:.4f} m  closure {reel.G_CLOSURE[f]:.3f}",
            f"  {'CLOSED' if reel.G_CLOSED[f] else 'OPEN  '}  close_hi {cal['close_hi']:.2f} -> "
            f"{cal['close_distance_m']:.4f} m   open_lo {cal['open_lo']:.2f} -> "
            f"{cal['open_distance_m']:.4f} m",
            f"  gizmo XYZ len {gr.AXIS_LENGTHS[0]:.2f}/{gr.AXIS_LENGTHS[1]:.2f}/{gr.AXIS_LENGTHS[2]:.2f} m"
            f"   smooth_alpha {reel.G_SMOOTH_ALPHA:.2f}"
            f"   keypts {int(reel.G_KEY_VALID[f].sum())}/5"
            f"   judge {'on' if reel.G_JUDGE[f] else 'off'}",
        ]
    for half in range(2):
        R.draw_hud(d, lines, half, font)

    if not inset:
        out = img.convert("RGB")
    else:
        # --- 放大图: 不放大很难判断"贴合", 1080 宽里差 5 px 肉眼看不出来 ---
        panels = []
        for half in range(2):
            c = _crop_zoom(
                img, reel, f, half, joints=gr.KEYPOINTS if two_fingers else None
            )
            if c is not None:
                panels.append(c)
        if not panels:
            out = img.convert("RGB")
        else:
            ph = 360
            panels = [
                p.resize((max(1, int(p.width * ph / p.height)), ph), Image.LANCZOS)
                for p in panels
            ]
            W = sum(p.width for p in panels) + 8 * (len(panels) + 1)
            canvas = Image.new(
                "RGB", (max(W, img.width), img.height + ph + 8), (20, 20, 20)
            )
            canvas.paste(img.convert("RGB"), (0, 0))
            x = 8
            for p in panels:
                canvas.paste(p, (x, img.height + 4))
                x += p.width + 8
            out = canvas

    if mode != "skeleton" and getattr(reel, "G_CLOSED", None) is not None:
        out = _stack_panel(reel, f, out)
    return out


def _stack_panel(reel: Reel, f: int, image: Image.Image) -> Image.Image:
    """把判据波形面板贴到画面下方 (整幅宽, 与 VideoWriter 的输出宽度对齐)。"""
    panel = R.draw_grasp_panel(
        max(image.width, EYE_W * 2),
        frame=f,
        distance=reel.G_PINCH,
        closure=reel.G_CLOSURE,
        closed=reel.G_CLOSED,
        calibration=reel.G_CALIB,
        fps=reel.fps,
    )
    canvas = Image.new("RGB", (panel.width, image.height + panel.height), (20, 20, 20))
    canvas.paste(image, (0, 0))
    canvas.paste(panel, (0, image.height))
    return canvas


def _crop_zoom(
    img: Image.Image, reel: Reel, f: int, half: int, pad: int = 70, *, joints=None
):
    """按手在画面上的外接框裁一块放大。

    `joints=None` (默认) 用 26 点全部 —— 与加这个参数之前逐像素一致;
    gripper 模式传 5 点, 免得被手指/掌心的框撑大而看不出夹爪。
    **不要**把 6DoF 轴端点算进外接框: 它们会伸出去十几厘米, 框一撑开手就小了。
    """
    if joints is None:
        m = reel.usable[f, half]
        u_all = reel.U[f, half]
        v_all = reel.V[f, half]
    else:
        idx = list(joints)
        m = reel.usable[f, half][idx]
        u_all = reel.U[f, half][idx]
        v_all = reel.V[f, half][idx]
    if m.sum() < 2:
        return None
    x0 = half * EYE_W
    u = u_all[m]
    v = v_all[m]
    x1 = int(max(x0, x0 + np.floor(u.min()) - pad))
    x2 = int(min(x0 + EYE_W, x0 + np.ceil(u.max()) + pad))
    y1 = int(max(0, np.floor(v.min()) - pad))
    y2 = int(min(EYE_H, np.ceil(v.max()) + pad))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return img.crop((x1, y1, x2, y2)).convert("RGB")


# ---------------------------------------------------------------- main


def _print_grasp_table(reel: Reel, c: dict, mode: str) -> None:
    """二值爪的标定/判据对照表: 看完再决定要不要调 --grasp-* 阈值。"""
    print()
    print(f"    二值爪 (mode={mode}): 5 点 = 腕/拇指根/食指根/拇指尖/食指尖 "
          f"(由 draw_hand 按原配色原半径画, 只是收窄了 joints/edges); "
          f"夹爪 = 两指尖中点 + MidpointFrameBuilder 的 6DoF 朝向")
    print(f"      实际关节 (PICO 26 点下标): W(1) / TB:THUMB_PROXIMAL(3) / "
          f"IB:INDEX_PROXIMAL(7) / T:THUMB_TIP(5) / I:INDEX_TIP(10)")
    print(f"      与 HumanEgo 的对应: Aria ThumbMCP(6) = PICO 3, Aria IndexMCP(8) = PICO 7 "
          f"—— Aria 的 21 点模型没有掌骨, 它的 MCP 就是 PICO 的近节, **不是** 2/6")
    print(f"      画面上的线: 拇指 (1-2-3-4-5) 与食指 (1-6-7-8-9-10) 的完整折线 "
          f"({len(gr.BONES)} 段), 折线穿过的 2/4/6/8/9 不画点")
    print(f"      朝向平滑: EMA alpha {reel.G_SMOOTH_ALPHA:.2f} "
          f"(HumanEgo opt 路径 AriaHandsOptimizer.py:267-274, x/y 基向量各一次; "
          f"1.0 = 不平滑)")
    print(f"      标定 (每条采集各自标定): q95 张开基准 {c['open_baseline_m']:.4f} m   "
          f"q10 闭合参考 {c['closed_reference_m']:.4f} m   动态范围 "
          f"{c['dynamic_range_m']:.4f} m (下限 {c['min_range_m']:.3f} m)")
    print(f"      阈值: close_hi {c['close_hi']:.2f} -> 判闭合于 {c['close_distance_m']:.4f} m"
          f"   open_lo {c['open_lo']:.2f} -> 判张开于 {c['open_distance_m']:.4f} m")
    print(f"      时间常数: median {c['median_window']} 帧  confirm {c['confirm_ticks']} 帧  "
          f"min_state {c['min_state_ticks']} 帧 (都按相机帧 @ {reel.fps:g} fps)")
    if c["nan_filled_frames"]:
        print(f"      [注意] {c['nan_filled_frames']} 帧指尖距是 NaN, 已用中位 "
              f"{c['nan_fill_value_m']:.4f} m 填上 (这些帧不参与投票)")
    if c["dynamic_range_too_flat"]:
        print(f"      [注意] 动态范围不足 {c['min_range_m']:.3f} m -> 整条采集恒判张开")
    clipped = int((reel.clipped | ~reel.mapping.in_span).sum())
    print(f"      参与投票 {int(reel.G_JUDGE.sum())}/{reel.n_frames} 帧 "
          f"(被 clip/超跨度 {clipped} 帧, 5 点不全 {int((~reel.G_POSE_VALID).sum())} 帧)")
    for name, state in (
        ("自适应 (画面用这个)", reel.G_CLOSED),
        (f"固定 {gr.METRIC_CLOSE_M:.3f} m + x{gr.METRIC_OPEN_FACTOR:g} 滞回 "
         f"(ego_relation_policy 的 pico 兜底, 只做对照)",
         reel.G_CLOSED_METRIC),
    ):
        s = gr.summarize(np.asarray(state))
        print(f"      {name}\n        闭合 {s['closed_frames']}/{s['frames']} "
              f"({s['closed_ratio'] * 100:.0f}%)  段 {s['runs_text']}")
    print("      注: 首/尾段不会被吸收 (上游 absorb_short_runs 刻意保留被截断的边界段), "
          "所以片头那段闭合是数据里本来就有的, 不是 bug")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="标称参数裸叠加 (Step 2)")
    ap.add_argument("stem", nargs="?", default="20260920_111300")
    ap.add_argument("--frames", default=None, help="逗号分隔的帧号")
    ap.add_argument("--n-auto", type=int, default=8)
    ap.add_argument("--lag", type=int, default=None, help="覆盖阶段 2 的 lag")
    ap.add_argument("--calib", default=None,
                    help="calib.json 路径 (默认 out/calib.json, 与 xrseg/hand_overlay.py 同约定)")
    ap.add_argument("--no-calib", action="store_true",
                    help="显式退回标称参数 (只做对照; 骨架会明显偏离手)")
    ap.add_argument("--radius", type=int, default=5)
    ap.add_argument("--video", action="store_true", help="渲染全片 mp4")
    ap.add_argument(
        "--mode",
        choices=("skeleton", "gripper", "both"),
        default="gripper",
        help="skeleton=26 点灵巧手 (原行为) / gripper=5 点二值爪 (默认) / both",
    )
    ap.add_argument("--grasp-close-hi", type=float, default=gr.DEFAULT_CLOSE_HI,
                    help="闭合成度超过它 -> 想闭合 (默认 0.70, 本仓库实测选定)")
    ap.add_argument("--grasp-open-lo", type=float, default=gr.DEFAULT_OPEN_LO,
                    help="闭合成度低于它 -> 想张开 (默认 0.60)")
    ap.add_argument("--grasp-min-range", type=float, default=gr.MIN_RANGE_M,
                    help="指尖距动态范围下限 (m), 低于它整条采集恒判张开")
    ap.add_argument("--grasp-median-window", type=int, default=gr.DEFAULT_MEDIAN_WINDOW,
                    help="中值滤波窗口 (正奇数, 相机帧)")
    ap.add_argument("--grasp-confirm-ticks", type=int, default=gr.DEFAULT_CONFIRM_TICKS,
                    help="状态要连续多少帧都'想要'才改")
    ap.add_argument("--grasp-min-state-ticks", type=int, default=gr.DEFAULT_MIN_STATE_TICKS,
                    help="吸收内部短于这么多帧的状态段")
    ap.add_argument("--grasp-smooth-alpha", type=float, default=gr.DEFAULT_SMOOTH_ALPHA,
                    help="夹爪朝向 EMA 平滑因子, 照 HumanEgo opt 路径 (默认 0.15; "
                         "1.0 = 不平滑, 只做对照)")
    ap.add_argument("--outdir", default=OUT)
    args = ap.parse_args(argv)

    # --- 标定: 默认就要读 out/calib.json ---
    # 不读标定的话 extra_t 全零 (camera.py:145 的 head_to_cam), 深度会差 5.24 cm,
    # 骨架和关键点整体偏 37~163 px —— 就是之前"关键点没贴合手"的真正原因。
    # 这里跟 xrseg/hand_overlay.py:62 对齐: 默认读, 读不到直接退出 (而不是静默用标称值),
    # 想标称跑得显式加 --no-calib。
    params = CameraParams()
    if args.no_calib:
        if args.calib:
            ap.error("--no-calib 和 --calib 不能同时给")
        print("[警告] --no-calib: 用标称参数 (extra_t=0), 骨架/关键点会明显偏离手, 仅供对照")
    else:
        cj_path = args.calib or os.path.join(args.outdir, "calib.json")
        if not os.path.isfile(cj_path):
            raise SystemExit(
                f"缺 {cj_path} —— 用标称参数画出来的骨架会明显偏, 不如不画; "
                f"先跑 python tools/make_calib.py, 或用 --no-calib 显式退回标称参数"
            )
        with open(cj_path, encoding="utf-8") as f:
            cj = json.load(f)
        params = CameraParams.from_json(cj["params"])
        extra_t = np.asarray(params.extra_t, dtype=float)
        print(f"已载入标定参数 {cj_path}"
              f" (d_f={params.d_f:+.3f}, d_cx={params.d_cx:+.3f}, d_cy={params.d_cy:+.3f},"
              f" extra_t=[{extra_t[0]:+.4f}, {extra_t[1]:+.4f}, {extra_t[2]:+.4f}] m)")

    reel = Reel(args.stem, args.lag, params)
    print(f"=== {reel.stem}: {reel.n_frames} 帧 @ {reel.fps} fps, "
          f"{len(reel.records)} 条记录")
    print(f"    {reel.lag_note()}")
    print(f"    flip_head_quat={params.flip_head_quat}  f={params.eff_f:.3f} "
          f"c=({params.eff_cx:.1f},{params.eff_cy:.1f})")

    # ---- 二值爪: 只在需要时算 (skeleton 模式一律不算, draw_frame 里也不会碰它) ----
    if args.mode != "skeleton":
        calib = reel.compute_gripper(
            close_hi=args.grasp_close_hi,
            open_lo=args.grasp_open_lo,
            min_range=args.grasp_min_range,
            median_window=args.grasp_median_window,
            confirm_ticks=args.grasp_confirm_ticks,
            min_state_ticks=args.grasp_min_state_ticks,
            smooth_alpha=args.grasp_smooth_alpha,
        )
        _print_grasp_table(reel, calib, args.mode)

    os.makedirs(args.outdir, exist_ok=True)

    if args.frames:
        frames = [int(x) for x in args.frames.split(",")]
    else:
        frames = auto_frames(reel, args.n_auto)

    # ---- 判据表: 偏移是否随深度变化 ----
    print()
    print(f"{'frame':>6} {'rec':>6} {'depth':>7} {'hh':>6} {'in%':>5} {'angv':>7} "
          f"{'u-med':>8} {'v-med':>7}  {'r_span':>7}")
    for f in frames:
        st = reel.stats(f)
        m = reel.usable[f, 0]
        um = float(np.median(reel.U[f, 0][m])) if m.any() else float("nan")
        vm = float(np.median(reel.V[f, 0][m])) if m.any() else float("nan")
        span = (
            float(reel.U[f, 0][m].max() - reel.U[f, 0][m].min()) if m.any() else float("nan")
        )
        print(f"{f:>6} {st['record']:>6} {st['median_depth_m_eye0']:>7.3f} "
              f"{st['hand_head_m']:>6.3f} {st['in_frac_eye0'] * 100:>4.0f}% "
              f"{st['head_ang_vel_deg_s']:>6.1f}d {um:>8.1f} {vm:>7.1f}  {span:>7.1f}")

    # 整段统计: "常数偏移"还是"随深度变化", 看这两行的对照
    print()
    print(f"    全片 hand-head 距离: {np.nanmin(reel.HH):.3f} ~ {np.nanmax(reel.HH):.3f} m "
          f"(中位 {np.nanmedian(reel.HH):.3f})")
    print(f"    全片 eye0 在画面内点比例: 中位 {np.median(reel.in_frac[:, 0]) * 100:.1f}%"
          f"   eye1: {np.median(reel.in_frac[:, 1]) * 100:.1f}%")
    print(f"    超出记录跨度被 clip 的帧: {int(reel.clipped.sum())} / {reel.n_frames}")
    print(f"    head 角速度: 中位 {np.median(reel.ang_vel):.1f}  p95 "
          f"{np.percentile(reel.ang_vel, 95):.1f}  max {reel.ang_vel.max():.1f} deg/s")

    # ---- 写静帧 ----
    want = set(frames)
    # 产物名按 mode 分开, 免得 gripper 模式覆盖磁盘上已有的 overlay_<stem>.mp4
    tag = {"skeleton": "overlay", "gripper": "gripper", "both": "gripper_both"}[args.mode]
    panel_h = R.PANEL_H if args.mode != "skeleton" else 0
    if args.video:
        vout = os.path.join(args.outdir, f"{tag}_{reel.stem}.mp4")
        # 视频高度 = 810 + 判据面板高。放大图仍然不进视频 (两半的裁剪框宽度随手掌
        # 大小变, 会把画布撑成不定宽而破坏 CFR 输出), 但面板是定高的, 可以进。
        # VideoWriter 建好就改不了尺寸, 所以高度必须先在这里算好。
        w = VideoWriter(vout, EYE_W * 2, EYE_H + panel_h, reel.fps)
        print(f"\n    渲染全片 ({tag}, {EYE_W * 2}x{EYE_H + panel_h}) -> {vout}")
    else:
        w = None

    for i, fr in enumerate(iter_frames(reel.mp4, EYE_W * 2, EYE_H)):
        # ffmpeg 管道按输出尺寸给帧; 原始就是 2160x810, 不需要缩放
        if w is not None:
            w.write(
                np.asarray(
                    draw_frame(reel, fr, i, args.radius, inset=False, mode=args.mode)
                )
            )
        elif i in want:
            p = os.path.join(args.outdir, f"{tag}_{reel.stem}_f{i:04d}.png")
            draw_frame(reel, fr, i, args.radius, inset=True, mode=args.mode).save(p)
            print(f"    -> {p}")

    if w is not None:
        w.close()
        print(f"    -> {vout} ({os.path.getsize(vout) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
