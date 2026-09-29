"""帧对齐: txt 的 90 Hz 记录流 <-> mp4 的 30 fps CFR 帧。

两个候选时钟 (plan §2.2), 语义完全不同:

  predictTime = PXR_Enterprise.GetPredictedDisplayTime() * 1000   [微秒]
      源码 TrackingData.cs:64 注释: "微秒，对应camera录制中帧插入的时间戳"
      -> 这是全仓库唯一一句关于"追踪数据与相机录制如何对齐"的官方说明, 故作主时钟。
      实测严格 90 Hz (dt 中位 11111.2 us, 抖 ±26 us)。

  timeStampNs = (DateTime.UtcNow - 1970)ms * 1e6                   [UTC 墙钟纳秒]
      非单调, 实测 dt 中位 11.09 ms, 抖动 7.5~16.8 ms。作独立交叉校验。

锚点 (plan §2.1): header.notice 原文
  "This is the timestamp and head pose information when obtaining the image for the first frame."
  -> 视频第 0 帧 <-> header.timeStampNs (墙钟)
  注意 header 实际并没有 head pose 字段, notice 措辞与内容有出入。

两个时钟用记录内同时出现的 (timeStampNs, predictTime) 做线性拟合互相换算。
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .io_tracking import Header, Record, quat_to_mat, record_arrays

# 测量端符号约定的版本号。改动 image_motion / image_motion_vec / phase_correlate
# 的符号语义时必须同步 +1 —— 它决定 lag 的正负号往哪边解释, 是个会静默出错的量。
# v1: phase_correlate 返回 "把 b 拉回 a 所需的位移" = −内容位移;
#     image_motion / image_motion_vec 在上层取负, 对外统一为**内容位移**。
MEASURE_SIGN_VERSION = 1


# ---------------------------------------------------------------- 时钟换算


def fit_clock_map(records: list[Record]) -> tuple[float, float, float]:
    """用记录内同时出现的两个时钟拟合 predict_us ~ a + b * ts_ms。

    返回 (a, b, rms_residual_us)。b 应非常接近 1000 (两钟同速率)。
    若 b 明显偏离 1000, 说明墙钟有漂移/跳变, 需要报警而不是忽略。
    """
    a = record_arrays(records)
    x = a["ts_ns"].astype(np.float64) / 1e6  # ms
    y = a["predict_us"]
    b, a0 = np.polyfit(x, y, 1)
    res = y - (a0 + b * x)
    return float(a0), float(b), float(np.sqrt(np.mean(res**2)))


@dataclass
class Mapping:
    """逐帧的追踪记录映射结果。"""

    frame_index: np.ndarray  # (n_frames,) 选中的记录下标
    t_query_us: np.ndarray  # (n_frames,) 该帧按映射算出的 predictTime
    t_record_us: np.ndarray  # (n_frames,) 选中记录的 predictTime
    residual_us: np.ndarray  # (n_frames,) 二者之差 (最近邻误差)
    clock: str
    offset_s: float
    rate: float
    fps: float
    in_span: np.ndarray  # (n_frames,) 该帧的时刻是否落在记录时间跨度内

    def to_json(self) -> dict:
        r = np.abs(self.residual_us)
        ok = self.in_span
        return {
            "clock": self.clock,
            "offset_s": self.offset_s,
            "rate": self.rate,
            "fps": self.fps,
            "n_frames": int(self.frame_index.size),
            "n_out_of_span": int((~ok).sum()),
            "out_of_span_frames": np.nonzero(~ok)[0].tolist(),
            # 只有落在记录跨度内的帧, 残差才有"最近邻误差"的意义;
            # 跨度外的帧是被 clip 到端点的, 单列出来不做统计
            "residual_us_median": float(np.median(r[ok])) if ok.any() else None,
            "residual_us_max": float(r[ok].max()) if ok.any() else None,
            "residual_us_max_out_of_span": float(r[~ok].max()) if (~ok).any() else None,
            "frame_index": self.frame_index.tolist(),
            "t_record_us": self.t_record_us.tolist(),
            "residual_us": self.residual_us.tolist(),
        }


def build_mapping(
    header: Header,
    records: list[Record],
    n_frames: int,
    fps: float,
    clock: str = "predict",
    offset_s: float = 0.0,
    rate: float = 1.0,
) -> Mapping:
    """把每一视频帧映射到最近的一条追踪记录。

    t_n = t_anchor + (n * rate + offset_s* fps) * (1/fps)   [按给定时钟]
    其中 t_anchor 是视频第 0 帧在该时钟上的时刻。
    为避免浮点累积, 直接写成秒域:  t_n[s] = t_anchor_s + n*rate/fps + offset_s
    """
    a = record_arrays(records)

    if clock == "predict":
        a0, b, _ = fit_clock_map(records)
        t_anchor_us = a0 + b * (header.time_stamp_ns / 1e6)
        t_axis_us = a["predict_us"]
    elif clock == "ts":
        t_anchor_us = header.time_stamp_ns / 1e3
        t_axis_us = a["ts_ns"].astype(np.float64) / 1e3
    else:
        raise ValueError(f"未知时钟 {clock!r}, 应为 'predict' 或 'ts'")

    n = np.arange(n_frames, dtype=np.float64)
    t_query = t_anchor_us + (n * rate / fps + offset_s) * 1e6

    # 视频比追踪区间长 (plan §2.1: StopRecord 先关 writer 再停预览),
    # 所以两端的帧会落到记录跨度之外。这些帧没有对应记录, 必须显式标出,
    # 不能让最近邻 clip 把"根本没数据"伪装成"残差有点大"。
    in_span = (t_query >= t_axis_us[0]) & (t_query <= t_axis_us[-1])

    idx = np.searchsorted(t_axis_us, t_query).clip(1, len(t_axis_us) - 1)
    left = np.abs(t_axis_us[idx - 1] - t_query)
    right = np.abs(t_axis_us[idx] - t_query)
    idx = np.where(left <= right, idx - 1, idx)

    return Mapping(
        frame_index=idx,
        t_query_us=t_query,
        t_record_us=t_axis_us[idx],
        residual_us=t_query - t_axis_us[idx],
        clock=clock,
        offset_s=offset_s,
        rate=rate,
        fps=fps,
        in_span=in_span,
    )


# ---------------------------------------------------------------- 图像全局运动


def _gray(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def phase_correlate(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """相位相关求 b 相对 a 的平移, **带亚像素峰值细化**。

    返回 (dx, dy, peak), 已做环绕解包。加 Hann 窗抑制边缘效应。

    ⚠ **符号: 返回的是「把 b 拉回 a 所需的位移」= −(a→b 的内容位移)**, 不是内容位移本身。
    这一点此前被写反过, 直接导致了"head 四元数约定"的假冲突, 所以这里写死:

        内容位移 = −(dx, dy)

    钉死方式 ("亮点质心"判据, 不依赖任何符号约定):
        A[40,30]=1; B = ndi.shift(A, (0, +5))  ->  亮点 30.00 -> 35.00
        内容位移 = +5.00,  而 phase_correlate(A,B) 返回 dx = −5.000

    需要"内容位移"语义的调用方请用 image_motion / image_motion_vec, 它们已经取过负。

    亚像素是必需的, 不是锦上添花: 下采样到 0.25 后整像素对应全分辨率的
    4 px, 而实测图像位移中位仅 ~5 px —— 不细化的话信号会被量化到 {0,4,8,...},
    把相关性彻底毁掉 (实测表现: r 随 lag 单调变化、没有峰)。
    细化用峰值 3x3 邻域的抛物线顶点。
    """
    h, w = a.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    A = np.fft.rfft2((a - a.mean()) * win)
    B = np.fft.rfft2((b - b.mean()) * win)
    cross = A * np.conj(B)
    mag = np.abs(cross)
    cross = np.where(mag < 1e-12, 0, cross / np.maximum(mag, 1e-12))
    r = np.fft.irfft2(cross, s=(h, w))
    pk = int(np.argmax(r))
    iy, ix = np.unravel_index(pk, r.shape)

    def _sub(vals):  # vals = (前, 峰, 后)
        d = vals[0] - 2 * vals[1] + vals[2]
        return 0.0 if abs(d) < 1e-12 else float(np.clip(0.5 * (vals[0] - vals[2]) / d, -1, 1))

    sy = _sub([r[(iy - 1) % h, ix], r[iy, ix], r[(iy + 1) % h, ix]])
    sx = _sub([r[iy, (ix - 1) % w], r[iy, ix], r[iy, (ix + 1) % w]])

    dy, dx = float(iy) + sy, float(ix) + sx
    if dy > h / 2:
        dy -= h
    if dx > w / 2:
        dx -= w
    return dx, dy, float(r.max())


def image_motion(
    video_path: str, n_frames: int, scale: float = 0.25, half: int = 0
) -> np.ndarray:
    """逐帧图像全局平移幅度 (第 i 项 = 第 i-1 -> i 帧的内容位移大小, 第 0 项恒 0)。

    half: 0=左半幅, 1=右半幅。只取单眼, 避免立体接缝干扰。

    幅度无符号, 所以 phase_correlate 的符号问题在这里**看不出来**
    (这也正是那个 bug 长期没被发现的原因)。但仍然显式取负, 与
    image_motion_vec 保持同一语义 —— 别让"碰巧对"变成"看起来对"。
    """
    from .camera import EYE_W
    from .video import iter_frames

    out = np.zeros(n_frames, dtype=np.float64)
    prev = None
    for i, fr in enumerate(iter_frames(video_path, 2160, 810, scale=scale)):
        if i >= n_frames:
            break
        x0 = int(half * EYE_W * scale)
        g = _gray(fr[:, x0 : x0 + int(EYE_W * scale)])
        if prev is not None:
            dx, dy, _ = phase_correlate(prev, g)
            out[i] = np.hypot(dx, dy) / scale  # 换回全分辨率像素
        prev = g
    return out


# ---------------------------------------------------------------- lag 估计


def head_angular_step(
    records: list[Record], frame_index: np.ndarray
) -> np.ndarray:
    """按给定帧映射采样 head 姿态, 求相邻帧之间的旋转角 (度)。"""
    q = np.stack([records[i].head_quat for i in frame_index])
    R = np.stack([quat_to_mat(x) for x in q])
    rel = np.einsum("nij,nkj->nik", R[1:], R[:-1])
    cos = np.clip((np.trace(rel, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def unproject_pixel(
    proj,
    u: float,
    v: float,
    head_pos: np.ndarray,
    head_quat: np.ndarray,
    eye: int = 0,
    dist: float = 100.0,
) -> np.ndarray:
    """把单眼像素 (u,v) 反投影成世界点, 该点在**相机系**下深度为 dist 米。

    存在的意义: 让"远点网格"的构造**与四元数约定无关**。
    之前的写法是用 R0 把 head 系方向转到世界系 —— 但那一步本身就用到了
    被检验的约定, 于是"预测与实测反号"可能只是构造不自洽的产物, 不构成证据。
    反投影是投影的逆, 无论约定取哪个都严格往返, 所以得到的世界点是**物理**的:
    它在第 0 条记录的姿态下恰好落在指定的像素上。

    注意深度必须定义在**相机系**, 不能定义成"距 head 原点 dist 米"。
    早先的实现在末尾做了 `head_pos + 单位方向 * dist`, 结果往返差 ~55 px / 5°:
    因为相机相对 head 有约 8 cm 平移, 投影链 `p_cam = R(p_dev − t)` 对
    "绕 head 的纯方向" **不是尺度不变的** —— 从 head 出发的同一条射线
    与从相机出发的同一条射线打在不同像素上。深度定义在相机系才与像素一一对应。
    """
    p = proj.p
    p_cam = np.array([(u - p.eff_cx) / p.eff_f, (v - p.eff_cy) / p.eff_f, 1.0]) * dist
    E = proj.E[eye]
    R, t = E[:3, :3], E[:3, 3]
    # head_to_cam 的逆
    if p.extrinsic_subtract_t:
        p_dev = p_cam @ R + t
    else:
        p_dev = (p_cam - t) @ R
    # extra 刚体变换的逆 (正向: p_dev = (p_head @ Rz.T − extra_t) @ extra_R.T)
    p_dev = p_dev @ np.asarray(p.extra_R) + np.asarray(p.extra_t)
    p_head = p_dev @ proj.Rz  # 正向: p_dev = p_head @ Rz.T
    # world_to_head 的逆
    R_h = quat_to_mat(head_quat)
    if p.flip_head_quat:
        R_h = R_h.T
    return p_head @ R_h.T + head_pos


def project_world_grid(
    header: Header,
    records: list[Record],
    params=None,
    dist: float = 1000.0,
    n_grid: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """一组**固定在世界系**的远点, 逐条记录求出其像素坐标。

    构造方式: 在第 0 条记录的画面上取一个覆盖视场的像素网格, 逐个**反投影**
    成世界点, 之后这些世界点固定不动。这样:
      - 点跟着头转的退化情况不会发生 (点在世界系里是静止的);
      - 构造过程不含"把 head 系方向转到世界系"这一步, 因此**与四元数约定无关**
        (反投影是投影的逆, 任何约定下都严格往返)。

    距离取 1000 m (定义在**相机系**, 见 unproject_pixel): head 自身平移
    (整段 span ~0.1 m) 造成的视差降到 ~0.08 px, 相对实测中位位移 ~3 px 可忽略,
    信号几乎纯由旋转决定; 但本函数仍做**完整投影**(含平移), 不引入"纯旋转"近似。
    (取 100 m 时视差 ~0.8 px, 占信号的 1/4, 偏大。)

    返回 (U, V, Z, P_world), 前三者形状 (n_records, G)。
    """
    from .camera import EYE_H, EYE_W, make_projector

    proj = make_projector(header, params)
    r0 = records[0]
    us = np.linspace(0.12, 0.88, n_grid) * EYE_W
    vs = np.linspace(0.12, 0.88, n_grid) * EYE_H
    P = [
        unproject_pixel(proj, float(u), float(v), r0.head_pos, r0.head_quat, eye=0, dist=dist)
        for v in vs
        for u in us
    ]
    P_world = np.stack(P)

    n, G = len(records), P_world.shape[0]
    U = np.zeros((n, G))
    V = np.zeros((n, G))
    Z = np.zeros((n, G))
    for i, r in enumerate(records):
        u, v, z, _ = proj.project(P_world, r.head_pos, r.head_quat, eye=0)
        U[i], V[i], Z[i] = u, v, z
    return U, V, Z, P_world


def head_image_shift(
    U: np.ndarray,
    V: np.ndarray,
    Z: np.ndarray,
    frame_index: np.ndarray,
    lag: int = 0,
    params=None,
) -> np.ndarray:
    """由 head 姿态预测的图像全局平移 (dx, dy), 长度 = len(frame_index)-1。

    对每个远方向求"在第 n-1 帧姿态下看到的像素"到"在第 n 帧姿态下看到的像素"的位移,
    只在两帧都可见且在画面内的方向上取平均 —— 这正是全局平移的定义。

    符号约定与 phase_correlate **一致**: 都是"内容位移"(同一套亮点质心自检钉死)。
    但前提是 head 四元数的旋转方向取对 (CameraParams.flip_head_quat) ——
    取反的话这里会与实测恰好反号 (实测互相关: 正取 r=+0.94, 取反 r=−0.90)。
    """
    from .camera import EYE_H, EYE_W

    fix = np.clip(frame_index + lag, 0, U.shape[0] - 1)
    i0, i1 = fix[:-1], fix[1:]
    u0, v0, z0 = U[i0], V[i0], Z[i0]
    u1, v1, z1 = U[i1], V[i1], Z[i1]
    vis = (
        (z0 > 0) & (z1 > 0)
        & (u0 > 0) & (u0 < EYE_W) & (v0 > 0) & (v0 < EYE_H)
    )
    du, dv = u1 - u0, v1 - v0
    w = vis.astype(np.float64)
    cnt = w.sum(axis=1)
    cnt = np.where(cnt < 1, 1, cnt)[:, None]
    dx = (np.where(vis, du, 0.0)).sum(axis=1) / cnt[:, 0]
    dy = (np.where(vis, dv, 0.0)).sum(axis=1) / cnt[:, 0]
    out = np.stack([dx, dy], axis=1)
    out[cnt[:, 0] < 1] = np.nan  # 没有共同可见方向 -> 该帧无预测
    return out


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson 相关, 长度必须相同。"""
    if a.size < 3:
        return float("nan")
    sa, sb = a.std(), b.std()
    if sa < 1e-12 or sb < 1e-12:
        return float("nan")
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def estimate_lag(
    header: Header,
    records: list[Record],
    image_motion_mag: np.ndarray,
    n_frames: int,
    fps: float,
    clock: str = "predict",
    lag_range: tuple[int, int] = (-15, 15),
    rate_range: np.ndarray | None = None,
    min_motion_pct: float = 60.0,
) -> dict:
    """用两个独立信号互相关估计视频相对追踪数据的时间偏移。

    信号 A: head 角速度 (由 head 四元数在视频帧时刻采样后求相邻帧角步长)
    信号 B: 图像全局运动幅度 (相位相关)

    lag 的含义: 正的 lag = 视频帧 n 实际对应追踪记录的 n+lag
               (即视频比追踪"晚" lag 帧)。

    plan §2.4 指出直接 argmax 会被尖峰驱动、峰很宽, 所以这里:
      1) 只在 head 运动较大的帧上算相关 (min_motion_pct 分位阈值), 提高信噪比;
      2) 报告整个相关曲线与置信区间 (相关系数 >= max - 0.02 的连续区间), 而不是裸 argmax;
      3) 同时扫 rate, 检验 30 fps 是否有漂移。
    """
    b_img = np.asarray(image_motion_mag, dtype=np.float64)
    n_use = min(n_frames, b_img.size)
    b_img = b_img[:n_use]

    # 只用运动足够大的帧, 避免静止段把相关拉平
    thr = np.percentile(b_img, min_motion_pct)
    use = b_img >= thr

    fw = np.arange(n_use, dtype=np.float64)  # 用于按 lag 平移
    del fw

    if rate_range is None:
        rate_range = np.array([1.0])

    best = None
    curve_by_rate: dict[str, list] = {}
    for rate in rate_range:
        lags = np.arange(lag_range[0], lag_range[1] + 1)
        rs = []
        for lag in lags:
            idx = np.arange(n_use) + lag
            ok = (idx >= 0) & (idx < len(records))
            if ok.sum() < 10:
                rs.append(np.nan)
                continue
            m = build_mapping(header, records, n_use, fps, clock=clock, rate=float(rate))
            # 用平移后的记录下标序列重算角步长
            fix = m.frame_index.copy()
            fix[ok] = m.frame_index[ok] + lag
            fix = np.clip(fix, 0, len(records) - 1)
            ang = head_angular_step(records, fix)
            mask = use[1:] & ok[1:]
            if mask.sum() < 10:
                rs.append(np.nan)
                continue
            rs.append(_corr(ang[mask], b_img[1:][mask]))
        rs = np.array(rs)
        curve_by_rate[f"{rate:.5f}"] = rs.tolist()

        k = int(np.nanargmax(rs))
        if best is None or rs[k] > best["r"]:
            best = {
                "r": float(rs[k]),
                "lag": int(lags[k]),
                "rate": float(rate),
                "lags": lags,
                "curve": rs,
            }

    assert best is not None
    lags, rs = best["lags"], best["curve"]
    rmax = best["r"]
    # 置信区间: 相关系数在 max-0.02 以内的 lag 构成的连续区间 (取含峰的那段)
    near = np.abs(rs - rmax) <= 0.02
    k = int(np.nanargmax(rs))
    lo = k
    while lo - 1 >= 0 and near[lo - 1]:
        lo -= 1
    hi = k
    while hi + 1 < len(lags) and near[hi + 1]:
        hi += 1

    # 质心估计 (对峰附近做加权平均), 比 argmax 更稳
    seg = rs[lo : hi + 1]
    seg = np.where(np.isfinite(seg), seg, 0.0)
    w = np.maximum(seg - np.nanmin(seg), 0.0)
    centroid = (
        float(np.sum(lags[lo : hi + 1] * w) / np.sum(w)) if np.sum(w) > 1e-9 else float(best["lag"])
    )

    return {
        "clock": clock,
        "best_lag_frames": int(best["lag"]),
        "best_lag_ms": float(best["lag"] / fps * 1000),
        "best_r": rmax,
        "ci_lag_frames": [int(lags[lo]), int(lags[hi])],
        "ci_lag_ms": [float(lags[lo] / fps * 1000), float(lags[hi] / fps * 1000)],
        "centroid_lag_frames": centroid,
        "centroid_lag_ms": float(centroid / fps * 1000),
        "best_rate": best["rate"],
        "curve": {f"{int(l)}": (None if not np.isfinite(v) else round(float(v), 4))
                  for l, v in zip(lags, rs)},
        "curve_by_rate": {k2: [None if not np.isfinite(v) else round(float(v), 4) for v in vv]
                          for k2, vv in curve_by_rate.items()},
        "n_frames_used": int(n_use),
        "motion_threshold_px": float(thr),
    }


def image_motion_vec(
    video_path: str, n_frames: int, scale: float = 0.25, half: int = 0
) -> np.ndarray:
    """逐帧图像全局平移**矢量** (dx, dy), 单位全分辨率像素, 形状 (n_frames, 2)。

    第 i 项 = 第 i-1 -> i 帧的**内容位移**, 第 0 项恒 (0,0)。符号与
    head_image_shift 的预测**同约定** (都是内容位移), 可直接比相关。

    ⚠ 这里必须取负: 底层 phase_correlate 返回的是"把当前帧拉回上一帧所需的位移",
    即内容位移的**相反数**。早先忘了取负, 结果预测与实测恒为反号, 被误读成
    "head 四元数旋转方向取反", 白查了一轮。标量幅度版无符号所以掩盖了这个 bug。

    比标量幅度多保留了方向 —— head 绕光轴 roll 时图像是**旋转**而非平移,
    标量幅度会把这种帧误当成"几乎没动"; 且平移量 ~ f·tan(θ) 只在**小角度**下
    才正比于旋转角。两个缺陷都是换成矢量后才消失的。
    """
    from .camera import EYE_W
    from .video import iter_frames

    out = np.zeros((n_frames, 2), dtype=np.float64)
    prev = None
    for i, fr in enumerate(iter_frames(video_path, 2160, 810, scale=scale)):
        if i >= n_frames:
            break
        x0 = int(half * EYE_W * scale)
        g = _gray(fr[:, x0 : x0 + int(EYE_W * scale)])
        if prev is not None:
            dx, dy, _ = phase_correlate(prev, g)
            out[i] = (-dx / scale, -dy / scale)  # ★ 取负 = 内容位移
        prev = g
    return out


def estimate_lag_vec(
    header: Header,
    records: list[Record],
    meas_shift: np.ndarray,
    n_frames: int,
    fps: float,
    clock: str = "predict",
    lag_range: tuple[int, int] = (-15, 15),
    min_motion_px: float = 1.0,
    min_motion_pct: float = 60.0,
    dist: float = 1000.0,
    params=None,
) -> dict:
    """用**二维平移矢量**互相关估计 lag, 取代标量幅度版本。

    lag 的含义与 estimate_lag 一致: 正 lag = 视频帧 n 对应追踪记录 n+lag。

    `meas_shift` 必须是**内容位移**(image_motion_vec 的输出)。若误传
    phase_correlate 的原始返回值(差一个负号), 最优 lag 会镜像到相反一侧 ——
    这正是之前"lag 是 −9 还是 +9"说不清的根源。
    """
    from .camera import EYE_W

    meas = np.asarray(meas_shift, dtype=np.float64)
    n_use = min(n_frames, meas.shape[0])
    meas = meas[:n_use]
    U, V, Z, _ = project_world_grid(header, records, params=params, dist=dist)
    m = build_mapping(header, records, n_use, fps, clock=clock)

    # 只在图像**确实动了**的帧上算相关。用「分位阈值 + 像素下限」:
    #   只取分位 -> 整段静止时会把纯噪声帧当信号; 只取像素下限 -> head 中位角速度
    #   仅 1.55°/s (≈0.6 px/帧), 通过阈值的帧太少。两者取大即可。
    mag = np.hypot(meas[:, 0], meas[:, 1])
    thr = max(float(min_motion_px), float(np.percentile(mag, min_motion_pct)))
    use = mag >= thr
    if use.sum() < 20:  # 整段几乎静止, 退化为取运动最大的若干帧并标注
        kk = min(30, mag.size)
        use = np.zeros(mag.shape, dtype=bool)
        use[np.argsort(mag)[-kk:]] = True

    lags = np.arange(lag_range[0], lag_range[1] + 1)
    rs = []
    for lag in lags:
        pred = head_image_shift(U, V, Z, m.frame_index, lag=lag, params=params)
        mask = use[1:] & (mag[1:] > 0) & np.isfinite(pred).all(axis=1)
        if mask.sum() < 20:
            rs.append(np.nan)
            continue
        a = pred[mask].reshape(-1)
        b = meas[1:][mask].reshape(-1)
        rs.append(_corr(a, b))
    rs = np.array(rs)

    k = int(np.nanargmax(rs))
    rmax = float(rs[k])
    near = np.abs(rs - rmax) <= 0.02
    lo = hi = k
    while lo - 1 >= 0 and near[lo - 1]:
        lo -= 1
    while hi + 1 < len(lags) and near[hi + 1]:
        hi += 1
    seg = np.where(np.isfinite(rs[lo : hi + 1]), rs[lo : hi + 1], 0.0)
    w = np.maximum(seg - np.nanmin(seg), 0.0)
    centroid = (
        float(np.sum(lags[lo : hi + 1] * w) / np.sum(w)) if np.sum(w) > 1e-9 else float(lags[k])
    )

    return {
        "method": "2d_vector",
        "clock": clock,
        "best_lag_frames": int(lags[k]),
        "best_lag_ms": float(lags[k] / fps * 1000),
        "best_r": rmax,
        "ci_lag_frames": [int(lags[lo]), int(lags[hi])],
        "ci_lag_ms": [float(lags[lo] / fps * 1000), float(lags[hi] / fps * 1000)],
        "centroid_lag_frames": centroid,
        "centroid_lag_ms": float(centroid / fps * 1000),
        "curve": {f"{int(l)}": (None if not np.isfinite(v) else round(float(v), 4))
                  for l, v in zip(lags, rs)},
        "n_frames_used": int(use.sum()),
        "n_samples": int((use[1:] & (mag[1:] > 0)).sum() * 2),
        "motion_threshold_px": float(thr),
        "median_image_motion_px": float(np.median(mag)),
        "half_width_px": float(EYE_W),
    }


def save_json(obj: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def lag_signals(
    header: Header,
    records: list[Record],
    image_motion_mag: np.ndarray,
    n_frames: int,
    fps: float,
    clock: str = "predict",
    lag: int = 0,
    rate: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """取出 (head 角步长, 图像运动幅度, 有效掩膜), 已按 lag 平移并对齐长度。

    主要用于画图复核 —— 把两条信号叠在一起看它们是否真的同步。
    """
    b = np.asarray(image_motion_mag, dtype=np.float64)
    n = min(n_frames, b.size)
    b = b[:n]
    m = build_mapping(header, records, n, fps, clock=clock, rate=rate)
    fix = np.clip(m.frame_index + lag, 0, len(records) - 1)
    ang = head_angular_step(records, fix)
    use = b >= np.percentile(b, 60.0)
    return ang, b, use


def plot_lag(
    res: dict,
    ang: np.ndarray,
    img: np.ndarray,
    use: np.ndarray,
    fps: float,
    out_png: str,
    title: str,
) -> None:
    """画 lag 相关曲线 + 两条信号的对齐叠图, 供人工复核 (plan §2.4)。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lags = np.array(sorted(int(k) for k in res["curve"]))
    r = np.array([res["curve"][str(int(l))] for l in lags], dtype=np.float64)
    lo, hi = res["ci_lag_frames"]
    best = res["best_lag_frames"]
    cen = res["centroid_lag_frames"]

    fig, ax = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)

    a0 = ax[0]
    a0.axvspan(lo - 0.5, hi + 0.5, color="tab:orange", alpha=0.18,
               label=f"CI (r >= max-0.02): [{lo}, {hi}] frames")
    a0.plot(lags, r, "o-", color="tab:blue", lw=1.6, ms=4, label="Pearson r")
    a0.axhline(0, color="k", lw=0.6, alpha=0.4)
    a0.axvline(best, color="tab:red", lw=1.4, ls="--",
               label=f"argmax = {best} frames ({best / fps * 1000:+.1f} ms), r = {res['best_r']:.4f}")
    a0.axvline(cen, color="tab:green", lw=1.4, ls=":",
               label=f"centroid = {cen:.2f} frames ({res['centroid_lag_ms']:+.1f} ms)")
    a0.axvline(0, color="tab:purple", lw=1.0, alpha=0.8, label="lag 0 (header anchor as-is)")
    a0.set_xlabel("lag [frames]   (positive = video frame n matches tracking record n+lag)")
    a0.set_ylabel("Pearson r")
    a0.set_title(f"lag correlation: predicted vs measured image shift ({res.get('method', '?')})"
                 + (f"   |   r(0) = {res['curve'].get('0', float('nan')):+.4f}" if "0" in res["curve"] else ""))
    a0.grid(alpha=0.3)
    a0.legend(fontsize=8, loc="lower left")

    # 两条信号在最佳 lag 下叠图 (各自归一化到 0..1)
    a1 = ax[1]
    n = min(ang.size, img.size - 1)
    x = np.arange(1, n + 1)

    def _norm(v):
        v = np.asarray(v, dtype=np.float64)
        lo_, hi_ = np.nanmin(v), np.nanmax(v)
        return (v - lo_) / (hi_ - lo_) if hi_ > lo_ else v * 0

    na = _norm(ang[:n])
    ni = _norm(img[1 : n + 1])
    a1.plot(x, na, lw=1.0, color="tab:blue", label="head angular step [deg/frame] (normalized)")
    a1.plot(x, ni, lw=1.0, color="tab:red", alpha=0.8,
            label="image global motion [px/frame] (normalized)")
    sel = use[1 : n + 1]
    a1.plot(x[sel], ni[sel], ".", color="tab:red", ms=3,
            label="frames used for r (top 40% motion)")
    a1.set_xlabel("video frame index")
    a1.set_ylabel("normalized")
    a1.set_title(f"signals at lag = {best} frames  (both normalized to 0..1)")
    a1.grid(alpha=0.3)
    a1.legend(fontsize=8, loc="upper right")

    # best_rate 只有标量版(会扫 rate)才有; 矢量版没有这一项, 用 .get 兜住。
    rate_s = f" | rate={res['best_rate']:.5f}" if "best_rate" in res else ""
    # 图里只用 ASCII: 环境里的 DejaVu Sans 没有中文字形, 中文会渲染成豆腐块。
    sc = res.get("cross_check_scalar")
    sc_s = (f"\ncross-check (scalar magnitude): argmax {sc['best_lag_frames']:+d} frames "
            f"r={sc['best_r']:.4f} CI {sc['ci_lag_frames']}" if sc else "")
    fig.suptitle(
        f"{title}\nbest lag {best} frames = {best / fps * 1000:+.1f} ms | "
        f"CI [{lo}, {hi}] | centroid {cen:.2f} frames | r = {res['best_r']:.4f}{rate_s}{sc_s}",
        fontsize=11,
    )
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- 阶段 2 驱动


def _stem(path: str) -> str:
    from pathlib import Path

    return Path(path).stem.replace("trackingData_", "")


def main(argv: list[str] | None = None) -> int:
    """阶段 2: 对每组 (txt, mp4) 产出 align_<stem>.json 与 lag_<stem>.png。

    mp4 路径默认按 CameraRecord_<stem>.mp4 在 txt 同目录下找。
    """
    import argparse
    import os
    from pathlib import Path

    ap = argparse.ArgumentParser(description="阶段 2: txt <-> mp4 帧对齐 + lag 估计")
    ap.add_argument("txt", nargs="+")
    ap.add_argument("--video", action="append", default=None,
                    help="与 --txt 一一对应; 省略则按 CameraRecord_<stem>.mp4 推断")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--scale", type=float, default=0.25, help="相位相关的下采样比")
    ap.add_argument("--lag-range", type=int, nargs=2, default=(-15, 15))
    ap.add_argument("--scan-rate", action="store_true",
                    help="额外扫描 rate (检验 30 fps 是否有漂移), 慢")
    args = ap.parse_args(argv)

    from .camera import CameraParams
    from .io_tracking import load
    from .video import count_frames, probe

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    vids = args.video
    if vids is None:
        vids = []
        for p in args.txt:
            cand = Path(p).parent / f"CameraRecord_{_stem(p)}.mp4"
            if not cand.exists():
                raise SystemExit(f"找不到视频 {cand}, 请用 --video 指定")
            vids.append(str(cand))
    if len(vids) != len(args.txt):
        raise SystemExit("--video 的个数必须与 --txt 一致")

    rate_range = np.array([1.0, 0.9995, 1.0005]) if args.scan_rate else None
    summary = {}

    for txt, vid in zip(args.txt, vids):
        stem = _stem(txt)
        print(f"\n=== {stem} ===")
        header, records = load(txt)
        info = probe(vid)
        fps = float(info["fps"])
        n_frames = count_frames(vid)
        span = len(records) / 89.9989
        print(f"  视频 {vid}: {info['width']}x{info['height']} @ {fps:g} fps, {n_frames} 帧, "
              f"{info['duration_s']:.4f} s; 记录 {len(records)} 条 ({span:.4f} s)")
        print(f"  时长差 视频−追踪 = {info['duration_s'] - span:+.4f} s")

        a0, b, rms = fit_clock_map(records)
        print(f"  时钟拟合 predict_us = {a0:.3f} + {b:.6f}·ts_ms, rms {rms:.1f} µs "
              f"({(b - 1000) * 1e3:+.0f} ppm)")

        # 两个时钟各算一遍逐帧映射 —— 互为交叉校验 (plan §2.2)
        maps = {}
        for clk in ("predict", "ts"):
            m = build_mapping(header, records, n_frames, fps, clock=clk)
            maps[clk] = m
            r = np.abs(m.residual_us)[m.in_span]
            nout = int((~m.in_span).sum())
            oos = np.nonzero(~m.in_span)[0]
            print(f"  [{clk:7s}] 跨度内逐帧残差 中位 {np.median(r):7.1f} µs, 最大 {r.max():7.1f} µs; "
                  f"跨度外 {nout} 帧" + (f" (帧 {oos.min()}..{oos.max()})" if nout else "")
                  + f", frame(200)={int(m.frame_index[min(200, n_frames - 1)])}")
        # 只在"两钟都在跨度内"的帧上比较 —— 跨度外的帧都被 clip 到同一个端点,
        # 那种"一致"是假的一致。
        both = maps["predict"].in_span & maps["ts"].in_span
        d = maps["predict"].frame_index != maps["ts"].frame_index
        n_diff = int((d & both).sum())
        same = n_diff == 0
        print(f"  两钟一致(跨度内 {int(both.sum())} 帧): {same}  不一致 {n_diff} 帧")
        if n_diff:
            bad = np.nonzero(d & both)[0]
            dd = (maps["predict"].frame_index - maps["ts"].frame_index)[bad]
            vals = sorted(set(dd.tolist()))
            benign = set(vals) <= {-1, 1}
            print(f"    不一致帧下标范围 {bad.min()}..{bad.max()}, "
                  f"记录下标差 predict−ts: 取值 {vals}, "
                  + ("全部为 ±1 条记录 —— 两钟的 t_anchor 差不足半个记录间隔 (5.6 ms), "
                     "最近邻在相邻两条之间摆动。**良性**: 一个视频帧 33.3 ms, "
                     "1 条记录 11.1 ms, 不改变对齐结论, 但按 plan §2.4 显式标注, 不静默二选一。"
                     if benign else
                     "**存在大于 1 条的差异, 需查**"))
            pr = np.abs(maps["predict"].residual_us)[bad]
            tr = np.abs(maps["ts"].residual_us)[bad]
            print(f"    这些帧上 predict 残差中位 {np.median(pr):.0f} µs vs ts 残差中位 {np.median(tr):.0f} µs")

        print(f"  读视频做相位相关 (单眼, scale={args.scale}) ...")
        img_motion = image_motion(vid, n_frames, scale=args.scale, half=0)
        img_vec = image_motion_vec(vid, n_frames, scale=args.scale, half=0)
        # 一致性自检: 两者的幅度应完全一致(同一遍解码的同一批 dx,dy), 只差符号语义
        dmag = np.abs(np.hypot(img_vec[:, 0], img_vec[:, 1]) - img_motion)
        print(f"  标量/矢量自洽: max|‖vec‖ − scalar| = {dmag.max():.3e} px"
              + ("  OK" if dmag.max() < 1e-9 else "  ** 不一致, 查 **"))

        # 主估计: 2D 矢量 (标量幅度对绕光轴的 roll 不敏感, 见 plan §2.3)
        lag = estimate_lag_vec(header, records, img_vec, n_frames, fps,
                               clock="predict", lag_range=tuple(args.lag_range))
        print(f"  lag(矢量): argmax {lag['best_lag_frames']:+d} 帧 ({lag['best_lag_ms']:+.1f} ms) "
              f"r={lag['best_r']:.4f}; CI {lag['ci_lag_frames']}; "
              f"质心 {lag['centroid_lag_frames']:.2f} 帧 ({lag['centroid_lag_ms']:+.1f} ms)")
        # 交叉校验: 标量幅度版 (符号无关, 所以它一直是对的; 但分辨力差)
        lag_sc = estimate_lag(header, records, img_motion, n_frames, fps,
                              clock="predict", lag_range=tuple(args.lag_range),
                              rate_range=rate_range)
        print(f"  lag(标量, 仅交叉校验): argmax {lag_sc['best_lag_frames']:+d} 帧 "
              f"r={lag_sc['best_r']:.4f}; CI {lag_sc['ci_lag_frames']}; "
              f"质心 {lag_sc['centroid_lag_frames']:.2f} 帧")
        lag["cross_check_scalar"] = {
            "best_lag_frames": lag_sc["best_lag_frames"],
            "best_r": lag_sc["best_r"],
            "ci_lag_frames": lag_sc["ci_lag_frames"],
            "centroid_lag_frames": lag_sc["centroid_lag_frames"],
        }

        ang, _, use = lag_signals(header, records, img_motion, n_frames, fps,
                                  clock="predict", lag=lag["best_lag_frames"])
        png = outdir / f"lag_{stem}.png"
        plot_lag(lag, ang, img_motion, use, fps, str(png),
                 f"{stem}  ({n_frames} frames @ {fps:g} fps)")

        npy = outdir / f"imgmotion_{stem}.npy"
        np.save(npy, img_motion)

        doc = {
            "stem": stem,
            "txt": os.path.abspath(txt),
            "video": os.path.abspath(vid),
            "video_info": info,
            "n_frames": int(n_frames),
            "n_records": len(records),
            "tracking_span_s": span,
            "duration_gap_s": info["duration_s"] - span,
            "header_time_stamp_ns": int(header.time_stamp_ns),
            "header_minus_first_record_ms": float(
                (header.time_stamp_ns - records[0].time_stamp_ns) / 1e6),
            "clock_fit": {"a0_us": a0, "b": b, "rms_us": rms, "ppm": (b - 1000) * 1e3},
            "mapping": {clk: maps[clk].to_json() for clk in maps},
            "mapping_clocks_agree": same,
            "mapping_n_frames_differ": n_diff,
            "lag": lag,
            "geometry_convention": {
                "flip_head_quat": bool(CameraParams().flip_head_quat),
                "meaning": "False = 数据四元数是 head->world (标准 OpenXR), "
                           "p_head = Rᵀ(p_world − t)。判定依据见 plan §0.2/§0.3。",
                "measure_sign_version": MEASURE_SIGN_VERSION,
            },
            "artifacts": {"lag_png": str(png), "imgmotion_npy": str(npy)},
            "notes": {
                "anchor": "frame0 <-> header.timeStampNs 只是 §2.1 论证过的「作者意图」初值; "
                          "实际以 lag 结论为准",
                "lag_sign": "正 lag = 视频帧 n 对应追踪记录 n+lag; "
                            "实测为负 => 视频内容比标称时刻旧 (plan §0.5)",
                "ci": "CI 是 r >= max−0.02 的连续区间, 宽度 7~9 帧 => "
                      "**不能声称 lag 精度优于 ±0.15 s**",
            },
        }
        save_json(doc, str(outdir / f"align_{stem}.json"))
        print(f"  已写出 {outdir / f'align_{stem}.json'} 与 {png}")
        summary[stem] = {
            "n_frames": int(n_frames),
            "n_records": len(records),
            "duration_gap_s": doc["duration_gap_s"],
            "clocks_agree": same,
            "n_frames_differ": n_diff,
            "lag_frames": lag["best_lag_frames"],
            "lag_ms": lag["best_lag_ms"],
            "lag_r": lag["best_r"],
            "lag_ci_frames": lag["ci_lag_frames"],
            "lag_centroid_frames": lag["centroid_lag_frames"],
        }

    save_json(summary, str(outdir / "align_summary.json"))
    print(f"\n汇总 -> {outdir / 'align_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
