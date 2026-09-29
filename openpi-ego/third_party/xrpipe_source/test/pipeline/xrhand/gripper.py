"""二值爪 (binary gripper): 每只手 5 个关键点 -> 夹爪位置 + 朝向 + 开/闭二值。

用户要的两句话分别来自两份参考, 它们说的其实**不是**同一件事, 这里各取所需:

  HumanEgo-main (Aria MPS 21 点) —— 提供「5 个关键点 + 两指尖中点 + 朝向」
    5 点的取法      preprocess/AriaHands.py:346-374
    夹爪位置        AriaHands.py:359      midpoint_w = (thumb_tip + index_tip) / 2.0
    夹爪朝向        preprocess/AriaHandsTypes.py:40-129  MidpointFrameBuilder.build
    退化兜底        AriaHands.py:369      if R_mid is None: R_mid = prev_R if prev_R is not None else r_world
    画法/配色       preprocess/AriaHandsOps.py:1244-1384 (_draw_opt_wrist_thumb_index_only)
    坐标架          AriaHandsOps.py:1030-1097 (_draw_axis; 轴长 x 0.06 / y 0.10 / z 0.06 m)

  ego_relation_policy (PICO 26 点) —— 提供「怎么把连续量判成开/闭」
    中值滤波 + 每条采集分位数标定 + 确认式滞回 + 最小驻留
                    src/ego_relation/s1_pico_mode2/grasp.py:26-164
                    (_bool_runs / absorb_short_runs / confirmed_hysteresis / adaptive_index_grasp)
    距离兜底判据    同上的 pinch 距离, s2 里按 0.035 m 固定阈值 + ×1.35 张开门限

两份的关节顺序不同, 这里做一次重映射 (Aria 21 点 -> PICO 26 点)。
**注意 Aria 的 21 点模型没有掌骨(CMC)关节**, 每根手指从 MCP 开始, 而 PICO 的 26 点多一节
掌骨 —— 所以 Aria 的 "MCP" 要落到 PICO 的 **Proximal**, 不是名字更像的 Metacarpal:

    Aria 0 ThumbTip -> PICO 5   THUMB_TIP
    Aria 1 IndexTip -> PICO 10  INDEX_TIP
    Aria 5 Wrist    -> PICO 1   WRIST
    Aria 6 ThumbMCP -> PICO 3   THUMB_PROXIMAL   (不是 THUMB_METACARPAL=2)
    Aria 8 IndexMCP -> PICO 7   INDEX_PROXIMAL   (不是 INDEX_METACARPAL=6)

(依据: WiLoRHands.py:78-86 的 Aria 对照表 + 实测骨长, 详见下面 THUMB_BASE/INDEX_BASE 处。)

坐标系: 全部在**世界系**。`io_tracking.Hand.pos` 是世界系 (io_tracking.py:103),
`camera.Projector.project` 收的也是世界系 (camera.py:168 内部第一步才转到头系),
HumanEgo 那份的 docstring 也写明 "in World Space"。

本模块只做几何与判据, 不 import cv2 / PIL —— 与 `camera.py` 同级, 两个解释器都能跑
(系统 anaconda python 没有 cv2, 而 `tools/overlay.py` 就是用它跑的)。
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from . import skeleton as sk

# ---------------------------------------------------------------- 5 个关键点

# 顺序 = HumanEgo 的绘制顺序 (腕、拇指根、食指根、拇指尖、食指尖), 下标取自 skeleton.py
WRIST = sk.WRIST  # 1
# 两个"指根"必须取**近节(Proximal)**, 不是掌骨(Metacarpal) —— 这是本文件最容易搞错的一处:
# HumanEgo 用的是 Aria 21 点模型的 "ThumbMCP(6) / IndexMCP(8)"
# (AriaHandsTypes.py:81-129 的 `build`, docstring: "Uses MCP bases (index 6 and 8)"),
# 而 Aria 的 21 点模型**没有掌骨(CMC)关节**, 每根手指的链条从 MCP 开始
# (WiLoRHands.py:78-86 的对照表: 6=ThumbMCP, 7=ThumbIP, 8=IndexMCP, 9=IndexPIP, 10=IndexDIP)。
# OpenXR/PICO 的 26 点模型比它多一节掌骨, 所以:
#     Aria ThumbMCP  = PICO THUMB_PROXIMAL(3)   ← 不是名字更像的 THUMB_METACARPAL(2)
#     Aria IndexMCP  = PICO INDEX_PROXIMAL(7)   ← 不是 INDEX_METACARPAL(6)
# 实测佐证 (两条采集, 世界系): PICO 6->7 骨长 62.6 mm = 掌骨(全手最长的一根), 而 6 距腕只有
# 29.4 mm(掌内贴腕), 7 距腕 88.6 mm = 指节(真人掌指关节 85~95 mm); 拇指同理, 2 距腕 39 mm
# 在掌内, 3 距腕 66 mm = 拇指 MCP。
THUMB_BASE = sk.FINGERS["thumb"][1]  # 3 = THUMB_PROXIMAL = Aria 的 ThumbMCP
INDEX_BASE = sk.FINGERS["index"][1]  # 7 = INDEX_PROXIMAL = Aria 的 IndexMCP
THUMB_TIP = sk.TIPS[0]  # 5
INDEX_TIP = sk.TIPS[1]  # 10

KEYPOINTS = (WRIST, THUMB_BASE, INDEX_BASE, THUMB_TIP, INDEX_TIP)  # (1, 3, 7, 5, 10)
# 只用于终端打印/HUD 说明。**画面上的 5 个点不写标签** —— 它们由 draw_hand 按原方法
# 绘制 (原方法本来就不给关节点写字母), 用户要求"完全照原样"。
# TB/IB 里的 "B" = base(指根), 指的是**近节**; 终端表里会把实际关节下标一起打出来。
KEYPOINT_NAMES = ("W", "TB", "IB", "T", "I")

# 5 点内部的局部下标 (KEYPOINTS 的位置), 画夹爪连线时用
K_WRIST, K_TBASE, K_IBASE, K_TTIP, K_ITIP = range(5)

# 画多少骨架: 拇指与食指这两根的**完整链条**(含掌骨段), 直接从既有拓扑 sk.EDGES
# 过滤得到, 不硬编码 —— 结果 (1,2),(2,3),(3,4),(4,5),(1,6),(6,7),(7,8),(8,9),(9,10) 共 9 段。
# sk.EDGES 本来是 "(WRIST, 各掌骨) + 相邻对 + (PALM, WRIST)", 过滤后自动排除 (PALM, WRIST)
# 与其余三根手指。
# 注意: 折线会穿过 2/4/6/8/9 这几个关节, 但**只给 KEYPOINTS 那 5 个画圆点**(用户选定)。
_TWO_FINGERS = {sk.WRIST, *sk.FINGERS["thumb"], *sk.FINGERS["index"]}
BONES = tuple(
    (a, b) for a, b in sk.EDGES if a in _TWO_FINGERS and b in _TWO_FINGERS
)

# 坐标架三条轴的长度 (m), AriaHandsOps.py:1035-1037
AXIS_LENGTHS = (0.06, 0.10, 0.06)

# MidpointFrameBuilder 的鲁棒性阈值 (AriaHandsTypes.py:52-54)
EPS_NORM = 1e-6
EPS_ARM = 1e-5
EPS_Y = 1e-5

# 距离量纲下的动态范围下限 (m)。上游 grasp_robust_minimum_range=0.12 是**归一化电机
# 指令**量纲的, 搬到距离上会把整条采集清零 (实测动态范围只有 ~0.047 m), 所以改 0.010。
MIN_RANGE_M = 0.010

# 默认阈值: **本仓库实测选定**, 不是上游值 —— 上游 close_hi/open_lo = 0.35/0.25 卡在
# 归一化电机指令上, 搬到距离信号会判 71%/75% 的帧为闭合 (实测), 而本数据里
# "耳机壳可见" 的那几段对应的是 0.70/0.60 (闭合 33%/35%)。
DEFAULT_CLOSE_HI = 0.70
DEFAULT_OPEN_LO = 0.60

# 时间常数沿用上游 (configs/default.yaml:34 control_hz: 30.0 == 本仓库视频 30 fps,
# 所以 tick 常数可以直接按**相机帧**用)
DEFAULT_MEDIAN_WINDOW = 5
DEFAULT_CONFIRM_TICKS = 5
DEFAULT_MIN_STATE_TICKS = 12

# HumanEgo opt 路径对 midpoint 基向量 x/y 的 EMA 平滑因子 (AriaHands.yaml:25-26
# smooth_ema_alpha_x / smooth_ema_alpha_y 都是 0.15)。照搬上游默认, 不是本仓库实测。
DEFAULT_SMOOTH_ALPHA = 0.15

# ego_relation_policy 给 PICO 自己的固定距离判据 (s1_pico_mode2 + s2 的 pinch fallback):
# 闭合门限 0.035 m, 要张开得超过 0.035*1.35 = 0.047 m。仅用于终端对照表。
METRIC_CLOSE_M = 0.035
METRIC_OPEN_FACTOR = 1.35


# ---------------------------------------------------------------- 几何


def midpoint(pos: np.ndarray) -> np.ndarray:
    """夹爪位置 = 拇指指尖与食指指尖的中点 (AriaHands.py:359)。"""
    pos = np.asarray(pos, dtype=np.float64)
    return (pos[THUMB_TIP] + pos[INDEX_TIP]) / 2.0


def pinch_distance(pos: np.ndarray) -> float:
    """两指尖距离 (m)。小 = 闭合, 大 = 张开 —— 注意方向与上游信号相反。"""
    pos = np.asarray(pos, dtype=np.float64)
    return float(np.linalg.norm(pos[THUMB_TIP] - pos[INDEX_TIP]))


def _safe_normalize(v: np.ndarray, eps: float = EPS_NORM):
    n = float(np.linalg.norm(v))
    if n < eps:
        return None
    return v / n


def midpoint_frame(pos: np.ndarray, prev_R: np.ndarray | None = None):
    """夹爪朝向 (3,3, 列 = x/y/z), 逐字移植 MidpointFrameBuilder.build。

        x = normalize(index_base - thumb_base)     # 两指根连线, 即"手指张开方向"
        y = normalize((两指根中点 - 腕) 在 x 正交补上的投影)   # Gram-Schmidt
        z = x × y ;  y = z × x
        跨帧符号一致: prev_R[:,0]·x < 0 时把 x,y 取反, 重算 z

    两个 base 是**近节 Proximal(3/7)**, 即 Aria 的 ThumbMCP/IndexMCP —— 见文件头那段
    说明。用近节而不是指尖, 是因为捏合时指节几乎不动, x 轴才稳 (上游 docstring:
    "Uses MCP bases (index 6 and 8) to maintain a rigid X-axis during pinches")。

    任何一步退化 (模长 < eps) 都返回 `prev_R` —— 与上游一致 (`build` 返回 None 后,
    调用方 AriaHands.py:369 再兜一层 `prev_R if prev_R is not None else wrist_R`)。
    上游 `build()` 的 `midpoint_w` 形参**函数体内从未使用** (docstring 说 y 用
    midpoint-wrist, 实际用的是**两指根中点**-wrist, 因为指尖一捏就退化), 这里按代码走。

    返回 None 表示"这一帧没有可用朝向且也没有历史", 调用方自己决定兜底。
    """
    pos = np.asarray(pos, dtype=np.float64)
    x = _safe_normalize(pos[INDEX_BASE] - pos[THUMB_BASE])
    if x is None:
        return prev_R

    arm = (pos[THUMB_BASE] + pos[INDEX_BASE]) / 2.0 - pos[WRIST]
    if float(np.linalg.norm(arm)) < EPS_ARM:
        return prev_R

    y = _safe_normalize(arm - float(np.dot(arm, x)) * x, EPS_Y)
    if y is None:
        return prev_R

    z = _safe_normalize(np.cross(x, y))
    if z is None:
        return prev_R

    y = _safe_normalize(np.cross(z, x), EPS_Y)
    if y is None:
        return prev_R

    if prev_R is not None and float(np.dot(prev_R[:, 0], x)) < 0.0:
        x, y = -x, -y
        z = np.cross(x, y)

    return np.column_stack([x, y, z])


def axis_points(mid: np.ndarray, R: np.ndarray) -> np.ndarray:
    """中点 + 三个轴端点, (4,3) 世界系 —— 交给 project() 一次投完。"""
    mid = np.asarray(mid, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    return np.stack(
        [mid, mid + AXIS_LENGTHS[0] * R[:, 0], mid + AXIS_LENGTHS[1] * R[:, 1],
         mid + AXIS_LENGTHS[2] * R[:, 2]]
    )


# ---------------------------------------------------------------- 朝向平滑 (HumanEgo opt)


def ema_unit_vec(v: np.ndarray, v_ema: np.ndarray | None, alpha: float):
    """对单位向量做指数滑动平均, 带符号一致 (防 180° 跳变)。逐字移植 `_ema_unit_vec`。

    AriaHandsOptimizer.py:319-335。首帧 (`v_ema is None`) 直接返回 v。
    """
    v = np.asarray(v, dtype=np.float64)
    v = v / (float(np.linalg.norm(v)) + 1e-6)
    if v_ema is None:
        return v, v.copy()
    if float(np.dot(v, v_ema)) < 0.0:
        v = -v
    v_new = (1.0 - float(alpha)) * v_ema + float(alpha) * v
    v_new = v_new / (float(np.linalg.norm(v_new)) + 1e-6)
    return v_new, v_new.copy()


def smooth_midpoint_frame(R_raw: np.ndarray, *, x_ema, y_ema, alpha: float):
    """把 `midpoint_frame` 的原始朝向再过一遍 HumanEgo opt 的平滑与再正交化。

    AriaHandsOptimizer.py:267-274:
        x = EMA(R_raw[:,0]);  y = EMA(R_raw[:,1])
        z = normalize(x × y);  y = normalize(z × x)
        R_opt = column_stack([x, y, z])

    语义仍是 x=开合 / y=接近 / z=掌心法向, **不换轴、不加修正矩阵** —— 已核实
    `MidpointFrameBuilder` 就是该仓库表示 EEF 姿态的权威方法, opt 只是对 raw 帧做
    平滑重建, 没有后续变换。返回 `(R_opt, x_ema, y_ema)` (EMA 状态要跨帧携带)。
    """
    R = np.asarray(R_raw, dtype=np.float64)
    x, x_ema = ema_unit_vec(R[:, 0], x_ema, alpha)
    y, y_ema = ema_unit_vec(R[:, 1], y_ema, alpha)
    z = np.cross(x, y)
    z = z / (float(np.linalg.norm(z)) + 1e-6)
    y = np.cross(z, x)
    return np.column_stack([x, y, z]), x_ema, y_ema


# ---------------------------------------------------------------- 二值判据


def bool_runs(mask: np.ndarray) -> list[tuple[int, int, bool]]:
    """连续同值段 [(start, end, value)], 逐字移植 grasp.py:26-35 `_bool_runs`。"""
    values = np.asarray(mask, dtype=bool)
    if len(values) == 0:
        return []
    cuts = np.flatnonzero(np.diff(values.astype(np.int8)) != 0) + 1
    bounds = [0, *cuts.tolist(), len(values)]
    return [
        (bounds[index], bounds[index + 1], bool(values[bounds[index]]))
        for index in range(len(bounds) - 1)
    ]


def absorb_short_runs(state: np.ndarray, minimum_length: int) -> np.ndarray:
    """吸收**内部**过短的段 (取前一段的值); 首尾段不动。

    逐字移植 grasp.py:38-54。首尾不吸收是上游刻意的: 采集被截断时开头/结尾那一段
    本来就短, 抹掉它会凭空造出一次状态变化。所以片头那一段"闭合"会原样留在结果里。
    """
    output = np.asarray(state, dtype=bool).copy()
    if minimum_length <= 1:
        return output
    while True:
        runs = bool_runs(output)
        interior = [
            (end - start, index)
            for index, (start, end, _value) in enumerate(runs)
            if 0 < index < len(runs) - 1 and (end - start) < minimum_length
        ]
        if not interior:
            return output
        _, index = min(interior)
        start, end, _value = runs[index]
        output[start:end] = runs[index - 1][2]


def confirmed_hysteresis(
    signal: np.ndarray,
    valid: np.ndarray,
    close_hi: float,
    open_lo: float,
    confirm_ticks: int,
    minimum_state_ticks: int,
) -> tuple[np.ndarray, np.ndarray]:
    """确认式滞回: 状态要连续 `confirm_ticks` 帧都"想要"才改, 并把切换**回填**到起点。

    逐字移植 grasp.py:86-123。两处非对称别改错:
      - 关门用 `value > close_hi` (严格), 开门用 `value >= open_lo` (带等号);
      - 无效帧只把 `pending` 清掉, **状态保持** (冻结, 不是复位)。
    """
    signal = np.asarray(signal, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if signal.shape != valid.shape:
        raise ValueError(f"signal/valid shape mismatch: {signal.shape} vs {valid.shape}")
    if not 0.0 <= open_lo <= close_hi <= 1.0:
        raise ValueError("阈值要满足 0 <= open_lo <= close_hi <= 1")
    if confirm_ticks < 1 or minimum_state_ticks < 1:
        raise ValueError("confirm_ticks / minimum_state_ticks 必须是正数")

    state = np.zeros(len(signal), dtype=bool)
    current = False
    pending: bool | None = None
    pending_start = 0
    for frame, value in enumerate(signal):
        state[frame] = current
        if not valid[frame]:
            pending = None
            continue
        wanted = bool(value > close_hi) if not current else bool(value >= open_lo)
        if wanted == current:
            pending = None
            continue
        if pending != wanted:
            pending = wanted
            pending_start = frame
        if frame - pending_start + 1 >= confirm_ticks:
            current = wanted
            state[pending_start : frame + 1] = current
            pending = None
    return state, absorb_short_runs(state, minimum_state_ticks)


def grasp_from_pinch(
    distance: np.ndarray,
    valid: np.ndarray,
    *,
    close_hi: float = DEFAULT_CLOSE_HI,
    open_lo: float = DEFAULT_OPEN_LO,
    median_window: int = DEFAULT_MEDIAN_WINDOW,
    confirm_ticks: int = DEFAULT_CONFIRM_TICKS,
    min_state_ticks: int = DEFAULT_MIN_STATE_TICKS,
    min_range: float = MIN_RANGE_M,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """指尖距 -> 闭合成度 + 开/闭二值。机制移植 adaptive_index_grasp (grasp.py:126-164)。

    信号方向与上游**相反**: 上游电机指令 1=闭合, 我们距离小=闭合, 所以归一化写成
    `(q95 - d) / span` —— q95 (最张开的那些帧) 当张开基准, q10 当闭合参考。

    中值滤波在**相机帧**这条序列上做 (不是 90 Hz 记录序列): 上游窗口 5 的单位是
    control_hz=30 的 tick, 与本仓库 30 fps 一一对应, 换到记录序列上窗口的含义就变了。

    返回 (closure, closed, calibration)。名字沿用上游的 score/before_dwell/closed,
    但顺序改成了"先给信号再给状态"。
    """
    distance = np.asarray(distance, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if distance.shape != valid.shape:
        raise ValueError(f"distance/valid shape mismatch: {distance.shape} vs {valid.shape}")
    if median_window < 1 or median_window % 2 == 0:
        raise ValueError("median_window 必须是正奇数")

    # 中值滤波前把 NaN 填掉: 一个 NaN 会顺着 median_filter 的窗口扩散到整条序列。
    # 填过的帧 valid=False, 不参与标定也不投票。
    filled_nan = int(np.count_nonzero(~np.isfinite(distance)))
    fill_value = 0.0
    if filled_nan:
        finite = np.isfinite(distance)
        fill_value = float(np.nanmedian(distance)) if finite.any() else 0.0
        distance = np.where(finite, distance, fill_value)

    filtered = median_filter(distance, size=median_window, mode="nearest")

    calibration_values = filtered[valid]
    if len(calibration_values):
        open_baseline, closed_reference = np.quantile(calibration_values, [0.95, 0.10])
    else:
        open_baseline = closed_reference = 0.0
    dynamic_range = float(open_baseline - closed_reference)
    usable_range = max(dynamic_range, float(min_range))
    too_flat = dynamic_range < float(min_range)

    closure = np.clip((float(open_baseline) - filtered) / usable_range, 0.0, 1.0)
    if too_flat:
        # 动态范围不够 -> 这一条采集里手就没怎么动, 没有抓取可言, 恒判张开
        closure[:] = 0.0

    before_dwell, closed = confirmed_hysteresis(
        closure, valid, close_hi, open_lo, confirm_ticks, min_state_ticks
    )

    calibration = {
        "open_baseline_m": float(open_baseline),
        "closed_reference_m": float(closed_reference),
        "dynamic_range_m": dynamic_range,
        "usable_range_m": float(usable_range),
        "min_range_m": float(min_range),
        "dynamic_range_too_flat": bool(too_flat),
        "close_hi": float(close_hi),
        "open_lo": float(open_lo),
        # 阈值换算成米, 方便和"耳机壳多大"对照
        "close_distance_m": float(open_baseline - close_hi * usable_range),
        "open_distance_m": float(open_baseline - open_lo * usable_range),
        "median_window": int(median_window),
        "confirm_ticks": int(confirm_ticks),
        "min_state_ticks": int(min_state_ticks),
        "nan_filled_frames": filled_nan,
        "nan_fill_value_m": fill_value,
        "valid_frames": int(valid.sum()),
        "frames": int(distance.size),
    }
    return closure, closed, calibration


def grasp_from_pinch_metric(
    distance: np.ndarray,
    valid: np.ndarray,
    *,
    close_m: float = METRIC_CLOSE_M,
    open_factor: float = METRIC_OPEN_FACTOR,
    minimum_dwell: int = 1,
) -> np.ndarray:
    """对照用的**固定阈值**判据: 指尖距 <= 0.035 m 判闭合, 要张开得超过 0.047 m。

    移植 grasp.py:57-83 `grasp_binary` 的滞回结构, 只是把信号换成"距离小=闭合"。
    上游这条是给 PICO 的兜底判据, 只用它做终端对照表, 不进画面。
    """
    distance = np.asarray(distance, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if distance.shape != valid.shape:
        raise ValueError(f"distance/valid shape mismatch: {distance.shape} vs {valid.shape}")
    open_m = float(close_m) * float(open_factor)
    state = np.zeros(len(distance), dtype=bool)
    first_valid = np.flatnonzero(valid)
    current = bool(distance[first_valid[0]] <= close_m) if len(first_valid) else False
    for frame in range(len(distance)):
        if valid[frame]:
            if not current and distance[frame] <= close_m:
                current = True
            elif current and distance[frame] >= open_m:
                current = False
        state[frame] = current
    return absorb_short_runs(state, minimum_dwell)


def state_runs_text(state: np.ndarray, *, limit: int = 6) -> str:
    """闭合段 [(start, end), ...] 的简短文本, 给终端打印用。"""
    runs = [(s, e - 1) for s, e, value in bool_runs(state) if value]
    if not runs:
        return "无"
    text = ", ".join(f"{s}-{e}({e - s + 1})" for s, e in runs[:limit])
    if len(runs) > limit:
        text += f", ... 共 {len(runs)} 段"
    return text


def summarize(state: np.ndarray) -> dict:
    """闭合帧数/占比/段数 —— 终端对照表用。"""
    state = np.asarray(state, dtype=bool)
    runs = [(s, e - 1) for s, e, value in bool_runs(state) if value]
    return {
        "closed_frames": int(state.sum()),
        "frames": int(state.size),
        "closed_ratio": float(state.mean()) if state.size else 0.0,
        "runs": runs,
        "runs_text": state_runs_text(state),
    }
