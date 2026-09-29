"""裁剪规则与 state/action 的对齐 —— step2 与 step4 共用的**唯一**一份实现。

规则 (一次采集 = 一个 episode):

    invalid_runs = 连续 valid=false 的段 (含首尾)
    keep_index   = 挖掉所有 len(run) >= max_invalid_gap 的段之后剩下的源帧号

长度 < `max_invalid_gap` 的无效段**保留** (原地桥接, 帧不丢, 只是那些帧的 state 不可信);
更长的整段挖掉, 前后剩余的帧**拼接进同一个 episode** (不新建 episode), 于是 episode
内部会出现一次源帧号跳变。一次采集永远只产出 1 个 episode。

对齐的关键: 所有张量用**同一个 `keep_index`** 切片, 并且 `action` 指向的是 **episode 内的
下一帧** (不是源帧的下一帧) —— 桥接段与拼接边界处这两者不同。所以 state/action/reference
一律由 `build_episode_arrays` 生成, step2 (keep = arange(N), 全长) 与 step4 (真正的
keep_index) 都调它, 不可能各写一套。
"""

from __future__ import annotations

import numpy as np

from . import (MAX_RECENTER_STEP_ROTATION_DEG, MAX_RECENTER_STEP_TRANSLATION_M,
               MAX_WORLD_STEP_ROTATION_DEG, MAX_WORLD_STEP_TRANSLATION_M)

MAX_INVALID_GAP = 30

# Exact match for ego_relation.contracts.se3.UNITY_TO_OPENXR. PICO tracking is
# Unity LH (x-right/y-up/z-forward); action uses OpenXR RH (x-right/y-up/z-back).
UNITY_TO_OPENXR = np.diag([1.0, 1.0, -1.0])


def unity_to_openxr_transforms(transforms: np.ndarray) -> np.ndarray:
    """Convert one or more Unity-LH poses to OpenXR-RH: t_new=M t, R_new=M R M."""
    values = np.asarray(transforms, dtype=np.float64)
    if values.shape[-2:] != (4, 4):
        raise ValueError(f"pose matrices must end in (4,4), got {values.shape}")
    change = np.eye(4, dtype=np.float64)
    change[:3, :3] = UNITY_TO_OPENXR
    return change @ values @ change


def check_action_coordinate_conversion(source: np.ndarray, converted: np.ndarray,
                                       atol: float = 1e-12) -> float:
    """Assert that action poses exactly follow the Unity-to-OpenXR basis change."""
    expected = unity_to_openxr_transforms(source)
    actual = np.asarray(converted, dtype=np.float64)
    if actual.shape != expected.shape:
        raise ValueError(f"coordinate conversion shape mismatch: {expected.shape} / {actual.shape}")
    error = float(np.max(np.abs(actual - expected))) if expected.size else 0.0
    if error > atol:
        raise AssertionError(f"Unity-to-OpenXR action pose error {error:.3e} > {atol:g}")
    rotations = actual[..., :3, :3].reshape(-1, 3, 3)
    if rotations.size and not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-9):
        raise AssertionError("Unity-to-OpenXR action pose has det(R) != +1")
    return error


def runs_of(mask: np.ndarray) -> list[list[int]]:
    """连续 True 的段 `[[start, end], ...]` (闭区间), 按帧号升序。"""
    flags = np.asarray(mask, dtype=bool)
    if flags.ndim != 1:
        raise ValueError(f"要一维掩码, 拿到 {flags.shape}")
    out: list[list[int]] = []
    start: int | None = None
    for i, on in enumerate(flags):
        if on and start is None:
            start = i
        elif not on and start is not None:
            out.append([start, i - 1])
            start = None
    if start is not None:
        out.append([start, len(flags) - 1])
    return out


def keep_index(valid: np.ndarray, max_invalid_gap: int = MAX_INVALID_GAP) -> np.ndarray:
    """按上面的规则给出保留的源帧号 (升序 int32)。

    短无效段保留 -> 它只从 `valid` 里体现, 不影响帧号; 长无效段整段挖掉。
    """
    if max_invalid_gap < 1:
        raise ValueError(f"max_invalid_gap 必须 >= 1, 拿到 {max_invalid_gap}")
    flags = np.asarray(valid, dtype=bool)
    n = len(flags)
    if n == 0:
        return np.zeros(0, dtype=np.int32)

    keep = np.ones(n, dtype=bool)
    for start, stop in runs_of(~flags):
        if stop - start + 1 >= max_invalid_gap:
            keep[start:stop + 1] = False
    return np.nonzero(keep)[0].astype(np.int32)


def split_description(valid: np.ndarray, max_invalid_gap: int = MAX_INVALID_GAP) -> dict:
    """裁剪过程的可追溯记录 (写进 JSON 报告, 便于复查为什么丢了那几帧)。"""
    flags = np.asarray(valid, dtype=bool)
    kept: list[list[int]] = []
    dropped: list[list[int]] = []
    for start, stop in runs_of(~flags):
        (dropped if stop - start + 1 >= max_invalid_gap else kept).append([start, stop])
    index = keep_index(flags, max_invalid_gap)
    return {
        "max_invalid_gap": int(max_invalid_gap),
        "valid_runs": runs_of(flags),
        "invalid_runs": runs_of(~flags),
        "kept_invalid_runs": kept,
        "dropped_invalid_runs": dropped,
        "keep_index": index.tolist(),
        "n_frames_source": int(len(flags)),
        "n_frames_kept": int(len(index)),
    }


def fill_invalid_poses(T: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, dict]:
    """把 `valid=false` 的帧按**前向填充**补上前一个有效帧的位姿; 返回 (副本, 计数)。

    为什么必须填: `hand_states` 给无效帧的 `T_world_midpoint` / `T_camera0_midpoint` 是
    **单位阵** (`relation.py:199-212`), 而 `T_camera0_world` 仍是真值 —— 不填的话那些帧的
    reference/action 会变成「指尖瞬移到世界原点」(≈0.4 m 的跳变, 比整段真实行程还大),
    还会被算进 `stats.json` / `episodes_stats.jsonl`, 污染 openpi 的归一化
    (`normalization_clip=5.0`)。而这些帧**不会**被全挖掉: `keep_index` 只挖
    >= `max_invalid_gap` 的段, 更短的原地桥接保留 (见模块 docstring)。

    填充是**因果的前向填充** (照 HumanEgo-main 的 `Forward Fill Hand if momentarily missing
    tracking`, `FlowMatchingDataloader.py:527-530`): 拿不到手的位置时保持上一个已知位姿。
    首帧起就无效的那些帧没有更早的值可填, 只能留单位阵 —— 计数里单独报出来
    (`n_unfilled`), 因为它们确实是垃圾值, 不该被当成「填过就好了」。

    填充要在**全长**数组上做、再按 `keep_index` 切 (不是切完再填): 桥接段的前几帧要用到
    该段**之前**的有效位姿。
    """
    values = np.asarray(T, dtype=np.float64)
    flags = np.asarray(valid, dtype=bool)
    if flags.ndim != 1 or len(flags) != len(values):
        raise ValueError(
            f"valid 该是 (N,) 且与位姿序列等长, 拿到 {flags.shape} vs {values.shape}"
        )
    filled = values.copy()
    last: np.ndarray | None = None
    n_filled = 0
    unfilled: list[int] = []
    for k in range(len(filled)):
        if flags[k]:
            last = values[k]
        elif last is None:
            unfilled.append(k)       # 首个有效帧之前: 没有可填的值
        else:
            filled[k] = last
            n_filled += 1
    return filled, {"n_invalid": int((~flags).sum()), "n_filled": n_filled,
                    "n_unfilled": len(unfilled), "unfilled_frames": unfilled}


# ---------------------------------------------------------------- 世界系的守卫


def invert_batch(T: np.ndarray) -> np.ndarray:
    """(N,4,4) 刚体逆 (Rᵀ, −Rᵀt) —— `ego_relation.contracts.se3.invert` 只吃单个 4x4。"""
    values = np.asarray(T, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (4, 4):
        raise ValueError(f"要 (N,4,4), 拿到 {values.shape}")
    rotation = np.swapaxes(values[:, :3, :3], -1, -2)
    out = np.zeros_like(values)
    out[:, :3, :3] = rotation
    out[:, :3, 3] = -np.einsum("nij,nj->ni", rotation, values[:, :3, 3])
    out[:, 3, 3] = 1.0
    return out


def check_world_chain(T_world: np.ndarray, T_camera0_midpoint: np.ndarray,
                      T_camera0_world: np.ndarray, hand_valid: np.ndarray,
                      atol: float = 1e-9) -> float:
    """定义式 `T_world == inv(T_camera0_world) @ T_camera0_midpoint`, **只对有效的帧**。

    原始 action 源位姿全押在 `T_world` 上, 而它来自 step1 的 `reel.G_MID`/`G_R`
    (由 `reel.npz` 直接落盘), 与相机链是**两条独立的存储**; 这条式子把两者钉在一起
    (实测本段 max 0.0)。返回最大逐元素误差, 超限抛 AssertionError。

    **只查有效帧**: 无效帧的 `T_camera0_midpoint` / `T_world` 是单位阵而 `T_camera0_world`
    是真值 (relation.py:205-212), 式子本来就不成立 —— 不挖掉要么误报, 要么被迫把 atol 放到
    没有意义。所以调用方传进来的应当是**已填充**的 `T_world`/`T_camera0_midpoint`
    (有效帧上与原始值逐位相同, 见 `fill_invalid_poses`)。
    """
    T_world = np.asarray(T_world, dtype=np.float64)
    mid = np.asarray(T_camera0_midpoint, dtype=np.float64)
    cam = np.asarray(T_camera0_world, dtype=np.float64)
    flags = np.asarray(hand_valid, dtype=bool)
    if not (T_world.shape == mid.shape == cam.shape):
        raise ValueError(f"三个位姿序列形状不一致: {T_world.shape} / {mid.shape} / {cam.shape}")
    if flags.shape != (len(T_world),):
        raise ValueError(f"hand_valid 该是 ({len(T_world)},), 拿到 {flags.shape}")
    if not flags.any():
        raise AssertionError("一帧有效的手位姿都没有 —— 这段采集没有任何可用的世界系位姿")
    chain = invert_batch(cam[flags]) @ mid[flags]
    error = float(np.abs(T_world[flags] - chain).max())
    if error > atol:
        bad = np.nonzero(flags)[0][np.abs(T_world[flags] - chain).max(axis=(1, 2)) > atol]
        raise AssertionError(
            f"T_world_midpoint 与 inv(T_camera0_world) @ T_camera0_midpoint 最大差 {error:.3e} "
            f"(atol {atol:g}; 前几个坏帧 {bad[:8].tolist()}) —— 世界链与相机链不是同一批位姿, "
            f"reference/action 会与 state 的相机系脱钩"
        )
    return error


def check_world_motion(T_world: np.ndarray, hand_valid: np.ndarray, *,
                       max_translation_m: float = MAX_WORLD_STEP_TRANSLATION_M,
                       max_rotation_deg: float = MAX_WORLD_STEP_ROTATION_DEG) -> dict:
    """世界系里手的逐帧位移/转角 —— **只统计, 不否决**。

    曾经拿它当 recenter 守卫 (`max_translation_m`/`max_rotation_deg` 只用于标注门槛, 供读报告的
    人对照), 但实测它抓不到: 门限 0.05 m 落在真实手部运动分布里面 (49 段逐段 max 的中位就有
    30.4 mm、p90 53.6 mm), 越线的 9 段逐条核对下来, 手在**相机系**里跳了同样大的量 —— 而
    recenter 的前提正是「相机系里看不见」。真正的守卫是 `check_recenter`。

    返回实测统计 (写进报告, 便于事后判断余量)。
    """
    T_world = np.asarray(T_world, dtype=np.float64)
    flags = np.asarray(hand_valid, dtype=bool)
    if flags.shape != (len(T_world),):
        raise ValueError(f"hand_valid 该是 ({len(T_world)},), 拿到 {flags.shape}")
    steps = np.nonzero(flags[:-1] & flags[1:])[0]      # 只看两帧都有效的相邻对
    if len(steps) == 0:
        return {"n_pairs": 0, "max_translation_m": 0.0, "max_rotation_deg": 0.0,
                "gate_translation_m": float(max_translation_m),
                "gate_rotation_deg": float(max_rotation_deg)}
    delta = invert_batch(T_world[steps]) @ T_world[steps + 1]
    translation = np.linalg.norm(delta[:, :3, 3], axis=1)
    trace = np.trace(delta[:, :3, :3], axis1=1, axis2=2)
    rotation = np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))
    worst_t, worst_r = float(translation.max()), float(rotation.max())
    n_over = int(((translation > max_translation_m) | (rotation > max_rotation_deg)).sum())
    return {"n_pairs": int(len(steps)), "max_translation_m": worst_t,
            "max_rotation_deg": worst_r, "n_over_gate": n_over,
            "gate_translation_m": float(max_translation_m),
            "gate_rotation_deg": float(max_rotation_deg)}


def check_recenter(T_cam_world: np.ndarray, valid: np.ndarray, *,
                   max_translation_m: float = MAX_RECENTER_STEP_TRANSLATION_M,
                   max_rotation_deg: float = MAX_RECENTER_STEP_ROTATION_DEG) -> dict:
    """**中途 recenter 守卫**: `T_camera0_world` 自身的逐帧位移/转角不得超过门限。

    为什么必须盯这个量: 录制中途一旦 recenter, runtime 会把此后所有世界系位姿左乘同一个 J,
    于是 `T_camera0_world -> T_camera0_world @ J^-1`, 而
    `T_camera0_midpoint = T_camera0_world @ T_world_midpoint` (relation.py:211) 在 J 与
    `T_camera0_world` 的新值下**逐字不变** —— 旧口径免疫、新口径不免疫, 且现有的每一条自检
    恒等式 (`check_world_chain`) 照样通过。**只有 `T_camera0_world` 自己的突变能暴露它。**

    这个量只由头部位姿决定, 与手部跟踪无关: 实测 49 段无一处是填充/单位阵, 手未跟踪区间里
    它仍是平滑的亚毫米值。门限 0.10 m / 45° 取自实测 (逐帧位移 max 25.4 mm、转角 max 10.0°)
    与 recenter 的量级 (~0.4 m) 之间 —— 两边各留 4 倍。
    """
    T_cam_world = np.asarray(T_cam_world, dtype=np.float64)
    flags = np.asarray(valid, dtype=bool)
    if flags.shape != (len(T_cam_world),):
        raise ValueError(f"valid 该是 ({len(T_cam_world)},), 拿到 {flags.shape}")
    steps = np.nonzero(flags[:-1] & flags[1:])[0]      # 只看两帧都在 episode 窗口内的相邻对
    if len(steps) == 0:
        return {"n_pairs": 0, "max_translation_m": 0.0, "max_rotation_deg": 0.0,
                "gate_translation_m": float(max_translation_m),
                "gate_rotation_deg": float(max_rotation_deg)}
    delta = invert_batch(T_cam_world[steps]) @ T_cam_world[steps + 1]
    translation = np.linalg.norm(delta[:, :3, 3], axis=1)
    trace = np.trace(delta[:, :3, :3], axis1=1, axis2=2)
    rotation = np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))
    worst_t, worst_r = float(translation.max()), float(rotation.max())
    stats = {"n_pairs": int(len(steps)), "max_translation_m": worst_t,
             "max_rotation_deg": worst_r,
             "gate_translation_m": float(max_translation_m),
             "gate_rotation_deg": float(max_rotation_deg)}
    if worst_t > max_translation_m or worst_r > max_rotation_deg:
        over = np.nonzero((translation > max_translation_m) | (rotation > max_rotation_deg))[0]
        frames = steps[over]
        raise AssertionError(
            f"T_camera0_world 在相邻帧间跳了 {worst_t * 1000:.1f} mm / {worst_r:.2f}° "
            f"(门限 {max_translation_m * 1000:.0f} mm / {max_rotation_deg:.0f}°; 首帧 "
            f"{int(frames[0])}, 共 {len(frames)} 处) —— 录制中途发生了 recenter, 世界系被整体"
            f"重置, 此后 action/reference 的绝对值不再与前面同源 (几何自检发现不了这件事, "
            f"因为 T_camera0_midpoint = T_camera0_world @ T_world_midpoint 逐字不变)"
        )
    return stats


def build_episode_arrays(T_mid: np.ndarray, T_obj: np.ndarray, closed: np.ndarray,
                         keep_index: np.ndarray, T_world: np.ndarray) -> dict:
    """把全长的原始量按 `keep_index` 切成一个 episode 的 state/action/reference。

    - `T_mid`   (N,4,4) f64    左眼相机 <- 指尖中点
    - `T_obj`   (N,M,4,4) f64  左眼相机 <- 物体 (**物体轴恒在**, M 个物体按 obj 序)
    - `closed`  (N,)   bool    二值爪
    - `keep_index` (L,) int    升序源帧号
    - `T_world` (N,4,4) f64    **PICO OpenXR 右手世界系** <- 指尖中点 (Step2 已转换)

    返回 dict:
      state        (L, 9M+1) f32  [vec9(inv(T_mid) @ T_obj_j) (j=0..M-1), closed]  本帧开闭进 state
      reference    (L, 9)    f32  vec9(T_world)   本帧在 **pico_world_openxr** 里的绝对位姿
      action       (L, 10)   f32  [OpenXR 右手世界系里的**下一帧**绝对位姿, 下一帧开闭]  末帧重复自身

    **action 是绝对位姿, 不是相对动作** —— 与 s4 逐字同构 (`action_storage: "absolute"` +
    `training_action_transform: "deferred: inv(T_current) @ T_absolute_target"`): 相对动作
    不进数据集, 由 openpi 训练期现算。`reference` 是**同一参考系里的本帧**位姿, 所以
    `action[t] = reference[t+1]`, 结构恒等式 `action[:, :9] == [reference[1:], reference[-1:]]`
    与旧版逐字相同 (末帧重复自身, 与 s4 的 `min(t+1, T-1)` 一致, `lerobot.py:196-198`);
    `inv(reference[t]) @ action[t]` 因此是**纯手自身在两帧间的相对运动**, 且对「每段各一个
    常量世界系」不变 —— 这正是训练期要的那个量。

    参考系为什么是世界系而不是左眼相机系 —— **头的位姿不是任何模型输入** (`state` 只有物体在
    手系里的关系, 世界系原点在 `state` 里完全不可观测), 而相机系绝对位姿把「当时头在哪」
    编进了监督目标: `inv(T_cam0_mid[t]) @ T_cam0_mid[t+1]` = `inv(T_w[t]) · J · T_w[t+1]`,
    中间夹着的 J 是**相机自己的逐帧运动** (本段实测每帧 ≤3.31 mm / ≤1.122°; 杠杆臂 = 手到
    相机的距离 ≤0.539 m), 于是同一段手部动作会因为头的位置不同而标成不同的数。这个差直接
    落在相对动作上 —— 本段 274 帧上, `inv(T_cam0_mid[t]) @ T_cam0_mid[t+1]` 与
    `inv(reference[t]) @ action[t]` 的**平移差** p50 1.77 / p99 7.56 / max 9.19 mm, 而这两个
    量本身的 p50 都只有 ~3.6 mm (相机系 3.74 / 世界系 3.63): 逐帧几毫米的假动作混在真行程
    里, 单帧看不出来, 累积起来就是「同一个动作被标成两组数」。世界系里
    `inv(reference[t]) @ action[t]` 就是手自身的相对运动, 且对「每段各一个常量世界系」不变
    —— 这正是 openpi 训练期要现算的那个量。

    世界系是 runtime 报的 tracking space: **段内不动, 但逐段各自一个** (原点由 runtime 定,
    本 pipeline 不做任何重新归零), 所以跨 episode 的 action **绝对值不可比**, 同一段内可比。
    与 s4 的 `g1_base` 逐 episode 不同是同一个情形。录制中途 recenter 会让世界系整体重置,
    由 `check_recenter` 挡住。

    `T_world` 必须**先过 `fill_invalid_poses`** (无效帧的它是单位阵, 直接用会写成
    「指尖瞬移到世界原点」); `T_mid` 同理, 否则 state 的物体块也是垃圾。
    """
    from ego_relation.contracts.se3 import compose, invert, transform_to_vec9

    T_mid = np.asarray(T_mid, dtype=np.float64)
    T_obj = np.asarray(T_obj, dtype=np.float64)
    if T_obj.ndim == 3:  # 老调用方给单物体的 (N,4,4): 补一个长度为 1 的物体轴
        T_obj = T_obj[:, None]
    closed = np.asarray(closed, dtype=bool)
    index = np.asarray(keep_index, dtype=np.int64)
    T_world = np.asarray(T_world, dtype=np.float64)
    if T_world.shape != T_mid.shape:
        raise ValueError(
            f"T_world 形状 {T_world.shape} != T_mid 形状 {T_mid.shape} —— "
            f"世界系与相机系必须是同一批帧 (T_world 取自 step1 的 T_world_midpoint)"
        )
    L = len(index)
    if L == 0:
        raise ValueError("keep_index 为空 —— 这一段采集没有任何可用的帧")

    T_m = T_mid[index]
    T_o = T_obj[index]
    T_w = T_world[index]
    grasp = closed[index]
    n_obj = T_o.shape[1]

    if not (np.isfinite(T_m).all() and np.isfinite(T_o).all() and np.isfinite(T_w).all()):
        bad = np.nonzero(~(np.isfinite(T_m).all(axis=(1, 2))
                           & np.isfinite(T_o).all(axis=(1, 2, 3))
                           & np.isfinite(T_w).all(axis=(1, 2))))[0]
        raise ValueError(
            f"保留帧里有非有限的位姿 (episode 局部下标 {bad[:8].tolist()} 等 "
            f"{len(bad)} 帧) —— 物体位姿在 SAM2 起始帧之前无定义, 不该被保留"
        )

    state = np.empty((L, 9 * n_obj + 1), dtype=np.float32)
    reference = np.empty((L, 9), dtype=np.float32)
    action = np.empty((L, 10), dtype=np.float32)
    for k in range(L):
        for j in range(n_obj):
            state[k, 9 * j: 9 * (j + 1)] = transform_to_vec9(compose(invert(T_m[k]), T_o[k, j]))
        state[k, -1] = 1.0 if grasp[k] else 0.0
        # 本帧的手, 在 **PICO OpenXR 右手世界系**里的绝对位姿 (不参照任何物体, 也不随头动)
        reference[k] = transform_to_vec9(T_w[k])
    # action = OpenXR 右手世界系里的下一帧位姿 —— 就是 reference 的下一个元素 (末帧重复自身)
    action[:-1, :9] = reference[1:]
    action[:-1, 9] = grasp[1:].astype(np.float32)
    action[L - 1, :9] = reference[L - 1]
    action[L - 1, 9] = state[L - 1, -1]
    return {"state": state, "reference": reference, "action": action}
