"""step2: DINO+SAM2 物体标定 -> 物体位姿 -> state / action。

三段:

  2a 分割     `tools/seg_object.py` 的 `stage_frames/image/video` 原样驱动
              (`xrseg.dinosam` 取提示帧 box/mask, `xrseg.sam2_video` 整段传播),
              产物落在本 pipeline 自己的 `step2/seg_<stem>/`。
  2b 物体位姿  `xrrel.stereo`(SGBM 深度) -> `xrrel.lift.build_clouds`(mask+深度反投影)
              -> `xrrel.objectpose.estimate/save`(参考帧 PCA 定向 + 逐帧 ICP + 门控 +
              HumanEgo 的「抓握锁存」手推)。参考帧 = SAM2 起始帧。
  2c state/action  `T_right_midpoint_object = compose(invert(T_camera0_midpoint),
              T_camera0_object)` —— 即 s4 的 `relation_direction: "T_tcp_object"`,
              也就是「物体在手系里」。编码用 `ego_relation.contracts.se3.transform_to_vec9`
              (与 s4 逐位同一个函数)。action/reference 则是手在 **PICO OpenXR 右手世界系**里的绝对位姿
              (`action[t]` = **下一帧**绝对位姿 + 下一帧开闭) —— 只存绝对量, 相对动作由 openpi
              训练期现算; 参考系逐段各自一个, 见 `episode.build_episode_arrays`。

**至多一个物体被锁存**: 规则本身在 2b (`blocked` 链 + `objectpose` 的 `owned_by_other`),
2c 只**校验** (`_check_single_latch`)。被锁存的帧里退化的是 `state` 的 obj 块 (物体位姿由手推
得, 不是测量), action 不受影响 —— 它只跟手自身有关。

世界系是这次新增的几何前提, 2c 因此多三道: `check_world_chain` (定义式, 只查有效帧) 挡不住
recenter, 所以另有 `check_recenter` (盯 `T_camera0_world` 自身的逐帧跳变) 否决,
`check_world_motion` (手在世界系里的逐帧位移) 只统计、不否决。无效帧的位姿先按 `hand_valid`
前向填充 (`fill_invalid_poses`), 否则那些帧会变成「指尖瞬移到世界原点」。

手的状态**只从 step1 那份 `hand_states()` 来** (由 CLI 传进来), 2b 的锁存与 2c 的
世界系位姿用同一份 `T_camera0_midpoint`/`T_world_midpoint`, 不会出现「锁存用一个末端、
报出来用另一个」。

注意方向: s4 要的是 `T_tcp_object` (物体表示在 TCP 系里), 不是 `T_object_tcp`。
两者平移范数相同, 只看平移区分不出来 —— 所以这里直接按定义算, 并断言与
`relation.run` 写出的 `T_right_midpoint_object` 一致、与反方向明显不同。

裁剪不在这里做: state/action 按**全长**算 (keep = arange(N)), 交给 step4 按
`episode.keep_index` 再切一次。两处调的是 `episode.build_episode_arrays` 同一份实现。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from . import (ACTION_COORDINATE_SYSTEM, ACTION_REFERENCE_FRAME, ACTION_SOURCE_COORDINATE_SYSTEM, ACTION_STORAGE, INSTANCE_IDS, LATCH_DISTANCE_M,
               bootstrap)
from .episode import (MAX_INVALID_GAP, build_episode_arrays, check_action_coordinate_conversion, check_recenter,
                      check_world_chain, check_world_motion, fill_invalid_poses, split_description,
                      unity_to_openxr_transforms)
from .names import action_names, reference_names, state_names
from .paths import PipePaths


def object_prompts(prompt) -> list[str]:
    """`--prompt "物体A|物体B"` -> `["物体A", "物体B"]`, **顺序即 obj1/obj2**。

    这段提示是给 DINO/SAM2 的**识别物体**提示词 (与 step4 的 `--task` 语言指令是两回事,
    别混)。不给 `|` 时就是一个物体 —— 与原来的命令行逐字相同。
    """
    values = prompt if isinstance(prompt, (list, tuple)) else str(prompt).split("|")
    prompts = [str(v).strip() for v in values if str(v).strip()]
    if not prompts:
        raise SystemExit("--prompt 是空的")
    return prompts


def object_ids(count: int) -> list[str]:
    """前 `count` 个规范实例 id (obj1..objN)。"""
    return list(INSTANCE_IDS[: int(count)])


def object_boxes(boxes, count: int) -> list[list[float]]:
    """`--box` 的扁平写法 -> 逐物体一个框, 顺序即 obj 序。

    CLI 上是 `nargs="+"`: 一个物体就是 `--box x1 y1 x2 y2` (4 个数, 与老命令行逐字相同),
    N 个物体就一次给 4N 个数。个数与物体数不符就报错 —— 少给一个会让第 2 个物体悄悄退回
    DINO, 那是最难查的一类错。
    """
    values = list(boxes or [])
    if not values:
        return []
    if all(np.isscalar(v) for v in values):
        if len(values) % 4:
            raise SystemExit(f"--box 给了 {len(values)} 个数, 必须是 4 的整数倍 (x1,y1,x2,y2)")
        values = [values[i: i + 4] for i in range(0, len(values), 4)]
    if len(values) != count:
        raise SystemExit(
            f"--box 给了 {len(values)} 个框, 但物体有 {count} 个 (--prompt 给了 {count} 个) —— "
            f"要么都给, 要么都不给"
        )
    return [[float(v) for v in box] for box in values]


# ---------------------------------------------------------------- 2a 分割


def segment(paths: PipePaths, *, prompts, prompt_frame: int, box=None,
            box_threshold: float = 0.3, video_in_vram: bool = False,
            force: bool = False, vis: bool = False) -> dict:
    """跑 2a, 产物落 `step2/seg_<stem>/`。`prompts` = 逐物体的检测提示词。

    `box` 与 `run` 的同名形参是**同一个东西** (CLI 的 `--box`: 一个物体给 4 个数, N 个物体
    一次给 4N 个), 由 `object_boxes` 拆成逐物体一个框 —— 名字在这里故意与 `run` 一致:
    上一版这一处叫 `boxes` 而两个调用点都传 `box=`, 于是**任何**走 `run()` 的 step2 都在
    进 DINO 之前就 TypeError 挂掉 (`segment() got an unexpected keyword argument 'box'`)。
    """
    from xrseg.common import SegPaths

    from . import tool_module

    prompts = object_prompts(prompts)
    wanted = object_ids(len(prompts))
    boxes = object_boxes(box, len(prompts))

    module = tool_module("seg_object")
    # 用它的 CLI 默认值 (与既有跑法逐项一致), 只覆盖本次要动的那几个。
    args = module.build_parser().parse_args([])
    args.stem = [paths.stem]
    args.prompt = list(prompts)          # 重复传 = 多个物体 (内层同一套语义)
    args.prompt_frame = int(prompt_frame)
    args.box = boxes or None           # 逐物体一个框 (内层 CLI 的 --box 就是重复给)
    args.box_threshold = float(box_threshold)
    args.outdir = str(paths.step2_dir)   # -> step2/seg_<stem>/
    args.force = bool(force)
    # SAM2 的帧张量 (12.58 MB/帧) 默认落**系统内存**: 274 帧就是 3.4 GB, 本机 15 GB RAM
    # 常被占到只剩 1-2 GB, 预检会直接拒绝。进显存 (8 GB 卡放得下) 才是这里的正解 ——
    # 别拿 --force 硬压预检, 那样真的会被 OOM 杀掉。
    args.video_in_vram = bool(video_in_vram)
    # 手部骨架叠加只画诊断图, 对物体 mask 判定没有影响; 关掉省一遍 Reel 构建。
    args.with_hand = False
    # 纯诊断产物 (quality.png / report.json) 默认不落盘 —— 要它们就得显式 --vis。
    args.no_artifacts = not vis

    seg = SegPaths.for_stem(paths.stem, paths.step2_dir)
    existing = seg.instance_ids()
    if (seg.metrics_npz.is_file() and existing == wanted
            and all(seg.prompt_mask(i).is_file() for i in existing) and not force):
        return {"seg_dir": str(seg.outdir), "reused": True, "instance_ids": existing,
                "start_frame": _rel_paths(paths).seg_start_frame(), "prompts": prompts}

    module.stage_frames(seg, args)
    module.stage_image(seg, args)
    module.stage_video(seg, args)
    # 起始帧 (SAM2 的 initial_frame) 只在 `xrrel.RelPaths.seg_start_frame()` 里实现,
    # `xrseg.common.SegPaths` 没有 —— 读的是同一份 report.json/metrics.npz, 复用它;
    # 跑完 stage 再读, 此时 report.json 才存在。
    return {"seg_dir": str(seg.outdir), "reused": False, "instance_ids": wanted,
            "start_frame": _rel_paths(paths).seg_start_frame(), "prompts": prompts}


def render_vis(paths: PipePaths, *, prompt_frame: int = 50) -> dict:
    """`--vis`: 从已有 Step1/Step2 产物渲染完整的多物体 TCP 关系视频。

    这里只读 mask、点云、关系位姿、原始视频、标定与 lag；不会重新运行
    DINO/SAM2、双目深度或 ICP。`prompt_frame` 保留在公共签名中以兼容 CLI。
    """
    del prompt_frame
    from xrrel import render, relation

    from . import DEFAULT_CALIB

    rel = _rel_paths(paths)
    if not rel.relation_npz.is_file():
        raise FileNotFoundError(f"缺 {rel.relation_npz} —— 先跑完整 step2")
    with np.load(rel.relation_npz) as archive:
        instance_ids = [str(v) for v in np.atleast_1d(archive["instance_ids"])]
    missing = [rel.clouds_npz_for(instance_id) for instance_id in instance_ids
               if not rel.clouds_npz_for(instance_id).is_file()]
    if missing:
        raise FileNotFoundError(f"缺物体点云: {', '.join(map(str, missing))} —— 先跑完整 step2")
    rel.ensure()
    _, _, reel = relation.load_reel(paths.stem, calib=DEFAULT_CALIB)
    return render.run(rel, reel=reel, frames=None, frame_print=False, verbose=True)


# ---------------------------------------------------------------- 2b 物体位姿


def _rel_paths(paths: PipePaths):
    """`xrrel.RelPaths`, 但 outdir 指向 step2, seg 目录也指向 step2 自己那份。"""
    from xrrel import RelPaths

    return RelPaths.for_stem(paths.stem, outdir=paths.step2_dir, segdir=paths.seg_dir)


def object_pose(paths: PipePaths, header, hands, *, instance_ids, num_disparities=None,
                erode_px=None, stride=None, reference_frame=None, force: bool = False,
                latch_distance_m: float = LATCH_DISTANCE_M, latch: bool = True) -> dict:
    """跑 2b: 深度 -> 点云 -> 逐物体位姿。

    深度与物体无关, `stereo.run` **只跑一次**; 之后逐物体走 `build_clouds ->
    estimate -> save`。锁存归属按**物体顺序**交接: 前一个物体锁存的帧, 通过 `blocked`
    传给后一个物体, 于是同一帧**至多一个物体被锁存** (见 `xrrel/objectpose.py` 文件头)。
    """
    from xrrel import lift as lift_mod
    from xrrel import objectpose, stereo

    rel = _rel_paths(paths)
    rel.ensure()
    n = len(hands["T_camera0_midpoint"])
    start = rel.seg_start_frame()
    if start is None:
        raise SystemExit(
            f"读不到 SAM2 起始帧 ({rel.seg_dir}) —— 2a 的分割没跑成, 或者 metrics.npz 里"
            f"没有任何 area>0 的帧"
        )
    reference = int(reference_frame) if reference_frame is not None else int(start)
    if reference != int(start):
        raise SystemExit(
            f"参考帧 {reference} != SAM2 起始帧 {start} —— 物体朝向的定向基准必须与分割一致"
        )

    stereo.run(rel, header, frames=range(int(start), n),
               num_disparities=num_disparities, force=force, verbose=False)
    extra = {}
    if erode_px is not None:
        extra["erode_px"] = int(erode_px)
    if stride is not None:
        extra["stride"] = int(stride)
    instance_ids = list(instance_ids)
    blocked = np.zeros(n, dtype=bool)   # 已被前面的物体锁存的帧位
    clouds_by_instance: dict[str, dict] = {}
    results: list = []
    for instance_id in instance_ids:
        clouds = lift_mod.build_clouds(rel, range(n), instance_ids=[instance_id], **extra)
        clouds_by_instance[instance_id] = clouds
        result = objectpose.estimate(
            rel, clouds, hands["T_camera0_midpoint"], hands["valid"], hands["closed"],
            reference_frame=reference, verbose=False,
            instance_id=instance_id, latch=latch, latch_distance_m=latch_distance_m,
            blocked=blocked,
        )
        results.append(result)
        blocked = blocked | np.asarray(result.latched, dtype=bool)
    objectpose.save(rel, results if len(results) > 1 else results[0])
    return {"relation_paths": rel, "start_frame": int(start), "reference_frame": reference,
            "instance_ids": instance_ids, "results": results,
            "clouds": clouds_by_instance, "latched": blocked}


# ---------------------------------------------------------------- 2c state/action


def _check_direction(state: np.ndarray, relation: dict, hand_valid) -> None:
    """方向护栏 (逐物体): state 的第 j 个 9 维块必须等于 `vec9(T_right_midpoint_object[:, j])`,
    且**不等于**反方向。

    两者平移范数相同 (都是同一个 |t| 的模), 只比平移区分不出来, 所以旋转也要比。

    **只比手有效的帧**: state 现在用**填充后**的手位姿算 (`fill_invalid_poses`), 而 relation
    里的 `T_right_midpoint_object` 是原始的 (无效帧上是 `inv(I) @ T_obj`); 有效帧上两者逐位
    相同, 无效帧上本来就不该比。
    """
    from ego_relation.contracts.se3 import transform_to_vec9

    flags = np.asarray(hand_valid, dtype=bool)
    forward_all = np.asarray(relation["T_right_midpoint_object"], dtype=np.float64)
    backward_all = np.asarray(relation["T_object_right_midpoint"], dtype=np.float64)
    n_obj = forward_all.shape[1] if forward_all.ndim == 4 else 1
    for j in range(n_obj):
        forward = np.stack([transform_to_vec9(m) for m in forward_all[flags, j]])
        block = state[flags, 9 * j: 9 * (j + 1)]
        if not np.allclose(block, forward, atol=1e-6):
            raise AssertionError(f"state 的第 {j} 个 9 维块与 T_right_midpoint_object 不一致")
        backward = np.stack([transform_to_vec9(m) for m in backward_all[flags, j]])
        gap = float(np.abs(block[:, 3:] - backward[:, 3:]).max())
        if gap < 1e-3:
            raise AssertionError(
                f"state 第 {j} 块与反方向 T_object_right_midpoint 的旋转只差 {gap:.2e} —— "
                f"可能取错了方向 (s4 要的是 T_tcp_object)"
            )


def _normalize_latched(latched) -> np.ndarray:
    """`latched` 统一成 `(T, N)` bool —— 单物体时产物里是 `(T,)` (`np.count_nonzero` 也照样能用)。

    与 `write_report` 里对 `object_valid`/`object_observed` 的归一同一写法。
    """
    values = np.asarray(latched, dtype=bool)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise AssertionError(f"latched 该是 (T,) 或 (T,N), 拿到 {values.shape}")
    return values


def _check_single_latch(latched) -> int:
    """**至多一个物体被锁存**: 返回每帧锁存数, 若有任何一帧 >1 就抛 AssertionError。

    规则本身在 2b (`blocked` 链逐个物体跑 + `objectpose` 拒绝 `owned_by_other`), 这里只是
    把「一次只能锁存一个物体」这条约束**钉死**成可失败的断言 —— 于是以后谁改了 2b 的
    优先级/释放逻辑, 都会在这里立刻炸, 而不是悄悄导出一份「同一帧两个物体都在手推」的数据。
    """
    values = _normalize_latched(latched)
    per_frame = values.sum(axis=1)
    worst = int(per_frame.max()) if len(per_frame) else 0
    if worst > 1:
        bad = np.nonzero(per_frame > 1)[0]
        raise AssertionError(
            f"第 {int(bad[0])} 帧 (共 {len(bad)} 帧, 前几个 {bad[:8].tolist()}) 有 "
            f"{worst} 个物体同时被锁存 —— 「一次只能锁存一个物体」被破坏, 该帧的 state 里"
            f"会有多个物体被手推成同一个位姿"
        )
    return worst


def assemble(paths: PipePaths, hands, *, verbose: bool = True) -> dict:
    """2c: 按**全长**组装 state/action/reference, 落 `step2/state_action.npz`。

    所有几何量与相机系一致, 只有 `action` / `observation.action_reference_tcp` 换成 **PICO OpenXR 右手
    世界系**里的绝对位姿 (参考系逐段各自一个, 见 `episode.build_episode_arrays`)。裁剪不在这
    里 (keep = arange(N)), 交给 step4 按 `episode.keep_index` 再切一次 —— 两处调的是
    `episode.build_episode_arrays` 同一份实现。

    落盘的 `T_camera0_midpoint` / `T_world_midpoint` 是**已按 `hand_valid` 前向填充**过的
    (无效帧的原始值是单位阵, 见 `episode.fill_invalid_poses`) —— step4 用前者校验世界链、用转换后的 `T_pico_world_openxr_midpoint` 重算 action/reference, 于是
    全长与裁剪后的两套数组逐位一致。`hand_valid` 一并落盘, 谁被填过一目了然。
    """
    from xrrel.relation import run as relation_run

    rel = _rel_paths(paths)
    relation_run(rel, hands, verbose=False)
    with np.load(rel.relation_npz) as archive:
        relation = {key: archive[key] for key in archive.files}

    T_mid_raw = np.asarray(hands["T_camera0_midpoint"], dtype=np.float64)
    T_world_raw = np.asarray(hands["T_world_midpoint"], dtype=np.float64)
    T_cam_world = np.asarray(hands["T_camera0_world"], dtype=np.float64)
    T_obj = np.asarray(relation["T_camera0_object"], dtype=np.float64)
    closed = np.asarray(hands["closed"], dtype=bool)
    valid = np.asarray(relation["valid"], dtype=bool)
    hand_valid = np.asarray(hands["valid"], dtype=bool)
    n = len(T_mid_raw)

    # 严格口径 (用户定): **全部物体有效** 才保留这一帧。世界系跳变守卫也只检查
    # 这些最终可能进入 episode 的帧；提示帧之前、追踪初始化期间等必然被裁掉的帧
    # 不应让整段数据组装失败。真正落入训练数据的 recenter 仍会被守卫拦截。
    state_valid = valid.all(axis=1) if valid.ndim > 1 else valid

    # 无效帧的位姿是单位阵 (relation.py:205-212) —— 不填就写成「指尖瞬移到世界原点」
    T_mid, fill = fill_invalid_poses(T_mid_raw, hand_valid)
    T_world, _ = fill_invalid_poses(T_world_raw, hand_valid)   # 同一个掩码 -> 计数同上
    chain_error = check_world_chain(T_world, T_mid, T_cam_world, hand_valid)
    motion = check_world_motion(T_world, hand_valid & state_valid)
    # recenter 守卫只跟头有关, 所以掩码**不**带 hand_valid —— 带上会缩小窗口, 反而可能漏掉
    # 手部掉跟踪期间发生的 recenter。
    recenter = check_recenter(T_cam_world, state_valid)
    T_action = unity_to_openxr_transforms(T_world)
    coordinate_error = check_action_coordinate_conversion(T_world, T_action)

    instance_ids = [str(v) for v in np.atleast_1d(relation["instance_ids"])]
    arrays = build_episode_arrays(T_mid, T_obj, closed, np.arange(n, dtype=np.int64), T_action)
    _check_direction(arrays["state"], relation, hand_valid)

    latched = _normalize_latched(relation["latched"])
    max_latched = _check_single_latch(latched)

    paths.ensure("step2")
    np.savez_compressed(
        paths.state_action_npz,
        stem=paths.stem,
        n_frames=np.int32(n),
        frame_index=np.arange(n, dtype=np.int32),
        timestamp_ns=np.asarray(hands["timestamps_ns"], dtype=np.int64),
        instance_ids=np.asarray(instance_ids),
        # 盘上产物的**语义指纹**: step4 拿这两个键判「这份 npz 是不是新口径」, 缺键即旧语义
        action_storage=np.asarray(ACTION_STORAGE),
        action_reference_frame=np.asarray(ACTION_REFERENCE_FRAME),
        action_coordinate_system=np.asarray(ACTION_COORDINATE_SYSTEM),
        action_source_coordinate_system=np.asarray(ACTION_SOURCE_COORDINATE_SYSTEM),
        state=arrays["state"],
        action=arrays["action"],
        # 键名不变: 仍对应 parquet 列 `observation.action_reference_tcp` (语义: 世界系本帧绝对位姿)
        action_reference_tcp=arrays["reference"],
        state_valid=state_valid,
        hand_valid=hand_valid,
        object_valid=np.asarray(relation["object_valid"], dtype=bool),
        object_observed=np.asarray(relation["object_observed"], dtype=bool),
        latched=np.asarray(relation["latched"], dtype=bool),
        grasp_closed=closed,
        T_camera0_midpoint=T_mid,
        T_world_midpoint=T_world,
        T_pico_world_openxr_midpoint=T_action,
        T_camera0_world=T_cam_world,
        T_camera0_object=T_obj,
        T_right_midpoint_object=np.asarray(relation["T_right_midpoint_object"], dtype=np.float64),
        T_object_right_midpoint=np.asarray(relation["T_object_right_midpoint"], dtype=np.float64),
    )
    if verbose:
        print(f"    state/action: {n} 帧全长, 有效 {int(state_valid.sum())} 帧 "
              f"(严格口径: 全部物体有效才留; 锁存手推 "
              f"{int(latched.sum())} 个物体帧, 每帧最多 {max_latched} 个)")
        print(f"    action/reference: {ACTION_STORAGE} @ {ACTION_REFERENCE_FRAME} (绝对位姿, "
              f"相对动作留到训练期现算); 世界链误差 {chain_error:.2e}")
        print(f"    recenter 守卫: T_camera0_world 逐帧位移 max "
              f"{recenter['max_translation_m'] * 1000:.1f} mm / {recenter['max_rotation_deg']:.2f}° "
              f"(门限 {recenter['gate_translation_m'] * 1000:.0f} mm / "
              f"{recenter['gate_rotation_deg']:.0f}°); 手在世界系里 "
              f"{motion['max_translation_m'] * 1000:.1f} mm / {motion['max_rotation_deg']:.2f}° "
              f"(只统计, 越门限 {motion['n_over_gate']} 处)")
    if fill["n_filled"] or fill["n_unfilled"]:
        print(f"  ⚠ step2 {paths.stem}: 手位姿无效 {fill['n_invalid']} 帧 —— 前向填充 "
              f"{fill['n_filled']} 帧, 另有 {fill['n_unfilled']} 帧 (首个有效帧之前) 无值可填、"
              f"仍是单位阵; 后者若被裁剪规则留下就是「指尖在世界原点」的假标签 "
              f"(write_report 里会再查一次)",
              file=sys.stderr)
    return {"relation_paths": rel, "relation": relation, "hands": hands,
            "state_valid": state_valid, "arrays": arrays,
            "max_simultaneous_latched": max_latched,
            "invalid_pose_fill": fill,
            "world_chain_max_error": chain_error,
            "world_motion": motion, "recenter": recenter,
            "action_coordinate_conversion_max_error": coordinate_error}


def write_report(paths: PipePaths, *, assembled: dict, categories,
                 seg_info: dict, max_invalid_gap: int = MAX_INVALID_GAP) -> dict:
    """落 `step2/state_action.json` (命名 + 溯源 + 裁剪预览)。

    刻意**不记任何 task / 语言指令**: 那由 step4 的 `--task` 给, 与这里的 `categories`
    (即 DINO/SAM2 的检测提示词) 是两回事。混在一处会让「task 是哪来的」变得说不清。
    """
    split = split_description(assembled["state_valid"], max_invalid_gap)
    relation = assembled["relation"]
    instance_ids = [str(v) for v in np.atleast_1d(relation["instance_ids"])]
    categories = object_prompts(categories) if not isinstance(categories, (list, tuple)) \
        else [str(v) for v in categories]
    if len(categories) != len(instance_ids):
        categories = (categories + instance_ids)[: len(instance_ids)]
    max_latched = int(assembled.get("max_simultaneous_latched", 1))
    fill = dict(assembled.get("invalid_pose_fill") or {})
    # 无值可填的那几帧 (首帧起就无效) 若是**短**无效段, 会被裁剪规则原地保留 —— 它们的
    # reference/action 是单位阵, 也就是「指尖在世界原点」, 必须点名。
    kept = set(int(v) for v in split["keep_index"])
    surviving = [int(f) for f in fill.get("unfilled_frames", []) if int(f) in kept]
    fill["unfilled_frames_kept"] = surviving
    if surviving:
        print(f"  ⚠ step2 {paths.stem}: {len(surviving)} 帧的手位姿**无值可填** (首帧起就无效) "
              f"却被裁剪规则保留了下来 (源帧号 {surviving[:8]}) —— 它们的 reference/action "
              f"是「指尖在世界原点」的单位阵, 不是真实动作; 加大 --max-invalid-gap 或换一段",
              file=sys.stderr)
    # 逐物体有效率 —— 严格口径下"哪个物体把帧裁掉了"必须一眼看得见
    object_valid = np.asarray(relation["object_valid"], dtype=bool)
    object_observed = np.asarray(relation["object_observed"], dtype=bool)
    if object_valid.ndim == 1:
        object_valid, object_observed = object_valid[:, None], object_observed[:, None]
    body = {
        "stem": paths.stem,
        "instance_ids": instance_ids,
        "categories": categories,
        "object_valid_ratio": {
            instance_id: float(object_valid[:, j].mean())
            for j, instance_id in enumerate(instance_ids)
        },
        "object_observed_ratio": {
            instance_id: float(object_observed[:, j].mean())
            for j, instance_id in enumerate(instance_ids)
        },
        "valid_reduction": "state_valid = 全部物体 object_valid ∧ hand_valid (严格口径)",
        "relation_direction": "T_tcp_object",
        "relation_pose_encoding": "[tx,ty,tz,R[:,0],R[:,1]]",
        "action_storage": ACTION_STORAGE,
        "action_reference_frame": ACTION_REFERENCE_FRAME,
        "action_coordinate_system": ACTION_COORDINATE_SYSTEM,
        "action_source_coordinate_system": ACTION_SOURCE_COORDINATE_SYSTEM,
        "action_coordinate_transform": {
            "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
            "translation": "t_openxr = M @ t_unity",
            "rotation": "R_openxr = M @ R_unity @ M",
            "max_error": float(assembled.get("action_coordinate_conversion_max_error", 0.0)),
        },
        "action_reference_frame_definition": (
            "runtime 报的 PICO tracking space (trackingData 的原始世界坐标, 本 pipeline **不做**"
            "任何重新归零): 同一段采集内固定, 逐段各自一个 ⇒ 跨 episode 的 action **绝对值不可比**, "
            "同段内可比。action/reference 已从 Unity 左手系转换为 OpenXR 右手系 (X右/Y上/Z后)"
        ),
        "action_semantics": (
            "action[t] = vec9(T_pico_world_openxr_right_tcp[t+1]) —— 世界系里的**下一帧绝对位姿** "
            "(episode 内下一帧, 不是源帧的下一帧) + 下一帧开闭; 末帧重复自身, 与 s4 的 "
            "min(t+1, T-1) 同义。**相对动作不进数据集**, 由训练期 inv(reference[t]) @ action[t] 现算"
        ),
        "action_reference_semantics": (
            "reference[t] = vec9(T_pico_world_openxr_right_tcp[t]) —— **同一世界系里的本帧**绝对位姿 "
            "(列名不变: observation.action_reference_tcp), 所以 action[t] = reference[t+1]; "
            "与 s4 的 right_tcp_absolute_current 逐字同构, 只是绝对量的落点不同"
        ),
        "training_action_transform": "deferred: inv(T_current) @ T_absolute_target",
        "hand_frame": (
            "指尖中点 (HumanEgo MidpointFrameBuilder, 拇指+食指), 不是 TCP/手腕语义; "
            "Invalid 帧已按 hand_valid 前向填充 (见 invalid_pose_fill)"
        ),
        "world_chain": {
            "definition": "T_world_midpoint == inv(T_camera0_world) @ T_camera0_midpoint (仅有效帧)",
            "max_error": float(assembled.get("world_chain_max_error", 0.0)),
        },
        # 手在世界系里的逐帧位移: 只统计, 不否决 (它抓不到 recenter, 见 check_world_motion)。
        "world_motion": dict(assembled.get("world_motion") or {}),
        # recenter 守卫的实测值 (**否决项**): T_camera0_world 自身的逐帧跳变。
        "recenter": dict(assembled.get("recenter") or {}),
        "invalid_pose_fill": fill,
        "single_latch_rule": (
            "任意时刻至多一个物体被锁存 (2b 的 blocked 链 + objectpose 拒绝 owned_by_other); "
            "本字段由 _check_single_latch 在组装时断言。被锁存的帧里退化的是 **state** 的 obj 块 "
            "(物位姿由手推得, 不是测量), action 不受影响 (它只跟手自身有关)"
        ),
        "max_simultaneous_latched": max_latched,
        "grasp_binary_semantics": "state[t]=closed[t]; action[t]=closed[episode-local t+1]",
        # 与 `export.features` 里写进 info.json 的那份同源: 9N+1, 按 obj 序。
        "state_names": state_names(*instance_ids),
        "action_names": action_names(),
        "reference_names": reference_names(),
        **split,
        "seg": seg_info,
    }
    paths.state_action_json.write_text(
        json.dumps(body, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return body


def reassemble(paths: PipePaths, hands, *, prompts, prompt_frame: int = 50, box=None,
               box_threshold: float = 0.3, video_in_vram: bool = False,
               max_invalid_gap: int = MAX_INVALID_GAP, quiet: bool = False,
               vis: bool = False) -> dict:
    """**只重算 2c** (assemble + write_report), 复用盘上 2a/2b 的产物 (CLI 的 `--reassemble`)。

    为什么需要这条窄路径: action 的参考系/编码换过一趟之后, 盘上旧产物的语义已经对不上,
    必须重算 —— 但**只有 2c 与参考系有关**。2a 的分割 (SAM2) 与 2b 的物体位姿 (SGBM + ICP)
    逐字不受影响, 而 `--force` 会一路传进 `segment(force=True)` / `object_pose(force=True)`,
    把整条重跑几十分钟。这里显式隔开: `segment` 以 `force=False` 走复用分支 (只读一遍
    metrics.npz 与 prompt mask 就返回), `object_pose` 根本不调。

    前置是 2b 的 `object_npz` (2c 的输入) —— 盘上没有就说明这条窄路径不适用, 直接报错。
    """
    rel = _rel_paths(paths)
    if not rel.object_npz.is_file():
        raise SystemExit(
            f"--reassemble 需要 2b 的物体位姿产物 {rel.object_npz} —— 盘上没有 (或换了 stem); "
            f"这条窄路径不适用, 去掉 --reassemble 跑完整 step2"
        )
    seg_info = segment(paths, prompts=prompts, prompt_frame=prompt_frame, box=box,
                       box_threshold=box_threshold, video_in_vram=video_in_vram,
                       force=False, vis=vis)
    assembled = assemble(paths, hands, verbose=not quiet)
    report = write_report(paths, assembled=assembled, categories=object_prompts(prompts),
                          seg_info=seg_info, max_invalid_gap=max_invalid_gap)
    if not quiet:
        print(f"  step2 {paths.stem}: --reassemble 只重算 2c -> 保留 "
              f"{report['n_frames_kept']}/{report['n_frames_source']} 帧")
    return report


def run(paths: PipePaths, header, hands, *, prompt, prompt_frame: int, box=None,
        box_threshold: float = 0.3, video_in_vram: bool = False, force: bool = False,
        max_invalid_gap: int = MAX_INVALID_GAP, quiet: bool = False, vis: bool = False,
        latch_distance_m: float = LATCH_DISTANCE_M, latch: bool = True,
        **pose_kwargs) -> dict:
    """`prompt` = `--prompt "物体A|物体B"` (竖线分隔, 顺序即 obj1/obj2)。

    `latch_distance_m` = 抓握锁存的门 (米, 默认 0.05); `latch=False` 不做锁存/手推。
    物体个数由提示词段数决定 (至少 `MIN_OBJECT_PROMPTS` 段, 任务口径是「抓物体1、放到物体2上」),
    与 action 无关 —— action 现在只跟手自身在世界系里的绝对位姿有关。
    """
    bootstrap()
    prompts = object_prompts(prompt)
    instance_ids = object_ids(len(prompts))
    seg_info = segment(paths, prompts=prompts, prompt_frame=prompt_frame, box=box,
                       box_threshold=box_threshold, video_in_vram=video_in_vram,
                       force=force, vis=vis)
    pose = object_pose(paths, header, hands, instance_ids=instance_ids,
                       force=force, latch_distance_m=latch_distance_m, latch=latch,
                       **pose_kwargs)
    assembled = assemble(paths, hands, verbose=not quiet)
    report = write_report(
        paths, assembled=assembled, categories=prompts,
        seg_info=seg_info, max_invalid_gap=max_invalid_gap,
    )
    if not quiet:
        print(
            f"  step2 {paths.stem}: 有效段 {report['valid_runs']} -> 保留 "
            f"{report['n_frames_kept']}/{report['n_frames_source']} 帧, "
            f"桥接的短无效段 {report['kept_invalid_runs']}"
        )
    return report
