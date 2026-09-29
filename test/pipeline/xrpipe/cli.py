"""`python -m xrpipe <step> [stems ...]` —— 四个 step 的分发。

step 之间只通过 `out/<stem>/<step>/` 的产物耦合, 不互相重跑; `all` 就是按顺序调四遍。
批量时一段采集失败不拖垮其余段 (收集起来, 最后汇总退出码); step4 是「一个数据集」,
失败即整体失败。

可视化全部收在 `--vis` 后面 (默认关), 且是**后置**的: 每步跑完后由 `_vis()` 按需出片,
所以产物已存在、该步被跳过时可视化仍然出得来。只打印每段的汇总行, `--verbose` 才展开细节。

两种「提示词」不要混: step2 的 `--prompt` / step3 的 `--mask-prompt` 是给 DINO/SAM2 的
**检测提示词**, step4 的 `--task` 是 openpi 训练用的**语言指令**。不同名, 不互相兜底。

「产物在就跳过」有个例外: **action 的语义换过一趟** (参考系相机系 -> PICO OpenXR 右手世界系, 编码从
相对 obj2 换成绝对位姿) 之后, 盘上旧产物的语义已经对不上, 跳过去就等于把旧语义原样留下。
所以 step2 的跳过分支会核一次 `state_action_storage`/`action_reference_frame` (见
`_check_stored_action`), 不一致就报错, 并给出**便宜的那条修法**: `--reassemble` 只重算 2c
(assemble + write_report), 而 `--force` 会把 SAM2 + SGBM + ICP 整条重跑几十分钟 ——
参考系只影响 2c。
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from . import (ACTION_REFERENCE_FRAME, ACTION_STORAGE, DEFAULT_CALIB, LATCH_DISTANCE_M,
               MIN_OBJECT_PROMPTS, OBSERVATION_SOURCES, OUT, bootstrap, resolve_stems)
from .episode import MAX_INVALID_GAP
from .export import DEFAULT_CODEC, DEFAULT_ROBOT_TYPE
from .paths import PipePaths

STEPS = ("step1", "step2", "step3", "step4", "all")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xrpipe",
        description="PICO ego 采集 -> LeRobot (对齐 ego_relation_policy 的 s4 格式)",
    )
    parser.add_argument("step", choices=STEPS)
    parser.add_argument("stems", nargs="*", help="采集 stem (可给多个); 不给则用 --stems/--all")
    parser.add_argument("--stems", dest="stems_opt",
                        help="逗号分隔的 stem 列表, 或 all")
    parser.add_argument("--out", default=str(OUT), help="各 step 产物的根目录")
    parser.add_argument("--calib", default=str(DEFAULT_CALIB), help="标定 json (读不到就报错)")
    parser.add_argument("--lag", type=int, default=None, help="覆盖 lag (省略时自动生成/复用 align_<stem>.json)")
    parser.add_argument("--force", action="store_true", help="覆盖已有产物")
    parser.add_argument("--vis", action="store_true",
                        help="额外导出可视化 (默认关): 用 stem 选 episode, 每个 step 出一份"
                             "对应的视频/图; 纯观察, 不参与任何下游计算")
    parser.add_argument("--quiet", action="store_true", default=True, help="只打印每段汇总 (默认)")
    parser.add_argument("--verbose", dest="quiet", action="store_false", help="打印每步详情")

    group = parser.add_argument_group("step2")
    group.add_argument("--prompt", action="append", default=None,
                       help="DINO 的**检测提示词** (物体类别), 决定框出哪个物体; "
                            "给多次则按 stem 顺序一一对应。任务指令在 step4 的 --task 给")
    group.add_argument("--prompt-frame", type=int, default=50, help="取 box/mask 的提示帧")
    group.add_argument("--box", type=float, nargs="+", default=None, metavar="X1 Y1 X2 Y2",
                       help="跳过检测, 直接用这个框 (左眼像素); 一个物体给 4 个数, "
                            "N 个物体一次给 4N 个数 (顺序即 obj 序)")
    group.add_argument("--box-threshold", type=float, default=0.3)
    group.add_argument("--video-in-vram", action="store_true",
                       help="把 SAM2 的帧张量放显存 (本机 RAM 只剩几 GB 时的正解, 别用 --force 硬压预检)")
    group.add_argument("--num-disparities", type=int, default=None, help="SGBM 视差搜索范围")
    group.add_argument("--erode-px", type=int, default=None, help="mask 腐蚀像素")
    group.add_argument("--stride", type=int, default=None, help="点云抽稀步长")
    group.add_argument("--latch-distance", type=float, default=LATCH_DISTANCE_M,
                       help=f"抓握锁存的门 (米, 默认 {LATCH_DISTANCE_M}); 只在"
                            f"「夹爪闭合 ∧ 手原点↔物体原点 < 这个值」时锁存。这就是参考"
                            f" perception.latch_distance_m 那个归属门, 只是值收小了 "
                            f"(参考默认 0.20)")
    group.add_argument("--reassemble", action="store_true",
                       help="只重算 2c (assemble + write_report), 不碰 2a 的分割与 2b 的物体"
                            "位姿 —— 盘上已有的 seg_<stem>/ + object pose 直接复用。"
                            "语义换过一趟 (action 的参考系/编码) 时用这个: --force 会一路传进"
                            " segment/object_pose, 把 SAM2 + SGBM + ICP 整条重跑几十分钟, "
                            "而参考系**只影响 2c**")

    group = parser.add_argument_group("step3")
    group.add_argument("--with-arm", action="store_true", help="连手臂一起 mask + 修复")
    group.add_argument("--mask-prompt", default=None,
                       help="给 DINO 的**手部检测提示词** (决定框出手/手臂); 默认只 mask 手。"
                            "这不是任务指令 —— 任务指令在 step4 的 --task 给")
    group.add_argument("--gripper-render", choices=("piper", "wireframe"), default="piper")
    group.add_argument("--lama-model", default=None)
    group.add_argument("--piper-model", default=None)
    group.add_argument("--piper-assets", default=None)
    group.add_argument("--piper-tcp-calibration", default=None)
    group.add_argument("--piper-open-joint", type=float, default=None)
    group.add_argument("--piper-closed-joint", type=float, default=None)
    group.add_argument("--only-keep", action="store_true",
                       help="只合成会被 step4 保留的帧 (纯提速, 不改结果)")

    group = parser.add_argument_group("step4")
    group.add_argument("--dataset", default=None, help="数据集名字 (落在 <out>/lerobot/<name>)")
    group.add_argument("--dataset-dir", default=None, help="直接给数据集目录 (优先于 --dataset)")
    group.add_argument("--task", action="append", default=None,
                       help="openpi 训练用的**语言指令** (写进 tasks.jsonl 的 task 字段); "
                            "给多次则按 stem 顺序一一对应, 给 1 个则所有段共用。必给, "
                            "且与 step2/step3 的检测提示词互不兜底")
    group.add_argument("--max-invalid-gap", type=int, default=MAX_INVALID_GAP,
                       help=f"无效段 >= 这个长度才挖掉, 更短的原地桥接 (默认 {MAX_INVALID_GAP})")
    group.add_argument("--robot-type", default=DEFAULT_ROBOT_TYPE)
    group.add_argument("--observation", choices=OBSERVATION_SOURCES, default="step3",
                       help="观测帧从哪来: step3 (默认, 手已修复 + piper 夹爪) 或 "
                            "step2 (直接解源 mp4 左半, 原始手、无夹爪 —— 于是 step3 可以整步跳过)。"
                            "只影响视频, state/action 与它无关")
    group.add_argument("--video-codec", default=DEFAULT_CODEC,
                       help="h264 (libx264, 默认, 什么播放器都打得开) | "
                            "mp4v (与 s4 的 cfg.export.video_codec 逐字一致, 但兼容性差)")
    return parser


def _prompt_for(args, index: int, stem: str) -> str:
    if not args.prompt:
        raise SystemExit(
            f"step2 需要 --prompt (DINO 的检测提示词, 例如 --prompt 'small white earbud case'); "
            f"stem {stem} 没给"
        )
    if len(args.prompt) == 1:
        return args.prompt[0]
    if index >= len(args.prompt):
        raise SystemExit(f"--prompt 给了 {len(args.prompt)} 个, 不够第 {index + 1} 段 ({stem})")
    return args.prompt[index]


def _task_for(args, index: int, stem: str) -> str:
    """`--task` 的映射, 规则同 `_prompt_for`: 1 个共用 / N 个一一对应 / **0 个报错**。

    与 `_prompt_for` 唯一但关键的区别: 这里**没有兜底**。task 是 openpi 训练要的语言
    指令, 从检测提示词推 (`pick up the <prompt>`) 会得到一个「跑得起来、但 task 是不是
    训练要的那句说不清」的数据集 —— 所以缺就是错。
    """
    if not args.task:
        raise SystemExit(
            f"step4 需要 --task (openpi 训练用的语言指令, 例如 --task 'pick up the earphone case'); "
            f"stem {stem} 没给。注意这与 step2 的 --prompt / step3 的 --mask-prompt "
            f"(都是检测提示词) 不同名、不互相兜底"
        )
    if len(args.task) == 1:
        return args.task[0]
    if index >= len(args.task):
        raise SystemExit(f"--task 给了 {len(args.task)} 个, 不够第 {index + 1} 段 ({stem})")
    return args.task[index]


def _vis(args, name: str, paths: PipePaths) -> None:
    """`--vis`: 某一步跑完之后出这一步的可视化。

    刻意做成**后置的**, 不塞进 step 内部 —— 于是「产物已存在、这一步被跳过」时可视化
    照样出得来 (不用 `--force` 重跑 DINO/SAM2/LaMa)。可视化失败只警告: 它不参与任何
    下游计算, 没理由把一次成功的 step 变成失败。
    """
    if not args.vis:
        return
    from . import step1, step2, step3

    try:
        if name == "step1":
            result = step1.render_vis(paths)
        elif name == "step2":
            result = step2.render_vis(paths, prompt_frame=args.prompt_frame)
        else:
            result = step3.render_vis(paths)
    except Exception as error:
        print(f"  ⚠ {name} {paths.stem} 可视化失败 (不影响结果): {error}", file=sys.stderr)
        return
    _print_vis(name, result)


def _print_vis(name: str, result: dict) -> None:
    for key, value in result.items():
        if isinstance(value, list):
            head = f", 首个 {value[0]}" if value else ""
            print(f"  {name} vis: {key} -> {len(value)} 个{head}")
        else:
            print(f"  {name} vis: {key} -> {value}")


def _vis_only(args) -> bool:
    """这次 step4 是不是"数据集已在盘上, 只要重渲染 vis/"(于是 --task 用不上)。"""
    if args.step != "step4" or not args.vis or args.force:
        return False
    destination = Path(args.dataset_dir) if args.dataset_dir \
        else (Path(args.out) / "lerobot" / args.dataset if args.dataset else None)
    return destination is not None and (destination / "meta" / "info.json").is_file()


def _check_prompts(args, stems: list[str]) -> None:
    """开跑**之前**把 `--prompt` / `--task` 够不够查一遍, 不做任何有副作用的事。

    `all` 会先跑完 step1-3 (DINO+SAM2+LaMa, 很贵) 才走到 step4; 没给 `--task` 的话,
    等几十分钟后才在最后一步报错是很糟的体验。这里复用 `_prompt_for` / `_task_for`
    同一份话术, 所以「够不够」的判定只有一处。

    另外在这里查 `--prompt` 的**段数 >= MIN_OBJECT_PROMPTS** (任务口径是「抓物体1、放到
    物体2上」, 不是 action 需要 obj2 —— action 只跟手自身在世界系里的绝对位姿有关, 与物体
    个数无关)。只给 1 段是个必然的错, 必须在几十分钟的 DINO+SAM2 之前就报出来。

    唯一的例外是「数据集已在盘上、只重渲染 vis/」: 那条路一个 parquet 都不写, task 根本
    用不上, 所以不逼着传。
    """
    if args.step in ("step2", "all"):
        from . import step2

        for index, stem in enumerate(stems):
            prompts = step2.object_prompts(_prompt_for(args, index, stem))
            if len(prompts) < MIN_OBJECT_PROMPTS:
                raise SystemExit(
                    f"step2 {stem}: --prompt 只给了 {len(prompts)} 段 ({prompts}), 至少要 "
                    f"{MIN_OBJECT_PROMPTS} 段, 用竖线分隔 (例如 "
                    f"--prompt 'small white earbud case|red cube')。任务口径是"
                    f"「抓物体1、放到物体2上」—— 场景里的**承载物**也要框出来, "
                    f"observation.state 的每个物体一块 9 维都是训练输入"
                )
    if args.step in ("step4", "all") and not _vis_only(args):
        for index, stem in enumerate(stems):
            _task_for(args, index, stem)


def _step1(args, paths: PipePaths) -> dict | None:
    from . import step1

    # 与 step2/step3 一样: 产物在就跳过, `--force` 才重做。批量 (`all`) 靠这个才是幂等的
    # —— 否则第二次跑会在 step1 的「拒绝覆盖」上挂掉。
    if paths.reel_npz.is_file() and not args.force:
        if not args.quiet:
            print(f"  step1 {paths.stem}: 已有 {paths.reel_npz.name}, 跳过 (--force 重做)")
        return None
    return step1.run(paths, calib=args.calib, lag=args.lag, force=args.force, quiet=args.quiet)


def _stale_message(paths: PipePaths, *, storage, frame, why: str) -> SystemExit:
    """旧口径产物的话术 (两处共用): 讲清「为什么不能跳过」与「怎么修、代价多大」。

    `--force` 会一路传进 `segment(force=True)` / `object_pose(force=True)`, 把 SAM2 + SGBM +
    ICP 整条重跑几十分钟; 而参考系/编码**只影响 2c** (2b 只读 instance_ids 与手位姿)。
    所以修法给两条, 便宜的在前。
    """
    return SystemExit(
        f"step2 {paths.stem}: {why}—— 盘上 {paths.state_action_json.name} 写的是 "
        f"action_storage={storage!r} / action_reference_frame={frame!r}, 与这次要求的 "
        f"{ACTION_STORAGE!r} / {ACTION_REFERENCE_FRAME!r} 不符。跳过就等于把旧语义的 "
        f"action 留在盘上、再被 step4 照抄进数据集。修法:\n"
        f"  · 加 --reassemble: 只重算 2c (assemble + write_report), 复用盘上已有的 "
        f"seg_<stem>/ 与物体位姿 —— 几十秒\n"
        f"  · 或加 --force: 会把 SAM2 + SGBM + ICP 整条重跑 (几十分钟), 而参考系只影响 2c"
    )


def _check_stored_action(paths: PipePaths) -> None:
    """盘上那份 `state_action.json` 的 action 语义是不是这次要的那一种。

    「产物在就跳过」([`_step2`]) 在 action 语义**换掉之后**会变成陷阱: 上一轮的产物是
    「相对 obj2」、更早是「相机系绝对位姿」写的, 跳过就等于把旧语义的 action 留在盘上,
    后面 step4 照抄进数据集。所以这里精确比一次 `action_storage` + `action_reference_frame`
    (两个都必须**全等**, 不用子串 —— 旧产物的 `action_reference_frame` 里恰好含 `camera0`,
    子串匹配会把新旧混起来), 不一致就要求 --reassemble/--force。

    JSON **读不动要硬报错**: 原来读不动就 `return`, 于是「报告文件坏了」表现为「step2 被
    静默跳过」(而 `write_report` 是非原子写, 半截文件是真会出现的), 之后 step4 拿着一份
    没有对应报告的 npz 继续跑。
    """
    try:
        body = json.loads(paths.state_action_json.read_text(encoding="utf-8"))
    except Exception as error:
        raise SystemExit(
            f"step2 {paths.stem}: 盘上的 {paths.state_action_json} 读不动/不是合法 JSON "
            f"({error}) —— 它和 {paths.state_action_npz.name} 是这一步的完成标记, 没法核"
            f"action 语义。加 --reassemble 重写这份报告 (或 --force 整步重跑)"
        ) from error
    storage = body.get("action_storage")
    frame = body.get("action_reference_frame")
    if storage == ACTION_STORAGE and frame == ACTION_REFERENCE_FRAME:
        return
    raise _stale_message(paths, storage=storage, frame=frame,
                         why="盘上的产物是**旧口径**的 action")


def _reassemble_step2(args, paths: PipePaths, prompt: str) -> dict:
    """`--reassemble`: 只重跑 2c (assemble + write_report), 复用盘上 2a/2b 的产物。

    参考系从相机系换成世界系**只影响 2c** —— 2a 的分割 (SAM2) 与 2b 的物体位姿 (SGBM +
    ICP) 与 action 的参考系无关。所以语义换了一趟之后, 正确且便宜的修法是重算 2c, 而不是
    `--force` 把整条重跑几十分钟。

    前置: `step1` 的 reel.npz (手位姿的来源) 与 2b 的物体位姿产物必须都在盘上 —— 缺任何
    一个说明这条窄路径不适用, 报错让用户改走完整 step2 (见 `step2.reassemble`)。
    """
    from . import step1, step2

    if not paths.reel_npz.is_file():
        raise SystemExit(f"--reassemble 需要 step1 的 {paths.reel_npz} —— 先跑 step1")
    return step2.reassemble(
        paths, step1.hands(paths), prompts=step2.object_prompts(prompt),
        prompt_frame=args.prompt_frame, box=args.box, box_threshold=args.box_threshold,
        video_in_vram=args.video_in_vram, max_invalid_gap=args.max_invalid_gap,
        quiet=args.quiet, vis=args.vis,
    )


def _step2(args, paths: PipePaths, index: int) -> dict | None:
    from . import step1, step2

    prompt = _prompt_for(args, index, paths.stem)
    if args.reassemble:
        # 窄路径是**无条件**重算 2c 的: 它存在的理由就是「盘上那份的语义已经对不上」,
        # 而新版 npz 在盘上时 `_step2` 的跳过分支会让它变成空操作。
        return _reassemble_step2(args, paths, prompt)
    # 完成标记 = 最后写出的那两个文件 (assemble -> state_action.npz, write_report -> json)。
    # 与 `_step1` 同一个理由: `--vis` 必须停在"读产物"上, 不能为了看一眼 overlay 就把
    # DINO+SAM2 整条重跑; 批量 (`all`) 的幂等也靠它。
    if paths.state_action_npz.is_file() and paths.state_action_json.is_file() and not args.force:
        _check_stored_action(paths)
        if not args.quiet:
            print(f"  step2 {paths.stem}: 已有 {paths.state_action_npz.name}, 跳过 (--force 重做)")
        return None
    if not paths.reel_npz.is_file():
        _step1(args, paths)
    pose_kwargs = {key: getattr(args, key) for key in ("num_disparities", "erode_px", "stride")
                   if getattr(args, key) is not None}
    return step2.run(
        paths, step1.header(paths), step1.hands(paths), prompt=prompt,
        prompt_frame=args.prompt_frame, box=args.box, box_threshold=args.box_threshold,
        video_in_vram=args.video_in_vram, max_invalid_gap=args.max_invalid_gap,
        latch_distance_m=args.latch_distance,
        force=args.force, quiet=args.quiet, vis=args.vis, **pose_kwargs,
    )


def _step3(args, paths: PipePaths) -> dict | None:
    from . import step3

    # 完成标记 = observation.json (step3.run 先写 observation/*.png, 再写它)。
    if paths.observation_json.is_file() and not args.force:
        if not args.quiet:
            print(f"  step3 {paths.stem}: 已有 {paths.observation_json.name}, 跳过 (--force 重做)")
        return None
    return step3.run(
        paths, with_arm=args.with_arm, gripper_render=args.gripper_render,
        lama_model=args.lama_model, piper_model=args.piper_model, piper_assets=args.piper_assets,
        piper_tcp_calibration=args.piper_tcp_calibration, piper_open_joint=args.piper_open_joint,
        piper_closed_joint=args.piper_closed_joint, mask_prompt=args.mask_prompt,
        only_keep=args.only_keep, max_invalid_gap=args.max_invalid_gap, force=args.force,
        quiet=args.quiet, vis=args.vis,
    )


def _step4(args, paths_list: list[PipePaths], *, index_of: dict[str, int] | None = None) -> dict:
    from . import step4

    if not args.dataset_dir and not args.dataset:
        raise SystemExit("step4 需要 --dataset <name> 或 --dataset-dir <path>")
    destination = Path(args.dataset_dir) if args.dataset_dir \
        else Path(args.out) / "lerobot" / args.dataset
    # 数据集已经在盘上、又只要看可视化 -> 从盘上重渲染 `vis/`, 一个字节的 parquet/mp4 都不动。
    # 这就是「`--vis` 只读产物」在 step4 上的落法: 它的"产物"就是数据集本身。
    if args.vis and not args.force and (destination / "meta" / "info.json").is_file():
        if not args.quiet:
            print(f"  step4: 数据集已存在 {destination}, 只按盘上产物重渲染 vis/ (--force 重导)")
        vis = step4.render_from_disk(destination)
        _print_vis("step4", vis)
        return {"vis": vis, "destination": str(destination)}
    # `all` 只对跑成功的那些段导数据集, 所以 index 得按**原始 stem 顺序**算, 不能用
    # 这里的 enumerate —— 否则前面失败一段, 后面每段的 --task 都会串位。
    order = index_of or {p.stem: i for i, p in enumerate(paths_list)}
    tasks = {p.stem: _task_for(args, order[p.stem], p.stem) for p in paths_list}
    # action 的权威在 step2 的产物 (step4 只搬运): 每段 npz/report 的 action 语义由
    # `_step2` 的完成标记那条路保证, 这里再核一遍只会重复 —— 参考系换了是新旧产物的
    # 问题, 交给 `step4.prepare` 的精确比对报「旧口径」。
    summary = step4.run(
        paths_list, destination, tasks=tasks,
        max_invalid_gap=args.max_invalid_gap, robot_type=args.robot_type,
        codec=args.video_codec, observation=args.observation,
        force=args.force, quiet=args.quiet, vis=args.vis,
    )
    if summary.get("vis"):
        _print_vis("step4", summary["vis"])
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bootstrap()
    spec = args.stems_opt or (",".join(args.stems) if args.stems else None)
    stems = resolve_stems(spec)
    paths_list = [PipePaths.for_stem(stem, args.out) for stem in stems]
    _check_prompts(args, stems)
    if not args.quiet:
        print(f"[xrpipe] {args.step}: {len(stems)} 段采集 -> {args.out}")

    failures: list[tuple[str, str]] = []
    if args.step == "step4":
        try:
            _step4(args, paths_list)
        except Exception as error:
            traceback.print_exc()
            failures.append((", ".join(stems), str(error)))
        return _report(failures)

    for index, paths in enumerate(paths_list):
        try:
            if args.step == "step1":
                _step1(args, paths)
                _vis(args, "step1", paths)
            elif args.step == "step2":
                _step2(args, paths, index)
                _vis(args, "step2", paths)
            elif args.step == "step3":
                _step3(args, paths)
                _vis(args, "step3", paths)
            else:
                _step1(args, paths)
                _vis(args, "step1", paths)
                _step2(args, paths, index)
                _vis(args, "step2", paths)
                _step3(args, paths)
                _vis(args, "step3", paths)
        except Exception as error:
            traceback.print_exc()
            failures.append((paths.stem, str(error)))

    if args.step == "all":
        done = [p for p in paths_list if p.stem not in {stem for stem, _ in failures}]
        if done:
            try:
                _step4(args, done, index_of={p.stem: i for i, p in enumerate(paths_list)})
            except Exception as error:
                traceback.print_exc()
                failures.append((", ".join(p.stem for p in done), str(error)))
    return _report(failures)


def _report(failures: list[tuple[str, str]]) -> int:
    if failures:
        print(f"[xrpipe] {len(failures)} 处失败:", file=sys.stderr)
        for stem, message in failures:
            print(f"  - {stem}: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
