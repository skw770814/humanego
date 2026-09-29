#!/usr/bin/env python3
"""耳机壳分割: step2 的 DINO→SAM2 链路复刻 + 可视化。

    GroundingDINO(文本提示 -> 框) -> SAM2 image predictor(框 -> mask, 仅提示帧)
      -> SAM2 video predictor(传播全片) -> mask 叠加视频 / 质量曲线 / 静帧

必须用本链路自己的 venv 跑 (torch / sam2 / transformers / cv2 都在里面):

    source tools/seg_env.sh
    .venv/bin/python tools/seg_object.py 20260920_111300 \\
        --prompt "earphone case" --stage all

分阶段跑 (产物都落盘, 可续; --stage all = frames+image+video+render):

    check    版本 / GPU / sam2 配置 / 视频 / 对齐产物自检, 不下任何东西
    fetch    预下权重 (DINO 689 MB + SAM2 156 MB, 走 hf-mirror 镜像)
    frames   解码 mp4 左半 eye0 -> out/seg_<stem>/frames/%05d.jpg
    image    DINO 文本提示 -> 框 -> SAM2 image predictor -> mask_obj1.png
             + prompt_frame.png  ← 这一步跑完**先看这张图**, 确认框住的是耳机壳
               且没吃进手指, 再往下跑 video
    video    传播全片 -> masks/obj1/%05d.png + metrics.npz + quality.png + report.json
    render   seg_overlay_<stem>.mp4 (2160x810) + still_f####.png

体例与 tools/overlay.py 对齐 (同样的 stem 位置参数、--outdir/--lag/--calib)。
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

from xrseg.common import (  # noqa: E402  (必须在 sys.path 之后)
    DEFAULT_CFG,
    EYE_H,
    EYE_W,
    INSTANCE_ID,
    INSTANCE_IDS,
    SAM2_FRAME_TENSOR_BYTES,
    SegPaths,
    human_gb,
    mem_available_bytes,
    models_cache_dir,
    normalize_prompt,
    preflight_memory,
    sam2_cuda_ext_available,
)

STAGES = ("check", "fetch", "frames", "image", "video", "render", "all")


# ------------------------------------------------------------------ 小工具


def fail(message: str) -> None:
    print(f"\n✗ {message}\n", file=sys.stderr)
    raise SystemExit(2)


def require_venv() -> None:
    missing = []
    for name in ("torch", "cv2", "sam2", "transformers", "yaml"):
        try:
            __import__(name)
        except Exception:
            missing.append(name)
    if missing:
        fail(
            f"当前解释器 {sys.executable} 缺 {'/'.join(missing)}。\n"
            f"  先 `source tools/seg_env.sh` (设 HF_ENDPOINT 等), 再用\n"
            f"  {WORK / '.venv/bin/python'} tools/seg_object.py ...\n"
            f"  环境没装就 `bash tools/seg_setup.sh`。"
        )


def parse_boxes(text: str) -> list[float]:
    parts = [float(v) for v in text.replace(" ", "").split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--box 需要 x1,y1,x2,y2 四个数")
    return parts


def load_metrics(paths: SegPaths) -> dict | None:
    import numpy as np

    if not paths.metrics_npz.is_file():
        return None
    with np.load(paths.metrics_npz, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def update_report(paths: SegPaths, payload: dict) -> Path:
    """把本次阶段的信息并进 report.json (分阶段跑也不会互相覆盖)。"""
    existing: dict = {}
    if paths.report_json.is_file():
        try:
            existing = json.loads(paths.report_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    existing.update(payload)
    paths.report_json.parent.mkdir(parents=True, exist_ok=True)
    paths.report_json.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return paths.report_json


def make_hand(stem: str, args) -> object | None:
    """右手骨架层; 拿不到就降级 --no-hand 并告警, 不中断整条链路 (plan R7)。"""
    if not args.with_hand:
        return None
    try:
        from xrseg.hand_overlay import HandLayer

        return HandLayer(stem, calib=args.calib, lag=args.lag)
    except Exception as exc:  # HandUnavailable / ImportError / 缺产物
        print(f"⚠ 手部骨架叠加不可用, 降级为 --no-hand: {exc}")
        return None


# ------------------------------------------------------------------ check


def stage_check(paths: SegPaths, args) -> None:
    import cv2
    import torch
    import torchvision
    import transformers
    import yaml
    import sam2

    from xrhand.video import count_frames, probe

    print(f"解释器      {sys.executable}")
    print(f"torch       {torch.__version__}   torchvision {torchvision.__version__}")
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability(0)
        print(f"cuda        True -> {torch.cuda.get_device_name(0)}  sm_{capability[0]}{capability[1]}"
              f"  ({torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB)")
    else:
        print("cuda        False  (会退回 CPU, 全片传播会从 1-3 min 变成十几分钟)")
    print(f"sam2        {getattr(sam2, '__version__', '(无版本号)')} @ {Path(sam2.__path__[0])}")
    print(f"transformers{transformers.__version__}   cv2 {cv2.__version__}")

    cfg_path = Path(sam2.__path__[0]) / "configs" / "sam2" / "sam2_hiera_t.yaml"
    if not cfg_path.is_file():
        fail(f"sam2 包里缺 hydra 配置 {cfg_path}")
    text = cfg_path.read_text(encoding="utf-8")
    for needle in ("feat_sizes: [64, 64]", "image_size: 1024"):
        if needle not in text:
            fail(
                f"{cfg_path} 里没有 {needle!r} —— 大概率是拿到了 HF 仓库根目录那份"
                f"同名的错误 yaml (feat_sizes: [32,32]), 与 sam2_hiera_tiny.pt 不匹配。"
            )
    print(f"sam2 config {cfg_path.name}: feat_sizes=[64,64] image_size=1024  OK")

    raw = yaml.safe_load(DEFAULT_CFG.read_text(encoding="utf-8"))
    print(f"dino        {raw['dino_model_id']}   box_threshold={raw['box_threshold']}")
    print(f"sam2 ckpt   {raw['sam2_repo_id']}/{raw['sam2_checkpoint_name']}")
    ext = sam2_cuda_ext_available()
    print(f"sam2._C     {'✓ 已编译' if ext else '✗ 未编译 (预期)'}"
          + ("" if ext else " -> 上游内置的 fill_hole_area=8 会静默跳过; 要填洞加 --fill-holes N"))
    cache = models_cache_dir()
    print(f"权重缓存    {cache}  (HF_ENDPOINT={os.environ.get('HF_ENDPOINT', '(未设)')})")

    info = probe(str(paths.mp4))
    frames = count_frames(str(paths.mp4))
    print(f"视频        {paths.mp4.name}  {info.get('width')}x{info.get('height')} "
          f"@ {info.get('fps')} fps  {frames} 帧  ({info.get('duration_s', 0):.2f} s)")
    if int(info.get("width", 0)) != EYE_W * 2 or int(info.get("height", 0)) != EYE_H:
        fail(f"视频不是预期的 {EYE_W * 2}x{EYE_H}, 双目布局假设不成立")

    tail = f"{paths.tracking.stat().st_size / 1e6:.1f} MB" if paths.tracking.is_file() else ""
    print(f"tracking    {paths.tracking.name}  "
          f"{'在位' if paths.tracking.is_file() else '缺失'}  {tail}")

    for label, path in (
        ("帧对齐", WORK / "out" / f"align_{paths.stem}.json"),
        ("标定", WORK / "out" / "calib.json"),
    ):
        print(f"{label}产物    {'✓' if path.is_file() else '✗'} {path}")

    print(f"内存        MemAvailable {human_gb(mem_available_bytes())}")
    warning = preflight_memory(frames, offload_video_to_cpu=True)
    print(f"SAM2 帧张量 预计 {human_gb(frames * SAM2_FRAME_TENSOR_BYTES)} "
          f"(offload_video_to_cpu=True -> 落系统内存)")
    if warning:
        print(f"⚠ {warning}")

    print(f"提示词       {[normalize_prompt(p) for p in (args.prompt or ['earphone case'])]}")


# ------------------------------------------------------------------ fetch


def stage_fetch(paths: SegPaths, args) -> None:
    import yaml
    from huggingface_hub import hf_hub_download, snapshot_download

    raw = yaml.safe_load(DEFAULT_CFG.read_text(encoding="utf-8"))
    cache = str(models_cache_dir()) if models_cache_dir() else None
    print(f"缓存目录 {cache}  HF_ENDPOINT={os.environ.get('HF_ENDPOINT', '(未设)')}")
    try:
        begin = time.perf_counter()
        dino_dir = snapshot_download(
            repo_id=str(raw["dino_model_id"]),
            cache_dir=cache,
            allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors"],
        )
        print(f"✓ DINO   {dino_dir}  ({time.perf_counter() - begin:.1f} s)")
        begin = time.perf_counter()
        ckpt = hf_hub_download(
            repo_id=str(raw["sam2_repo_id"]),
            filename=str(raw["sam2_checkpoint_name"]),
            cache_dir=cache,
        )
        print(f"✓ SAM2   {ckpt}  ({time.perf_counter() - begin:.1f} s)  "
              f"{Path(ckpt).stat().st_size / 1e6:.0f} MB")
    except Exception as exc:
        print(f"⚠ 下载失败: {type(exc).__name__}: {exc}")
        print("  本机直连 huggingface.co 会超时, 必须 HF_ENDPOINT=https://hf-mirror.com "
              "(source tools/seg_env.sh 已设)。仍失败时试 HF_HUB_DISABLE_XET=1 再跑一次。")


# ------------------------------------------------------------------ frames


def stage_frames(paths: SegPaths, args) -> int:
    from xrseg.frames import prepare_eye0_frames

    begin = time.perf_counter()
    count = prepare_eye0_frames(
        paths, max_frames=args.max_frames, force=args.force
    )
    print(f"✓ 帧缓存 {count} 帧 -> {paths.frames_dir}  ({time.perf_counter() - begin:.1f} s)")
    return count


# ------------------------------------------------------------------ image


def object_prompts(args) -> list[str]:
    """本次要分割的物体提示词, 顺序即 obj1..objN。

    `--prompt` 重复给就是多个物体 (今天的内层 CLI 只读 `[0]`, 所以这条重新解释对
    单物体的旧命令完全兼容)。不给就用默认那一个。
    """
    return [str(p) for p in (args.prompt or ["earphone case"])]


def object_boxes(args, count: int) -> list[list[float] | None]:
    """`--box` 按物体顺序一一对应; 没给就全是 None (走 DINO)。

    与物体数不符就报错 —— 少给一个会让第 2 个物体悄悄用 DINO, 那是最难查的一类错。
    """
    boxes = list(args.box or [])
    if not boxes:
        return [None] * count
    if len(boxes) != count:
        fail(f"--box 给了 {len(boxes)} 个, 但有 {count} 个物体 (--prompt 给了 {count} 个) —— "
             f"要么都给, 要么都不给")
    return [list(b) for b in boxes]


def stage_image(paths: SegPaths, args) -> dict:
    """提示帧上逐物体跑 DINO(+SAM2 image predictor), 落 `mask_<instance_id>.png`。

    一个 DINO engine 复用到底 (`set_image` 每帧/每物体各一次), 所以 N 个物体的开销
    基本就是 N 次前向, 不是 N 份模型。
    """
    import cv2
    import numpy as np

    from xrseg.dinosam import DINOSAM, PromptNotFoundError
    from xrseg.render import render_prompt_frame

    frame = int(args.prompt_frame)
    jpg = paths.frame_jpg(frame)
    if not jpg.is_file():
        print(f"帧缓存里没有 {jpg.name}, 先跑 frames 阶段")
        stage_frames(paths, args)
    if not jpg.is_file():
        fail(f"还是缺 {jpg}; --max-frames={args.max_frames} 太小了?")

    prompts = object_prompts(args)
    boxes = object_boxes(args, len(prompts))
    instance_ids = list(INSTANCE_IDS[:len(prompts)])
    print(f"提示帧 {frame}  物体 {list(zip(instance_ids, prompts))}  "
          f"box_threshold={args.box_threshold}")

    image_np = cv2.imread(str(jpg))
    if image_np is None:
        fail(f"读不了 {jpg}")
    objects: dict[str, dict] = {}
    # 一个 DINO engine 到底: 每物体只是再 set_image + 一次前向, 不重复加载模型。
    engine = DINOSAM(str(args.cfg))
    engine.cfg.box_threshold = float(args.box_threshold)
    try:
        for instance_id, raw_prompt, box in zip(instance_ids, prompts, boxes):
            prompt = normalize_prompt(raw_prompt)
            print(f"║ [{instance_id}] 提示词 {prompt!r}")
            begin = time.perf_counter()
            if box is not None:
                # 手工框: 完全绕过 DINO (plan R3 的兜底), 直接在已 set_image 的 predictor 上出 mask
                engine.engine.predictor.set_image(image_np)
                np_boxes = np.asarray([box], dtype=np.float32)
                masks, _, _ = engine.engine.predictor.predict(box=np_boxes,
                                                             multimask_output=False)
                mask = (np.any(masks.squeeze(), axis=0) if masks.ndim > 3 else masks.squeeze())
                mask = (mask.astype(np.uint8) * 255)
                info = {
                    "mask": mask,
                    "boxes": np_boxes,
                    "box_confidences": np.asarray([float("nan")]),
                    "avg_confidence": float("nan"),
                    "latency_s": time.perf_counter() - begin,
                }
                print(f"║ [{instance_id}] 用手工框绕过 DINO: {[int(round(v)) for v in box]}")
            else:
                try:
                    info = engine.segment(jpg, prompt,
                                          save_mask_to=paths.prompt_mask(instance_id))
                except PromptNotFoundError as exc:
                    fail(f"[{instance_id}] {exc}")

            mask = info["mask"]
            if mask is None:
                fail(f"[{instance_id}] SAM2 没给出 mask")
            # 手工框那条路不走 engine.segment, 所以 mask 得自己落盘。
            if box is not None:
                if not cv2.imwrite(str(paths.prompt_mask(instance_id)), mask):
                    fail(f"写不了 {paths.prompt_mask(instance_id)}")
            area = int(np.count_nonzero(mask > 127))
            np_boxes = np.asarray(info["boxes"]).reshape(-1, 4)
            confidences = np.asarray(info["box_confidences"], dtype=float).reshape(-1)
            print(f"║ [{instance_id}] DINO boxes={len(np_boxes)}  "
                  f"conf={np.round(confidences, 3).tolist()}  "
                  f"avg={float(info['avg_confidence']):.3f}   mask {area} px  "
                  f"({float(info['latency_s']) * 1000:.0f} ms)")
            for one, confidence in zip(np_boxes, confidences):
                corner = [int(round(float(v))) for v in one]
                print(f"║   [{instance_id}] 框 (eye0 像素) x1,y1,x2,y2 = {corner}  "
                      f"conf {confidence:.3f}")
            if area == 0:
                fail(f"[{instance_id}] SAM2 给出的 mask 是空的 —— 框可能落在没有目标的地方, "
                     f"换帧或改手工框。")

            render_prompt_frame(
                paths,
                frame_rgb=image_np[:, :, ::-1].copy(),
                mask=mask,
                boxes=info["boxes"],
                confidences=info["box_confidences"],
                prompt=prompt,
                frame=frame,
                box_threshold=args.box_threshold,
                avg_confidence=info["avg_confidence"],
                latency_s=info["latency_s"],
                instance_id=instance_id,
            )
            objects[instance_id] = {
                "instance_id": instance_id,
                "prompt": raw_prompt,
                "normalized_prompt": prompt,
                "prompt_frame": frame,
                "box_threshold": float(args.box_threshold),
                "manual_box": box,
                "boxes": np.asarray(info["boxes"]).tolist(),
                "box_confidences": [
                    float(v) for v in np.asarray(info["box_confidences"]).reshape(-1)
                ],
                "avg_confidence": info["avg_confidence"],
                "mask_area_px": area,
                "mask_shape": list(mask.shape),
                "mask_path": str(paths.prompt_mask(instance_id)),
                "sam2_input_colorspace": "bgr (与上游 process_and_save 一致, 不是 RGB)",
                "latency_s": round(float(info["latency_s"]), 3),
            }
            print(f"✓ [{instance_id}] 提示帧 mask {area} px -> {paths.prompt_mask(instance_id)}")
    finally:
        try:
            engine.cleanup()
        except Exception:
            pass

    first = instance_ids[0]
    payload = {
        "stage_image": {
            # 顶层这几个键是**第一个物体**的 (单物体时就是全部), 老读法不变。
            **objects[first],
            "instance_ids": instance_ids,
            "objects": objects,
        }
    }
    if not getattr(args, "no_artifacts", False):
        update_report(paths, payload)
    print(f"→ 现在看 {paths.prompt_frame_png}: 确认框住的是目标物体、没吃掉手指, "
          f"再跑 --stage video")
    return payload["stage_image"]


def stage_prompt_scan(paths: SegPaths, args) -> None:
    """在若干帧上各跑一次 DINO, 看提示词在哪儿框得最稳 (plan R3)。"""
    import cv2

    from xrseg.dinosam import DINOSAM, PromptNotFoundError

    prompts = [normalize_prompt(p) for p in args.prompt or ["earphone case"]]
    count = len(sorted(paths.frames_dir.glob("*.jpg")))
    if count == 0:
        fail(f"{paths.frames_dir} 是空的, 先跑 --stage frames")
    frames = sorted({0, count // 4, count // 2, 3 * count // 4, count - 1})
    engine = DINOSAM(str(args.cfg))
    engine.cfg.box_threshold = float(args.box_threshold)
    try:
        for prompt in prompts:
            print(f"\n提示词 {prompt!r}  box_threshold={args.box_threshold}")
            for frame in frames:
                jpg = paths.frame_jpg(frame)
                image_np = cv2.imread(str(jpg))
                engine.engine.predictor.set_image(image_np)
                try:
                    _, avg_conf, boxes, confs = engine.engine.predict_frame_internal(
                        image_np, prompt
                    )
                    print(f"  frame {frame:4d}: boxes={len(boxes):2d}  avg_conf={avg_conf:.3f}  "
                          f"max={confs.max():.3f}")
                except PromptNotFoundError:
                    print(f"  frame {frame:4d}: boxes= 0  (该帧框不到)")
    finally:
        engine.cleanup()
    print("\n挑 boxes>=1 且 avg_conf 最高的帧作 --prompt-frame。")


# ------------------------------------------------------------------ video


def stage_video(paths: SegPaths, args) -> dict:
    import numpy as np
    import yaml

    from xrseg.render import render_quality_chart
    from xrseg.sam2_video import run_sam2_video_masks

    prompts = object_prompts(args)
    instance_ids = list(INSTANCE_IDS[:len(prompts)])
    initial_masks = []
    for instance_id in instance_ids:
        mask_path = paths.prompt_mask(instance_id)
        if not mask_path.is_file():
            fail(f"缺提示帧 mask {mask_path}, 先跑 --stage image")
        initial_masks.append((instance_id, mask_path))
    count = len(sorted(paths.frames_dir.glob("*.jpg")))
    if count == 0:
        fail(f"{paths.frames_dir} 是空的, 先跑 --stage frames")

    offload = not args.video_in_vram
    if offload:
        warning = preflight_memory(count, offload_video_to_cpu=True)
        if warning and not args.force:
            fail(f"{warning}\n  (确认要硬跑就加 --force)")

    hand = make_hand(paths.stem, args)
    extra_masks = []
    for spec in args.extra_mask or []:
        index, _, path = spec.partition("=")
        if not path:
            fail(f"--extra-mask 要写成 N=path 的形式, 收到 {spec!r}")
        extra_masks.append((int(index), Path(path)))

    raw = yaml.safe_load(Path(args.cfg).read_text(encoding="utf-8"))
    fps = float(args.fps or 30.0)
    result = run_sam2_video_masks(
        paths,
        initial_masks=initial_masks,
        fps=fps,
        initial_frame=int(args.prompt_frame),
        cfg_path=Path(args.cfg),
        sam2_config=args.sam2_config,
        sam2_checkpoint=args.sam2_ckpt,
        extra_masks=extra_masks,
        fill_hole_area=int(args.fill_holes),
        keep_largest_component=args.keep_largest_component,
        offload_video_to_cpu=offload,
        overlap_fn=hand.overlap_ratio if hand is not None else None,
        frame_count=count,
    )
    for instance_id, metrics in result["objects"].items():
        print(f"\n质量指标 [{instance_id}] (区间 {metrics['metrics_scope']}, "
              f"提示帧之前的 {metrics['leading_empty_frames']} 帧是空帧不计入):")
        for key in (
            "nonempty_ratio",
            "area_ratio_p05",
            "area_ratio_p95",
            "score_median",
            "tiny_mask_ratio",
            "multi_component_frames",
        ):
            print(f"  {key:24s} {metrics[key]}")
        print(f"  {'empty_mask_runs':24s} {metrics['empty_mask_runs']}")
    first = result["metrics"]
    if first.get("hand_overlap_median") is not None:
        print(f"  {'hand_overlap_median':24s} {first['hand_overlap_median']} "
              f"({instance_ids[0]} 的 mask ∩ 右手凸包)")
    if len(instance_ids) > 1:
        print(f"  {'overlap_pixels_p95':24s} {first['overlap_pixels_p95']} "
              f"(逐帧被 >=2 个物体同时覆盖的像素)")

    prompt = ""
    if paths.prompt_json.is_file():
        prompt = json.loads(paths.prompt_json.read_text(encoding="utf-8")).get(
            "normalized_prompt", ""
        )
    # --no-artifacts: 纯诊断产物 (质量曲线 + 报告 json) 不落盘。下面的指标行照旧打印 ——
    # 那是给人看的, 不产生文件。
    if not getattr(args, "no_artifacts", False):
        metrics_dict = load_metrics(paths) or {}
        render_quality_chart(
            paths,
            prompt=prompt or INSTANCE_ID,
            areas=metrics_dict.get("areas", np.zeros((count, len(instance_ids)))),
            scores=metrics_dict.get("scores", np.zeros((count, len(instance_ids)))),
            hand_overlap=metrics_dict.get("hand_overlap"),
            instance_ids=metrics_dict.get("instance_ids"),
            fps=fps,
            prompt_frame=int(args.prompt_frame),
        )

        update_report(
            paths,
            {
                "method": result["report"]["method"],
                "device": result["report"]["device"],
                "frames": result["report"]["frames"],
                "ms_per_frame": result["report"]["ms_per_frame"],
                "sam2_config": args.sam2_config or str(raw["sam2_config"]),
                "sam2_checkpoint": args.sam2_ckpt or str(raw["sam2_checkpoint_name"]),
                "hand_overlay": bool(hand is not None),
                "prompt": prompt,
                "outputs_are_diagnostic_only": True,
                "stage_video": result["report"],
            },
        )
        print(f"\n✓ 报告 {paths.report_json}")
    return result


# ------------------------------------------------------------------ render


def stage_render(paths: SegPaths, args) -> None:
    from xrseg.render import auto_still_frames, render_overlay_video, render_stills

    arrays = load_metrics(paths)
    if arrays is None:
        fail(f"缺 {paths.metrics_npz}, 先跑 --stage video")
    areas = arrays["areas"]
    instance_ids = [str(v) for v in arrays.get("instance_ids", [INSTANCE_ID])]
    scores = arrays["scores"]
    hand = make_hand(paths.stem, args)
    fps = float(args.fps or 30.0)

    prompt = ""
    if paths.prompt_json.is_file():
        prompt = json.loads(paths.prompt_json.read_text(encoding="utf-8")).get(
            "normalized_prompt", ""
        )

    render_overlay_video(
        paths,
        prompt=prompt,
        fps=fps,
        hand=hand,
        scores=scores,
        instance_ids=instance_ids,
        frame_count=int(areas.shape[0]),
        prompt_frame=int(args.prompt_frame),
        radius=int(args.radius),
    )
    if not args.no_stills:
        # 静帧与叠加视频按**并集**取帧: 任一物体出现/消失的时刻都是要看的那一帧。
        frames = auto_still_frames(areas.any(axis=1), int(args.prompt_frame))
        render_stills(
            paths,
            frames=frames,
            prompt=prompt,
            fps=fps,
            hand=hand,
            scores=scores,  # (帧, 物体); render_stills 目前只按 masks 画, 收着备用
            instance_ids=instance_ids,
            radius=int(args.radius),
        )
    if not getattr(args, "no_artifacts", False):
        update_report(paths, {"stage_render": {"video": str(paths.overlay_video),
                                               "hand_overlay": bool(hand is not None)}})


# ------------------------------------------------------------------ main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="耳机壳分割 (DINO->SAM2) + 可视化",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("stem", nargs="*", help="采集 stem, 可给多个 (默认 111300 那条)")
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--prompt", action="append",
                        help="GroundingDINO 文本提示; **重复给 = 多个物体** "
                             "(第 1 个是 obj1, 第 2 个是 obj2, 顺序即 state 里的拼接顺序)。"
                             "只给 1 个时与旧行为逐位一致")
    parser.add_argument("--prompt-frame", type=int, default=50,
                        help="提示帧序号 (默认 50: 两条采集的前若干帧画面里还没有耳机壳, "
                             "第 0 帧框不到东西; 提示帧之前的帧写全零 mask)。"
                             "**全局单值**: 所有物体共享同一个提示帧")
    parser.add_argument("--box", type=parse_boxes, action="append", default=None,
                        help="手工框 x1,y1,x2,y2 (给了就完全绕过 DINO); "
                             "多个物体按顺序重复给, 个数必须与 --prompt 相同")
    parser.add_argument("--box-threshold", type=float, default=0.3)
    parser.add_argument("--prompt-scan", action="store_true",
                        help="在 5 个帧上各跑一次 DINO, 打印框数/置信度后退出")
    parser.add_argument("--max-frames", type=int, default=None, help="只处理前 N 帧 (冒烟用)")
    parser.add_argument("--cfg", default=str(DEFAULT_CFG))
    parser.add_argument("--sam2-config", default=None, help="覆盖 hydra 配置 (默认用 step2 那对)")
    parser.add_argument("--sam2-ckpt", default=None, help="覆盖 SAM2 权重路径")
    parser.add_argument("--extra-mask", action="append", default=None,
                        help="中途补点 N=mask.png, 可重复 (默认空 = 与 step2 一致)")
    parser.add_argument("--fill-holes", type=int, default=0,
                        help="填掉面积<=N 的内部空洞 (0=关; sam2._C 没编译时上游那条路会静默失效)")
    parser.add_argument("--keep-largest-component", action="store_true",
                        help="每帧只留最大连通域 (mask 裂开时用)")
    parser.add_argument("--video-in-vram", action="store_true",
                        help="offload_video_to_cpu=False, 帧张量放显存 (RAM 不够时的兜底)")
    parser.add_argument("--with-hand", dest="with_hand", action="store_true", default=True,
                        help="叠加右手骨架")
    parser.add_argument("--no-hand", dest="with_hand", action="store_false")
    parser.add_argument("--no-stills", action="store_true", help="render 阶段不出静帧")
    parser.add_argument("--no-artifacts", action="store_true", default=False,
                        help="跳过纯诊断产物 (quality.png / report.json); mask + metrics 不受影响")
    parser.add_argument("--calib", default=None, help="calib.json (默认 out/calib.json)")
    parser.add_argument("--lag", type=int, default=None, help="覆盖 align_*.json 的 lag")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--radius", type=int, default=4, help="手部关节半径")
    parser.add_argument("--outdir", default=str(WORK / "out"))
    parser.add_argument("--force", action="store_true", help="重算已有产物 / 忽略内存告警")
    return parser


def run_one(stem: str, args) -> int:
    paths = SegPaths.for_stem(stem, args.outdir)
    paths.outdir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 78}\n=== {stem}  ->  {paths.outdir}\n{'=' * 78}")
    require_venv()

    if args.stage == "check":
        stage_check(paths, args)
        return 0
    if args.stage == "fetch":
        stage_fetch(paths, args)
        return 0
    if args.prompt_scan:
        stage_prompt_scan(paths, args)
        return 0

    if args.stage in ("frames", "all"):
        stage_frames(paths, args)
    if args.stage in ("image", "all"):
        info = stage_image(paths, args)
        paths.prompt_json.write_text(
            json.dumps(info, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
    if args.stage in ("video", "all"):
        stage_video(paths, args)
    if args.stage in ("render", "all"):
        stage_render(paths, args)
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    stems = args.stem or ["20260920_111300"]
    for stem in stems:
        try:
            run_one(stem, args)
        except KeyboardInterrupt:
            raise
        except BaseException as exc:  # 单条失败不要拖累另一条
            if len(stems) == 1:
                raise
            traceback.print_exc()
            print(f"✗ {stem} 失败 ({type(exc).__name__}: {exc}), 继续下一条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
