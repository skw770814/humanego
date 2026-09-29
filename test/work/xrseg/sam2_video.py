"""SAM2 视频传播: 把提示帧上的 mask 推到全片 (step2 链路的下半段)。

复制自 ego_relation_policy/src/ego_relation/s2_object_relations/sam2_video.py。

原样保留:
  - `_COLORS`
  - `_mask_sequence_metrics` / `_true_runs` (逐行照抄)
  - `run_sam2_video_masks` 里的传播循环: init_state(offload_video_to_cpu=True,
    offload_state_to_cpu=(device=="cuda"), async_loading_frames=False) →
    add_new_mask → torch.autocast("cuda", bfloat16) → propagate_in_video →
    `logits > 0` 取支持集 → 写 0/255 PNG → 面积/质心/sigmoid(score)

适配 (相对上游):
  1. 删 `from ego_relation.config import ProjectConfig` —— 那个包不在本仓库。
  2. 目录不再由 episode_dir 推导, 改用 `xrseg.common.SegPaths`。
  3. `hf_hub_download(local_files_only=True)` → False + cache_dir (权重还没下过)。
  4. `_prepare_jpeg_frames` 不要了 —— 本仓库的帧缓存由 `xrseg.frames` 直接从 mp4
     解码左半眼写出 (上游是从 staged 的 rgb.png 拷), 覆盖同一件事。
  5. 多目标: 初始 mask 由调用方按 `[(instance_id, mask_path)]` 的有序列表给, 不再读
     object_catalog.json。一个 predictor state + 每物体一次 `add_new_mask` (`obj_id` 1..N)
     + 一次 `propagate_in_video` (与上游 `:419-451` 同一套); 指标/质心/面积都是
     `(frame_count, object_count)`, `report["objects"]` 逐物体一份。给一个物体时与
     单目标逐位一致。
  6. `_render_metrics_chart` / `_render_video` / `render_sam2_pose_video` 移到
     render.py —— 前两个要加"右手骨架"这一层, 后者要 6DoF pose (本仓库没有)。
  7. fps 由 `cfg.timeline.control_hz` 改成显式入参。
  8. 新增 `fill_hole_area` / `keep_largest_component` 后处理 (plan R4/R5 兜底) 与
     `overlap_fn` 回调 (每帧算 mask∩右手凸包, R5 的诊断量), 默认都不改变上游行为。
  9. 提示帧 > 0 时: 提示帧**之前**的帧写全零 mask (目标那时还没出现, 复制提示帧的
     mask 回去等于凭空画一块), 质量指标也只在 [提示帧, 末帧] 这一段上算 —— 否则
     `_mask_sequence_metrics` 的 initial = areas[0] = 0 会让所有 area_ratio 爆掉。
     提示帧 = 0 时这两处都与上游逐位一致。
"""

from __future__ import annotations

from contextlib import nullcontext
import gc
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np
import yaml

from xrseg.common import (
    DEFAULT_CFG,
    INSTANCE_ID,
    MASK_COLORS,
    SegPaths,
    models_cache_dir,
    sam2_cuda_ext_available,
)


_COLORS = MASK_COLORS


def _mask_sequence_metrics(areas: np.ndarray, centroids: np.ndarray) -> dict[str, Any]:
    areas = np.asarray(areas, dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64)
    initial = max(float(areas[0]), 1.0)
    ratios = areas / initial
    valid_centers = np.isfinite(centroids).all(axis=1)
    jumps = np.linalg.norm(np.diff(centroids[valid_centers], axis=0), axis=1)
    return {
        "initial_area_px": int(areas[0]),
        "nonempty_ratio": float(np.mean(areas > 0)),
        "area_ratio_median": float(np.median(ratios)),
        "area_ratio_p05": float(np.percentile(ratios, 5)),
        "area_ratio_p95": float(np.percentile(ratios, 95)),
        "tiny_mask_ratio": float(np.mean(ratios < 0.10)),
        "large_mask_ratio": float(np.mean(ratios > 4.0)),
        "centroid_jump_p95_px": float(np.percentile(jumps, 95)) if len(jumps) else None,
    }


def _true_runs(mask: np.ndarray) -> list[list[int]]:
    padded = np.r_[False, np.asarray(mask, dtype=bool), False]
    changes = np.diff(padded.astype(np.int8))
    return [
        [int(start), int(end - 1)]
        for start, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1), strict=True)
    ]


def _fill_small_holes(support: np.ndarray, max_area: int) -> np.ndarray:
    """填掉面积 <= max_area 的内部空洞 (scipy 实现, 供 --fill-holes 用)。

    为什么要自己写: `build_sam2_video_predictor` 的 apply_postprocessing 会给
    `++model.fill_hole_area=8`, 于是 SAM2 每帧都调 `fill_holes_in_mask_scores`;
    那个函数内部 `from sam2 import _C`, 而本机没编译 CUDA 扩展 -> ImportError 被它
    自己的 `except Exception` 吞掉 -> warn 一次、**不填洞** (plan R4, 已在
    sam2/utils/misc.py:325-334 与 utils/misc.py:61 对照确认)。
    这里用 scipy 复刻同样的语义: 只填**不与画面边界相连**的小连通域。
    """
    from scipy import ndimage

    inverse = ~support
    labels, count = ndimage.label(inverse)
    if count == 0:
        return support
    border = np.unique(
        np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    )
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    filled = support.copy()
    for label in range(1, count + 1):
        if label in border or sizes[label] > max_area:
            continue
        filled[labels == label] = True
    return filled


def _largest_component(support: np.ndarray) -> tuple[np.ndarray, int]:
    """只留最大连通域; 返回 (mask, 连通域个数)。对应上游 `_expand_instances` 的思路。"""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        support.astype(np.uint8), connectivity=8
    )
    if count <= 2:  # 0=背景, 1=唯一目标
        return support, max(count - 1, 0)
    areas = stats[1:, cv2.CC_STAT_AREA]
    return labels == int(np.argmax(areas)) + 1, count - 1


def run_sam2_video_masks(
    paths: SegPaths,
    *,
    initial_masks: Sequence[tuple[str, Path]],
    fps: float,
    initial_frame: int = 0,
    cfg_path: Path = DEFAULT_CFG,
    sam2_config: str | None = None,
    sam2_checkpoint: str | None = None,
    extra_masks: Sequence[tuple[int, Path]] = (),
    fill_hole_area: int = 0,
    keep_largest_component: bool = False,
    offload_video_to_cpu: bool = True,
    overlap_fn: Callable[[int, np.ndarray], float] | None = None,
    frame_count: int | None = None,
    progress_every: int = 25,
) -> dict[str, Any]:
    """把 frame `initial_frame` 的 mask 传播到全片, 逐帧写 `masks/<instance_id>/%05d.png`。

    `initial_masks` 是 **[(instance_id, mask_path)] 的有序列表** —— 一个 predictor state +
    每个物体一次 `add_new_mask` (`obj_id` = 1..N) + **一次** `propagate_in_video`, 与参考
    `sam2_video.py:419-451` 同一套。顺序即 obj1..objN, 也就是 state 里物体块的拼接顺序。

    `extra_masks` 是 [(frame_idx, mask_path)] 形式的中途补点 (`add_new_mask` 再来一次),
    默认空 = 与 step2 完全一致 (只在首帧给一次提示); 补点只作用于**第一个**物体, 与上游
    单目标时的语义逐字一致。
    `overlap_fn(frame, support) -> float` 由调用方注入 (算 mask 与右手凸包的重叠率),
    保持本模块不依赖手部那一套; 多物体时只对第一个物体算 (它就是原来的那个量)。
    """
    import torch
    from huggingface_hub import hf_hub_download
    from sam2.build_sam import build_sam2_video_predictor

    frame_dir = paths.frames_dir
    if not frame_dir.is_dir():
        raise FileNotFoundError(f"缺少帧缓存目录, 先跑 --stage frames: {frame_dir}")
    if frame_count is None:
        frame_count = len(sorted(frame_dir.glob("*.jpg")))
    if frame_count <= 0:
        raise RuntimeError(f"{frame_dir} 里没有 .jpg 帧")
    if not 0 <= int(initial_frame) < frame_count:
        raise ValueError(
            f"--prompt-frame={initial_frame} 超出范围 (帧缓存共 {frame_count} 帧, "
            f"合法范围 0..{frame_count - 1})"
        )
    if not initial_masks:
        raise ValueError("initial_masks 是空的 —— 至少给一个物体")

    raw_config = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    config_name = sam2_config or str(raw_config["sam2_config"])
    if sam2_checkpoint:
        checkpoint = sam2_checkpoint
    else:
        checkpoint = hf_hub_download(
            repo_id=str(raw_config["sam2_repo_id"]),
            filename=str(raw_config["sam2_checkpoint_name"]),
            cache_dir=str(models_cache_dir()) if models_cache_dir() else None,
            local_files_only=False,
        )

    mask_root = paths.masks_dir
    # 逐物体清旧 mask (参考 `:401-405`): 只认这次要跑的这几个 instance_id,
    # 不去动别的目录 —— 免得把调用方还想要的东西删掉。
    for instance_id, _ in initial_masks:
        object_dir = paths.instance_mask_dir(instance_id)
        object_dir.mkdir(parents=True, exist_ok=True)
        for stale in object_dir.glob("*.png"):
            stale.unlink()

    resolved: list[tuple[str, np.ndarray]] = []
    for instance_id, mask_path in initial_masks:
        initial_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if initial_mask is None:
            raise FileNotFoundError(mask_path)
        resolved.append((str(instance_id), initial_mask))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"║ SAM2 video: device={device} frames={frame_count} 物体 "
          f"{[i for i, _ in resolved]} config={config_name}")
    # sam2._C 没编译时, 每帧都会走一次 fill_holes_in_mask_scores 的 except 分支并 warn;
    # 内容完全一样, 静音掉, 结论记进 report (要不要填洞由 --fill-holes 决定)。
    warnings.filterwarnings(
        "ignore", message=".*Skipping the post-processing step.*", category=UserWarning
    )
    predictor = build_sam2_video_predictor(config_name, checkpoint, device=device)
    state = predictor.init_state(
        video_path=str(frame_dir),
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=device == "cuda",
        async_loading_frames=False,
    )

    # 一个 state, 每个物体一次 add_new_mask, obj_id = 1..N (参考 `:419-426`)。
    object_id_map: dict[int, str] = {}
    for numeric_id, (instance_id, initial_mask) in enumerate(resolved, start=1):
        predictor.add_new_mask(
            state, frame_idx=int(initial_frame), obj_id=numeric_id, mask=initial_mask > 127
        )
        object_id_map[numeric_id] = instance_id
    first_instance = resolved[0][0]
    for extra_frame, extra_path in extra_masks:
        extra = cv2.imread(str(extra_path), cv2.IMREAD_GRAYSCALE)
        if extra is None:
            raise FileNotFoundError(extra_path)
        predictor.add_new_mask(
            state, frame_idx=int(extra_frame), obj_id=1, mask=extra > 127
        )
        print(f"║ 中途补点: frame {extra_frame} <- {extra_path} ({first_instance})")

    object_count = len(resolved)
    index_by_instance = {instance_id: index for index, (instance_id, _) in enumerate(resolved)}
    areas = np.zeros((frame_count, object_count), dtype=np.int32)
    scores = np.zeros((frame_count, object_count), dtype=np.float32)
    centroids = np.full((frame_count, object_count, 2), np.nan, dtype=np.float32)
    components = np.zeros((frame_count, object_count), dtype=np.int32)
    overlaps = np.full(frame_count, np.nan, dtype=np.float32)

    context = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
    t_start = time.perf_counter()
    try:
        with context:
            for frame, output_ids, mask_logits in predictor.propagate_in_video(state):
                index = int(frame)
                for output_index, numeric_id in enumerate(output_ids):
                    instance_id = object_id_map[int(numeric_id)]
                    object_index = index_by_instance[instance_id]
                    logits = mask_logits[output_index, 0]
                    support_tensor = logits > 0
                    support = support_tensor.detach().cpu().numpy()
                    score = float(
                        torch.sigmoid(logits[support_tensor]).mean().item()
                    ) if bool(support_tensor.any()) else 0.0

                    if fill_hole_area > 0:
                        support = _fill_small_holes(support, fill_hole_area)
                    if keep_largest_component:
                        support, component_count = _largest_component(support)
                    else:
                        component_count = cv2.connectedComponents(
                            support.astype(np.uint8), connectivity=8
                        )[0] - 1

                    components[index, object_index] = component_count
                    mask = support.astype(np.uint8) * 255
                    mask_path = paths.mask(index, instance_id)
                    if not cv2.imwrite(str(mask_path), mask):
                        raise RuntimeError(f"无法写入 mask: {mask_path}")
                    area = int(support.sum())
                    areas[index, object_index] = area
                    if area:
                        ys, xs = np.nonzero(support)
                        centroids[index, object_index] = [float(xs.mean()), float(ys.mean())]
                        scores[index, object_index] = score
                        # 手部重叠只对第一个物体算 —— 它就是单物体时的那个诊断量,
                        # 多物体下再有第二个名字反而会让旧读法悄悄改义。
                        if overlap_fn is not None and instance_id == first_instance:
                            overlaps[index] = float(overlap_fn(index, support))
                if progress_every and index % progress_every == 0:
                    elapsed = time.perf_counter() - t_start
                    print(
                        f"║   frame {index:4d}/{frame_count - 1}  "
                        + "  ".join(
                            f"{instance_id} {int(areas[index, j]):6d}"
                            for instance_id, j in index_by_instance.items()
                        )
                        + f"  {elapsed / max(index + 1, 1) * 1000:.0f} ms/frame"
                    )
    finally:
        del state
        del predictor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    elapsed = time.perf_counter() - t_start

    # 提示帧不在 0 时, 前面的帧 propagate_in_video 不覆盖 (它从提示帧往后推)。
    # 这些帧写**全零 mask**, 不复制提示帧的 mask 回去: 把提示帧选在 N 就意味着
    # "目标到第 N 帧才出现"(耳机壳就是这样, 前 50 帧画面里还没有它), 复制等于在
    # 目标不存在的地方凭空画一块。空着并由 empty_mask_runs 如实报出来。
    leading_empty = 0
    for frame in range(int(initial_frame)):
        for instance_id, initial_mask in resolved:
            mask_path = paths.mask(frame, instance_id)
            if not mask_path.is_file():
                if not cv2.imwrite(str(mask_path), np.zeros_like(initial_mask)):
                    raise RuntimeError(f"无法写入前导空 mask: {mask_path}")
                areas[frame, index_by_instance[instance_id]] = 0
        leading_empty += 1

    missing = [
        str(paths.mask(f, instance_id))
        for f in range(frame_count)
        for instance_id, _ in resolved
        if not paths.mask(f, instance_id).is_file()
    ]
    if missing:
        raise RuntimeError(f"SAM2 Video 缺少 {len(missing)} 个输出，首个: {missing[0]}")

    # 指标只在**真正传播过的区间**上算: 提示帧之前是硬写的空帧, 混进去会让
    # _mask_sequence_metrics 的 initial = areas[0] = 0, 所有 area_ratio 爆成天文数字。
    # 那个函数本身一行没改, 只是喂给它的切片变了 —— 提示帧 = 0 时切片就是全长, 与上游一致。
    first = int(initial_frame)
    scope_overlaps = overlaps[first:]
    object_metrics: dict[str, dict[str, Any]] = {}
    for instance_id, index in index_by_instance.items():
        object_metrics[instance_id] = {
            "category": instance_id,
            **_mask_sequence_metrics(areas[first:, index], centroids[first:, index]),
            "score_median": float(np.median(scores[first:, index])),
            "empty_mask_runs": _true_runs(areas[first:, index] == 0),
            "multi_component_frames": int(np.count_nonzero(components[first:, index] > 1)),
            "leading_empty_frames": leading_empty,
            "metrics_scope": f"frames {first}..{frame_count - 1}",
        }
    metrics = object_metrics[first_instance]
    metrics.update({
        "hand_overlap_median": (
            float(np.nanmedian(scope_overlaps)) if np.isfinite(scope_overlaps).any() else None
        ),
        "hand_overlap_p95": (
            float(np.nanpercentile(scope_overlaps, 95))
            if np.isfinite(scope_overlaps).any() else None
        ),
        # 逐物体的 mask 相交像素 (参考 `sam2_video.py:468-478` 的 overlap_pixels):
        # 两个物体抢同一片像素 = 分割把它们糊在一起了, 该看的诊断量。
        "overlap_pixels_p95": (
            float(np.percentile(_overlap_pixels(paths, frame_count, [i for i, _ in resolved]), 95))
            if object_count > 1 else 0.0
        ),
    })
    report = {
        "method": "SAM2VideoPredictor frame-0 mask propagation",
        "device": device,
        "frames": frame_count,
        "fps": fps,
        "instance_ids": [i for i, _ in resolved],
        # 单物体时仍是 {obj1: {...}}; 第一个物体的那份同时提到顶层 (metrics) 供旧读法使用。
        "objects": object_metrics,
        "initial_frame": int(initial_frame),
        "extra_masks": [[int(f), str(p)] for f, p in extra_masks],
        "leading_empty_frames": leading_empty,
        "metrics_scope": f"frames {first}..{frame_count - 1}",
        "fill_hole_area": int(fill_hole_area),
        "sam2_builtin_fill_hole_area": 8 if sam2_cuda_ext_available() else (
            "8 (但 sam2._C 未编译 -> 上游那条路静默跳过, 见 plan R4)"
        ),
        "fill_hole_impl": "scipy.ndimage (--fill-holes)" if fill_hole_area > 0 else None,
        "keep_largest_component": bool(keep_largest_component),
        "offload_video_to_cpu": bool(offload_video_to_cpu),
        "seconds": round(elapsed, 2),
        "ms_per_frame": round(elapsed / max(frame_count, 1) * 1000, 1),
        "outputs_are_diagnostic_only": True,
    }
    paths.outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        paths.metrics_npz,
        instance_ids=np.asarray([i for i, _ in resolved]),
        areas=areas,
        scores=scores,
        centroids=centroids,
        components=components,
        hand_overlap=overlaps,
    )
    # report.json 由 CLI 汇总 (还有其他阶段的信息), 这里只把本阶段的片段返回
    return {"metrics": metrics, "objects": object_metrics, "report": report,
            "paths": {"masks": mask_root, "metrics": paths.metrics_npz},
            "areas": areas, "scores": scores}


def _overlap_pixels(paths: SegPaths, frame_count: int, instance_ids: list[str]) -> np.ndarray:
    """逐帧「被 >= 2 个物体同时覆盖」的像素数 (参考 `sam2_video.py:468-478`)。"""
    out = np.zeros(frame_count, dtype=np.int32)
    for frame in range(frame_count):
        occupancy = None
        for instance_id in instance_ids:
            mask = cv2.imread(str(paths.mask(frame, instance_id)), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            support = (mask > 127).astype(np.uint8)
            occupancy = support if occupancy is None else occupancy + support
        if occupancy is not None:
            out[frame] = int(np.count_nonzero(occupancy > 1))
    return out
