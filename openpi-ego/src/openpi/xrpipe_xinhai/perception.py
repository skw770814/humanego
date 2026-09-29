"""Resident DINO/SAM2 models and causal pipeline-compatible object pose tracking."""
from __future__ import annotations

from collections import deque
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np
from PIL import Image

from .config import pipeline_imports


class ResidentPerception:
    """Loads each model once. Reset only discards tracking state, never weights.

    Streaming adapter targets SAM2VideoPredictor's per-object state layout (the
    version installed with test/pipeline). Unsupported layouts fail explicitly.
    Frame images are supplied in memory after one public init_state call.
    """

    def __init__(self, args, prompts):
        pipeline_imports()  # also pins the bundled SAM2 1.1.0 source
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        from sam2.build_sam import build_sam2_video_predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        start = time.perf_counter()
        if not args.sam2_checkpoint or not Path(args.sam2_checkpoint).is_file():
            raise ValueError("--sam2-checkpoint must point to a local SAM2 checkpoint")
        self.torch, self.device = torch, args.perception_device
        self.prompts = tuple(" ".join(p.strip().lower().replace(".", " ").split()) + " ." for p in prompts)
        self.threshold = args.box_threshold
        self.processor = AutoProcessor.from_pretrained(args.dino_checkpoint, local_files_only=True)
        self.dino = AutoModelForZeroShotObjectDetection.from_pretrained(args.dino_checkpoint, local_files_only=True).to(self.device).eval()
        self.video = build_sam2_video_predictor(args.sam2_config, args.sam2_checkpoint, device=self.device)
        self.image = SAM2ImagePredictor(self.video)  # SAME backbone/weights, not another model
        self.load_seconds = time.perf_counter() - start
        self.reset()

    def reset(self):
        self.state = None
        self.frame = -1
        self.initial_areas = None
        self.last_centres = None
        self.last_stamp = None
        self.image.reset_predictor()

    def _tensor(self, rgb):
        a = np.asarray(Image.fromarray(rgb).resize((self.video.image_size, self.video.image_size)))
        t = self.torch.from_numpy(a.copy()).permute(2, 0, 1).float() / 255.
        mean = self.torch.tensor([.485, .456, .406])[:, None, None]
        std = self.torch.tensor([.229, .224, .225])[:, None, None]
        return (t - mean) / std

    def _start(self, rgb, masks):
        # Public initializer establishes all version-specific state fields.
        with tempfile.TemporaryDirectory(prefix="xrpipe-sam2-") as temp:
            Image.fromarray(rgb).save(Path(temp) / "00000.jpg")
            state = self.video.init_state(temp, offload_video_to_cpu=True, offload_state_to_cpu=True)
        required = {"output_dict_per_obj", "frames_tracked_per_obj", "cached_features"}
        if not required <= state.keys():
            raise RuntimeError("Unsupported SAM2 state layout; use pipeline's SAM2 installation")
        state["images"] = {0: self._tensor(rgb)}
        state["cached_features"].clear()  # use lossless current RGB, not initializer JPEG
        self.state, self.frame = state, 0
        for i, mask in enumerate(masks, 1):
            self.video.add_new_mask(state, frame_idx=0, obj_id=i, mask=mask)
        self._propagate()

    def _propagate(self):
        records = list(self.video.propagate_in_video(
            self.state, start_frame_idx=self.frame, max_frame_num_to_track=0))
        if len(records) != 1 or records[0][0] != self.frame:
            raise RuntimeError("SAM2 did not propagate exactly the requested frame")
        _, ids, logits = records[0]
        order = [list(ids).index(i) for i in (1, 2)]
        masks = (logits[order, 0] > 0).detach().cpu().numpy()
        scores = []
        for logit, mask in zip(logits[order, 0], masks):
            scores.append(float(logit.sigmoid()[self.torch.as_tensor(mask, device=logit.device)].mean())
                          if mask.any() else 0.)
        return masks, scores

    def _detect(self, rgb, allowed_missing=()):
        masks, boxes, scores = [], [], []
        detection_s = segmentation_s = 0.
        # Keep pipeline DINOSAMEngine image-predictor BGR convention; video uses RGB.
        self.image.set_image(rgb[..., ::-1].copy())
        for slot, prompt in enumerate(self.prompts):
            start = time.perf_counter()
            inputs = self.processor(images=Image.fromarray(rgb), text=prompt, return_tensors="pt").to(self.device)
            output = self.dino(**inputs)
            conf = output.logits.sigmoid()[0].max(dim=-1).values
            keep = conf > self.threshold
            raw = output.pred_boxes[0][keep].detach().cpu().numpy()
            confidence = conf[keep].detach().cpu().numpy()
            detection_s += time.perf_counter() - start
            if not len(raw):
                if slot in allowed_missing:
                    masks.append(np.zeros(rgb.shape[:2], bool))
                    boxes.append([0, 0, 0, 0])
                    scores.append(0.)
                    continue
                raise RuntimeError(f"DINO found no object for prompt {prompt!r}")
            xyxy = np.c_[raw[:, :2] - raw[:, 2:] / 2, raw[:, :2] + raw[:, 2:] / 2]
            xyxy *= [rgb.shape[1], rgb.shape[0], rgb.shape[1], rgb.shape[0]]
            box = xyxy[int(np.argmax(confidence))]  # same best-instance selection as pipeline
            start = time.perf_counter()
            mask, _, _ = self.image.predict(box=box, multimask_output=False)
            segmentation_s += time.perf_counter() - start
            masks.append(np.asarray(mask).reshape(-1, *rgb.shape[:2]).any(axis=0))
            boxes.append(box.tolist())
            scores.append(float(confidence.mean()))
        return np.asarray(masks), boxes, scores, detection_s, segmentation_s

    def warmup(self):
        start = time.perf_counter()
        torch = self.torch
        rgb = np.zeros((480, 640, 3), np.uint8)
        with torch.inference_mode():
            inputs = self.processor(images=Image.fromarray(rgb), text=self.prompts[0], return_tensors="pt").to(self.device)
            self.dino(**inputs)
            self.image.set_image(rgb)
            self.image.predict(box=np.array([100, 100, 200, 200]), multimask_output=False)
            masks = np.zeros((2, 480, 640), bool)
            masks[0, 100:200, 100:200] = True
            masks[1, 250:350, 300:400] = True
            self._start(rgb, masks)
            self.frame = 1
            self.state["images"][1] = self._tensor(rgb)
            self.state["num_frames"] = 2
            self._propagate()
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        self.warmup_seconds = time.perf_counter() - start
        self.reset()
        print(f"Perception ready: load={self.load_seconds:.2f}s warmup={self.warmup_seconds:.2f}s", flush=True)

    def process(self, rgb, stamp, allowed_missing=()):
        if self.last_stamp is not None and stamp <= self.last_stamp:
            raise ValueError("Duplicate/out-of-order RGBD must not update perception")
        start = time.perf_counter()
        redetect = self.state is None
        detection_s = segmentation_s = 0.
        with self.torch.inference_mode():
            if not redetect:
                self.frame += 1
                self.state["images"] = {self.frame: self._tensor(rgb)}
                self.state["num_frames"] = self.frame + 1
                masks, scores = self._propagate()
                areas = masks.sum(axis=(1, 2))
                centres = np.array([np.argwhere(m).mean(axis=0) if m.any() else [np.nan, np.nan] for m in masks])
                required = [i for i in range(2) if i not in allowed_missing]
                redetect = any(areas[i] < max(30, self.initial_areas[i] * .1)
                               or areas[i] > self.initial_areas[i] * 4
                               or not np.isfinite(centres[i]).all()
                               or np.linalg.norm(centres[i] - self.last_centres[i]) > 150 for i in required)
                # Bounded history, enough for default SAM2 temporal memory and object pointers.
                keep = max(64, int(getattr(self.video, "max_obj_ptrs_in_encoder", 16)) * 2,
                           int(getattr(self.video, "num_maskmem", 7)) *
                           int(getattr(self.video, "memory_temporal_stride_for_eval", 1)) + 2)
                for outputs in self.state["output_dict_per_obj"].values():
                    for key in list(outputs["non_cond_frame_outputs"]):
                        if key < self.frame - keep:
                            del outputs["non_cond_frame_outputs"][key]
                for tracked in self.state["frames_tracked_per_obj"].values():
                    for key in list(tracked):
                        if key < self.frame - keep:
                            del tracked[key]
            if redetect:
                masks, boxes, scores, detection_s, segmentation_s = self._detect(rgb, allowed_missing)
                if any(masks[i].sum() < 30 for i in range(2) if i not in allowed_missing):
                    raise RuntimeError("Empty/tiny SAM2 initial mask")
                self._start(rgb, masks)
                self.initial_areas = masks.sum(axis=(1, 2))
            else:
                boxes = []
                for mask in masks:
                    y, x = np.where(mask)
                    boxes.append([int(x.min()), int(y.min()), int(x.max()), int(y.max())] if mask.any() else [0, 0, 0, 0])
            overlap = np.logical_and(*masks).sum() / max(1, np.logical_or(*masks).sum())
            if overlap > .8:
                raise RuntimeError("Both prompts resolved to the same mask; object slots are ambiguous")
            self.last_centres = np.array([np.argwhere(m).mean(axis=0) if m.any() else [np.nan, np.nan] for m in masks])
        self.last_stamp = stamp
        return dict(masks=masks, boxes=boxes, scores=scores, redetected=redetect,
                    timing=dict(perception=time.perf_counter() - start, detection=detection_s,
                                segmentation=time.perf_counter() - start - detection_s),
                    score_kind="dino" if redetect else "sam2_mask_probability")


class ObjectTracker:
    """Online version of pipeline PCA/reference ICP, gates, smoothing and one-object latch."""

    def __init__(self):
        pipeline_imports()
        from xrrel import objectpose
        self.op = objectpose
        self.cfg = objectpose.load_ego_config()[0].perception
        self.items = [None, None]
        self.lock = None
        self.last_time = None

    def feedback(self, hand, closed, stamp):
        if not closed:
            self.lock = None
        if hand is None:
            return
        if self.lock is None and closed:
            for i, item in enumerate(self.items):
                if (item is not None and stamp - item["time"] <= .5
                        and np.linalg.norm(item["pose"][:3, 3] - hand[:3, 3]) < .05):
                    self.lock = (i, np.linalg.inv(hand) @ item["pose"])
                    break
        if self.lock is not None:
            i, rel = self.lock
            self.items[i]["pose"] = hand @ rel
            self.items[i]["history"].append(self.items[i]["pose"][:3, 3].copy())

    def update(self, clouds, centroids, hand, closed, stamp):
        if not closed:
            self.lock = None
        op, cfg = self.op, self.cfg
        reports = []
        for i, points in enumerate(clouds):
            item = self.items[i]
            if self.lock is not None and self.lock[0] == i and hand is not None:
                item["pose"] = hand @ self.lock[1]
                item["history"].append(item["pose"][:3, 3].copy())
                reports.append(dict(valid=True, latched=True, point_count=len(points)))
                continue
            if len(points) < 30:
                raise RuntimeError(f"obj{i+1}: fewer than 30 valid depth points")
            if item is None:
                pose, info = op.pca_frame(points)
                canonical = (points - pose[:3, 3]) @ pose[:3, :3]
                item = dict(pose=pose, canonical=canonical, measurement=pose.copy(),
                            time=stamp, centroid=centroids[i],
                            history=deque([pose[:3, 3].copy()], maxlen=max(1, cfg.object_translation_median_window)))
                self.items[i] = item
                reports.append(dict(valid=True, latched=False, point_count=len(points), **info))
                continue
            gap = max(1, round((stamp - item["time"]) * 30))
            candidate, residual, ratio, fitted = op.icp_to_reference(item["canonical"], points, item["pose"])
            recovered = False
            if not fitted or ratio < cfg.minimum_pose_inlier_ratio:
                coarse = max(op.ICP_THRESHOLD_M, min(cfg.maximum_object_translation_step_m * min(gap, 5),
                                                     2 * op.rms_radius(item["canonical"])))
                retry = op.recover_icp(item["canonical"], points, item["pose"], coarse_m=coarse)
                if retry[3] and (not fitted or retry[2] > ratio + 1e-9):
                    candidate, residual, ratio, fitted = retry
                    recovered = True
            if not fitted or ratio < cfg.minimum_pose_inlier_ratio:
                raise RuntimeError(f"obj{i+1}: ICP failed (ratio={ratio:.3f})")
            if np.linalg.norm(candidate[:3, 3] - item["measurement"][:3, 3]) > cfg.maximum_object_translation_step_m * min(gap, 5):
                raise RuntimeError(f"obj{i+1}: translation gate rejected measurement")
            if op.sf._rotation_step_deg(item["measurement"], candidate) > cfg.maximum_object_rotation_step_deg * min(np.sqrt(min(gap, 5)), 2):
                candidate[:3, :3] = item["measurement"][:3, :3]
            motion = np.linalg.norm(centroids[i] - item["centroid"]) if gap == 1 else float("nan")
            item["measurement"] = candidate.copy()
            item["history"].append(candidate[:3, 3].copy())
            alpha = op.sf._adaptive_translation_alpha(cfg.object_translation_smoothing, motion,
                cfg.object_translation_motion_deadband_px, cfg.object_translation_full_response_px)
            ratio_motion = (alpha - cfg.object_translation_smoothing) / max(1 - cfg.object_translation_smoothing, 1e-6)
            candidate[:3, 3] = op.sf._adaptive_translation_measurement(list(item["history"]), ratio_motion)
            item["pose"] = op.sf._smooth_pose(item["pose"], candidate, translation_alpha=alpha,
                                             rotation_alpha=cfg.object_rotation_smoothing)
            item["time"], item["centroid"] = stamp, centroids[i]
            reports.append(dict(valid=True, latched=False, recovered=recovered, point_count=len(points),
                                inlier_ratio=ratio, residual_m=float(np.median(residual))))
        self.feedback(hand, closed, stamp)  # latch AFTER this frame's visual measurement
        if self.lock is not None:
            reports[self.lock[0]]["latched"] = True
        return [item["pose"].copy() for item in self.items], reports
