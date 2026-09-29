# -*- coding: utf-8 -*-
# @FileName: DINOSAM.py

"""
====================================================================================================
Project Aria DINO-SAM2 Segmentation Pipeline (DINOSAM.py)
====================================================================================================

Description:
    This script processes RGB frames using Grounding DINO for object detection and SAM2
    for mask generation. It supports multi-object prompts and generates combined masks
    for downstream tasks.

Technical Specifics:
    - Grounding DINO: Text-to-Box detection.
    - SAM2: Box-to-Mask segmentation.
====================================================================================================

复制自 ego_relation_policy/third_party/humanego_runtime/preprocess/DINOSAM.py (step2 主流程)。

适配 (相对上游只有这 4 处):
  1. `run_dinosam_subprocess` 删掉 —— 它 `-m preprocess.DINOSAM` 调的是上游包名, 这里没有。
  2. import 从 `utils.*` 改成 `xrseg.utils.*`。
  3. `hf_hub_download` 显式传 cache_dir (本机 huggingface.co 不可达, 走 hf-mirror 镜像)。
  4. `predict_frame_internal` 在单帧没有候选框时返回 HumanEgo-main 兼容的
     全零 mask、空框和 0 置信度；批处理调用方可以继续处理后续帧。整段视频
     是否完全没有检测由上层批处理统计后决定。

`DINOSAMEngine` / `DINOSAM` 两个类的其余部分逐行照抄。
另加了一个 `segment()`: 与 `process_and_save` 的单提示词分支等价, 但把框/置信度/mask
原样返回给调用方 (上游只返回可视化图, 框和置信度会被丢掉)。
"""

import os
import cv2
import torch
import numpy as np
import gc
import time
from pathlib import Path
from PIL import Image
from huggingface_hub import hf_hub_download

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from xrseg.utils.utils_vis import draw_glass_rect, draw_status_bar, C_CYAN, C_GREEN, C_RED, C_GOLD, C_WHITE, C_GRAY
from xrseg.utils.utils_io import load_cfg


class PromptNotFoundError(RuntimeError):
    """DINO 在给定阈值下没框到任何东西 —— 提示词或阈值不对, 不是"这个物体不存在"。"""


class DINOSAMEngine:
    """Model Engine for DINO and SAM2."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"║ [System] Initializing Models on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(self.cfg.dino_model_id)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.cfg.dino_model_id).to(self.device)
        
        ckpt_path = hf_hub_download(
            repo_id=self.cfg.sam2_repo_id,
            filename=self.cfg.sam2_checkpoint_name,
            cache_dir=os.environ.get("HUGGINGFACE_HUB_CACHE") or None,
        )
        self.predictor = SAM2ImagePredictor(build_sam2(self.cfg.sam2_config, ckpt_path, device=self.device))

    def predict_frame_internal(self, image_np, text_prompt, *, raise_on_empty=False):
        """Internal prediction function using pre-set image in predictor."""
        image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        W, H = image_pil.size
        
        inputs = self.processor(images=image_pil, text=text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.dino_model(**inputs)
        
        logits = outputs.logits.sigmoid()[0]
        boxes = outputs.pred_boxes[0]
        
        mask_filter = logits.max(-1)[0] > self.cfg.box_threshold
        filtered_logits = logits[mask_filter]
        filtered_boxes = boxes[mask_filter]
        
        if len(filtered_boxes) == 0:
            if raise_on_empty:
                raise PromptNotFoundError(
                    f"GroundingDINO 在 box_threshold={self.cfg.box_threshold} 下没有框到 "
                    f"任何目标 (提示词 {text_prompt!r})。"
                )
            # HumanEgo-main 的逐帧预处理行为：单帧无检测是正常情况，
            # 返回空结果而不是中断整个视频。上层可依据所有帧的统计决定
            # 是否把提示词/模型配置视为不可用。
            return (
                np.zeros((H, W), dtype=np.uint8),
                0.0,
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )

        confidences = filtered_logits.max(-1)[0].cpu().numpy()
        avg_conf = np.mean(confidences)
        
        pixel_boxes = filtered_boxes * torch.Tensor([W, H, W, H]).to(self.device)
        cx, cy, w, h = pixel_boxes.unbind(-1)
        x1, y1 = cx - 0.5 * w, cy - 0.5 * h
        x2, y2 = cx + 0.5 * w, cy + 0.5 * h
        input_boxes = torch.stack([x1, y1, x2, y2], dim=-1).cpu().numpy()

        masks, _, _ = self.predictor.predict(box=input_boxes, multimask_output=False)
        
        combined_mask = np.any(masks.squeeze(), axis=0) if masks.ndim > 3 else masks.squeeze()
        if combined_mask.ndim > 2:
             combined_mask = np.any(combined_mask, axis=0)

        return (combined_mask.astype(np.uint8) * 255), avg_conf, input_boxes, confidences

    def cleanup(self):
        """Release resources."""
        print("║ [Cleanup] Releasing Engine Resources...")
        if hasattr(self, 'dino_model'): self.dino_model.to("cpu"); del self.dino_model
        if hasattr(self, 'predictor'): self.predictor.model.to("cpu"); del self.predictor
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

class DINOSAM:
    """Inference and Visualization Manager."""
    def __init__(self, cfg_path):
        self.cfg = load_cfg(cfg_path)
        self.engine = DINOSAMEngine(self.cfg)

    def _render_vis(self, img, mask, boxes, box_confs, avg_conf, latency, text_prompt):
        """Render side-by-side visualization."""
        left_vis = img.copy()
        num_objects = 0
        if boxes is not None:
            num_objects = len(boxes)
            for box, b_conf in zip(boxes, box_confs):
                bx1, by1, bx2, by2 = box.astype(int)
                cv2.rectangle(left_vis, (bx1, by1), (bx2, by2), C_GREEN, 2)
                cv2.putText(left_vis, f"{b_conf:.2f}", (bx1, by1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_GREEN, 1, cv2.LINE_AA)
        
        mask_vis = mask if mask is not None else np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        heatmap_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
        
        draw_glass_rect(heatmap_vis, (10, 10), (350, 240))
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        header_col = C_GREEN if avg_conf > self.cfg.box_threshold else C_GRAY
        cv2.putText(heatmap_vis, "DINO-SAM2 ANALYZER", (20, 30), font, 0.5, C_GOLD, 1, cv2.LINE_AA)
        status = "[SIGNAL ACTIVE]" if avg_conf > 0 else "[SEARCHING...]"
        cv2.putText(heatmap_vis, status, (20, 50), font, 0.4, header_col, 1, cv2.LINE_AA)
        cv2.putText(heatmap_vis, f"OBJECTS: {num_objects}", (20, 120), font, 0.5, C_GOLD, 1, cv2.LINE_AA)
        
        draw_status_bar(heatmap_vis, (20, 170), 300, avg_conf, 1.0, f"Conf: {avg_conf:.2f}", C_CYAN)
        cv2.putText(heatmap_vis, f"LATENCY: {latency*1000:.1f}ms", (20, 210), font, 0.4, C_WHITE, 1, cv2.LINE_AA)
        cv2.putText(heatmap_vis, f"PROMPT: {text_prompt[:20]}...", (20, 225), font, 0.3, C_GRAY, 1, cv2.LINE_AA)

        return cv2.hconcat([left_vis, heatmap_vis])


    def process_single(self, img, prompt, save_path=None):
        """Process a single image or numpy array."""
        if isinstance(img, str):
            image_np = cv2.imread(img)
        else:
            image_np = img

        if image_np is None: return None

        self.engine.predictor.set_image(image_np)
        t_start = time.perf_counter()
        mask, avg_conf, boxes, box_confs = self.engine.predict_frame_internal(image_np, prompt)
        
        if self.engine.device == "cuda": torch.cuda.synchronize()
        latency = time.perf_counter() - t_start

        mask_out = mask if mask is not None else np.zeros(image_np.shape[:2], dtype=np.uint8)
        if save_path: cv2.imwrite(save_path, mask_out)
        return mask_out
    

    def process_and_save(self, image_path, prompts_dict):
        """Process multiple prompts and save individual/combined masks."""
        if not os.path.exists(image_path): return None, 0
        
        img = cv2.imread(image_path)
        base_dir = os.path.dirname(image_path)
        combined_all_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        
        self.engine.predictor.set_image(img)
        last_vis = None
        prompts_count = 0 

        for key, prompt in prompts_dict.items():
            if not prompt.strip(): continue
            prompts_count += 1 

            t_start = time.perf_counter()
            mask, avg_conf, boxes, box_confs = self.engine.predict_frame_internal(img, prompt)
            
            if self.engine.device == "cuda": torch.cuda.synchronize()
            latency = time.perf_counter() - t_start

            mask_out = mask if mask is not None else np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            cv2.imwrite(os.path.join(base_dir, f"mask_{key}.png"), mask_out)
            combined_all_mask = cv2.bitwise_or(combined_all_mask, mask_out)

            last_vis = self._render_vis(img, combined_all_mask, boxes, box_confs, avg_conf, latency, prompt)
            
        cv2.imwrite(os.path.join(base_dir, "mask_arm_and_obj.png"), combined_all_mask)
        return last_vis, prompts_count

    def segment(self, image_path, prompt: str, save_mask_to=None) -> dict:
        """单提示词分割一张图, 把 mask / 框 / 置信度原样返回。

        与 `process_and_save` 的单提示词分支等价 (同样 set_image 后
        predict_frame_internal, 同样把 mask 落盘), 区别只是上游只把可视化图
        返回给调用方, 框和置信度被丢掉了 —— report.json 需要它们。

        颜色: 上游 `process_and_save` 用 cv2.imread 得到 BGR, 一路把 BGR 传给
        `predictor.set_image()` (DINO 那边单独 cvtColor 转 RGB)。这里**不修**它,
        要与 step2 的输出对齐就不能顺手换颜色空间。
        """
        image_np = cv2.imread(str(image_path))
        if image_np is None:
            raise FileNotFoundError(image_path)

        self.engine.predictor.set_image(image_np)
        t_start = time.perf_counter()
        mask, avg_conf, boxes, box_confs = self.engine.predict_frame_internal(image_np, prompt)
        if self.engine.device == "cuda":
            torch.cuda.synchronize()
        latency = time.perf_counter() - t_start

        if save_mask_to is not None:
            save_mask_to = Path(save_mask_to)
            save_mask_to.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(save_mask_to), mask):
                raise RuntimeError(f"无法写入提示帧 mask: {save_mask_to}")

        return {
            "mask": mask,
            "boxes": boxes,
            "box_confidences": box_confs,
            "avg_confidence": float(avg_conf),
            "latency_s": float(latency),
        }

    def cleanup(self):
        self.engine.cleanup()
