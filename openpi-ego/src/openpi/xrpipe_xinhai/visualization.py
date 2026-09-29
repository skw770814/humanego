"""Pipeline drawing primitives and colours with D405 distorted optical projection."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .config import pipeline_imports
from .geometry import se3
from .observation import save_snapshot


def project(points, camera):
    points = np.asarray(points, float)
    uv = cv2.projectPoints(points, np.zeros(3), np.zeros(3), np.asarray(camera["K"], float),
                           np.asarray(camera["D"], float))[0][:, 0]
    visible = (np.isfinite(uv).all(axis=1) & (points[:, 2] > 0)
               & (uv[:, 0] >= 0) & (uv[:, 0] < camera["width"])
               & (uv[:, 1] >= 0) & (uv[:, 1] < camera["height"]))
    return uv, visible


def visualize(obs, directory, prompts):
    pipeline_imports()
    from xrhand import render as draw
    from xrseg.common import instance_color

    path = save_snapshot(obs, directory)
    rgb = obs["rgb"]
    Image.fromarray(rgb).save(path / "rgb.png")
    valid = np.isfinite(obs["depth"]) & (obs["depth"] > 0)
    cv2.imwrite(str(path / "depth_valid.png"), valid.astype(np.uint8) * 255)
    overlay = rgb.astype(float)
    perception = obs["perception"]
    colors = []
    for i, mask in enumerate(perception["masks"]):
        color = tuple(reversed(instance_color(f"obj{i+1}")))
        colors.append(color)
        overlay[mask] = overlay[mask] * .55 + np.asarray(color) * .45
    image = Image.fromarray(overlay.astype(np.uint8))
    pen = ImageDraw.Draw(image)
    for i, box in enumerate(perception["boxes"]):
        pen.rectangle(box, outline=colors[i], width=2)
        draw._text(pen, (box[0], max(0, box[1] - 20)),
                   f"obj{i+1}: {prompts[i]} {perception['scores'][i]:.2f}", draw._font(16), colors[i])
    image.save(path / "segmentation.png")
    projections = {}
    poses = {f"obj{i+1}": t for i, t in enumerate(obs["T_C_O"])}
    if obs["T_C_M"] is not None:
        poses["fingertip_midpoint"] = obs["T_C_M"]
    for name, t in poses.items():
        se3(t)
        points = t[:3, 3] + np.r_[np.zeros((1, 3)), np.eye(3) * .05] @ t[:3, :3].T
        uv, inside = project(points, obs["metadata"]["camera"])
        projections[name] = dict(uv=uv.tolist(), inside=inside.tolist(), camera_depth_m=points[:, 2].tolist())
        if inside[0]:
            origin = tuple(uv[0])
            draw._disc(pen, origin, 5, (255, 255, 0))
            draw._text(pen, (origin[0] + 8, origin[1] + 8), name, draw._font(16), (255, 255, 255))
            for axis in range(3):
                if inside[axis + 1]:
                    end = tuple(uv[axis + 1])
                    pen.line([origin, end], fill=draw.COLOR_AXIS[axis], width=3)
                    draw._arrow_tip(pen, origin, end, draw.COLOR_AXIS[axis])
                    draw._text(pen, end, "XYZ"[axis], draw._font(16), draw.COLOR_AXIS[axis])
    image.save(path / "geometry.png")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    for i, points in enumerate(obs["clouds"]):
        ax.scatter(*points[::max(1, len(points) // 3000)].T, s=1, color=np.asarray(colors[i]) / 255, label=f"obj{i+1}")
    for name, t in poses.items():
        ax.text(*t[:3, 3], name)
        for axis, color in enumerate(("r", "g", "b")):
            ax.quiver(*t[:3, 3], *(t[:3, axis] * .05), color=color)
    ax.set(xlabel="camera X (m)", ylabel="camera Y (m)", zlabel="camera Z (m)")
    all_points = np.concatenate([*obs["clouds"], *[t[None, :3, 3] for t in poses.values()]])
    centre = (all_points.min(axis=0) + all_points.max(axis=0)) / 2
    radius = max(.05, np.ptp(all_points, axis=0).max() / 2)
    ax.set_xlim(centre[0]-radius, centre[0]+radius)
    ax.set_ylim(centre[1]-radius, centre[1]+radius)
    ax.set_zlim(centre[2]-radius, centre[2]+radius)
    ax.set_box_aspect((1, 1, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path / "camera_scene_3d.png")
    plt.close(fig)
    report = dict(**obs["metadata"], depth_valid_ratio=float(valid.mean()),
                  matrix_valid=True, projections=projections, prompts=list(prompts),
                  boxes=perception["boxes"], scores=perception["scores"], score_kind=perception["score_kind"],
                  complete_state=obs["state"] is not None,
                  raw_state=obs["raw_state"].tolist() if obs["raw_state"] is not None else None,
                  training_state=obs["state"].tolist() if obs["state"] is not None else None,
                  note="Single frame validates initial geometry only, not tracking/latch. Inspect projections manually.")
    (path / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return path

