from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from ego_relation.common.pipeline import episode_work_dir
from ego_relation.config import ProjectConfig
from ego_relation.s1_pico_mode2.grasp import (
    plot_brainco_grasp_timeline,
    render_brainco_grasp_overlay,
    write_brainco_grasp_binary,
)
from ego_relation.s1_pico_mode2.mode2 import run_mode2
from ego_relation.s1_pico_mode2.pico import prepare_pico_episode
from ego_relation.s1_pico_mode2.quality import write_step1_quality_report
from ego_relation.s1_pico_mode2.smoothing import write_smoothed_mode2
from ego_relation.visualization.reports import generate_index
from ego_relation.visualization.reports import generate_step1_report


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _run_g1_replay(episode_dir: Path, fps: float) -> dict[str, str]:
    script = PROJECT_ROOT / "scripts" / "visualize_step1_g1_tcp.py"
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--episode-dir",
            str(episode_dir),
            "--fps",
            str(fps),
            "--trajectory",
            "smoothed",
            "--target-mode",
            "ready_delta",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )
    output_dir = episode_dir / "qa" / "step1_g1_tcp_replay"
    return {
        "video": str(output_dir / "g1_tcp_smoothed_ready_delta_replay.mp4"),
        "report": str(output_dir / "g1_tcp_smoothed_ready_delta_ik_report.json"),
        "error_plot": str(output_dir / "g1_tcp_smoothed_ready_delta_ik_error.png"),
        "joint_trajectory": str(output_dir / "g1_arm_ik_smoothed_ready_delta_trajectory.npz"),
    }


def run_episode(
    cfg: ProjectConfig,
    source: str | Path,
    episode_dir: str | Path,
    *,
    force: bool = False,
    render_videos: bool = True,
    g1_replay: bool = False,
) -> dict:
    source = Path(source).expanduser().resolve()
    episode_dir = Path(episode_dir).expanduser().resolve()
    episode_dir.mkdir(parents=True, exist_ok=True)
    (episode_dir / "config.snapshot.json").write_text(
        json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[step1 1/7] PICO camera, hands, pixel keypoints: {source.name}", flush=True)
    pico_manifest = prepare_pico_episode(cfg, source, episode_dir)

    print("[step1 2/7] task, three object instances, static-world TCP labels", flush=True)
    mode2_manifest = run_mode2(cfg, source, episode_dir, force=force)

    print("[step1 3/7] repair and smooth TCP trajectory", flush=True)
    smoothed_state, smoothed_action, smoothing_report = write_smoothed_mode2(
        episode_dir,
        fps=float(cfg.timeline.control_hz),
    )

    print("[step1 4/7] BrainCo binary grasp supplement", flush=True)
    grasp_path, grasp_report_path, _grasp_report = write_brainco_grasp_binary(
        cfg,
        episode_dir,
    )
    print("[step1 5/7] consolidate 30 Hz alignment and data quality QA", flush=True)
    quality_report = write_step1_quality_report(episode_dir)

    grasp_overlay: Path | None = None
    grasp_timeline: Path | None = None
    if render_videos:
        print("[step1 6/7] render hand pixels and BrainCo grasp QA", flush=True)
        grasp_overlay = render_brainco_grasp_overlay(cfg, source, episode_dir)
        grasp_timeline = plot_brainco_grasp_timeline(cfg, episode_dir)
    else:
        print("[step1 6/7] skip QA videos", flush=True)

    g1_outputs: dict[str, str] | None = None
    if g1_replay:
        print("[step1 7/7] MuJoCo replay from G1 ready using local TCP increments", flush=True)
        g1_outputs = _run_g1_replay(episode_dir, float(cfg.timeline.control_hz))
    else:
        print("[step1 7/7] G1 replay not requested", flush=True)

    outputs = {
        "schema": "ego_relation_step1_bundle_v1",
        "source": str(source),
        "episode_dir": str(episode_dir),
        "task_semantics": mode2_manifest.outputs["task_semantics"],
        "object_instances": mode2_manifest.outputs["object_instances"],
        "action_contract": mode2_manifest.outputs["action_contract"],
        "state_abs_raw": mode2_manifest.outputs["state_abs"],
        "action_abs_raw": mode2_manifest.outputs["action_abs"],
        "state_abs_smoothed": str(smoothed_state),
        "action_abs_smoothed": str(smoothed_action),
        "frame_table_30hz": mode2_manifest.outputs["frame_table_30hz"],
        "timeline_30hz_qa": mode2_manifest.outputs["timeline_qa"],
        "step1_quality_qa": str(quality_report),
        "tcp_smoothing_qa": str(smoothing_report),
        "brainco_grasp_binary": str(grasp_path),
        "brainco_grasp_binary_qa": str(grasp_report_path),
        "hand_keypoints_brainco_grasp_video": str(grasp_overlay) if grasp_overlay else None,
        "brainco_grasp_timeline": str(grasp_timeline) if grasp_timeline else None,
        "hand_keypoints_pixels": pico_manifest.outputs["hand_pixels"],
        "hand_keypoints_pixels_qa": pico_manifest.outputs["hand_pixels_qa"],
        "wrist_tcp_poses_camera0": pico_manifest.outputs["wrist_tcp_poses"],
        "wrist_tcp_poses_qa": pico_manifest.outputs["wrist_tcp_poses_qa"],
        "g1_ready_delta_replay": g1_outputs,
        "reference_frame": "static frame-0 pelvis origin/yaw with PICO world gravity",
        "execution": "rebase demo-local SE(3) increments at the G1 ready TCP poses",
    }
    output_path = episode_dir / "step1" / "outputs.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(outputs, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[step1 ok] {output_path}", flush=True)
    return outputs


def run(
    cfg: ProjectConfig,
    sources: list[Path],
    *,
    force: bool = False,
    render_videos: bool = True,
    g1_replay: bool = False,
) -> list[Path]:
    output = []
    for source in sources:
        episode_dir = episode_work_dir(cfg, source)
        run_episode(
            cfg,
            source,
            episode_dir,
            force=force,
            render_videos=render_videos,
            g1_replay=g1_replay,
        )
        if cfg.visualization.enabled:
            generate_step1_report(cfg, source, episode_dir)
        output.append(episode_dir)
    if cfg.visualization.enabled:
        generate_index(cfg, sources)
    return output
