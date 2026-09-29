from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_step1_quality_report(episode_dir: str | Path) -> Path:
    episode_dir = Path(episode_dir).expanduser().resolve()
    qa_dir = episode_dir / "qa"
    mode2 = _read_json(episode_dir / "mode2.manifest.json")
    pico = _read_json(qa_dir / "pico_report.json")
    timeline = _read_json(qa_dir / "step1_timeline_qa.json")
    hand_pixels = _read_json(qa_dir / "hand_keypoints_pixels_report.json")
    wrist_tcp = _read_json(qa_dir / "wrist_tcp_poses_report.json")
    smoothing = _read_json(qa_dir / "step1_tcp_smoothing_report.json")
    grasp = _read_json(qa_dir / "brainco_grasp_binary_report.json")

    state_path = episode_dir / "mode2/state_abs.npy"
    action_path = episode_dir / "mode2/action_abs.npy"
    state = np.load(state_path)
    action = np.load(action_path)
    expected_action = np.concatenate((state[1:], state[-1:]), axis=0)
    finite = bool(np.isfinite(state).all() and np.isfinite(action).all())
    action_shift_error = float(np.max(np.abs(action - expected_action)))

    metrics = mode2.get("metrics", {})
    checks = {
        "uniform_30hz_timeline": bool(timeline.get("target_30hz_verified", False)),
        "aligned_valid_ratio": bool(timeline.get("accepted", False)),
        "pico_geometry": bool(pico.get("geometry_deployable", False)),
        "finite_state_action": finite,
        "action_shift": action_shift_error <= 1e-6,
        "tcp_motion": bool(metrics.get("motion_deployable", False)),
    }
    descriptions = {
        "uniform_30hz_timeline": "control timestamps are a uniform 30 Hz grid",
        "aligned_valid_ratio": "body, hands and camera satisfy alignment gap/validity thresholds",
        "pico_geometry": "PICO intrinsics/extrinsics and hand projection QA are deployable",
        "finite_state_action": "state/action contain no NaN or Inf",
        "action_shift": "action[t] equals the next absolute state",
        "tcp_motion": "TCP and Revo2 trajectories satisfy motion limits",
    }
    failures = [descriptions[name] for name, passed in checks.items() if not passed]
    report = {
        "schema": "step1_quality_report_v1",
        "accepted": not failures,
        "checks": checks,
        "failures": failures,
        "timeline": timeline,
        "label_qa": {
            "state_shape": list(state.shape),
            "action_shape": list(action.shape),
            "finite": finite,
            "action_shift_max_abs_error": action_shift_error,
        },
        "pico": pico,
        "motion": metrics.get("motion_qa", {}),
        "diagnostics": {
            "hand_pixels": hand_pixels,
            "wrist_tcp": wrist_tcp,
            "tcp_smoothing": smoothing,
            "brainco_grasp": grasp,
        },
        "policy": (
            "accepted episodes may continue to Step2"
            if not failures
            else "preserve outputs for diagnosis; exclude episode from Step2/export/training"
        ),
    }
    output = qa_dir / "step1_quality_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output
