from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ego_relation.config import ProjectConfig
from ego_relation.contracts.manifest import StageManifest, file_sha256
from ego_relation.contracts.se3 import compose, invert, rotation_angle_deg, vec9_to_transform
from ego_relation.s1_pico_mode2.contracts import write_step1_contracts
from ego_relation.s1_pico_mode2.native_mode2 import build_mode2_labels


def _gap_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "max_ms": float(np.max(values)),
    }


def _alignment_qa(cfg: ProjectConfig, arrays: dict[str, np.ndarray], native_meta: dict) -> dict:
    ticks = np.asarray(arrays["ticks_ns"], dtype=np.int64)
    intervals = np.diff(ticks)
    expected_interval_ns = int(round(1e9 / cfg.timeline.control_hz))
    interval_error_ns = np.abs(intervals - expected_interval_ns)
    streams = {}
    for name, match_key, gap_key, valid_key, maximum_gap_ms in (
        ("body", "body_match", "body_gap_ms", "body_valid", cfg.timeline.max_tracking_gap_ms),
        (
            "left_hand",
            "left_hand_match",
            "left_hand_gap_ms",
            "left_hand_valid",
            cfg.timeline.max_tracking_gap_ms,
        ),
        (
            "right_hand",
            "right_hand_match",
            "right_hand_gap_ms",
            "right_hand_valid",
            cfg.timeline.max_tracking_gap_ms,
        ),
        ("camera", "cam_match", "cam_gap_ms", "camera_valid", cfg.timeline.max_camera_gap_ms),
    ):
        match = np.asarray(arrays[match_key], dtype=np.int64)
        gaps = np.asarray(arrays[gap_key], dtype=np.float64)
        valid = np.asarray(arrays[valid_key], dtype=bool)
        streams[name] = {
            **_gap_stats(gaps),
            "maximum_allowed_gap_ms": float(maximum_gap_ms),
            "valid_ratio": float(valid.mean()),
            "unique_source_samples": int(np.unique(match).size),
            "repeated_control_frames": int(np.count_nonzero(np.diff(match) == 0)),
            "nonmonotonic_matches": int(np.count_nonzero(np.diff(match) < 0)),
        }

    combined_valid_ratio = float(np.asarray(arrays["valid"], dtype=bool).mean())
    target_30hz_verified = bool(
        np.isclose(cfg.timeline.control_hz, 30.0)
        and len(intervals) > 0
        and int(interval_error_ns.max(initial=0)) <= 1
    )
    violations = []
    if not target_30hz_verified:
        violations.append("control timeline is not a uniform 30 Hz grid")
    if combined_valid_ratio < cfg.timeline.minimum_aligned_valid_ratio:
        violations.append(
            f"aligned valid ratio {combined_valid_ratio:.3f} is below "
            f"{cfg.timeline.minimum_aligned_valid_ratio:.3f}"
        )
    for name, report in streams.items():
        if report["p95_ms"] > report["maximum_allowed_gap_ms"]:
            violations.append(
                f"{name} match gap p95 {report['p95_ms']:.3f} ms exceeds "
                f"{report['maximum_allowed_gap_ms']:.3f} ms"
            )
        if report["nonmonotonic_matches"]:
            violations.append(f"{name} has non-monotonic source matches")
    return {
        "schema": "step1_timeline_qa_v1",
        "target_hz": float(cfg.timeline.control_hz),
        "target_30hz_verified": target_30hz_verified,
        "frames": int(len(ticks)),
        "duration_s": float((ticks[-1] - ticks[0]) / 1e9) if len(ticks) > 1 else 0.0,
        "expected_interval_ns": expected_interval_ns,
        "interval_error_ns_max": int(interval_error_ns.max(initial=0)),
        "source_rates_hz": native_meta.get("source_rates_hz", {}),
        "streams": streams,
        "combined_valid_ratio": combined_valid_ratio,
        "minimum_aligned_valid_ratio": float(cfg.timeline.minimum_aligned_valid_ratio),
        "accepted": not violations,
        "violations": violations,
    }


def _motion_qa(state: np.ndarray, ticks_ns: np.ndarray) -> dict:
    dt = np.diff(np.asarray(ticks_ns, dtype=np.int64)) / 1e9
    if len(dt) == 0 or np.any(dt <= 0):
        raise ValueError("Mode2 timestamps must be strictly increasing")
    report: dict = {}
    for name, item in (("left", state[:, :9]), ("right", state[:, 9:18])):
        translation_speed = []
        rotation_speed = []
        for previous, current, seconds in zip(item[:-1], item[1:], dt, strict=True):
            delta = compose(invert(vec9_to_transform(previous)), vec9_to_transform(current))
            translation_speed.append(float(np.linalg.norm(delta[:3, 3]) / seconds))
            rotation_speed.append(float(rotation_angle_deg(delta[:3, :3]) / seconds))
        report[name] = {
            "translation_speed_m_s_p99": float(np.percentile(translation_speed, 99)),
            "translation_speed_m_s_max": float(np.max(translation_speed)),
            "rotation_speed_deg_s_p99": float(np.percentile(rotation_speed, 99)),
            "rotation_speed_deg_s_max": float(np.max(rotation_speed)),
        }
    brainco_speed = np.abs(np.diff(state[:, 18:], axis=0)) / dt[:, None]
    report["brainco_speed_s_p99"] = float(np.percentile(brainco_speed, 99))
    report["brainco_speed_s_max"] = float(np.max(brainco_speed))
    return report


def run_mode2(cfg: ProjectConfig, source: str | Path, episode_dir: str | Path, *, force: bool = False) -> StageManifest:
    """运行已内化的 Mode2；绝不进入 Mode1/G1 初始位姿分支。"""
    source = Path(source).resolve()
    episode_dir = Path(episode_dir).resolve()
    output_dir = episode_dir / "mode2"
    output_dir.mkdir(parents=True, exist_ok=True)
    contracts = write_step1_contracts(cfg, source, episode_dir)

    manifest_file = episode_dir / "mode2.manifest.json"
    previous = StageManifest.read(manifest_file) if manifest_file.is_file() else None
    reusable = (
        previous is not None
        and (output_dir / "state_abs.npy").is_file()
        and (output_dir / "action_abs.npy").is_file()
        and (output_dir / "ticks_ns.npy").is_file()
        and (episode_dir / "sync/frame_table_30hz.parquet").is_file()
        and (episode_dir / "qa/step1_timeline_qa.json").is_file()
        and all(path.is_file() for path in contracts.values())
        and float(previous.config.get("control_hz", -1.0)) == float(cfg.timeline.control_hz)
        and previous.config.get("engine") == "ego_relation.native_mode2_v4_30hz_qa"
        and previous.config.get("learning_target") == "next_absolute_tcp_target_in_g1_base"
    )
    if reusable and not force:
        return previous

    arrays, native_meta = build_mode2_labels(cfg, source)
    (output_dir / "upstream.log").write_text(
        "portable native Mode2; reference semantics internalized from egodata_targeting_project\n"
        + json.dumps(native_meta, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )

    state = arrays["state"].astype(np.float32)
    action = arrays["action"].astype(np.float32)
    if state.ndim != 2 or state.shape[1] != 30:
        raise ValueError(f"Mode2 state 必须为 (T,30)，实际 {state.shape}")
    expected = np.concatenate([state[1:], state[-1:]], axis=0)
    if not np.allclose(action, expected, atol=1e-6):
        raise ValueError("Mode2 action 不满足 action[t]=state[min(t+1,T-1)]")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("Mode2 state/action 含 NaN 或 Inf")
    motion_qa = _motion_qa(state, arrays["ticks_ns"])
    alignment_qa = _alignment_qa(cfg, arrays, native_meta)
    motion_violations: list[str] = []
    for side in ("left", "right"):
        if motion_qa[side]["translation_speed_m_s_max"] > cfg.mode2.maximum_eef_speed_m_s:
            motion_violations.append(f"{side} EEF translation jump: {motion_qa[side]}")
        if motion_qa[side]["rotation_speed_deg_s_max"] > cfg.mode2.maximum_eef_rotation_deg_s:
            motion_violations.append(f"{side} EEF rotation jump: {motion_qa[side]}")
    if motion_qa["brainco_speed_s_max"] > cfg.mode2.maximum_brainco_speed_s:
        motion_violations.append(f"BrainCo joint jump: {motion_qa['brainco_speed_s_max']}")
    camera_gap_p95_ms = float(np.percentile(arrays["cam_gap_ms"], 95))

    np.save(output_dir / "state_abs.npy", state)
    np.save(output_dir / "action_abs.npy", action)
    np.save(output_dir / "ticks_ns.npy", arrays["ticks_ns"].astype(np.int64))
    np.save(output_dir / "camera_match.npy", arrays["cam_match"].astype(np.int64))
    np.save(output_dir / "valid.npy", arrays["valid"].astype(bool))
    if "arm_qpos" in arrays:
        np.save(output_dir / "arm_qpos_qa.npy", arrays["arm_qpos"].astype(np.float32))

    table = pa.table(
        {
            "control_index": np.arange(len(state), dtype=np.int64),
            "timestamp_ns": arrays["ticks_ns"].astype(np.int64),
            "body_index": arrays["body_match"].astype(np.int64),
            "body_gap_ms": arrays["body_gap_ms"].astype(np.float32),
            "body_valid": arrays["body_valid"].astype(bool),
            "left_hand_index": arrays["left_hand_match"].astype(np.int64),
            "left_hand_gap_ms": arrays["left_hand_gap_ms"].astype(np.float32),
            "left_hand_valid": arrays["left_hand_valid"].astype(bool),
            "right_hand_index": arrays["right_hand_match"].astype(np.int64),
            "right_hand_gap_ms": arrays["right_hand_gap_ms"].astype(np.float32),
            "right_hand_valid": arrays["right_hand_valid"].astype(bool),
            "camera_index": arrays["cam_match"].astype(np.int64),
            "camera_gap_ms": arrays["cam_gap_ms"].astype(np.float32),
            "camera_valid": arrays["camera_valid"].astype(bool),
            "mode2_valid": arrays["valid"].astype(bool),
        }
    )
    pq.write_table(table, episode_dir / "sync" / "frame_table.parquet")
    pq.write_table(table, episode_dir / "sync" / "frame_table_30hz.parquet")
    timeline_qa_path = episode_dir / "qa" / "step1_timeline_qa.json"
    timeline_qa_path.write_text(
        json.dumps(alignment_qa, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    quality_violations = [*alignment_qa["violations"], *motion_violations]

    metrics = {
        "frames": int(len(state)),
        "fps": float(cfg.timeline.control_hz),
        "valid_ratio": float(np.mean(arrays["valid"])),
        "camera_gap_p95_ms": camera_gap_p95_ms,
        "action_shift_verified": True,
        "upstream_mode": native_meta.get("mode"),
        "upstream_mapping": native_meta.get("mapping"),
        "engine": native_meta.get("engine"),
        "source_rates_hz": native_meta.get("source_rates_hz"),
        "timeline_qa": alignment_qa,
        "pelvis_motion_qa": native_meta.get("pelvis_motion_qa"),
        "ik_err_cm": None,
        "motion_qa": motion_qa,
        "motion_deployable": not motion_violations,
        "step1_deployable": not quality_violations,
        "motion_violations": motion_violations,
        "quality_violations": quality_violations,
    }
    manifest = StageManifest(
        schema_version=cfg.schema_version,
        stage="mode2",
        episode=source.stem,
        source_path=str(source),
        source_sha256=file_sha256(source),
        config={
            "pipeline_mode": "mode2_state",
            "control_hz": cfg.timeline.control_hz,
            "hand_retargeter": cfg.mode2.hand_retargeter,
            "engine": "ego_relation.native_mode2_v4_30hz_qa",
            "tcp_reference_frame": "static initial pelvis origin/yaw; PICO world gravity",
            "learning_target": "next_absolute_tcp_target_in_g1_base",
        },
        outputs={
            "state_abs": str(output_dir / "state_abs.npy"),
            "action_abs": str(output_dir / "action_abs.npy"),
            "action_contract": str(contracts["action_contract"]),
            "task_semantics": str(contracts["task_semantics"]),
            "object_instances": str(contracts["object_instances"]),
            "ticks": str(output_dir / "ticks_ns.npy"),
            "frame_table": str(episode_dir / "sync" / "frame_table.parquet"),
            "frame_table_30hz": str(episode_dir / "sync" / "frame_table_30hz.parquet"),
            "timeline_qa": str(timeline_qa_path),
            "log": str(output_dir / "upstream.log"),
        },
        metrics=metrics,
        warnings=[
            "portable Mode2 intentionally omits G1 arm IK: labels are absolute T_g1_base_tcp poses plus Revo2 commands; "
            "action[t] is the next absolute target state; downstream training may apply its own action transform",
            *(
                ["episode quarantined from Step2/export: " + "; ".join(quality_violations)]
                if quality_violations
                else []
            ),
        ],
    )
    manifest.write(episode_dir / "mode2.manifest.json")
    quarantine_path = episode_dir / "qa" / "mode2_quarantine.json"
    if quality_violations:
        quarantine_path.write_text(
            json.dumps(
                {
                    "accepted": False,
                    "reason": "Step1 30 Hz alignment or motion QA failed",
                    "violations": quality_violations,
                    "policy": "preserve raw output; exclude the complete episode from perception/export/training",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    elif quarantine_path.exists():
        quarantine_path.unlink()
    return manifest
