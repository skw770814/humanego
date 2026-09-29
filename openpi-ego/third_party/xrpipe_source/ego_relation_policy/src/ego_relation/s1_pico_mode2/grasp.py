from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import median_filter

from ego_relation.config import ProjectConfig
from ego_relation.s1_pico_mode2.pico import PicoEpisode


MOTOR_ORDER = ("thumb_flex", "thumb_rot", "index", "middle", "ring", "pinky")
HAND_CHAINS = (
    (1, 0),
    (1, 2, 3, 4, 5),
    (1, 6, 7, 8, 9, 10),
    (1, 11, 12, 13, 14, 15),
    (1, 16, 17, 18, 19, 20),
    (1, 21, 22, 23, 24, 25),
)


def _bool_runs(mask: np.ndarray) -> list[tuple[int, int, bool]]:
    values = np.asarray(mask, dtype=bool)
    if len(values) == 0:
        return []
    cuts = np.flatnonzero(np.diff(values.astype(np.int8)) != 0) + 1
    bounds = [0, *cuts.tolist(), len(values)]
    return [
        (bounds[index], bounds[index + 1], bool(values[bounds[index]]))
        for index in range(len(bounds) - 1)
    ]


def absorb_short_runs(state: np.ndarray, minimum_length: int) -> np.ndarray:
    """Absorb short interior runs while preserving truncated episode edges."""
    output = np.asarray(state, dtype=bool).copy()
    if minimum_length <= 1:
        return output
    while True:
        runs = _bool_runs(output)
        interior = [
            (end - start, index)
            for index, (start, end, _value) in enumerate(runs)
            if 0 < index < len(runs) - 1 and (end - start) < minimum_length
        ]
        if not interior:
            return output
        _, index = min(interior)
        start, end, _value = runs[index]
        output[start:end] = runs[index - 1][2]


def grasp_binary(
    signal: np.ndarray,
    valid: np.ndarray,
    close_hi: float,
    open_lo: float,
    minimum_dwell: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply open/closed hysteresis followed by non-causal min-dwell debounce."""
    signal = np.asarray(signal, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if signal.shape != valid.shape:
        raise ValueError(f"signal/valid shape mismatch: {signal.shape} vs {valid.shape}")
    if open_lo > close_hi:
        raise ValueError("grasp_open_lo must be <= grasp_close_hi")
    if minimum_dwell < 1:
        raise ValueError("minimum_dwell must be positive")
    state = np.zeros(len(signal), dtype=bool)
    first_valid = np.flatnonzero(valid)
    current = bool(signal[first_valid[0]] >= close_hi) if len(first_valid) else False
    for frame in range(len(signal)):
        if valid[frame]:
            if not current and signal[frame] > close_hi:
                current = True
            elif current and signal[frame] < open_lo:
                current = False
        state[frame] = current
    return state, absorb_short_runs(state, minimum_dwell)


def confirmed_hysteresis(
    signal: np.ndarray,
    valid: np.ndarray,
    close_hi: float,
    open_lo: float,
    confirm_ticks: int,
    minimum_state_ticks: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Debounce state changes and backdate confirmed transitions to their onset."""
    signal = np.asarray(signal, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if signal.shape != valid.shape:
        raise ValueError(f"signal/valid shape mismatch: {signal.shape} vs {valid.shape}")
    if not 0.0 <= open_lo <= close_hi <= 1.0:
        raise ValueError("robust grasp thresholds must satisfy 0 <= open <= close <= 1")
    if confirm_ticks < 1 or minimum_state_ticks < 1:
        raise ValueError("grasp temporal lengths must be positive")
    state = np.zeros(len(signal), dtype=bool)
    current = False
    pending: bool | None = None
    pending_start = 0
    for frame, value in enumerate(signal):
        state[frame] = current
        if not valid[frame]:
            pending = None
            continue
        wanted = bool(value > close_hi) if not current else bool(value >= open_lo)
        if wanted == current:
            pending = None
            continue
        if pending != wanted:
            pending = wanted
            pending_start = frame
        if frame - pending_start + 1 >= confirm_ticks:
            current = wanted
            state[pending_start : frame + 1] = current
            pending = None
    return state, absorb_short_runs(state, minimum_state_ticks)


def adaptive_index_grasp(
    index_command: np.ndarray,
    valid: np.ndarray,
    cfg: ProjectConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Normalize index flexion per episode, then apply robust temporal binarization."""
    command = np.asarray(index_command, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    window = int(cfg.mode2.grasp_robust_median_window)
    if window < 1 or window % 2 == 0:
        raise ValueError("grasp_robust_median_window must be a positive odd number")
    filtered = median_filter(command, size=window, mode="nearest")
    calibration_values = filtered[valid]
    if len(calibration_values):
        low, high = np.quantile(
            calibration_values,
            [cfg.mode2.grasp_robust_low_quantile, cfg.mode2.grasp_robust_high_quantile],
        )
    else:
        low, high = 0.0, 0.0
    dynamic_range = float(high - low)
    usable_range = max(dynamic_range, float(cfg.mode2.grasp_robust_minimum_range))
    score = np.clip((filtered - float(low)) / usable_range, 0.0, 1.0)
    if dynamic_range < cfg.mode2.grasp_robust_minimum_range:
        score[:] = 0.0
    before_dwell, closed = confirmed_hysteresis(
        score,
        valid,
        cfg.mode2.grasp_robust_close_hi,
        cfg.mode2.grasp_robust_open_lo,
        cfg.mode2.grasp_robust_confirm_ticks,
        cfg.mode2.grasp_robust_minimum_state_ticks,
    )
    calibration = {
        "index_open_baseline": float(low),
        "index_closed_reference": float(high),
        "index_dynamic_range": dynamic_range,
    }
    return score.astype(np.float32), before_dwell, closed, calibration


def _transitions(state: np.ndarray) -> list[int]:
    return (np.flatnonzero(np.diff(np.asarray(state, dtype=np.int8)) != 0) + 1).tolist()


def write_brainco_grasp_binary(
    cfg: ProjectConfig,
    episode_dir: str | Path,
) -> tuple[Path, Path, dict]:
    episode_dir = Path(episode_dir).expanduser().resolve()
    mode2_dir = episode_dir / "mode2"
    state = np.load(mode2_dir / "state_abs.npy").astype(np.float64)
    ticks_ns = np.load(mode2_dir / "ticks_ns.npy").astype(np.int64)
    camera_match = np.load(mode2_dir / "camera_match.npy").astype(np.int64)
    valid = np.load(mode2_dir / "valid.npy").astype(bool)
    if state.ndim != 2 or state.shape[1] != 30:
        raise ValueError(f"Mode2 state must be (T,30), got {state.shape}")
    if not (len(state) == len(ticks_ns) == len(camera_match) == len(valid)):
        raise ValueError("Mode2 grasp inputs do not share one timeline")

    signal_indices = [MOTOR_ORDER.index(name) for name in cfg.mode2.grasp_signal_fingers]
    index_motor = MOTOR_ORDER.index("index")
    output: dict[str, np.ndarray] = {
        "ticks_ns": ticks_ns,
        "camera_match": camera_match,
        "motor_order": np.asarray(MOTOR_ORDER),
        "signal_fingers": np.asarray(cfg.mode2.grasp_signal_fingers),
        "valid": valid,
        "main_close_hi": np.asarray(cfg.mode2.grasp_close_hi, dtype=np.float32),
        "main_open_lo": np.asarray(cfg.mode2.grasp_open_lo, dtype=np.float32),
        "main_min_dwell_ticks": np.asarray(cfg.mode2.grasp_min_dwell_ticks, dtype=np.int32),
        "diagnostic_index_threshold": np.asarray(
            cfg.mode2.grasp_diagnostic_index_threshold,
            dtype=np.float32,
        ),
        "diagnostic_min_dwell_ticks": np.asarray(
            cfg.mode2.grasp_diagnostic_min_dwell_ticks,
            dtype=np.int32,
        ),
    }
    report: dict = {
        "schema": "brainco_grasp_binary_v1",
        "frames": int(len(state)),
        "fps": float(cfg.timeline.control_hz),
        "motor_order": list(MOTOR_ORDER),
        "recommended_rule": {
            "signal": "per-side adaptive normalized BrainCo index motor",
            "calibration_quantiles": [
                float(cfg.mode2.grasp_robust_low_quantile),
                float(cfg.mode2.grasp_robust_high_quantile),
            ],
            "minimum_motor_range": float(cfg.mode2.grasp_robust_minimum_range),
            "median_window": int(cfg.mode2.grasp_robust_median_window),
            "close_hi": float(cfg.mode2.grasp_robust_close_hi),
            "open_lo": float(cfg.mode2.grasp_robust_open_lo),
            "confirm_ticks": int(cfg.mode2.grasp_robust_confirm_ticks),
            "minimum_state_ticks": int(cfg.mode2.grasp_robust_minimum_state_ticks),
        },
        "legacy_mean_rule": {
            "signal_fingers": list(cfg.mode2.grasp_signal_fingers),
            "signal": "mean selected motor command; each [0,1], 1=closed",
            "close_hi": float(cfg.mode2.grasp_close_hi),
            "open_lo": float(cfg.mode2.grasp_open_lo),
            "minimum_dwell_ticks": int(cfg.mode2.grasp_min_dwell_ticks),
        },
        "legacy_index_rule": {
            "signal": "index motor command",
            "threshold": float(cfg.mode2.grasp_diagnostic_index_threshold),
            "minimum_dwell_ticks": int(cfg.mode2.grasp_diagnostic_min_dwell_ticks),
        },
        "hands": {},
    }

    hands_path = episode_dir / "camera" / "hands_camera0.npz"
    with np.load(hands_path, allow_pickle=False) as hands:
        pinch_by_side = {
            side: hands[f"{side}_pinch_m"].astype(np.float64)[camera_match]
            for side in ("left", "right")
        }

    for side, command_slice in (("left", slice(18, 24)), ("right", slice(24, 30))):
        commands = state[:, command_slice]
        closure = commands[:, signal_indices].mean(axis=1)
        legacy_mean_before, legacy_mean = grasp_binary(
            closure,
            valid,
            cfg.mode2.grasp_close_hi,
            cfg.mode2.grasp_open_lo,
            cfg.mode2.grasp_min_dwell_ticks,
        )
        legacy_index_before, legacy_index = grasp_binary(
            commands[:, index_motor],
            valid,
            cfg.mode2.grasp_diagnostic_index_threshold,
            cfg.mode2.grasp_diagnostic_index_threshold,
            cfg.mode2.grasp_diagnostic_min_dwell_ticks,
        )
        robust_score, robust_before, closed, robust_calibration = adaptive_index_grasp(
            commands[:, index_motor],
            valid,
            cfg,
        )
        pinch_m = pinch_by_side[side]
        pinch_closed = pinch_m < cfg.perception.grasp_distance_m
        output[f"commands_{side}"] = commands.astype(np.float32)
        output[f"closure_{side}"] = closure.astype(np.float32)
        output[f"robust_score_{side}"] = robust_score
        output[f"closed_before_dwell_{side}"] = robust_before
        output[f"closed_{side}"] = closed
        output[f"legacy_mean_before_dwell_{side}"] = legacy_mean_before
        output[f"legacy_mean_closed_{side}"] = legacy_mean
        output[f"legacy_index_before_dwell_{side}"] = legacy_index_before
        output[f"legacy_index025_dwell5_{side}"] = legacy_index
        # Compatibility aliases retained until a Step2 grasp contract is selected.
        output[f"diagnostic_before_dwell_{side}"] = legacy_index_before
        output[f"diagnostic_index025_dwell5_{side}"] = legacy_index
        output[f"pico_pinch_m_{side}"] = pinch_m.astype(np.float32)
        output[f"pico_pinch_closed_{side}"] = pinch_closed
        report["hands"][side] = {
            "valid_ratio": float(valid.mean()),
            "closure_min_median_max": [
                float(np.min(closure)),
                float(np.median(closure)),
                float(np.max(closure)),
            ],
            "robust_calibration": robust_calibration,
            "closed_ratio": float(closed.mean()),
            "transition_frames": _transitions(closed),
            "minimum_state_changed_frames": int(np.count_nonzero(robust_before != closed)),
            "legacy_mean_closed_ratio": float(legacy_mean.mean()),
            "legacy_mean_transition_frames": _transitions(legacy_mean),
            "legacy_index_closed_ratio": float(legacy_index.mean()),
            "legacy_index_transition_frames": _transitions(legacy_index),
            "recommended_legacy_index_agreement_ratio": float(np.mean(closed == legacy_index)),
            "pico_pinch_agreement_ratio": float(np.mean(closed[valid] == pinch_closed[valid]))
            if valid.any()
            else 0.0,
        }

    output_path = mode2_dir / "brainco_grasp_binary.npz"
    report_path = episode_dir / "qa" / "brainco_grasp_binary_report.json"
    np.savez_compressed(output_path, **output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path, report_path, report


def _draw_hand_skeleton(
    image: np.ndarray,
    uv: np.ndarray,
    valid: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    for chain in HAND_CHAINS:
        for first, second in zip(chain[:-1], chain[1:], strict=True):
            if valid[first] and valid[second]:
                cv2.line(
                    image,
                    tuple(np.rint(uv[first]).astype(int)),
                    tuple(np.rint(uv[second]).astype(int)),
                    color,
                    2,
                    cv2.LINE_AA,
                )
    for joint, point in enumerate(uv):
        if not valid[joint]:
            continue
        center = tuple(np.rint(point).astype(int))
        cv2.circle(image, center, 4, (15, 15, 15), -1, cv2.LINE_AA)
        cv2.circle(image, center, 3, color, -1, cv2.LINE_AA)


def _wave_points(
    values: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    xs = np.linspace(x, x + width, len(values))
    ys = y + height - np.clip(values, 0.0, 1.0) * height
    return np.rint(np.column_stack((xs, ys))).astype(np.int32)


def _draw_digital_wave(
    panel: np.ndarray,
    values: np.ndarray,
    x: int,
    y: int,
    width: int,
    color: tuple[int, int, int],
) -> None:
    points = _wave_points(np.asarray(values, dtype=np.float64), x, y, width, 11)
    cv2.polylines(panel, [points], False, color, 1, cv2.LINE_AA)


def _draw_waveform_panel(
    data: dict[str, np.ndarray],
    frame: int,
    width: int,
    height: int,
    cfg: ProjectConfig,
) -> np.ndarray:
    panel = np.full((height, width, 3), (18, 22, 27), dtype=np.uint8)
    cv2.putText(
        panel,
        "BrainCo two-finger grasp | full episode waveform",
        (14, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (238, 242, 245),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "score  mean  thumb  index   ROBUST  MEAN30  IDX25  PICO",
        (14, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (185, 195, 202),
        1,
        cv2.LINE_AA,
    )
    colors = {
        "mean": (255, 185, 70),
        "score": (255, 210, 80),
        "thumb": (80, 155, 255),
        "index": (90, 205, 110),
        "main": (245, 245, 245),
        "alt": (150, 155, 160),
        "pico": (0, 205, 255),
    }
    for row, side in enumerate(("left", "right")):
        top = 50 + row * 213
        chart_x, chart_y = 66, top + 27
        chart_width, chart_height = width - 82, 95
        commands = data[f"commands_{side}"]
        closure = data[f"closure_{side}"]
        robust_score = data[f"robust_score_{side}"]
        main = data[f"closed_{side}"]
        legacy_mean = data[f"legacy_mean_closed_{side}"]
        legacy_index = data[f"legacy_index025_dwell5_{side}"]
        pico = data[f"pico_pinch_closed_{side}"]
        state_text = "CLOSED" if bool(main[frame]) else "OPEN"
        cv2.putText(
            panel,
            (
                f"{side.upper()} ROBUST {state_text}  score={robust_score[frame]:.3f}  "
                f"index={commands[frame, 2]:.3f}  mean={closure[frame]:.3f}"
            ),
            (14, top + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (0, 185, 255) if side == "left" else (255, 90, 210),
            1,
            cv2.LINE_AA,
        )
        cv2.rectangle(
            panel,
            (chart_x, chart_y),
            (chart_x + chart_width, chart_y + chart_height),
            (70, 76, 82),
            1,
        )
        for value, label in ((1.0, "1"), (0.5, ".5"), (0.0, "0")):
            y = chart_y + chart_height - int(value * chart_height)
            cv2.line(panel, (chart_x, y), (chart_x + chart_width, y), (45, 50, 55), 1)
            cv2.putText(
                panel,
                label,
                (42, y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.31,
                (160, 166, 172),
                1,
                cv2.LINE_AA,
            )
        for threshold, color in (
            (cfg.mode2.grasp_robust_close_hi, (220, 220, 220)),
            (cfg.mode2.grasp_robust_open_lo, (105, 110, 115)),
        ):
            y = chart_y + chart_height - int(float(threshold) * chart_height)
            cv2.line(panel, (chart_x, y), (chart_x + chart_width, y), color, 1)
        for values, color, thickness in (
            (robust_score, colors["score"], 2),
            (closure, colors["mean"], 1),
            (commands[:, 0], colors["thumb"], 1),
            (commands[:, 2], colors["index"], 1),
        ):
            cv2.polylines(
                panel,
                [_wave_points(values, chart_x, chart_y, chart_width, chart_height)],
                False,
                color,
                thickness,
                cv2.LINE_AA,
            )
        digital_rows = (
            ("ROB", main, colors["main"]),
            ("MEAN", legacy_mean, colors["mean"]),
            ("IDX", legacy_index, colors["alt"]),
            ("PICO", pico, colors["pico"]),
        )
        for digital_index, (label, values, color) in enumerate(digital_rows):
            y = chart_y + chart_height + 8 + digital_index * 15
            cv2.putText(
                panel,
                label,
                (14, y + 9),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                color,
                1,
                cv2.LINE_AA,
            )
            _draw_digital_wave(panel, values, chart_x, y, chart_width, color)
        cursor_x = chart_x + int(frame * chart_width / max(len(closure) - 1, 1))
        cv2.line(
            panel,
            (cursor_x, chart_y),
            (cursor_x, chart_y + chart_height + 67),
            (30, 235, 255),
            1,
            cv2.LINE_AA,
        )
    return panel


def render_brainco_grasp_overlay(
    cfg: ProjectConfig,
    source: str | Path,
    episode_dir: str | Path,
    output: str | Path | None = None,
) -> Path:
    source = Path(source).expanduser().resolve()
    episode_dir = Path(episode_dir).expanduser().resolve()
    grasp_path = episode_dir / "mode2" / "brainco_grasp_binary.npz"
    if not grasp_path.is_file():
        write_brainco_grasp_binary(cfg, episode_dir)
    with np.load(grasp_path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    with np.load(episode_dir / "camera" / "hand_keypoints_pixels.npz", allow_pickle=False) as archive:
        pixels = {name: archive[name] for name in archive.files}

    output_path = (
        Path(output).expanduser().resolve()
        if output is not None
        else episode_dir / "qa" / "step1_hand_keypoints_brainco_grasp.mp4"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path = episode_dir / "qa" / "brainco_grasp_binary_overlay.mp4"
    if legacy_path.is_file():
        legacy_path.unlink()
    width, image_height = (int(value) for value in pixels["image_size_wh"])
    panel_width = width
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(cfg.timeline.control_hz),
        (width + panel_width, image_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create grasp QA video: {output_path}")

    side_color = {"left": (0, 185, 255), "right": (255, 90, 210)}
    with PicoEpisode(source, cfg) as episode:
        try:
            for frame, camera_index_value in enumerate(data["camera_match"]):
                camera_index = int(camera_index_value)
                image = cv2.cvtColor(episode.image(camera_index), cv2.COLOR_RGB2BGR)
                if image.shape[:2] != (image_height, width):
                    image = cv2.resize(image, (width, image_height), interpolation=cv2.INTER_AREA)
                cv2.rectangle(image, (0, 0), (width, 48), (10, 15, 20), -1)
                cv2.putText(
                    image,
                    f"PICO hand keypoints + BrainCo grasp | {frame:04d}/{len(data['camera_match']) - 1:04d}",
                    (12, 23),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.53,
                    (240, 243, 245),
                    1,
                    cv2.LINE_AA,
                )
                for side in ("left", "right"):
                    color = side_color[side]
                    main = bool(data[f"closed_{side}"][frame])
                    legacy_index = bool(data[f"legacy_index025_dwell5_{side}"][frame])
                    uv = pixels[f"{side}_uv"][camera_index]
                    pixel_valid = pixels[f"{side}_pixel_valid"][camera_index]
                    _draw_hand_skeleton(image, uv, pixel_valid, color)
                    wrist_valid = bool(pixels[f"{side}_pixel_valid"][camera_index, 1])
                    if wrist_valid:
                        wrist = tuple(np.rint(pixels[f"{side}_uv"][camera_index, 1]).astype(int))
                        label = (
                            f"{side[0].upper()} ROB:{'C' if main else 'O'} "
                            f"IDX:{'C' if legacy_index else 'O'}"
                        )
                        cv2.putText(
                            image,
                            label,
                            (wrist[0] + 8, max(wrist[1] - 10, 48)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.47,
                            color,
                            2,
                            cv2.LINE_AA,
                        )
                panel = _draw_waveform_panel(data, frame, panel_width, image_height, cfg)
                writer.write(np.hstack((image, panel)))
        finally:
            writer.release()
    return output_path


def plot_brainco_grasp_timeline(
    cfg: ProjectConfig,
    episode_dir: str | Path,
    output: str | Path | None = None,
) -> Path:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episode_dir = Path(episode_dir).expanduser().resolve()
    with np.load(episode_dir / "mode2" / "brainco_grasp_binary.npz", allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    output_path = (
        Path(output).expanduser().resolve()
        if output is not None
        else episode_dir / "qa" / "brainco_grasp_binary_timeline.png"
    )
    time_s = np.arange(len(data["ticks_ns"])) / float(cfg.timeline.control_hz)
    figure, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    for axis, side in zip(axes, ("left", "right"), strict=True):
        commands = data[f"commands_{side}"]
        axis.plot(time_s, data[f"robust_score_{side}"], label="adaptive index score", linewidth=2.0)
        axis.plot(time_s, data[f"closure_{side}"], label="legacy mean(thumb,index)", alpha=0.7)
        axis.plot(time_s, commands[:, 0], label="thumb_flex", alpha=0.5)
        axis.plot(time_s, commands[:, 2], label="index", alpha=0.7)
        axis.axhline(
            cfg.mode2.grasp_robust_close_hi,
            color="black",
            linestyle="--",
            linewidth=1,
            label="robust close",
        )
        axis.axhline(
            cfg.mode2.grasp_robust_open_lo,
            color="gray",
            linestyle=":",
            linewidth=1,
            label="robust open",
        )
        axis.fill_between(
            time_s,
            0.0,
            1.0,
            where=data[f"closed_{side}"],
            color="tab:green",
            alpha=0.12,
            label="robust CLOSED",
        )
        axis.set_ylim(-0.02, 1.02)
        axis.set_ylabel(side)
        axis.grid(alpha=0.2)
    axes[0].legend(ncol=6, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("time (s)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path
