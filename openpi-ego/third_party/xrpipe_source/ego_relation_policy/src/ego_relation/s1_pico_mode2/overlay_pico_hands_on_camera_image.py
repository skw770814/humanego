#!/usr/bin/env python3
"""Render the Step1 PICO hand-keypoint pixel contract on camera frames."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from ego_relation.config import ProjectConfig, load_config
from ego_relation.s1_pico_mode2.pico import PicoEpisode, prepare_pico_episode


HAND_CHAINS = (
    (1, 0),
    (1, 2, 3, 4, 5),
    (1, 6, 7, 8, 9, 10),
    (1, 11, 12, 13, 14, 15),
    (1, 16, 17, 18, 19, 20),
    (1, 21, 22, 23, 24, 25),
)


def _draw_hand(
    image: np.ndarray,
    uv: np.ndarray,
    valid: np.ndarray,
    color: tuple[int, int, int],
    prefix: str,
    *,
    show_indices: bool,
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
    labelled = {1, 5, 10, 15, 20, 25}
    for joint, point in enumerate(uv):
        if not valid[joint]:
            continue
        center = tuple(np.rint(point).astype(int))
        cv2.circle(image, center, 4, (12, 12, 12), -1, cv2.LINE_AA)
        cv2.circle(image, center, 3, color, -1, cv2.LINE_AA)
        if show_indices or joint in labelled:
            cv2.putText(
                image,
                f"{prefix}{joint}",
                (center[0] + 4, center[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                color,
                1,
                cv2.LINE_AA,
            )


def _estimate_fps(timestamps_ns: np.ndarray) -> float:
    if len(timestamps_ns) < 2:
        return 30.0
    intervals = np.diff(np.asarray(timestamps_ns, dtype=np.int64)).astype(np.float64)
    interval = float(np.median(intervals)) / 1e9
    return 1.0 / interval if interval > 0 else 30.0


def render_pixel_overlay(
    cfg_source: str | Path | ProjectConfig,
    source: Path,
    episode_dir: Path,
    output: Path | None = None,
    *,
    max_frames: int | None = None,
    fps: float | None = None,
    show_indices: bool = False,
) -> Path:
    cfg = cfg_source if isinstance(cfg_source, ProjectConfig) else load_config(cfg_source)
    source = source.expanduser().resolve()
    episode_dir = episode_dir.expanduser().resolve()
    pixels_path = episode_dir / "camera" / "hand_keypoints_pixels.npz"
    if not pixels_path.is_file():
        prepare_pico_episode(cfg, source, episode_dir)
    output = (
        output.expanduser().resolve()
        if output is not None
        else episode_dir / "qa" / "hand_keypoints_pixels_overlay.mp4"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    with np.load(pixels_path, allow_pickle=False) as archive:
        timestamps = archive["timestamp_ns"].astype(np.int64)
        image_size = archive["image_size_wh"].astype(int)
        left_uv = archive["left_uv"].astype(np.float64)
        right_uv = archive["right_uv"].astype(np.float64)
        left_valid = archive["left_pixel_valid"].astype(bool)
        right_valid = archive["right_pixel_valid"].astype(bool)
    frame_count = len(timestamps)
    if max_frames is not None:
        frame_count = min(frame_count, int(max_frames))
    width, height = (int(value) for value in image_size)
    output_fps = float(fps) if fps is not None else _estimate_fps(timestamps)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create overlay video: {output}")

    with PicoEpisode(source, cfg) as episode:
        try:
            for frame in range(frame_count):
                image = cv2.cvtColor(episode.image(frame), cv2.COLOR_RGB2BGR)
                if image.shape[:2] != (height, width):
                    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                _draw_hand(
                    image,
                    left_uv[frame],
                    left_valid[frame],
                    (0, 185, 255),
                    "L",
                    show_indices=show_indices,
                )
                _draw_hand(
                    image,
                    right_uv[frame],
                    right_valid[frame],
                    (255, 90, 210),
                    "R",
                    show_indices=show_indices,
                )
                cv2.rectangle(image, (0, 0), (width, 58), (10, 15, 20), -1)
                cv2.putText(
                    image,
                    f"PICO hand keypoints -> left-camera pixels | frame {frame:04d}/{frame_count - 1:04d}",
                    (14, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.57,
                    (240, 243, 245),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    f"valid joints L {int(left_valid[frame].sum()):02d}/26  R {int(right_valid[frame].sum()):02d}/26 | OpenCV u-right v-down",
                    (14, 47),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (180, 215, 225),
                    1,
                    cv2.LINE_AA,
                )
                writer.write(image)
        finally:
            writer.release()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay Step1 PICO hand keypoints in camera pixels")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--show-indices", action="store_true")
    args = parser.parse_args()
    output = render_pixel_overlay(
        args.config,
        args.source,
        args.episode_dir,
        args.output,
        max_frames=args.max_frames,
        fps=args.fps,
        show_indices=args.show_indices,
    )
    print(output)


if __name__ == "__main__":
    main()
