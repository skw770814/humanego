#!/usr/bin/env python3
"""Create a self-contained LeRobot v2.1 dataset from its first N episodes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

from lerobot.common.datasets.compute_stats import aggregate_stats
from lerobot.common.datasets.utils import load_episodes_stats
from lerobot.common.datasets.utils import write_stats


def _dataset_root(path: Path) -> Path:
    """Accept either a dataset root or an outer directory containing one dataset."""
    path = path.resolve()
    if (path / "meta" / "info.json").is_file():
        return path

    candidates = [child for child in path.iterdir() if (child / "meta" / "info.json").is_file()]
    if len(candidates) != 1:
        raise ValueError(f"Expected one LeRobot dataset below {path}, found {len(candidates)}")
    return candidates[0]


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)
        file.write("\n")


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for value in values:
            file.write(json.dumps(value, ensure_ascii=False) + "\n")


def _link_or_copy(source: Path, destination: Path, *, copy_files: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy_files:
        shutil.copy2(source, destination)
        return

    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def create_subset(source: Path, output: Path, *, num_episodes: int, copy_files: bool) -> None:
    source = _dataset_root(source)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")

    info = _read_json(source / "meta" / "info.json")
    episodes = _read_jsonl(source / "meta" / "episodes.jsonl")
    episode_stats_rows = _read_jsonl(source / "meta" / "episodes_stats.jsonl")
    tasks = _read_jsonl(source / "meta" / "tasks.jsonl")
    if num_episodes > len(episodes):
        raise ValueError(f"Requested {num_episodes} episodes, but source only has {len(episodes)}")

    episodes = episodes[:num_episodes]
    episode_stats_rows = episode_stats_rows[:num_episodes]
    expected_indices = list(range(num_episodes))
    if [episode["episode_index"] for episode in episodes] != expected_indices:
        raise ValueError("The source prefix is not consecutively indexed from zero")
    if [row["episode_index"] for row in episode_stats_rows] != expected_indices:
        raise ValueError("Episode stats are not aligned with the episode metadata")

    used_task_indices = {episode["task_index"] for episode in episodes}
    tasks = [task for task in tasks if task["task_index"] in used_task_indices]
    video_keys = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]

    info["total_episodes"] = num_episodes
    info["total_frames"] = sum(episode["length"] for episode in episodes)
    info["total_tasks"] = len(tasks)
    info["total_chunks"] = (num_episodes + info["chunks_size"] - 1) // info["chunks_size"]
    info["total_videos"] = num_episodes * len(video_keys)

    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)

    try:
        _write_json(temporary / "meta" / "info.json", info)
        _write_jsonl(temporary / "meta" / "episodes.jsonl", episodes)
        _write_jsonl(temporary / "meta" / "episodes_stats.jsonl", episode_stats_rows)
        _write_jsonl(temporary / "meta" / "tasks.jsonl", tasks)

        for episode_index in expected_indices:
            episode_chunk = episode_index // info["chunks_size"]
            fields = {"episode_chunk": episode_chunk, "episode_index": episode_index}
            data_path = Path(info["data_path"].format(**fields))
            if not (source / data_path).is_file():
                raise FileNotFoundError(source / data_path)
            _link_or_copy(source / data_path, temporary / data_path, copy_files=copy_files)

            for video_key in video_keys:
                video_fields = fields | {"video_key": video_key}
                video_path = Path(info["video_path"].format(**video_fields))
                if not (source / video_path).is_file():
                    raise FileNotFoundError(source / video_path)
                _link_or_copy(source / video_path, temporary / video_path, copy_files=copy_files)

        loaded_episode_stats = load_episodes_stats(temporary)
        write_stats(aggregate_stats(list(loaded_episode_stats.values())), temporary)
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--num-episodes", type=int, required=True)
    parser.add_argument(
        "--copy-files",
        action="store_true",
        help="Physically copy media and parquet files instead of space-saving hard links.",
    )
    args = parser.parse_args()
    create_subset(args.source, args.output, num_episodes=args.num_episodes, copy_files=args.copy_files)


if __name__ == "__main__":
    main()
