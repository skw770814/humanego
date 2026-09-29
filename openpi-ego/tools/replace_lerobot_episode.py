#!/usr/bin/env python3
"""Replace one LeRobot v2.1 episode with a copy of another episode.

The source dataset is never modified. The output keeps the same episode ids,
but the destination episode receives the source episode's parquet rows and
videos. Episode/global indices and metadata statistics are rewritten so the
result remains a self-contained, internally consistent LeRobot dataset.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil

from lerobot.common.datasets.compute_stats import aggregate_stats
from lerobot.common.datasets.utils import load_episodes_stats
from lerobot.common.datasets.utils import write_stats
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


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


def _replace_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    position = table.schema.get_field_index(name)
    if position < 0:
        raise ValueError(f"Required parquet column is missing: {name}")
    field = table.schema.field(position)
    return table.set_column(position, field, pa.array(values, type=field.type))


def _constant_stats(value: int, length: int) -> dict:
    return {
        "min": [value],
        "max": [value],
        "mean": [float(value)],
        "std": [0.0],
        "count": [length],
    }


def _index_stats(start: int, length: int) -> dict:
    values = np.arange(start, start + length, dtype=np.float64)
    return {
        "min": [int(values[0])],
        "max": [int(values[-1])],
        "mean": [float(values.mean())],
        "std": [float(values.std())],
        "count": [length],
    }


def _link_or_copy(source: Path, destination: Path, *, copy_files: bool) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy_files:
        shutil.copy2(source, destination)
        return "copy"
    try:
        os.link(source, destination)
        return "link"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def replace_episode(
    source: Path,
    output: Path,
    *,
    destination_episode: int,
    source_episode: int,
    random_seed: int | None,
    output_task: str | None,
    copy_videos: bool,
) -> None:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if source == output:
        raise ValueError("Output cannot overwrite the source dataset")

    info = _read_json(source / "meta" / "info.json")
    episodes = _read_jsonl(source / "meta" / "episodes.jsonl")
    stats_rows = _read_jsonl(source / "meta" / "episodes_stats.jsonl")
    episode_by_id = {int(row["episode_index"]): row for row in episodes}
    stats_by_id = {int(row["episode_index"]): row["stats"] for row in stats_rows}
    ids = sorted(episode_by_id)
    if ids != list(range(len(ids))):
        raise ValueError("This tool requires contiguous episode ids starting at zero")
    for episode_id in (destination_episode, source_episode):
        if episode_id not in episode_by_id:
            raise ValueError(f"Unknown episode id: {episode_id}")
    if destination_episode == source_episode:
        raise ValueError("Source and destination episodes must differ")

    chunks_size = int(info["chunks_size"])
    video_keys = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)

    linked_videos = copied_videos = total_frames = 0
    output_episodes: list[dict] = []
    output_stats: list[dict] = []
    try:
        for destination_id in ids:
            selected_id = source_episode if destination_id == destination_episode else destination_id
            selected_chunk = selected_id // chunks_size
            destination_chunk = destination_id // chunks_size
            source_parquet = source / info["data_path"].format(
                episode_chunk=selected_chunk, episode_index=selected_id
            )
            table = pq.read_table(source_parquet)
            length = table.num_rows
            table = _replace_column(
                table, "episode_index", np.full(length, destination_id, dtype=np.int64)
            )
            table = _replace_column(
                table, "index", np.arange(total_frames, total_frames + length, dtype=np.int64)
            )
            destination_parquet = temporary / info["data_path"].format(
                episode_chunk=destination_chunk, episode_index=destination_id
            )
            destination_parquet.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, destination_parquet)

            episode = copy.deepcopy(episode_by_id[selected_id])
            episode["episode_index"] = destination_id
            episode["length"] = length
            if output_task is not None:
                episode["tasks"] = [output_task]
            output_episodes.append(episode)

            stats = copy.deepcopy(stats_by_id[selected_id])
            stats["episode_index"] = _constant_stats(destination_id, length)
            stats["index"] = _index_stats(total_frames, length)
            output_stats.append({"episode_index": destination_id, "stats": stats})

            for video_key in video_keys:
                source_video = source / info["video_path"].format(
                    episode_chunk=selected_chunk,
                    video_key=video_key,
                    episode_index=selected_id,
                )
                destination_video = temporary / info["video_path"].format(
                    episode_chunk=destination_chunk,
                    video_key=video_key,
                    episode_index=destination_id,
                )
                mode = _link_or_copy(source_video, destination_video, copy_files=copy_videos)
                linked_videos += mode == "link"
                copied_videos += mode == "copy"
            total_frames += length

        output_info = copy.deepcopy(info)
        output_info["total_frames"] = total_frames
        if output_task is not None:
            output_info["total_tasks"] = 1
            tasks = [{"task_index": 0, "task": output_task}]
        else:
            tasks = _read_jsonl(source / "meta" / "tasks.jsonl")

        _write_json(temporary / "meta" / "info.json", output_info)
        _write_jsonl(temporary / "meta" / "episodes.jsonl", output_episodes)
        _write_jsonl(temporary / "meta" / "episodes_stats.jsonl", output_stats)
        _write_jsonl(temporary / "meta" / "tasks.jsonl", tasks)
        _write_json(
            temporary / "meta" / "replacement_manifest.json",
            {
                "source": str(source),
                "destination_episode": destination_episode,
                "replacement_source_episode": source_episode,
                "random_seed": random_seed,
                "output_task": output_task,
                "linked_videos": linked_videos,
                "copied_videos": copied_videos,
            },
        )
        modality = source / "meta" / "modality.json"
        if modality.is_file():
            shutil.copy2(modality, temporary / "meta" / "modality.json")
        write_stats(aggregate_stats(list(load_episodes_stats(temporary).values())), temporary)
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(f"Done: {output}")
    print(f"  episodes: {len(ids)}")
    print(f"  frames: {total_frames}")
    print(f"  episode {destination_episode} <- episode {source_episode}")
    print(f"  videos: {linked_videos} hard-linked, {copied_videos} copied")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Source LeRobot v2.1 dataset")
    parser.add_argument("output", type=Path, help="New output dataset")
    parser.add_argument("--destination-episode", type=int, required=True)
    parser.add_argument("--source-episode", type=int, required=True)
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--output-task", help="Assign this task text to every output episode")
    parser.add_argument("--copy-videos", action="store_true", help="Copy instead of hard-linking videos")
    args = parser.parse_args()
    replace_episode(
        args.source,
        args.output,
        destination_episode=args.destination_episode,
        source_episode=args.source_episode,
        random_seed=args.random_seed,
        output_task=args.output_task,
        copy_videos=args.copy_videos,
    )


if __name__ == "__main__":
    main()
