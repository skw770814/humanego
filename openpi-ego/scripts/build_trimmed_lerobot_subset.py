#!/usr/bin/env python3
"""Build a self-contained LeRobot v2.1 dataset from a hand-picked set of episodes,
dropping a per-episode number of leading frames from each one.

This is a standalone tool. It does not modify the source dataset or any other
file in the repository. Given a spec that maps *original* episode ids to the
number of leading frames to drop, it:

  1. Selects exactly the episodes named in the spec, sorted by original id, and
     re-indexes them to 0..K-1.
  2. Trims the first N rows of each episode's parquet and re-bases
     ``frame_index`` (0..L-1), ``timestamp`` (frame_index / fps) and the global
     ``index``.
  3. Re-encodes each of the four camera videos so frame 0 of the output is the
     first kept frame and its PTS restarts at 0 -- keeping image, state and
     action strictly time-aligned. Videos are trimmed in parallel across cores.
  4. Rewrites every meta file (info.json, episodes.jsonl, episodes_stats.jsonl,
     tasks.jsonl, stats.json). Numeric per-episode stats are recomputed exactly;
     image per-episode stats are carried over from the source (they are robust
     to dropping a short prefix and are not used for openpi action/state norm).

Spec format -- JSON object or CSV, keyed by the ORIGINAL episode id:

  JSON:  {"3": 45, "17": 30, "58": 0, ...}
  CSV:   one "episode_id,drop_frames" pair per line (a header row is allowed).

Every episode you want in the output must appear in the spec; drop may be 0.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess

from lerobot.common.datasets.compute_stats import aggregate_stats
from lerobot.common.datasets.utils import load_episodes_stats
from lerobot.common.datasets.utils import write_stats
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Small IO helpers
# ---------------------------------------------------------------------------
def _dataset_root(path: Path) -> Path:
    """Accept either a dataset root or an outer directory containing exactly one."""
    path = path.expanduser().resolve()
    if (path / "meta" / "info.json").is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(path)
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


def _load_spec(path: Path) -> dict[int, int]:
    """Return {original_episode_id: leading_frames_to_drop}.

    Accepts a JSON object ({"3": 45, ...}) or a CSV. A CSV may either be a bare
    "id,drop" pair per line, or a header row naming an id column (id/episode_id)
    and a drop column (drop/drop_frames) -- rows whose drop cell is blank are
    treated as *not selected* and skipped, so the episodes_overview.csv table
    doubles as the spec: just fill drop_frames for the episodes you want.
    """
    text = path.read_text(encoding="utf-8").strip()
    spec: dict[int, int] = {}
    if text.startswith("{"):
        for key, value in json.loads(text).items():
            spec[int(key)] = int(value)
    else:
        lines = text.splitlines()
        header = next(csv.reader(lines[:1]), [])
        id_aliases, drop_aliases = {"id", "episode_id", "episode"}, {"drop", "drop_frames"}
        if any(cell.strip().lower() in id_aliases for cell in header):
            id_key = next(c for c in header if c.strip().lower() in id_aliases)
            drop_key = next((c for c in header if c.strip().lower() in drop_aliases), None)
            if drop_key is None:
                raise ValueError("CSV header has no drop/drop_frames column")
            for record in csv.DictReader(lines):
                raw_drop = (record.get(drop_key) or "").strip()
                if raw_drop == "":
                    continue  # blank drop == not selected
                spec[int(record[id_key])] = int(raw_drop)
        else:
            for row in csv.reader(lines):
                if not row or not row[0].strip():
                    continue
                if len(row) < 2 or not row[1].strip():
                    raise ValueError(f"CSV spec row needs 'id,drop': {row!r}")
                spec[int(row[0])] = int(row[1])
    if not spec:
        raise ValueError(f"No episodes selected in spec (fill the drop column): {path}")
    for episode_id, drop in spec.items():
        if drop < 0:
            raise ValueError(f"drop must be >= 0, got {drop} for episode {episode_id}")
    return spec


# ---------------------------------------------------------------------------
# Per-episode statistics
# ---------------------------------------------------------------------------
def _numeric_stats(values: np.ndarray, length: int) -> dict:
    """min/max/mean/std over axis 0 with a per-feature count, matching LeRobot."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [length],
    }


def _episode_stats(table: pa.Table, source_stats: dict, length: int) -> dict:
    """Recompute numeric feature stats exactly; carry image stats over unchanged."""
    stats: dict = {}
    for name in table.column_names:
        column = table.column(name)
        first = column[0].as_py()
        if isinstance(first, list):
            array = np.asarray(column.to_pylist(), dtype=np.float64)
        else:
            array = column.to_numpy(zero_copy_only=False).astype(np.float64)
        stats[name] = _numeric_stats(array, length)
    # Image features are not columns in the parquet; keep the source stats.
    for name, value in source_stats.items():
        if name not in stats:
            stats[name] = copy.deepcopy(value)
    return stats


# ---------------------------------------------------------------------------
# Video trimming
# ---------------------------------------------------------------------------
def _count_frames(path: Path) -> int:
    """Fast packet count (no full decode); packets == frames for these mp4s."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-count_packets", "-show_entries", "stream=nb_read_packets",
            "-of", "csv=p=0", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    return int(result.stdout.strip())


def _trim_video(job: dict) -> tuple[str, int]:
    """Drop the first ``drop`` frames, restart PTS at 0, re-encode. Returns (dst, frames)."""
    source, destination, drop = Path(job["source"]), Path(job["destination"]), job["drop"]
    fps, codec, crf, preset = job["fps"], job["codec"], job["crf"], job["preset"]
    destination.parent.mkdir(parents=True, exist_ok=True)

    encoder = {"h264": "libx264", "hevc": "libx265", "av1": "libaom-av1"}[codec]
    if drop == 0:
        video_filter = "setpts=PTS-STARTPTS"
    else:
        video_filter = rf"select=gte(n\,{drop}),setpts=PTS-STARTPTS"

    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(source),
        "-vf", video_filter,
        "-an", "-vsync", "cfr", "-r", str(fps),
        "-c:v", encoder, "-crf", str(crf), "-preset", preset,
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(destination),
    ]
    if codec == "av1":  # libaom needs an explicit constant-quality setup
        command[command.index("-preset") + 1] = "8"
        command[command.index("-preset")] = "-cpu-used"
        command += ["-b:v", "0", "-row-mt", "1"]

    subprocess.run(command, check=True, capture_output=True)
    return str(destination), _count_frames(destination)


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------
def build_dataset(
    source: Path | str,
    output: Path | str,
    spec: dict[int, int],
    *,
    jobs: int,
    video_codec: str = "h264",
    crf: int = 20,
    preset: str = "veryfast",
    min_frames: int = 2,
    progress=None,
) -> dict:
    """Build a trimmed subset from an in-memory ``{original_episode_id: drop}`` spec.

    Returns a summary dict. ``progress`` is an optional ``callable(str)`` used for
    log lines (defaults to :func:`print`); pass a no-op to silence it.
    """
    log = progress or print
    source = _dataset_root(Path(source))
    output = Path(output).expanduser().resolve()
    if output == source:
        raise ValueError("Output cannot overwrite the source dataset")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if not spec:
        raise ValueError("Empty spec: no episodes selected")

    info = _read_json(source / "meta" / "info.json")
    episodes = {row["episode_index"]: row for row in _read_jsonl(source / "meta" / "episodes.jsonl")}
    source_stats = {row["episode_index"]: row["stats"] for row in _read_jsonl(source / "meta" / "episodes_stats.jsonl")}
    tasks = {row["task_index"]: row["task"] for row in _read_jsonl(source / "meta" / "tasks.jsonl")}
    fps = int(info["fps"])
    chunks_size = int(info["chunks_size"])
    video_keys = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]

    spec = {int(k): int(v) for k, v in spec.items()}
    selected = sorted(spec)
    missing = [episode_id for episode_id in selected if episode_id not in episodes]
    if missing:
        raise ValueError(f"Spec references unknown episode ids: {missing}")
    for episode_id in selected:
        length, drop = episodes[episode_id]["length"], spec[episode_id]
        if drop < 0:
            raise ValueError(f"episode {episode_id}: drop must be >= 0, got {drop}")
        if drop >= length - min_frames:
            raise ValueError(
                f"episode {episode_id}: dropping {drop} of {length} frames leaves "
                f"< {min_frames} frames"
            )

    # task_index is authoritative in the parquet (episodes.jsonl only lists task strings).
    # Cheap single-column pre-pass to learn which task ids the selection uses.
    old_task_index: dict[int, np.ndarray] = {}
    for episode_id in selected:
        chunk = episode_id // chunks_size
        parquet = source / info["data_path"].format(episode_chunk=chunk, episode_index=episode_id)
        old_task_index[episode_id] = pq.read_table(parquet, columns=["task_index"]).column(0).to_numpy()
    used_tasks = sorted({int(v) for array in old_task_index.values() for v in np.unique(array)})
    task_remap = {old: new for new, old in enumerate(used_tasks)}

    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)

    try:
        new_episodes: list[dict] = []
        new_stats_rows: list[dict] = []
        video_jobs: list[dict] = []
        global_index = 0

        for new_id, old_id in enumerate(selected):
            drop = spec[old_id]
            new_chunk = new_id // chunks_size
            old_chunk = old_id // chunks_size

            src_parquet = source / info["data_path"].format(episode_chunk=old_chunk, episode_index=old_id)
            table = pq.read_table(src_parquet).slice(drop)
            length = table.num_rows

            new_frame_index = np.arange(length, dtype=np.int64)
            remapped_task = np.array([task_remap[int(t)] for t in old_task_index[old_id][drop:]], dtype=np.int64)
            new_columns = {
                "frame_index": new_frame_index,
                "timestamp": (new_frame_index / fps).astype(np.float32),
                "episode_index": np.full(length, new_id, dtype=np.int64),
                "index": np.arange(global_index, global_index + length, dtype=np.int64),
                "task_index": remapped_task,
            }
            for name, array in new_columns.items():
                field = table.schema.field(name)
                position = table.schema.get_field_index(name)
                table = table.set_column(position, field, pa.array(array, type=field.type))
            global_index += length

            dst_parquet = temporary / info["data_path"].format(episode_chunk=new_chunk, episode_index=new_id)
            dst_parquet.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, dst_parquet)

            new_episodes.append({
                "episode_index": new_id,
                "tasks": episodes[old_id]["tasks"],
                "length": length,
            })
            stats = _episode_stats(table, source_stats[old_id], length)
            new_stats_rows.append({"episode_index": new_id, "stats": stats})

            for video_key in video_keys:
                src_video = source / info["video_path"].format(
                    episode_chunk=old_chunk, video_key=video_key, episode_index=old_id
                )
                dst_video = temporary / info["video_path"].format(
                    episode_chunk=new_chunk, video_key=video_key, episode_index=new_id
                )
                video_jobs.append({
                    "source": str(src_video), "destination": str(dst_video), "drop": drop,
                    "expected": length, "old_id": old_id, "new_id": new_id, "video_key": video_key,
                    "fps": fps, "codec": video_codec, "crf": crf, "preset": preset,
                })

        # Trim all videos in parallel; this is the expensive stage.
        log(f"Trimming {len(video_jobs)} videos with {jobs} workers ({video_codec})...")
        expected_by_dst = {job["destination"]: job for job in video_jobs}
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
            for done, (destination, frames) in enumerate(pool.map(_trim_video, video_jobs), start=1):
                job = expected_by_dst[destination]
                if frames != job["expected"]:
                    raise RuntimeError(
                        f"Frame/row mismatch (episode {job['new_id']} {job['video_key']}): "
                        f"video={frames} parquet={job['expected']}"
                    )
                if done % 20 == 0 or done == len(video_jobs):
                    log(f"  {done}/{len(video_jobs)} videos done")

        # Meta files.
        info = copy.deepcopy(info)
        info["total_episodes"] = len(selected)
        info["total_frames"] = sum(episode["length"] for episode in new_episodes)
        info["total_tasks"] = len(used_tasks)
        info["total_videos"] = len(selected) * len(video_keys)
        info["total_chunks"] = (len(selected) + chunks_size - 1) // chunks_size
        info["splits"] = {"train": f"0:{len(selected)}"}
        for video_key in video_keys:
            info["features"][video_key]["info"]["video.codec"] = video_codec

        _write_json(temporary / "meta" / "info.json", info)
        _write_jsonl(temporary / "meta" / "episodes.jsonl", new_episodes)
        _write_jsonl(temporary / "meta" / "episodes_stats.jsonl", new_stats_rows)
        _write_jsonl(
            temporary / "meta" / "tasks.jsonl",
            [{"task_index": task_remap[old], "task": tasks[old]} for old in used_tasks],
        )
        _write_json(
            temporary / "meta" / "trim_manifest.json",
            {"source": str(source), "fps": fps, "video_codec": video_codec,
             "episodes": [{"new_id": new_id, "old_id": old_id, "dropped": spec[old_id],
                           "length": new_episodes[new_id]["length"]}
                          for new_id, old_id in enumerate(selected)]},
        )

        # Global stats.json via LeRobot aggregation.
        loaded = load_episodes_stats(temporary)
        write_stats(aggregate_stats(list(loaded.values())), temporary)

        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    total_frames = sum(episode["length"] for episode in new_episodes)
    log(f"\nDone: {output}")
    log(f"  episodes: {len(selected)}   frames: {total_frames}   codec: {video_codec}")
    return {
        "output": str(output),
        "episodes": len(selected),
        "frames": total_frames,
        "video_codec": video_codec,
        "mapping": [{"new_id": new_id, "old_id": old_id, "dropped": spec[old_id]}
                    for new_id, old_id in enumerate(selected)],
    }


def build(args: argparse.Namespace) -> None:
    build_dataset(
        source=args.source,
        output=args.output,
        spec=_load_spec(args.spec),
        jobs=args.jobs,
        video_codec=args.video_codec,
        crf=args.crf,
        preset=args.preset,
        min_frames=args.min_frames,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="Source LeRobot v2.1 dataset root.")
    parser.add_argument("--output", type=Path, required=True, help="Output dataset root (must not exist).")
    parser.add_argument("--spec", type=Path, required=True, help="JSON/CSV mapping original episode id -> leading frames to drop.")
    parser.add_argument("--jobs", type=int, default=min(32, os.cpu_count() or 8), help="Parallel ffmpeg workers.")
    parser.add_argument("--video-codec", choices=["h264", "hevc", "av1"], default="h264", help="Output video codec (h264 is fastest here).")
    parser.add_argument("--crf", type=int, default=20, help="Encoder CRF (quality; lower = better/larger).")
    parser.add_argument("--preset", default="veryfast", help="x264/x265 preset (ignored for av1).")
    parser.add_argument("--min-frames", type=int, default=2, help="Refuse to leave fewer than this many frames.")
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
