#!/usr/bin/env python3
"""A responsive local web GUI for reviewing and exporting LeRobot datasets."""

from __future__ import annotations

import argparse
import copy
from datetime import UTC
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import json
import math
import mimetypes
import os
from pathlib import Path
import shutil
import sys
import threading
from urllib.parse import parse_qs
from urllib.parse import urlparse
import webbrowser

from lerobot.common.datasets.compute_stats import aggregate_stats
from lerobot.common.datasets.utils import load_episodes_stats
from lerobot.common.datasets.utils import write_stats
import pyarrow as pa
import pyarrow.parquet as pq

# The trimming/re-encoding builder lives one directory up (scripts/). Reuse it so
# GUI export and the standalone CLI share exactly the same, tested pipeline.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import build_trimmed_lerobot_subset as trimmer  # noqa: E402

UI_ROOT = Path(__file__).resolve().parent / "static"
VALID_DECISIONS = {"undecided", "keep", "reject"}


def resolve_dataset_root(path: Path) -> Path:
    """Accept a LeRobot root or an outer directory containing exactly one root."""
    path = path.expanduser().resolve()
    if (path / "meta" / "info.json").is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(path)

    candidates = [child for child in path.iterdir() if (child / "meta" / "info.json").is_file()]
    if len(candidates) != 1:
        raise ValueError(f"Expected one LeRobot dataset below {path}, found {len(candidates)}")
    return candidates[0]


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for value in values:
            file.write(json.dumps(value, ensure_ascii=False) + "\n")


def link_or_copy(source: Path, destination: Path, *, copy_files: bool) -> str:
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


class SelectionStore:
    def __init__(self, path: Path, dataset_root: Path):
        self.path = path
        self.dataset_root = dataset_root
        self.lock = threading.RLock()
        self.decisions: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        value = read_json(self.path)
        if value.get("dataset_root") != str(self.dataset_root):
            raise ValueError(f"Selection file belongs to another dataset: {self.path}")
        self.decisions = value.get("decisions", {})

    def snapshot(self) -> dict[str, dict]:
        with self.lock:
            return copy.deepcopy(self.decisions)

    @staticmethod
    def _default() -> dict:
        return {"status": "undecided", "note": "", "start_frame": 0}

    def _prune_locked(self, key: str, decision: dict) -> dict:
        """Drop a decision only when it carries no information at all."""
        if decision["status"] == "undecided" and not decision["note"] and not decision["start_frame"]:
            self.decisions.pop(key, None)
            return self._default()
        self.decisions[key] = decision
        return decision

    def update(self, episode_index: int, status: str, note: str) -> dict:
        if status not in VALID_DECISIONS:
            raise ValueError(f"Invalid decision: {status}")
        with self.lock:
            key = str(episode_index)
            decision = {**self._default(), **self.decisions.get(key, {})}
            decision["status"] = status
            decision["note"] = note.strip()
            decision = self._prune_locked(key, decision)
            self._save_locked()
            return copy.deepcopy(decision)

    def set_trim(self, episode_index: int, start_frame: int) -> dict:
        """Set the number of leading frames to drop for this episode (0 clears it)."""
        if start_frame < 0:
            raise ValueError("start_frame must be >= 0")
        with self.lock:
            key = str(episode_index)
            decision = {**self._default(), **self.decisions.get(key, {})}
            decision["start_frame"] = int(start_frame)
            decision = self._prune_locked(key, decision)
            self._save_locked()
            return copy.deepcopy(decision)

    def _save_locked(self) -> None:
        payload = {
            "version": 1,
            "dataset_root": str(self.dataset_root),
            "updated_at": datetime.now(tz=UTC).isoformat(),
            "decisions": self.decisions,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        write_json(temporary, payload)
        temporary.replace(self.path)


class DatasetCatalog:
    def __init__(self, root: Path, state_path: Path | None = None):
        self.root = resolve_dataset_root(root)
        self.info = read_json(self.root / "meta" / "info.json")
        self.episodes = read_jsonl(self.root / "meta" / "episodes.jsonl")
        self.episode_stats = read_jsonl(self.root / "meta" / "episodes_stats.jsonl")
        task_rows = read_jsonl(self.root / "meta" / "tasks.jsonl")
        self.tasks = {row["task_index"]: row["task"] for row in task_rows}
        self.video_keys = [
            key for key, feature in self.info["features"].items() if feature.get("dtype") == "video"
        ]
        self.episodes_by_index = {row["episode_index"]: row for row in self.episodes}
        self.stats_by_index = {row["episode_index"]: row for row in self.episode_stats}
        if len(self.episodes_by_index) != len(self.episodes):
            raise ValueError("Duplicate episode indices in metadata")
        default_state = self.root / ".lerobot-curator" / "selection.json"
        self.selection = SelectionStore((state_path or default_state).resolve(), self.root)
        self.export_lock = threading.Lock()

    def video_path(self, episode_index: int, video_key: str) -> Path:
        if episode_index not in self.episodes_by_index:
            raise KeyError(f"Unknown episode: {episode_index}")
        if video_key not in self.video_keys:
            raise KeyError(f"Unknown video key: {video_key}")
        episode_chunk = episode_index // self.info["chunks_size"]
        relative = self.info["video_path"].format(
            episode_chunk=episode_chunk,
            episode_index=episode_index,
            video_key=video_key,
        )
        path = self.root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def catalog_payload(self) -> dict:
        decisions = self.selection.snapshot()
        episodes = []
        counts = {"undecided": 0, "keep": 0, "reject": 0}
        for row in self.episodes:
            index = row["episode_index"]
            decision = decisions.get(str(index), {})
            status = decision.get("status", "undecided")
            counts[status] = counts.get(status, 0) + 1
            # episodes.jsonl may store either task_index or a list of task strings.
            task_index = row.get("task_index")
            if task_index is not None:
                task = self.tasks.get(task_index, f"Task {task_index}")
            else:
                episode_tasks = row.get("tasks") or []
                task = episode_tasks[0] if episode_tasks else ""
            episodes.append(
                {
                    "episode_index": index,
                    "length": row["length"],
                    "duration": row["length"] / self.info["fps"],
                    "task": task,
                    "status": status,
                    "note": decision.get("note", ""),
                    "start_frame": int(decision.get("start_frame", 0)),
                }
            )
        return {
            "name": self.root.name,
            "root": str(self.root),
            "fps": self.info["fps"],
            "total_episodes": len(self.episodes),
            "video_keys": self.video_keys,
            "episodes": episodes,
            "counts": counts,
            "state_path": str(self.selection.path),
            "default_output": str(self.root.parent / f"{self.root.name}_curated"),
        }

    def export_trimmed(
        self, output: Path, *, jobs: int, video_codec: str = "h264", crf: int = 20, preset: str = "veryfast"
    ) -> dict:
        """Export kept episodes, dropping each one's ``start_frame`` leading frames.

        Writes a spec file next to the output *and* builds the trimmed, re-encoded
        LeRobot dataset by delegating to the shared ``build_trimmed_lerobot_subset``
        pipeline. Only episodes marked "keep" are exported; a kept episode with no
        trim start drops zero frames.
        """
        decisions = self.selection.snapshot()
        selected = sorted(
            index
            for index in self.episodes_by_index
            if decisions.get(str(index), {}).get("status") == "keep"
        )
        if not selected:
            raise ValueError("No episodes are marked as keep")

        output = output.expanduser()
        if not output.is_absolute():
            output = self.root.parent / output
        output = output.resolve()
        if output == self.root:
            raise ValueError("Output cannot overwrite the source dataset")
        if output.exists():
            raise FileExistsError(f"Output already exists: {output}")

        spec = {index: int(decisions.get(str(index), {}).get("start_frame", 0)) for index in selected}

        with self.export_lock:
            # Persist the spec next to the output so the same cut is reproducible from the CLI.
            spec_path = output.parent / f"{output.name}.trim_spec.csv"
            spec_path.parent.mkdir(parents=True, exist_ok=True)
            with spec_path.open("w", encoding="utf-8") as file:
                file.write("episode_id,drop_frames\n")
                for index in selected:
                    file.write(f"{index},{spec[index]}\n")

            summary = trimmer.build_dataset(
                source=self.root,
                output=output,
                spec=spec,
                jobs=jobs,
                video_codec=video_codec,
                crf=crf,
                preset=preset,
                progress=lambda message: None,
            )
        summary["spec_path"] = str(spec_path)
        summary["trimmed_episodes"] = sum(1 for value in spec.values() if value > 0)
        return summary

    def _export_selected(
        self,
        selected: list[int],
        decisions: dict[str, dict],
        output: Path,
        *,
        copy_files: bool,
    ) -> dict:
        temporary = output.with_name(f".{output.name}.tmp")
        if temporary.exists():
            shutil.rmtree(temporary)

        selected_rows = [self.episodes_by_index[index] for index in selected]
        used_tasks = sorted({row["task_index"] for row in selected_rows})
        task_remap = {old_index: new_index for new_index, old_index in enumerate(used_tasks)}
        task_rows = [
            {"task_index": task_remap[old_index], "task": self.tasks[old_index]} for old_index in used_tasks
        ]
        info = copy.deepcopy(self.info)
        info["total_episodes"] = len(selected)
        info["total_frames"] = sum(row["length"] for row in selected_rows)
        info["total_tasks"] = len(task_rows)
        info["total_chunks"] = math.ceil(len(selected) / info["chunks_size"])
        info["total_videos"] = len(selected) * len(self.video_keys)

        exported_episodes = []
        exported_stats = []
        provenance = []
        global_frame_index = 0
        linked_files = 0
        copied_files = 0
        try:
            for new_index, old_index in enumerate(selected):
                source_episode = self.episodes_by_index[old_index]
                length = source_episode["length"]
                new_task_index = task_remap[source_episode["task_index"]]
                new_episode = copy.deepcopy(source_episode)
                new_episode["episode_index"] = new_index
                new_episode["task_index"] = new_task_index
                exported_episodes.append(new_episode)

                new_chunk = new_index // info["chunks_size"]
                old_chunk = old_index // self.info["chunks_size"]
                source_data_path = self.root / self.info["data_path"].format(
                    episode_chunk=old_chunk,
                    episode_index=old_index,
                )
                destination_data_path = temporary / info["data_path"].format(
                    episode_chunk=new_chunk,
                    episode_index=new_index,
                )
                destination_data_path.parent.mkdir(parents=True, exist_ok=True)
                table = pq.read_table(source_data_path)
                if table.num_rows != length:
                    raise ValueError(f"Frame count mismatch in {source_data_path}")
                table = _replace_int64_column(table, "episode_index", [new_index] * length)
                table = _replace_int64_column(
                    table,
                    "index",
                    range(global_frame_index, global_frame_index + length),
                )
                table = _replace_int64_column(table, "task_index", [new_task_index] * length)
                pq.write_table(table, destination_data_path, compression="snappy")

                new_stats = copy.deepcopy(self.stats_by_index[old_index])
                new_stats["episode_index"] = new_index
                _set_constant_stats(new_stats["stats"]["episode_index"], new_index, length)
                _set_sequence_stats(new_stats["stats"]["index"], global_frame_index, length)
                _set_constant_stats(new_stats["stats"]["task_index"], new_task_index, length)
                exported_stats.append(new_stats)

                for video_key in self.video_keys:
                    source_video = self.video_path(old_index, video_key)
                    destination_relative = info["video_path"].format(
                        episode_chunk=new_chunk,
                        episode_index=new_index,
                        video_key=video_key,
                    )
                    mode = link_or_copy(source_video, temporary / destination_relative, copy_files=copy_files)
                    linked_files += mode == "link"
                    copied_files += mode == "copy"

                provenance.append(
                    {
                        "episode_index": new_index,
                        "source_episode_index": old_index,
                        "status": "keep",
                        "note": decisions.get(str(old_index), {}).get("note", ""),
                    }
                )
                global_frame_index += length

            write_json(temporary / "meta" / "info.json", info)
            write_jsonl(temporary / "meta" / "episodes.jsonl", exported_episodes)
            write_jsonl(temporary / "meta" / "episodes_stats.jsonl", exported_stats)
            write_jsonl(temporary / "meta" / "tasks.jsonl", task_rows)
            loaded_stats = load_episodes_stats(temporary)
            write_stats(aggregate_stats(list(loaded_stats.values())), temporary)
            write_json(
                temporary / "meta" / "curation.json",
                {
                    "version": 1,
                    "created_at": datetime.now(tz=UTC).isoformat(),
                    "source_dataset": str(self.root),
                    "episodes": provenance,
                },
            )
            temporary.rename(output)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

        return {
            "output": str(output),
            "episodes": len(selected),
            "frames": global_frame_index,
            "linked_videos": linked_files,
            "copied_videos": copied_files,
        }


def _replace_int64_column(table: pa.Table, name: str, values) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        raise KeyError(f"Missing parquet column: {name}")
    return table.set_column(index, table.schema.field(index), pa.array(values, type=pa.int64()))


def _set_constant_stats(stats: dict, value: int, count: int) -> None:
    stats.update({"min": [value], "max": [value], "mean": [float(value)], "std": [0.0], "count": [count]})


def _set_sequence_stats(stats: dict, start: int, count: int) -> None:
    end = start + count - 1
    std = math.sqrt((count**2 - 1) / 12) if count > 1 else 0.0
    stats.update(
        {
            "min": [start],
            "max": [end],
            "mean": [(start + end) / 2],
            "std": [std],
            "count": [count],
        }
    )


class CuratorHandler(BaseHTTPRequestHandler):
    catalog: DatasetCatalog
    server_version = "LeRobotCurator/1.0"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/health":
                self._send_json({"ok": True})
            elif parsed.path == "/api/catalog":
                self._send_json(self.catalog.catalog_payload())
            elif parsed.path == "/api/video":
                query = parse_qs(parsed.query)
                episode_index = int(query["episode"][0])
                video_key = query["key"][0]
                self._send_file(self.catalog.video_path(episode_index, video_key), allow_range=True)
            else:
                self._send_static(parsed.path)
        except (KeyError, ValueError) as error:
            self._send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
        except FileNotFoundError as error:
            self._send_json({"error": str(error)}, status=HTTPStatus.NOT_FOUND)
        except BrokenPipeError:
            pass

    def do_HEAD(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/video":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            query = parse_qs(parsed.query)
            path = self.catalog.video_path(int(query["episode"][0]), query["key"][0])
            self._send_file(path, allow_range=True, head_only=True)
        except (KeyError, ValueError, FileNotFoundError) as error:
            self.send_error(HTTPStatus.NOT_FOUND, str(error))

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            payload = self._read_json_body()
            if parsed.path == "/api/decision":
                episode_index = int(payload["episode_index"])
                if episode_index not in self.catalog.episodes_by_index:
                    raise ValueError(f"Unknown episode: {episode_index}")
                decision = self.catalog.selection.update(
                    episode_index,
                    str(payload["status"]),
                    str(payload.get("note", "")),
                )
                self._send_json({"ok": True, "decision": decision})
            elif parsed.path == "/api/trim":
                episode_index = int(payload["episode_index"])
                if episode_index not in self.catalog.episodes_by_index:
                    raise ValueError(f"Unknown episode: {episode_index}")
                start_frame = int(payload["start_frame"])
                length = self.catalog.episodes_by_index[episode_index]["length"]
                if not 0 <= start_frame < length:
                    raise ValueError(f"start_frame must be in [0, {length}) for episode {episode_index}")
                decision = self.catalog.selection.set_trim(episode_index, start_frame)
                self._send_json({"ok": True, "decision": decision})
            elif parsed.path == "/api/export":
                jobs = int(payload.get("jobs") or (min(32, os.cpu_count() or 8)))
                if jobs < 1:
                    raise ValueError("jobs must be >= 1")
                codec = payload.get("video_codec", "h264")
                if codec not in {"h264", "hevc", "av1"}:
                    raise ValueError("video_codec must be h264, hevc or av1")
                crf = int(payload.get("crf") if payload.get("crf") not in (None, "") else 20)
                if not 0 <= crf <= 63:
                    raise ValueError("crf must be in [0, 63]")
                result = self.catalog.export_trimmed(
                    Path(payload["output"]), jobs=jobs, video_codec=codec, crf=crf
                )
                self._send_json({"ok": True, **result})
            else:
                self._send_json({"error": "Unknown endpoint"}, status=HTTPStatus.NOT_FOUND)
        except (KeyError, ValueError, FileExistsError) as error:
            self._send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as error:
            self._send_json({"error": f"{type(error).__name__}: {error}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 1_000_000:
            raise ValueError("Invalid request body")
        return json.loads(self.rfile.read(length))

    def _send_json(self, value: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, url_path: str) -> None:
        relative = "index.html" if url_path in {"", "/"} else url_path.lstrip("/")
        path = (UI_ROOT / relative).resolve()
        if UI_ROOT.resolve() not in path.parents or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._send_file(path, allow_range=False)

    def _send_file(self, path: Path, *, allow_range: bool, head_only: bool = False) -> None:
        size = path.stat().st_size
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        start = 0
        end = size - 1
        status = HTTPStatus.OK
        range_header = self.headers.get("Range") if allow_range else None
        if range_header:
            try:
                start, end = _parse_byte_range(range_header, size)
                status = HTTPStatus.PARTIAL_CONTENT
            except ValueError:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return

        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        if allow_range:
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "private, max-age=3600")
        else:
            self.send_header("Cache-Control", "no-cache")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if head_only:
            return

        with path.open("rb") as file:
            file.seek(start)
            remaining = length
            try:
                while remaining:
                    chunk = file.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            except (ConnectionResetError, BrokenPipeError):
                # The browser aborted the video request (seek, episode switch, tab close)
                # while we were still streaming. Harmless -- stop quietly.
                return

    def log_message(self, message_format: str, *args) -> None:
        if not self.path.startswith("/api/video"):
            super().log_message(message_format, *args)


def _parse_byte_range(header: str, size: int) -> tuple[int, int]:
    if not header.startswith("bytes=") or "," in header:
        raise ValueError("Unsupported range")
    start_text, end_text = header[6:].split("-", maxsplit=1)
    if not start_text:
        suffix_length = int(end_text)
        if suffix_length <= 0:
            raise ValueError("Invalid suffix range")
        return max(0, size - suffix_length), size - 1
    start = int(start_text)
    end = min(int(end_text), size - 1) if end_text else size - 1
    if start < 0 or start >= size or end < start:
        raise ValueError("Invalid range")
    return start, end


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Local LeRobot dataset directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--state-file", type=Path, default=None)
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser automatically")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog = DatasetCatalog(args.dataset, args.state_file)
    CuratorHandler.catalog = catalog
    server = ThreadingHTTPServer((args.host, args.port), CuratorHandler)
    server.daemon_threads = True
    url = f"http://{args.host}:{args.port}"
    print(f"LeRobot Curator: {url}")
    print(f"Dataset: {catalog.root}")
    print(f"Selection state: {catalog.selection.path}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
