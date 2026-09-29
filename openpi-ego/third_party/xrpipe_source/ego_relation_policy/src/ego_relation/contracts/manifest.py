from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True)
class StageManifest:
    schema_version: int
    stage: str
    episode: str
    source_path: str
    source_sha256: str
    config: dict[str, Any]
    outputs: dict[str, str]
    metrics: dict[str, Any] = dataclasses.field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def write(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dataclasses.asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def read(cls, path: str | Path) -> "StageManifest":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        raw["warnings"] = tuple(raw.get("warnings", ()))
        return cls(**raw)
