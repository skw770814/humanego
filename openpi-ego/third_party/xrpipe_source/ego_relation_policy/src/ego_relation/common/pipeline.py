from __future__ import annotations

from pathlib import Path
import json

from ego_relation.config import ProjectConfig


def find_episodes(cfg: ProjectConfig, values: list[str] | None) -> list[Path]:
    if not values:
        episodes = sorted(cfg.paths.raw_dir.glob("episode_*.hdf5"), key=lambda path: int(path.stem.rsplit("_", 1)[-1]))
    else:
        episodes = []
        for value in values:
            path = Path(value).expanduser()
            if not path.is_absolute():
                candidate = cfg.paths.raw_dir / value
                path = candidate if candidate.exists() else path
            if path.is_dir():
                episodes.extend(sorted(path.glob("episode_*.hdf5")))
            else:
                episodes.append(path)
    missing = [str(path) for path in episodes if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"找不到 episode: {missing}")
    if not episodes:
        raise FileNotFoundError(f"{cfg.paths.raw_dir} 下没有 episode_*.hdf5")
    return [path.resolve() for path in episodes]


def episode_work_dir(cfg: ProjectConfig, source: Path) -> Path:
    return (cfg.paths.work_dir / source.stem).resolve()


def filter_motion_accepted(cfg: ProjectConfig, sources: list[Path], *, stage: str) -> list[Path]:
    accepted = []
    for source in sources:
        manifest_path = episode_work_dir(cfg, source) / "mode2.manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"{source.stem} 尚未完成 Step1")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metrics = manifest.get("metrics", {})
        deployable = metrics.get("step1_deployable", metrics.get("motion_deployable"))
        if deployable is False:
            print(f"[{stage}] 隔离 {source.stem}: Step1 QA 未通过（见 qa/mode2_quarantine.json）")
            continue
        accepted.append(source)
    if not accepted:
        raise RuntimeError(f"{stage} 没有通过 Mode2 motion QA 的 episode")
    return accepted
