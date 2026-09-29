from __future__ import annotations

from pathlib import Path

from ego_relation.config import ProjectConfig
from ego_relation.common.pipeline import episode_work_dir
from ego_relation.s3_brainco_visual.hand_swap import run_hand_swap
from ego_relation.visualization.reports import generate_index
from ego_relation.visualization.reports import generate_step3_report


def run(cfg: ProjectConfig, sources: list[Path]) -> list[Path]:
    output = []
    for source in sources:
        episode_dir = episode_work_dir(cfg, source)
        if not (episode_dir / "relations.manifest.json").is_file():
            raise FileNotFoundError(f"{source.stem} 尚未完成 Step2")
        run_hand_swap(cfg, source, episode_dir)
        if cfg.visualization.enabled:
            generate_step3_report(cfg, source, episode_dir)
        output.append(episode_dir)
    if cfg.visualization.enabled:
        generate_index(cfg, sources)
    return output
