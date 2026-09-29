from __future__ import annotations

from pathlib import Path

from ego_relation.config import ProjectConfig
from ego_relation.common.pipeline import episode_work_dir
from ego_relation.common.pipeline import filter_motion_accepted
from ego_relation.s2_object_relations.encoding import process_relations
from ego_relation.s2_object_relations.humanego import run_humanego
from ego_relation.s2_object_relations.stereo_depth import run_stereo_depth
from ego_relation.visualization.reports import generate_index
from ego_relation.visualization.reports import generate_step2_report


def run(
    cfg: ProjectConfig,
    sources: list[Path],
    *,
    allow_unverified_calibration: bool = False,
) -> list[Path]:
    requested_sources = list(sources)
    sources = filter_motion_accepted(cfg, sources, stage="Step2")
    output = []
    for source in sources:
        episode_dir = episode_work_dir(cfg, source)
        if not (episode_dir / "mode2.manifest.json").is_file():
            raise FileNotFoundError(f"{source.stem} 尚未完成 Step1")
        run_stereo_depth(cfg, source, episode_dir)
        run_humanego(
            cfg,
            source,
            episode_dir,
            allow_unverified_calibration=allow_unverified_calibration,
        )
        process_relations(cfg, source, episode_dir)
        if cfg.visualization.enabled:
            generate_step2_report(cfg, source, episode_dir)
        output.append(episode_dir)
    if cfg.visualization.enabled:
        generate_index(cfg, requested_sources)
    return output
