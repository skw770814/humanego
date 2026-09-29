from __future__ import annotations

from pathlib import Path

from ego_relation.config import ProjectConfig
from ego_relation.common.pipeline import episode_work_dir
from ego_relation.common.pipeline import filter_motion_accepted
from ego_relation.s4_lerobot_export.lerobot import export_variants
from ego_relation.visualization.reports import generate_index
from ego_relation.visualization.reports import generate_step4_report


def run(
    cfg: ProjectConfig,
    sources: list[Path],
    variants: tuple[str, ...] | None = None,
    *,
    force: bool = False,
) -> dict:
    requested_sources = list(sources)
    sources = filter_motion_accepted(cfg, sources, stage="Step4")
    episode_dirs = [episode_work_dir(cfg, source) for source in sources]
    for source, episode_dir in zip(sources, episode_dirs, strict=True):
        if not (episode_dir / "relations.manifest.json").is_file():
            raise FileNotFoundError(f"{source.stem} 尚未完成 Step2")
        if cfg.visual.source == "brainco_swap" and not (episode_dir / "hand_swap.manifest.json").is_file():
            raise FileNotFoundError(f"{source.stem} 要导出 brainco_swap，但尚未完成 Step3")
    result = export_variants(cfg, sources, episode_dirs, variants, force=force)
    if cfg.visualization.enabled:
        safe_prefix = cfg.export.repo_id_prefix.replace("/", "_")
        for variant in result:
            generate_step4_report(cfg, cfg.paths.output_dir / f"{safe_prefix}_{variant}")
        generate_index(cfg, requested_sources)
    return result
