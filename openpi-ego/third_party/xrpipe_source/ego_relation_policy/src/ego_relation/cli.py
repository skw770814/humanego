from __future__ import annotations

import argparse
import json
from pathlib import Path

from ego_relation.config import load_config
from ego_relation.preflight import run_preflight
from ego_relation.common.pipeline import find_episodes
from ego_relation.s1_pico_mode2 import pipeline as step1_prepare
from ego_relation.s2_object_relations import pipeline as step2_relations
from ego_relation.s3_brainco_visual import pipeline as step3_visual
from ego_relation.s4_lerobot_export import pipeline as step4_export
from ego_relation.visualization.reports import generate_available_reports
from ego_relation.visualization.server import serve_reports


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PICO ego relation Step1--Step4 数据工程")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("step1", "step2", "step3", "step4", "all", "check", "report", "serve-reports"):
        command = subparsers.add_parser(name)
        if name != "serve-reports":
            command.add_argument("--episodes", nargs="*")
        if name in {"step1", "step4", "all"}:
            command.add_argument("--force", action="store_true")
        if name in {"step1", "all"}:
            command.add_argument(
                "--g1-replay",
                action="store_true",
                help="额外生成平滑轨迹的 G1 ready-delta MuJoCo 回放",
            )
            command.add_argument(
                "--skip-step1-videos",
                action="store_true",
                help="保留像素关键点数据，但不生成 Step1 QA 视频",
            )
        if name in {"step2", "all"}:
            command.add_argument("--allow-unverified-calibration", action="store_true")
        if name in {"step4", "all"}:
            command.add_argument("--variants", nargs="*", choices=sorted({"continuous", "binary"}))
        if name == "check":
            command.add_argument(
                "--scope",
                choices=("core", "perception", "visual", "pipeline", "openpi", "all"),
                default="all",
            )
            command.add_argument("--openpi-project", type=Path)
        if name == "report":
            command.add_argument(
                "--steps",
                nargs="*",
                choices=("step1", "step2", "step3", "step4"),
                default=("step1", "step2", "step3", "step4"),
            )
        if name == "serve-reports":
            command.add_argument("--host", default="127.0.0.1")
            command.add_argument("--port", type=int, default=8765)
            command.add_argument("--open", action="store_true", help="启动本地 HTTP 服务后尝试打开浏览器")
    return parser


def main() -> None:
    args = _parser().parse_args()
    cfg = load_config(args.config)
    if args.command == "serve-reports":
        serve_reports(cfg, host=args.host, port=args.port, open_browser=args.open)
        return

    sources = find_episodes(cfg, args.episodes)
    if args.command == "check":
        report = run_preflight(cfg, sources, scope=args.scope, openpi_project=args.openpi_project)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["ok"]:
            raise SystemExit(1)
    elif args.command == "report":
        generated = generate_available_reports(cfg, sources, tuple(args.steps))
        serializable = {
            key: [str(path) for path in value] if isinstance(value, list) else str(value)
            for key, value in generated.items()
        }
        print(json.dumps(serializable, ensure_ascii=False, indent=2))
    elif args.command == "step1":
        step1_prepare.run(
            cfg,
            sources,
            force=args.force,
            render_videos=not args.skip_step1_videos,
            g1_replay=args.g1_replay,
        )
    elif args.command == "step2":
        step2_relations.run(
            cfg, sources, allow_unverified_calibration=args.allow_unverified_calibration
        )
    elif args.command == "step3":
        step3_visual.run(cfg, sources)
    elif args.command == "step4":
        step4_export.run(cfg, sources, tuple(args.variants) if args.variants else None, force=args.force)
    elif args.command == "all":
        step1_prepare.run(
            cfg,
            sources,
            force=args.force,
            render_videos=not args.skip_step1_videos,
            g1_replay=args.g1_replay,
        )
        step2_relations.run(
            cfg, sources, allow_unverified_calibration=args.allow_unverified_calibration
        )
        if cfg.visual.enabled or cfg.visual.source == "brainco_swap":
            step3_visual.run(cfg, sources)
        step4_export.run(cfg, sources, tuple(args.variants) if args.variants else None, force=args.force)


if __name__ == "__main__":
    main()
