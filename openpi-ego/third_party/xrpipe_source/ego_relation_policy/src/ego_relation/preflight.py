"""Fail-fast checks for the PICO/HumanEgo/visual pipeline environments."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from ego_relation.config import ProjectConfig
from ego_relation.common.pipeline import episode_work_dir


def _command_path(value: str | Path) -> str | None:
    path = Path(value).expanduser()
    if path.is_file() and path.stat().st_mode & 0o111:
        return str(path)
    return shutil.which(str(value))


def _probe_python(executable: str, imports: tuple[str, ...], *, pythonpath: Path | None = None) -> dict[str, Any]:
    resolved = _command_path(executable)
    result: dict[str, Any] = {"requested": str(executable), "resolved": resolved, "imports": list(imports)}
    if resolved is None:
        result["ok"] = False
        result["error"] = "interpreter not found"
        return result
    code = "import " + ", ".join(imports)
    environment = None
    if pythonpath is not None:
        import os

        environment = {**os.environ, "PYTHONPATH": str(pythonpath)}
    process = subprocess.run(
        [resolved, "-c", code],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    result["ok"] = process.returncode == 0
    if process.returncode:
        result["stderr"] = process.stderr[-2000:]
    return result


def run_preflight(
    cfg: ProjectConfig,
    sources: list[Path],
    *,
    scope: str = "all",
    openpi_project: Path | None = None,
) -> dict[str, Any]:
    if scope not in {"core", "perception", "visual", "openpi", "pipeline", "all"}:
        raise ValueError(f"unsupported preflight scope: {scope}")
    report: dict[str, Any] = {"scope": scope, "episodes": [str(path) for path in sources]}
    pipeline_scope = scope in {"pipeline", "all"}
    if scope in {"core"} or pipeline_scope:
        core: dict[str, Any] = {
            "mode2_interpreter": _probe_python(sys.executable, ("numpy", "scipy", "h5py")),
            "mode2_engine": str(Path(__file__).with_name("s1_pico_mode2") / "native_mode2.py"),
            "revo2_assets": str(cfg.paths.assets_dir / "revo2"),
            "episodes": [],
        }
        for source in sources:
            episode_dir = episode_work_dir(cfg, source)
            entry: dict[str, Any] = {"episode": source.stem, "work_dir": str(episode_dir)}
            pico_report = episode_dir / "qa" / "pico_report.json"
            mode2_manifest = episode_dir / "mode2.manifest.json"
            if pico_report.is_file():
                entry["pico"] = json.loads(pico_report.read_text(encoding="utf-8"))
                entry["pico_geometry_deployable"] = bool(entry["pico"].get("geometry_deployable"))
            else:
                entry["pico_error"] = "run Step1 first"
            if mode2_manifest.is_file():
                mode2 = json.loads(mode2_manifest.read_text(encoding="utf-8"))
                entry["mode2_motion_deployable"] = mode2.get("metrics", {}).get("motion_deployable")
                entry["mode2_quarantined"] = mode2.get("metrics", {}).get("motion_deployable") is False
            else:
                entry["mode2_error"] = "run Step1 first"
            core["episodes"].append(entry)
        core["ok"] = bool(
            core["mode2_interpreter"]["ok"]
            and Path(core["mode2_engine"]).is_file()
            and Path(core["revo2_assets"]).is_dir()
            and any(item.get("mode2_motion_deployable") is True for item in core["episodes"])
        )
        report["core"] = core

    if scope in {"perception"} or pipeline_scope:
        imports = (
            "numpy",
            "cv2",
            "torch",
            "transformers",
            "huggingface_hub",
            "cotracker",
            "PIL",
            "sam2",
            "open3d",
        )
        if cfg.perception.pose_method == "vlm":
            imports += ("orient_anything",)
        perception = {
            "python": _probe_python(str(cfg.perception.python), imports, pythonpath=cfg.paths.humanego_project),
            "humanego_root": str(cfg.paths.humanego_project),
            "worker": str(Path(__file__).with_name("s2_object_relations") / "humanego_worker.py"),
            "model_cache": str(cfg.paths.models_dir / "huggingface"),
        }
        perception["ok"] = bool(
            perception["python"]["ok"]
            and Path(perception["humanego_root"]).is_dir()
            and Path(perception["worker"]).is_file()
        )
        report["perception"] = perception

    if scope in {"visual"} or pipeline_scope:
        visual_tool_imports = ("numpy", "cv2", "h5py", "pyarrow", "imageio", "mediapy")
        visual_render_imports = ("sapien", "dex_retargeting", "pytransform3d")
        visual_mask_imports = ("torch", "sam2")
        visual = {
            "python": str(cfg.visual.python),
            "tool": _probe_python(str(cfg.visual.python), visual_tool_imports),
            "render": _probe_python(str(cfg.visual.python), visual_render_imports),
            "mask": _probe_python(str(cfg.visual.python), visual_mask_imports),
            "hand_swap_root": str(cfg.paths.hand_swap_project),
            "assets_dir": str(cfg.paths.assets_dir),
            "e2fgvi_runtime": str(Path(__file__).resolve().parents[2] / "third_party/e2fgvi_runtime/E2FGVI"),
            "e2fgvi_checkpoint": str(cfg.paths.models_dir / "e2fgvi/E2FGVI-HQ-CVPR22.pth"),
        }
        visual["ok"] = bool(
            visual["tool"]["ok"]
            and visual["render"]["ok"]
            and visual["mask"]["ok"]
            and Path(visual["hand_swap_root"]).is_dir()
            and Path(visual["assets_dir"]).is_dir()
            and Path(visual["e2fgvi_runtime"]).is_dir()
            and Path(visual["e2fgvi_checkpoint"]).is_file()
        )
        report["visual"] = visual

    if scope in {"openpi"}:
        project = (openpi_project or (Path(__file__).resolve().parents[3] / "openpi")).resolve()
        python_candidate = project / ".venv/bin/python"
        # The checked-out OpenPI sources and its large ML environment are often
        # kept in separate directories.  Prefer an explicit override, then the
        # conventional sibling environment used by this workspace.
        if not python_candidate.is_file():
            override = os.environ.get("OPENPI_PYTHON")
            sibling = project.parents[2] / "openpi/.venv/bin/python"
            for candidate in (Path(override).expanduser() if override else None, sibling):
                if candidate is not None and candidate.is_file():
                    python_candidate = candidate
                    break
        openpi = {
            "project": str(project),
            "python": _probe_python(str(python_candidate), ("jax", "flax", "torch", "lerobot")),
            "train_script": str(project / "scripts/train_ego_relation.py"),
            "serve_script": str(project / "scripts/serve_ego_relation.py"),
        }
        openpi["ok"] = bool(
            openpi["python"]["ok"]
            and Path(openpi["train_script"]).is_file()
            and Path(openpi["serve_script"]).is_file()
        )
        report["openpi"] = openpi
    report["ok"] = all(
        bool(report[key]["ok"]) for key in report if key in {"core", "perception", "visual", "openpi"}
    )
    return report
