from __future__ import annotations

import json
from pathlib import Path

import h5py

from ego_relation.config import ProjectConfig


def _task_instruction(cfg: ProjectConfig, source: Path) -> tuple[str, str]:
    configured = cfg.task.instruction.strip()
    if configured:
        return configured, "config.task.instruction"
    with h5py.File(source, "r") as file:
        recorded = str(file.attrs.get("task_instruction", "")).strip()
    if not recorded:
        raise ValueError("未配置 task.instruction，HDF5 也没有 task_instruction")
    return recorded, "hdf5.task_instruction"


def _object_instances(cfg: ProjectConfig) -> tuple[list[dict], str]:
    if cfg.task.objects:
        rows = [dict(row) for row in cfg.task.objects]
        source = "config.task.objects"
    else:
        rows = []
        for category, prompt in cfg.perception.object_prompts.items():
            count = int(cfg.perception.instance_counts.get(category, 1))
            for _ in range(count):
                rows.append({"category": category, "prompt": prompt})
        source = "config.perception compatibility fallback"
    if not rows:
        raise ValueError("未配置 task.objects，也没有可兼容的 perception.object_prompts")

    instance_ids: set[str] = set()
    result = []
    for index, row in enumerate(rows, start=1):
        instance_id = str(row.get("instance_id", f"obj{index}")).strip()
        category = str(row.get("category", "")).strip(" ,")
        prompt = str(row.get("prompt", "")).strip()
        if not instance_id or instance_id in instance_ids:
            raise ValueError(f"task.objects instance_id 必须非空且唯一，实际 {instance_id!r}")
        if not category or not prompt:
            raise ValueError(f"{instance_id} 必须配置 category 和 prompt")
        if not prompt.endswith("."):
            prompt += " ."
        instance_ids.add(instance_id)
        result.append(
            {
                "instance_id": instance_id,
                "category": category,
                "prompt": prompt,
                "is_anchor": False,
                "graspable": bool(row.get("graspable", True)),
                "expected_instances": 1,
            }
        )
    return result, source


def write_step1_contracts(cfg: ProjectConfig, source: str | Path, episode_dir: str | Path) -> dict[str, Path]:
    source = Path(source).resolve()
    output_dir = Path(episode_dir).resolve() / "step1"
    output_dir.mkdir(parents=True, exist_ok=True)

    instruction, instruction_source = _task_instruction(cfg, source)
    objects, object_source = _object_instances(cfg)
    task_path = output_dir / "task_semantics.json"
    objects_path = output_dir / "object_instances.json"
    action_path = output_dir / "action_contract.json"

    task_path.write_text(
        json.dumps(
            {
                "schema_version": cfg.schema_version,
                "episode": source.stem,
                "instruction": instruction,
                "source": instruction_source,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    objects_path.write_text(
        json.dumps(
            {
                "schema_version": cfg.schema_version,
                "episode": source.stem,
                "source": object_source,
                "pose_reference_frame": "camera0_static_when_estimated_in_step2",
                "instances": objects,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    action_path.write_text(
        json.dumps(
            {
                "schema_version": cfg.schema_version,
                "state_file": "mode2/state_abs.npy",
                "action_file": "mode2/action_abs.npy",
                "dimensions": 30,
                "action_semantics": "action[t] = state_abs[min(t+1, T-1)]",
                "tcp_pose_semantics": "absolute TCP target pose expressed in the G1 robot-base frame",
                "pose_transform": "T_g1_base_tcp",
                "capture_reference_frame": "static frame-0 pelvis origin/yaw with PICO world gravity",
                "dynamic_pelvis_compensation": False,
                "base_frame_origin": "static PICO world reference mapped into the G1 robot-base convention",
                "execution_default": "start from G1 ready TCP poses and apply demo-local SE(3) increments",
                "execution_formula": (
                    "delta_T[t] = inv(T_demo_state[t]) @ T_demo_action[t]; "
                    "T_g1[t+1] = T_g1[t] @ delta_T[t]"
                ),
                "pose_encoding": "[tx, ty, tz, R[:,0], R[:,1]]",
                "translation_unit": "meter",
                "layout": [
                    {"slice": [0, 9], "name": "left_tcp_pose_abs", "frame": "g1_robot_base"},
                    {"slice": [9, 18], "name": "right_tcp_pose_abs", "frame": "g1_robot_base"},
                    {"slice": [18, 24], "name": "left_revo2_command", "range": [0.0, 1.0]},
                    {"slice": [24, 30], "name": "right_revo2_command", "range": [0.0, 1.0]},
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"task_semantics": task_path, "object_instances": objects_path, "action_contract": action_path}
