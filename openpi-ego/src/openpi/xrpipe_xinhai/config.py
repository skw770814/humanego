from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import sys

import numpy as np

from .geometry import se3

ROOT = Path(__file__).resolve().parents[3]
STATE_TRANSFORM = "per relation pose T_midpoint_object: S @ T @ S, S=diag(1,1,-1,1); current gripper is unchanged"
ADAPTER = "xrpipe_xinhai_source_frame_v1"


def pipeline_imports():
    source = ROOT / "third_party/xrpipe_source"
    for path in (source, source / "test/pipeline", source / "ego_relation_policy/src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def contract(path):
    path = Path(path)
    if path.is_dir():
        path = path / "assets/runtime_manifest.json"
    m = read_json(path)
    validate_contract(m)
    return m


def validate_contract(m):
    if not m.get("observation_only"):
        mode = m.get("mode", {})
        for key, value in {"name": "xrpipe_mode1", "camera_mode": "single", "model_uses_state": True,
                           "action_representation": "relative", "relative_norm": "shared"}.items():
            if mode.get(key) != value:
                raise ValueError(f"Unsupported training mode: {key}={mode.get(key)!r}")
        if m.get("training_config") != "xrpipe_mode1_train":
            raise ValueError("Checkpoint is not the existing XRPipe Mode1 training preset")
    expected = dict(schema_version=1, preset="xrpipe_mode1_rel_shared", state_dim=19,
                    action_dim=10, reference_dim=9, action_horizon=50, dataset_fps=30,
                    model_action_dim=32, normalization="shared_quantile", normalization_clip=5.)
    for key, value in expected.items():
        if m.get(key) != value:
            raise ValueError(f"Unsupported contract {key}: {m.get(key)!r}; expected {value!r}")
    c = m["dataset_contract"]
    for key, value in {
        "schema_version": "xrpipe_action_v1", "state_dim": 19, "reference_dim": 9,
        "action_dim": 10, "object_order": ["obj1", "obj2"],
        "coordinate_system": "openxr_rh_x_right_y_up_z_back",
        "source_coordinate_system": "unity_lh_x_right_y_up_z_forward",
        "reference_frame": "pico_world_openxr", "stored_action": "absolute_next_target",
        "control_point": "right_thumb_index_fingertip_midpoint",
        "relative_formula": "inv(reference[t]) @ action[t+k]",
        "training_state_transform": STATE_TRANSFORM,
    }.items():
        if c.get(key) != value:
            raise ValueError(f"Dataset contract mismatch: {key}={c.get(key)!r}")
    if len(c.get("object_categories", [])) != 2 or not c.get("digest") or not m.get("asset_id"):
        raise ValueError("Missing object categories, digest or normalization asset_id")


@dataclasses.dataclass
class ObservationArgs:
    contract: str  # checkpoint directory OR exported assets/runtime_manifest.json
    object_prompts: tuple[str, ...] = ()
    robot_ip: str = "172.16.0.30"
    robot_port: int = 4242
    extrinsics: str | None = str(ROOT / "extrinsics.json")
    camera_config: str = str(ROOT / "configs/xrpipe_xinhai/d405.json")
    tcp_geometry: str = str(ROOT / "configs/xrpipe_xinhai/right_gripper_train_aligned.json")
    perception_device: str = "cuda"
    dino_checkpoint: str = str(ROOT / "models/xrpipe_xinhai/grounding-dino-tiny")
    sam2_checkpoint: str = str(ROOT / "models/xrpipe_xinhai/sam2_hiera_tiny.pt")
    sam2_config: str = "configs/sam2/sam2_hiera_t.yaml"
    box_threshold: float = .3
    capture_timeout: float = 8.
    max_observation_age: float = 5.
    max_tcp_bracket_translation: float = .005
    max_tcp_bracket_rotation_deg: float = 2.
    output_dir: str = "outputs/xrpipe_xinhai"


def prompts_for(args, manifest):
    prompts = args.object_prompts or tuple(manifest["dataset_contract"]["object_categories"])
    if len(prompts) != 2 or any(not p.strip() for p in prompts):
        raise ValueError("Exactly two nonempty object prompts required, in obj1/obj2 contract order")
    return tuple(prompts)


def load_geometry(args):
    for name in ("capture_timeout", "max_observation_age", "max_tcp_bracket_translation",
                 "max_tcp_bracket_rotation_deg"):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(args.box_threshold) or not 0 < args.box_threshold < 1:
        raise ValueError("box_threshold must be inside (0,1)")
    g = read_json(args.tcp_geometry)
    if g.get("units") != "m" or g.get("axis_convention") != "source_five_keypoint_analogue":
        raise ValueError("Unsupported TCP units or axis convention")
    if g.get("ee_frame") != "right_gripper_link":
        raise ValueError("TCP profile must refer to the original right_gripper_link")
    tem = se3(g["T_E_M"])
    e = read_json(args.extrinsics) if args.extrinsics else None
    if e is not None:
        if (e.get("units") != "m" or not e.get("control_frame") or not e.get("camera_frame")
                or e.get("T_B_C") is None):
            raise ValueError("Extrinsics require calibrated T_B_C, units=m and both frame names")
        se3(e["T_B_C"])
    return g, tem, e

