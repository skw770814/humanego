from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, get_origin, get_type_hints

import yaml


@dataclasses.dataclass(frozen=True)
class PathsConfig:
    raw_dir: Path
    work_dir: Path
    output_dir: Path
    assets_dir: Path
    models_dir: Path
    humanego_project: Path
    hand_swap_project: Path
    egodex_project: Path


@dataclasses.dataclass(frozen=True)
class TaskConfig:
    instruction: str = ""
    objects: tuple[dict[str, Any], ...] = ()


@dataclasses.dataclass(frozen=True)
class TimelineConfig:
    control_hz: float = 30.0
    source_rate_tolerance_hz: float = 1.0
    max_camera_gap_ms: float = 80.0
    max_tracking_gap_ms: float = 80.0
    minimum_aligned_valid_ratio: float = 0.95


@dataclasses.dataclass(frozen=True)
class CameraConfig:
    eye: str = "left"
    head_pose_source: str = "tracking"
    extrinsic_direction: str = "head_to_camera"
    require_verified_extrinsics: bool = True
    calibration_verified: bool = False
    accept_sdk_factory_extrinsics: bool = True
    minimum_hand_projection_ratio: float = 0.50
    minimum_hand_positive_depth_ratio: float = 0.95
    minimum_scan_translation_m: float = 0.08
    minimum_scan_rotation_deg: float = 12.0
    maximum_stereo_delta_ms: float = 8.0


@dataclasses.dataclass(frozen=True)
class DepthConfig:
    method: str = "stereo_sgbm"
    min_depth_m: float = 0.15
    max_depth_m: float = 4.0
    num_disparities: int = 128
    block_size: int = 7
    uniqueness_ratio: int = 10
    speckle_window_size: int = 80
    speckle_range: int = 2
    rectify_pinhole_stereo: bool = True
    left_right_consistency: bool = True
    left_right_max_diff_px: float = 1.0
    minimum_dense_coverage: float = 0.15
    minimum_epipolar_matches: int = 50
    maximum_epipolar_p95_px: float = 1.5
    debug_minimum_epipolar_matches: int = 30
    debug_maximum_epipolar_p95_px: float = 2.5
    debug_maximum_sgbm_lk_disparity_p95_px: float = 2.5
    debug_minimum_epipolar_ransac_inlier_ratio: float = 0.50
    debug_maximum_epipolar_median_px: float = 1.0
    debug_maximum_epipolar_p90_px: float = 2.0
    debug_maximum_sgbm_lk_disparity_median_px: float = 1.0
    patch_radius_px: int = 3
    minimum_keypoint_depths: int = 3


@dataclasses.dataclass(frozen=True)
class Mode2Config:
    hand_retargeter: str = "geometric"
    max_repair_gap: int = 2
    smooth_window: int = 5
    max_wrist_jump_m: float = 0.10
    max_pelvis_jump_m: float = 0.15
    max_hand_keypoint_jump_m: float = 0.08
    maximum_eef_speed_m_s: float = 2.0
    maximum_eef_rotation_deg_s: float = 720.0
    maximum_brainco_speed_s: float = 5.0
    grasp_signal_fingers: tuple[str, ...] = ("thumb_flex", "index")
    grasp_close_hi: float = 0.30
    grasp_open_lo: float = 0.30
    grasp_min_dwell_ticks: int = 3
    grasp_diagnostic_index_threshold: float = 0.25
    grasp_diagnostic_min_dwell_ticks: int = 5
    grasp_robust_low_quantile: float = 0.10
    grasp_robust_high_quantile: float = 0.95
    grasp_robust_minimum_range: float = 0.12
    grasp_robust_median_window: int = 5
    grasp_robust_close_hi: float = 0.35
    grasp_robust_open_lo: float = 0.25
    grasp_robust_confirm_ticks: int = 5
    grasp_robust_minimum_state_ticks: int = 12


@dataclasses.dataclass(frozen=True)
class PerceptionConfig:
    python: Path = Path("python")
    max_entities: int = 8
    keypoints_per_object: int = 24
    dino_box_threshold: float = 0.50
    pose_method: str = "vlm"
    triangulation_step: int = 2
    visibility_threshold: float = 0.5
    grasp_distance_m: float = 0.035
    latch_distance_m: float = 0.20
    anchor_category: str = ""
    instance_counts: dict[str, int] = dataclasses.field(default_factory=dict)
    minimum_instance_area_px: int = 400
    sam2_pose_gate_enabled: bool = True
    sam2_pose_gate_dilation_px: int = 4
    sam2_pose_gate_minimum_area_ratio: float = 0.10
    sam2_pose_gate_maximum_area_ratio: float = 4.0
    sam2_pose_gate_minimum_score: float = 0.80
    minimum_pose_inlier_ratio: float = 0.50
    maximum_object_translation_step_m: float = 0.04
    maximum_object_rotation_step_deg: float = 25.0
    object_translation_smoothing: float = 0.45
    object_translation_median_window: int = 3
    object_translation_motion_deadband_px: float = 2.0
    object_translation_full_response_px: float = 7.0
    object_rotation_smoothing: float = 0.20
    rotation_symmetry_categories: tuple[str, ...] = ("red cube", "yellow cube")
    object_prompts: dict[str, str] = dataclasses.field(default_factory=dict)
    allow_debug_fallback: bool = False


@dataclasses.dataclass(frozen=True)
class RelationsConfig:
    translation_scale_m: float = 0.50
    distance_bins_m: tuple[float, ...] = (0.06, 0.15, 0.35)
    approach_speed_m_s: float = 0.04
    contact_distance_m: float = 0.06


@dataclasses.dataclass(frozen=True)
class VisualConfig:
    enabled: bool = False
    source: str = "raw"
    python: Path = Path("python")


@dataclasses.dataclass(frozen=True)
class ExportConfig:
    variants: tuple[str, ...] = ("continuous", "binary")
    repo_id_prefix: str = "ego/relation"
    robot_type: str = "unitree_g1_brainco_revo2"
    video_codec: str = "mp4v"
    chunks_size: int = 1000


@dataclasses.dataclass(frozen=True)
class RealtimeConfig:
    camera_hz: float = 30.0
    tracker_hz: float = 20.0
    detector_hz: float = 2.0
    orientation_hz: float = 0.2
    policy_hz: float = 10.0
    control_hz: float = 30.0
    max_observation_age_ms: float = 150.0
    lost_timeout_ms: float = 500.0


@dataclasses.dataclass(frozen=True)
class VisualizationConfig:
    enabled: bool = True
    max_chart_points: int = 800
    preview_frames: int = 8
    jpeg_quality: int = 82


@dataclasses.dataclass(frozen=True)
class ProjectConfig:
    schema_version: int
    paths: PathsConfig
    task: TaskConfig
    timeline: TimelineConfig
    camera: CameraConfig
    depth: DepthConfig
    mode2: Mode2Config
    perception: PerceptionConfig
    relations: RelationsConfig
    visual: VisualConfig
    export: ExportConfig
    realtime: RealtimeConfig
    visualization: VisualizationConfig

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if dataclasses.is_dataclass(value):
                return {field.name: convert(getattr(value, field.name)) for field in dataclasses.fields(value)}
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [convert(item) for item in value]
            return value

        return convert(self)


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    # Preserve a final executable symlink such as ``.venv/bin/python``.
    # Dereferencing it to uv's base interpreter would lose the virtualenv's
    # site-packages when the resolved target is executed directly.
    return path.absolute() if path.is_absolute() else (base_dir / path).absolute()


def _section(cls, raw: dict[str, Any], *, base_dir: Path | None = None):
    # ``from __future__ import annotations`` makes ``Field.type`` a string on
    # some Python versions.  Resolve it once so paths never silently remain
    # strings (that would only fail much later in a processing step).
    annotations = get_type_hints(cls)
    unknown = set(raw) - set(annotations)
    if unknown:
        raise ValueError(f"{cls.__name__} 存在未知配置项: {sorted(unknown)}")
    values = dict(raw)
    for field in dataclasses.fields(cls):
        if field.name not in values:
            continue
        resolved_type = annotations[field.name]
        if resolved_type is Path:
            values[field.name] = (
                _resolve_path(values[field.name], base_dir)
                if base_dir is not None
                else Path(values[field.name]).expanduser()
            )
        elif get_origin(resolved_type) is tuple and isinstance(values[field.name], list):
            values[field.name] = tuple(values[field.name])
    return cls(**values)


def load_config(path: str | Path) -> ProjectConfig:
    path = Path(path).expanduser().resolve()
    project_root = path.parent.parent
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {field.name for field in dataclasses.fields(ProjectConfig)}
    unknown = set(raw) - expected
    if unknown:
        raise ValueError(f"ProjectConfig 存在未知配置段: {sorted(unknown)}")
    return ProjectConfig(
        schema_version=int(raw["schema_version"]),
        paths=_section(PathsConfig, raw["paths"], base_dir=project_root),
        task=_section(TaskConfig, raw.get("task", {})),
        timeline=_section(TimelineConfig, raw.get("timeline", {})),
        camera=_section(CameraConfig, raw.get("camera", {})),
        depth=_section(DepthConfig, raw.get("depth", {})),
        mode2=_section(Mode2Config, raw["mode2"], base_dir=project_root),
        perception=_section(PerceptionConfig, raw.get("perception", {}), base_dir=project_root),
        relations=_section(RelationsConfig, raw.get("relations", {})),
        visual=_section(VisualConfig, raw.get("visual", {}), base_dir=project_root),
        export=_section(ExportConfig, raw.get("export", {})),
        realtime=_section(RealtimeConfig, raw.get("realtime", {})),
        visualization=_section(VisualizationConfig, raw.get("visualization", {})),
    )
