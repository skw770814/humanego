from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq

from ego_relation.config import ProjectConfig
from ego_relation.contracts.manifest import StageManifest, file_sha256
from ego_relation.contracts.se3 import compose, invert, transform_to_vec9
from ego_relation.s1_pico_mode2.pico import PicoEpisode

TYPE_PAD = 0.0
TYPE_HAND_LEFT = 1.0
TYPE_HAND_RIGHT = 2.0
TYPE_OBJECT_ANCHOR = 3.0
TYPE_OBJECT_OTHER = 4.0
RELATION_DIM = 29


def classify_initial_layout(
    instance_ids: np.ndarray,
    categories: np.ndarray,
    initial_poses: np.ndarray,
    left_wrist: np.ndarray,
    right_wrist: np.ndarray,
    *,
    ambiguity_margin_m: float = 0.02,
) -> dict:
    """Classify the initial red/yellow arrangement along static camera0 x."""
    ids = np.asarray(instance_ids).astype(str)
    labels = np.char.lower(np.asarray(categories).astype(str))
    poses = np.asarray(initial_poses, dtype=np.float64)

    def find_object(*terms: str) -> int | None:
        matches = [i for i, label in enumerate(labels) if all(term in label for term in terms)]
        return matches[0] if len(matches) == 1 else None

    red_index = find_object("red", "cube")
    yellow_index = find_object("yellow", "cube")
    holder_index = find_object("pen", "holder")
    result: dict = {
        "layout_class": "unclassified",
        "reference_frame": "camera0_static_frame",
        "classification_axis": "camera0_x",
        "ambiguity_margin_m": float(ambiguity_margin_m),
    }
    if red_index is None or yellow_index is None:
        result["reason"] = "catalog must contain one red cube and one yellow cube"
        return result

    red_position = poses[red_index, :3, 3]
    yellow_position = poses[yellow_index, :3, 3]
    red_minus_yellow_x = float(red_position[0] - yellow_position[0])
    if abs(red_minus_yellow_x) <= ambiguity_margin_m:
        layout_class = "ambiguous_red_yellow_side"
    elif red_minus_yellow_x < 0:
        layout_class = "red_left_yellow_right"
    else:
        layout_class = "red_right_yellow_left"

    wrists = np.stack([left_wrist[:3, 3], right_wrist[:3, 3]])
    wrist_names = ("left_wrist", "right_wrist")

    def nearest_wrist(position: np.ndarray) -> tuple[str, list[float]]:
        distances = np.linalg.norm(wrists - position, axis=1)
        return wrist_names[int(np.argmin(distances))], [float(value) for value in distances]

    red_nearest, red_distances = nearest_wrist(red_position)
    yellow_nearest, yellow_distances = nearest_wrist(yellow_position)
    result.update(
        {
            "layout_class": layout_class,
            "red_instance_id": str(ids[red_index]),
            "yellow_instance_id": str(ids[yellow_index]),
            "red_x_m": float(red_position[0]),
            "yellow_x_m": float(yellow_position[0]),
            "red_minus_yellow_x_m": red_minus_yellow_x,
            "red_nearest_initial_wrist": red_nearest,
            "yellow_nearest_initial_wrist": yellow_nearest,
            "red_to_initial_wrists_m": red_distances,
            "yellow_to_initial_wrists_m": yellow_distances,
        }
    )
    if holder_index is not None:
        holder_x = float(poses[holder_index, 0, 3])
        low_x, high_x = sorted((float(red_position[0]), float(yellow_position[0])))
        result.update(
            {
                "holder_instance_id": str(ids[holder_index]),
                "holder_x_m": holder_x,
                "holder_between_cubes": bool(low_x <= holder_x <= high_x),
            }
        )
    return result


def _grasp_hysteresis(pinch: np.ndarray, close_threshold: float) -> np.ndarray:
    open_threshold = close_threshold * 1.35
    result = np.zeros(len(pinch), dtype=bool)
    active = False
    for index, distance in enumerate(pinch):
        if active and distance > open_threshold:
            active = False
        elif not active and distance < close_threshold:
            active = True
        result[index] = active
    return result


def _align_step1_grasp(
    episode_dir: Path,
    timestamps_ns: np.ndarray,
    camera_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str] | None:
    """Load the recommended Step1 grasp state on the Step2 relation timeline."""
    path = episode_dir / "mode2" / "brainco_grasp_binary.npz"
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as archive:
        left = archive["closed_left"].astype(bool)
        right = archive["closed_right"].astype(bool)
        source_ticks = archive["ticks_ns"].astype(np.int64)
        source_camera = archive["camera_match"].astype(np.int64)
    if not (len(left) == len(right) == len(source_ticks) == len(source_camera)):
        raise ValueError(f"Step1 grasp arrays do not share one timeline: {path}")
    if len(left) == len(timestamps_ns) and np.array_equal(source_camera, camera_indices):
        return left, right, "step1_brainco_robust_exact"
    if not len(source_ticks):
        raise ValueError(f"Step1 grasp timeline is empty: {path}")
    insertion = np.searchsorted(source_ticks, timestamps_ns)
    upper = np.clip(insertion, 0, len(source_ticks) - 1)
    lower = np.clip(insertion - 1, 0, len(source_ticks) - 1)
    choose_upper = np.abs(source_ticks[upper] - timestamps_ns) < np.abs(
        source_ticks[lower] - timestamps_ns
    )
    nearest = np.where(choose_upper, upper, lower)
    return left[nearest], right[nearest], "step1_brainco_robust_nearest_timestamp"


def _graspable_object_mask(
    cfg: ProjectConfig,
    instance_ids: np.ndarray,
    categories: np.ndarray,
    is_anchor: np.ndarray,
) -> np.ndarray:
    """Keep coordinate anchoring separate from the task's grasp affordance."""
    specs_by_id = {str(row.get("instance_id", "")): row for row in cfg.task.objects}
    specs_by_category = {
        str(row.get("category", "")).strip(" ,").lower(): row for row in cfg.task.objects
    }
    result = np.empty(len(instance_ids), dtype=bool)
    for index, (instance_id, category) in enumerate(
        zip(instance_ids.astype(str), categories.astype(str), strict=True)
    ):
        spec = specs_by_id.get(instance_id)
        if spec is None:
            spec = specs_by_category.get(category.strip(" ,").lower())
        result[index] = bool(spec.get("graspable", not is_anchor[index])) if spec else not bool(
            is_anchor[index]
        )
    return result


def latch_object_poses(
    initial_poses: np.ndarray,
    left_hand: np.ndarray,
    right_hand: np.ndarray,
    left_grasp: np.ndarray,
    right_grasp: np.ndarray,
    *,
    latch_distance_m: float,
    visual_poses: np.ndarray | None = None,
    visual_valid: np.ndarray | None = None,
    eligible_objects: np.ndarray | None = None,
    left_valid: np.ndarray | None = None,
    right_valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Latch objects to TCP with a fixed contact transform until release."""
    initial = np.asarray(initial_poses, dtype=np.float64)
    current = initial.copy()
    output = np.empty((len(left_hand), len(initial), 4, 4), dtype=np.float64)
    dynamic = np.zeros((len(left_hand), len(initial)), dtype=bool)
    owner = np.full((len(left_hand), len(initial)), -1, dtype=np.int8)
    eligible = (
        np.ones(len(initial), dtype=bool)
        if eligible_objects is None
        else np.asarray(eligible_objects, dtype=bool)
    )
    if eligible.shape != (len(initial),):
        raise ValueError(f"eligible_objects must have shape {(len(initial),)}, got {eligible.shape}")
    hand_valid_sequences = {
        0: np.ones(len(left_hand), dtype=bool)
        if left_valid is None
        else np.asarray(left_valid, dtype=bool),
        1: np.ones(len(right_hand), dtype=bool)
        if right_valid is None
        else np.asarray(right_valid, dtype=bool),
    }
    states = {
        0: {"object": None, "T_tcp_object": None},
        1: {"object": None, "T_tcp_object": None},
    }
    hand_sequences = {0: left_hand, 1: right_hand}
    grasp_sequences = {0: left_grasp, 1: right_grasp}
    for frame in range(len(left_hand)):
        released: set[int] = set()
        for hand_id in (0, 1):
            state = states[hand_id]
            if not grasp_sequences[hand_id][frame] and state["object"] is not None:
                released.add(int(state["object"]))
                state["object"] = None
                state["T_tcp_object"] = None
        claimed: set[int] = {
            int(state["object"]) for state in states.values() if state["object"] is not None
        }
        if visual_poses is not None:
            validity = np.ones(len(current), dtype=bool) if visual_valid is None else visual_valid[frame]
            for object_index in range(len(current)):
                if (
                    object_index not in claimed
                    and object_index not in released
                    and validity[object_index]
                ):
                    current[object_index] = visual_poses[frame, object_index]
        for hand_id in (0, 1):
            state = states[hand_id]
            grasp = bool(grasp_sequences[hand_id][frame])
            hand_pose = hand_sequences[hand_id][frame]
            hand_valid = bool(hand_valid_sequences[hand_id][frame])
            # A close event may precede physical contact. Keep looking while
            # closed so one early frame cannot permanently miss the latch.
            if grasp and state["object"] is None and hand_valid:
                distances = np.linalg.norm(current[:, :3, 3] - hand_pose[:3, 3], axis=1)
                order = np.argsort(distances)
                candidate = next(
                    (
                        int(index)
                        for index in order
                        if eligible[int(index)] and int(index) not in claimed
                    ),
                    None,
                )
                if candidate is not None and distances[candidate] <= latch_distance_m:
                    state["object"] = candidate
                    state["T_tcp_object"] = compose(invert(hand_pose), current[candidate])
            if grasp and state["object"] is not None:
                object_index = int(state["object"])
                if hand_valid:
                    current[object_index] = compose(hand_pose, state["T_tcp_object"])
                dynamic[frame, object_index] = True
                owner[frame, object_index] = hand_id
                claimed.add(object_index)
        output[frame] = current
    return output, dynamic, owner


def build_relation_tokens(
    left_hand: np.ndarray,
    right_hand: np.ndarray,
    object_poses: np.ndarray,
    is_anchor: np.ndarray,
    left_grasp: bool,
    right_grasp: bool,
    object_dynamic: np.ndarray,
    *,
    max_entities: int,
    translation_scale_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    tokens: list[np.ndarray] = []

    def token(type_id: float, entity: np.ndarray, flag: float) -> np.ndarray:
        return np.concatenate(
            [
                [type_id],
                transform_to_vec9(entity, translation_scale_m),
                transform_to_vec9(left_hand, translation_scale_m),
                transform_to_vec9(right_hand, translation_scale_m),
                [flag],
            ]
        ).astype(np.float32)

    tokens.append(token(TYPE_HAND_LEFT, left_hand, float(left_grasp)))
    tokens.append(token(TYPE_HAND_RIGHT, right_hand, float(right_grasp)))
    order = list(range(len(object_poses)))
    for index in order:
        type_id = TYPE_OBJECT_ANCHOR if is_anchor[index] else TYPE_OBJECT_OTHER
        tokens.append(token(type_id, object_poses[index], float(object_dynamic[index])))

    output = np.zeros((max_entities, RELATION_DIM), dtype=np.float32)
    mask = np.zeros(max_entities, dtype=bool)
    count = min(len(tokens), max_entities)
    if count:
        output[:count] = np.stack(tokens[:count])
        mask[:count] = True
    return output, mask


def _distance_word(distance: float, bins: tuple[float, ...]) -> str:
    labels = ("contact", "near", "reachable", "far")
    return labels[min(int(np.searchsorted(np.asarray(bins), distance, side="right")), len(labels) - 1)]


def _direction_word(relative_translation: np.ndarray) -> str:
    axis = int(np.argmax(np.abs(relative_translation)))
    if axis == 0:
        return "right" if relative_translation[axis] > 0 else "left"
    if axis == 1:
        return "below" if relative_translation[axis] > 0 else "above"
    return "behind" if relative_translation[axis] > 0 else "in front"


def relation_text(
    categories: np.ndarray,
    object_poses: np.ndarray,
    left_hand: np.ndarray,
    right_hand: np.ndarray,
    previous_distances: np.ndarray | None,
    dt: float,
    cfg: ProjectConfig,
) -> tuple[str, np.ndarray]:
    fragments = []
    distances = np.zeros((len(object_poses), 2), dtype=np.float64)
    for index, (category, object_pose) in enumerate(zip(categories, object_poses, strict=True)):
        left_delta_camera = left_hand[:3, 3] - object_pose[:3, 3]
        right_delta_camera = right_hand[:3, 3] - object_pose[:3, 3]
        distances[index] = [np.linalg.norm(left_delta_camera), np.linalg.norm(right_delta_camera)]
        states = []
        for hand_name, delta_camera, distance, hand_index in (
            ("left", left_delta_camera, distances[index, 0], 0),
            ("right", right_delta_camera, distances[index, 1], 1),
        ):
            motion = "steady"
            if previous_distances is not None:
                speed = (distance - previous_distances[index, hand_index]) / max(dt, 1e-6)
                if speed < -cfg.relations.approach_speed_m_s:
                    motion = "approaching"
                elif speed > cfg.relations.approach_speed_m_s:
                    motion = "leaving"
            states.append(
                f"{hand_name} {_distance_word(float(distance), cfg.relations.distance_bins_m)} "
                f"{_direction_word(delta_camera)} {motion}"
            )
        fragments.append(f"{category}: " + ", ".join(states))
    return "; ".join(fragments), distances


def process_relations(cfg: ProjectConfig, source: str | Path, episode_dir: str | Path) -> StageManifest:
    source = Path(source).resolve()
    episode_dir = Path(episode_dir).resolve()
    entities_dir = episode_dir / "entities"
    objects_path = entities_dir / "objects_initial.npz"
    if not objects_path.is_file():
        raise FileNotFoundError(f"缺少 {objects_path}，请先运行 perception")
    with np.load(objects_path, allow_pickle=False) as archive:
        instance_ids = archive["instance_ids"]
        categories = archive["categories"]
        is_anchor = archive["is_anchor"].astype(bool)
        initial_objects = archive["T_camera0_object"].astype(np.float64)
        object_confidence = archive["confidence"].astype(np.float32)
    graspable_objects = _graspable_object_mask(cfg, instance_ids, categories, is_anchor)
    stereo_track_path = entities_dir / "objects_stereo_track.npz"
    if not stereo_track_path.is_file():
        raise FileNotFoundError(f"缺少 {stereo_track_path}；稳定主方案要求双目物体 3D track")
    with np.load(stereo_track_path, allow_pickle=False) as archive:
        visual_object_camera = archive["T_camera0_object"].astype(np.float64)
        visual_valid_camera = archive["valid"].astype(bool)
    pose_contract_path = episode_dir / "camera" / "wrist_tcp_poses_camera0.npz"
    if pose_contract_path.is_file():
        with np.load(pose_contract_path, allow_pickle=False) as archive:
            left_wrist_camera = archive["T_camera0_left_wrist"].astype(np.float64)
            right_wrist_camera = archive["T_camera0_right_wrist"].astype(np.float64)
            left_tcp_camera = archive["T_camera0_left_tcp"].astype(np.float64)
            right_tcp_camera = archive["T_camera0_right_tcp"].astype(np.float64)
            left_wrist_valid_camera = archive["left_valid"].astype(bool)
            right_wrist_valid_camera = archive["right_valid"].astype(bool)
        with np.load(episode_dir / "camera" / "hands_camera0.npz", allow_pickle=False) as archive:
            left_palm_camera = archive["T_camera0_left_hand"].astype(np.float64)
            right_palm_camera = archive["T_camera0_right_hand"].astype(np.float64)
            left_pinch = archive["left_pinch_m"].astype(np.float64)
            right_pinch = archive["right_pinch_m"].astype(np.float64)
        pose_source = "Step1 camera/wrist_tcp_poses_camera0.npz"
    else:
        # Backward compatibility for episodes prepared before the explicit
        # Step1 wrist/TCP pose contract was introduced.
        with PicoEpisode(source, cfg) as episode:
            hdf_hands = episode.hands_at_camera()
        left_wrist_camera = hdf_hands["T_camera0_left_wrist"].astype(np.float64)
        right_wrist_camera = hdf_hands["T_camera0_right_wrist"].astype(np.float64)
        left_tcp_camera = hdf_hands["T_camera0_left_tcp"].astype(np.float64)
        right_tcp_camera = hdf_hands["T_camera0_right_tcp"].astype(np.float64)
        left_palm_camera = hdf_hands["T_camera0_left_hand"].astype(np.float64)
        right_palm_camera = hdf_hands["T_camera0_right_hand"].astype(np.float64)
        left_pinch = hdf_hands["left_pinch_m"].astype(np.float64)
        right_pinch = hdf_hands["right_pinch_m"].astype(np.float64)
        left_wrist_valid_camera = hdf_hands["left_wrist_valid"].astype(bool)
        right_wrist_valid_camera = hdf_hands["right_wrist_valid"].astype(bool)
        pose_source = "runtime fallback from HDF tracking/hands"

    frame_table = pq.read_table(episode_dir / "sync" / "frame_table.parquet").to_pydict()
    camera_indices = np.asarray(frame_table["camera_index"], dtype=np.int64)
    timestamps = np.asarray(frame_table["timestamp_ns"], dtype=np.int64)
    left = left_tcp_camera[camera_indices]
    right = right_tcp_camera[camera_indices]
    left_wrist = left_wrist_camera[camera_indices]
    right_wrist = right_wrist_camera[camera_indices]
    left_palm = left_palm_camera[camera_indices]
    right_palm = right_palm_camera[camera_indices]
    left_wrist_valid = left_wrist_valid_camera[camera_indices]
    right_wrist_valid = right_wrist_valid_camera[camera_indices]
    layout = classify_initial_layout(instance_ids, categories, initial_objects, left[0], right[0])
    layout_path = entities_dir / "layout_classification.json"
    layout_path.write_text(json.dumps(layout, ensure_ascii=False, indent=2), encoding="utf-8")
    pico_left_grasp = _grasp_hysteresis(
        left_pinch[camera_indices], cfg.perception.grasp_distance_m
    )
    pico_right_grasp = _grasp_hysteresis(
        right_pinch[camera_indices], cfg.perception.grasp_distance_m
    )
    step1_grasp = _align_step1_grasp(episode_dir, timestamps, camera_indices)
    if step1_grasp is None:
        left_grasp = pico_left_grasp
        right_grasp = pico_right_grasp
        grasp_source = "pico_thumb_index_distance_fallback"
    else:
        left_grasp, right_grasp, grasp_source = step1_grasp
    objects, dynamic, owner = latch_object_poses(
        initial_objects,
        left,
        right,
        left_grasp,
        right_grasp,
        latch_distance_m=cfg.perception.latch_distance_m,
        visual_poses=visual_object_camera[camera_indices],
        visual_valid=visual_valid_camera[camera_indices],
        eligible_objects=graspable_objects,
        left_valid=left_wrist_valid,
        right_valid=right_wrist_valid,
    )

    tokens = np.zeros((len(left), cfg.perception.max_entities, RELATION_DIM), dtype=np.float32)
    masks = np.zeros((len(left), cfg.perception.max_entities), dtype=bool)
    T_object_left_tcp = np.zeros((len(left), len(initial_objects), 4, 4), dtype=np.float64)
    T_object_right_tcp = np.zeros_like(T_object_left_tcp)
    T_left_tcp_object = np.zeros_like(T_object_left_tcp)
    T_right_tcp_object = np.zeros_like(T_object_left_tcp)
    T_object_left_wrist = np.zeros_like(T_object_left_tcp)
    T_object_right_wrist = np.zeros_like(T_object_left_tcp)
    T_left_wrist_object = np.zeros_like(T_object_left_tcp)
    T_right_wrist_object = np.zeros_like(T_object_left_tcp)
    text_rows = []
    matrix_rows = []
    previous_distances = None
    task_contract = episode_dir / "step1" / "task_semantics.json"
    if task_contract.is_file():
        task_instruction = str(json.loads(task_contract.read_text(encoding="utf-8"))["instruction"])
    else:
        with h5py.File(source, "r") as file:
            task_instruction = str(file.attrs.get("task_instruction", ""))
    for frame in range(len(left)):
        tokens[frame], masks[frame] = build_relation_tokens(
            left[frame],
            right[frame],
            objects[frame],
            is_anchor,
            bool(left_grasp[frame]),
            bool(right_grasp[frame]),
            dynamic[frame],
            max_entities=cfg.perception.max_entities,
            translation_scale_m=cfg.relations.translation_scale_m,
        )
        for object_index in range(len(initial_objects)):
            object_pose = objects[frame, object_index]
            T_object_left_tcp[frame, object_index] = compose(invert(object_pose), left[frame])
            T_object_right_tcp[frame, object_index] = compose(invert(object_pose), right[frame])
            T_left_tcp_object[frame, object_index] = compose(invert(left[frame]), object_pose)
            T_right_tcp_object[frame, object_index] = compose(invert(right[frame]), object_pose)
            T_object_left_wrist[frame, object_index] = compose(
                invert(object_pose), left_wrist[frame]
            )
            T_object_right_wrist[frame, object_index] = compose(
                invert(object_pose), right_wrist[frame]
            )
            T_left_wrist_object[frame, object_index] = compose(
                invert(left_wrist[frame]), object_pose
            )
            T_right_wrist_object[frame, object_index] = compose(
                invert(right_wrist[frame]), object_pose
            )
        dt = 1.0 / cfg.timeline.control_hz if frame == 0 else (timestamps[frame] - timestamps[frame - 1]) / 1e9
        text, previous_distances = relation_text(
            categories,
            objects[frame],
            left[frame],
            right[frame],
            previous_distances,
            dt,
            cfg,
        )
        text_rows.append(
            {
                "frame_index": frame,
                "timestamp_ns": int(timestamps[frame]),
                "task": task_instruction,
                "layout": layout["layout_class"],
                "relation": text,
                "prompt": f"{task_instruction} Initial layout: {layout['layout_class']}. Spatial context: {text}",
            }
        )
        matrix_rows.append(
            {
                "frame_index": frame,
                "objects": {
                    str(instance_ids[index]): {
                        "T_object_left_tcp": T_object_left_tcp[frame, index].tolist(),
                        "T_object_right_tcp": T_object_right_tcp[frame, index].tolist(),
                        "T_left_tcp_object": T_left_tcp_object[frame, index].tolist(),
                        "T_right_tcp_object": T_right_tcp_object[frame, index].tolist(),
                        "T_object_left_wrist": T_object_left_wrist[frame, index].tolist(),
                        "T_object_right_wrist": T_object_right_wrist[frame, index].tolist(),
                        "T_left_wrist_object": T_left_wrist_object[frame, index].tolist(),
                        "T_right_wrist_object": T_right_wrist_object[frame, index].tolist(),
                    }
                    for index in range(len(initial_objects))
                },
            }
        )

    np.save(entities_dir / "relation_tokens.npy", tokens)
    np.save(entities_dir / "relation_mask.npy", masks)
    np.savez_compressed(
        entities_dir / "poses.npz",
        instance_ids=instance_ids,
        categories=categories,
        is_anchor=is_anchor,
        graspable=graspable_objects,
        object_confidence=object_confidence,
        layout_class=np.asarray(layout["layout_class"]),
        T_camera0_left_tcp=left,
        T_camera0_right_tcp=right,
        T_camera0_left_wrist=left_wrist,
        T_camera0_right_wrist=right_wrist,
        T_camera0_left_palm_geometric=left_palm,
        T_camera0_right_palm_geometric=right_palm,
        left_wrist_valid=left_wrist_valid,
        right_wrist_valid=right_wrist_valid,
        # Compatibility aliases: Step2 hand means the HDF XR wrist frame.
        T_camera0_left_hand=left_wrist,
        T_camera0_right_hand=right_wrist,
        T_camera0_object=objects,
        T_object_left_tcp=T_object_left_tcp,
        T_object_right_tcp=T_object_right_tcp,
        T_left_tcp_object=T_left_tcp_object,
        T_right_tcp_object=T_right_tcp_object,
        T_object_left_wrist=T_object_left_wrist,
        T_object_right_wrist=T_object_right_wrist,
        T_left_wrist_object=T_left_wrist_object,
        T_right_wrist_object=T_right_wrist_object,
        T_object_left_hand=T_object_left_wrist,
        T_object_right_hand=T_object_right_wrist,
        left_grasp=left_grasp,
        right_grasp=right_grasp,
        pico_pinch_grasp_left=pico_left_grasp,
        pico_pinch_grasp_right=pico_right_grasp,
        object_dynamic=dynamic,
        object_owner=owner,
    )
    (entities_dir / "relation_text.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in text_rows), encoding="utf-8"
    )
    (entities_dir / "relation_matrices_debug.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in matrix_rows), encoding="utf-8"
    )
    metadata = {
        "schema": "camera_frame_entities_v2_tcp",
        "dimension": RELATION_DIM,
        "max_entities": cfg.perception.max_entities,
        "translation_scale_m": cfg.relations.translation_scale_m,
        "reference_frame": "camera0_static_frame",
        "token_order": ["left_tcp", "right_tcp", "objects_in_catalog_order"],
        "token_layout": ["type:1", "entity_in_cam0:9", "left_tcp_in_cam0:9", "right_tcp_in_cam0:9", "flag:1"],
        "pose_source": pose_source,
        "grasp_source": grasp_source,
        "graspable_instances": [
            str(instance_id)
            for instance_id, graspable in zip(instance_ids, graspable_objects, strict=True)
            if graspable
        ],
        "tcp_source": "geometric palm axes plus fixed side TCP alignment; origin at XR wrist",
        "pose_sync": "nearest hand XrTime to camera exposure XrTime, then camera0_static_frame",
        "matrix_direction": [
            "relation token stores all entity/TCP poses directly in camera0_static_frame",
            "debug matrices include both object-TCP and object-wrist relative transforms",
        ],
        "layout_classification": layout,
        "flag": "hand token: grasp; object token: latched_dynamic",
    }
    (entities_dir / "relation_schema.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = StageManifest(
        schema_version=cfg.schema_version,
        stage="relations",
        episode=source.stem,
        source_path=str(source),
        source_sha256=file_sha256(source),
        config=metadata,
        outputs={
            "tokens": str(entities_dir / "relation_tokens.npy"),
            "mask": str(entities_dir / "relation_mask.npy"),
            "poses": str(entities_dir / "poses.npz"),
            "text": str(entities_dir / "relation_text.jsonl"),
            "matrix_debug": str(entities_dir / "relation_matrices_debug.jsonl"),
            "layout": str(layout_path),
        },
        metrics={
            "frames": len(left),
            "objects": len(initial_objects),
            "left_grasp_ratio": float(left_grasp.mean()),
            "right_grasp_ratio": float(right_grasp.mean()),
            "left_step1_pico_grasp_agreement_ratio": float(
                np.mean(left_grasp == pico_left_grasp)
            ),
            "right_step1_pico_grasp_agreement_ratio": float(
                np.mean(right_grasp == pico_right_grasp)
            ),
            "dynamic_ratio": float(dynamic.mean()),
            "layout_class": layout["layout_class"],
            "left_wrist_valid_ratio": float(left_wrist_valid.mean()),
            "right_wrist_valid_ratio": float(right_wrist_valid.mean()),
        },
    )
    manifest.write(episode_dir / "relations.manifest.json")
    return manifest
