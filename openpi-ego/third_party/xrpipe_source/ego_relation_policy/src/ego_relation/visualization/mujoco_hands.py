"""Step1 BrainCo Revo2-only MuJoCo replay renderer.

The renderer deliberately contains no G1 body.  Both free-floating Revo2
hands are driven directly by the Mode2 TCP pose9 and normalized 6-motor
commands in one fixed robot-base world frame.
"""

from __future__ import annotations

from pathlib import Path
import ctypes
import os
import warnings

import cv2
import numpy as np


MOTOR_ORDER = ("thumb_flex", "thumb_rot", "index", "middle", "ring", "pinky")
ACTUATOR_JOINT = {
    "left": {
        "thumb_flex": "left_thumb_proximal_joint",
        "thumb_rot": "left_thumb_metacarpal_joint",
        "index": "left_index_proximal_joint",
        "middle": "left_middle_proximal_joint",
        "ring": "left_ring_proximal_joint",
        "pinky": "left_pinky_proximal_joint",
    },
    "right": {
        "thumb_flex": "right_thumb_proximal_joint",
        "thumb_rot": "right_thumb_metacarpal_joint",
        "index": "right_index_proximal_joint",
        "middle": "right_middle_proximal_joint",
        "ring": "right_ring_proximal_joint",
        "pinky": "right_pinky_proximal_joint",
    },
}
DISTAL_RATIO = {"thumb": 1.0, "index": 1.155, "middle": 1.155, "ring": 1.155, "pinky": 1.155}
TCP_TO_INWARD_PALM = {
    "left": np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),
    "right": np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
}
MOUNT_POSITION = {
    "left": np.asarray([0.0415, 0.003, 0.0]),
    "right": np.asarray([0.0415, -0.003, 0.0]),
}


def _rot6d_to_mat(d6: np.ndarray) -> np.ndarray:
    a, b = np.asarray(d6, dtype=np.float64)[:3], np.asarray(d6, dtype=np.float64)[3:6]
    x = a / max(float(np.linalg.norm(a)), 1e-12)
    b = b - float(np.dot(x, b)) * x
    y = b / max(float(np.linalg.norm(b)), 1e-12)
    return np.stack((x, y, np.cross(x, y)), axis=1)


def _quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Stable matrix to MuJoCo scalar-first quaternion conversion."""
    matrix = np.asarray(matrix, dtype=np.float64)
    q = np.empty(4, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2
        q[:] = (0.25 * scale, (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale, (matrix[1, 0] - matrix[0, 1]) / scale)
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            q[:] = ((matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale)
        elif index == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            q[:] = ((matrix[0, 2] - matrix[2, 0]) / scale, (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale, (matrix[1, 2] + matrix[2, 1]) / scale)
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            q[:] = ((matrix[1, 0] - matrix[0, 1]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale)
    return q / max(float(np.linalg.norm(q)), 1e-12)


def _pose9_transform(pose9: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _rot6d_to_mat(pose9[3:9])
    transform[:3, 3] = pose9[:3]
    return transform


def _tcp_to_hand_transforms(assets: Path) -> dict[str, np.ndarray]:
    """Mode2 visualization-only TCP -> Revo2 base transform.

    This is the same chirality-aware installation used by
    egodata_targeting_project's G1HandsBackend.action_hand_pose_from_base_tcp.
    """
    output: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        with np.load(assets / f"fk_tables_{side}.npz") as archive:
            robot_palm = archive["robot_palm"]
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = TCP_TO_INWARD_PALM[side] @ robot_palm.T
        transform[:3, 3] = MOUNT_POSITION[side]
        output[side] = transform
    return output


def _build_model(assets: Path):
    # Set the backend before importing MuJoCo.  EGL is preferred on the data
    # workstation; users can explicitly select glfw/OSMesa in their shell.
    os.environ.setdefault("MUJOCO_GL", "egl")
    if os.environ.get("MUJOCO_GL", "").lower() == "egl":
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    # pyrender pins PyOpenGL==3.1.0, while recent MuJoCo only needs one EGL
    # extension typedef which that old binding forgot to expose.  Defining the
    # official opaque handle type keeps the unified UV environment solvable
    # and avoids modifying site-packages.
    if os.environ.get("MUJOCO_GL", "").lower() == "egl":
        from OpenGL import EGL

        if not hasattr(EGL, "EGLDeviceEXT"):
            EGL.EGLDeviceEXT = ctypes.c_void_p
    import mujoco

    spec = mujoco.MjSpec()
    spec.worldbody.add_geom(
        name="base_floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=(0, 0, 0),
        size=(2.0, 2.0, 0.01),
        rgba=(0.13, 0.16, 0.19, 1.0),
        contype=0,
        conaffinity=0,
    )
    axis_length = 0.16
    for name, endpoint, color in (
        ("base_x", (axis_length, 0, 0), (0.95, 0.18, 0.20, 1)),
        ("base_y", (0, axis_length, 0), (0.20, 0.88, 0.36, 1)),
        ("base_z", (0, 0, axis_length), (0.20, 0.48, 0.98, 1)),
    ):
        spec.worldbody.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=(0, 0, 0, *endpoint),
            size=(0.006,),
            rgba=color,
            contype=0,
            conaffinity=0,
        )
    left_frame = spec.worldbody.add_frame(name="left_mount")
    right_frame = spec.worldbody.add_frame(name="right_mount")
    left = mujoco.MjSpec.from_file(str(assets / "xml_left/brainco-lefthand-v2-patched-free.xml"))
    right = mujoco.MjSpec.from_file(str(assets / "xml_right/brainco-righthand-v2-patched-free.xml"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec.attach(left, prefix="L_", frame=left_frame)
        spec.attach(right, prefix="R_", frame=right_frame)
        model = spec.compile()
    model.vis.headlight.ambient[:] = (0.58, 0.58, 0.58)
    model.vis.headlight.diffuse[:] = (0.78, 0.78, 0.78)
    model.vis.headlight.specular[:] = (0.12, 0.12, 0.12)
    return mujoco, model


def _set_hand(
    mujoco,
    model,
    data,
    side: str,
    pose9: np.ndarray,
    command: np.ndarray,
    tcp_to_hand: np.ndarray,
) -> None:
    prefix = "L_" if side == "left" else "R_"
    free = model.joint(prefix + "floating_base_joint")
    address = int(free.qposadr[0])
    hand_pose = _pose9_transform(pose9) @ tcp_to_hand
    data.qpos[address:address + 3] = hand_pose[:3, 3]
    data.qpos[address + 3:address + 7] = _quat_wxyz(hand_pose[:3, :3])

    proximal: dict[str, int] = {}
    for motor, value in zip(MOTOR_ORDER, np.clip(command, 0.0, 1.0)):
        name = prefix + ACTUATOR_JOINT[side][motor]
        joint = model.joint(name)
        joint_address = int(joint.qposadr[0])
        data.qpos[joint_address] = float(value) * float(joint.range[1])
        proximal[motor] = joint_address
    for finger, ratio in DISTAL_RATIO.items():
        motor = "thumb_flex" if finger == "thumb" else finger
        distal = model.joint(prefix + f"{side}_{finger}_distal_joint")
        data.qpos[int(distal.qposadr[0])] = np.clip(
            ratio * data.qpos[proximal[motor]], float(distal.range[0]), float(distal.range[1])
        )
    mujoco.mj_forward(model, data)


def render_revo2_replay(
    state: np.ndarray,
    action: np.ndarray,
    assets: Path,
    output_dir: Path,
    *,
    jpeg_quality: int = 82,
    width: int = 960,
    height: int = 540,
) -> tuple[list[Path], dict[str, object]]:
    """Render Mode2 action TCP targets plus fixed state-hand closeups."""
    state = np.asarray(state, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    if state.ndim != 2 or state.shape[1] < 30:
        raise ValueError(f"Mode2 state should be (T, >=30), got {state.shape}")
    if action.shape != state.shape:
        raise ValueError(f"Mode2 action should match state, got {action.shape} vs {state.shape}")
    assets = Path(assets).resolve()
    required = (
        assets / "xml_left/brainco-lefthand-v2-patched-free.xml",
        assets / "xml_right/brainco-righthand-v2-patched-free.xml",
    )
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("BrainCo Revo2 MJCF assets are incomplete: " + ", ".join(map(str, required)))

    mujoco, model = _build_model(assets)
    tcp_to_hand = _tcp_to_hand_transforms(assets)
    try:
        from mujoco.rendering.classic.renderer import Renderer
    except ImportError as error:
        raise RuntimeError(
            "MuJoCo offscreen renderer unavailable. Use MUJOCO_GL=egl on a machine with an EGL device."
        ) from error
    data = mujoco.MjData(model)
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), int(width))
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), int(height))
    renderer = Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    positions = np.concatenate((action[:, 0:3], action[:, 9:12]), axis=0)
    finite = positions[np.isfinite(positions).all(axis=1)]
    center = np.mean(finite, axis=0) if len(finite) else np.zeros(3)
    extent = np.ptp(finite, axis=0) if len(finite) else np.ones(3) * 0.3
    camera.lookat[:] = center
    camera.distance = max(0.55, float(np.linalg.norm(extent)) * 1.8 + 0.28)
    camera.azimuth = 138
    camera.elevation = -24
    geom_body_names = [model.body(model.geom_bodyid[index]).name or "" for index in range(model.ngeom)]
    hand_geom = {
        "left": np.asarray([name.startswith("L_") for name in geom_body_names]),
        "right": np.asarray([name.startswith("R_") for name in geom_body_names]),
    }

    def render_closeup(side: str, row: np.ndarray) -> np.ndarray:
        mujoco.mj_resetData(model, data)
        fixed = np.asarray([0.0, 0.0, 0.35, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        offset = 0 if side == "left" else 9
        command_offset = 18 if side == "left" else 24
        _set_hand(
            mujoco,
            model,
            data,
            side,
            fixed,
            row[command_offset : command_offset + 6],
            tcp_to_hand[side],
        )
        other = "right" if side == "left" else "left"
        other_fixed = fixed.copy()
        other_fixed[:3] = (10.0, 10.0, 10.0)
        other_command_offset = 18 if other == "left" else 24
        _set_hand(
            mujoco,
            model,
            data,
            other,
            other_fixed,
            row[other_command_offset : other_command_offset + 6],
            tcp_to_hand[other],
        )
        keep = hand_geom[side]
        close_camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, close_camera)
        close_camera.lookat[:] = data.geom_xpos[keep].mean(axis=0)
        close_camera.distance = 0.29
        close_camera.azimuth = 135 if side == "left" else 225
        close_camera.elevation = -20
        renderer.update_scene(data, close_camera)
        # The classic EGL renderer is double-buffered. After switching the
        # camera three times per control tick, the first read can contain a
        # partially stale static-geom buffer; discard it and keep the second.
        renderer.render()
        return renderer.render().copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for frame, row in enumerate(state):
        output = output_dir / f"{frame:05d}.jpg"
        if not output.is_file():
            mujoco.mj_resetData(model, data)
            target = action[frame]
            _set_hand(mujoco, model, data, "left", target[0:9], target[18:24], tcp_to_hand["left"])
            _set_hand(mujoco, model, data, "right", target[9:18], target[24:30], tcp_to_hand["right"])
            renderer.update_scene(data, camera)
            renderer.render()
            main = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            left_close = cv2.cvtColor(render_closeup("left", row), cv2.COLOR_RGB2BGR)
            right_close = cv2.cvtColor(render_closeup("right", row), cv2.COLOR_RGB2BGR)
            main_width = int(round(width * 2 / 3))
            side_width = width - main_width
            main = cv2.resize(main, (main_width, height), interpolation=cv2.INTER_AREA)
            left_close = cv2.resize(left_close, (side_width, height // 2), interpolation=cv2.INTER_AREA)
            right_close = cv2.resize(right_close, (side_width, height - height // 2), interpolation=cv2.INTER_AREA)
            bgr = np.zeros((height, width, 3), dtype=np.uint8)
            bgr[:, :main_width] = main
            bgr[: height // 2, main_width:] = left_close
            bgr[height // 2 :, main_width:] = right_close
            cv2.line(bgr, (main_width, 0), (main_width, height), (55, 78, 94), 2)
            cv2.line(bgr, (main_width, height // 2), (width, height // 2), (55, 78, 94), 2)
            cv2.rectangle(bgr, (0, 0), (main_width, 42), (10, 15, 20), -1)
            cv2.rectangle(bgr, (main_width, 0), (width, 32), (10, 15, 20), -1)
            cv2.rectangle(bgr, (main_width, height // 2), (width, height // 2 + 32), (10, 15, 20), -1)
            cv2.putText(
                bgr,
                f"ACTION target | robot_base | frame {frame:05d}",
                (18, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (232, 238, 242),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(bgr, "STATE left Revo2", (main_width + 12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 220, 210), 1, cv2.LINE_AA)
            cv2.putText(bgr, "STATE right Revo2", (main_width + 12, height // 2 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 180, 255), 1, cv2.LINE_AA)
            if not cv2.imwrite(str(output), bgr, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]):
                raise RuntimeError(f"Failed to write MuJoCo frame: {output}")
        outputs.append(output)
    renderer.close()
    return outputs, {
        "renderer": "mujoco",
        "model": "BrainCo Revo2 Mode2 action hands + state closeups",
        "mount": "TCP_TO_INWARD_PALM @ robot_palm.T",
        "world_frame": "robot_base",
        "fixed_camera": True,
        "camera_lookat": center.tolist(),
        "camera_distance": float(camera.distance),
        "frames": len(outputs),
    }
