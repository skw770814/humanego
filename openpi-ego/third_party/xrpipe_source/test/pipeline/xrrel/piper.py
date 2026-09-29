"""Official AgileX Piper gripper mesh loader and camera renderer.

The loader consumes the upstream Piper URDF/Xacro assets and keeps only the
gripper links.  It exposes the existing Step3 TCP contract: a camera-frame
TCP pose plus a normalized open ratio produces an RGBA mesh image and depth.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


class PiperGripperModel:
    def __init__(self, model_path=None, assets_root=None, *, width=1080, height=810,
                 fx=None, fy=None, cx=None, cy=None):
        try:
            # X11 is the most reliable path on the workstation (DISPLAY=:0);
            # EGL is used automatically only for truly headless runs.  The
            # explicit PIPER_GL_PLATFORM override is useful on multi-GPU hosts.
            requested_platform = os.environ.get("PIPER_GL_PLATFORM", "auto")
            if requested_platform == "auto":
                # pyrender selects its X11/pyglet backend when the variable is
                # absent; "pyglet" is not a valid PYOPENGL_PLATFORM value.
                if os.environ.get("DISPLAY"):
                    os.environ.pop("PYOPENGL_PLATFORM", None)
                else:
                    os.environ["PYOPENGL_PLATFORM"] = "egl"
            else:
                os.environ["PYOPENGL_PLATFORM"] = requested_platform
            import pyrender
            from yourdfpy import URDF
        except ImportError as exc:
            raise RuntimeError("Piper mesh rendering requires yourdfpy, trimesh and pyrender") from exc
        self.pyrender = pyrender
        self.width, self.height = int(width), int(height)
        self.model_path = Path(model_path) if model_path else (
            Path(__file__).resolve().parents[1] / "assets/piper/piper_gripper.urdf"
        )
        if assets_root:
            self.assets_root = Path(assets_root)
        else:
            self.assets_root = Path(__file__).resolve().parents[1] / "assets/piper/agx_arm_urdf/piper"
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.model_path.is_file():
            self._make_gripper_urdf(self.model_path)
        self.urdf = URDF.load(
            str(self.model_path),
            mesh_dir=str(self.assets_root),
            filename_handler=self._filename_handler,
            force_mesh=False,
        )
        if "gripper" not in self.urdf.actuated_joint_names:
            raise ValueError("Piper URDF does not contain actuated gripper joint")
        self.open_joint = float(self.urdf.joint_map["gripper"].limit.upper)
        self.closed_joint = float(self.urdf.joint_map["gripper"].limit.lower)
        self.scene = self.urdf.scene
        self.fx, self.fy = float(fx or 1.0), float(fy or 1.0)
        self.cx, self.cy = float(cx or width / 2), float(cy or height / 2)
        self.renderer = pyrender.OffscreenRenderer(self.width, self.height)
        self._last_q = None

    def _filename_handler(self, filename=None, **kwargs):
        filename = filename if filename is not None else kwargs.get("fname", "")
        prefix = "package://agx_arm_description/agx_arm_urdf/piper/"
        if filename.startswith(prefix):
            return str(self.assets_root / filename[len(prefix):])
        if not Path(filename).is_absolute():
            return str(self.assets_root / filename)
        return filename

    def _make_gripper_urdf(self, out_path: Path):
        """Resolve the official xacro include into a ROS-independent gripper URDF."""
        piper_root = self.assets_root
        xacro = piper_root / "urdf/piper_with_gripper_description.xacro"
        base = piper_root / "urdf/piper_description.urdf"
        if not xacro.is_file() or not base.is_file():
            raise FileNotFoundError(f"Missing Piper URDF assets under {piper_root}")
        base_root = ET.parse(base).getroot()
        extra_root = ET.parse(xacro).getroot()
        links = {"gripper_base", "gripper_link", "gripper_link1", "gripper_link2"}
        root = ET.Element("robot", {"name": "piper_gripper"})
        for child in extra_root:
            tag = child.tag.rsplit("}", 1)[-1]
            if tag == "link" and child.attrib.get("name") in links:
                root.append(ET.fromstring(ET.tostring(child)))
        for child in extra_root:
            tag = child.tag.rsplit("}", 1)[-1]
            if tag != "joint":
                continue
            parent = child.find("parent")
            joint_child = child.find("child")
            if parent is None or joint_child is None:
                continue
            if parent.attrib.get("link") == "gripper_base" and joint_child.attrib.get("link") in links:
                root.append(ET.fromstring(ET.tostring(child)))
        for mesh in root.iter("mesh"):
            f = mesh.attrib.get("filename", "")
            prefix = "package://agx_arm_description/agx_arm_urdf/piper/"
            if f.startswith(prefix):
                mesh.set("filename", f[len(prefix):])
        ET.indent(root, space="  ")
        out_path.write_bytes(ET.tostring(root, encoding="utf-8", xml_declaration=True))

    def set_open_ratio(self, ratio: float):
        ratio = float(np.clip(ratio, 0.0, 1.0))
        q = self.closed_joint + ratio * (self.open_joint - self.closed_joint)
        if self._last_q != q:
            self.urdf.update_cfg([q])
            self._last_q = q

    def finger_origins_model(self):
        """The two finger link-frame origins in gripper_base coords, (2,3).

        Every geometry node of `gripper_link1`/`gripper_link2` carries a visual
        origin of identity, so its node transform IS the link transform.
        """
        origins = {}
        for node in self.scene.graph.nodes_geometry:
            for side in ("gripper_link1", "gripper_link2"):
                if side in str(node) and side not in origins:
                    T_node, _ = self.scene.graph[node]
                    origins[side] = np.asarray(T_node[:3, 3], dtype=np.float64)
        missing = [s for s in ("gripper_link1", "gripper_link2") if s not in origins]
        if missing:
            raise RuntimeError(f"Piper mesh has no geometry for {missing}")
        return np.stack([origins["gripper_link1"], origins["gripper_link2"]])

    def fingertip_midpoint_model(self) -> np.ndarray:
        """Fingertip midpoint of the two fingers in gripper_base coords.

        The official SolidWorks-exported URDF places each finger's link frame
        AT the fingertip -- the finger mesh hangs in -Z from its frame origin
        and the tip-face sub-mesh (gripper_link*.dae #2) has its distal edge
        exactly on that origin.  So the midpoint of the two link origins IS the
        fingertip midpoint: (0, 0, 0.138) at every open ratio (the fingers only
        translate along +/-Y, so the midpoint stays on the tool axis).

        Do not go back to a "min-Z vertex face" heuristic here: the finger mesh
        spans z in [0.0615, 0.138], and its LOW-z end is the mounting carriage
        that meets the gripper housing -- picking that end shortens the TCP by
        7.65 cm and draws the gripper past the hand.
        """
        midpoint = self.finger_origins_model().mean(axis=0)
        if midpoint.shape != (3,) or not np.isfinite(midpoint).all():
            raise RuntimeError("Invalid Piper fingertip midpoint")
        return midpoint

    def mesh_fingertip_extent_model(self, distal: bool = True) -> np.ndarray:
        """Midpoint of the two fingers' distal (or proximal) mesh end faces.

        Used only as a QA cross-check: on the official mesh the distal value
        must agree with `fingertip_midpoint_model()` to within a millimetre.
        """
        ends = []
        for side in ("gripper_link1", "gripper_link2"):
            verts = []
            for node in self.scene.graph.nodes_geometry:
                if side not in str(node):
                    continue
                T_node, geom_name = self.scene.graph[node]
                mesh = self.scene.geometry.get(geom_name)
                if mesh is None or not hasattr(mesh, "vertices"):
                    continue
                v = np.asarray(mesh.vertices, dtype=np.float64)
                verts.append((np.asarray(T_node[:3, :3]) @ v.T).T + np.asarray(T_node[:3, 3]))
            if not verts:
                raise RuntimeError(f"Piper mesh has no geometry for {side}")
            v = np.concatenate(verts, axis=0)
            if distal:
                z = float(np.max(v[:, 2]))
                face = v[v[:, 2] >= z - 1e-5]
            else:
                z = float(np.min(v[:, 2]))
                face = v[v[:, 2] <= z + 1e-5]
            ends.append(np.mean(face, axis=0))
        return 0.5 * (ends[0] + ends[1])

    def pose_from_tcp(self, T_cam_hand: np.ndarray, T_hand_from_piper: np.ndarray,
                      open_ratio: float, tcp_offset_model: np.ndarray | None = None) -> np.ndarray:
        """Construct a Piper pose whose mesh fingertip midpoint is T_cam_hand.

        `tcp_offset_model` is the explicit gripper_base -> fingertip offset (the
        Piper analogue of HumanEgo-main's `t_flange_tool`).  When it is None the
        offset is taken from the URDF itself via forward kinematics.
        """
        self.set_open_ratio(open_ratio)
        self.last_mesh_distal_extent_model = self.mesh_fingertip_extent_model(distal=True)
        if tcp_offset_model is None:
            p_model = self.fingertip_midpoint_model()
            self.last_anchor_source = "fk_link_origins"
        else:
            p_model = np.asarray(tcp_offset_model, dtype=np.float64)
            if p_model.shape != (3,) or not np.isfinite(p_model).all():
                raise ValueError(f"Invalid Piper TCP offset: {tcp_offset_model!r}")
            self.last_anchor_source = "calibration"
        self.last_fingertip_midpoint_model = p_model.copy()
        # T_hand_from_piper maps Piper local vectors into the HumanEgo hand
        # frame.  Solve the root translation directly from the fingertip anchor;
        # no root/TCP sign convention or fixed translation is used.
        R_hand_from_piper = np.asarray(T_hand_from_piper[:3, :3], dtype=np.float64)
        R_model = np.asarray(T_cam_hand[:3, :3], dtype=np.float64) @ R_hand_from_piper
        out = np.eye(4, dtype=np.float64)
        out[:3, :3] = R_model
        out[:3, 3] = np.asarray(T_cam_hand[:3, 3], dtype=np.float64) - R_model @ p_model
        self.last_tcp_alignment_error_m = float(np.linalg.norm(
            out[:3, :3] @ p_model + out[:3, 3] - np.asarray(T_cam_hand[:3, 3])
        ))
        return out

    def render(self, T_cam_model: np.ndarray, open_ratio: float):
        self.set_open_ratio(open_ratio)
        pr = self.pyrender
        scene = pr.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.25, 0.25, 0.25, 1.0])
        cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0])
        for node in self.scene.graph.nodes_geometry:
            T_model, geom_name = self.scene.graph[node]
            if geom_name not in self.scene.geometry:
                continue
            mesh = self.scene.geometry[geom_name]
            if not hasattr(mesh, "vertices"):
                continue
            material = pr.MetallicRoughnessMaterial(
                baseColorFactor=[0.02, 0.02, 0.02, 1.0],
                metallicFactor=0.25, roughnessFactor=0.45,
            )
            pm = pr.Mesh.from_trimesh(mesh, material=material, smooth=False)
            scene.add(pm, pose=cv_to_gl @ np.asarray(T_cam_model) @ T_model)
        camera = pr.IntrinsicsCamera(self.fx, self.fy, self.cx, self.cy, znear=0.01, zfar=10.0)
        scene.add(camera, pose=np.eye(4))
        light = pr.DirectionalLight(color=np.ones(3), intensity=3.0)
        scene.add(light, pose=np.eye(4))
        color, depth = self.renderer.render(scene, flags=pr.RenderFlags.RGBA)
        return color, depth

    def close(self):
        if self.renderer is not None:
            self.renderer.delete()

    def manifest(self):
        anchor = getattr(self, "last_fingertip_midpoint_model", None)
        if anchor is None:
            anchor = self.fingertip_midpoint_model()
        return {
            "model": str(self.model_path),
            "assets_root": str(self.assets_root),
            "open_joint": self.open_joint,
            "closed_joint": self.closed_joint,
            "actuated_joint": "gripper",
            "mimic_joints": {"gripper_joint1": 0.5, "gripper_joint2": -0.5},
            "mesh_units": "meter",
            "material": "black",
            "pose_contract": "HumanEgo VisualKpts fingertip-midpoint frame",
            "tcp_alignment": "fingertip anchor placed on the Step1 5-keypoint TCP midpoint",
            "tcp_anchor_source": getattr(self, "last_anchor_source", "fk_link_origins"),
            "tcp_offset_model_m": np.asarray(anchor, dtype=np.float64).tolist(),
            # gripper_base sits 4.5 mm off the flange (gripper_base_joint), so the
            # flange->fingertip distance is the Piper analogue of HumanEgo-main's
            # t_flange_tool (0.145 m for their Trossen gripper).
            "flange_to_tip_m": float(anchor[2]) + 0.0045,
        }


def _load_calibration_payload(path=None) -> dict:
    if path is None:
        default = Path(__file__).resolve().parents[1] / "assets/piper/tcp_calibration.json"
        path = default if default.is_file() else None
    if path is None:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_tcp_calibration(path=None):
    """Load T_hand_from_piper (Piper local axes -> HumanEgo hand axes).

    Only the rotation takes part in placing the gripper: the root translation is
    solved every frame so that the fingertip anchor lands on T_cam_hand.  The
    `translation_m` field therefore does NOT move the gripper -- use
    `load_tcp_offset_model()` for that.
    """
    payload = _load_calibration_payload(path)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(payload.get("rotation", np.eye(3)), dtype=np.float64)
    T[:3, 3] = np.asarray(payload.get("translation_m", [0.0, 0.0, 0.0]), dtype=np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError(f"Invalid Piper TCP calibration: {path}")
    return T


def load_tcp_offset_model(path=None):
    """gripper_base -> fingertip offset in metres, or None to use URDF FK.

    This is the Piper analogue of HumanEgo-main's `t_flange_tool`
    ("cfg/inference/example_dualarm/RobotArmTrossen*.yaml"): one explicit
    constant that says where the fingertip is, instead of deriving it from mesh
    vertices.  For the official Piper gripper it is [0, 0, 0.138].
    """
    payload = _load_calibration_payload(path)
    offset = payload.get("tcp_offset_model_m")
    if offset is None:
        return None
    offset = np.asarray(offset, dtype=np.float64)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError(f"Invalid Piper TCP offset: {offset!r}")
    return offset
