"""投影链路: 世界坐标关键点 -> 单眼像素坐标。

链路出处见 plan §1.5/§1.6/§1.7, 全部来自源码, 不是猜的:

  源码 Assets/Scripts/Utils.cs (PICO 客户端 v1.1.1):
    PXR_Enterprise.GetCameraIntrinsicsfor4U(width, height, 76.35f, 61.05f)
    PXR_EnterprisePlugin.GetCameraExtrinsics(out left, out right)

  反解出的内参公式 (吻合到 1e-8 相对误差):
    fx = (W/2)/tan(hfov/2)    W=2160 -> 1373.6681014263   (数据 1373.6681390725917)
    fy = (H/2)/tan(vfov/2)    H= 810 ->  686.8680544306   (数据  686.8680648828482)
    cx = W/2 - 0.5 = 1079.5 ;  cy = H/2 - 0.5 = 404.5

  关键: 调用方传的宽是"双眼总宽 2160", 高是"单眼高 810"。
  ⇒ fy=686.868 才是正确的单眼焦距, fx 因为宽度按双眼算而大了一倍 (fx/fy=1.99990)。
  换算到单眼 1080x810: f = 686.868 (方形像素), c_local = (539.5, 404.5)。

链路:
  p_world  --head 逆变换: p_head = R_head^T (p_world - t_head)-->  p_head  [OpenXR 右/上/后]
  p_head   --Rz(-90)------------------------------------------>  p_dev   [上/左/后]
  p_dev    --E: p_cam = R (p_dev - t)------------------------->  p_cam   [OpenCV 右/下/前]
  p_cam    --针孔 u = f X/Z + cx, v = f Y/Z + cy-------------->  单眼像素
  右眼: u_full = u + 1080
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np

from .io_tracking import quat_to_mat

# --- 视频几何 (plan §1.4) ---
EYE_W = 1080
EYE_H = 810
FULL_W = 2160
FULL_H = 810

# --- 标称内参 (plan §1.5) ---
NOMINAL_F = 686.868
NOMINAL_CX = 539.5
NOMINAL_CY = 404.5

# --- device 系相对 Head 系的绕 Z 转角 (plan §1.6 / §1.7) ---
RZ_DEVICE_DEG = -90.0


def rotz(deg: float) -> np.ndarray:
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


_RZ_DEV = rotz(RZ_DEVICE_DEG)


@dataclass
class CameraParams:
    """一次投影所需的全部几何参数。

    全部可配置, 生效值会写进 calib.json 便于复核 —— 不硬编码猜测。
    """

    f: float = NOMINAL_F
    cx: float = NOMINAL_CX
    cy: float = NOMINAL_CY
    rz_device_deg: float = RZ_DEVICE_DEG
    # 外参施加形式: True -> p_cam = R(p_dev - t), False -> p_cam = R p_dev + t
    extrinsic_subtract_t: bool = True
    # 眼别归属: 第 0 个外参 (数据里的 "left") 对应画面的哪半幅
    eye0_is_left_half: bool = True
    # head 四元数的旋转方向: False -> p_head = Rᵀ(p_world − t) (数据给的是 head->world)
    #                          True  -> p_head = R (p_world − t)  (即数据给的其实是 world->head)
    # 由**视频本身**判定 (图像运动互相关: 正取 r=+0.94, 取反 r=−0.90), 不靠猜。
    flip_head_quat: bool = False
    # 标定增量 (加在标称值上)
    d_f: float = 0.0
    d_cx: float = 0.0
    d_cy: float = 0.0
    # 完整 6-DoF 微调: 施加在 device 系的额外刚体变换
    extra_R: np.ndarray = field(default_factory=lambda: np.eye(3))
    extra_t: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def eff_f(self) -> float:
        return self.f + self.d_f

    @property
    def eff_cx(self) -> float:
        return self.cx + self.d_cx

    @property
    def eff_cy(self) -> float:
        return self.cy + self.d_cy

    def to_json(self) -> dict:
        d = asdict(self)
        d["extra_R"] = np.asarray(self.extra_R).tolist()
        d["extra_t"] = np.asarray(self.extra_t).tolist()
        d["effective"] = {
            "f": self.eff_f,
            "cx": self.eff_cx,
            "cy": self.eff_cy,
        }
        return d

    @staticmethod
    def from_json(d: dict) -> "CameraParams":
        d = dict(d)
        d.pop("effective", None)
        if "extra_R" in d:
            d["extra_R"] = np.asarray(d["extra_R"], dtype=np.float64)
        if "extra_t" in d:
            d["extra_t"] = np.asarray(d["extra_t"], dtype=np.float64)
        known = {k: v for k, v in d.items() if k in CameraParams.__dataclass_fields__}
        return CameraParams(**known)


class Projector:
    """把世界坐标点投到单眼像素坐标。

    注意 step 1 (head 逆变换) 是**不可关闭**的固定环节 —— 手部关键点是世界坐标,
    相机固连在头上, 不补偿会彻底错位 (plan §1.2b 的量化表)。
    """

    def __init__(self, extrinsics: np.ndarray, params: CameraParams | None = None):
        self.E = np.asarray(extrinsics, dtype=np.float64)  # (2,4,4)
        self.p = params or CameraParams()
        self.Rz = rotz(self.p.rz_device_deg)

    # ---- step 1 ----
    def world_to_head(self, p_world: np.ndarray, head_pos: np.ndarray, head_quat: np.ndarray):
        """p_head = R_headᵀ (p_world − t_head)。

        head_pos/head_quat 直接取自数据 Head.pose (与手部同一个 tracking space,
        都是绝对量, 无父子关系 —— 源码 GetSensorJson / GetHandJsonData)。

        `flip_head_quat` 控制把数据的四元数当作 head->world 还是 world->head。
        这个方向**不能靠推理定**: 两种取法给出的旋转角完全相同、只差符号,
        而 "Z<0 在头前方" 这类静态自检对反号不敏感(小角度下都能过)。
        所以交给视频判定 (图像运动互相关: 正取 r=+0.94, 取反 r=−0.90)。
        """
        R = quat_to_mat(head_quat)
        if self.p.flip_head_quat:
            R = R.T
        return (np.asarray(p_world, dtype=np.float64) - head_pos) @ R

    # ---- step 2/3 ----
    def head_to_cam(self, p_head: np.ndarray, eye: int) -> np.ndarray:
        p_dev = p_head @ self.Rz.T
        p_dev = (p_dev - self.p.extra_t) @ np.asarray(self.p.extra_R).T
        E = self.E[eye]
        R, t = E[:3, :3], E[:3, 3]
        if self.p.extrinsic_subtract_t:
            return (p_dev - t) @ R.T
        return p_dev @ R.T + t

    # ---- step 4 ----
    def cam_to_pixel(self, p_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """返回 (u, v) 单眼坐标与深度 z。z<=0 的点在相机后方, 不可见。"""
        z = p_cam[..., 2]
        safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
        u = self.p.eff_f * p_cam[..., 0] / safe + self.p.eff_cx
        v = self.p.eff_f * p_cam[..., 1] / safe + self.p.eff_cy
        return u, v, z

    def project(
        self, p_world: np.ndarray, head_pos: np.ndarray, head_quat: np.ndarray, eye: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """完整链路。返回 (u_eye, v, z, p_head)。

        u_eye 是单眼局部坐标; 加 1080 (左半) 得到整幅坐标。
        """
        p_head = self.world_to_head(p_world, head_pos, head_quat)
        p_cam = self.head_to_cam(p_head, eye)
        u, v, z = self.cam_to_pixel(p_cam)
        return u, v, z, p_head

    # ---- 眼别 -> 半幅映射 ----
    def eye_of_half(self, half: int) -> int:
        """half: 0=画面左半 (u_full<1080), 1=画面右半。

        返回应该使用哪个外参下标 (0 或 1)。
        """
        if self.p.eye0_is_left_half:
            return half
        return 1 - half

    def u_full(self, u_eye: np.ndarray, half: int) -> np.ndarray:
        return u_eye + EYE_W * half


def make_projector(header, params: CameraParams | None = None) -> Projector:
    return Projector(header.extrinsics, params)


def head_motion_stats(head_pos: np.ndarray, head_quat: np.ndarray, dt: float) -> dict:
    """head 位移跨度与角速度, 用于 plan §1.2b 自检与 §3.2 的误差归属。"""
    span = np.ptp(head_pos, axis=0)
    R = np.stack([quat_to_mat(q) for q in head_quat])
    # 相邻帧相对旋转角
    rel = np.einsum("nij,nkj->nik", R[1:], R[:-1])
    cos = np.clip((np.trace(rel, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    ang = np.degrees(np.arccos(cos))
    return {
        "pos_span_m": span.tolist(),
        "max_dist_from_origin_m": float(np.linalg.norm(head_pos, axis=1).max()),
        "ang_step_deg_median": float(np.median(ang)),
        "ang_step_deg_max": float(ang.max()),
        "ang_vel_deg_s_median": float(np.median(ang) / dt),
        "ang_vel_deg_s_max": float(ang.max() / dt),
        "lin_vel_m_s_max": float(
            (np.linalg.norm(np.diff(head_pos, axis=0), axis=1) / dt).max()
        ),
    }
