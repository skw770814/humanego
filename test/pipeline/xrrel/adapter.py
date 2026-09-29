"""输入适配器: 把我们的数据喂成 `ego_relation_policy` 的 `stereo_depth.py` 期望的形状。

用户口径 (原话): 「我的数据是和 ego_relation_policy 用一样的 PICO4ULTRA 采的, 相机参数等
和他一样, 所以物体深度计算用他的骨架。」 —— 所以深度这块**不移植、不重写**, 直接 import
他的模块调用。他那边只有 `run_stereo_depth` 绑死了 HDF5, 其余函数 (几何/校正/SGBM/QA)
的入参本来就是通用的:

    stereo_geometry(file) / _rectification(file, image_size)   把 file 只当**字典**用
    compute_stereo_depth(left_rgb, right_rgb, K, baseline_m, cfg, ...)   纯函数, 无 h5py

于是这里只做三件事:
  1. `load_ego_config()` —— 读他仓库的 configs/default.yaml (参数一个不改);
  2. `StereoShim`      —— 一个支持 `__getitem__` 的小对象, 冒充 h5py.File, 只提供他要用
                          的 4 个键 (`camera/extrinsics_left|right`、`camera/K_left|right`);
  3. `pose7_from_matrix` / `eye_pair_iter` —— 4x4 <-> pose7 无损往返, 以及逐帧读左右眼。

`matrix_from_pose7` (`contracts/se3.py:23`) 是 `[x,y,z,qx,qy,qz,qw] -> R|t` 的**纯转换,
不做坐标轴变换**, 所以我们的 4x4 外参可以无损往返成 pose7, `stereo_geometry` 于是给出
与他 HDF5 路径下**同一个** `T_left_right` (往返误差在 check stage 里断言 <1e-12)。

他的模块顶层 `import h5py`、并转手 import `s1_pico_mode2.pico` (那里 `import pyarrow`),
所以 pipeline/.venv 里必须装上这两个包 —— 我们一行都不用, 纯粹为了过 import 链。
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np

from . import (
    EGO_CONFIG,
    EGO_SRC,
    EYE_CX,
    EYE_CY,
    EYE_F,
    EYE_H,
    EYE_W,
    FULL_H,
    FULL_W,
)

# 允许重复 import (tools/rel_object.py 的各 stage 会各自调用)
if str(EGO_SRC) not in sys.path:
    sys.path.insert(0, str(EGO_SRC))

import cv2  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from ego_relation.config import ProjectConfig, load_config  # noqa: E402
from ego_relation.contracts.manifest import file_sha256  # noqa: E402
from ego_relation.contracts.se3 import matrix_from_pose7  # noqa: E402
from ego_relation.s2_object_relations import stereo_depth as sd  # noqa: E402

# 单眼针孔内参矩阵 —— 他 HDF5 里的 camera/K_left|K_right 就是这个 (3,3)。
# 他的两个 K 是分开存的, 但 PICO 双目是对称的 (同一 fx/fy, cx/cy 各自在半幅内),
# 我们的 header 只给一组 (全幅 2160 坐标), 拆到单眼就是这个矩阵; check stage 里
# 对着 header 断言 `intrinsics[2]/2 == EYE_F`。
EYE_K = np.array(
    [[EYE_F, 0.0, EYE_CX], [0.0, EYE_F, EYE_CY], [0.0, 0.0, 1.0]], dtype=np.float64
)

# 他 run_stereo_depth 里用于汇总 QA 的两栏, 数据源是他 HDF5 的 camera/stereo_valid
# 与 camera/stereo_pair_delta_ns —— 我们的 TXT 没有对应字段。为了仍然能**原样调用**
# 他的 _stereo_qa_status (它直接下标取这两项并做比较), 用「不构成否决」的值填上,
# 同时在 qa_notes 里显式记明来源, 不让它冒充真实测量。
QA_NO_SOURCE_VALUE = {"stereo_valid_ratio": 1.0, "pair_delta_max_ms": 0.0}
QA_NO_SOURCE_NOTE = (
    "stereo_valid_ratio / pair_delta_max_ms 的数据源是他 HDF5 的 camera/stereo_valid 与 "
    "camera/stereo_pair_delta_ns, 我们的 txt header 里没有这两项, 故按「不构成否决」填 "
    "1.0 / 0.0 以复用他的 _stereo_qa_status, 它们**不代表测量值**, 不参与判定。"
)


def load_ego_config(num_disparities: int | None = None) -> tuple[ProjectConfig, list[str]]:
    """读他的 configs/default.yaml (原样), 返回 (cfg, 偏离说明)。

    `num_disparities` 默认 None = **用他的 128**。只有明确要看更近的深度时才覆盖,
    返回的 notes 会写进报告, 免得事后不知道跑的是哪套参数。

    `distance_covered_m` = f·B / num_disparities 是他那套参数能覆盖的最近距离,
    日志里必须打印 —— 128 视差只覆盖 Z >= 0.344 m。
    """
    cfg = load_config(EGO_CONFIG)
    notes: list[str] = []
    if num_disparities is not None and int(num_disparities) != int(cfg.depth.num_disparities):
        notes.append(
            f"DEVIATION: depth.num_disparities 从他 yaml 的 {cfg.depth.num_disparities} "
            f"覆盖为 {int(num_disparities)}"
        )
        cfg = dataclasses.replace(
            cfg, depth=dataclasses.replace(cfg.depth, num_disparities=int(num_disparities))
        )
    return cfg, notes


def nearest_covered_depth_m(cfg: ProjectConfig, baseline_m: float) -> float:
    """这套 num_disparities 能测到的**最近**距离 (视差上限对应的 Z)。"""
    return float(EYE_F * baseline_m) / float(cfg.depth.num_disparities)


def focal_baseline_mm(cfg: ProjectConfig, baseline_m: float) -> float:
    return float(EYE_F) * float(baseline_m)


# ---------------------------------------------------------------- pose7 往返


def quat_xyzw_from_matrix(rotation: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(np.asarray(rotation, dtype=np.float64)).as_quat()


def pose7_from_matrix(transform: np.ndarray) -> np.ndarray:
    """4x4 -> [x, y, z, qx, qy, qz, qw], 与他 matrix_from_pose7 互逆。"""
    transform = np.asarray(transform, dtype=np.float64)
    q = quat_xyzw_from_matrix(transform[:3, :3])
    return np.concatenate([transform[:3, 3], q])


def pose7_roundtrip_error(transform: np.ndarray) -> float:
    """往返误差, check stage 用它断言无损 (要求 <1e-12)。"""
    back = matrix_from_pose7(pose7_from_matrix(transform))
    return float(np.max(np.abs(back - np.asarray(transform, dtype=np.float64))))


class StereoShim:
    """冒充 h5py.File 的只读字典 —— 只实现他 stereo_geometry/_rectification 用到的取法。

    他那边是 `file["camera/extrinsics_left"][:]` 这种, 所以我们返回的必须是**数组**,
    不是 Dataset; 于是他拿到的东西行为完全一致。
    """

    def __init__(self, extrinsics: np.ndarray, K_left: np.ndarray, K_right: np.ndarray):
        extrinsics = np.asarray(extrinsics, dtype=np.float64)
        if extrinsics.shape != (2, 4, 4):
            raise ValueError(f"extrinsics 应为 (2,4,4), 实际 {extrinsics.shape}")
        self._data = {
            "camera/extrinsics_left": pose7_from_matrix(extrinsics[0]),
            "camera/extrinsics_right": pose7_from_matrix(extrinsics[1]),
            "camera/K_left": np.asarray(K_left, dtype=np.float64),
            "camera/K_right": np.asarray(K_right, dtype=np.float64),
        }

    def __getitem__(self, key: str) -> np.ndarray:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def keys(self):
        return self._data.keys()

    def get(self, key: str, default=None):
        return self._data.get(key, default)


def make_shim(header) -> StereoShim:
    """我们的 io_tracking.Header (extrinsics (2,4,4) + intrinsics) -> 他的 file。"""
    _assert_header_intrinsics(header)
    return StereoShim(header.extrinsics, EYE_K, EYE_K)


# 四项内参不变量允许的偏差 (px)。取 1e-3 是因为标称值本身是四舍五入到 3 位的:
# 数据 fy=686.8680648828482 -> 标称 686.868 (差 6.5e-5), 见 camera.py:11。
# 1e-3 足够松到不误报, 又足够紧到能抓住「intrinsics 顺序变了 / 换了别的设备」。
INTRINSIC_TOLERANCE_PX = 1e-3


def _assert_header_intrinsics(header) -> dict:
    """header 的 `cameraIntrinsics = [cx, cy, fx, fy]` 是**全幅 2160 坐标**下的值。

    断言的是四条**真正**的不变量 (不是精确等式 —— 上一轮把标称关系当成精确等式, 一跑就挂):

      1. `fy` 就是单眼焦距           (垂直半幅没被翻倍, 因为取内参时 height 传的就是单眼高 810)
      2. `fx / fy == 2`              (水平方向被算成两倍 —— 因为取内参时 width 传的是双眼总宽
                                      2160; 注意是 **≈2** 不是 =2, 实测 1.99990101, camera.py:15)
      3. `cy == FULL_H/2 - 0.5`      (单眼高直接给的主点)
      4. `cx == FULL_W/2 - 0.5`      (全幅主点; **单眼主点是重新算的** `EYE_W/2-0.5 = 539.5`,
                                      它 **不等于** cx/2 = 539.75 —— 差 0.25 px, 这是本轮踩的坑)

    返回实测差值 (dict), 给 check stage 打印用。
    """
    cx, cy, fx, fy = (float(v) for v in np.asarray(header.intrinsics).reshape(-1)[:4])
    expected_cx = FULL_W / 2.0 - 0.5
    expected_cy = FULL_H / 2.0 - 0.5
    focal_ratio = fx / fy if fy else float("nan")

    if abs(fy - EYE_F) > INTRINSIC_TOLERANCE_PX:
        raise AssertionError(
            f"header fy={fy!r} 与单眼焦距 {EYE_F} 差 {fy - EYE_F:+.3e} px "
            f"(容差 {INTRINSIC_TOLERANCE_PX}) —— cameraIntrinsics 的顺序不是 [cx,cy,fx,fy]?"
        )
    if abs(focal_ratio - 2.0) > INTRINSIC_TOLERANCE_PX:
        raise AssertionError(
            f"header fx/fy={focal_ratio!r} 不是 2 (fx={fx!r}, fy={fy!r}) —— "
            f"「宽度按双眼总宽算, 所以 fx 翻倍」这个前提不成立了, 别再用 fx/2 当单眼焦距"
        )
    if abs(cy - expected_cy) > INTRINSIC_TOLERANCE_PX:
        raise AssertionError(f"header cy={cy!r} != FULL_H/2-0.5 = {expected_cy!r}")
    if abs(cx - expected_cx) > INTRINSIC_TOLERANCE_PX:
        raise AssertionError(f"header cx={cx!r} != FULL_W/2-0.5 = {expected_cx!r}")

    return {
        "header_cx": cx,
        "header_cy": cy,
        "header_fx": fx,
        "header_fy": fy,
        "fx_over_fy": focal_ratio,
        "fy_minus_nominal_f": fy - EYE_F,
        "half_fx_minus_nominal_f": fx / 2.0 - EYE_F,
        "cx_minus_expected": cx - expected_cx,
        "cy_minus_expected": cy - expected_cy,
        "eye_cx_minus_half_cx": EYE_CX - cx / 2.0,
        "tolerance_px": INTRINSIC_TOLERANCE_PX,
        "fed_K": EYE_K.tolist(),
    }


def stereo_source_hash() -> dict:
    """他那份 stereo_depth.py 的 sha256 —— 证明「直接调他的骨架」是只读调用, 不是改写。"""
    path = Path(sd.__file__)
    return {"path": str(path), "sha256": file_sha256(path)}


# ---------------------------------------------------------------- 读帧


def eye_pair_iter(mp4: str | Path):
    """逐帧产出 (left_rgb, right_rgb), 各自 1080x810 RGB uint8。

    用仓库既有的 `xrhand.video.iter_frames` (ffmpeg 管道, 整幅 2160x810 RGB),
    再切两半 —— 左半是 eye0, 与 mask / 深度的像素系一致 (mask 只画在左半)。
    """
    from xrhand.video import iter_frames

    for frame in iter_frames(str(mp4), 2 * EYE_W, EYE_H):
        left = np.ascontiguousarray(frame[:, :EYE_W])
        right = np.ascontiguousarray(frame[:, EYE_W:])
        yield left, right


def read_eye_pair(mp4: str | Path, frame: int) -> tuple[np.ndarray, np.ndarray]:
    """只取第 frame 帧的左右眼 (静帧用; 逐帧循环请用 eye_pair_iter, 别在循环里调它)。"""
    for index, pair in enumerate(eye_pair_iter(mp4)):
        if index == frame:
            return pair
    raise IndexError(f"{mp4} 没有第 {frame} 帧")


def to_gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def video_size(mp4: str | Path) -> tuple[int, int]:
    from xrhand.video import probe

    info = probe(str(mp4))
    return int(info["width"]), int(info["height"])
