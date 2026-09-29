"""解析 trackingData_*.txt。

数据由 XRoboToolkit-Unity-Client (PICO 客户端, tag v1.1.1) 的
Assets/Scripts/TrackingData.cs 写出。

文件结构 (源码 TrackingData.cs + 实测):
  第 1 行: header, 含 notice / timeStampNs / cameraExtrinsics / cameraIntrinsics
  其余行: 每行一个 JSON 记录

字段语义 (源码依据):
  predictTime  = PXR_Enterprise.GetPredictedDisplayTime() * 1000   [微秒]
                 源码注释: "微秒，对应camera录制中帧插入的时间戳"
  timeStampNs  = Utils.GetCurrentTimestamp()  -> (DateTime.UtcNow-1970)ms*1e6
                 [UTC 墙钟纳秒, 非单调]
  Head.pose    = GetPoseStr(PxrSensorState2.pose)  -> "x,y,z,qx,qy,qz,qw"
  Head.status  = PxrSensorState2.status
  Hand.<side>.isActive = HandJointLocations.isActive  [质量等级 0=low 1=high]
  Hand.<side>.HandJointLocations[i].p = "x,y,z,qx,qy,qz,qw"
                                     .s = (ulong)HandLocationStatus  [位掩码]
                                     .r = HandJointLocation.radius   [本数据恒 0]
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

# --- 源码依据: PXR_HandTracking.cs (PICO Unity Integration SDK) ---
NUM_JOINTS = 26

# HandLocationStatus 位掩码 (源码 PXR_HandTracking.cs)
HL_ORIENTATION_VALID = 0x1
HL_POSITION_VALID = 0x2
HL_ORIENTATION_TRACKED = 0x4
HL_POSITION_TRACKED = 0x8
HL_ALL = (
    HL_ORIENTATION_VALID | HL_POSITION_VALID | HL_ORIENTATION_TRACKED | HL_POSITION_TRACKED
)  # = 15


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """四元数 (x,y,z,w) -> 3x3 旋转矩阵。

    数据里的四元数顺序是 xyzw —— 源码 GetPoseStr(Vector3, Quaternion) 按
    Unity Quaternion 的 x,y,z,w 顺序写出。已实测模长 == 1。
    """
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _floats(s: str) -> np.ndarray:
    """解析逗号分隔的浮点串, 容忍外层方括号与空白。

    header 里 cameraExtrinsics / cameraIntrinsics 形如 "[a, b, c]|[d, e, f]",
    Unity 侧 ToString("E16") 输出大写 E 的科学计数法, Python float() 可直接吃。
    """
    s = s.strip()
    if s.startswith("["):
        s = s[1:]
    if s.endswith("]"):
        s = s[:-1]
    return np.array([float(t) for t in s.split(",") if t.strip()], dtype=np.float64)


def parse_pose7(s: str) -> tuple[np.ndarray, np.ndarray]:
    """解析 "x,y,z,qx,qy,qz,qw"。"""
    v = _floats(s)
    if v.size != 7:
        raise ValueError(f"pose 字段应有 7 个数, 实际 {v.size}: {s!r}")
    return v[:3].copy(), v[3:].copy()


@dataclass
class Header:
    notice: str
    time_stamp_ns: int
    # 4x4 行主序, [left, right], 源码 Utils.GetCameraExtrinsicsStrE() -> "left|right"
    extrinsics: np.ndarray  # (2,4,4)
    # 源码 Utils.GetCameraIntrinsicsStrE() -> [cx, cy, fx, fy]
    intrinsics: np.ndarray  # (4,)
    n_bad_lines: int = 0


@dataclass
class Hand:
    """单侧手的一帧数据。"""

    is_active: int
    scale: float
    pos: np.ndarray  # (26,3) world
    quat: np.ndarray  # (26,4) xyzw
    status: np.ndarray  # (26,) uint64 位掩码

    @property
    def quality_high(self) -> bool:
        """isActive 是质量等级: 0=low, 1=high (SDK 注释)。"""
        return int(self.is_active) >= 1

    @property
    def all_tracked(self) -> np.ndarray:
        """每点是否 '位置+朝向均被跟踪且有效' (s == 15)。"""
        return self.status == HL_ALL

    @property
    def valid_mask(self) -> np.ndarray:
        """可直接用于投影的点: 质量等级为 high 且该点 status 四位全置。"""
        if not self.quality_high:
            return np.zeros(NUM_JOINTS, dtype=bool)
        return self.all_tracked


@dataclass
class Record:
    predict_time_us: float
    time_stamp_ns: int
    head_pos: np.ndarray  # (3,)
    head_quat: np.ndarray  # (4,) xyzw
    head_status: int
    left: Hand
    right: Hand
    input_device: int


def _parse_extrinsics(s: str) -> np.ndarray:
    left_s, right_s = s.split("|")
    out = []
    for part in (left_s, right_s):
        m = _floats(part)
        if m.size != 16:
            raise ValueError(f"外参应为 16 个数, 实际 {m.size}")
        out.append(m.reshape(4, 4))
    return np.stack(out)  # (2,4,4)


def _parse_hand(d: dict) -> Hand:
    n = int(d.get("count", NUM_JOINTS))
    if n != NUM_JOINTS:
        raise ValueError(f"关节数应为 {NUM_JOINTS}, 实际 {n}")
    pos = np.zeros((NUM_JOINTS, 3))
    quat = np.zeros((NUM_JOINTS, 4))
    st = np.zeros(NUM_JOINTS, dtype=np.uint64)
    for i, j in enumerate(d["HandJointLocations"][:NUM_JOINTS]):
        p, q = parse_pose7(j["p"])
        pos[i] = p
        quat[i] = q
        st[i] = np.uint64(int(float(j["s"])))
    return Hand(
        is_active=int(d.get("isActive", 0)),
        scale=float(d.get("scale", 1.0)),
        pos=pos,
        quat=quat,
        status=st,
    )


def load(path: str) -> tuple[Header, list[Record]]:
    """读取一个 trackingData_*.txt。"""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    hdr_raw = json.loads(lines[0])
    header = Header(
        notice=hdr_raw.get("notice", ""),
        time_stamp_ns=int(hdr_raw["timeStampNs"]),
        extrinsics=_parse_extrinsics(hdr_raw["cameraExtrinsics"]),
        intrinsics=_floats(hdr_raw["cameraIntrinsics"]),
    )

    records: list[Record] = []
    bad = 0
    for ln in lines[1:]:
        ln = ln.strip()
        if not ln:
            continue
        try:
            d = json.loads(ln)
        except json.JSONDecodeError:
            bad += 1
            continue

        hp, hq = parse_pose7(d["Head"]["pose"])
        hand = d["Hand"]
        records.append(
            Record(
                predict_time_us=float(d["predictTime"]),
                time_stamp_ns=int(d["timeStampNs"]),
                head_pos=hp,
                head_quat=hq,
                head_status=int(d["Head"]["status"]),
                left=_parse_hand(hand["leftHand"]),
                right=_parse_hand(hand["rightHand"]),
                input_device=int(d.get("Input", 0)),
            )
        )

    header.n_bad_lines = bad
    return header, records


def record_arrays(records: list[Record]) -> dict:
    """把记录列表摊平成便于向量化使用的数组。"""
    return {
        "predict_us": np.array([r.predict_time_us for r in records]),
        "ts_ns": np.array([r.time_stamp_ns for r in records], dtype=np.int64),
        "head_pos": np.stack([r.head_pos for r in records]),
        "head_quat": np.stack([r.head_quat for r in records]),
        "head_status": np.array([r.head_status for r in records]),
    }


def summarize(path: str) -> dict:
    """给 plan §1 的自检用: 打印数据的关键统计量。"""
    header, records = load(path)
    a = record_arrays(records)
    dt_pred = np.diff(a["predict_us"])
    dt_ns = np.diff(a["ts_ns"]) / 1e6  # ms

    r0 = records[0]
    return {
        "n_records": len(records),
        "n_bad_lines": header.n_bad_lines,
        "intrinsics_cx_cy_fx_fy": header.intrinsics.tolist(),
        "predict_us_span_s": float((a["predict_us"][-1] - a["predict_us"][0]) / 1e6),
        "ts_ns_span_s": float((a["ts_ns"][-1] - a["ts_ns"][0]) / 1e9),
        "predict_dt_us_median": float(np.median(dt_pred)),
        "predict_dt_us_min": float(dt_pred.min()),
        "predict_dt_us_max": float(dt_pred.max()),
        "ts_dt_ms_median": float(np.median(dt_ns)),
        "ts_dt_ms_min": float(dt_ns.min()),
        "ts_dt_ms_max": float(dt_ns.max()),
        "header_ts_minus_first_record_ms": float(
            (header.time_stamp_ns - a["ts_ns"][0]) / 1e6
        ),
        "head_pos_span": np.ptp(a["head_pos"], axis=0).tolist(),
        "left_isActive": sorted({int(r.left.is_active) for r in records}),
        "right_isActive": sorted({int(r.right.is_active) for r in records}),
        "left_status_uniq": sorted({int(v) for r in records for v in r.left.status}),
        "right_status_uniq": sorted({int(v) for r in records for v in r.right.status}),
        "first_head_pose": r0.head_pos.tolist(),
    }


if __name__ == "__main__":
    import sys

    for p in sys.argv[1:]:
        print(f"=== {p}")
        for k, v in summarize(p).items():
            print(f"  {k}: {v}")
