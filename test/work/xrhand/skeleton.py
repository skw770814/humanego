"""26 关节的名称与骨骼拓扑。

关节顺序直接来自源码而非推断 —— PICO Unity Integration SDK 的
`PXR_HandTracking.cs` 中 `HandJoint` 枚举:

    JointPalm = 0,  JointWrist = 1,
    JointThumbMetacarpal = 2,  JointThumbProximal = 3,  JointThumbDistal = 4,  JointThumbTip = 5,
    JointIndexMetacarpal = 6,  JointIndexProximal = 7,  JointIndexIntermediate = 8,
    JointIndexDistal = 9,      JointIndexTip = 10,
    JointMiddleMetacarpal = 11, ... JointMiddleTip = 15,
    JointRingMetacarpal = 16,   ... JointRingTip = 20,
    JointLittleMetacarpal = 21, ... JointLittleTip = 25,
    JointMax = 26

TrackingData.cs 就是按 `jointLocations[i]` 的原生下标逐条写出的, 所以该枚举即数据顺序。

拓扑与论文 §II-B 一致: "4 joints on the thumb, 5 joints on each remaining finger,
plus palm and wrist joints" (引文 [17] = OpenXR 规范)。
"""

from __future__ import annotations

NUM_JOINTS = 26

NAMES = [
    "PALM",
    "WRIST",
    # thumb (无 INTERMEDIATE)
    "THUMB_METACARPAL", "THUMB_PROXIMAL", "THUMB_DISTAL", "THUMB_TIP",
    # index
    "INDEX_METACARPAL", "INDEX_PROXIMAL", "INDEX_INTERMEDIATE", "INDEX_DISTAL", "INDEX_TIP",
    # middle
    "MIDDLE_METACARPAL", "MIDDLE_PROXIMAL", "MIDDLE_INTERMEDIATE", "MIDDLE_DISTAL", "MIDDLE_TIP",
    # ring
    "RING_METACARPAL", "RING_PROXIMAL", "RING_INTERMEDIATE", "RING_DISTAL", "RING_TIP",
    # little
    "LITTLE_METACARPAL", "LITTLE_PROXIMAL", "LITTLE_INTERMEDIATE", "LITTLE_DISTAL", "LITTLE_TIP",
]
assert len(NAMES) == NUM_JOINTS

PALM = 0
WRIST = 1
TIPS = (5, 10, 15, 20, 25)

# 5 根手指的 (metacarpal, [chain...]) , 下标即源码枚举值
FINGERS = {
    "thumb": [2, 3, 4, 5],
    "index": [6, 7, 8, 9, 10],
    "middle": [11, 12, 13, 14, 15],
    "ring": [16, 17, 18, 19, 20],
    "little": [21, 22, 23, 24, 25],
}

# 骨骼边: wrist -> 各掌骨, 掌骨 -> ... -> 指尖
EDGES: list[tuple[int, int]] = []
for chain in FINGERS.values():
    EDGES.append((WRIST, chain[0]))
    for a, b in zip(chain[:-1], chain[1:]):
        EDGES.append((a, b))
EDGES.append((PALM, WRIST))

# 掌心多边形 (用于半透明填充, 帮助判断姿态)
PALM_POLY = [WRIST, 6, 11, 16, 21]  # wrist + 4 根非拇指掌骨

# 每根手指的颜色 (RGB)
FINGER_COLORS = {
    "thumb": (255, 64, 64),
    "index": (64, 220, 64),
    "middle": (64, 128, 255),
    "ring": (255, 200, 32),
    "little": (220, 64, 220),
}

# 关节索引 -> 颜色
JOINT_COLOR: dict[int, tuple[int, int, int]] = {PALM: (255, 255, 255), WRIST: (255, 255, 255)}
for fname, chain in FINGERS.items():
    for j in chain:
        JOINT_COLOR[j] = FINGER_COLORS[fname]

# 关节索引 -> 所属手指 (用于按手指统计骨长)
JOINT_FINGER: dict[int, str] = {PALM: "palm", WRIST: "palm"}
for fname, chain in FINGERS.items():
    for j in chain:
        JOINT_FINGER[j] = fname


def bone_lengths(pos):
    """(26,3) -> {边: 长度}, 用于骨长稳定性自检 (plan §五.5)。"""
    import numpy as np

    pos = np.asarray(pos)
    return {f"{NAMES[a]}-{NAMES[b]}": float(np.linalg.norm(pos[a] - pos[b])) for a, b in EDGES}
