"""PICO ego 采集 -> LeRobot 的 4-step pipeline。

    step1  对齐 + 5 关键点 -> 左眼相机下的指尖中点位姿      out/<stem>/step1/
    step2  DINO+SAM2 物体标定 -> 物体位姿 -> state/action   out/<stem>/step2/
    step3  手 mask + LaMa 修复 + piper 夹爪合成             out/<stem>/step3/
    step4  裁剪对齐 + 写出 LeRobot v2.1 数据集              out/lerobot/<name>/

`test/pipeline` 是自包含的实现与编排根目录: 几何/分割/深度/位姿/修复
直接使用同目录下的 `xrrel` / `xrhand` / `xrseg` / `tools`，运行时不依赖
`test/pipeline`。step 之间只通过 `out/<stem>/<step>/` 的产物耦合。

坐标系: 观测与 `observation.state` 只有**左眼 (eye0 / camera0)** 一个相机系, 观测也只用
左眼 1080x810; 但 `action` / `observation.action_reference_tcp` 的参考系是 **PICO OpenXR 右手世界系**
—— runtime 报的 tracking space, **段内不动**, 而且是**逐段各自一个** (见 `ACTION_REFERENCE_FRAME`)。
末端 = HumanEgo 的 MidpointFrameBuilder (两指尖中点), **不是** TCP/手腕语义。
"""

from __future__ import annotations

import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent          # test/pipeline/xrpipe
PIPE = PKG.parent                              # test/pipeline
WORK = PIPE                                     # test/pipeline (兼容迁入模块的旧常量名)
DATA = PIPE.parent                              # test/
OUT = PIPE / "out"

if str(WORK) not in sys.path:
    sys.path.insert(0, str(WORK))

# 观测/几何共用的常量。EYE_* 与 xrrel/__init__.py 一致 (左半 = eye0)。
EYE_W, EYE_H = 1080, 810
FULL_W, FULL_H = 2160, 810
INSTANCE_ID = "obj1"
# 物体实例 id 的规范顺序 (obj1..objN) —— 与 `xrseg.common.INSTANCE_IDS` / `xrrel.INSTANCE_IDS`
# 同一张表, 也是 `--prompt "A|B"` 里 `|` 分隔出来的顺序 (第 i 段提示 = obj{i+1})。
MAX_OBJECTS = 3
INSTANCE_IDS = tuple(f"obj{i}" for i in range(1, MAX_OBJECTS + 1))
FPS = 30.0

# `--prompt "A|B"` 至少要 2 段提示词 —— 这是**任务口径** (「抓物体1、放到物体2上」),
# 与 action 无关 (action 现在与世界系里的绝对位姿有关, 不参照任何物体)。
# 落点在 `cli._check_prompts`: 开跑前就报, 不浪费几十分钟的 DINO+SAM2。
MIN_OBJECT_PROMPTS = 2

# action / `observation.action_reference_tcp` 的语义标记 —— **唯一一处定义**, cli 的防呆、
# step2 的 report、step4 的 report 与 info.json 都引用它, 免得四份字面量各自漂移。
#   ACTION_STORAGE       与 s4 的 `lerobot.py:409` 逐字一致: 存的是**绝对**位姿
#   ACTION_REFERENCE_FRAME  `pico_world_openxr` = runtime 报的 PICO tracking space, 段内不动;
#                        相对动作 (delta) 不进数据集, 由 openpi 训练期按
#                        `training_action_transform: inv(T_current) @ T_absolute_target` 现算
# 这两个 token 一起构成盘上产物的**语义指纹**: 旧的相机系产物是 `absolute` + `camera0 (…)`,
# 只比 `ACTION_STORAGE` 分辨不出来, 所以两处都必须精确相等地比 (见 cli._check_stored_action)。
ACTION_STORAGE = "absolute"
ACTION_REFERENCE_FRAME = "pico_world_openxr"
ACTION_COORDINATE_SYSTEM = "openxr_rh_x_right_y_up_z_back"
ACTION_SOURCE_COORDINATE_SYSTEM = "unity_lh_x_right_y_up_z_forward"

# 手在世界系里的逐帧运动门。**只统计, 不否决** —— 它抓不到 recenter: 实测 49 段里 9 段越线,
# 而逐条核对那 9 段在手在**相机系**里跳了同样大的量 (recenter 的前提正是「相机系里看不见」,
# 所以这 9 段全是真实手部运动)。真正的守卫见 `MAX_RECENTER_STEP_*`。
# 顺带: 这两个数也早已不反映观测 —— 逐段 max 的中位就有 30.4 mm、最大 173.0 mm。
MAX_WORLD_STEP_TRANSLATION_M = 0.05
MAX_WORLD_STEP_ROTATION_DEG = 15.0

# recenter 守卫的门: `T_camera0_world` **自身**的逐帧跳变。录制中途一旦 recenter, runtime 把
# 整个世界系左乘一个 J, 于是 `T_camera0_world -> T_camera0_world @ J^-1`, 在这里直接可见
# (而 `T_camera0_midpoint = T_camera0_world @ T_world_midpoint` 逐字不变, 几何自检照样通过)。
# 这个量只由头部位姿决定, 与手部跟踪无关 (实测 49 段无填充/无效值, 手未跟踪区间仍平滑):
# 逐帧位移 max 25.4 mm、转角 max 10.0°, 而 recenter 跳变量级 ~0.4 m —— 取 0.10 m / 45°,
# 比观测噪声大 4 倍、比 recenter 小 4 倍。
MAX_RECENTER_STEP_TRANSLATION_M = 0.10
MAX_RECENTER_STEP_ROTATION_DEG = 45.0

# 抓握锁存的门 (米): 只有「夹爪闭合 ∧ 手原点 ↔ 物体位姿原点 < 这个值」才锁存。
# 这就是参考 `perception.latch_distance_m` 那个归属门 (`s2_object_relations/encoding.py:246`
# 的「最近 + 未被占用 + ≤阈值」), 只是把值从参考默认的 0.20 收到 0.05。
# 别和 `perception.grasp_distance_m = 0.035` 搞混: 那个量的是**两指尖**距 (手捏没捏上),
# 本 pipeline 一点没读 (开闭来自 `xrhand/gripper.py` 自己的阈值)。
LATCH_DISTANCE_M = 0.05

# 观测源的两种选法 (`--observation`): 帧从哪来。只影响视频, 不碰任何标签列。
#   step3  用 `step3/observation/%05d.png` (手已修复 + piper 夹爪) —— 默认, 就是原来的行为
#   step2  直接解源 `CameraRecord_<stem>.mp4` 左半 (原始手、无夹爪) —— 于是 step3 可整步跳过
OBSERVATION_SOURCES = ("step2", "step3")

# 默认的 mask 提示词: 默认只 mask 手, `--with-arm` 时才连手臂 (与 work 里那版一致)。
MASK_PROMPT_HANDS = "human hands ."
MASK_PROMPT_HANDS_ARMS = "human arms . human hands ."

# 默认标定随 pipeline 分发；lag 产物由 step1 自动生成到 OUT。
DEFAULT_CALIB = PIPE / "config" / "calib.json"


def bootstrap() -> None:
    """把自包含的 pipeline 与 `ego_relation_policy/src` 放上 sys.path。

    `ego_relation` 只在 step2 的相对位姿与 step4 的命名/统计里用, import 它需要
    `src` 在路径上; `xrrel` 的模块级常量已经替我们找到了仓库位置。
    """
    if str(WORK) not in sys.path:
        sys.path.insert(0, str(WORK))
    from xrrel import EGO_SRC

    if str(EGO_SRC) not in sys.path:
        sys.path.insert(0, str(EGO_SRC))


def tool_module(name: str):
    """按文件路径载入 `pipeline/tools/<name>.py` (`tools/` 不是包, 不能 import)。

    载入一次就缓存进 `sys.modules` —— 这几个脚本各自都做 `sys.path.insert` 与
    重型 import, 重复 exec 既慢又容易出两套模块状态。
    """
    import importlib.util

    key = f"xrpipe_tool_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, WORK / "tools" / f"{name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"载入不了 {WORK / 'tools' / f'{name}.py'}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


def discover_stems() -> list[str]:
    """`test/CameraRecord_<stem>.mp4` 的 stem, 排序后返回 (episode_index 按此定序)。"""
    return sorted(p.stem[len("CameraRecord_"):] for p in DATA.glob("CameraRecord_*.mp4"))


def resolve_stems(spec: str | None) -> list[str]:
    """`--stems a,b` 或 `--all`/None。"""
    if not spec or spec == "all":
        stems = discover_stems()
    else:
        stems = [s.strip() for s in spec.split(",") if s.strip()]
    if not stems:
        raise SystemExit(f"没有找到采集: {DATA}/CameraRecord_*.mp4")
    for stem in stems:
        if not (DATA / f"CameraRecord_{stem}.mp4").is_file():
            raise SystemExit(f"缺 {DATA / f'CameraRecord_{stem}.mp4'}")
    return stems
