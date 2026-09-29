"""物体位姿: 参考帧 PCA 定向 + 逐帧对参考帧做鲁棒刚体拟合 + 门控/平滑 + 抓取锁存。

用户定的口径: **参考帧是 SAM2 分割的起始帧 (111342 = 帧 50), 不是第 0 帧。**
首帧(即参考帧)用点云 PCA 定初始姿态, 之后每帧把**参考帧的点云**刚体拟合到当前帧的
点云上, 得到 `T_camera0_object`。

出处 (照抄的部分都标了行号, 直接调用的部分标了函数):

  - 参考帧定向: HumanEgo `preprocess/OrientAnything.py:149-258` `estimate_frame_pca2`
    的 **is_anchor 分支** —— 取 PCA 三主轴 (按方差降序), `y = v2` 且翻到朝下
    (`y·(0,-1,0) <= 0`; 各向异性 > 0.15 时) `x = v1` 翻到朝相机右侧 (`x·(1,0,0) >= 0`,
    否则退化成把 cam_right 对 y 正交化), `z = x × y`, 最后 `R = [x|y|z]` 并修 det。
    第 50 帧就是"锚"帧, 所以只走 is_anchor 这一支。
  - 刚体拟合: ego_relation_policy `s2_object_relations/stereo_fusion.py:49-73`
    `_rigid_fit` (Kabsch/SVD) 与 `robust_rigid_fit(threshold_m)` (按残差剔外点再拟一次)。
    **直接 import 调用**, 不重写。
  - 平滑: 同文件 `_smooth_pose` / `_adaptive_translation_alpha` /
    `_adaptive_translation_measurement` / `_rotation_step_deg` —— 也直接调用。
  - 门控阈值与平滑系数: 全部读**他的** `configs/default.yaml` 的 `perception` 段
    (`maximum_object_translation_step_m=0.04`、`maximum_object_rotation_step_deg=25.0`、
    `object_translation_smoothing=0.45`、`object_translation_median_window=3`、
    `object_rotation_smoothing=0.20`、`minimum_pose_inlier_ratio=0.50`)。
  - 抓取锁存: `s2_object_relations/encoding.py:167` `latch_object_poses` —— 锁存时刻记
    `T_hand_object = inv(T_hand) @ T_obj`, 之后握持中 `T_obj := T_hand @ T_hand_object`。
    触发信号用我们 step1 **已有的**二值爪状态 `G_CLOSED` (判据见 xrhand/gripper.py)。
    **三处必须照抄的细节**:
      * 判锁存要在**本帧位姿算完之后** —— 他是先用本帧 visual pose 更新 `current[object]`,
        再用它锁存; 我原先写在循环开头, 读到的 `valid[i]/T[i]` 还是初始值, 于是锁存
        **一次都没触发**。 (这是照抄的那一处。)
      * 锁存前有一道 `latch_distance_m` 的门 (手离物体位姿原点太远就不锁) —— 没有它,
        物体不可见时会把一个陈旧位姿硬绑到手上, 等于让它瞬移。**这道门在本文件里被用在
        两条路上, 且值由调用方定** (见下)。
      * **握持期间一律手推, 不跑拟合** —— 这条照 HumanEgo-main 的原样 (用户 2026-09-21 指定;
        见下面「参照实现」一节)。`latched[i]=True` 的含义是「本帧位姿来自手推, 不是测量」。

**参照实现: HumanEgo-main 的 "Latch & Propagate"** (用户 2026-09-21 指定锁存期间照它, 不做观测优先)。
同一套机制他写了四处, 语义一致: `preprocess/DatasetGen.py:462-498`、`preprocess/RobotDatasetGen.py:527-584`、
`training/FlowMatchingDataloader.py:541`、`inference/run_inference.py:155-207`:

  * 闭合 (他 `grasp > 0.5`) 且**距手 < 0.20 m** 时记 `T_lock_h2obj = inv(T_hand) @ T_obj`,
    **只在建立那一帧算一次**, 此后**不再重锚**;
  * 握持期间 `T_obj := T_hand @ T_lock_h2obj` —— **纯前向运动学直接覆盖视觉位姿**, 不融合、
    不平滑、不跑拟合、不设门; 张开即清空锁存, 下一帧回到视觉。
    (`FlowMatchingDataloader.py:11` 把用意写明: "Freezes T_obj_in_hand upon grasp to fix visual
    occlusion jitter via FK.")
  * **他的 hand 就是拇指-食指指尖中点系, 与我们 step1 的中点系同构** ——
    `preprocess/AriaHandsTypes.py:82-129` `MidpointFrameBuilder.build`: 原点 = 指尖中点,
    `x = index_base - thumb_base` (MCP 基准), `y` = (两 MCP 中点 - 手腕) 对 x 正交化, `z = x×y`;
    `preprocess/DatasetGen.py:136-137` `_get_hand_pose_world` 读的正是
    `midpoint_translation_opt_world` / `midpoint_orientation_opt_world`。
    所以"指尖中点与物体不刚性、他的 TCP 才刚性"这个前提**对 HumanEgo-main 不成立**
    (它对 `ego_relation_policy` 的 `T_tcp_*` 才成立), 刚性假设两边其实相同。
  * `0.20 m` 那道距离门也是他的 (`run_inference.py:198`、`DatasetGen.py:482`) —— 我们的
    `latch_distance_m` 就是它。

**「夹爪闭合 ∧ 距离 < 5 cm 才锁存」+「同一时刻至多一个物体被锁存」** (用户 2026-09-23 定的规则):

  1. 距离量 = **手位姿原点 (两指尖中点) ↔ 物体位姿原点**, 就是代码里现成那个
     `norm(T_obj[:3,3] - hand[:3,3])`。门值默认仍读他的 `perception.latch_distance_m`,
     由调用方 (pipeline 传 0.05, `xrpipe.LATCH_DISTANCE_M`) 收小 —— 这是同一个机制换个数,
     不是新机制。
  2. **两条路都设门**: 建立 (`finish_frame` 的 accepted 分支, 原先**无门**) 与就地锚定
     (无测量、退到保持位姿那一条)。已建立之后的"每有测量就重锚"不再单独设门 —— 它走不到
     没有门的那一步, 因为建立本身已经被门挡过。
  3. **归属 (`blocked` 形参)**: 多物体时调用方按物体顺序逐个跑, 把**前面物体已经锁存的帧**
     以 `blocked` 传进来 —— 那些帧里本物体不得锁存, 且已建立的锁存要立即释放。于是
     `latched` 在任何一帧上**至多一个物体为真**, 物体不会被一起冻结到手上 (这就是参考
     `encoding.py:208-256` 里 `claimed` 集合的作用, 我们把它放在逐物体的调用之间)。
     `blocked` 的优先级是**物体顺序**而不是"最近优先": 门只有 5 cm, 手心里同时有两个物体
     各自距指尖中点 5 cm 以内的情况实际不存在, 所以两种选法落到同一帧上几乎总是同一个物体。

`latch=False` = 只做测量/门控/平滑: 两条锁存路都不走, `latch_pose` 永远是 `None`,
握持期间也照常跑 ICP —— 给"要一份纯测量流"的调用方用。

本文件**只保留**他这套语义, 不额外加"锁存期间也看测量"的分支。唯一属于我们自己的是
**解握那几帧的账**: 握持期间仍然维护 `current` / `recent_translations` (它们是"我们当前认为物体
在哪", 不是"一次测量"), 否则解握首帧的 ICP 种子与平移中值窗口会停在一个 100 mm 外的陈旧位姿上;
而 `last_measurement` / `last_measurement_pos` **故意不动** —— `gap` 随握持时长增长、门预算
`0.04 x min(gap,5)` 随之放宽, 这正是他 `stereo_fusion.py` 遮挡门控的语义, 收紧它反而会白丢解握首帧。

门控的**顺序与语义**照 `stereo_fusion.py:328-365` 一步不差, 两处容易写错的地方:

  * 内点比例不够 / 平移步长超限 -> **整帧丢弃**, 沿用上帧位姿;
  * 旋转步长超限 -> **只**把旋转按住不动 (取上帧朝向), 平移测量**继续用**, 不是整帧丢弃。

**三个适配点 (必须知道)**:

  1. 他是靠 CoTracker 跟踪的同一批关键点拿对应关系; 我们不入 CoTracker (重、要 GPU 与权重),
     改成 **point-to-point ICP** —— 对应关系由"参考点云经当前估计变换过去后取最近邻"给出
     (scipy cKDTree)。语义仍是"对参考帧做鲁棒刚体拟合", 刚体求解用的还是他的
     `robust_rigid_fit`。ICP 的迭代门限用他 `robust_rigid_fit` 的默认 `threshold_m=0.035`
     (与 `relations.grasp_distance_m` 同量级)。
  2. 他的 `keypoint_motion_px` (决定平移 EMA 的响应速度) 我们拿不到同款量, 用 **mask 质心的
     像素位移** 代替 —— 只在"与上次测量紧邻"的那一帧算 (他去程里也是 `measurement_gap == 1`
     才算, 否则保持 NaN 走基础 alpha), 其余帧原样传 NaN 给他的
     `_adaptive_translation_alpha`。
  3. ICP 的**捕获半径**是固定的 `ICP_THRESHOLD_M = 0.035 m`, 而参考点云的 rms 半径只有约
     19 mm —— 种子偏差超过约 3.5 cm 就一对都配不上, 一旦丢跟就**永久**丢跟 (种子不再更新)。
     门控那边给 `min(gap,5)` 帧的位移预算是 `0.04 x 5 = 0.20 m`, 两者口径不一致。
     所以原路径失败的帧会走 `recover_icp`: **质心预对齐 + 按同一个预算放大的粗门限**,
     再用名义容差精配一遍 (报出来的内点比例仍是 0.035 m 下的, 门控照旧复核)。
     只在失败帧上走, 已经跑对的帧逐位不变; 是否走过记在 `recovered` 里。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from . import INSTANCE_ID, RelPaths
from .adapter import load_ego_config
from .lift import _runs, cloud_at
from ego_relation.contracts.se3 import invert, transform_points
from ego_relation.s2_object_relations import stereo_fusion as sf

# ICP 的对应点距离门限 (米)。取他 robust_rigid_fit 的默认阈值。
ICP_THRESHOLD_M = 0.035
ICP_ITERATIONS = 5
ICP_MIN_PAIRS = 30
# 一次刚体拟合至少要有这么多个**不同**的目标点参与。一个目标点被反复匹配也能凑够
# ICP_MIN_PAIRS 对数, 并使残差恒 0 -> ratio 显示 1.0, 但那不是一次测量。
ICP_MIN_UNIQUE_TARGET = 10

CAM_UP = np.array([0.0, -1.0, 0.0])  # OpenCV: Y 向下, 所以"上"是 -Y
CAM_RIGHT = np.array([1.0, 0.0, 0.0])
ANISOTROPY_MIN = 0.15  # HumanEgo OrientAnything.py 里判"够不够细长"的阈值


def pca_frame(points: np.ndarray) -> tuple[np.ndarray, dict]:
    """参考帧的点云 -> 物体系 (HumanEgo estimate_frame_pca2 的 is_anchor 分支)。

    返回 (T_camera0_object (4,4), info)。info 里带特征值与选了哪条分支, 便于复核。
    """
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < 3:
        raise ValueError(f"点云只有 {points.shape[0]} 个点, 定不了姿态")
    center = points.mean(axis=0)
    centered = points - center
    covariance = centered.T @ centered
    evals, evecs = np.linalg.eigh(covariance)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    v1, v2 = evecs[:, 0], evecs[:, 1]

    # y = 中间方差轴, 翻到"朝下"
    y_axis = v2.copy()
    if float(np.dot(y_axis, CAM_UP)) > 0.0:
        y_axis = -y_axis

    anisotropy = float((evals[0] - evals[1]) / (evals[0] + 1e-12))
    if anisotropy > ANISOTROPY_MIN:
        x_axis = v1.copy()
        if float(np.dot(x_axis, CAM_RIGHT)) < 0.0:
            x_axis = -x_axis
        method = f"anchor_pca (aniso:{anisotropy:.2f})"
    else:
        projected = CAM_RIGHT - float(np.dot(CAM_RIGHT, y_axis)) * y_axis
        x_axis = projected / (np.linalg.norm(projected) + 1e-12)
        method = f"anchor_symmetric (aniso:{anisotropy:.2f})"

    x_axis = x_axis - float(np.dot(x_axis, y_axis)) * y_axis
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-12)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-12)

    rotation = np.stack([x_axis, y_axis, z_axis], axis=1)
    if np.linalg.det(rotation) < 0.0:
        x_axis = -x_axis
        rotation = np.stack([x_axis, y_axis, z_axis], axis=1)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = center
    return transform, {
        "method": method,
        "pca_evals": [float(v) for v in evals],
        "anisotropy": anisotropy,
        "centroid": [float(v) for v in center],
    }


def icp_to_reference(
    canonical: np.ndarray,
    target: np.ndarray,
    init: np.ndarray,
    *,
    iterations: int = ICP_ITERATIONS,
    threshold_m: float = ICP_THRESHOLD_M,
    min_pairs: int = ICP_MIN_PAIRS,
    min_unique_target: int = ICP_MIN_UNIQUE_TARGET,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """把参考点云 (canonical, 已在物体系) 刚体拟合到当前帧点云 (target, 相机系)。

    `canonical -> target` 的最近邻当对应关系, 求解调用他的 `robust_rigid_fit`。
    返回 (T_camera0_object, 逐点残差 (N,), 内点比例, fitted)。

    内点比例**恒按 `ICP_THRESHOLD_M` 算** (不是按 `threshold_m`) —— 粗配阶段传大半径进来时,
    报出来的仍是「名义容差下能配上多少」, 跨调用可比, 也是门控判据用的那个量。

    `fitted` 是**诚实性护栏**: 只在「拟合真的成功更新过」时为 True —— 满足 `min_pairs` 对
    且来自 `min_unique_target` 个**不同**的目标点。目标点太少时 (例如整帧只有 3 个点),
    一个点被反复匹配也能凑够对数、残差恒 0、ratio 显示 1.0, 而那不是一次测量。
    `fitted=False` 表示返回的 transform **就是传进来的 init** (没动过), 调用方别把它当成测量。
    """
    canonical = np.asarray(canonical, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    tree = cKDTree(target)
    transform = np.asarray(init, dtype=np.float64).copy()
    fitted = False
    for _ in range(max(1, iterations)):
        moved = transform_points(transform, canonical)
        distance, index = tree.query(moved, k=1)
        keep = distance <= threshold_m
        if int(keep.sum()) < min_pairs:
            break
        if int(np.unique(index[keep]).size) < min_unique_target:
            break
        candidate, _, _ = sf.robust_rigid_fit(
            canonical[keep], target[index[keep]], threshold_m=threshold_m
        )
        transform = candidate
        fitted = True
    moved = transform_points(transform, canonical)
    distance, _ = tree.query(moved, k=1)
    residual = distance
    inlier_ratio = float(np.mean(residual <= ICP_THRESHOLD_M))
    return transform, residual, inlier_ratio, fitted


def rms_radius(points: np.ndarray) -> float:
    """点云绕质心的 rms 半径 (米) —— 物体自身的尺度。给捕获半径封顶用。"""
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] == 0:
        return 0.0
    centered = points - points.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))


def recover_icp(
    canonical: np.ndarray,
    target: np.ndarray,
    init: np.ndarray,
    *,
    coarse_m: float,
    iterations: int = ICP_ITERATIONS,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """丢跟后的**接回**: 质心预对齐 + 粗门限 -> 精门限 两段拟合。

    只在原路径失败 (`fitted=False` 或内点比例不够) 的帧上调用, 所以已经跑对的帧**逐位不变**。

    为什么需要它 (本轮实测): `icp_to_reference` 用固定 `ICP_THRESHOLD_M = 0.035` 找对应点,
    而参考点云的 rms 半径只有 18.8 mm —— 种子偏差超过约 3.5 cm 就**一对都配不上**,
    循环第一轮就 break、transform 停在 init、ratio = 0 -> 整帧丢弃 -> `current` 不更新 ->
    下一帧还是同一个旧种子 -> **永久丢跟**。而门控本身给 `min(gap,5)` 帧的位移预算是
    `0.04 x 5 = 0.20 m`, 即「门控允许 20 cm, 但 ICP 连 3.5 cm 都找不到」——两者口径不一致。

    所以这里把**捕获半径也按同一个预算放大** (`coarse_m` 由调用方按 gap 算), 并用质心对齐
    先吃掉纯平移那部分; `coarse_m` 由调用方用 `2 x rms_radius` 封顶 —— 超过物体自身尺度的
    匹配没有意义, 不允许到别的表面上去。粗配之后**再在名义容差下精配一遍**,
    报出来的 ratio 仍是 0.035 m 下的, 门控照旧复核, 粗配不会把离谱的结果偷渡进来。
    """
    target = np.asarray(target, dtype=np.float64)
    canonical = np.asarray(canonical, dtype=np.float64)
    if target.shape[0] == 0:
        return np.asarray(init, dtype=np.float64).copy(), np.full(canonical.shape[0], np.inf), 0.0, False
    # 质心预对齐: 纯平移那部分先吃掉, 粗配只需管朝向与残差
    prealigned = np.asarray(init, dtype=np.float64).copy()
    prealigned[:3, 3] += target.mean(axis=0) - transform_points(prealigned, canonical).mean(axis=0)
    coarse_m = float(max(coarse_m, ICP_THRESHOLD_M))
    coarse = icp_to_reference(
        canonical, target, prealigned, iterations=iterations, threshold_m=coarse_m
    )
    # 再以粗配结果为种子, 在名义容差下精配一遍 (粗配只管把朝向/位置摆进捕获范围)
    refine = icp_to_reference(
        canonical, target, coarse[0], iterations=iterations, threshold_m=ICP_THRESHOLD_M
    )
    # 两者的 ratio 都是在名义容差 (ICP_THRESHOLD_M) 下算的, 直接可比: 取内点多的那个,
    # 平手时取精配 —— 它没有放宽任何门限。
    return refine if refine[2] >= coarse[2] else coarse


@dataclass
class PoseResult:
    frame_index: np.ndarray
    T: np.ndarray
    valid: np.ndarray
    observed: np.ndarray
    confidence: np.ndarray
    residual_m: np.ndarray
    latched: np.ndarray
    inlier_ratio: np.ndarray
    rejected_translation: np.ndarray  # 平移步长超限 -> 整帧丢弃
    rejected_rotation: np.ndarray  # 旋转步长超限 -> 只按住旋转, 平移仍采用
    recovered: np.ndarray  # 原路径丢跟、靠"质心预对齐+粗门限"接回**且被采用**的帧
    low_points: np.ndarray  # 有观测但目标点少于 ICP_MIN_UNIQUE_TARGET 的帧 (拟合不成立 -> 判无效)
    area_px: np.ndarray
    reference_frame: int
    reference_pos: int
    reference_info: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    # 多物体/锁存口径: 物体是谁、门值与开关、以及被门挡下的帧 (都要能复查)
    instance_id: str = INSTANCE_ID
    latch_enabled: bool = True
    latch_distance_m: float = float("nan")
    refused_latch: list = field(default_factory=list)  # [(帧位, 原因码, 距离)]


def estimate(
    paths: RelPaths,
    clouds: dict,
    hand_frames: np.ndarray,
    hand_valid: np.ndarray,
    closed: np.ndarray,
    *,
    reference_frame: int,
    cfg=None,
    reference_pos: int | None = None,
    verbose: bool = True,
    instance_id: str = INSTANCE_ID,
    latch: bool = True,
    latch_distance_m: float | None = None,
    blocked: np.ndarray | None = None,
) -> PoseResult:
    """逐帧估计**一个物体**的 `T_camera0_object`。

    多物体 = 调用方逐物体各调一次 (点云/参考帧都逐物体), 并用 `blocked` 交接归属。

    - `latch=False`: 不做锁存、不手推, 只留测量/门控/平滑那一路。
    - `latch_distance_m=None`: 读配置的 `perception.latch_distance_m` (他的 0.20);
      pipeline 传 0.05 (用户规则)。`blocked` 见文件头。
    """
    cfg = cfg or load_ego_config()[0]
    perception = cfg.perception
    frame_index = np.asarray(clouds["frame_index"], dtype=np.int32)
    n = len(frame_index)
    notes: list[str] = []

    # ---- 锁存口径 (门值 / 开关 / 归属) ----
    # 门值默认读他的 perception.latch_distance_m (0.20); pipeline 显式传 0.05 (用户规则)。
    latch_distance_m = float(
        perception.latch_distance_m if latch_distance_m is None else latch_distance_m
    )
    if blocked is None:
        blocked = np.zeros(n, dtype=bool)
    else:
        blocked = np.asarray(blocked, dtype=bool)
        if blocked.shape != (n,):
            raise ValueError(f"blocked 形状 {blocked.shape} 与 {n} 帧不符")
    if not latch:
        notes.append("latch=False: 只做测量/门控/平滑, 不做抓取锁存与手推传播")
    if blocked.any():
        notes.append(
            f"{int(blocked.sum())} 帧的物体位姿已归属另一个物体 -> 本物体 ({instance_id}) "
            f"在这几帧不锁存"
        )

    # ---- 参考帧 ----
    if reference_pos is None:
        matches = np.nonzero(frame_index == int(reference_frame))[0]
        reference_pos = int(matches[0]) if matches.size else int(np.searchsorted(frame_index, reference_frame))
    reference_pos = int(np.clip(reference_pos, 0, n - 1))
    if not bool(clouds["valid"][reference_pos]):
        moved = np.nonzero(clouds["valid"])[0]
        if moved.size == 0:
            raise RuntimeError("所有帧都没有观测, 无法定参考帧姿态")
        nearest = int(moved[np.argmin(np.abs(moved - reference_pos))])
        notes.append(
            f"参考帧 {int(frame_index[reference_pos])} 没有可用观测, 退到最近的 "
            f"{int(frame_index[nearest])} (原计划的 SAM2 起始帧仍记在报告里)"
        )
        if verbose:
            print(f"    [warn] {notes[-1]}")
        reference_pos = nearest

    reference_points = cloud_at(clouds, reference_pos)
    thin_reference: str | None = None
    if reference_points.shape[0] < ICP_MIN_UNIQUE_TARGET:
        # 参考点云本身太稀 -> 后面每一帧的 ICP 配上都不可信, 这事必须说出来而不是照跑
        thin_reference = (
            f"参考帧 {int(frame_index[reference_pos])} 只有 {reference_points.shape[0]} 个点 "
            f"(< {ICP_MIN_UNIQUE_TARGET}), 参考系本身太稀, 逐帧 ICP 结果不可信"
        )
        notes.append(thin_reference)
    T_reference, reference_info = pca_frame(reference_points)
    canonical = transform_points(invert(T_reference), reference_points)
    if verbose:
        print(
            f"    参考帧 {int(frame_index[reference_pos])}: {reference_points.shape[0]} 点, "
            f"PCA {reference_info['method']}, 质心 "
            f"[{reference_info['centroid'][0]:+.3f} {reference_info['centroid'][1]:+.3f} "
            f"{reference_info['centroid'][2]:+.3f}] m"
        )
        if thin_reference is not None:
            print(f"    [warn] {thin_reference}")

    # ---- 逐帧 ----
    T = np.repeat(np.eye(4)[None], n, axis=0)
    valid = np.zeros(n, dtype=bool)
    observed = np.asarray(clouds["valid"], dtype=bool).copy()
    confidence = np.zeros(n, dtype=np.float32)
    residual_m = np.full(n, np.nan, dtype=np.float32)
    inlier_ratio = np.full(n, np.nan, dtype=np.float32)
    latched = np.zeros(n, dtype=bool)
    rejected_t = np.zeros(n, dtype=bool)
    rejected_r = np.zeros(n, dtype=bool)

    T[reference_pos] = T_reference
    valid[reference_pos] = True
    confidence[reference_pos] = 1.0
    residual_m[reference_pos] = 0.0
    inlier_ratio[reference_pos] = 1.0

    current = T_reference.copy()
    last_measurement = T_reference.copy()
    last_measurement_pos = reference_pos
    recent_translations: list[np.ndarray] = [T_reference[:3, 3].copy()]
    centroid_uv: list[np.ndarray | None] = [None] * n
    # 逐帧 mask 质心像素 (给"运动量"用, 相当于他的 keypoint_motion_px)
    for i in range(n):
        pts = cloud_at(clouds, i)
        centroid_uv[i] = pts[:, :2].mean(axis=0) if pts.shape[0] else None

    # 锁存: `latch_pose` = T_hand_object (`None` = 未就绪)。他那边一旦建立就冻结这个变换,
    # 我们这边**每有一帧可用测量就重新锚定**一次 —— 原因见 finish_frame 的注释。
    latch_pose: np.ndarray | None = None
    latch_start_frame: int | None = None
    latch_start_pos: int | None = None
    # 最近一次"物体位姿有来源"的帧位 (参考帧 / 一次测量 / 一次手推)。在此之前物体位姿
    # 还是初始的单位阵, 不能拿来锁存。
    last_known_pos: int | None = None
    latch_events: list[str] = []
    refused_latch: list[tuple[int, str, float]] = []  # (帧位, 原因码, 距离)
    recovered_notes: list[str] = []
    canonical_rms = rms_radius(canonical)
    recovered = np.zeros(n, dtype=bool)
    # "点数过少"只对**有观测**的帧有意义 —— 无观测帧的 n_points 本来就是 0, 不记进来
    # (否则 116 帧全是噪声, 真正该看的 196 / 218 两帧被淹掉)。
    low_points = observed & (np.asarray(clouds["n_points"], dtype=np.int32) < ICP_MIN_UNIQUE_TARGET)

    def finish_frame(i: int, is_closed: bool, hand_ok: bool, accepted: bool) -> None:
        """每帧**收尾**才判锁存 —— 顺序照他 `encoding.py:167` `latch_object_poses`:
        先用本帧的物体位姿更新 `current`, **再**判 `grasp and state["object"] is None and
        hand_valid` 并记下 `T_tcp_object = inv(hand) @ current[object]`。
        (上一轮把这段写在循环体**开头**, 读的是本帧还没算出来的 `valid[i]` / `T[i]`,
        永远是初始的 False / 单位阵, 于是锁存一次都没触发过。)

        语义照 HumanEgo-main (见文件头「参照实现」): **建立那一帧锚一次, 之后握持期间不再重锚**
        —— 真正的逐帧手推在循环顶部无条件做掉 (那里才是"已锁存"的常态路径), 这里只负责
        **建立**与**就地锚定**两件事。`latched[i] = True` 的含义是「**本帧位姿来自手推**, 不是测量」。

        只用本帧的测量锚定 (他 `_get_hand_pose_world` 拿的也是本帧 `static_objs` 的位姿);
        本帧没有可用测量时退到"保持的位姿", 但**仍过 `latch_distance_m` 那道门** —— 没有它,
        物体不可见时会把一个陈旧位姿硬绑到手上, 等于让它瞬移 (这道门也是他的:
        `run_inference.py:198` / `DatasetGen.py:482`, 同一个 0.20 m)。

        **建立也要过同一道门** (用户 2026-09-23 规则: 夹爪闭合 ∧ 距手 < 5 cm 才锁存)。
        原来建立那一条**无门** —— 多物体时后果是: 闭合那一刻**可见的每个物体各自锚一次**,
        随后握持期间它们的位姿全被覆盖成 `手 x latch_pose`, 物体被一起冻结到手上, 而且
        这些帧 `valid=True` (手推置真、`observed` 保持假) -> **不会被裁**, 成了静默污染的
        "看起来有效的错位姿"。`blocked[i]` = 这一帧已归属另一个物体 -> 本物体同样不锁存
        (以及已建立的锁存要在主循环里释放)。`latch=False` 时整个函数直接返回。
        `blocked` 见 `estimate`。

        128-164 段的实测代价 (记录在案, 不是 bug): 冻结的锁存位姿与可见点云中位差 45.1 mm /
        最大 51.8 mm, 而同期 ICP 测量只差 2.3 mm。这条**不能单独证明**"物体在手里转了 45 mm"
        —— 握持时手遮挡物体、可见面变了, ICP 在残缺点云上拟合本身就会偏。用户要的是
        「抓取时物体锁存到夹爪上」, 就把这个分歧按 0 处理。
        """
        nonlocal latch_pose, latch_start_frame, latch_start_pos, last_known_pos
        if not latch:  # 只测量: 两条锁存路都不走
            return
        if not (is_closed and hand_ok):
            return
        if blocked[i]:
            # 这一帧的物体位姿归**另一个**物体: 本物体不得建立/重锚 (主循环另外负责释放)
            refused_latch.append((i, "owned_by_other", float("nan")))
            return

        if accepted:
            # 本帧有可用测量 -> 用它把 T_hand_object 重新锚定到最新
            distance = float(np.linalg.norm(T[i][:3, 3] - hand_frames[i][:3, 3]))
            if latch_pose is None:
                # 建立: 距手太远就不认作"握在手里" (他的归属门, 值由调用方定)
                if distance > latch_distance_m:
                    refused_latch.append((i, "too_far", distance))
                    return
                latch_start_frame = int(frame_index[i])
                latch_start_pos = i
                latch_events.append(
                    f"帧 {frame_index[i]} 闭合 -> 锁存就绪 (用本帧测量锚定, "
                    f"距手 {distance * 1000:.1f} mm, 门 {latch_distance_m * 1000:.1f} mm)"
                )
            # 已建立: 照旧每有一帧测量就重锚 (能走到这里说明建立时已过门)
            latch_pose = invert(hand_frames[i]) @ T[i]
            return

        # 已锁存时不会走到这里: 闭合且手有效 -> 循环顶部已经手推并 continue (见那里的注释)。

        # 还没就绪: 用"已知的"物体位姿就地锚定 (他的 `current` 就是保持值), 受距离门约束
        if last_known_pos is None:
            refused_latch.append((i, "no_measurement", float("nan")))
            return
        distance = float(np.linalg.norm(T[i][:3, 3] - hand_frames[i][:3, 3]))
        if distance > latch_distance_m:
            refused_latch.append((i, "too_far", distance))
            return
        latch_pose = invert(hand_frames[i]) @ T[i]
        latch_start_frame = int(frame_index[i])
        latch_start_pos = i
        latch_events.append(
            f"帧 {frame_index[i]} 闭合 -> 锁存就绪 (用保持的位姿锚定, "
            f"距手 {distance * 1000:.1f} mm, 门 {latch_distance_m * 1000:.1f} mm)"
        )
        valid[i] = True
        latched[i] = True
        last_known_pos = i

    for i in range(n):
        frame = int(frame_index[i])
        is_closed = bool(closed[i]) if i < len(closed) else False
        hand_ok = bool(hand_valid[i]) if i < len(hand_valid) else False

        # ---- 张开 / 让位 -> 解除锁存 ----
        # 让位 = 这一帧的物体位姿归另一个物体了 (多物体时按物体顺序交接归属)。
        if latch_pose is not None and (not is_closed or blocked[i]):
            span = int(latched[latch_start_pos:i].sum()) if latch_start_pos is not None else 0
            reason = "张开" if not is_closed else "归属另一个物体"
            latch_pose = None
            latch_events.append(
                f"帧 {frame} {reason} -> 解除锁存 (起于帧 {latch_start_frame}; "
                f"这期间 {i - (latch_start_pos or i)} 帧里手推了 {span} 帧, "
                f"其余用本帧测量)"
            )

        # ---- 已锁存: 他的原样 (HumanEgo-main DatasetGen.py:496-498 / run_inference.py:204-206) ----
        # 握持期间物体位姿一律 `手 x T_hand_object`: 不跑拟合、不重锚、不设门, 直接覆盖。
        # (上一轮曾按"指尖中点与物体不刚性"改成"测量优先"; 他的 hand 就是同一个指尖中点系,
        #  假设相同, 那个判断已撤销 —— 见文件头「参照实现」。)
        if latch_pose is not None and is_closed and hand_ok and not blocked[i]:
            pushed = hand_frames[i] @ latch_pose
            T[i] = pushed
            valid[i] = True
            latched[i] = True
            # 本帧没跑拟合 -> 如实记"没有测量", 不沿用上帧的数冒充测量
            confidence[i] = float("nan")
            residual_m[i] = float("nan")
            inlier_ratio[i] = float("nan")
            # 只维护 `current` 与 `recent_translations` —— 它们是"我们当前认为物体在哪",
            # 解握首帧要拿它们当 ICP 种子 / 平移中值窗口 (不维护就停在一个 100 mm 外的陈旧位姿)。
            # `last_measurement` / `last_measurement_pos` **故意不动**: `gap` 随握持时长增长,
            # 门预算 `0.04 x min(gap,5)` 随之放宽 —— 那正是他遮挡门控的语义; 收紧它反而会
            # 让解握首帧被 40 mm 平移门白丢一帧 (实测有效帧 158 -> 157)。
            current = pushed
            recent_translations.append(pushed[:3, 3].copy())
            recent_translations = recent_translations[
                -max(int(perception.object_translation_median_window), 1):
            ]
            continue

        if i == reference_pos:
            last_known_pos = i
            finish_frame(i, is_closed, hand_ok, accepted=True)
            continue

        if not observed[i]:
            # 无观测: 保持上帧估计, 如实记 valid=False (不硬凑)。
            # (若手上正握着它, 上面那条"已锁存"分支已经接管并 continue 了, 走不到这里。)
            T[i] = T[i - 1]
            valid[i] = False
            confidence[i] = 0.0
            finish_frame(i, is_closed, hand_ok, accepted=False)
            continue

        points = cloud_at(clouds, i)
        gap = max(i - last_measurement_pos, 1)
        candidate, residual, ratio, fitted = icp_to_reference(canonical, points, current)

        # ---- 丢跟接回: 只在这条路径**失败**时走, 已经跑对的帧逐位不变 ----
        # (门控允许 min(gap,5) 帧走 0.04 x 5 = 0.20 m, 而 ICP 的固定捕获半径只有 3.5 cm ——
        #  有观测却 ratio=0 的帧就是这么来的, 见 recover_icp 的注释)
        # 是否**采用**接回结果由下面的两道门决定 —— 走完门才记 recovered[i],
        # 免得"接回了但被门丢掉"的帧在日志里冒充一次有效测量。
        retry_note: str | None = None
        if not fitted or ratio < perception.minimum_pose_inlier_ratio:
            coarse_m = float(
                np.clip(
                    perception.maximum_object_translation_step_m * min(gap, 5),
                    ICP_THRESHOLD_M,
                    2.0 * canonical_rms,
                )
            )
            retry, retry_residual, retry_ratio, retry_fitted = recover_icp(
                canonical, points, current, coarse_m=coarse_m
            )
            # `not fitted` 也要采用接回: 原路径没拟合时那个 ratio 是拿 **seed** 算的
            # (`icp_to_reference` 的 fitted=False 表示 transform 就是传进去的 init),
            # 它不该赢过一次**真的**刚体拟合。实测 f210/f212: 原路径 fitted=False 而
            # ratio 也是 1.000, 于是 `retry_ratio > ratio` 不成立、接回被丢, 紧接着又被
            # 下面那道 `not fitted` 判掉 —— 两帧白丢 (f206-217 那段从中间断两个洞)。
            if retry_fitted and (not fitted or retry_ratio > ratio + 1e-9):
                retry_note = (
                    f"帧 {frame}: 原路径 ratio={ratio:.3f}(fitted={fitted}) 丢跟 -> "
                    f"质心预对齐 + 粗门限 {coarse_m * 1000:.1f} mm (gap={gap}) -> "
                    f"ratio={retry_ratio:.3f}"
                )
                candidate, residual, ratio, fitted = retry, retry_residual, retry_ratio, retry_fitted

        # 拟合不成立 / 内点比例不够 -> 整帧丢弃 (对应他 stereo_fusion.py:315-321 的 inlier 判据)
        # `not fitted` 也要丢: 点数不足时 `icp_to_reference` 根本没做刚体求解, 返回的还是
        # `init` 那个位姿, 它的残差小只说明"上帧位姿看着还行", 不构成**本帧的一次测量**。
        # 不设这道, 4 个点的 f196 会带着 confidence=0.941 冒充一次测量 (见 ICP_MIN_UNIQUE_TARGET)。
        if not fitted or ratio < perception.minimum_pose_inlier_ratio:
            T[i] = T[i - 1]
            valid[i] = False
            confidence[i] = float(ratio)
            inlier_ratio[i] = float(ratio)
            residual_m[i] = float(np.median(residual))
            if retry_note:
                recovered_notes.append(
                    f"{retry_note} -> 仍低于 minimum_pose_inlier_ratio="
                    f"{perception.minimum_pose_inlier_ratio}, 丢弃"
                )
            finish_frame(i, is_closed, hand_ok, accepted=False)
            continue

        # ---- 门控 (照 stereo_fusion.py:328-365 的顺序, 一步不差) ----
        gap_scale = min(gap, 5)
        # 他的 motion_px 只在"与上次测量紧邻"时才算, 否则保持 NaN
        # (motion_px 用 mask 质心的像素位移代替他的 keypoint_motion_px)
        motion_px = float("nan")
        if gap == 1:
            if centroid_uv[i] is not None and centroid_uv[last_measurement_pos] is not None:
                motion_px = float(
                    np.linalg.norm(centroid_uv[i] - centroid_uv[last_measurement_pos])
                )
        translation_step = float(np.linalg.norm(candidate[:3, 3] - last_measurement[:3, 3]))
        # 平移超限: 整帧丢弃, 沿用上帧
        if translation_step > perception.maximum_object_translation_step_m * gap_scale:
            rejected_t[i] = True
            T[i] = T[i - 1]
            valid[i] = False
            confidence[i] = float(ratio)
            inlier_ratio[i] = float(ratio)
            residual_m[i] = float(np.median(residual))
            if retry_note:
                recovered_notes.append(
                    f"{retry_note} -> 但平移步长 {translation_step * 1000:.1f} mm 超门 "
                    f"{perception.maximum_object_translation_step_m * gap_scale * 1000:.1f} mm, 丢弃"
                )
            finish_frame(i, is_closed, hand_ok, accepted=False)
            continue
        # 旋转超限: **只**把旋转按住不动 (保留上帧朝向), 平移测量继续用 —— 他的原样
        rotation_step = sf._rotation_step_deg(last_measurement, candidate)
        rotation_gap_scale = min(np.sqrt(gap_scale), 2.0)
        if rotation_step > perception.maximum_object_rotation_step_deg * rotation_gap_scale:
            rejected_r[i] = True
            candidate[:3, :3] = last_measurement[:3, :3]

        # ---- 平滑 (照 stereo_fusion.py:366-395, 全部调用他的函数) ----
        last_measurement = candidate.copy()
        last_measurement_pos = i
        recent_translations.append(candidate[:3, 3].copy())
        recent_translations = recent_translations[-max(int(perception.object_translation_median_window), 1):]
        alpha = sf._adaptive_translation_alpha(
            perception.object_translation_smoothing,
            motion_px,
            perception.object_translation_motion_deadband_px,
            perception.object_translation_full_response_px,
        )
        motion_ratio = (
            (alpha - perception.object_translation_smoothing)
            / max(1.0 - perception.object_translation_smoothing, 1e-6)
        )
        candidate[:3, 3] = sf._adaptive_translation_measurement(recent_translations, motion_ratio)
        current = sf._smooth_pose(
            current,
            candidate,
            translation_alpha=alpha,
            rotation_alpha=perception.object_rotation_smoothing,
        )

        T[i] = current
        valid[i] = True
        confidence[i] = float(ratio)
        inlier_ratio[i] = float(ratio)
        inliers = residual <= ICP_THRESHOLD_M
        residual_m[i] = float(np.median(residual[inliers])) if inliers.any() else float(np.median(residual))
        last_known_pos = i
        if retry_note:
            # 接回结果**真的被采用了**才记 recovered (过完两道门才走到这里)
            recovered[i] = True
            recovered_notes.append(f"{retry_note} -> 采用 (平移步长 {translation_step * 1000:.1f} mm)")
        # 本帧位姿已经算出来了, 现在才能判锁存 (顺序见 finish_frame)
        finish_frame(i, is_closed, hand_ok, accepted=True)

    def _latch_refusal_lines() -> list[str]:
        """拒绝锁存的帧压成区间 + 写明原因 (逐帧刷 100 行没用)。"""
        lines: list[str] = []
        specs = (
            ("no_measurement", "物体位姿此前没有任何测量 (SAM2 全程无观测)"),
            ("too_far", f"物体位姿距手超过 latch_distance_m={latch_distance_m} m"),
            ("owned_by_other", "这一帧的物体位姿已归属另一个物体 (同一时刻只锁一个)"),
        )
        for code, label in specs:
            picked = [(p, d) for (p, c, d) in refused_latch if c == code]
            if not picked:
                continue
            spans: list[str] = []
            for run in _runs([p for p, _ in picked]):
                tail = f"-{frame_index[run[1]]}" if run[1] > run[0] else ""
                spans.append(f"帧 {frame_index[run[0]]}{tail}")
            distances = [d for _, d in picked if np.isfinite(d)]
            extra = (
                f"; 实测距手 {min(distances):.3f}~{max(distances):.3f} m" if distances else ""
            )
            lines.append(f"{', '.join(spans)}: 手闭合但{label}{extra}")
        return lines

    if verbose:
        print(
            f"    [{instance_id}] 位姿: {int(valid.sum())}/{n} 帧有效 "
            f"(观测 {int(observed.sum())}, 手推 {int(latched.sum())}, "
            f"接回 {int(recovered.sum())}, 点数过少 {int(low_points.sum())}, "
            f"平移超限丢帧 {int(rejected_t.sum())}, 旋转按住 {int(rejected_r.sum())})"
            + (f"  [latch 门 {latch_distance_m * 1000:.1f} mm]" if latch else "  [latch 关]")
        )
        for note in recovered_notes:
            print(f"    [接回] {note}")
        for event in latch_events:
            print(f"    [latch] {event}")
        for line in _latch_refusal_lines():
            print(f"    [latch] 拒绝锁存 {line}")
        if low_points.any():
            spans = [
                (int(frame_index[run[0]]), int(frame_index[run[1]]))
                for run in _runs(list(np.nonzero(low_points)[0]))
            ]
            print(
                f"    [warn] {int(low_points.sum())} 帧点数过少 (mask 内有效深度点 < "
                f"{ICP_MIN_UNIQUE_TARGET}, 拟合不成立 -> 判无效): {spans}"
            )
    return PoseResult(
        frame_index=frame_index,
        T=T,
        valid=valid,
        observed=observed,
        confidence=confidence,
        residual_m=residual_m,
        latched=latched,
        inlier_ratio=inlier_ratio,
        rejected_translation=rejected_t,
        rejected_rotation=rejected_r,
        recovered=recovered,
        low_points=low_points,
        area_px=np.asarray(clouds["area_px"], dtype=np.int32),
        reference_frame=int(reference_frame),
        reference_pos=reference_pos,
        reference_info={**reference_info, "used_frame": int(frame_index[reference_pos])},
        notes=notes,
        instance_id=str(instance_id),
        latch_enabled=bool(latch),
        latch_distance_m=float(latch_distance_m),
        refused_latch=list(refused_latch),
    )


def save(paths: RelPaths, result, *, extra: dict | None = None) -> None:
    """落 `pose/T_camera0_object.npz` + `pose/object_pose_meta.json`。

    `result` 可以是一个 `PoseResult` (单物体, 形状与旧产物逐位相同) 或一串
    (多物体 -> 每个逐帧数组都多一个**物体轴**, 顺序即 obj 序)。
    """
    results = list(result) if isinstance(result, (list, tuple)) else [result]
    if not results:
        raise ValueError("save 至少要有一个物体的 PoseResult")
    instance_ids = [r.instance_id for r in results]

    def stack(name: str) -> np.ndarray:
        """逐物体的逐帧数组 -> 单个物体时原样 (旧形状), 多物体时沿新的物体轴堆起来。"""
        values = [np.asarray(getattr(r, name)) for r in results]
        return values[0] if len(values) == 1 else np.stack(values, axis=1)

    np.savez_compressed(
        paths.object_npz,
        instance_ids=np.asarray([str(v) for v in instance_ids]),
        categories=np.asarray([paths.seg_category(v) for v in instance_ids]),
        T_camera0_object=stack("T").astype(np.float64),
        valid=stack("valid"),
        observed=stack("observed"),
        confidence=stack("confidence"),
        residual_m=stack("residual_m"),
        inlier_ratio=stack("inlier_ratio"),
        latched=stack("latched"),
        rejected_translation=stack("rejected_translation"),
        rejected_rotation=stack("rejected_rotation"),
        recovered=stack("recovered"),
        low_points=stack("low_points"),
        frame_index=results[0].frame_index,
        area_px=stack("area_px"),
        reference_frame_index=np.asarray([r.reference_frame for r in results], dtype=np.int32),
        reference_used_frame=np.asarray(
            [r.reference_info["used_frame"] for r in results], dtype=np.int32
        ),
        latch_distance_m=np.asarray([r.latch_distance_m for r in results], dtype=np.float64),
        latch_enabled=np.asarray([r.latch_enabled for r in results], dtype=bool),
        **(extra or {}),
    )
    if len(results) > 1:
        # 单物体那份 meta 是历史的单一事实来源, 多物体时逐物体各写一份,
        # 免得"哪个物体的参考帧/锁存门是多少"被一份平均值糊掉。
        for r in results:
            _write_meta(paths, r, suffix=f"_{r.instance_id}")
        _write_meta(paths, results[0])
        return
    _write_meta(paths, results[0])


def _write_meta(paths: RelPaths, result: PoseResult, *, suffix: str = "") -> None:
    meta = {
        "instance_id": result.instance_id,
        "latch_enabled": bool(result.latch_enabled),
        "latch_distance_m": float(result.latch_distance_m),
        "latch_refused_frames": int(len(result.refused_latch)),
        "reference_frame_planned": result.reference_frame,
        "reference_frame_used": result.reference_info["used_frame"],
        "reference_method": result.reference_info["method"],
        "reference_anisotropy": result.reference_info["anisotropy"],
        "reference_centroid_m": result.reference_info["centroid"],
        "pca_evals": result.reference_info["pca_evals"],
        "valid_frames": int(result.valid.sum()),
        "observed_frames": int(result.observed.sum()),
        "latched_frames": int(result.latched.sum()),
        "recovered_frames": int(result.recovered.sum()),
        "low_point_frames": int(result.low_points.sum()),
        "rejected_translation_frames": int(result.rejected_translation.sum()),
        "rejected_rotation_frames": int(result.rejected_rotation.sum()),
        "frame_semantics": {
            "latched": "本帧位姿由「手 x T_hand_object」推出 (握持期间一律手推, 本帧未做拟合), "
            "与 HumanEgo-main 的锁存同义; 该帧 confidence/residual_m/inlier_ratio 记 nan 表示"
            "本帧没有测量",
            "rejected_translation": "平移步长超限 -> 该帧整帧丢弃, 沿用上帧位姿, valid=false",
            "rejected_rotation": "旋转步长超限 -> 该帧只按住旋转(取上帧朝向), 平移测量仍采用, valid 可为 true",
            "recovered": "原路径丢跟 (ICP 固定捕获半径 3.5cm 配不上) -> 质心预对齐+粗门限接回, **过完门控且实际采用**才置位",
            "low_points": "有观测但 mask 内有效深度点少于 ICP_MIN_UNIQUE_TARGET, 刚体拟合不成立 -> 该帧 valid=false (位姿沿用上帧)",
        },
        "notes": result.notes,
    }
    (paths.pose_dir / f"object_pose_meta{suffix}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
