"""step4: 裁剪 -> 对齐 -> 写出 LeRobot v2.1 数据集。

裁剪规则与 state/action 的生成**不在这里实现**, 而是调 `episode.py` 的那一份 ——
step2 用 `keep = arange(N)` 调过它 (全长), step4 用真正的 `keep_index` 再调一次。
同一份实现, 所以「裁完之后帧还对齐」是构造上的结论, 不是靠两处小心翼翼地对齐。

对齐的关键点: `action[t]` 指向的是 **episode 内的下一帧**, 不是源帧的下一帧。桥接的
短无效段与拼接边界处这两者不同 (`build_episode_arrays` 里 `T_w[k+1]`, k 是 episode
局部下标)。这里额外用几套独立算出来的量交叉验证:

  1. 与 step2 全长数组比 —— `state`/`reference` 必须**逐帧**等于全长数组按
     `keep_index` 取的切片 (它们不依赖邻居); `action` 只在「段尾」允许不同, 别处必须相等。
  2. 世界链 —— `T_world_midpoint == inv(T_camera0_world) @ T_camera0_midpoint` (仅手有效的
     帧), 以及用世界链**重算**一遍 state 的物体块 (`inv(T_world) @ inv(T_cam0_world) @
     T_cam0_object`), 与相机链现算的 state 逐帧比。两条路径的算术完全不同, 能抓出
     `T_camera0_world` 非刚体或取错帧。
  3. 与 step3 的观测帧比 —— 视频第 k 帧必须就是 `observation/<keep_index[k]>.png`。

校验都读**写盘后的** parquet / mp4 / meta json, 不是对内存数组断言。

task (openpi 训练用的语言指令) **只从 `--task` 来**, 见 `run()` 的 `tasks` 形参: 这里
不读 step2 的 report、也不从检测提示词推导。`--vis` 时另外画一张裁剪时间轴与每个
episode 的 check 视频 (`render_vis`), 它们落在数据集的 `vis/` 里, 不属于 LeRobot 契约。
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

from . import (ACTION_COORDINATE_SYSTEM, ACTION_REFERENCE_FRAME, ACTION_SOURCE_COORDINATE_SYSTEM, ACTION_STORAGE, EYE_H, EYE_W, FPS, FULL_H, FULL_W,
               INSTANCE_ID, OBSERVATION_SOURCES, bootstrap)
from .episode import (MAX_INVALID_GAP, build_episode_arrays, check_action_coordinate_conversion, check_world_chain, invert_batch,
                      keep_index, split_description)
from .export import (CHUNKS_SIZE, DEFAULT_CODEC, DEFAULT_ROBOT_TYPE, OBSERVATION_HEIGHT,
                     OBSERVATION_SIZE, OBSERVATION_WIDTH, BgrSink, action_semantics,
                     features as build_features, fit_frame, info_dict, probe_video,
                     validate_transforms, write_stream)

from . import names
from .paths import PipePaths
# 「至多一个物体被锁存」这两条校验与 step2 组装时用的是**同一份实现** (纯函数, 与 IO 无关),
# 所以裁剪后再查一次是同一个函数换一组输入, 不是另写一套。
from .step2 import _check_single_latch, _normalize_latched

VIDEO_KEY_FALLBACK = "observation.images.camera0"

# 每一个观测源在 `info.json` / `extraction_meta.json` 里的溯源标记。
VISUAL_SOURCE = {"step3": "pipeline_step3_composite", "step2": "pipeline_step2_raw"}


class _PngFrames:
    """观测帧 = step3 合成的 `observation/%05d.png` (已修手 + 夹爪), 按 episode 顺序读。"""

    def __init__(self, paths: PipePaths, index: np.ndarray):
        self.stem = paths.stem
        self.index = np.asarray(index, dtype=np.int64)
        self.paths = [paths.observation(int(f)) for f in self.index]
        missing = [str(p) for p in self.paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                f"{self.stem} 缺 {len(missing)} 张观测帧 (step3 没跑完, 或只跑了 --only-keep "
                f"的另一组帧), 例如 {missing[:3]}"
            )

    def __len__(self) -> int:
        return len(self.paths)

    def __iter__(self):
        import cv2

        for path in self.paths:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise FileNotFoundError(f"读不到观测帧 {path}")
            yield frame

    def read(self, k: int) -> np.ndarray:
        import cv2

        frame = cv2.imread(str(self.paths[k]), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(f"读不到观测帧 {self.paths[k]}")
        return frame

    def label(self, k: int) -> str:
        return f"observation/{int(self.index[k]):05d}.png"


class _SourceFrames:
    """观测帧 = **直接解源 mp4 的左半** (`--observation step2`): 原始手, 没有修复/夹爪。

    这是「跳过 step3」的定义本身。只取 `keep_index` 里的源帧, 所以帧号仍然与标签一一对应。

    顺序解码而不是逐帧 seek: 274 帧整段解一遍很快, 且不受关键帧分布影响 (`read()` 那条
    抽帧校验才用 seek, 一共只抽 4 帧)。
    """

    def __init__(self, paths: PipePaths, index: np.ndarray):
        self.stem = paths.stem
        self.mp4 = paths.mp4
        self.index = np.asarray(index, dtype=np.int64)
        if not self.mp4.is_file():
            raise FileNotFoundError(f"缺源视频 {self.mp4}")

    def __len__(self) -> int:
        return len(self.index)

    def __iter__(self):
        import cv2
        from xrhand.video import iter_frames

        wanted = [int(v) for v in self.index]
        if not wanted:
            return
        found, last_seen = 0, -1
        for i, frame in enumerate(iter_frames(str(self.mp4), FULL_W, FULL_H)):
            last_seen = i
            if i == wanted[found]:
                # iter_frames 给 RGB, BgrSink 收 BGR —— 与 `step3._write_sbs` 同一处转换。
                yield cv2.cvtColor(frame[:, :EYE_W], cv2.COLOR_RGB2BGR)
                found += 1
                if found == len(wanted):
                    return
            if i >= wanted[-1]:
                break
        raise RuntimeError(
            f"{self.mp4.name} 只解到第 {last_seen} 帧, 但 keep_index 要到第 {wanted[-1]} 帧"
        )

    def read(self, k: int) -> np.ndarray:
        import cv2

        frame_index = int(self.index[k])
        capture = cv2.VideoCapture(str(self.mp4))
        if not capture.isOpened():
            raise RuntimeError(f"opencv 打不开 {self.mp4}")
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok or frame is None:
            raise RuntimeError(f"{self.mp4.name} 第 {frame_index} 帧读不到")
        return frame[:, :EYE_W]

    def label(self, k: int) -> str:
        return f"{self.mp4.name} 左半第 {int(self.index[k]):05d} 帧"


# ---------------------------------------------------------------- 一段采集 -> 一个 episode


def _object_table(report: dict, key: str, legacy_key: str, default: str) -> list[str]:
    """读物体表 (有序的 `obj1..objN` / 对应类别)。

    多物体后 step2 写的是列表 (`instance_ids` / `categories`); 单物体的老产物只有
    `instance_id` / `category` 两个标量。两种都读得进来, 且**顺序即 obj 序**。
    """
    value = report.get(key)
    if value is None:
        value = [report.get(legacy_key, default)]
    values = [str(v) for v in value]
    if not values:
        raise ValueError(f"{key} 是空的 —— step2 的 report 里没有物体表")
    return values


def prepare(paths: PipePaths, *, task: str, max_invalid_gap: int = MAX_INVALID_GAP,
            observation: str = "step3") -> dict:
    """把 step2 的产物裁成一个 episode 的数组 + 帧清单。

    `task` 由**调用方**给 (CLI 的 `--task`), 不从这里读任何东西推导 —— 语言指令与
    step2/step3 的检测提示词 (`report["category"]` / `obs["mask_prompt"]`) 是两回事,
    混在一处会让「task 是哪来的」说不清。

    `observation` 选**帧从哪来** (`--observation`), 只影响视频:
      `step3` 读 `step3/observation/%05d.png` (已修手 + piper 夹爪);
      `step2` 直接解源 mp4 左半 (原始手, 没有夹爪) —— 此时**不读** step3 的任何产物。
    无论哪种, 标签列 (state/action/reference/timestamp) 完全一样。
    """
    bootstrap()

    if observation not in OBSERVATION_SOURCES:
        raise ValueError(f"--observation 只能是 {OBSERVATION_SOURCES}, 拿到 {observation!r}")
    if not paths.state_action_npz.is_file():
        raise FileNotFoundError(f"缺 {paths.state_action_npz} —— 先跑 step2")
    with np.load(paths.state_action_npz) as archive:
        sa = {key: archive[key] for key in archive.files}
    report = json.loads(paths.state_action_json.read_text(encoding="utf-8"))

    # 盘上产物的**语义指纹**, 两处都要 (npz + report), 且必须精确相等 —— 缺键 = 旧口径的
    # step2 产物 (只存相机系绝对位姿 / 相对参照物), 这里给出明确报错而不是让下游隐式 KeyError。
    # 只比 `action_storage` 分不出新旧 (旧的相机系产物也是 "absolute"), 所以两个 token 一起比。
    missing = [k for k in ("T_world_midpoint", "T_pico_world_openxr_midpoint",
                           "T_camera0_world", "action_storage", "action_reference_frame",
                           "action_coordinate_system", "action_source_coordinate_system") if k not in sa]
    if missing:
        raise ValueError(
            f"{paths.stem} 的 {paths.state_action_npz.name} 缺 {missing} —— 这是旧口径的 step2 "
            f"产物 (action/reference 不是 PICO OpenXR 右手世界系里的绝对位姿), 回 step2 重跑"
        )
    npz_tokens = (str(sa["action_storage"]), str(sa["action_reference_frame"]))
    json_tokens = (report.get("action_storage"), report.get("action_reference_frame"))
    expected_tokens = (ACTION_STORAGE, ACTION_REFERENCE_FRAME)
    if npz_tokens != expected_tokens or json_tokens != expected_tokens:
        raise ValueError(
            f"{paths.stem} 的 step2 产物是旧语义写的 (npz {npz_tokens}, report {json_tokens}, "
            f"期望 {expected_tokens}) —— 那份 action/reference 不是 PICO OpenXR 右手世界系里的绝对位姿, "
            f"要新语义请回 step2 重跑 (--prompt 至少给 2 段)"
        )
    npz_coordinates = (str(sa["action_coordinate_system"]), str(sa["action_source_coordinate_system"]))
    expected_coordinates = (ACTION_COORDINATE_SYSTEM, ACTION_SOURCE_COORDINATE_SYSTEM)
    json_coordinates = (report.get("action_coordinate_system"),
                        report.get("action_source_coordinate_system"))
    if npz_coordinates != expected_coordinates or json_coordinates != expected_coordinates:
        raise ValueError(
            f"{paths.stem} 的 step2 坐标系标记是 {npz_coordinates} / {json_coordinates}, "
            f"期望 {expected_coordinates} —— 请用 step2 --reassemble 重算 2c"
        )
    if npz_tokens != json_tokens:
        # write_report 不是原子写, npz 与 report 可能来自两次不同的跑法 —— 静默沿用正是要防的。
        raise ValueError(
            f"{paths.stem} 的 state_action.npz {npz_tokens} 与 state_action.json {json_tokens} "
            f"语义标记不一致 —— 两个产物不是同一次写出来的, 回 step2 重跑"
        )

    task = str(task).strip()
    if not task:
        # 与 s4 `_load_task` 同样的态度: 宁可报错, 不编一个字符串进 tasks.jsonl。
        raise ValueError(f"{paths.stem} 没有任务指令 —— step4 的 --task 给的是空串")

    valid = np.asarray(sa["state_valid"], dtype=bool)
    split = split_description(valid, max_invalid_gap)
    index = keep_index(valid, max_invalid_gap)
    if len(index) == 0:
        raise ValueError(
            f"{paths.stem} 按 --max-invalid-gap {max_invalid_gap} 裁完一帧不剩 "
            f"(无效段 {split['invalid_runs']}) —— 跳过这段采集"
        )

    obs = None
    if observation == "step3":
        if not paths.observation_json.is_file():
            raise FileNotFoundError(
                f"缺 {paths.observation_json} —— 先跑 step3, 或改 --observation step2 "
                f"(直接用源 mp4 的左半当观测)"
            )
        obs = json.loads(paths.observation_json.read_text(encoding="utf-8"))
        width, height = int(obs["observation"]["width"]), int(obs["observation"]["height"])
        if (width, height) != (EYE_W, EYE_H):
            raise ValueError(f"{paths.stem} 观测分辨率 {width}x{height} != {EYE_W}x{EYE_H}")
        if int(obs["n_frames"]) != split["n_frames_source"]:
            raise ValueError(
                f"{paths.stem} step3 的 n_frames {obs['n_frames']} != step2 的 "
                f"{split['n_frames_source']} —— 两步的帧号不在同一条时间轴上"
            )
        frames = _PngFrames(paths, index)
    else:
        # step2 模式: 时间轴只能跟 step1 对。不读 observation.json, 所以 step3 没跑过也行。
        if not paths.reel_json.is_file():
            raise FileNotFoundError(f"缺 {paths.reel_json} —— 先跑 step1")
        reel = json.loads(paths.reel_json.read_text(encoding="utf-8"))
        if int(reel["n_frames"]) != split["n_frames_source"]:
            raise ValueError(
                f"{paths.stem} step1 的 n_frames {reel['n_frames']} != step2 的 "
                f"{split['n_frames_source']} —— 两步的帧号不在同一条时间轴上"
            )
        frames = _SourceFrames(paths, index)

    # `T_camera0_midpoint` / `T_world_midpoint` 是 step2 按 hand_valid **填充过**的版本
    # (无效帧的原始值/世界值是单位阵), 全长与裁剪后因此逐位一致; `T_camera0_world` 未填
    # (它在无效帧上本来就是真值), 两条链的交叉验证靠它。
    T_mid = np.asarray(sa["T_camera0_midpoint"], dtype=np.float64)
    T_world = np.asarray(sa["T_world_midpoint"], dtype=np.float64)
    T_action = np.asarray(sa["T_pico_world_openxr_midpoint"], dtype=np.float64)
    coordinate_error = check_action_coordinate_conversion(T_world, T_action)
    T_cam_world = np.asarray(sa["T_camera0_world"], dtype=np.float64)
    T_obj = np.asarray(sa["T_camera0_object"], dtype=np.float64)
    closed = np.asarray(sa["grasp_closed"], dtype=bool)
    hand_valid = np.asarray(sa["hand_valid"], dtype=bool)
    instance_ids = _object_table(report, "instance_ids", "instance_id", INSTANCE_ID)
    arrays = build_episode_arrays(T_mid, T_obj, closed, index, T_action)
    validate_transforms(f"{paths.stem} T_camera0_midpoint(保留帧)", T_mid[index])
    validate_transforms(f"{paths.stem} T_camera0_object(保留帧)", T_obj[index])
    validate_transforms(f"{paths.stem} T_world_midpoint(保留帧)", T_world[index])
    validate_transforms(f"{paths.stem} T_pico_world_openxr_midpoint(保留帧)", T_action[index])
    # 世界链的定义式 (只对手有效的帧成立) —— `reference` 来自 `T_pico_world_openxr_midpoint`；其原始 `T_world_midpoint` 与
    # 相机链是两条独立落盘的阵列, 这条把两者钉在一起。
    world_chain_error = check_world_chain(T_world, T_mid, T_cam_world, hand_valid)

    # 至多一个物体被锁存 —— 裁剪后再断言一次 (写盘前的**真**断言; step2 那次查的是全长)
    latched = _normalize_latched(sa["latched"])
    max_latched = _check_single_latch(latched[index])

    episode = {
        "stem": paths.stem,
        "task": task,
        "category": str(report.get("category", "")),
        "instance_ids": instance_ids,
        "categories": _object_table(report, "categories", "category", str(report.get("category", ""))),
        # action 语义 (与 step2 的产物同源; 数据集级一致性由 `_write_all` 断言)
        "action_storage": npz_tokens[0],
        "action_reference_frame": npz_tokens[1],
        "action_coordinate_system": npz_coordinates[0],
        "action_source_coordinate_system": npz_coordinates[1],
        "action_coordinate_conversion_max_error": coordinate_error,
        "world_chain_max_error": world_chain_error,
        "max_simultaneous_latched": max_latched,
        "keep_index": index.astype(np.int64),
        "arrays": arrays,
        "split": split,
        "frames": frames,
        "observation": observation,
        "visual_source": VISUAL_SOURCE[observation],
        "n_frames": len(index),
        # 全长原量: 逐帧交叉验证与「本段锚点」都要用 (写盘前断言的对象)
        "T_mid": T_mid,
        "T_world": T_world,
        "T_action": T_action,
        "T_camera0_world": T_cam_world,
        "T_obj": T_obj,
        "hand_valid": hand_valid,
        "state_full": np.asarray(sa["state"], dtype=np.float32),
        "reference_full": np.asarray(sa["action_reference_tcp"], dtype=np.float32),
        "action_full": np.asarray(sa["action"], dtype=np.float32),
        "grasp_full": closed,
        "latched": latched,
        "ticks": np.asarray(sa["timestamp_ns"], dtype=np.int64)[index],
        "relation_direction": str(report.get("relation_direction", "T_tcp_object")),
        "seg_prompt": (report.get("seg") or {}).get("prompt"),
        # 只有 step3 模式有「手部检测提示词」可溯源。
        "mask_prompt": None if obs is None else obs.get("mask_prompt"),
        "lag": (json.loads(paths.reel_json.read_text(encoding="utf-8")).get("lag") or {}).get("frames")
        if paths.reel_json.is_file() else None,
    }
    check_episode(episode)
    return episode


def _runs(index: np.ndarray) -> list[tuple[int, int]]:
    """`keep_index` 里连续的那几段 (episode 局部的 [a,b] 闭区间)。"""
    out: list[tuple[int, int]] = []
    start = 0
    for k in range(1, len(index)):
        if index[k] != index[k - 1] + 1:
            out.append((start, k - 1))
            start = k
    out.append((start, len(index) - 1))
    return out


def check_episode(episode: dict) -> dict:
    """写盘前的交叉验证: 裁剪后各张量是否始终对齐。"""
    index = episode["keep_index"]
    state, reference, action = (episode["arrays"][k] for k in ("state", "reference", "action"))
    full_state, full_ref, full_act = (episode[k] for k in ("state_full", "reference_full", "action_full"))
    L = len(index)
    if not (len(state) == len(reference) == len(action) == L):
        raise AssertionError(f"{episode['stem']} 裁剪后各数组帧数不一致")

    # 1a. state / reference 不依赖邻居 -> 必须逐帧等于 step2 全长数组的对应行。
    if not np.allclose(state, full_state[index], atol=1e-6, rtol=0):
        bad = np.nonzero(~np.isclose(state, full_state[index], atol=1e-6, rtol=0).all(axis=1))[0]
        raise AssertionError(f"{episode['stem']} state 与 step2 全长数组不同步, 首帧 {bad[:5].tolist()}")
    if not np.allclose(reference, full_ref[index], atol=1e-6, rtol=0):
        raise AssertionError(f"{episode['stem']} reference 与 step2 全长数组不同步")
    # 1a'. **世界链**: reference 的右手系来源由 `T_world_midpoint` 经 M 共轭转换；原始位姿与相机链 (`T_camera0_world`
    # 由外参算出、`T_camera0_midpoint` 由 `T_camera0_world @ T_world_midpoint` 算出) 是
    # 两条独立落盘的阵列 —— 定义式把两者钉在一起 (只对手有效的帧)。再用世界链**重算**一遍
    # state 的物体块 (`inv(T_world) @ inv(T_cam0_world) @ T_cam0_obj`, 与相机链
    # `inv(T_cam0_mid) @ T_cam0_obj` 的算术完全不同), 两条路径逐帧比。
    flag = episode["hand_valid"]
    chain = invert_batch(episode["T_camera0_world"][flag]) @ episode["T_mid"][flag]
    if not np.allclose(episode["T_world"][flag], chain, atol=1e-6, rtol=0):
        raise AssertionError(
            f"{episode['stem']} 的 T_world_midpoint != inv(T_camera0_world) @ T_camera0_midpoint "
            f"(最大差 {np.abs(episode['T_world'][flag] - chain).max():.3e}) —— "
            f"reference/action 的世界系与 state 的相机系不同源"
        )
    from ego_relation.contracts.se3 import transform_to_vec9

    # 这一段比的是**落盘的那份 state** (已按 keep_index 裁过), 所以这一步的每个量都要先裁:
    # `flag` 是全长掩码, 拿它去索引裁剪后的数组会 "boolean index did not match"。
    kept = episode["keep_index"]
    kept_valid = flag[kept]
    # `[:, None]` 也是必须的: T_obj 是 (L, N, 4, 4), 而 matmul 的 batch 维**右对齐** ——
    # 直接把 (L,4,4) 与 (L,N,4,4) 相乘会拿 L 去对 N 而报 "operands could not be broadcast"。
    # 补一个长度为 1 的物体轴, 后面那次 @ 再广播到 N。
    world_obj = invert_batch(episode["T_world"][kept])[:, None] \
        @ invert_batch(episode["T_camera0_world"][kept])[:, None] @ episode["T_obj"][kept]
    state_chain = np.asarray(episode["arrays"]["state"], dtype=np.float64)[kept_valid]
    for j in range(episode["T_obj"].shape[1]):
        block = np.stack([transform_to_vec9(m) for m in world_obj[kept_valid, j]])
        if not np.allclose(state_chain[:, 9 * j:9 * (j + 1)], block, atol=1e-6, rtol=0):
            bad = np.nonzero(~np.isclose(state_chain[:, 9 * j:9 * (j + 1)], block, atol=1e-6,
                                         rtol=0).all(axis=1))[0]
            raise AssertionError(
                f"{episode['stem']} state 的第 {j} 个物体块用世界链重算后不一致 "
                f"(有效帧里第 {bad[:5].tolist()} 处) —— T_camera0_world 可能非刚体或取错帧"
            )

    # 1b. action 只在「段尾」与源帧的下一帧不同 (段尾的 episode 下一帧落在下一段的开头)。
    tails = {b for _, b in _runs(index)}
    interior = [k for k in range(L - 1) if k not in tails]
    if interior:
        rows = np.asarray(interior)
        if not np.allclose(action[rows], full_act[index[rows]], atol=1e-6, rtol=0):
            bad = rows[~np.isclose(action[rows], full_act[index[rows]], atol=1e-6, rtol=0).all(axis=1)]
            raise AssertionError(
                f"{episode['stem']} action 在段内位置与源帧下一帧不同: {bad[:5].tolist()}"
            )

    # 1c. 结构性恒等式 (与 s4 `lerobot.py:196` 同义): action = [x[1:], x[-1:]]。
    # 爪在 state 的**最后一维** (state = 9N 关系 + 1 爪), action 的爪仍在第 9 维。
    if not np.allclose(action[:, :9], np.concatenate([reference[1:], reference[-1:]]), atol=1e-6, rtol=0):
        raise AssertionError(f"{episode['stem']} action[:, :9] != [reference[1:], reference[-1:]]")
    if not np.array_equal(action[:, -1], np.concatenate([state[:, -1][1:], state[:, -1][-1:]])):
        raise AssertionError(f"{episode['stem']} action[:, -1] != [state[:, -1][1:], state[:, -1][-1:]]")

    # 1d. 二值爪只能是 0/1。
    if not np.isin(state[:, -1], (0.0, 1.0)).all() or not np.isin(action[:, -1], (0.0, 1.0)).all():
        raise AssertionError(f"{episode['stem']} 二值爪字段不是 0/1")
    if not np.array_equal(state[:, -1].astype(bool), episode["grasp_full"][index]):
        raise AssertionError(f"{episode['stem']} state 的开闭与 grasp_closed 不同步")
    return {"runs": _runs(index), "interior_checked": len(interior)}


# ---------------------------------------------------------------- 写出


def _write_episode(destination: Path, episode: dict, episode_index: int, task_index: int,
                   total_frames: int, codec: str) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    state = episode["arrays"]["state"]
    reference = episode["arrays"]["reference"]
    action = episode["arrays"]["action"]
    ticks = np.asarray(episode["ticks"], dtype=np.int64)
    length = len(state)

    chunk = episode_index // CHUNKS_SIZE
    video_key = _video_key()
    video_path = destination / "videos" / f"chunk-{chunk:03d}" / video_key / f"episode_{episode_index:06d}.mp4"
    # 进数据集的 mp4 固定 640x480 (与 ego_relation_policy 的训练图像口径一致);
    # 源帧本身是多少 (1080x810 / 源 mp4 左半) 不影响这条约束。
    write_stream(episode["frames"], video_path, length,
                 codec=codec, size=OBSERVATION_SIZE, fps=FPS)

    # LeRobot 的 timestamp 描述的是写出后固定 FPS 视频的数据时间轴，不是采集设备带抖动的
    # wall-clock 时间。视频已经把 keep_index 对应的帧重新编码成严格 30 FPS；如果这里继续写
    # PICO 原始 timestamp_ns（通常每帧约 32.5--33.9 ms），LeRobotDataset 会因相邻时间戳
    # 不等于 1/FPS 而拒绝整个数据集。原始时间/源帧对应关系仍保留在 Step2 产物与
    # extraction_meta.json 的 keep_index 中，不参与训练时间轴。
    timestamp = (np.arange(length, dtype=np.float64) / float(FPS)).astype(np.float32)
    parquet_path = destination / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "observation.state": list(state),
                "observation.action_reference_tcp": list(reference),
                "action": list(action),
                "timestamp": timestamp,
                "frame_index": np.arange(length, dtype=np.int64),
                "episode_index": np.full(length, episode_index, dtype=np.int64),
                "index": np.arange(total_frames, total_frames + length, dtype=np.int64),
                "task_index": np.full(length, task_index, dtype=np.int64),
            }
        ),
        parquet_path,
    )
    return {
        "episode_index": episode_index,
        "length": length,
        "task": episode["task"],
        "task_index": task_index,
        "parquet": parquet_path,
        "video": video_path,
        "state": state,
        "reference": reference,
        "action": action,
        "timestamp": timestamp,
        "ticks": ticks,
    }


def _video_key() -> str:
    from .export import _s4

    try:
        return _s4().VIDEO_KEY
    except Exception:  # pragma: no cover - 只在 ego_relation 不可 import 时兜底
        return VIDEO_KEY_FALLBACK


def write_dataset(destination: Path, episodes: list[dict], *, robot_type: str = DEFAULT_ROBOT_TYPE,
                  codec: str = DEFAULT_CODEC, force: bool = False, quiet: bool = False) -> dict:
    """写整个数据集 (1 段采集 = 1 个 episode, 按传入顺序编号)。"""
    destination = Path(destination)
    if destination.exists():
        if not force:
            raise FileExistsError(f"输出已存在, 避免覆盖: {destination} (--force 覆盖)")
        target = _atomic_target(destination)
    else:
        target = destination
    try:
        summary = _write_all(target, episodes, robot_type=robot_type, codec=codec)
        verify_dataset(target, summary, episodes)
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise
    if target != destination:
        _swap_in(target, destination)
    if not quiet:
        print(f"  step4: {summary['n_episodes']} 个 episode / {summary['total_frames']} 帧 -> {destination}")
    return summary


def _atomic_target(destination: Path) -> Path:
    rebuilding = destination.with_name(destination.name + ".rebuilding")
    previous = destination.with_name(destination.name + ".previous")
    if previous.exists():
        raise FileExistsError(f"发现未清理的备份, 拒绝覆盖: {previous}")
    shutil.rmtree(rebuilding, ignore_errors=True)
    return rebuilding


def _swap_in(rebuilding: Path, destination: Path) -> None:
    previous = destination.with_name(destination.name + ".previous")
    destination.rename(previous)
    try:
        rebuilding.rename(destination)
    except Exception:
        previous.rename(destination)
        raise
    shutil.rmtree(previous, ignore_errors=True)


def _write_all(destination: Path, episodes: list[dict], *, robot_type: str, codec: str) -> dict:
    # 物体是**数据集级**的, 不是逐 episode 的: s4 的 `object_order` / `object_categories`
    # 是一对同长常量 (`lerobot.py:404-405`), 而且它对每个 episode dir 都断言与那两个常量
    # 一致 (`:203-206`), 不符直接报错 —— 参考的语义就是「一个数据集 = 一套固定的、**有序的**
    # 物体表」。照此: 各段的有序 `(instance_id, category)` 表必须**完全相等**, 否则同一个
    # 数据集里 state 各维的含义会逐段漂移 (第 2 段的前 9 维是另一个物体), 训练侧无从察觉。
    # 放在最前面: 不一致就别写盘了 (parquet/mp4 都在后面)。
    tables = [list(zip(e["instance_ids"], e["categories"], strict=True)) for e in episodes]
    first = tables[0]
    if not first:
        raise ValueError(f"{episodes[0]['stem']} 的物体表是空的 —— step2 没写出 instance_id")
    mismatch = {e["stem"]: t for e, t in zip(episodes, tables, strict=True) if t != first}
    if mismatch:
        raise ValueError(
            f"一个数据集只能有一套固定的有序物体表, 以 {episodes[0]['stem']} 的 {first} 为准, "
            f"但这些段不一致: {mismatch} —— 多套物体请分开导成多个数据集"
        )
    object_order = [instance_id for instance_id, _ in first]
    object_categories = [category or instance_id for instance_id, category in first]

    # action/reference 的参考系是**逐段各自一个** PICO OpenXR 右手世界系, 而 info.json 里只有一份全局
    # 声明 —— 于是每段的**锚点**必须落盘: 没有它, 绝对量对真机部署没有落点, 也没法判断两段
    # 的世界系差了多远。写盘前先钉住「每段都有锚点、且是合法刚体」。
    anchors: dict[str, list[float]] = {}
    for episode in episodes:
        anchor = np.asarray(episode["arrays"]["reference"][0], dtype=np.float64)
        if anchor.shape != (9,) or not np.isfinite(anchor).all():
            raise ValueError(f"{episode['stem']} 的 PICO OpenXR 右手世界系锚点不是 9 个有限数: {anchor!r}")
        anchors[episode["stem"]] = [float(v) for v in anchor]
    frames_kind = {e["action_reference_frame"] for e in episodes}
    storages = {e["action_storage"] for e in episodes}
    coordinate_systems = {e["action_coordinate_system"] for e in episodes}
    source_coordinate_systems = {e["action_source_coordinate_system"] for e in episodes}
    if (frames_kind != {ACTION_REFERENCE_FRAME} or storages != {ACTION_STORAGE}
            or coordinate_systems != {ACTION_COORDINATE_SYSTEM}
            or source_coordinate_systems != {ACTION_SOURCE_COORDINATE_SYSTEM}):
        raise ValueError(
            f"一个数据集的 action 只能有一种帧种类/存法, 拿到 {sorted(frames_kind)} / "
            f"{sorted(storages)} (期望 {ACTION_REFERENCE_FRAME!r} / {ACTION_STORAGE!r}) —— "
            f"语义/坐标系不同的采集请分开导成多个数据集"
        )

    visual_sources = {e["visual_source"] for e in episodes}
    if len(visual_sources) != 1:
        raise ValueError(f"一个数据集只能有一种观测来源, 拿到 {sorted(visual_sources)}")
    visual_source = visual_sources.pop()

    meta = destination / "meta"
    meta.mkdir(parents=True, exist_ok=False)

    task_to_index: dict[str, int] = {}
    for episode in episodes:
        task_to_index.setdefault(episode["task"], len(task_to_index))

    rows = []
    total_frames = 0
    for episode_index, episode in enumerate(episodes):
        row = _write_episode(destination, episode, episode_index,
                             task_to_index[episode["task"]], total_frames, codec)
        rows.append(row)
        total_frames += row["length"]

    state_dim = int(rows[0]["state"].shape[1])
    reference_dim = int(rows[0]["reference"].shape[1])
    action_dim = int(rows[0]["action"].shape[1])
    for row in rows:
        if (row["state"].shape[1], row["reference"].shape[1], row["action"].shape[1]) != (
                state_dim, reference_dim, action_dim):
            raise ValueError(f"episode {row['episode_index']} 的向量维度与其他 episode 不一致")
    # state 的形状是物体数**决定**的, 不是碰巧: 9N 关系 + 1 爪。写盘前把这条钉住。
    if state_dim != 9 * len(object_order) + 1:
        raise ValueError(
            f"observation.state 维度 {state_dim} != 9*{len(object_order)}+1 "
            f"(物体表 {object_order}) —— 物体轴没接上"
        )

    codec_name = codec
    info = info_dict(
        features=build_features(state_dim, action_dim, reference_dim,
                                height=OBSERVATION_HEIGHT, width=OBSERVATION_WIDTH,
                                codec=codec_name, fps=FPS,
                                state_names=names.state_names(*object_order),
                                reference_names=names.reference_names(),
                                action_names=names.action_names()),
        n_episodes=len(rows), total_frames=total_frames, tasks=list(task_to_index),
        robot_type=robot_type, fps=FPS, codec=codec_name, variant="binary",
        object_order=object_order, object_categories=object_categories,
        latched_frames=sum(int(e["latched"][e["keep_index"]].sum()) for e in episodes),
        max_invalid_gap=int(episodes[0]["split"]["max_invalid_gap"]),
        visual_source=visual_source,
        max_simultaneous_latched=max(int(e["max_simultaneous_latched"]) for e in episodes),
    )
    (meta / "info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # openpi 的 Mode2 契约 —— 缺这份声明它会直接拒收 (形状与列名分不出「绝对」与「相对」)。
    # 完整 XRPipe Mode1 契约由 `export.action_semantics()` 一处生成。
    (meta / "action_semantics.json").write_text(
        json.dumps(action_semantics(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    tasks_by_index = sorted(task_to_index.items(), key=lambda item: item[1])
    (meta / "tasks.jsonl").write_text(
        "".join(json.dumps({"task_index": index, "task": task}, ensure_ascii=False) + "\n"
                for task, index in tasks_by_index),
        encoding="utf-8",
    )
    (meta / "episodes.jsonl").write_text(
        "".join(json.dumps({"episode_index": row["episode_index"], "tasks": [row["task"]],
                            "length": row["length"]}, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    from .export import stats as compute_stats

    (meta / "episodes_stats.jsonl").write_text(
        "".join(json.dumps({"episode_index": row["episode_index"], "stats": {
            "observation.state": compute_stats(row["state"]),
            "observation.action_reference_tcp": compute_stats(row["reference"]),
            "action": compute_stats(row["action"]),
        }}) + "\n" for row in rows),
        encoding="utf-8",
    )
    (meta / "stats.json").write_text(
        json.dumps({
            "observation.state": compute_stats(np.concatenate([r["state"] for r in rows])),
            "observation.action_reference_tcp": compute_stats(np.concatenate([r["reference"] for r in rows])),
            "action": compute_stats(np.concatenate([r["action"] for r in rows])),
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    extraction = {
        # v4 = XRPipe Mode1 explicitly preserves the measured thumb/index
        # fingertip midpoint and forbids wrist/palm reinterpretation.
        "schema_version": "xrpipe_v4",
        "timestamp_semantics": "uniform_frame_index_over_fps",
        "variant": "binary",
        "dataset": destination.name,
        "robot_type": robot_type,
        "fps": FPS,
        "object_order": object_order,
        "object_categories": object_categories,
        "action_storage": ACTION_STORAGE,
        "action_reference_frame": ACTION_REFERENCE_FRAME,
        "action_coordinate_system": ACTION_COORDINATE_SYSTEM,
        "action_source_coordinate_system": ACTION_SOURCE_COORDINATE_SYSTEM,
        "action_coordinate_transform": {
            "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
            "translation": "t_openxr = M @ t_unity",
            "rotation": "R_openxr = M @ R_unity @ M",
        },
        "action_note": (
            "action / observation.action_reference_tcp 是 **PICO OpenXR 右手世界系**里的指尖中点绝对位姿 "
            "(不是相机系, 也不是相对某个物体); 相对动作由训练期 "
            "inv(T_current) @ T_absolute_target 现算。物体位姿本身仍在 observation.state 的 "
            "9N 个关系维里 (物体在该手 TCP 系里)"
        ),
        # 参考系**逐段各自一个** (原点由 runtime 定), 所以每段必须有自己的锚点: 首帧手在世界系
        # 里的绝对位姿 (vec9, 与 reference/action 同一套编码)。跨段只有相对运动可比。
        "pico_world_openxr_anchors": {
            "definition": "T_pico_world_openxr_tcp at the episode's first retained frame (vec9: "
                          "[tx,ty,tz,R[:,0],R[:,1]]); one world frame per recording",
            "episodes": anchors,
        },
        "observation": {
            "eye": "left (camera0 / eye0)",
            # width/height = **源帧** (step3 PNG 或源 mp4 左半) 的分辨率
            "width": EYE_W, "height": EYE_H,
            "key": _video_key(), "codec": codec,
            # dataset_* = 真正进 mp4 与 features.shape 的分辨率 (与 ego_relation_policy 对齐)
            "dataset_width": OBSERVATION_WIDTH, "dataset_height": OBSERVATION_HEIGHT,
            "visual_source": visual_source,
            "note": "进数据集的 mp4 与 features 里的 shape 用 dataset_* (640x480, 与 "
                    "ego_relation_policy 的训练图像口径一致); width/height 是源帧, 编码时"
                    "按 INTER_AREA 等比缩放 (1080x810 与 640x480 同为 4:3, 不变形)",
        },
        "cropping": {
            "rule": "one episode per recording; invalid runs < max_invalid_gap are bridged in "
                    "place, longer ones are excised and the remainder spliced into the same episode",
            "max_invalid_gap": int(episodes[0]["split"]["max_invalid_gap"]),
            "invalid_definition": "state_valid = all(object_valid) & hand_valid (step2)",
        },
        "episodes": [{
            "episode_index": row["episode_index"],
            "source_stem": episode["stem"],
            "source_video": f"CameraRecord_{episode['stem']}.mp4",
            "source_dir": str(episode["source_dir"]),
            "n_frames_source": int(episode["split"]["n_frames_source"]),
            "n_frames": int(episode["n_frames"]),
            "keep_index": [int(v) for v in episode["keep_index"]],
            "valid_runs": episode["split"]["valid_runs"],
            "invalid_runs": episode["split"]["invalid_runs"],
            "kept_invalid_runs": episode["split"]["kept_invalid_runs"],
            "dropped_invalid_runs": episode["split"]["dropped_invalid_runs"],
            "episode_local_runs": [[int(a), int(b)] for a, b in _runs(episode["keep_index"])],
            "latched_frames": int(episode["latched"][episode["keep_index"]].sum()),
            # 逐物体的锁存帧数 —— 那些帧里退化的是 **state** 的该物体块 (位姿由手推得, 不是
            # 测量), action 不受影响 (它只跟手自身有关)。
            "latched_frames_per_object": {
                instance_id: int(episode["latched"][episode["keep_index"]][:, j].sum())
                for j, instance_id in enumerate(object_order)
            },
            "task": episode["task"],
            "pico_world_openxr_anchor": anchors[episode["stem"]],
            "action_coordinate_conversion_max_error": float(
                episode["action_coordinate_conversion_max_error"]),
            "world_chain_max_error": float(episode["world_chain_max_error"]),
            "category": episode["category"],
            "seg_prompt": episode["seg_prompt"],
            "mask_prompt": episode["mask_prompt"],
            "lag_frames": episode["lag"],
            "relation_direction": episode["relation_direction"],
            "ticks_ns_first": int(row["ticks"][0]),
            "ticks_ns_last": int(row["ticks"][-1]),
        } for row, episode in zip(rows, episodes, strict=True)],
    }
    (destination / "extraction_meta.json").write_text(
        json.dumps(extraction, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return {"info": info, "rows": rows, "n_episodes": len(rows), "total_frames": total_frames,
            "tasks": list(task_to_index), "extraction": extraction}


# ---------------------------------------------------------------- 写盘后读回校验


def verify_dataset(destination: Path, summary: dict, episodes: list[dict]) -> None:
    """全部读**写盘后**的文件: parquet / mp4 / meta json。"""
    import pyarrow.parquet as pq

    rows = summary["rows"]
    info = json.loads((destination / "meta/info.json").read_text(encoding="utf-8"))
    if info["total_episodes"] != len(rows) or info["total_frames"] != summary["total_frames"]:
        raise AssertionError("info.json 的 total_* 与实际不符")
    if info["splits"] != {"train": f"0:{len(rows)}"}:
        raise AssertionError(f"info.json splits 异常: {info['splits']}")
    if info["total_tasks"] != len(summary["tasks"]):
        raise AssertionError("info.json total_tasks 与 tasks.jsonl 不符")
    if info["fps"] != FPS or info["chunks_size"] != CHUNKS_SIZE:
        raise AssertionError("info.json fps / chunks_size 异常")

    video_key = _video_key()
    # 导出的 mp4 固定 640x480, 与 info.json 的 features + ego_relation.visual_size 三者必须一致。
    video_shape = [OBSERVATION_HEIGHT, OBSERVATION_WIDTH, 3]
    if info["features"][video_key]["shape"] != video_shape:
        raise AssertionError(
            f"features[{video_key}].shape {info['features'][video_key]['shape']} != {video_shape}"
        )
    if list(info["ego_relation"]["visual_size"]) != video_shape[:2]:
        raise AssertionError(
            f"ego_relation.visual_size {info['ego_relation']['visual_size']} != {video_shape[:2]}"
        )
    # 维度由物体数**推出**, 不是写死的常量 —— 这样它仍是一条真断言 (物体轴接错就报)。
    object_count = len(info["ego_relation"]["object_order"])
    if object_count != len(info["ego_relation"]["object_categories"]):
        raise AssertionError("info.json 的 object_order / object_categories 长度不等")
    layers = {
        "observation.state": 9 * object_count + 1,
        "observation.action_reference_tcp": 9,
        "action": 10,
    }
    for key, dim in layers.items():
        feature = info["features"][key]
        if feature["shape"] != [dim]:
            raise AssertionError(f"features[{key}].shape {feature['shape']} != [{dim}]")
        # s4 的向量维度名是**列表套列表**, openpi 的校验读 names[0]。
        # 变量别叫 `names` —— 它会遮住模块 `names` (下面按 object_order 现算期望名字要用)。
        declared_names = feature["names"]
        if not (isinstance(declared_names, list) and len(declared_names) == 1
                and isinstance(declared_names[0], list)):
            raise AssertionError(
                f"features[{key}].names 必须是列表套列表 (s4 如此), 拿到 {declared_names!r}"
            )
        if len(declared_names[0]) != dim:
            raise AssertionError(f"features[{key}] 的维度名个数 {len(declared_names[0])} != {dim}")
    # reference/action 的**名字**必须与 s4 逐字同一份 (存的就是绝对位姿, 名字里没有任何物体
    # id), state 的名字必须与 object_order 逐位对上 —— 一个「名字说一套、数值是另一套」的
    # 数据集在训练侧无从察觉, 所以这里查**内容**, 不只查个数。
    expected_names = {
        "observation.state": names.state_names(*info["ego_relation"]["object_order"]),
        "observation.action_reference_tcp": names.REFERENCE_NAMES,
        "action": names.ACTION_NAMES,
    }
    for key, expected in expected_names.items():
        got = info["features"][key]["names"][0]
        if got != expected:
            raise AssertionError(
                f"features[{key}].names 与期望的逐维名字不一致:\n  实际 {got}\n  期望 {expected}"
            )
    # 帧/存法两个 token 必须与代码里的常量精确相等 (旧产物是 "camera0 (left eye, eye0)" /
    # "relative_to_object"), 不能用子串比 —— 只比 action_storage 分不出新旧。
    if info["ego_relation"]["action_storage"] != ACTION_STORAGE:
        raise AssertionError(
            f"info.json 的 action_storage {info['ego_relation']['action_storage']!r} "
            f"!= {ACTION_STORAGE!r}"
        )
    if info["ego_relation"]["action_reference_frame"] != ACTION_REFERENCE_FRAME:
        raise AssertionError(
            f"info.json 的 action_reference_frame "
            f"{info['ego_relation']['action_reference_frame']!r} != {ACTION_REFERENCE_FRAME!r}"
        )
    if info["ego_relation"].get("action_coordinate_system") != ACTION_COORDINATE_SYSTEM:
        raise AssertionError("info.json 的 action_coordinate_system 与代码契约不一致")
    if info["ego_relation"].get("action_source_coordinate_system") != ACTION_SOURCE_COORDINATE_SYSTEM:
        raise AssertionError("info.json 的 action_source_coordinate_system 与代码契约不一致")
    for key in ("action_reference_frame_definition", "training_action_transform"):
        if not str(info["ego_relation"].get(key, "")).strip():
            raise AssertionError(f"info.json 的 ego_relation.{key} 是空的")
    if int(info["ego_relation"]["max_simultaneous_latched"]) > 1:
        raise AssertionError(
            f"ego_relation.max_simultaneous_latched = "
            f"{info['ego_relation']['max_simultaneous_latched']} > 1 —— "
            f"「一次只能锁存一个物体」被破坏"
        )
    # OpenPI XRPipe Mode1 contract: fail at export time instead of allowing a
    # later trainer to reinterpret the fingertip midpoint as a wrist/palm TCP.
    declared = json.loads((destination / "meta/action_semantics.json").read_text(encoding="utf-8"))
    if declared != action_semantics():
        raise AssertionError(
            f"meta/action_semantics.json 与期望的 XRPipe Mode1 声明不一致: "
            f"{declared} != {action_semantics()}"
        )
    # 每段的 PICO OpenXR 右手世界系锚点必须落了盘 (没有它, 绝对量在跨段/真机部署时没有落点)。
    extraction = summary["extraction"]
    anchor_episodes = (extraction.get("pico_world_openxr_anchors") or {}).get("episodes") or {}
    for episode in episodes:
        anchor = anchor_episodes.get(episode["stem"])
        if not isinstance(anchor, list) or len(anchor) != 9 or not all(
                isinstance(v, (int, float)) for v in anchor):
            raise AssertionError(f"extraction_meta.json 里 {episode['stem']} 的锚点异常: {anchor!r}")
    if extraction.get("schema_version") != "xrpipe_v4":
        raise AssertionError(
            f"extraction_meta.json 的 schema_version "
            f"{extraction.get('schema_version')!r} != 'xrpipe_v4'"
        )
    if extraction.get("timestamp_semantics") != "uniform_frame_index_over_fps":
        raise AssertionError("extraction_meta.json 未声明严格的 frame_index/FPS 训练时间轴")
    for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        if info["features"][key]["names"] is not None:
            raise AssertionError(f"features[{key}].names 应为 null (s4 如此)")
    if info["features"][video_key]["info"]["video.codec"] != summary["extraction"]["observation"]["codec"]:
        raise AssertionError("features 里的 video.codec 与写出用的 codec 不一致")

    seen_index: list[int] = []
    for episode, row in zip(episodes, rows, strict=True):
        table = pq.read_table(row["parquet"])
        expect_order = ["observation.state", "observation.action_reference_tcp", "action", "timestamp",
                        "frame_index", "episode_index", "index", "task_index"]
        if table.column_names != expect_order:
            raise AssertionError(f"parquet 列序 {table.column_names} != s4 的 {expect_order}")
        length = row["length"]
        for name, source in (("observation.state", row["state"]),
                             ("observation.action_reference_tcp", row["reference"]),
                             ("action", row["action"])):
            got = np.asarray(table.column(name).to_pylist(), dtype=np.float32)
            if got.shape != source.shape or not np.array_equal(got, source):
                raise AssertionError(f"episode {row['episode_index']} 读回的 {name} 与写出的不一致")
        if not np.array_equal(table.column("frame_index").to_numpy(), np.arange(length)):
            raise AssertionError(f"episode {row['episode_index']} frame_index != arange(L)")
        if not np.array_equal(table.column("episode_index").to_numpy(),
                              np.full(length, row["episode_index"])):
            raise AssertionError(f"episode {row['episode_index']} episode_index 不是常量")
        task_index = table.column("task_index").to_numpy()
        if len(set(task_index.tolist())) != 1 or int(task_index[0]) != row["task_index"]:
            raise AssertionError(f"episode {row['episode_index']} task_index 异常")
        seen_index.extend(table.column("index").to_numpy().tolist())
        timestamp = table.column("timestamp").to_numpy()
        expected_timestamp = (np.arange(length, dtype=np.float64) / float(FPS)).astype(np.float32)
        if not np.array_equal(timestamp, expected_timestamp):
            max_error = float(np.max(np.abs(timestamp - expected_timestamp)))
            raise AssertionError(
                f"episode {row['episode_index']} timestamp 不是严格的 frame_index/{FPS}: "
                f"max_error={max_error:.3e}"
            )
        probe = probe_video(row["video"])
        if probe["frames"] != length:
            raise AssertionError(f"episode {row['episode_index']} 视频 {probe['frames']} 帧 != 标签 {length} 帧")
        if (probe["width"], probe["height"]) != (OBSERVATION_WIDTH, OBSERVATION_HEIGHT):
            raise AssertionError(
                f"episode {row['episode_index']} 视频 {probe['width']}x{probe['height']} "
                f"!= {OBSERVATION_WIDTH}x{OBSERVATION_HEIGHT}"
            )
        _check_video_frames(episode, row)

    if seen_index != list(range(summary["total_frames"])):
        raise AssertionError("全局 index 列不是 0..total_frames-1 的连续序列")

    task_lines = [json.loads(line) for line in
                  (destination / "meta/tasks.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if len(task_lines) != len(summary["tasks"]):
        raise AssertionError("tasks.jsonl 行数与唯一任务数不符")
    episode_lines = [json.loads(line) for line in
                     (destination / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if len(episode_lines) != len(rows):
        raise AssertionError("episodes.jsonl 行数与 episode 数不符")
    stats_lines = [json.loads(line) for line in
                   (destination / "meta/episodes_stats.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if len(stats_lines) != len(rows) or stats_lines[0]["stats"]["observation.state"]["count"] != [rows[0]["length"]]:
        raise AssertionError("episodes_stats.jsonl 异常")


def _check_video_frames(episode: dict, row: dict) -> None:
    """抽帧比对: 视频第 k 帧必须就是该观测源的第 k 帧 (按同一个缩放比到 640x480)。"""
    import cv2

    length = row["length"]
    picks = sorted({0, length // 3, (2 * length) // 3, length - 1})
    frames = episode["frames"]
    capture = cv2.VideoCapture(str(row["video"]))
    if not capture.isOpened():
        raise RuntimeError(f"opencv 打不开 {row['video']}")
    try:
        for k in picks:
            capture.set(cv2.CAP_PROP_POS_FRAMES, k)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"{row['video']} 第 {k} 帧读不到")
            # 参照帧按**同一个目标尺寸**缩一遍再比 (mp4 是 1080x810 缩到 640x480 编出来的,
            # 不缩就永远对不上)。阈值 20.0 不动: 缩放在两边都做, 差异仍然只来自那一次有损编码。
            source = fit_frame(frames.read(k), (frame.shape[1], frame.shape[0]))
            if source.shape != frame.shape:
                raise AssertionError(f"{episode['stem']} 第 {k} 帧尺寸 {source.shape} != 视频 {frame.shape}")
            diff = float(np.abs(source.astype(np.float32) - frame.astype(np.float32)).mean())
            if diff > 20.0:
                raise AssertionError(
                    f"{episode['stem']} 第 {k} 帧与 {frames.label(k)} 差太多 (平均 {diff:.1f}) —— "
                    f"视频与标签可能不同源或不同序"
                )
    finally:
        capture.release()


# ---------------------------------------------------------------- 可视化 (--vis)

# 裁剪时间轴与 HUD 的配色 (BGR —— cv2 的约定)。
_COLOR_KEPT = (80, 175, 80)
_COLOR_BRIDGED = (0, 165, 255)
_COLOR_DROPPED = (95, 95, 95)
_COLOR_INK = (235, 235, 235)
_COLOR_DIM = (150, 150, 150)
_COLOR_BG = (32, 32, 32)
_COLOR_PANEL = (40, 40, 40)
_FONT = None  # 在函数里取 cv2.FONT_HERSHEY_SIMPLEX (模块级 import cv2 会拖慢普通跑法)

_LABEL_W = 300   # 左边的文字区宽度 (px)
_CELL_W = 2      # 一个源帧占的宽度 (px)
_CELL_H = 26     # 一行的高度 (px)
_ROW_GAP = 8
_HUD_H = 112     # 标题 + 4 行 (末行是 action - reference 的相对量, 见 _relative_hud)


def render_vis(destination: Path, summary: dict, episodes: list[dict]) -> dict:
    """`--vis`: 把「裁了什么」画出来, 落到 `destination/vis/`。

      `crop_timeline.png`          每段采集一行: 每个**源帧**按保留 / 桥接 / 挖掉上色,
                                   最下面再压一条按 episode 顺序拼好的总览
      `episode_%06d_check.mp4`     导出的那份 mp4 **逐帧解码** + 底部 HUD (宽度 x 112):
                                   episode 内 k、对应源帧号、处在第几个有效段、这一帧
                                   是不是拼接缝 / 桥接段, state/action 的平移与开闭, 以及
                                   `action - reference` 的相对量 (绝对值看不出对错, 见
                                   `_relative_hud`)

    check 视频刻意从**导出后的 mp4** 解码, 而不是回头读 `observation/*.png`: 这样看到
    的就是训练时真正会读到的那串像素。
    """
    import cv2

    global _FONT
    _FONT = cv2.FONT_HERSHEY_SIMPLEX
    vis_dir = Path(destination) / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    timeline = vis_dir / "crop_timeline.png"
    if not cv2.imwrite(str(timeline), _crop_timeline(episodes)):
        raise RuntimeError(f"写不出 {timeline}")
    checks = []
    for row, episode in zip(summary["rows"], episodes, strict=True):
        path = vis_dir / f"episode_{row['episode_index']:06d}_check.mp4"
        _episode_check(cv2, path, episode, row)
        checks.append(str(path))
    return {"timeline": str(timeline), "checks": checks}


def render_from_disk(destination: str | Path) -> dict:
    """`--vis`: **只读已冻结的数据集**, 重建 `render_vis` 要的那两个结构后出片。

    刻意不重跑 `prepare()`: 那要读 step2/step3 的产物, 而数据集一旦与产物脱钩 (或产物被
    `--force` 覆盖成了另一组), 就画不出「这份数据集**当时**裁了什么」。这里只读数据集自己的
    `meta/info.json` + `meta/episodes.jsonl` + `extraction_meta.json` + 每个 parquet ——
    于是 step1-3 的产物都不在时也能出片, 且只往 `vis/` 写, 与数据产物完全不相交。
    """
    import pyarrow.parquet as pq

    destination = Path(destination)
    info_path, extraction_path = destination / "meta/info.json", destination / "extraction_meta.json"
    episodes_path = destination / "meta/episodes.jsonl"
    for path in (info_path, extraction_path, episodes_path):
        if not path.is_file():
            raise FileNotFoundError(f"缺 {path} —— 这不是一个已导出的数据集 (先跑 step4)")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    lines = [json.loads(line) for line in
             episodes_path.read_text(encoding="utf-8").splitlines() if line]
    # HUD 上的 `action` 标签必须按**这份数据集自己的**元数据打: 盘上可能躺着旧语义的数据集
    # (相机系/相对参照物), 拿今天的新口径去标它就是把标签画错。缺字段的老数据集按 None 走,
    # 只报警告 —— 这正是原来那版 `--vis` 会 KeyError 的那条路径。
    reference_frame = (info.get("ego_relation") or {}).get("action_reference_frame")
    if reference_frame != ACTION_REFERENCE_FRAME:
        print(f"  ⚠ {destination}: ego_relation.action_reference_frame 是 {reference_frame!r}, "
              f"不是当前口径的 {ACTION_REFERENCE_FRAME!r} —— HUD 的 action 标签按它自己的元数据"
              f"打印 (旧数据集的 action 不是 PICO OpenXR 右手世界系的绝对位姿), 别按今天的语义读",
              file=sys.stderr)

    # `render_vis` 只用这几项: `_crop_timeline`/`_episode_check` 读 split 的 n_frames_source
    # 与 kept_invalid_runs, 以及 keep_index / n_frames / stem; HUD 还要 reference (算相对量)
    # 与 action_reference_frame (打标签)。
    episodes = [{
        "stem": str(entry["source_stem"]),
        "keep_index": np.asarray(entry["keep_index"], dtype=np.int64),
        "n_frames": int(entry["n_frames"]),
        "action_reference_frame": reference_frame,
        "split": {"n_frames_source": int(entry["n_frames_source"]),
                  "kept_invalid_runs": [[int(a), int(b)] for a, b in entry["kept_invalid_runs"]]},
    } for entry in extraction["episodes"]]
    if len(episodes) != len(lines):
        raise AssertionError(
            f"episodes.jsonl 有 {len(lines)} 行, extraction_meta.json 有 {len(episodes)} 个 episode"
        )

    rows = []
    for line, episode in zip(lines, episodes, strict=True):
        episode_index = int(line["episode_index"])
        fmt = {"episode_chunk": episode_index // int(info["chunks_size"]),
               "episode_index": episode_index, "video_key": _video_key()}
        table = pq.read_table(destination / info["data_path"].format(**fmt))
        rows.append({
            "episode_index": episode_index,
            "length": int(line["length"]),
            "video": destination / info["video_path"].format(**fmt),
            "state": np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32),
            "reference": np.asarray(table.column("observation.action_reference_tcp").to_pylist(),
                                    dtype=np.float32),
            "action": np.asarray(table.column("action").to_pylist(), dtype=np.float32),
        })
    missing = [str(row["video"]) for row in rows if not row["video"].is_file()]
    if missing:
        raise FileNotFoundError(f"数据集里缺视频: {missing[:3]}")
    return render_vis(destination, {"rows": rows}, episodes)


def _crop_timeline(episodes: list[dict]) -> np.ndarray:
    """源帧轴上色 + 一条 episode 顺序的总览 + 图例。"""
    import cv2

    n_source = max(int(e["split"]["n_frames_source"]) for e in episodes)
    n_packed = sum(int(e["n_frames"]) for e in episodes)
    rows_h = len(episodes) * (_CELL_H + _ROW_GAP)
    header, order_label, order_gap, legend_h = 56, 22, 16, 5 * 22 + 8
    height = header + rows_h + order_gap + order_label + _CELL_H + 16 + legend_h
    width = _LABEL_W + max(n_source, n_packed) * _CELL_W + 20
    canvas = np.full((height, width, 3), _COLOR_BG, np.uint8)

    cv2.putText(canvas, "crop timeline", (12, 26), _FONT, 0.7, _COLOR_INK, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"{len(episodes)} episode(s), 1 cell = 1 source frame",
                (12, 46), _FONT, 0.42, _COLOR_DIM, 1, cv2.LINE_AA)

    for index, episode in enumerate(episodes):
        y = header + index * (_CELL_H + _ROW_GAP)
        split = episode["split"]
        n = int(split["n_frames_source"])
        kept = set(int(v) for v in episode["keep_index"])
        bridged: set[int] = set()
        for a, b in split["kept_invalid_runs"]:
            bridged.update(range(int(a), int(b) + 1))
        for i in range(n):
            color = _COLOR_BRIDGED if i in bridged else (_COLOR_KEPT if i in kept else _COLOR_DROPPED)
            x = _LABEL_W + i * _CELL_W
            canvas[y:y + _CELL_H, x:x + _CELL_W] = color
        _mark_splices(canvas, y, _LABEL_W, episode["keep_index"], 0)
        cv2.putText(canvas, f"ep{index} {episode['stem']}", (8, y + 12), _FONT, 0.42,
                    _COLOR_INK, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{len(kept)}/{n} kept, {len(bridged)} bridged", (8, y + 25),
                    _FONT, 0.38, _COLOR_DIM, 1, cv2.LINE_AA)
        if n < n_source:   # 各段源长可以不同, 右边留白 = 该段确实短这么多
            canvas[y:y + _CELL_H, _LABEL_W + n * _CELL_W:width - 20] = _COLOR_BG

    y = header + rows_h + order_gap
    cv2.putText(canvas, "episode order", (8, y + 18), _FONT, 0.45, _COLOR_INK, 1, cv2.LINE_AA)
    offset = 0
    for index, episode in enumerate(episodes):
        length = int(episode["n_frames"])
        x0 = _LABEL_W + offset * _CELL_W
        canvas[y:y + _CELL_H, x0:_LABEL_W + (offset + length) * _CELL_W] = _episode_color(index)
        _mark_splices(canvas, y, _LABEL_W, episode["keep_index"], offset)
        offset += length

    legend = [
        (_COLOR_KEPT, "kept      valid frame, lands in the episode"),
        (_COLOR_BRIDGED, "bridged   invalid run < max_invalid_gap, kept in place"),
        (_COLOR_DROPPED, "dropped   invalid run excised, both ends spliced together"),
        (None, "white tick = splice boundary (episode-local frame jumps here)"),
        (None, "coloured bar = packed episode order, same colour as its row above"),
    ]
    y = header + rows_h + order_gap + order_label + _CELL_H + 16
    for i, (color, text) in enumerate(legend):
        row_y = y + i * 22
        if color is not None:
            canvas[row_y:row_y + 12, 12:32] = color
        cv2.putText(canvas, text, (40, row_y + 11), _FONT, 0.42, _COLOR_DIM, 1, cv2.LINE_AA)
    return canvas


def _episode_color(index: int) -> tuple[int, int, int]:
    """按 episode 序号取一个可区分的颜色 (BGR)。"""
    hue = int(180 * (index * 0.6180339887 % 1.0))
    pixel = np.uint8([[[hue, 190, 235]]])
    import cv2

    bgr = cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _mark_splices(canvas: np.ndarray, y: int, origin: int, index: np.ndarray, offset: int) -> None:
    """在 episode 局部段边界处画一道白竖线 (竖线位置由 `origin + (offset+a)*_CELL_W` 定)。"""
    import cv2

    for a, _ in _runs(np.asarray(index))[1:]:
        x = origin + (offset + int(a)) * _CELL_W
        cv2.line(canvas, (x, y - 3), (x, y + _CELL_H + 2), _COLOR_INK, 1)


def _episode_check(cv2, destination: Path, episode: dict, row: dict) -> None:
    """导出的 mp4 逐帧解码 + 底部 HUD。"""
    length = int(row["length"])
    index = np.asarray(episode["keep_index"], dtype=np.int64)
    runs = _runs(index)
    bridged: set[int] = set()
    for a, b in episode["split"]["kept_invalid_runs"]:
        bridged.update(range(int(a), int(b) + 1))
    # k -> 处在第几个有效段 (段尾且后面还有段 = 这里发生了拼接)
    segment_of = np.zeros(length, dtype=np.int64)
    for s, (a, b) in enumerate(runs):
        segment_of[a:b + 1] = s
    tails = {b for _, b in runs[:-1]} if len(runs) > 1 else set()

    capture = cv2.VideoCapture(str(row["video"]))
    if not capture.isOpened():
        raise RuntimeError(f"opencv 打不开 {row['video']}")
    written = 0
    sink = None
    try:
        while written < length:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if sink is None:
                # HUD 按**解码帧自己的宽度**画, 不放大到 1080 —— 这样 check 视频里就是
                # 训练时真正读到的那 640x480 像素, 原样 1:1。
                height, width = frame.shape[:2]
                sink = BgrSink(destination, codec=DEFAULT_CODEC,
                               size=(width, height + _HUD_H), fps=FPS)
            state, action = row["state"][written], row["action"][written]
            selframe = int(index[written])
            lines = [
                f"k={written}/{length - 1}  src={selframe:05d}  "
                f"seg={int(segment_of[written]) + 1}/{len(runs)}  "
                f"splice={'yes' if written in tails else 'no'}  "
                f"bridged={'yes' if selframe in bridged else 'no'}",
                _state_hud("state ", state, per_object=True),
                # action 的前 9 维是**手在世界系里的绝对位姿** —— 别按物体切 (只有一个平移)。
                # 绝对值本身在 HUD 上看不出对错 (整段是同一个常量偏移), 所以必须同时打出
                # action - reference 的**相对量**。
                _state_hud(f"action {episode.get('action_reference_frame') or '?'}",
                           action, per_object=False),
                # 注意传的是**整段**数组 (row 里的那两列), 不是上面那个单帧的 `action`:
                # 这一行的价值全在「两列长度相同」这条交叉检查上。
                _relative_hud(row.get("reference"), row.get("action"), written),
            ]
            hud = np.full((_HUD_H, frame.shape[1], 3), _COLOR_PANEL, np.uint8)
            cv2.putText(hud, f"ep{row['episode_index']} {episode['stem']}", (10, 20),
                        _FONT, 0.5, _COLOR_INK, 1, cv2.LINE_AA)
            for i, text in enumerate(lines):
                cv2.putText(hud, text, (10, 42 + i * 20), _FONT, 0.45, _COLOR_DIM, 1, cv2.LINE_AA)
            sink.write(np.vstack([frame, hud]))
            written += 1
    finally:
        if sink is not None:
            sink.close()
        capture.release()
    if written != length:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"{row['video']} 只解出 {written}/{length} 帧 —— 视频与标签帧数不符")


def _state_hud(tag: str, vector: np.ndarray, *, per_object: bool, max_objects: int = 3) -> str:
    """HUD 里一行 state/action。

    `per_object=True` (state = 9N 关系 + 爪): 逐物体打印该物体在手系里的平移。
    `per_object=False` (action = 9 位姿 + 爪): 只打印手自己**在世界系**里的平移 ——
    调用方在 tag 里标出帧种类 (例如 `action pico_world_openxr`)。
    物体多了就截断, 免得画出画面。
    """
    body = np.asarray(vector, dtype=np.float64)
    if not per_object:
        return f"{tag} t=({body[0]:+.3f},{body[1]:+.3f},{body[2]:+.3f})  grasp_next={int(body[-1])}"
    count = max(0, (len(body) - 1) // 9)
    parts = []
    for j in range(min(count, max_objects)):
        block = body[9 * j:9 * j + 3]
        parts.append(f"o{j + 1}=({block[0]:+.3f},{block[1]:+.3f},{block[2]:+.3f})")
    if count > max_objects:
        parts.append(f"+{count - max_objects} more")
    return f"{tag} {' '.join(parts)}  grasp={int(body[-1])}"


def _relative_hud(reference, action: np.ndarray, k: int) -> str:
    """HUD 最后一行: `inv(reference[k]) @ action[k]` 的平移与转角。

    action/reference 都是世界系里的**绝对**位姿, 绝对值在画面上看不出对错 (整段是同一个
    ~0.4 m 的常量偏移, 标签写错也照样「看着正常」), 所以必须同时给出这个**相对量** ——
    它与每帧的手部行程同量级 (本段实测 p50 3.6 mm), 明显偏大就说明帧种类或对齐出了问题。
    用 `vec9_to_transform` 解回 SE(3), 与 openpi 训练期现算相对动作是同一套解码: 解错这里先显形。

    `reference`/`action` 都是**整个 episode** 的 (L,9) / (L,10) 数组 (两条路径的 row 里都有
    这两列); 没有就明说。**别把单帧的 `action` 传进来** —— 那个是 (10,), 长度永远对不上,
    这一行就会恒打成 `n/a`, 而它本来正是「两列同源同序」的唯一一处交叉检查。
    """
    if reference is None or action is None:
        return "action - reference: n/a (这份 parquet 里缺 reference/action 列)"
    from ego_relation.contracts.se3 import compose, invert, rotation_angle_deg, vec9_to_transform

    reference = np.asarray(reference, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    if len(reference) != len(action) or k >= len(reference):
        return (f"action - reference: n/a (reference {len(reference)} 帧 "
                f"vs action {len(action)} 帧, 或 k={k} 越界)")
    current = vec9_to_transform(reference[k])
    target = vec9_to_transform(action[k][:9])
    relative = compose(invert(current), target)
    return (f"action - reference  dt={np.linalg.norm(relative[:3, 3]) * 1000:6.1f} mm  "
            f"dR={rotation_angle_deg(relative[:3, :3]):5.2f} deg")


# ---------------------------------------------------------------- 入口


def run(paths_list: list[PipePaths], destination: str | Path, *,
        tasks: dict[str, str], max_invalid_gap: int = MAX_INVALID_GAP,
        robot_type: str = DEFAULT_ROBOT_TYPE, codec: str = DEFAULT_CODEC,
        observation: str = "step3", force: bool = False, quiet: bool = False,
        vis: bool = False) -> dict:
    """把若干段采集写成一个数据集 (每段 1 个 episode, 按传入顺序编号)。

    `tasks` 是 **stem -> 语言指令**。没有兜底: 缺哪个 stem 就报哪个 —— 从检测提示词
    (「small white earbud case」) 推 `pick up the ...` 那种做法看起来能跑, 但 task
    到底是不是训练要的那句话就说不清了, 所以这里刻意不做。

    `observation` 选观测帧的来源 (`--observation`), 见 `prepare()`: 它只影响视频,
    标签列与它无关。
    """
    if not paths_list:
        raise ValueError("没有可导出的采集")
    missing = [p.stem for p in paths_list if not str(tasks.get(p.stem, "")).strip()]
    if missing:
        raise ValueError(f"这些采集没给 --task: {missing} (step4 不接受从检测提示词推导)")

    episodes = []
    for paths in paths_list:
        episode = prepare(paths, task=tasks[paths.stem], max_invalid_gap=max_invalid_gap,
                          observation=observation)
        episode["source_dir"] = str(paths.outdir)
        episodes.append(episode)
        if not quiet:
            print(
                f"  step4 {paths.stem}: 保留 {episode['n_frames']}/"
                f"{episode['split']['n_frames_source']} 帧, 桥接 {episode['split']['kept_invalid_runs']}, "
                f"挖掉 {episode['split']['dropped_invalid_runs']}, 观测 {episode['observation']}"
            )
    summary = write_dataset(destination, episodes, robot_type=robot_type, codec=codec,
                            force=force, quiet=quiet)
    if vis:
        summary["vis"] = render_vis(Path(destination), summary, episodes)
    return summary
