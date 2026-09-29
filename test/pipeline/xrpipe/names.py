"""向量各维的名字。

`state` 的前缀与 s4 逐字一致 (`right_tcp_to_<slug>`), 只有**个数**变了: s4 是左右手 x
{holder,red,yellow} 三个物体, 本数据是右手 + N 个物体, 所以 relation 的 9 维块 = N 个
(按 obj 序拼) 而不是六个, `state` = 9N+1。

`reference`/`action` 与 s4 **逐字同一份** (`right_tcp_absolute_current` /
`right_tcp_absolute_target`): 参考系换成了 PICO OpenXR 右手世界系 (段内不动), 但「存的是绝对位姿、
相对动作留到训练期现算」这件事与 s4 一样, 所以名字不动 (见 `episode.build_episode_arrays`
的 docstring)。列名 (`observation.action_reference_tcp` / `action`) 与列序/维度也没动。

**一处刻意的差异**: s4 的 `<slug>` 用语义名 (`holder`/`red`/`yellow`), 我们统一用**实例 id**
(`obj1`/`obj2`/`obj3`) —— 与 `info.json` 的 `object_order` / `object_categories` 同一套命名,
免得同一个物体在元数据里有两种叫法 (参考那边这两处本身就是两套, 见
`s4_lerobot_export/lerobot.py:134-143` vs `:21-22`)。

`_pose_names` 直接从 `ego_relation.s4_lerobot_export.lerobot` import 复用, 不另抄一份 ——
名字漂移是 openpi 侧最难查的一类错。
"""

from __future__ import annotations

from . import INSTANCE_ID, bootstrap


def _pose_names(stem: str) -> list[str]:
    bootstrap()
    from ego_relation.s4_lerobot_export.lerobot import _pose_names as s4_pose_names

    return list(s4_pose_names(stem))


def relation_names(instance_id: str = INSTANCE_ID) -> list[str]:
    """`observation.state` 前 9 维: 物体在该手 TCP 系里的位姿。"""
    return _pose_names(f"right_tcp_to_{instance_id}")


def state_names(*instance_ids) -> list[str]:
    """`observation.state` 的 9N+1 个名字: N 个物体的 9 维块 (按 obj 序) + 末尾的爪。

    单物体时与 `state_names()` 逐字相同 (只是不再带 `right_tcp_to_obj1_` 之外的前缀差异)。
    """
    ids = [str(v) for v in (instance_ids or (INSTANCE_ID,))]
    names: list[str] = []
    for instance_id in ids:
        names.extend(relation_names(instance_id))
    return names + ["right_grasp_binary"]


def reference_names() -> list[str]:
    """`observation.action_reference_tcp`: **本帧**指尖中点在 PICO OpenXR 右手世界系里的绝对位姿。

    与 s4 的 `_pose_names("right_tcp_absolute_current")` 逐字相同 —— s4 的「绝对」是
    `g1_base` 系, 我们的是 `pico_world_openxr` 系, 差异写在 `info.json` 的 `ego_relation` 块里。
    """
    return _pose_names("right_tcp_absolute_current")


def action_names() -> list[str]:
    """`action`: 世界系里的**下一帧**绝对位姿 + 下一帧开闭。"""
    return _pose_names("right_tcp_absolute_target") + ["right_grasp_binary"]


# 模块级常量: 这几个列表是纯字符串, 提前算好省得每处都 import ego_relation。
# `STATE_NAMES` = 单物体那一份; 多物体用 `state_names("obj1", "obj2")` 现算 (见 export.info_dict)。
STATE_NAMES = state_names()
REFERENCE_NAMES = reference_names()
ACTION_NAMES = action_names()
