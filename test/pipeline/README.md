# xrpipe —— PICO ego 采集 → LeRobot（对齐 `ego_relation_policy`）

把 `test/` 下的采集（双目拼接 MP4 + 90Hz 跟踪 txt）整理成一条 4-step pipeline，产出 key 与
`ego_relation_policy/src/ego_relation/s4_lerobot_export/lerobot.py` 对齐的 LeRobot v2.1 数据集，供 openpi 训练。

`test/pipeline/` 自包含几何、分割、深度、位姿、修复、Piper 资源与四步编排，
运行时不再导入旧 work 目录。step 之间只通过 `out/<stem>/<step>/` 的产物耦合，不互相重跑。

## 跑法

首次使用先执行 `bash tools/seg_setup.sh`，环境、模型缓存和默认标定都位于 `test/pipeline/`。
Step1 在未传 `--lag` 且缺少对齐产物时，会自动生成 `out/align_<stem>.json`。

```bash
bash run.sh step1 20260920_111342
bash run.sh step2 --stems 20260920_111342 --prompt "small white earbud case|computer mouse"
bash run.sh step3 --stems 20260920_111342 --only-keep
bash run.sh step4 --stems 20260920_111342 --dataset demo --task "pick up the earphone case"
bash run.sh all   --stems 20260920_111342,20260920_111300 --dataset smoke \
                  --prompt "small white earbud case|computer mouse" --prompt "<另一段的 A|B>" \
                  --task   "pick up the earphone case" --task "put the mug on the shelf"
```

- `--prompt` **至少 2 段**（竖线分隔）。这是**任务口径**（「抓物体1、放到物体2上」，承载物也要框出来），
  与 `action` 无关；只给 1 段在 `cli._check_prompts` 里**开跑前**报错，不进 DINO+SAM2。
- stem 也可用位置参数给多个：`bash run.sh step1 20260920_111342 20260920_111300`。
- 每步加 `--vis` 额外出一份可视化（见「可视化」），**用 stem 选要看的 episode**。
- 只要观测不要夹爪（跳过整个 step3）就把 step4 换成 `--observation step2`（见「观测源」）。

## 数据事实

- MP4 是**双目拼接**（2160×810 = 左眼 eye0 + 右眼 eye1），但标签与观测**只用左眼**（1080×810），
  全程只有 `camera0` 一个相机系。
- 末端是**二值爪**，不是灵巧手：`grasp_closed` ∈ {0,1}。
- 只有**右手**；物体 N 个，数据集里叫 `obj1..objN`（id 表到 `obj3`）。
- 一次采集 = **一个 episode**（见「裁剪」）。

### 检测提示词 ≠ 语言指令

| | 是什么 | 在哪传 | 进哪 |
|---|---|---|---|
| step2 `--prompt`、step3 `--mask-prompt` | **检测提示词**：给 GroundingDINO / SAM2 的文本，决定框出哪个物体、框出手 | 跑 step2 / step3 时按段传 | 只进 seg 产物与 `state_action.json` 的 `categories`、`observation.json` 的 `mask_prompt`（都是溯源） |
| step4 `--task` | **openpi 训练用的语言指令** | 跑 step4 时传，可逐段 | `meta/tasks.jsonl`、parquet 的 `task_index`、`episodes.jsonl` 的 `tasks` |

两者**不同名、不互相兜底**：step2 不产出任何 task 字符串（`state_action.json` 里没有 `instruction`
这个 key），step4 的 task 也决不从检测提示词推导（没有 `pick up the <prompt>` 这种兜底）。
不给 `--task` 就直接报错退出，不产出半个数据集。

映射规则相同：给 1 个 → 所有段共用；给 N 个 → 按 stem 顺序一一对应；给 0 个 → 报错。

## 坐标系与语义

### 三个坐标系

| 量 | 参考系 |
|---|---|
| 观测帧 | `camera0`（左眼，1080×810） |
| 手位姿（`T_camera0_midpoint`） | `camera0` |
| `observation.state` 的物体块 | **手（指尖中点）系** —— `inv(T_camera0_midpoint) @ T_camera0_object`，即 s4 的 `T_tcp_object` |
| `observation.action_reference_tcp`、`action` | **PICO OpenXR 右手世界系**（runtime 报的 tracking space，段内不动） |

只有「手位姿」在相机系里；`state` 是它派生出来的相对量（物体在手系），所以物体块**不在**相机系里。

末端 = HumanEgo 的 `MidpointFrameBuilder`：原点 = 拇指尖 / 食指尖的中点，`x = 食指根 − 拇指根`，
`y` 由两手根中点与腕的 Gram-Schmidt 得到，`z = x × y`。**不是** TCP / 手腕语义。

### state / action 的定义

设 `T_u = T_world_midpoint`（Unity 左手世界系 ← 指尖中点）、`M = diag(1,1,-1)`、
`T_w = diag(M,1) @ T_u @ diag(M,1)`（OpenXR 右手世界系 ← 指尖中点）、`T_m = T_camera0_midpoint`、
`T_o = T_camera0_object`、`L` = episode 帧数：

- `state`（9N+1）= N 个 `vec9(inv(T_m[t]) @ T_o[t, j])` 块（按 obj 序拼接，即「物体在手系里」）+ 本帧开闭；
- `observation.action_reference_tcp`（9）= `vec9(T_w[t])` —— **本帧手在世界系的绝对位姿**；
- `action`（10）= `vec9(T_w[t+1])` + **下一帧**开闭（末帧重复自身，与 s4 的 `min(t+1, T-1)` 同义）。

于是 `action[t, :9] == reference[t+1]`，而 `inv(reference[t]) @ action[t]` = **手自身在两帧间的相对运动**
—— 这正是 openpi 训练期要现算的那个量。**相对动作不进数据集。**

编码统一用 `ego_relation.contracts.se3.transform_to_vec9`（`[tx,ty,tz,R[:,0],R[:,1]]`，米）。

**为什么是绝对量**：与 s4 的 `right_tcp_absolute_current` / `right_tcp_absolute_target` 逐字同构，
差别只在**落点**（s4 映射到 G1 base，这里是 PICO OpenXR 右手世界系）。列名（`observation.action_reference_tcp`）、
列序、维度都没动；`state` 里 s4 的 `right_tcp_to_<slug>` 也照旧，只是 `<slug>` 用实例 id。

### 世界系：逐段一个

原点由 runtime 定（实测首帧平移 ≈ (0.093, −0.005, −0.055) m，即启动时刻头 / 眼附近，**不是地板系**），
本 pipeline **不做任何重新归零**。⇒ **跨 episode 的 `action` 绝对值不可比**，同段内可比。

这不是精度问题：`state` 是 `inv(T_m) @ T_o`，对共同刚体变换不变 ⇒ **世界系原点在模型输入里完全不可观测**，
于是同一任务在不同 episode 里被标成不同的数，而没有任何输入能解释这个差异。s4 敢用绝对量，是因为它显式把
demo 的 frame-0 pelvis 映射到 G1 base，本 pipeline **没有**这个映射。
每段锚点（该段首帧手的绝对位姿）落在 `extraction_meta.json` 的 `pico_world_openxr_anchors`，真机部署从这里接。

**轴向 / 手性已声明并转换**：原始 tracking 是 Unity 左手系（X右/Y上/Z前），Step2 仅对
`action` 与 `observation.action_reference_tcp` 应用 `t'=M·t, R'=M·R·M`，得到 OpenXR 右手系
（X右/Y上/Z后）。`state`、相机几何和物体关系不做此转换。

**录制中途 recenter**：世界系被重置时所有 world 位姿同时左乘同一个 `J`，于是
`T_camera0_midpoint = T_camera0_world @ T_world_midpoint` **逐字不变** —— 相机系的自检照样通过
（实测最大差 2.8e-17，而世界系里同一帧动了 400 mm）。只有 `check_world_motion` 能看见它。

### XRPipe Mode1 契约：`meta/action_semantics.json`

```json
{"schema_version": "xrpipe_action_v1",
 "mode": "xrpipe_mode1",
 "stored_action": "absolute_next_target",
 "reference_field": "observation.action_reference_tcp",
 "reference_frame": "pico_world_openxr",
 "coordinate_system": "openxr_rh_x_right_y_up_z_back",
 "control_point": "right_thumb_index_fingertip_midpoint",
 "relative_formula": "inv(reference[t]) @ action[t+k]"}
```

**不写这份声明，openpi 会直接拒收数据**。XRPipe 使用独立的
`xrpipe_mode1_rel_shared` preset；它保持指尖中点控制点，不进入 Unitree 的
`wrist_yaw_tcp -> palm` 变换。相对动作在训练期由当前 reference 与绝对 target 现算。
`verify_dataset` 会断言完整契约。

**训练侧落点**：`training_action_transform: "deferred: inv(T_current) @ T_absolute_target"` 里的 `T_current`
必须取 `observation.action_reference_tcp` 这一列。XRPipe Mode1 的独立输入变换已固定这样做；
`observation.state` 只承载物体关系与当前夹爪，不会被当成动作参考。

训练侧使用独立 preset（先统计、再训练）：

```bash
cd /home/skw/humanego/openpi
uv run scripts/compute_xrpipe_norm_stats.py \
  --preset xrpipe_mode1_rel_shared --dataset /path/to/lerobot/dataset
uv run scripts/train_xrpipe.py \
  --preset xrpipe_mode1_rel_shared --dataset /path/to/lerobot/dataset --exp-name <name>
```

旧 `xrpipe_v3` 数据只需用已有 Step1–3 产物重新执行 Step4；不需要重跑 SAM2、深度或 ICP。

## 四个 step

### step1 — 对齐 + 5 关键点 → 左眼相机下的指尖中点位姿

`out/<stem>/step1/{reel.npz, reel.json}`

Step1 自动调用 `xrhand.align` 生成/复用 lag，再由 `tools/overlay.py::Reel` 建立 30Hz 帧映射、
`xrhand.gripper` 的 5 关键点（PICO 26 点里的 1/3/7/5/10）与 `xrrel.relation.hand_states`（世界 → 左眼相机）。
启动时跑一次 `xrrel.relation.self_check_camera0`，它把 `camera_from_world` 与 `xrhand.camera.Projector`
逐点比对并要求 < 1e-8 —— 这是防「物体一个系、手另一个系」那类老坑的锁，别删。

关键 key：`T_camera0_midpoint`、**`T_world_midpoint`**（原始 Unity 位姿）与 **`T_pico_world_openxr_midpoint`**（action/reference 来源）、
`T_camera0_world`、`pose_valid`、`grasp_closed`、`timestamp_ns`、`eff_f/eff_cx/eff_cy`。

### step2 — DINO+SAM2 物体标定 → 物体位姿 → state / action

`out/<stem>/step2/{seg_<stem>/, pose/, relation/, state_action.npz, state_action.json}`

- **2a 分割**：`tools/seg_object.py` 的 `stage_frames/image/video` 原样驱动（GroundingDINO 取提示帧的
  box/mask，SAM2 video predictor 整段传播）。多物体时内层 `--prompt` **重复传**（逐物体一份 `masks/obj<i>/`）。
- **2b 物体位姿**：`xrrel.stereo`（SGBM 深度，与物体无关 → **只跑一次**）→ `xrrel.lift.build_clouds`
  （mask+深度反投影 → 点云，逐物体一份）→ `xrrel.objectpose.estimate/save`（参考帧 PCA 定向 + 逐帧 ICP
  + 门控 + 抓握锁存手推，**逐物体各跑一次**）。参考帧固定为 SAM2 起始帧，所有物体共用。
- **2c state / action**：`T_right_midpoint_object = compose(invert(T_camera0_midpoint), T_camera0_object)`，
  即 s4 的 `relation_direction: "T_tcp_object"`；`reference`/`action` 走 `T_pico_world_openxr_midpoint = M @ T_world_midpoint @ M`
  （原始 `T_world_midpoint` 保留用于世界链校验）。

方向容易搞反：`T_tcp_object` 与 `T_object_tcp` 的**平移范数相同**，只看平移区分不出来。
`step2._check_direction` 因此**逐物体**断言 (a) state 的第 j 个 9 维块等于 `T_right_midpoint_object[:, j]`，
(b) 与反方向的旋转明显不同。

**只重算 2c**：`--reassemble` 复用盘上已有的 `seg_<stem>/` 与物体位姿，只跑 `assemble + write_report`
（0.5 s 量级）。参考系 / 编码这类语义改动**只影响 2c**，而 `--force` 会一路传进 `segment` / `object_pose`，
把 SAM2 + SGBM + ICP 整条重跑几十分钟。

#### 抓握锁存

**规则**：`夹爪闭合 ∧ 手原点 ↔ 物体位姿原点 < --latch-distance`（默认 **0.05 m**）才锁存。锁存后这段时间的
物体位姿由 `hand @ T_hand_object` 的闭式**手推**（不再跑 ICP），手张开就释放。

这就是参考**自己的**归属门 `perception.latch_distance_m`（`s2_object_relations/encoding.py:236-248`：
手闭合且还没拿东西时，把所有物体按距手排序，取最近的、eligible 且**未被占用**的，
`distances[candidate] <= latch_distance_m` 才认作握持）——**同一个机制换了个值**（参考默认 0.20）。

别和另一个距离量混：`perception.grasp_distance_m = 0.035` 量的是**拇指尖 ↔ 食指尖**（手捏没捏上），
本 pipeline **一点没读**（开闭来自 `xrhand/gripper.py` 自己的阈值）。

多物体时归属按**物体顺序**交接：前一个物体锁存的帧通过 `blocked` 传给下一个，于是同一帧
**至多一个物体被锁存**（否则闭合期间可见的物体会被一起冻结到手，成为「看起来有效」的错位姿）。
没抢到锁存的帧走**纯测量 / 无效**，不对抗视觉。

**「至多一个」是断言，不只是实现**：`step2._check_single_latch` 在组装时对全长断言一次，step4 写盘前对
裁剪后的 `latched` 再断言一次（同一个纯函数，`>1` 就抛）。计数落在 `info.json` 的 `latched_frames_total`
与 `extraction_meta.json` 每段的 `latched_frames` / `latched_frames_per_object`。

可预期的后果（不是回归）：**丢跟的握持段不再锁存**。111342 的 f243–273 那段握持距手 252~270 mm，
是丢跟后的陈旧位姿 —— 收紧到 5 cm 之后这段锁不上，宁可回到无效。

#### 两个守卫

- `check_world_chain`：`T_world == inv(T_camera0_world) @ T_camera0_midpoint`，**只对有效帧**断言
  （atol 1e-9）—— 防「reference/action 的世界系与 state 的相机系脱钩」。
- `check_world_motion`：世界系里手逐帧的平移 / 转角上限（**0.05 m / 15°**）。实测手每帧 max 22 mm /
  1.97°，而 recenter 跳变约 0.4 m，量级差约 20 倍。

**无效帧**：`pose_valid=false` 的帧 `T_world_midpoint` 在数据源里保持**单位阵**（而 `T_camera0_world`
是真值），不处理就是「指尖瞬移到世界原点」（≈0.4 m 跳变，比真实行程还大）并污染归一化统计。
照 HumanEgo-main 的 *Forward Fill Hand if momentarily missing tracking* 做**因果前向填充**
（`fill_invalid_poses`），填充帧数记在 `state_action.json` 的 `invalid_pose_fill`；
首个有效帧之前没有值可填的帧单独报出。

step2 按**全长**算（`keep = arange(N)`），裁剪交给 step4。

### step3 — 手 mask + 修复 + piper 夹爪 → 纯 RGB 观测帧

`out/<stem>/step3/{observation/%05d.png, observation.json}`

和 `xrrel.step3.run` 唯一的区别是**合成基座**：这里用原始左眼帧，不是那份 render 叠加视频
（关系视频宽 2160、高 `988 + 168×物体数`，里面已经烤进了各物体 XYZ 轴 / REL 面板 /
关键点 / mask 叠加，拿它当底图就永远去不掉）。
合成 = 原 RGB 上只替换手（手臂）区域的 LaMa 修复结果 + piper 夹爪 alpha 混合：

```
clean  = bg.mkv 第 k 帧                    # LaMa 修复后的左眼原 RGB
raw    = CameraRecord 的左半第 i 帧
canvas = np.where(hand_mask > 0, clean, raw)
+ piper.render(pose_from_tcp(T_camera0_midpoint[i], …))  # 夹爪
```

夹爪位姿**直接取 step1 的 `T_camera0_midpoint`**，不重推 —— step1 与 step3 看到的末端因此是同一个矩阵。
默认只 mask 手（`human hands .`），`--with-arm` 才连手臂一起；`--mask-prompt` 可以直接指定这条
**手部检测提示词**（它决定框出手还是连手臂，不是任务指令）。

帧名是 `observation/%05d.png`，编号是**源帧号**，所以 `--only-keep`（只合成 step4 会保留的帧）
纯属提速，输出与全跑一遍逐像素一致。

**这一步可以整步跳过**：step4 用 `--observation step2` 时直接用源 mp4 左半当观测，不读 step3 的任何产物。

### step4 — 裁剪 → 对齐 → 写出 LeRobot v2.1

`out/lerobot/<dataset>/`

`meta/{info.json,tasks.jsonl,episodes.jsonl,episodes_stats.jsonl,stats.json,action_semantics.json}`、
`data/chunk-000/episode_%06d.parquet`、`videos/chunk-000/observation.images.camera0/episode_%06d.mp4`、
`extraction_meta.json`（`--vis` 时另加 `vis/`，不属于 LeRobot 契约）。
手写 pyarrow + cv2，不引入 `lerobot` 库（没装，s4 也没装）；
命名 / 统计 / 编码直接 import s4 的 `_stats` / `_pose_names` / `VIDEO_KEY` 复用，不另抄一份。
`tasks.jsonl` 里的指令**只来自 `--task`**。

#### 裁剪（一次采集 = 一个 episode）

```
invalid_runs = 连续 state_valid=false 的段（含首尾）
keep_index   = 挖掉所有 len(run) >= --max-invalid-gap（默认 30）的段之后剩下的源帧号
```

- 长度 **< 30 的无效段保留**（原地桥接，帧不丢，只是这些帧的 `state` 不可信）。
- 长度 ≥ 30 的整段挖掉，前后剩下的帧**拼接进同一个 episode**，不新建 episode。
- 一次采集永远只产出 1 个 episode；裁完一帧不剩就报错跳过这段采集。

`state_valid = all(object_valid) & hand_valid`（**严格口径**：N 个物体**全部**有效才保留这一帧；
约简只在 `step2.assemble` 里做一次）。111342 两物体时的实况：
`invalid_runs = [[0,49],[95,126],[135,147],[183,192],[196,205],[218,273]]`，其中
`[135,147]`/`[183,192]`/`[196,205]` 共 33 帧桥接保留，`[0,49]`/`[95,126]`/`[218,273]` 共 138 帧挖掉
→ `keep_index = [50..217]`，**L = 136**。多物体时只要有一个物体无效这段就会被挖掉 ——
`state_action.json` 里逐物体打印有效率，正是为了看出「是哪个物体把帧裁掉了」。

#### 帧对齐

所有张量用**同一个 `keep_index`** 切片，`action[t]` 指向 **episode 内的下一帧**（不是源帧的下一帧）
—— 桥接段与拼接边界处这两者不同，是最容易错的地方。所以生成逻辑只写一份
（`episode.build_episode_arrays`）：step2 用 `arange(N)` 调它、step4 用真正的 `keep_index` 再调一次，
两处不可能走偏。

因为桥接 / 拼接，裁剪后的 `timestamp` **不再**等于 `k/30`，会带一次或多次跳变 —— 这正确反映了
episode 内部的真实时间，训练侧按 `timestamp` 对齐即可。

#### 写回断言（对文件断言，不是对内存数组）

1. `state` / `reference` **逐帧**等于 step2 全长数组按 `keep_index` 取的切片（它们不依赖邻居）；
2. **世界链**：`T_world == inv(T_camera0_world) @ T_camera0_midpoint`（只对有效帧）；再用 world 链
   `inv(T_world) @ inv(T_camera0_world) @ T_obj` 重算 `state` 的每个物体块，与 camera 链逐帧比
   —— 这是真的两条路径，能抓出 `T_camera0_world` 非刚体或取错帧；
3. `action` 只在「段尾」与源帧下一帧不同，段内必须相等 —— 段尾的下一帧落在下一段的开头；
4. `action[:, :9] == [reference[1:], reference[-1:]]`、
   `action[:, -1] == [state[:, -1][1:], state[:, -1][-1:]]`（爪在 `state` 的**最后一维**，不在第 9 维）；
5. 二值爪 ∈ {0,1}；`timestamp` 从 0 起严格递增；`len(state) == len(action) == 视频帧数`；
6. parquet 列序 / dtypes / features 的 `shape` 与 `names`（必须是**列表套列表**，openpi 读 `names[0]`）；
   三处 `names` 与 s4 的逐维名字**精确相等** —— 只查个数的名字对不上内容，一个「名字说绝对位姿、
   数值是相对位姿」的数据集在训练侧无从察觉；
7. 视频抽 4 帧与该观测源的第 k 帧比对（step3 源 = `observation/<keep_index[k]>.png`，
   step2 源 = 源 mp4 左半的第 `keep_index[k]` 帧；防「视频与标签不同源或不同序」）；
8. `meta/action_semantics.json` 的完整 XRPipe Mode1 契约、`extraction_meta.json` 的
   `schema_version == "xrpipe_v4"`
   与每段锚点。

数据集级：各 episode 的有序 `(instance_id, category)` 表必须完全相等；`action_reference_frame` /
`action_storage` 必须唯一；每段锚点是 9 个有限数。**世界系逐段不同是正常且必然的**，
这几条断言拦不住它，也不需要拦。

## 可视化

pipeline 跑的时候**不产出**可视化。加 `--vis` 才出，**用 stem 选要看的 episode**：

```bash
bash run.sh step1 --stems 20260920_111342 --vis
bash run.sh step2 --stems 20260920_111342 --prompt "A|B" --vis
bash run.sh step3 --stems 20260920_111342 --vis
bash run.sh step4 --stems 20260920_111342 --dataset demo --task "…" --vis
```

**`--vis` 只读已算完的产物，不重跑任何 step**，也不改任何数据产物。这是刻意的：可视化的意义就是
「检验这一步处理对不对」，所以它必须看到**已经落在盘上的那份结果**。每一步的「产物在就跳过」守卫
让这件事成立，于是这些命令都不需要 `--force`。Step2 `--vis` 只显示渲染进度，不逐帧刷相对位姿；
两个或多个物体是在**同一个视频、同一帧**里同时绘制，不会按物体拆成多个视频。

| step | 产物 | 看什么 |
|---|---|---|
| step1 | `out/<stem>/step1/gripper_<stem>.mp4` | 手部骨架 + 夹爪叠在 RGB 上，用来确认对齐与 5 关键点没歪 |
| step2 | `out/<stem>/step2/rel_<stem>/render/rel_<stem>.mp4` | 同一视频画面同时叠加所有物体的 mask、点云、物体 TCP、手指中点 TCP、连线/距离；底部每个物体各一块 REL 曲线面板 |
| step3 | `out/<stem>/step3/composite_piper.mp4`、`composite_sbs.mp4` | 前者就是进数据集的观测帧；后者 = 原始左眼帧 ‖ 合成帧，用来判断手修得干不干净 |
| step4 | `out/lerobot/<name>/vis/crop_timeline.png`、`vis/episode_%06d_check.mp4` | 时间轴：每个**源帧**按 保留 / 桥接 / 挖掉 上色；check 视频 = 导出的 mp4 逐帧解码 + 底部 HUD |

step4 那条走 `cli._vis_only()` / `step4.render_from_disk()`：数据集一旦存在且没给 `--force`，就只从盘上
重建渲染要的结构，一个字节的 parquet/mp4 都不动，因此**连 `--task` 都不用给**。

HUD 是标题 + 4 行：

```
ep0 20260920_111342
k=120/135  src=00202  seg=2/2  splice=no  bridged=yes
state  o1=(-0.021,-0.092,-0.160) o2=(+0.052,-0.214,-0.215)  grasp=0
action pico_world_openxr_openxr t=(+0.018,-0.260,-0.412)  grasp_next=0
action - reference  dt=  8.6 mm  dR= 1.43 deg
```

最后一行是 `inv(reference[k]) @ action[k]`，与 openpi 训练期现算相对动作是同一套解码。绝对值全是同一个
~0.4 m 的常量偏移，**标签写错也照样「看着正常」**，所以必须看这个相对量：段内它与每帧手部行程同量级
（本段 p50 4.8 mm、p99 20.9 mm），而最大值出现在**拼接缝**（k=44，28.4 mm / 22.29°），因为
`action[k]` 指向的是被挖掉那段之后的下一帧 —— 判读时连 `splice=` 一起看。

`--vis` 是**后置**的（每步跑完后由 `cli._vis()` 调用），所以「产物已存在、这一步被跳过」时可视化照样
出得来；可视化失败只打一条警告，不会把一次成功的 step 变成失败（它不参与任何下游计算）。

收在 `--vis` 后面的还有纯诊断产物：step2 的 `quality.png` / `report.json` 与 step3 的
`dino_boxes/*.json`。这两处开关**默认关**，所以不带 `--vis` 跑 pipeline 时它们不落盘。

最终观测帧里**只有原 RGB + 手修复 + 夹爪**，没有 XYZ 轴 / REL 面板 / 关键点 / mask 叠加。
`--verbose` 才展开每步的细节打印。

## 与 s4 的差异

| | s4 | 这里 |
|---|---|---|
| 手 / 物体 | 左右手 × {holder, red, yellow} | 右手 × N 物体 `obj1..objN`（`--prompt "A\|B"`） |
| `observation.state` | 54 + 爪 | 9N + 爪（同样 `right_tcp_to_*`，`<slug>` 换成实例 id） |
| `action` | 18 + 爪 | 9 + 爪 = 10 |
| `observation.action_reference_tcp` | 18 | 9 |
| 夹爪 | BrainCo 连续 / 二值两种 variant | 只有二值（`variant: "binary"`） |
| 观测 | 640×480 | 同 **640×480**（左眼 1080×810 等比缩放，见下） |
| 观测里**手位姿**的参考系 | `g1_base` | `camera0`（左眼；`state` 由它派生成手系里的物体位姿） |
| 相对位姿 | `T_tcp_object` | 同（`state` 的物体块；`reference`/`action` 是绝对的） |
| `action` / `reference` | **绝对**位姿（`right_tcp_absolute_*`），落点 `g1_base` | **绝对**位姿，同一套列名，落点 **PICO OpenXR 右手世界系**（逐段不同；相对动作在 openpi 训练期现算） |
| action 的「下一帧」 | 全局 30Hz 网格 | **episode 内**下一帧（裁剪后可能跨源帧跳变） |
| 向量维度名 / `object_order` | 语义 slug（`holder`/`red`/`yellow`），两处叫法还不一样 | 统一用实例 id `obj1..objN` |

物体数只影响 `observation.state` —— `action` 恒为 10、`observation.action_reference_tcp` 恒为 9。
爪因此仍在 `state` 的**最后一维**（`state[:, -1]`），不在第 9 维。

**观测分辨率对齐 `ego_relation_policy`**：进数据集的 mp4 与 `features` / `visual_size` 都用 640×480
（`export.OBSERVATION_WIDTH/HEIGHT`），左眼原图 1080×810 同为 4:3，`INTER_AREA` 等比缩放在**编码这一处**做。
观测 PNG 与所有可视化仍是 1080×810，`verify_dataset` 会把三者钉在一起。

## 多物体

物体数由 `--prompt` 里的竖线决定，**顺序即 obj 序**：

```bash
bash run.sh step2 --stems 20260920_111342 --prompt "black, metal pen holder|red cube"
#                                  -> obj1 = pen holder, obj2 = red cube
```

- **至少 2 段提示词**（见「跑法」）。
- `--box` 与物体顺序一一对应：一个物体给 4 个数（`--box x1 y1 x2 y2`，与老命令行逐字相同），
  N 个物体一次给 4N 个数。**个数与物体数不符就报错** —— 少给一个会让第 2 个物体悄悄退回 DINO。
- `--prompt-frame` 仍是**全局单值**：SAM2 从提示帧才开始写 mask，所有物体共享同一个起始帧，
  所以 `objectpose` 的参考帧仍是单值。代价是**每个物体都必须在提示帧被 DINO 框到**（框不到就报错）。
- 成本：SAM2 是「一个 predictor state + N 次 `add_new_mask` + **一次** `propagate_in_video`」，
  帧张量 12.58 MB/帧与物体数无关 → 多物体在最贵的那一步几乎免费。深度（SGBM）也只跑一次，
  逐物体只多一份点云 + 一次 ICP。
- **一个数据集 = 一套固定的、有序的物体表**（照 s4 `lerobot.py:203-206` 的语义）：`--stems all` 时各段的
  `(obj_i, category)` 表必须完全相等，否则报错并指出是哪几段不同 —— 不然同一个数据集里 state 各维的含义
  会逐段漂移，训练侧无从察觉。不一致就拆成两个数据集导。
- step3 会把**所有**物体的 mask 从手部 mask 里抠掉（`PipePaths.seg_masks`）。只抠 obj1 的话，
  第二个物体会被当成手一起 LaMa 修掉。

`--prompt` / `--task` 的按段映射规则不变（给 1 个共用 / 给 N 个一一对应），竖线切分发生在
**拿到字符串之后**，所以 `--prompt A|B --prompt C|D` 就是「第 1 段两个物体、第 2 段两个物体」。

## 观测源（`--observation step2|step3`，默认 `step3`）

只选**视频帧从哪来**，标签列（state / action / reference / timestamp）完全不受影响：

| | `step3`（默认） | `step2` |
|---|---|---|
| 帧 | `step3/observation/%05d.png`（手已修 + piper 夹爪） | 直接解源 `CameraRecord_<stem>.mp4` 左半 |
| 依赖 | `observation.json` 必须存在 | **不读 step3 的任何产物**（step3 可以整步跳过） |
| 时间轴校验 | `obs["n_frames"] == step2 的 n_frames_source` | step1 的 `n_frames` 同上 |
| `visual_source` | `pipeline_step3_composite` | `pipeline_step2_raw` |
| 画面 | 原 RGB + 手修复 + 夹爪 | **原始手，没有夹爪** |

step2 模式就是「跳过 step3」的定义本身。缩放到 640×480 与抽帧校验对两种源一视同仁。

## 口径上的决定（写在这里以免以后当成 bug）

- **世界系逐段一个、不重新归零，action 轴向固定为 OpenXR 右手系**（见「世界系」）。这是本轮与参考差异最大的一处，
  也是接真机前必须做一次方向核对的原因。
- **锁存门 = 0.05 m**（`--latch-distance`）。参考默认 0.20。这是算法口径上与参考不同的一条
  （见「抓握锁存」），也是 N=1 时结果会变的那一处（丢跟的握持段不再锁存）。
- 观测视频 codec 默认 **`h264`**（libx264 + yuv420p + faststart，走 `xrhand.video`）：常见播放器 /
  浏览器直接打得开，体积也比 `mp4v` 小。`--video-codec mp4v` 才是 s4 `cfg.export.video_codec` 那个值，
  只有在要「与 s4 逐字一致」时才用 —— 它很多播放器打不开。
- 没有 `observation.valid` 之类的额外列（保持 key 集合与 s4 完全一致）。需要逐帧有效性掩码时，
  用 `extraction_meta.json` 里的 `keep_index` 回查 step2 的 `state_valid`。

## 依赖

解释器固定 `test/pipeline/.venv/bin/python`（cv2 / pyarrow / onnxruntime / pyrender 只装在那里），
`run.sh` 已经设好 CUDA 库路径与 HuggingFace 镜像（本机直连 huggingface.co 会超时）。
`config/calib.json` 必须有（标定读不到就直接报错，不用标称值兜底）。
