# OpenPI：Unitree G1-D + BrainCo 训练与评估

Human 从 EEF20-gripper 数据读取、但混训只监督双臂 EEF18 的说明：
[docs/unitree_human_eef_only_training.md](docs/unitree_human_eef_only_training.md)。

Robot 单头部相机训练、归一化和真机推理说明：
[docs/unitree_robot_single_camera.md](docs/unitree_robot_single_camera.md)。

Ego Mode1/Mode2 的 pose/action、TCP、归一化、单训和混训安全矩阵：
[docs/unitree_ego_data_contract.md](docs/unitree_ego_data_contract.md)。

本文是当前 G1-D BrainCo 实验的完整操作手册，覆盖：

1. joint26 数据转换为 EEF30；
2. Robot/Human 分别计算归一化；
3. Robot-only、Human-only 和 Robot+Human 混合训练；
4. Validation、W&B 诊断、checkpoint 保存；
5. 独立评估服务和 Unitree G1-D 真机推理。

新实验使用独立的训练配置
[`src/openpi/training/unitree_train_config.py`](src/openpi/training/unitree_train_config.py) 和评估配置
[`src/openpi/training/unitree_eval_config.py`](src/openpi/training/unitree_eval_config.py)。原来的
`unitree_config.py`、旧 checkpoint 和 legacy 评估入口保持不变；policy、模型和训练主体仍然共用。

## 1. Mode 定义

| Mode | 数据域 | 模型输入 | state/action 数值定义 |
| --- | --- | --- | --- |
| Robot mode1 | 真机 | state + 默认三路相机；支持单头部 | 26D 绝对 joint：左臂7 + 右臂7 + 左BrainCo6 + 右BrainCo6 |
| Robot mode2 | 真机 | state + 默认三路相机；支持单头部 | 30D：左EEF9 + 右EEF9 + 左BrainCo6 + 右BrainCo6 |
| Human mode3 | 人类 | state + 单路头部相机 | 30D EEF + BrainCo |
| Human mode4 | 人类 | 不向模型输入 state + 单路头部相机 | 30D EEF + BrainCo |
| Human mode5 | 人类 | state + 三相机接口 | 30D EEF + BrainCo |
| Human mode6 | 人类 | 不向模型输入 state + 三相机接口 | 30D EEF + BrainCo |

当前 Human 数据只有真实头部相机。mode5/6 的左右腕部相机槽位使用零图，并设置
`image_mask=false`，不会复制头部图像伪装腕部相机。

混合训练只允许：

```text
Robot mode2 + Human mode3/4/5/6 中的一个
```

不能在同一个训练中混入多个 Human mode，也不能把 joint26 Robot mode1 与 EEF30 Human 混合。
默认 Robot:Human 为 1:1；`batch_size=32` 时每个 batch 固定为 16 条 Robot、16 条所选 Human。

Robot 现有 preset 全部保持三相机行为。在任意包含 Robot 的训练 preset 末尾追加 `_single`，
即可只读取头部相机，例如 `robot1_abs_single`、`robot2_rel_shared_single` 或
`mix_robot2_human3_abs_progress_single`。单相机模式不会复制头部图像到腕部槽位，而是使用零图并设置
`image_mask=false`。

## 2. 坐标系契约

### 2.1 base 和 EEF link

本项目把“机器人 base”明确固定为 G1-D URDF 中的：

```text
B = torso_link
E_L = left_hand_palm_link
E_R = right_hand_palm_link
```

因此绝对 EEF pose 始终表示：

```text
T_B_E = T_torso_link,hand_palm_link
```

这里的 base 不是 `pelvis`、`base_link` 或 world。如果真机系统中所说的 base 实际是其他 link，必须在
数据转换、Human 对齐、训练和推理四个环节一起修改，不能只修改其中一个字符串。

joint→EEF 转换和真机推理 FK 使用相同定义：

```text
T_B_E = inverse(T_world_B) @ T_world_E
```

Human Ego 数据已经做过人手→机器人 TCP retargeting，但仍需统一 G1 机器人 link convention：
Mode1 是 G1 pelvis→wrist-yaw TCP，归一化和训练（Human-only 与混训）统一转换为
torso→hand-palm；当前 Mode2 是 recording frame→wrist-yaw TCP，保留 recording reference 并统一
控制点到 palm。默认 profile 按 Ego mode 自动选择 `g1_base_tcp` 或 `recording_tcp`。Mode2 absolute
不能安全混训。完整规则见
[Ego 数据契约](docs/unitree_ego_data_contract.md)。

### 2.2 EEF30 排列

单手 EEF9 为 XYZ 加旋转矩阵前两列：

```text
[x, y, z, r00, r10, r20, r01, r11, r21]
```

这套 Rot6D 格式叫 `columns_grouped`，完整 EEF30 顺序是：

```text
[0:9]   left  T_torso,left_palm
[9:18]  right T_torso,right_palm
[18:24] left  BrainCo 6D absolute
[24:30] right BrainCo 6D absolute
```

不要与 legacy `rotation_format="columns"` 的交错排列混用。

### 2.3 absolute 和 relative 动作

Robot absolute 模式的 state/action 是 `torso_link -> hand_palm_link`。Mode1 Human-only 和混训均
转换到同一 torso→palm 契约；Mode2 Human-only 保留 recording reference，但将 wrist TCP 统一到 palm。

Relative 模式采用局部 SE(3) 表达：

```text
delta_p[t,k] = R_state[t].T @ (p_action[t+k] - p_state[t])
delta_R[t,k] = R_state[t].T @ R_action[t+k]
```

关键点：

- action chunk 内所有 `action[t+k]` 都相对于同一个当前观测 `state[t]`；
- 不是相对于 `action[t+k-1]`，也不会逐步累积；
- relative XYZ 和 rotation 表达在当前 EEF 的局部坐标中；
- Robot 与 Mode1 Human absolute state/action 位于 `torso_link`；Mode2 保留 recording reference；
- BrainCo12 在 absolute/relative 两种实验中始终使用绝对角度。

推理输出 relative EEF 时，会使用严格逆变换恢复 absolute EEF：

```text
p_action = p_state + R_state @ delta_p
R_action = R_state @ delta_R
```

### 2.4 URDF 一致性

joint→EEF 转换器会记录 G1-D URDF 的 SHA256、reference link、两个 EEF link 和 Rot6D 格式。之后：

- 归一化和训练会检查 EEF 数据集内的这些元数据；
- checkpoint 会保存相同的 frame/link/URDF SHA256；
- 真机客户端会根据 checkpoint 指定的 frame/link 建立 FK/IK；
- 客户端 URDF 哈希与 checkpoint 不一致时，在连接机器人前直接报错。

因此转换和真机推理必须使用内容完全相同的 G1-D URDF；文件路径可以不同，文件内容不能不同。

## 3. 数据集路径

当前数据集：

```bash
ROBOT_JOINT_DATASET="/home/zh/w_ego_collect/IL/openpi/dataset/G1_Brainco_Abc_fold_clothes_7_7/G1_Brainco_Abc_fold_clothes_7_7"
ROBOT_EEF_DATASET="/home/zh/w_ego_collect/IL/openpi/dataset/G1_Brainco_Abc_fold_clothes_7_7_eef30_columns_grouped"
HUMAN_DATASET="/home/zh/w_ego_collect/IL/openpi/dataset/pick_bottle_put_in_box_have_state"
```

三个变量同时存在于：

- [`compute_unitree_norm_stats.sh`](compute_unitree_norm_stats.sh)
- [`train_unitree.sh`](train_unitree.sh)

修改数据集时，必须保证两个脚本中的路径一致。路径必须指向 LeRobot 数据集根目录，即目录内直接包含：

```text
meta/info.json
meta/episodes.jsonl
meta/tasks.jsonl
data/
videos/
```

`robot1_*` 使用 joint26 数据；`robot2_*` 和所有混训使用转换后的 EEF30 数据；`human_*` 使用 Human
数据。当前 `ROBOT_EEF_DATASET` 在执行全量转换之前不存在，这是正常的。

## 4. joint26 转换为 EEF30

### 4.1 固定 G1-D URDF

在 OpenPI 根目录执行：

```bash
cd /home/zh/w_ego_collect/IL/openpi

G1_D_URDF="/home/zh/openpi/g1_d_description/g1_d.urdf"
sha256sum "$G1_D_URDF"
```

原始 joint26 顺序必须为：

```text
左臂7D + 右臂7D + 左BrainCo6D + 右BrainCo6D
```

转换器会读取 `meta/info.json` 中的完整关节名并严格检查顺序。`observation.state` 和 `action` 的前
14D 分别经过相同的 G1-D FK：

- joint state 转成当前左右 EEF；
- joint action 转成目标左右 EEF；
- BrainCo12 原样复制，不参与 FK。

### 4.2 少量 episode 只读测试

先测试 3 个 episode。`--dry-run` 不创建输出数据集：

```bash
.venv/bin/python tools/convert_g1_joint_to_eef.py \
  "$ROBOT_JOINT_DATASET" \
  /tmp/g1d_eef_smoke \
  --urdf "$G1_D_URDF" \
  --task 'fold clothes.' \
  --max-episodes 3 \
  --dry-run
```

输出中需要检查：

```text
reference_link = torso_link
eef_links = left_hand_palm_link, right_hand_palm_link
frame_convention = T_reference_eef
rotation_6d = columns_grouped:[r00,r10,r20,r01,r11,r21]
brainco_passthrough = true
```

### 4.3 转换全部数据

输出目录必须事先不存在：

```bash
.venv/bin/python tools/convert_g1_joint_to_eef.py \
  "$ROBOT_JOINT_DATASET" \
  "$ROBOT_EEF_DATASET" \
  --urdf "$G1_D_URDF" \
  --task 'fold clothes.'
```

转换器逐 episode 读取和写入 Parquet，不会把整个数据集加载到内存。视频默认在同一文件系统上使用
hardlink，避免复制大体积视频；如果必须复制视频，额外添加 `--copy-videos`。

只转换指定 episode 时可以使用：

```bash
--episodes 0,2,5-8
```

转换完成后会重新生成 LeRobot stats，并保存：

```text
<ROBOT_EEF_DATASET>/meta/info.json
<ROBOT_EEF_DATASET>/meta/joint_to_eef_conversion.json
```

检查转换记录：

```bash
sed -n '/eef_forward_kinematics/,+10p' "$ROBOT_EEF_DATASET/meta/info.json"
sed -n '1,30p' "$ROBOT_EEF_DATASET/meta/joint_to_eef_conversion.json"
```

其中 URDF SHA256 必须与 `sha256sum "$G1_D_URDF"` 一致。

## 5. 计算归一化

### 5.1 为什么不提供 mix 归一化

Robot 和 Human 是两个不同数据域，不能混在一起计算一套统计。混合训练时：

- Robot component 使用 Robot 数据集内部的 stats；
- Human component 使用 Human 数据集内部的 stats；
- 两份 stats 在训练时分别加载。

打开 [`compute_unitree_norm_stats.sh`](compute_unitree_norm_stats.sh)，确保只有一条 `NORM_PRESET`
没有被注释，然后运行：

```bash
./compute_unitree_norm_stats.sh
```

所有归一化 preset：

```text
robot1_abs
robot2_abs
robot2_rel_shared
robot2_rel_per_step
robot2_rel_hybrid
human_abs
human_rel_shared
human_rel_per_step
human_rel_hybrid
```

### 5.2 训练 preset 与归一化对应关系

| 训练类型 | 需要提前计算的 stats |
| --- | --- |
| `robot1_abs` | `robot1_abs` |
| `robot2_abs` | `robot2_abs` |
| `robot2_rel_shared` | `robot2_rel_shared` |
| `robot2_rel_per_step` | `robot2_rel_per_step` |
| `robot2_rel_hybrid` | `robot2_rel_hybrid` |
| `human3/4/5/6_abs` | `human_abs` |
| `human3/4/5/6_rel_shared` | `human_rel_shared` |
| `human3/4/5/6_rel_per_step` | `human_rel_per_step` |
| `human3/4/5/6_rel_hybrid` | `human_rel_hybrid` |
| `mix_robot2_humanX_abs` | 分别计算 `robot2_abs` 和 `human_abs` |
| `mix_robot2_humanX_rel_shared` | 分别计算 `robot2_rel_shared` 和 `human_rel_shared` |
| `mix_robot2_humanX_rel_per_step` | 分别计算 `robot2_rel_per_step` 和 `human_rel_per_step` |
| `mix_robot2_humanX_rel_hybrid` | 分别计算 `robot2_rel_hybrid` 和 `human_rel_hybrid` |

这里 `X` 只能是 3、4、5、6 中的一个。

例如训练 `mix_robot2_human3_rel_shared`：

1. 在归一化脚本中选择 `NORM_PRESET="robot2_rel_shared"`，运行一次；
2. 改为 `NORM_PRESET="human_rel_shared"`，再运行一次；
3. 然后启动混合训练。

Robot 相机数量不会改变 state/action 数值分布，因此 `_single` 与对应的原 preset 共享同一份 stats。
例如 `robot1_abs_single` 可直接使用 `robot1_abs` 的 stats；运行
`NORM_PRESET=robot1_abs_single` 也会解析到完全相同的 `asset_id`。

### 5.3 Human mode3/4/5/6 为什么共用 stats

四个 Human mode 使用同一 Human 数据集、同一 EEF/BrainCo 数值转换和同一 train split。它们仅改变：

- state 是否作为 token 暴露给模型；
- 使用单相机接口还是三相机接口。

这些差异不会改变 action 数值分布，所以不需要重复计算四份 Human action stats。mode4/6 的 relative
预处理仍会读取原始 state 来计算 EEF 相对动作，但不会把 state token 输入模型。

### 5.4 shared、per-step、hybrid 的区别

- `rel_shared`：所有 action horizon step 共用一套 action stats。样本最多、最稳定，建议作为第一组
  relative 基线。
- `rel_per_step`：每个 horizon step 单独计算 stats。能适配远近动作尺度差异，但近端动作可能接近零，
  容易放大同步、FK 和标定噪声，建议作为消融实验。
- `rel_hybrid`：EEF18 每个 step 单独统计，BrainCo12 所有 step 共用统计。它更符合“EEF 相对、手指
  绝对”的语义，建议在 `shared` 基线之后测试。

绝对模式沿用当前 OpenPI quantile normalization。所有模式都对低方差/常量维度采用通用 scale floor，
不针对某三个 BrainCo 关节硬编码；归一化后的数值限制在 `[-5,5]`，避免长时间不动维度造成异常放大。

### 5.5 stats 保存位置

归一化只使用 train episodes。默认保留 Robot 20 个、Human 10 个完整 episode 做 validation。

统计保存在各自数据集内部：

```text
<dataset>/meta/openpi_assets/<asset_id>/norm_stats.json
<dataset>/meta/openpi_assets/<asset_id>/norm_stats_manifest.json
```

`asset_id` 包含动作表示、归一化方式和 split digest，因此 absolute/shared/per-step/hybrid 可以同时存在，
不会互相覆盖。

检查结果：

```bash
find "$ROBOT_EEF_DATASET/meta/openpi_assets" -name norm_stats.json -print
find "$HUMAN_DATASET/meta/openpi_assets" -name norm_stats.json -print
```

只用少量 episode 测试归一化时，必须写到临时目录，不能覆盖正式数据集统计：

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_unitree_norm_stats.py \
  --norm-preset human_rel_shared \
  --human-dataset "$HUMAN_DATASET" \
  --max-episodes 3 \
  --output-root /tmp/g1d_norm_smoke
```

## 6. 启动训练

### 6.1 选择训练 preset

打开 [`train_unitree.sh`](train_unitree.sh)，确保只有一条 `PRESET` 没有被注释。

Robot-only：

```text
robot1_abs
robot2_abs
robot2_rel_shared
robot2_rel_per_step
robot2_rel_hybrid
```

Human-only：

```text
human{3|4|5|6}_abs
human{3|4|5|6}_rel_shared
human{3|4|5|6}_rel_per_step
human{3|4|5|6}_rel_hybrid
```

Robot+Human：

```text
mix_robot2_human{3|4|5|6}_abs
mix_robot2_human{3|4|5|6}_rel_shared
mix_robot2_human{3|4|5|6}_rel_per_step
mix_robot2_human{3|4|5|6}_rel_hybrid
```

例如：

```bash
PRESET="mix_robot2_human3_rel_shared"
```

表示 Robot mode2 与 Human mode3 进行 1:1 混训，两边 EEF 使用 relative action 和 shared stats，
BrainCo 保持 absolute。

### 6.2 常用参数

脚本顶部可直接修改：

```bash
BATCH_SIZE=32
NUM_TRAIN_STEPS=30000
NUM_WORKERS=8
VALIDATION_INTERVAL=500
VALIDATION_BATCHES=10
MODALITY_DIAGNOSTICS_INTERVAL=2000
SAVE_INTERVAL=1000
KEEP_PERIOD="${SAVE_INTERVAL}"  # 每次定期保存的 checkpoint 都永久保留
```

checkpoint 生命周期三选一：

```bash
TRAIN_LIFECYCLE="new"         # 默认；同名目录存在时安全退出
# TRAIN_LIFECYCLE="resume"    # 从最新 checkpoint 继续
# TRAIN_LIFECYCLE="overwrite" # 删除同名旧实验后重新训练
```

W&B 二选一：

```bash
WANDB_MODE="on"
# WANDB_MODE="off"
```

### 6.3 运行训练

确认需要的 Robot/Human stats 已经分别计算后：

```bash
./train_unitree.sh
```

训练开始前会检查：

- 数据集是否存在；
- Robot EEF 数据是否包含正确的 frame/link/Rot6D/URDF SHA256；
- 所选 preset 是否是合法单域或 `Robot mode2 + 一个 Human mode`；
- 对应的 norm stats 是否存在；
- train/validation episode split 是否有效。

checkpoint 保存到：

```text
checkpoints/unitree_g1d_brainco_train/<EXP_NAME>/<STEP>/
```

训练会把实际使用的每个 component stats 复制到 checkpoint 的 `assets/`，并写入：

```text
assets/runtime_manifest.json
```

因此评估依赖 checkpoint 自身，不再依赖原始训练数据目录。

## 7. Validation 和 W&B

默认每 500 step 进行一次 validation，每个域使用 10 个 batch。混训分别记录：

```text
val/loss/robot_mode2
val/loss/human_modeX
val/loss_weighted
```

`val/loss_weighted` 使用训练 mixture 权重；1:1 混训时就是 Robot/Human validation loss 的等权组合。

每隔 `MODALITY_DIAGNOSTICS_INTERVAL` 还会对同一 validation batch、同一 flow noise/timestep 分别屏蔽
text、state 和 image，记录受控 loss-ablation 指标：

```text
val/modality_reliance/<domain>/text_delta
val/modality_reliance/<domain>/state_delta
val/modality_reliance/<domain>/image_delta
val/modality_reliance/<domain>/text_share
val/modality_reliance/<domain>/state_share
val/modality_reliance/<domain>/image_share
```

这些指标表示模型对输入模态的 loss sensitivity，不是简单平均 Transformer attention 权重。no-state
mode 的 state span 为空，因此 `state_delta` 应接近零。

## 8. 评估与真机推理

### 8.1 启动独立评估服务

Robot-only 或 Robot+Human checkpoint 使用：

```bash
uv run scripts/serve_unitree_experiment.py \
  --checkpoint /home/zh/w_ego_collect/IL/openpi/checkpoints/robot2_abs/robot2_abs_25000 \
  --default-prompt 'fold clothes.'
```

评估入口会：

1. 读取 `assets/runtime_manifest.json`；
2. 只选择 Robot component 的 norm stats；     
3. 检查 Robot mode、absolute/relative、Rot6D、reference frame 和 EEF links；
4. 检查 component URDF SHA256 与 checkpoint 总 manifest 一致；
5. 使用 checkpoint 参数启动 websocket policy server。

Human-only checkpoint 没有 Robot component，不能直接用于 G1-D 真机控制。

### 8.2 真机客户端必须使用同一个 URDF

在真机客户端机器上准备与 joint→EEF 转换阶段内容完全相同的 G1-D URDF：

```bash
G1_D_URDF="/path/on/robot/to/the/same/g1_d.urdf"
sha256sum "$G1_D_URDF"
```

不要依赖客户端内置的 legacy Unitree-deploy URDF；它与转换用 G1-D URDF 的文件内容/哈希可能不同。

启动同步推理：

```bash
examples/unitree_inference/client.sh \
  sync 'fold clothes.' 127.0.0.1 8000 auto \
  --urdf-path "third_party/g1_d_description/g1_d.urdf" \
  --dt 0.0333333333
```

其他支持的执行策略：

```text
sync
async
temporal_ensembling
temporal_smoothing
rtc
```

客户端执行链路是：

```text
当前 joint14
  -> 同一 URDF 的 torso_link→palm FK
  -> EEF18 + 当前 BrainCo12
  -> policy state
  -> policy absolute EEF30 输出
  -> 同一 URDF、同一 frame/link 的 IK
  -> joint14 + absolute BrainCo12
```

relative checkpoint 的模型输出会在服务端根据当前 observation state 恢复为 absolute EEF，再发送给
客户端做 IK。

### 8.3 控制频率

Robot/Human 数据和 action chunk 都是 30 Hz。客户端必须使用：

```text
dt = 1 / 30 = 0.0333333333 秒
```

也可以由控制器把 30 Hz action 显式插值到更高频率，但不能直接把 30 Hz action 以默认 40 Hz 逐点
执行，否则 50-step chunk 会从约 1.67 秒压缩到 1.25 秒。

## 9. 推荐实验顺序

建议按下面顺序建立可靠基线：

1. `robot1_abs`：确认原始 joint policy 和评估链路正常；
2. `robot2_abs`：确认 joint→EEF、归一化、FK/IK 坐标统一；
3. `robot2_rel_shared`：relative EEF 稳定基线；
4. `robot2_rel_hybrid`：比较 EEF per-step、BrainCo shared；
5. `mix_robot2_human3_rel_shared`：第一组 1:1 混训；
6. mode4/5/6 分别作为独立实验，不在同一次训练中合并。

每组实验至少比较：

- 固定 Robot validation loss；
- 真机折衣成功率；
- 完成时间；
- EEF 轨迹稳定性和 IK 失败/限位次数；
- BrainCo 手指动作是否出现饱和或抖动。

## 10. 常见问题

### Robot mode2 提示数据集不存在

先执行第 4 节 joint26→EEF30 全量转换，并确认 `ROBOT_EEF_DATASET` 与两个 shell 脚本中的路径一致。

### 提示缺少 norm stats

训练 preset 与归一化 preset 必须严格对应。混训需要分别运行一次 Robot stats 和一次 Human stats。

### 提示 EEF dataset convention incompatible

不要手工删除 `eef_forward_kinematics`。使用当前
`tools/convert_g1_joint_to_eef.py` 和正确 G1-D URDF 重新转换数据。

### 真机客户端提示 URDF SHA mismatch

客户端 `--urdf-path` 指向了与数据转换不同的 URDF。把转换时的同一个 URDF 文件复制到客户端机器，
重新传入该路径。只修改文件名或 link 参数不能解决哈希不一致。

### BrainCo 某些关节长期为零

这是数据现象，不应删除这些维度。归一化会通过通用低尺度保护处理常量/低方差维度，并继续保留完整
BrainCo12 输出。检查 `norm_stats_manifest.json` 中的低方差记录即可。

### 旧 checkpoint 如何评估

旧 checkpoint 继续使用原 `src/openpi/training/unitree_config.py` 和 legacy serve 脚本。不要用
`scripts/serve_unitree_experiment.py` 强行加载旧 checkpoint。

更深入的实现和设计分析见
[`docs/unitree_g1d_brainco_training.md`](docs/unitree_g1d_brainco_training.md)。
