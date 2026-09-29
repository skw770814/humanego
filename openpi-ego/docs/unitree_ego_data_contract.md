# Unitree Ego 数据契约与全链路安全规则

本文是 Human Ego 数据在归一化、单独训练、Robot/Human 混训、恢复训练和评估中的唯一语义依据。
OpenPI 的 Human mode3/4/5/6 只表示 state/相机是否可见，与 Ego 转换器的 Mode1/Mode2 完全是两组概念。

## 1. 已核对的 Ego 转换器输出

源工程：

```text
/home/zh/ego_project/IL-EGO/egodata_targeting_project
```

当前转换器在每个 LeRobot 数据集写入 `meta/action_semantics.json`。OpenPI 会读取并校验该文件，
而不是根据 20D/30D shape 猜测动作含义。

| Ego 数据模式 | LeRobot state EEF | LeRobot action EEF | 参考系 | 控制点 |
|---|---|---|---|---|
| Mode1 | IK 后 FK 得到的绝对 pose | 同一时刻的理想绝对目标 pose | G1 `pelvis` base | `wrist_yaw` TCP |
| 当前 Mode2 | 当前绝对 TCP pose | 下一帧绝对 TCP pose；末帧重复 | 固定 recording frame | `wrist_yaw` TCP |

两种模式的 pose9 都是：

```text
[x, y, z, r00, r10, r20, r01, r11, r21]
```

也就是 grouped first-two-columns Rot6D，与本项目的 `columns_grouped` 相同。

Mode2 的 `replay/*.npz` 还包含相邻绝对动作之间的相对 delta，但它仅用于可视化回放，不是 LeRobot
Parquet 的训练 action。旧版 Mode2 数据曾把这种相对量写进 Parquet；shape 和 feature name 无法把它
与修复后的绝对 action 区分。因此：

- 当前 Mode2 必须有 `meta/action_semantics.json`，且声明
  `absolute_next_target_in_recording_frame`；
- 缺少该文件的旧 Mode2 一律拒绝，必须用当前 Ego 转换器重新导出；
- 旧 Mode1 只有在 `extraction_meta.json` 明确声明 `mode1_ideal` 时才兼容；
- 既无 `action_semantics.json`、也无可用 extraction provenance 的数据，即使命令行手填 Mode1 也拒绝。

可先只读审计：

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_unitree_human_dataset.py \
  /path/to/human_lerobot --mode mode1

PYTHONPATH=src .venv/bin/python scripts/audit_unitree_human_dataset.py \
  /path/to/human_lerobot --mode mode2
```

## 2. 为什么不能再对所有 Human 做同一个变换

真机 Robot EEF 契约是：

```text
torso_link -> left/right_hand_palm_link
```

Ego 数据虽然已经完成“人手到机器人 TCP”的 retargeting，但它保存的是 `wrist_yaw` TCP；参考系则因
Mode1/Mode2 而不同。它不是未经处理的人体 pelvis→wrist 数据，也不能再无条件套一遍旧
pelvis→wrist 变换。

这里的转换是 G1 机器人 link convention 统一，不是再次做人体→机器人 retarget。代码按 Ego mode
自动选择 profile，也允许命令行显式写出同一个值：

| `HUMAN_INPUT_FRAME` | 行为 |
|---|---|
| `native` | 仅诊断用；当前 Ego Mode1/2 正式配置会拒绝 |
| `g1_base_tcp` | Mode1 默认：G1 pelvis base→torso，再将 wrist-yaw TCP→hand palm |
| `recording_tcp` | Mode2 默认：保留 recording reference，只将 wrist-yaw TCP→hand palm |
| `torso_palm` | 保留给未来有元数据证明为 OpenPI torso→palm 的数据；当前 Ego Mode1/2 会拒绝 |
| `pelvis_wrist` | `g1_base_tcp` 的旧名称兼容项；新实验不再推荐 |

Mode1 转换使用 Ego 仿真模型零腰部关节时的：

```text
p_torso = p_g1_base - [-0.0039635, 0, 0.044]
p_palm  = p_torso + R_tcp @ wrist_to_palm
```

左右 wrist→palm 分别为 `[0.0415, 0.003, 0]` 和 `[0.0415, -0.003, 0]`。旋转不因这两个纯
translation offset 改变。

## 3. 支持矩阵

| 训练场景 | Ego Mode | 动作表示 | 必须设置 | 结果 |
|---|---|---|---|---|
| Human-only | Mode1 | absolute/relative | `HUMAN_INPUT_FRAME=g1_base_tcp` | 支持，统一为 torso→palm |
| Human-only | 当前 Mode2 | absolute/relative | `HUMAN_INPUT_FRAME=recording_tcp` | 支持，统一到 recording→palm |
| Robot+Human | Mode1 | absolute/relative | `HUMAN_INPUT_FRAME=g1_base_tcp` | 支持，对齐到 torso→palm |
| Robot+Human | 当前 Mode2 | relative | `HUMAN_INPUT_FRAME=recording_tcp` | 支持，先对齐控制点再求局部 delta |
| Robot+Human | 当前 Mode2 | absolute | 无 | 拒绝；缺少 recording→Robot 的固定放置关系 |
| 任意 Human | 任意 | 任意 | `HUMAN_INPUT_FRAME=native` | 拒绝；正式训练必须统一 G1 link/TCP convention |
| 任意 Human | 旧 Mode2 | 任意 | 任意 | 拒绝；无法证明 Parquet action 是绝对还是旧相对量 |

Mode2 relative 混训不需要 recording→torso 的全局放置，是因为 OpenPI 的局部 SE(3) 动作：

```text
delta_p[t,k] = R_state[t].T @ (p_action[t+k] - p_state[t])
delta_R[t,k] = R_state[t].T @ R_action[t+k]
```

对一个固定的全局刚体坐标变换不变。但 wrist TCP 和 palm 是不同控制点，旋转时会产生不同平移轨迹，
所以仍必须先用 `recording_tcp` 做 TCP→palm。

注意：这里不是 Ego replay 中“相邻 action 对 action”的 delta。OpenPI action chunk 中每个未来点
都相对于同一个当前 `state[t]`；训练输出在真机端再用严格逆变换恢复 Robot torso→palm 绝对目标。

## 4. 正确命令

Human-only Mode1：

```bash
HUMAN_DATASET_MODE=mode1 HUMAN_INPUT_FRAME=g1_base_tcp \
NORM_PRESET=human_abs bash compute_unitree_norm_stats.sh

HUMAN_DATASET_MODE=mode1 HUMAN_INPUT_FRAME=g1_base_tcp \
PRESET=human3_abs bash train_unitree.sh
```

Human-only 当前 Mode2：

```bash
HUMAN_DATASET_MODE=mode2 HUMAN_INPUT_FRAME=recording_tcp \
NORM_PRESET=human_rel_shared bash compute_unitree_norm_stats.sh

HUMAN_DATASET_MODE=mode2 HUMAN_INPUT_FRAME=recording_tcp \
PRESET=human3_rel_shared bash train_unitree.sh
```

Mode1 absolute 混训需要先分别计算 Robot 和 Human stats。Human 那次使用：

```bash
HUMAN_DATASET_MODE=mode1 HUMAN_INPUT_FRAME=g1_base_tcp \
NORM_PRESET=human_abs bash compute_unitree_norm_stats.sh

HUMAN_DATASET_MODE=mode1 HUMAN_INPUT_FRAME=g1_base_tcp \
PRESET=mix_robot2_human3_abs bash train_unitree.sh
```

Mode2 relative 混训的 Human stats 和训练都使用：

```bash
HUMAN_DATASET_MODE=mode2 HUMAN_INPUT_FRAME=recording_tcp \
NORM_PRESET=human_rel_shared bash compute_unitree_norm_stats.sh

HUMAN_DATASET_MODE=mode2 HUMAN_INPUT_FRAME=recording_tcp \
PRESET=mix_robot2_human3_rel_shared bash train_unitree.sh
```

`human_abs`/`human_rel_shared` 需要按实际 BrainCo、gripper 或 EEF-only preset 选择；数据模式与 frame
参数必须在计算 stats 和训练时完全一致。

## 5. stats、resume 与评估保护

Human norm asset ID 现在同时编码：

```text
ego_mode + source_profile + dataset_contract_digest + action_representation + split_digest
```

`norm_stats_manifest.json` 还会逐项记录 dataset root、action horizon、input frame 和完整 Ego contract。
训练时只要已有 stats 缺 manifest 或任一字段不匹配就直接报错，不会静默复用。

checkpoint 的 `runtime_manifest.json` 使用 schema v2，保存 Human dataset mode、input frame、完整 component
contract 和 asset ID。以下情况都会阻止 resume/评估：

- 旧 checkpoint 没有 Human mode/frame；
- component contract 缺失或声明的 action 不是 absolute；
- checkpoint 与当前训练的 dataset mode、frame、dataset path 或 contract 不一致；
- Human asset ID 不含完整 contract 摘要。

只读扫描旧资产：

```bash
PYTHONPATH=src .venv/bin/python scripts/purge_legacy_unitree_human_stats.py \
  /path/to/human_dataset

PYTHONPATH=src .venv/bin/python scripts/purge_legacy_unitree_human_stats.py \
  --checkpoint-root checkpoints

PYTHONPATH=src .venv/bin/python scripts/purge_legacy_unitree_human_checkpoints.py \
  checkpoints
```

stats 确认后可加 `--apply` 永久删除。带 `--checkpoint-root` 时只删除 checkpoint 内错误的 Human
stats，保留模型权重和 Robot stats。checkpoint 清理脚本则删除整个无效 step，通常体积很大，确认实验
目录后再加 `--apply`。

旧 Mode1 checkpoint 只有在使用同一数据 split 和 progress contract 重新计算出带
`ego_mode1_source_g1_base_tcp_contract_<digest>` 的 stats 后，才允许升级。先 dry-run，再执行：

```bash
PYTHONPATH=src .venv/bin/python scripts/migrate_unitree_mode1_checkpoint_contract.py \
  /path/to/checkpoint/STEP \
  /path/to/recomputed_mode1_stats

PYTHONPATH=src .venv/bin/python scripts/migrate_unitree_mode1_checkpoint_contract.py \
  /path/to/checkpoint/STEP \
  /path/to/recomputed_mode1_stats --apply
```

迁移器会核对 Mode1、absolute action、数据维度、split digest、progress digest、进度缩放和完整
source contract，拒绝覆盖任何已有目标资产。该工具刻意不能迁移 Mode2；旧 Mode2 checkpoint
继续无效，不能通过补写 manifest 或复制 Mode1 stats 恢复。
