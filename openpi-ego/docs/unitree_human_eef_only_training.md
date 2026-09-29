# Unitree 人机混训：Human 仅 EEF18

本文说明新增的 Human EEF-only 动作域。它只影响显式包含 `eef_only` 的新 preset；
已有 BrainCo、二指爪、Robot-only、Human-only 和普通人机混训 preset 保持原行为。
Ego Mode1/Mode2 的完整源语义和支持矩阵见
[unitree_ego_data_contract.md](unitree_ego_data_contract.md)。

## 1. 数据契约

`EEF-only` 表示训练监督域只有 18D，并不要求重新制作一份 18D 数据。默认直接读取现有
Human EEF20-gripper 数据：

```text
[0:9]   left EEF：xyz + grouped-columns Rot6D
[9:18]  right EEF：xyz + grouped-columns Rot6D
[18]    kLeftGripper
[19]    kRightGripper
```

程序要求 `observation.state` 和 `action` 维度一致。源为 20D 时会严格校验上述尾部名称，并在任何
归一化和训练变换之前只保留 `[0:18]`；源为原生 EEF18 时该投影是无操作。原数据文件不会被修改，
普通 `human_gripper_*` preset 仍正常使用夹爪2D。已经裁好的原生 EEF18 数据继续兼容，但不是必需输入。

`HUMAN_EEF_ONLY_DATASET` 默认等于 `HUMAN_GRIPPER_DATASET`。只有需要读取另一份数据时才单独
覆盖它。Ego 数据已做人手→机器人 TCP retargeting，但 Mode1/Mode2 的参考系不同，控制点仍是
`wrist_yaw` TCP。Mode1 的 Human-only 与混训都使用 `g1_base_tcp` 统一为 torso→palm；Mode2 的
Human-only 与 relative 混训都使用 `recording_tcp`，保留 recording reference 并将控制点移到 palm。
这些规则与是否裁掉 gripper2 无关，归一化和训练使用完全相同的 profile。

Human mode 仍表示相机/state 是否对模型可见：

| mode | state 输入模型 | 相机接口 |
|---|---|---|
| 3 | 是 | 单头部相机 |
| 4 | 否 | 单头部相机 |
| 5 | 是 | 头部 + 双腕 |
| 6 | 否 | 头部 + 双腕 |

即使 mode4/6 不把 state 编码给模型，源数据仍需提供 state；EEF20 的 state 同样先裁成 EEF18，
再用于坐标统一和 relative action 计算。

## 2. 训练语义

模型公共动作宽度仍为 32：

```text
Robot BrainCo: 监督 [0:30]
Robot gripper: 监督 [0:20]
Human EEF-only: 监督 [0:18]
```

Human 源数据的夹爪2D先被删除，剩余 EEF18 进入模型前补到 32D；`action_dim_mask` 会屏蔽
`[18:32]` 的 loss。因此被记录的 Human 夹爪值既不参与归一化，也不作为训练标签；Robot 的真实
样本继续负责学习 BrainCo12 或 gripper2。episode 尾部的 action padding 仍由独立时间掩码排除。

完整处理顺序：

```text
读取 Human EEF20
-> 校验最后两维名称
-> 删除 gripper[18:20]
-> 校验 Ego dataset mode/action contract
-> 按 Mode1/Mode2 profile 统一 G1 robot link/TCP convention
-> 可选 progress 重采样
-> 可选 relative 变换
-> EEF18 独立归一化
-> 补到模型32维并用 action_dim_mask 只监督前18维
```

checkpoint 顶层部署契约始终来自 Robot component。Human EEF-only 只在 component metadata 中记录：

```text
action_domain=eef_only
action_tail=none
dimension=18
supervised_action_dimensions=18
source_projection: 20 -> 18, ignored_tail=gripper2
```

## 3. 归一化

Robot 和 Human 必须分别计算统计。虽然 Human 从 EEF20-gripper 数据读取，但 EEF-only
统计只包含裁剪后的 18D，并使用独立 asset，不复用 20D/30D 统计：

```text
human_eef_only_abs
human_eef_only_rel_shared
human_eef_only_rel_per_step
human_eef_only_rel_hybrid
```

其中 EEF-only 没有绝对尾部，所以 `rel_hybrid` 等价于 EEF18 的 per-step 统计；仍保留独立
asset id，避免训练 preset 与统计类型混淆。`per_step/hybrid` 要求数据对 action horizon 的每个
位置至少提供两个有效样本；默认 horizon 为 50，正式数据应包含足够长的 episode。

Robot 为 BrainCo、absolute 混训：

```bash
cd /home/zh/w_ego_collect/IL/openpi

ROBOT_EEF_DATASET=/path/to/robot_eef30 \
HUMAN_EEF_ONLY_DATASET=/path/to/human_eef20_gripper \
NORM_PRESET=robot2_abs \
bash compute_unitree_norm_stats.sh

ROBOT_EEF_DATASET=/path/to/robot_eef30 \
HUMAN_EEF_ONLY_DATASET=/path/to/human_eef20_gripper \
NORM_PRESET=human_eef_only_abs \
bash compute_unitree_norm_stats.sh
```

Robot 为二指爪、relative shared 混训：

```bash
ROBOT_GRIPPER_EEF_DATASET=/path/to/robot_eef20 \
HUMAN_EEF_ONLY_DATASET=/path/to/human_eef20_gripper \
NORM_PRESET=robot2_gripper_rel_shared \
bash compute_unitree_norm_stats.sh

ROBOT_GRIPPER_EEF_DATASET=/path/to/robot_eef20 \
HUMAN_EEF_ONLY_DATASET=/path/to/human_eef20_gripper \
NORM_PRESET=human_eef_only_rel_shared \
bash compute_unitree_norm_stats.sh
```

## 4. 训练

BrainCo Robot + EEF18 Human：

```bash
ROBOT_EEF_DATASET=/path/to/robot_eef30 \
HUMAN_EEF_ONLY_DATASET=/path/to/human_eef20_gripper \
PRESET=mix_robot2_human3_eef_only_abs \
EXP_NAME=robot_brainco_human_eef18_abs \
BATCH_SIZE=32 NUM_TRAIN_STEPS=30000 NUM_WORKERS=8 \
ROBOT_FRACTION=0.5 \
SAVE_INTERVAL=5000 KEEP_PERIOD=5000 WANDB_MODE=off \
bash train_unitree.sh
```

二指爪 Robot + EEF18 Human：

```bash
ROBOT_GRIPPER_EEF_DATASET=/path/to/robot_eef20 \
HUMAN_EEF_ONLY_DATASET=/path/to/human_eef20_gripper \
PRESET=mix_robot2_human3_gripper_eef_only_rel_shared \
EXP_NAME=robot_gripper_human_eef18_rel_shared \
BATCH_SIZE=32 NUM_TRAIN_STEPS=30000 NUM_WORKERS=8 \
ROBOT_FRACTION=0.5 \
SAVE_INTERVAL=5000 KEEP_PERIOD=5000 WANDB_MODE=off \
bash train_unitree.sh
```

将 `human3` 改为 `human4`、`human5` 或 `human6` 即可切换可见模态。完整命名规则：

```text
mix_robot2_human{3|4|5|6}_eef_only_abs
mix_robot2_human{3|4|5|6}_eef_only_rel_{shared|per_step|hybrid}

mix_robot2_human{3|4|5|6}_gripper_eef_only_abs
mix_robot2_human{3|4|5|6}_gripper_eef_only_rel_{shared|per_step|hybrid}
```

所有组合都可追加 `_progress`。progress 训练要求 Robot/Human 任务文本一致。Human 统计名称：

```text
# BrainCo Robot 作为任务时长参考
human_eef_only_abs_progress
human_eef_only_rel_shared_progress
human_eef_only_rel_per_step_progress
human_eef_only_rel_hybrid_progress

# gripper Robot 作为任务时长参考
human_eef_only_robot_gripper_abs_progress
human_eef_only_robot_gripper_rel_shared_progress
human_eef_only_robot_gripper_rel_per_step_progress
human_eef_only_robot_gripper_rel_hybrid_progress
```

计算 progress Human 统计时必须同时传入 Robot 真机数据路径；Robot 统计仍使用对应的普通
`robot2_*` preset。训练 preset 的表示和 `_progress` 后缀必须与 Human 统计完全一致。

默认 `ROBOT_FRACTION=0.5`，保持原来的 1:1 批次比例。Human EEF-only 不直接监督 Robot
BrainCo/gripper 尾部；若真机评估显示尾部学习不足，可以做 `0.6` 或 `0.7` 的对照实验。该比例只改变
混合采样，不改变 Robot/Human 各自的归一化统计。

## 5. 推理部署

部署命令无需新增 Human 模式。服务端从 checkpoint 中选取唯一 Robot component 和 Robot
归一化资产：

```bash
uv run scripts/serve_unitree_experiment.py \
  --checkpoint checkpoints/unitree_g1d_brainco_train/<EXP>/<STEP> \
  --default-prompt "fold clothes."
```

BrainCo Robot checkpoint 仍输出 EEF30，Dex1 checkpoint 仍输出 EEF20。客户端继续按 Robot
manifest 做 IK，并原样转发 Robot 的 BrainCo/gripper 尾部。Human EEF18 统计不会参与部署反归一化。

不要尝试部署只有 Human EEF18、没有 Robot component 的 checkpoint；它没有真实机器人末端执行器
契约。正式部署应始终使用本文的 `mix_robot2_..._eef_only_...` preset。

## 6. 兼容性

- 旧 preset 不会自动启用 EEF-only。
- 旧 30D BrainCo 和完整 20D gripper preset 的校验、归一化与监督范围不变。
- 只有名称包含 `eef_only` 的 preset 才会丢弃 Human `[18:20]`；普通 `human_gripper_*`
  仍使用并监督完整 EEF20。
- Robot/Human 默认仍按 1:1 采样。
- 所有 preset 默认使用全部合格 episode，不创建验证集。
- Human 只提供 EEF 监督，手部动作质量最终取决于 Robot 数据覆盖量。

## 7. 迁移与自检

不要只复制两个 Shell 文件。EEF20 到 EEF18 的投影、维度 loss mask、progress source horizon
和 checkpoint 部署契约跨越多个模块。推荐按
[`unitree_human_eef_only_migration_files.txt`](unitree_human_eef_only_migration_files.txt)
同步全部文件，然后检查：

所有相对路径都以实际工程根目录为准，也就是同时包含 `pyproject.toml`、`src/`、`scripts/` 的目录。
不要把文件覆盖到工程里的嵌套备份目录。先确认当前解释器实际导入的是根目录源码：

```bash
cd /path/to/openpi
tar -czf openpi_human_eef_only_patch.tar.gz \
  -T docs/unitree_human_eef_only_migration_files.txt
```

```bash
cd /path/to/openpi

UV_NO_SYNC=1 .venv/bin/python scripts/prepare_unitree_norm_stats.py --help \
  | grep human-eef-only-dataset

UV_NO_SYNC=1 .venv/bin/python scripts/train_unitree.py --help \
  | grep human-eef-only-dataset

UV_NO_SYNC=1 .venv/bin/python -c \
  "import openpi.training.unitree_train_config as m; print(m.__file__)"
```

最后一条必须指向当前工程根目录下的 `src/openpi/training/unitree_train_config.py`；若指向另一份
`openpi/src/...`、`openpi_zh/src/...` 或旧 site-packages，说明仍在运行旧副本。

使用 EEF20 Human 数据执行 `human_eef_only_*` 后，生成的
`norm_stats_manifest.json` 必须包含：

```text
canonical_layout.dimension = 18
canonical_layout.action_domain = eef_only
source_projection.input_dimension = 20
source_projection.output_dimension = 18
source_projection.ignored_slice = [18, 20]
```

迁移清单文本是唯一文件列表，可直接传给 `tar -T`；不要手工挑选其中几项。模型、transform 和
data loader 文件负责 18D loss mask 与 progress source horizon，不能只复制新增 preset 文件。
归一化资产不要从旧 EEF18 实验直接复用，应在目标服务器上从正式 Human EEF20 数据重新计算。
