# G1-D joint 转 EEF

`convert_g1_joint_to_eef.py` 会根据数据字段自动识别两种输入：

- `arm14 + BrainCo12` -> `EEF18 + BrainCo12`（30D）
- `arm14 + left/right gripper2` -> `EEF18 + left/right gripper2`（20D）

夹爪和 BrainCo 尾部均逐值复制，不参与 FK。EEF 使用
`torso_link -> left/right_hand_palm_link`，Rot6D 顺序为 grouped columns。

## 当前夹爪数据

输入数据：

```text
/home/zh/w_ego_collect/IL/openpi/dataset/clothers_3eps
```

已经转换并验证：3 episodes、7651 frames，输出 state/action 均为
`[left EEF9, right EEF9, kLeftGripper, kRightGripper]`：

```text
/home/zh/w_ego_collect/IL/openpi/dataset/clothers_3eps_eef20_columns_grouped
```

## 终端命令

先只读检查一个 episode：

```bash
cd /home/zh/w_ego_collect/IL/openpi
.venv/bin/python tools/convert_g1_joint_to_eef.py \
  /home/zh/w_ego_collect/IL/openpi/dataset/clothers_3eps \
  /tmp/clothers_3eps_eef20_check \
  --urdf /home/zh/w_ego_collect/IL/openpi/third_party/g1_d_description/g1_d.urdf \
  --max-episodes 1 \
  --dry-run
```

首次转换全部数据：

```bash
cd /home/zh/w_ego_collect/IL/openpi
.venv/bin/python tools/convert_g1_joint_to_eef.py \
  /home/zh/w_ego_collect/IL/openpi/dataset/clothers_3eps \
  /home/zh/w_ego_collect/IL/openpi/dataset/clothers_3eps_eef20_columns_grouped \
  --urdf /home/zh/w_ego_collect/IL/openpi/third_party/g1_d_description/g1_d.urdf
```

输出目录必须不存在。当前目录已经生成，不需要重复转换；重新实验时请换一个新输出目录。

直接训练原始 joint16-gripper absolute（不需要 EEF 转换）：

```bash
cd /home/zh/w_ego_collect/IL/openpi
NORM_PRESET=robot1_gripper_abs bash compute_unitree_norm_stats.sh
PRESET=robot1_gripper_abs bash train_unitree.sh
```

训练转换后的 EEF20-gripper relative：

```bash
cd /home/zh/w_ego_collect/IL/openpi
NORM_PRESET=robot2_gripper_rel_shared bash compute_unitree_norm_stats.sh
PRESET=robot2_gripper_rel_shared bash train_unitree.sh
```

所有 preset 均不划验证集，全部 episode 用于归一化和训练。

## 替换失败回合并转 EEF20

下面这组命令保留原数据，使用当前 `episode 30` 完整替换失败的
`episode 20`，同时把任务文本统一为 `fold clothes.`：

```bash
cd /home/zh/w_ego_collect/IL/openpi
.venv/bin/python tools/replace_lerobot_episode.py \
  /home/zh/w_ego_collect/IL/openpi/dataset/gripper_clothers/clothers_data_curated \
  /home/zh/w_ego_collect/IL/openpi/dataset/gripper_clothers/clothers_data_curated_success50 \
  --destination-episode 20 \
  --source-episode 30 \
  --random-seed 20260724 \
  --output-task "fold clothes."

.venv/bin/python tools/convert_g1_joint_to_eef.py \
  /home/zh/w_ego_collect/IL/openpi/dataset/gripper_clothers/clothers_data_curated_success50 \
  /home/zh/w_ego_collect/IL/openpi/dataset/gripper_clothers/clothers_data_curated_success50_eef20 \
  --urdf /home/zh/w_ego_collect/IL/openpi/third_party/g1_d_description/g1_d.urdf \
  --output-task "fold clothes."
```

两个输出目录必须事先不存在。视频默认使用硬链接，不会重复占用视频磁盘空间；
`meta/replacement_manifest.json` 会记录替换来源和随机种子。
