# Unitree Robot 单头部相机训练与推理

本文描述新训练链路中的 Robot 单头部相机模式。它同时适用于：

- mode1 关节动作：joint26 BrainCo、joint16 gripper；
- mode2 EEF 动作：EEF30 BrainCo、EEF20 gripper；
- mode2 Robot 与任一 Human mode 的混合训练；
- absolute、relative shared/per-step/hybrid 和 progress alignment。

原有 preset 和 checkpoint 行为不变：不带后缀的 Robot preset 仍然读取头部、左腕、右腕三路相机。

## 1. Preset 命名

在任意包含 Robot 的训练 preset 最后追加 `_single`：

```text
robot1_abs_single
robot1_gripper_abs_single
robot2_abs_single
robot2_rel_shared_single
robot2_rel_per_step_single
robot2_rel_hybrid_single
robot2_gripper_abs_single
robot2_gripper_rel_shared_single
mix_robot2_human3_abs_progress_single
mix_robot2_human6_gripper_eef_only_rel_shared_progress_single
```

`_single` 必须是最后一个后缀。例如 progress 模式写成：

```text
mix_robot2_human3_abs_progress_single
```

Human-only preset 不接受 `_single`，因为 Human mode3/4/5/6 已经单独定义相机接口。

## 2. 数据字段和模型 mask

Robot 单相机数据只要求：

```text
observation.images.cam_left_high
```

训练 repack 后它会成为 `cam_high`，再映射到模型的 `base_0_rgb`。模型结构仍保留三个固定图像槽位：

```text
base_0_rgb          = 真实头部图像，mask=true
left_wrist_0_rgb    = 同尺寸零图，mask=false
right_wrist_0_rgb   = 同尺寸零图，mask=false
```

不会把头部图像复制成腕部图像。三相机 preset 的字段和 mask 行为保持原样。

## 3. 归一化

图像不参与 state/action 数值归一化，所以单相机和三相机故意共享 stats 与 `asset_id`：

```bash
export PYTHON="$CONDA_PREFIX/bin/python"

ROBOT_TASK_CONTAINS="make sandwich" \
ROBOT_JOINT_DATASET="$PWD/dataset/sandwich/pi05_unitree_g1_brainco_make_sandwich_7_7_mixture" \
NORM_PRESET=robot1_abs_single \
bash compute_unitree_norm_stats.sh
```

上面的输出位置与 `NORM_PRESET=robot1_abs` 完全相同。如果对应三相机 preset 的正式 stats 已存在，
无需重新计算。

EEF 示例：

```bash
ROBOT_TASK_CONTAINS="make sandwich" \
ROBOT_EEF_DATASET="/path/to/robot_eef30_dataset" \
NORM_PRESET=robot2_rel_shared_single \
bash compute_unitree_norm_stats.sh
```

`ROBOT_TASK_CONTAINS` 是大小写不敏感的 Robot task 子串。设置为空字符串表示使用数据集内所有合格
episode：

```bash
ROBOT_TASK_CONTAINS="" ...
```

默认仍为 `fold clothes`，因此原有实验不受影响。

## 4. 训练

关节动作示例：

```bash
UV_NO_SYNC=1 \
UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX" \
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
ROBOT_TASK_CONTAINS="make sandwich" \
ROBOT_JOINT_DATASET="$PWD/dataset/sandwich/pi05_unitree_g1_brainco_make_sandwich_7_7_mixture" \
PRESET=robot1_abs_single \
EXP_NAME=robot1_abs_single \
DEVICE_IDS=0,1,2,3,4,5,6,7 \
FSDP_DEVICES=1 \
BATCH_SIZE=128 \
NUM_TRAIN_STEPS=30000 \
NUM_WORKERS=16 \
SAVE_INTERVAL=5000 \
KEEP_PERIOD=5000 \
TRAIN_LIFECYCLE=new \
bash train_unitree.sh
```

EEF 只需改为对应 preset 和数据集变量，例如：

```bash
ROBOT_EEF_DATASET="/path/to/robot_eef30_dataset" \
PRESET=robot2_rel_shared_single \
EXP_NAME=robot2_rel_shared_single \
bash train_unitree.sh
```

训练写入 checkpoint 的 `assets/runtime_manifest.json` 会同时记录：

```json
{
  "robot_camera_mode": "single",
  "components": [
    {
      "kind": "robot",
      "camera_mode": "single"
    }
  ]
}
```

## 5. 真机推理

启动方式不增加额外相机参数：

```bash
uv run scripts/serve_unitree_experiment.py \
  --checkpoint /path/to/checkpoint/30000 \
  --default-prompt "make sandwich." \
  --port 8000
```

服务端从 checkpoint manifest 恢复 `single` 或 `three`，因此训练与推理不会因手工参数写错而不一致。
旧 checkpoint 如果没有相机字段，按历史行为回退为 `three`。

真机客户端 `examples/unitree_inference/robot_interface.py` 已经只强制要求头部图像，并会把
`cam_left_high` 重命名为 `cam_high`。单相机 checkpoint 可以不提供左右腕部图像；如果真机环境仍返回
腕部图像，单相机服务端会忽略它们。

mode2 EEF 推理的 URDF SHA256、reference frame、EEF links、Rot6D、relative/absolute 解码和 IK 检查
保持不变。
