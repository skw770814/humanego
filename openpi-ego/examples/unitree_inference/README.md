# Unitree OpenPI inference

本目录只包含 Unitree 真机接口、独立 FK/IK 和五种基于 Kai0 思路的推理模式。旧工程
`examples/unitree_real` 的推理代码没有被复用。

## 配置

默认 EEF pose 为 `xyz + rotation matrix 前两列`。临时兼容配置
`robot_mode4_three_with_state_rows` 使用 `xyz + rotation matrix 前两行`，其余配置均不变。相对 EEF 动作使用
`R_state.T @ (xyz_action - xyz_state)` 和 `R_state.T @ R_action`；Dex1/BrainCo 末端量保持绝对值。

| Policy | Mode | 动作/状态 | 绝对/相对 | 维度 |
|---|---:|---|---|---:|
| human | 1/2 | 双臂 EEF + Dex1 | 绝对/相对 | 20 |
| human | 3/4 | 双臂 EEF + BrainCo | 绝对/相对 | 30 |
| human | 5/6 | 双臂 EEF、无末端 | 绝对/相对 | 18 |
| robot | 1/2 | 双臂 EEF + Dex1 | 绝对/相对 | 20 |
| robot | 3/4 | 双臂 EEF + BrainCo | 绝对/相对 | 30 |
| robot | 5/6 | 双臂关节 + Dex1 | 绝对/相对 | 16 |
| robot | 7/8 | 双臂关节 + BrainCo | 绝对/相对 | 26 |

Human 配置名为 `human_mode{1..6}_{single|three}_{with_state|no_state}`；Robot 配置名为
`robot_mode{1..8}_{single|three}_with_state`，共 40 组标准配置，另有 1 组临时 rows 兼容配置。每组在
`src/openpi/training/unitree_config.py` 中有独立的 dataset、checkpoint 和 asset ID。服务端只从
`<checkpoint>/assets/<asset_id>/norm_stats.json` 加载归一化统计。

当前两份可直接测试的配置：

- `human_mode4_single_with_state`
- `human_mode4_single_no_state`

Mode 4 三相机 robot checkpoint 的前两行临时兼容配置：

- `robot_mode4_three_with_state_rows`

## 启动服务端

在 OpenPI 根目录执行：

```bash
examples/unitree_inference/serve.sh --list
examples/unitree_inference/serve.sh human_mode4_single_with_state
examples/unitree_inference/serve.sh robot_mode4_three_with_state_rows
```

第二、第三个可选参数分别覆盖 checkpoint 和端口：

```bash
examples/unitree_inference/serve.sh CONFIG /path/to/checkpoint 8000
```

新 G1-D BrainCo 训练体系不使用上面的 legacy config 名，而是由 checkpoint manifest 构建独立 eval config：

```bash
uv run scripts/serve_unitree_experiment.py \
  --checkpoint checkpoints/unitree_g1d_brainco_train/EXP/STEP \
  --default-prompt 'fold clothes.'
```

新 EEF30-BrainCo 和 EEF20-gripper checkpoint 的 Rot6D 都是 `columns_grouped`，客户端 FK/IK 已支持。
EEF20 的动作顺序为 EEF18 + 左右夹爪2D；IK 只处理 EEF18，夹爪值直接写入 Dex1 动作尾部。

使用 Human EEF-only 监督混训时，源数据可以仍是 EEF20-gripper；训练前会忽略 Human 夹爪2D。
部署格式仍由 checkpoint 的 Robot component 决定：BrainCo Robot 仍为 EEF30，Dex1 Robot 仍为
EEF20。Human 的 18D 归一化资产不会被评估或真机客户端加载。训练数据为
30 FPS，调用客户端时应显式传 `--dt 0.0333333333`；原有 legacy 客户端默认 40 Hz 未被修改。新 checkpoint
还记录了 joint→EEF 转换所用 URDF 的 SHA256、`torso_link` reference 和两个 palm link，因此客户端
必须用 `--urdf-path` 传入内容完全相同的 G1-D URDF；路径可以不同，哈希不能不同。不要依赖客户端
内置的 legacy Unitree-deploy URDF：

```bash
examples/unitree_inference/client.sh \
  sync 'fold clothes.' SERVER_IP 8000 auto \
  --urdf-path /path/on/robot/to/the/same/g1_d.urdf \
  --dt 0.0333333333
```

# examples/unitree_inference/serve.sh robot_mode5_three_with_state



###########################################################################
真机测试：
joint baseline:
examples/unitree_inference/serve.sh robot_mode7_three_with_state 
examples/unitree_inference/client.sh sync "fold clothes" 127.0.0.1 8000



## 启动真机客户端

客户端脚本首次运行会用 `unitree_deploy` conda Python 和 uv 创建本目录 `.venv`，并通过本地路径加载
当前工程的 `openpi-client` 与 `/home/zh/unitree-deploy`。机器人类型默认从服务端配置自动取得。
## SERVER_IP  127.0.0.1

```bash
examples/unitree_inference/client.sh sync "fold clothes" 127.0.0.1 8000
examples/unitree_inference/client.sh async "pick up the bottle" SERVER_IP 8000
examples/unitree_inference/client.sh temporal_ensembling "pick up the bottle" SERVER_IP 8000
examples/unitree_inference/client.sh temporal_smoothing "pick up the bottle" SERVER_IP 8000
examples/unitree_inference/client.sh rtc "pick up the bottle" SERVER_IP 8000
```
 
仅支持 `unitree_g1_dex1` 和 `unitree_g1_brainco`。连接真机后程序会等待人工按 Enter，之后才开始发送动作。
EEF 配置会使用 `g1_kinematics.py` 将机器人 14 维双臂关节经 FK 转为策略状态，并将策略动作逐帧经 IK
转回关节；末端自由度不进入 IK。

RTC 服务端使用与普通 Pi0.5 完全相同的参数树，额外消费客户端发送的 `prev_action_chunk`、
`inference_delay` 和 `execute_horizon`。历史动作块会先经过与训练一致的相对动作和 checkpoint
归一化变换，再参与 RTC guidance。

#额外的
cd /home/zh/w_ego_collect/IL/openpi
examples/unitree_inference/serve.sh robot_mode4_three_with_state_rows
