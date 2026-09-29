# Unitree G1-D + BrainCo / EEF20-gripper 训练实验

Human 从 EEF20-gripper 数据读取、但只使用双臂 EEF18 监督时，使用独立说明：
[unitree_human_eef_only_training.md](unitree_human_eef_only_training.md)。
Ego Mode1/Mode2 的完整源数据语义和支持矩阵见
[unitree_ego_data_contract.md](unitree_ego_data_contract.md)。

这套配置只负责新训练实验。原有 `src/openpi/training/unitree_config.py` 和
`scripts/serve_unitree_policy.py` 继续服务旧 checkpoint，不共享新实验的 mode 语义。
模型、policy、训练 step、RTC 推理实现仍然共用。

## 1. 动作和坐标定义

本项目把“机器人 base”固定为 G1-D URDF 中的 `torso_link`，不是 `pelvis`、`base_link` 或 world。
记 `B=torso_link`，`E_L=left_hand_palm_link`，`E_R=right_hand_palm_link`。绝对 pose 始终是
`T_B_E`。joint→EEF 转换和真机客户端 FK 都计算同一个式子：

```text
T_B_E = inverse(T_world_B) @ T_world_E
```

转换所用 URDF 的 SHA256 会写入 EEF 数据集、归一化/训练运行时 manifest 和 checkpoint。新 EEF
客户端从 checkpoint 取得 `B/E` link 与 SHA256，必须通过 `--urdf-path` 加载内容相同的 URDF，哈希
不同会在控制机器人前报错。

新实验的 EEF30 始终为：

```text
[0:3]   left xyz，torso_link -> left_hand_palm_link
[3:6]   left R[:, 0]
[6:9]   left R[:, 1]
[9:12]  right xyz，torso_link -> right_hand_palm_link
[12:15] right R[:, 0]
[15:18] right R[:, 1]
[18:24] left BrainCo 6D，绝对值
[24:30] right BrainCo 6D，绝对值
```

这里的 Rot6D 是 grouped columns：`[r00,r10,r20,r01,r11,r21]`。旧配置中的
`rotation_format="columns"` 是交错排列，含义不同；为了保证旧 checkpoint 不变，未修改旧格式。

Human Ego LeRobot 已经完成“人体→G1”的 retargeting，但还要统一 G1 机器人内部 link convention。
Mode1 保存 G1 pelvis base 下的 `wrist_yaw` TCP 绝对目标；Human-only 和混训都转换为
torso→hand-palm。当前 Mode2 保存 recording frame 下的当前/下一帧绝对 TCP；保留 recording reference，
但把控制点从 wrist-yaw TCP 移到 palm。默认按数据源自动使用：

```text
Mode1 absolute/relative: HUMAN_DATASET_MODE=mode1 HUMAN_INPUT_FRAME=g1_base_tcp
Mode2 relative only:     HUMAN_DATASET_MODE=mode2 HUMAN_INPUT_FRAME=recording_tcp
```

Mode2 absolute 混训和显式 `native` 会直接报错。旧 Mode2 如果没有
`meta/action_semantics.json` 也会拒绝，不能再根据 shape 猜测动作语义。

迁移后先清理旧 Human stats（第一条只列出，确认后第二条删除）：

```bash
python scripts/purge_legacy_unitree_human_stats.py "$HUMAN_DATASET"
python scripts/purge_legacy_unitree_human_stats.py "$HUMAN_DATASET" --apply
```

若使用 `HUMAN_GRIPPER_DATASET` 或 `HUMAN_EEF_ONLY_DATASET`，也要把对应目录一并传给脚本。任何包含
Human component、但 `runtime_manifest.json` 没有完整 dataset mode、input frame 和 action contract
的旧 checkpoint 均视为无效；评估服务和 `TRAIN_LIFECYCLE=resume` 会拒绝加载，必须重新计算 Human
stats 并用新实验名从基础权重训练。

相对 EEF 使用当前观测 state 为参考：

```text
delta_p = R_state.T @ (p_action - p_state)
delta_R = R_state.T @ R_action
```

对 action chunk 中每个 horizon `k`，这里都是 `state[t]` 对 `action[t+k]`；不会改为
`action[t+k-1]` 参考，也不会累计 delta。相对 translation/rotation 表达在当前 EEF 局部坐标中。
Robot 与 Mode1 Human absolute pose 位于 `torso_link`；Mode2 保留 recording reference。所有 Human
训练都会先按 source mode 统一机器人控制点，再计算 relative。

BrainCo 12D 在绝对、相对两种 EEF 实验中始终是绝对动作。

新增的 `gripper` 动作域与上述路径并行，布局固定为：

```text
[0:3]   left xyz
[3:6]   left R[:, 0]
[6:9]   left R[:, 1]
[9:12]  right xyz
[12:15] right R[:, 0]
[15:18] right R[:, 1]
[18]    kLeftGripper，绝对值
[19]    kRightGripper，绝对值
```

它使用原生 EEF20 数据，不会把 BrainCo12 做平均、抽维或阈值化来伪造夹爪标签。数据集的
`observation.state` 和 `action` 都必须是 20D，且最后两个 feature name 必须明确为
`kLeftGripper`、`kRightGripper`。相对模式仍只变换前 18D，两个夹爪值保持绝对。

## 2. joint26 / joint16 转 EEF

先做 3 个 episode 的只读验证：

```bash
.venv/bin/python tools/convert_g1_joint_to_eef.py \
  dataset/G1_Brainco_Abc_fold_clothes_7_7/G1_Brainco_Abc_fold_clothes_7_7 \
  /tmp/g1d_eef_smoke \
  --urdf /home/zh/openpi/g1_d_description/g1_d.urdf \
  --task 'fold clothes.' --max-episodes 3 --dry-run
```

确认后转换全部折衣服 episode：

```bash
.venv/bin/python tools/convert_g1_joint_to_eef.py \
  dataset/G1_Brainco_Abc_fold_clothes_7_7/G1_Brainco_Abc_fold_clothes_7_7 \
  dataset/G1_Brainco_Abc_fold_clothes_7_7_eef30_columns_grouped \
  --urdf /home/zh/openpi/g1_d_description/g1_d.urdf \
  --task 'fold clothes.'
```

工具逐 episode 读取 Parquet，视频优先 hardlink，不把全量数据放进内存；同时重算统计、重排
episode/task/index，并保存 URDF SHA、frame 和 Rot6D 布局。默认使用仓库内的
`third_party/g1_d_description/g1_d.urdf`，也可以通过 `--urdf` 指定 G1-D URDF。

同一个工具会按 feature schema 自动识别 `arm14+BrainCo12` 和 `arm14+gripper2`。例如将仓库中的
Dex1 16D 数据转换为 EEF20：

```bash
.venv/bin/python tools/convert_g1_joint_to_eef.py \
  dataset/clothers_3eps \
  dataset/clothers_3eps_eef20_columns_grouped \
  --urdf third_party/g1_d_description/g1_d.urdf
```

转换器逐值透传两个夹爪维度，并在 `eef_forward_kinematics` 中写入
`"gripper": "copied_without_transformation"`。训练配置会再次校验这项元数据、20D shape 和尾部名称。

## 3. 归一化

归一化只使用配置中的 train episodes，也不解码视频。旧 BrainCo 配置仍排除 validation episodes；
当前 gripper 配置不划验证集，因此全部 episode 都属于 train split。

Robot 和 Human 必须单独计算，不能把两个数据域混在一起求一套统计。例如准备
`mix_robot2_human3_rel_shared` 训练时，先在脚本顶部选择 `robot2_rel_shared`：

```bash
./compute_unitree_norm_stats.sh
```

然后把脚本改为 `human_rel_shared`，再运行一次。Human mode3/4/5/6 的数值 state/action、split 和
asset id 相同，因此只需要这一份 Human 统计。结果分别写入数据集内部：

```text
<robot_dataset>/meta/openpi_assets/<robot_asset_id>/norm_stats.json
<human_dataset>/meta/openpi_assets/<human_asset_id>/norm_stats.json
```

训练时每个 component 自动加载自己数据集中的统计；checkpoint 保存时会把两份统计复制到 checkpoint
的 `assets/`，保证评估不依赖原始训练数据目录。

三种相对量方案：

- `shared`：所有有效 horizon step 共用一套 action 统计，默认推荐作为稳定基线。参数少、长短
  episode 更稳，但远期位姿变化通常更大，可能压小近端动作的归一化幅度。
- `per_step`：每个相对步各有一套统计，能平衡近端和远端监督；但 step 0 的真实变化接近零，容易
  放大 FK、同步和标定噪声，后期 step 的有效样本也更少，因此适合作为对照实验，不建议第一个跑。
- `hybrid`：EEF18 使用 per-step，BrainCo12 使用 shared 后广播到所有 step。它符合“EEF 是相对、手是
  绝对”的语义，通常是 shared 基线之后最值得跑的第二组实验。

EEF20-gripper 支持完全相同的四种统计模式。`hybrid` 在该域中表示 EEF18 per-step、gripper2 shared；
20D 统计使用独立 asset id，绝不会复用 EEF30-BrainCo 的统计。

对所有维度统一采用通用低尺度保护，而不是硬编码“某三根手指”：按 XYZ、Rot6D、左右 BrainCo
（joint 模式则按左右手臂/左右 BrainCo）分组计算 scale floor。若 `q01 == q99`，以数据中心对称扩展
区间，使常量归一化到 0，而不是当前公式中的 -1；稀疏但偶尔非零的维度会写进
`norm_stats_manifest.json`，不会被自动删除。新实验还把归一化后的输入/监督限制在 `[-5, 5]`，使任意
语义组的罕见长尾都不能产生几十万量级的值；旧配置保持不裁剪。action chunk 越过 episode 尾部的
重复 padding 不参与统计。

绝对模式仍沿用 OpenPI 的 quantile normalization 语义，只有低方差维度使用上述通用安全下限。

Robot mode1 使用 `robot1_abs`（joint26-BrainCo）或 `robot1_gripper_abs`（joint16-gripper）；Robot mode2
和 Human 的 absolute/shared/per-step/hybrid 都已经逐条列在
脚本中。底层高级入口仍然是 `scripts/prepare_unitree_norm_stats.py`；smoke test 必须显式使用
`--output-root /tmp/...`，避免把不完整统计写入数据集。

gripper 统计 preset 为 `robot2_gripper_*` 和 `human_gripper_*`。当前仓库已为
`dataset/clothers_3eps_eef20_columns_grouped` 生成 `robot2_gripper_rel_shared` 统计；其他表示在训练前按需
选择相应 preset 再运行脚本。

## 4. 训练命令

所选动作表示和 normalization 类型必须已经分别为 Robot/Human 计算完成。preset 可以在终端传入：

```bash
NORM_PRESET=robot1_gripper_abs bash compute_unitree_norm_stats.sh
PRESET=robot1_gripper_abs bash train_unitree.sh
```

混合训练严格限定为 `Robot mode2 + 一个 Human mode`。例如
`mix_robot2_human3_rel_shared`、`mix_robot2_human4_rel_shared` 是两个独立实验，不能在同一次训练里
同时选择 Human mode3 和 mode4。默认每批 Robot:Human 为 1:1；batch size 32 即 `16 + 16`。
两个 component 独立洗牌、耗尽后重新洗牌，批内再打乱，并分别使用自己的归一化资产，避免 150 万帧
Robot 数据吞没较小 Human 域的统计。

当前 Human 数据的真实任务是拿瓶放盒，并不是折衣服。混训会保留各数据集原始 task prompt，把它当作
多任务/跨域辅助数据；不会把 Human 轨迹伪标成 `fold clothes.`。因此至少要同时跑 robot-only mode2
基线，并比较固定真机任务的成功率。若 1:1 出现负迁移，可以在底层训练入口提高
`robot_fraction`，而不是改写 Human 文本标签。

### ABC absolute 任务进度对齐混训

下面两份 ABC 数据都使用 `fold clothes.`，Human 数据中的旧错误任务文本已经修正：

```text
/home/zh/w_ego_collect/IL/openpi/dataset/ABC/robot_abc_eef
/home/zh/w_ego_collect/IL/openpi/dataset/ABC/human_abc_mode1
```

`human_abc_mode1` 是数据目录自己的命名；按本训练体系，它提供state和单路头部相机，因此对应
Human mode3。

任务进度对齐是通用混训后缀模式：在任意 `mix_robot2_human3/4/5/6` EEF preset 末尾添加
`_progress` 才会启用。它支持BrainCo或gripper、absolute或relative，例如：

```text
mix_robot2_human3_abs_progress
mix_robot2_human5_rel_hybrid_progress
mix_robot2_human4_gripper_abs_progress
mix_robot2_human6_gripper_rel_shared_progress
```

所有旧 preset、Robot组件和普通单域训练均保持原来的逐帧action chunk。每次解析训练配置时，程序
都会从当前传入的数据集、当前train split和各自FPS读取所有完整episode时长，自动计算
`gamma=median(T_robot)/median(T_human)`；代码中不保存ABC的固定gamma。任务文本不一致时直接拒绝
进度对齐。当前ABC数据自动得到`gamma=2.325`；Robot仍使用未来50点，Human从未来约21.075个原始
帧间隔取样，读取23个整数源点后重采样为50点。XYZ与BrainCo/gripper线性插值，两只手的Rot6D先
恢复旋转矩阵再做四元数SLERP。图像和state始终对应chunk起始帧；relative模式在重采样之后再相对
当前state计算EEF动作。

正式统计已经生成；在新服务器或统计需要重建时执行：

```bash
cd /home/zh/w_ego_collect/IL/openpi

ROBOT_EEF_DATASET=/home/zh/w_ego_collect/IL/openpi/dataset/ABC/robot_abc_eef \
HUMAN_DATASET=/home/zh/w_ego_collect/IL/openpi/dataset/ABC/human_abc_mode1 \
NORM_PRESET=robot2_abs bash compute_unitree_norm_stats.sh

ROBOT_EEF_DATASET=/home/zh/w_ego_collect/IL/openpi/dataset/ABC/robot_abc_eef \
HUMAN_DATASET=/home/zh/w_ego_collect/IL/openpi/dataset/ABC/human_abc_mode1 \
NORM_PRESET=human_abs_progress bash compute_unitree_norm_stats.sh
```

一张GPU的训练示例：

```bash
ROBOT_EEF_DATASET=/home/zh/w_ego_collect/IL/openpi/dataset/ABC/robot_abc_eef \
HUMAN_DATASET=/home/zh/w_ego_collect/IL/openpi/dataset/ABC/human_abc_mode1 \
PRESET=mix_robot2_human3_abs_progress \
EXP_NAME=abc_abs_progress_mix \
BATCH_SIZE=32 NUM_TRAIN_STEPS=30000 NUM_WORKERS=8 \
SAVE_INTERVAL=1000 WANDB_MODE=on \
bash train_unitree.sh
```

checkpoint manifest 会保存gamma、估计器、Robot/Human episode数量、中位时长、Human源时间跨度、
插值方法和配置摘要。Human progress统计使用带digest的独立asset id，不会覆盖或复用普通
`human_abs` 统计。

其他组合的归一化规则是：Robot仍计算对应的普通统计，Human使用同名表示加`_progress`。例如
`mix_robot2_human5_rel_hybrid_progress`需要`robot2_rel_hybrid`和
`human_rel_hybrid_progress`；`mix_robot2_human4_gripper_abs_progress`需要
`robot2_gripper_abs`和`human_gripper_abs_progress`。

当前 Human 数据只有真实 `cam_high`。因此 mode5/6 的两个 wrist 槽位为零图且
`image_mask=false`，不会复制头部图像冒充三相机。这可以测试三相机模型接口下的缺相机鲁棒性，但
不能当作“真实 Human 三相机”结论；要做该结论必须补采 wrist 视频。

模型公共动作宽度为 32。新实验会在 loss 中屏蔽 16D/20D/26D/30D 后面的补零维度，并屏蔽 episode 尾部
padding timestep；旧训练配置默认不启用此变化。

EEF20-gripper 同样保持模型公共宽度 32，但只监督 `[0:20]`，`[20:32]` 由 action dimension mask 屏蔽。
训练 preset 使用 `robot2_gripper_abs`、`robot2_gripper_rel_shared/per_step/hybrid`；Human 和混训分别使用
`human{3..6}_gripper_*`、`mix_robot2_human{3..6}_gripper_*`。Robot/Human 混训时两边必须都是同一动作域。

新增 EEF-only Human 混训允许 Robot 保持 EEF30-BrainCo 或 EEF20-gripper。Human 默认仍读取原有
EEF20-gripper 数据，但在归一化和训练前丢弃 `[18:20]`，只监督 `[0:18]`；随后补齐的模型维度由
action dimension mask 排除，不会把 Human 夹爪值或伪零标签用于监督。普通 `human_gripper_*` 仍完整
使用20D。新功能仅由名称中包含 `eef_only` 的 preset 启用；完整命令见上面的独立说明。

## 5. 训练数据范围

所有 Robot、Human、BrainCo 和 gripper preset 均不划 validation episodes，筛选后的全部 episode 都进入
训练集和归一化统计。`validation_interval=0`、`validation_batches=0`、
`modality_diagnostics_interval=0`，训练进程不会创建 validation loader，也不会执行 validation 或模态消融前向。
默认仍排除少于 10 帧的异常短 episode；可用 `--minimum-episode-frames` 调整，归一化与训练必须一致。

## 6. 独立评估

新 checkpoint 使用独立、只含 Robot component 的 eval config：

```bash
uv run scripts/serve_unitree_experiment.py \
  --checkpoint checkpoints/unitree_g1d_brainco_train/<EXP>/<STEP> \
  --default-prompt 'fold clothes.'
```

评估启动时读取 checkpoint 内的 `assets/runtime_manifest.json`，校验 action representation、robot
asset、`torso -> palm` frame 和 grouped Rot6D；不允许把 relative checkpoint 配成 absolute，也不会
误用 Human norm stats。旧 checkpoint 继续使用原 `scripts/serve_unitree_policy.py`。

两套数据均为 30 FPS，50-step chunk 对应约 1.633 秒。真机执行必须使用 30 Hz，或显式将 30 Hz
动作插值到控制器频率；不能直接以 40 Hz 逐点执行，否则会把 chunk 压缩为 1.25 秒。

真机客户端必须显式给出转换阶段同一个 URDF：

```bash
examples/unitree_inference/client.sh \
  sync 'fold clothes.' SERVER_IP 8000 auto \
  --urdf-path /path/on/robot/to/the/same/g1_d.urdf \
  --dt 0.0333333333
```

EEF20 checkpoint 的 manifest 会记录 `action_domain=gripper`、`action_tail=gripper2`、
`robot_type=unitree_g1_dex1`、`model_dim=20`。客户端把前 18D 经 IK 转回双臂 14D 关节，并把 `[18:20]`
原样写入部署动作 `[14:16]` 的左右夹爪槽位。服务端和客户端都会校验这些字段，20D checkpoint 无法
误接到 BrainCo 机器人。
