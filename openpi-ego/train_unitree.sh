#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Mode 含义
#
# Robot mode1：真机数据；输入 state；动作是 26D 绝对关节角：
#   左臂7D + 右臂7D + 左BrainCo6D + 右BrainCo6D。
# Robot mode1 + gripper：输入 state；动作是 16D 绝对关节量：
#   左臂7D + 右臂7D + 左夹爪1D + 右夹爪1D。
#
# Robot mode2：真机数据；输入 state；动作是 30D EEF：
#   左EEF9D(xyz + Rot6D前两列) + 右EEF9D + 左BrainCo6D + 右BrainCo6D。
#   支持 absolute；也支持 EEF18 relative + BrainCo12 absolute。
#   坐标固定为 torso_link -> left/right_hand_palm_link；Rot6D 为 grouped columns。
#   joint转EEF与真机客户端必须加载内容相同的G1-D URDF，训练/checkpoint会校验URDF SHA256。
# Robot mode2 + gripper 动作域：使用独立的原生 20D EEF 数据集：
#   左EEF9D + 右EEF9D + 左夹爪1D + 右夹爪1D。
#   relative 只作用于 EEF18，左右夹爪始终是绝对动作。不会从 BrainCo12 自动生成夹爪值。
#
# Human mode3：人类数据；输入 state + 单路头部相机；30D EEF 动作。
# Human mode4：人类数据；不向模型输入 state + 单路头部相机；30D EEF 动作。
# Human mode5：人类数据；输入 state + 三相机接口；30D EEF 动作。
# Human mode6：人类数据；不向模型输入 state + 三相机接口；30D EEF 动作。
# 新 eef_only preset 让任一 Human mode 只监督 18D 双臂 EEF；若源数据为
# EEF20-gripper，会在归一化和训练变换前丢弃最后 2D，不补伪标签。
# 注意：当前 Human 数据只有真实头部相机，mode5/6 的腕部相机为 mask=false 的零图。
# mode3/4/5/6 使用同一数据集、动作变换和 train split，因此共用 Human action stats；
# mode4/6 中保存的 state stats 不会作为 state token 输入模型。
#
# abs：EEF/关节和 BrainCo 都使用绝对动作。
# rel：EEF 使用相对动作，BrainCo 始终使用绝对动作。
# rel_shared：所有 action horizon step 共用一套归一化统计（默认推荐）。
# rel_per_step：每个 horizon step 分别使用一套归一化统计。
# rel_hybrid：EEF 每步独立统计，BrainCo 所有 step 共用统计。
#
# mix：只允许 Robot mode2 + Human mode3/4/5/6 中的一个，默认 Robot:Human=1:1。
#
# Robot 相机：
#   现有所有 preset 不变，仍使用头部 + 左右腕部三相机。
#   在任意包含 Robot 的 preset 末尾添加 _single，即只读取头部相机；
#   例如 robot1_abs_single、robot2_rel_shared_single、
#   mix_robot2_human3_abs_progress_single。模型的两路腕部图像会填零且 mask=false。
# =============================================================================

# =============================================================================
# 可修改下面默认 PRESET，也可在终端用 PRESET=... 临时覆盖。
# 混训前需在 compute_unitree_norm_stats.sh 分别计算同表示的 Robot 和 Human stats。
# =============================================================================

# Robot only
# PRESET="robot1_abs"
# PRESET="robot1_gripper_abs"
# PRESET="robot2_abs"
# PRESET="robot2_rel_shared"
# PRESET="robot2_rel_per_step"
# PRESET="robot2_rel_hybrid"
# 单头部相机：在上述任一 Robot preset 末尾添加 "_single"。
# PRESET="robot1_abs_single"
# PRESET="robot2_abs_single"
# PRESET="robot2_rel_shared_single"

# Robot EEF20-gripper only
# PRESET="robot2_gripper_abs"
PRESET="${PRESET:-robot2_gripper_rel_shared}"
# PRESET="robot2_gripper_rel_per_step"
# PRESET="robot2_gripper_rel_hybrid"

# Human mode3 only
# PRESET="human3_abs"
# PRESET="human3_rel_shared"
# PRESET="human3_rel_per_step"
# PRESET="human3_rel_hybrid"

# Human mode4 only
# PRESET="human4_abs"
# PRESET="human4_rel_shared"
# PRESET="human4_rel_per_step"
# PRESET="human4_rel_hybrid"

# Human mode5 only
# PRESET="human5_abs"
# PRESET="human5_rel_shared"
# PRESET="human5_rel_per_step"
# PRESET="human5_rel_hybrid"

# Human mode6 only
# PRESET="human6_abs"
# PRESET="human6_rel_shared"
# PRESET="human6_rel_per_step"
# PRESET="human6_rel_hybrid"

# Human EEF20-gripper 示例（mode3；mode4/5/6 名字规则相同）
# PRESET="human3_gripper_abs"
# PRESET="human3_gripper_rel_shared"
# PRESET="human3_gripper_rel_per_step"
# PRESET="human3_gripper_rel_hybrid"

# Robot mode2 + Human mode3（严格 1:1）
# PRESET="mix_robot2_human3_abs"
# PRESET="mix_robot2_human3_abs_progress"  # 仅 Human action 按任务进度重采样
# PRESET="mix_robot2_human3_rel_shared"
# PRESET="mix_robot2_human3_rel_shared_progress"
# PRESET="mix_robot2_human3_rel_per_step"
# PRESET="mix_robot2_human3_rel_per_step_progress"
# PRESET="mix_robot2_human3_rel_hybrid"
# PRESET="mix_robot2_human3_rel_hybrid_progress"

# Robot mode2 + Human mode4（严格 1:1）
# PRESET="mix_robot2_human4_abs"
# PRESET="mix_robot2_human4_abs_progress"
# PRESET="mix_robot2_human4_rel_shared"
# PRESET="mix_robot2_human4_rel_shared_progress"
# PRESET="mix_robot2_human4_rel_per_step"
# PRESET="mix_robot2_human4_rel_per_step_progress"
# PRESET="mix_robot2_human4_rel_hybrid"
# PRESET="mix_robot2_human4_rel_hybrid_progress"

# Robot mode2 + Human mode5（严格 1:1）
# PRESET="mix_robot2_human5_abs"
# PRESET="mix_robot2_human5_abs_progress"
# PRESET="mix_robot2_human5_rel_shared"
# PRESET="mix_robot2_human5_rel_shared_progress"
# PRESET="mix_robot2_human5_rel_per_step"
# PRESET="mix_robot2_human5_rel_per_step_progress"
# PRESET="mix_robot2_human5_rel_hybrid"
# PRESET="mix_robot2_human5_rel_hybrid_progress"

# Robot mode2 + Human mode6（严格 1:1）
# PRESET="mix_robot2_human6_abs"
# PRESET="mix_robot2_human6_abs_progress"
# PRESET="mix_robot2_human6_rel_shared"
# PRESET="mix_robot2_human6_rel_shared_progress"
# PRESET="mix_robot2_human6_rel_per_step"
# PRESET="mix_robot2_human6_rel_per_step_progress"
# PRESET="mix_robot2_human6_rel_hybrid"
# PRESET="mix_robot2_human6_rel_hybrid_progress"

# EEF20-gripper 混训示例（其他 Human mode 名字规则相同）
# PRESET="mix_robot2_human3_gripper_abs"
# PRESET="mix_robot2_human3_gripper_abs_progress"
# PRESET="mix_robot2_human3_gripper_rel_shared"
# PRESET="mix_robot2_human3_gripper_rel_shared_progress"
# PRESET="mix_robot2_human3_gripper_rel_per_step"
# PRESET="mix_robot2_human3_gripper_rel_per_step_progress"
# PRESET="mix_robot2_human3_gripper_rel_hybrid"
# PRESET="mix_robot2_human3_gripper_rel_hybrid_progress"

# Human 只监督 EEF18、Robot 使用 EEF30-BrainCo（Human 源数据默认仍为 EEF20-gripper）
# PRESET="mix_robot2_human3_eef_only_abs"
# PRESET="mix_robot2_human3_eef_only_abs_progress"
# PRESET="mix_robot2_human3_eef_only_rel_shared"
# PRESET="mix_robot2_human3_eef_only_rel_shared_progress"
# PRESET="mix_robot2_human3_eef_only_rel_per_step"
# PRESET="mix_robot2_human3_eef_only_rel_per_step_progress"
# PRESET="mix_robot2_human3_eef_only_rel_hybrid"
# PRESET="mix_robot2_human3_eef_only_rel_hybrid_progress"

# Human 只监督 EEF18、Robot 使用 EEF20-gripper（Human 源数据默认仍为 EEF20-gripper）
# PRESET="mix_robot2_human3_gripper_eef_only_abs"
# PRESET="mix_robot2_human3_gripper_eef_only_abs_progress"
# PRESET="mix_robot2_human3_gripper_eef_only_rel_shared"
# PRESET="mix_robot2_human3_gripper_eef_only_rel_shared_progress"
# PRESET="mix_robot2_human3_gripper_eef_only_rel_per_step"
# PRESET="mix_robot2_human3_gripper_eef_only_rel_per_step_progress"
# PRESET="mix_robot2_human3_gripper_eef_only_rel_hybrid"
# PRESET="mix_robot2_human3_gripper_eef_only_rel_hybrid_progress"

# 数据集路径：必须与 compute_unitree_norm_stats.sh 中完全一致。
ROBOT_JOINT_DATASET="${ROBOT_JOINT_DATASET:-/home/zh/w_ego_collect/IL/openpi/dataset/G1_Brainco_Abc_fold_clothes_7_7/G1_Brainco_Abc_fold_clothes_7_7}"
ROBOT_GRIPPER_JOINT_DATASET="${ROBOT_GRIPPER_JOINT_DATASET:-/home/zh/w_ego_collect/IL/openpi/dataset/clothers_3eps}"
ROBOT_EEF_DATASET="${ROBOT_EEF_DATASET:-/home/zh/w_ego_collect/IL/openpi/dataset/G1_Brainco_Abc_fold_clothes_7_7/G1_Brainco_Abc_fold_clothes_7_7_curated_eef30_columns_grouped}"
HUMAN_DATASET="${HUMAN_DATASET:-/home/zh/w_ego_collect/IL/openpi/dataset/pick_bottle_put_in_box_have_state}"
# 新 gripper preset 只读取下面两条路径；数据必须原生为 EEF20=[EEF18,left_gripper,right_gripper]。
ROBOT_GRIPPER_EEF_DATASET="${ROBOT_GRIPPER_EEF_DATASET:-/home/zh/w_ego_collect/IL/openpi/dataset/clothers_3eps_eef20_columns_grouped}"
HUMAN_GRIPPER_DATASET="${HUMAN_GRIPPER_DATASET:-/home/zh/w_ego_collect/IL/openpi/dataset/pick_bottle_put_in_box_have_state_gripper_eef20}"
# 默认复用带二指爪的 Human 数据；EEF-only preset 会在训练前丢弃 [18:20]。
HUMAN_EEF_ONLY_DATASET="${HUMAN_EEF_ONLY_DATASET:-${HUMAN_GRIPPER_DATASET}}"
# Ego 源模式与 OpenPI Human mode3/4/5/6 无关。旧默认数据是 Mode1。
HUMAN_DATASET_MODE="${HUMAN_DATASET_MODE:-mode1}"
# Mode-aware canonicalization is identical in normalization and training:
#   Mode1: G1 pelvis/wrist-yaw TCP -> OpenPI torso/palm.
#   Mode2: keep recording reference, move wrist-yaw TCP -> palm.
if [[ -z "${HUMAN_INPUT_FRAME:-}" ]]; then
  if [[ "${HUMAN_DATASET_MODE}" == "mode1" ]]; then
    HUMAN_INPUT_FRAME="g1_base_tcp"
  else
    HUMAN_INPUT_FRAME="recording_tcp"
  fi
fi
# Robot episode 任务过滤；空字符串表示不过滤。默认值保持原有 fold clothes 行为。
ROBOT_TASK_CONTAINS="${ROBOT_TASK_CONTAINS-fold clothes}"

# 实验名；默认直接使用 PRESET，通常不用改。
EXP_NAME="${EXP_NAME:-${PRESET}}"

# 训练设备。可直接修改默认值，也可在启动时临时覆盖，例如：
#   DEVICE_IDS=2,3 bash train_unitree.sh
# 注意：JAX 会将可见的物理卡重新编号为 0..N-1。
DEVICE_IDS="${DEVICE_IDS:-0}"
export CUDA_VISIBLE_DEVICES="${DEVICE_IDS}"

# checkpoint 行为：三选一，其余注释。
# TRAIN_LIFECYCLE="overwrite"
# TRAIN_LIFECYCLE="resume"
TRAIN_LIFECYCLE="${TRAIN_LIFECYCLE:-new}"

# 常用训练参数。
BATCH_SIZE="${BATCH_SIZE:-1}"
FSDP_DEVICES="${FSDP_DEVICES:-1}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-16}"
NUM_WORKERS="${NUM_WORKERS:-1}"
# 默认保持原来的 Robot:Human=1:1。EEF-only Human 不监督手部尾部时，可按实验需要提高到 0.6/0.7。
ROBOT_FRACTION="${ROBOT_FRACTION:-0.5}"
# 所有 preset 都不划验证集，也不执行 validation forward 或模态诊断。
VALIDATION_INTERVAL=0
VALIDATION_BATCHES=0
MODALITY_DIAGNOSTICS_INTERVAL=0
SAVE_INTERVAL="${SAVE_INTERVAL:-8}"
# 与 SAVE_INTERVAL 相同时，每次定期保存的 checkpoint 都会保留。
# 可在终端单独覆盖；默认与 SAVE_INTERVAL 相同，保持原有行为不变。
KEEP_PERIOD="${KEEP_PERIOD:-${SAVE_INTERVAL}}"

# 所有 Robot/Human episode 都进入训练集。
ROBOT_VALIDATION_EPISODES=0
HUMAN_VALIDATION_EPISODES=0

# W&B：二选一。
# OPENPI_WANDB_MODE 是脚本内部开关；兼容终端原有的 WANDB_MODE=on/off 写法。
# 读取后清除 WANDB_MODE，避免将非官方值 "on"/"off" 传给 wandb SDK。
OPENPI_WANDB_MODE="${OPENPI_WANDB_MODE:-${WANDB_MODE:-off}}"
unset WANDB_MODE

# JAX 显存比例。
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# =============================================================================
# 下面无需修改。
# =============================================================================

if [[ $# -ne 0 ]]; then
  echo "本脚本不需要命令行参数；请在文件顶部取消注释一个 PRESET。" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

command=(
  uv run scripts/train_unitree.py
  --exp-name "${EXP_NAME}"
  --preset "${PRESET}"
  --robot-joint-dataset "${ROBOT_JOINT_DATASET}"
  --robot-gripper-joint-dataset "${ROBOT_GRIPPER_JOINT_DATASET}"
  --robot-eef-dataset "${ROBOT_EEF_DATASET}"
  --human-dataset "${HUMAN_DATASET}"
  --human-eef-only-dataset "${HUMAN_EEF_ONLY_DATASET}"
  --robot-gripper-eef-dataset "${ROBOT_GRIPPER_EEF_DATASET}"
  --human-gripper-dataset "${HUMAN_GRIPPER_DATASET}"
  --human-dataset-mode "${HUMAN_DATASET_MODE}"
  --human-input-frame "${HUMAN_INPUT_FRAME}"
  --robot-task-contains "${ROBOT_TASK_CONTAINS}"
  --robot-fraction "${ROBOT_FRACTION}"
  --robot-validation-episodes "${ROBOT_VALIDATION_EPISODES}"
  --human-validation-episodes "${HUMAN_VALIDATION_EPISODES}"
  --batch-size "${BATCH_SIZE}"
  --fsdp-devices "${FSDP_DEVICES}"
  --num-train-steps "${NUM_TRAIN_STEPS}"
  --num-workers "${NUM_WORKERS}"
  --validation-interval "${VALIDATION_INTERVAL}"
  --validation-batches "${VALIDATION_BATCHES}"
  --modality-diagnostics-interval "${MODALITY_DIAGNOSTICS_INTERVAL}"
  --save-interval "${SAVE_INTERVAL}"
  --keep-period "${KEEP_PERIOD}"
)

case "${TRAIN_LIFECYCLE}" in
  overwrite) command+=(--overwrite) ;;
  resume) command+=(--resume) ;;
  new) ;;
  *)
    echo "TRAIN_LIFECYCLE 只能是 overwrite、resume 或 new。" >&2
    exit 2
    ;;
esac

case "${OPENPI_WANDB_MODE}" in
  on) ;;
  off) command+=(--no-wandb) ;;
  *)
    echo "OPENPI_WANDB_MODE（兼容 WANDB_MODE）只能是 on 或 off。" >&2
    exit 2
    ;;
esac

echo "PRESET=${PRESET} EXP_NAME=${EXP_NAME} HUMAN_DATASET_MODE=${HUMAN_DATASET_MODE} HUMAN_INPUT_FRAME=${HUMAN_INPUT_FRAME} ROBOT_FRACTION=${ROBOT_FRACTION} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} FSDP_DEVICES=${FSDP_DEVICES}"
exec "${command[@]}"
