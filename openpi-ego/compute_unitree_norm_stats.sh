#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Mode 含义
#
# Robot mode1：真机 26D 绝对关节动作：
#   左臂7D + 右臂7D + 左BrainCo6D + 右BrainCo6D。
# Robot mode1 + gripper：真机 16D 绝对关节动作：
#   左臂7D + 右臂7D + 左夹爪1D + 右夹爪1D。
#
# Robot mode2：真机 30D EEF 动作：
#   左EEF9D(xyz + Rot6D前两列) + 右EEF9D + 左BrainCo6D + 右BrainCo6D。
#   坐标固定为 torso_link -> left/right_hand_palm_link；Rot6D 为 grouped columns。
#   mode2归一化会先检查转换数据内保存的frame/link/URDF SHA256元数据。
# Robot mode2 + gripper：独立的原生 20D EEF 动作：
#   左EEF9D + 右EEF9D + 左夹爪1D + 右夹爪1D。夹爪两维始终按绝对值统计。
#
# Human mode3：state + 单路头部相机。
# Human mode4：不向模型输入 state + 单路头部相机。
# Human mode5：state + 三相机接口。
# Human mode6：不向模型输入 state + 三相机接口。
# Human mode3/4/5/6 使用同一数据集、同一 EEF/BrainCo 变换和同一 train split，
# 所以共用一份 Human action stats。mode4/6 虽不向模型暴露 state token，relative EEF
# 仍需原始 state 计算参考位姿；文件中可保留同一份 state stats，但模型不会消费它。
#
# abs：EEF/关节和 BrainCo 都使用绝对动作。
# rel：EEF 使用相对动作，BrainCo 始终使用绝对动作。
# rel_shared：所有 action horizon step 共用一套归一化统计（默认推荐）。
# rel_per_step：每个 horizon step 分别使用一套归一化统计。
# rel_hybrid：EEF 每步独立统计，BrainCo 所有 step 共用统计。
#
# 归一化没有 mix 模式：Robot 和 Human 必须分开计算并写入各自数据集的 meta/openpi_assets。
# 相机不参与数值归一化；Robot 的 _single preset 与对应的原三相机 preset
# 故意使用完全相同的 asset_id 和 norm_stats。
# Human asset_id 必须同时包含 Ego Mode、源坐标/TCP profile 和数据契约摘要；
# 任何旧的、缺少完整 `_ego_mode..._source..._contract...` 标记的 stats 均废弃。
# =============================================================================

# =============================================================================
# 每次只计算一个数据域的归一化：Robot 和 Human 必须分开运行。
# 可修改下面默认值，也可在终端用 NORM_PRESET=... 临时覆盖。
#
# 如果训练是 mix_robot2_human3_rel_shared：
#   1. 先选择 robot2_rel_shared，运行一次本脚本；
#   2. 再选择 human_rel_shared，再运行一次本脚本；
#   3. 最后运行 train_unitree.sh。
#
# Human mode3/4/5/6 只改变 state/camera 是否对模型可见，不改变数值动作域，共用 Human stats。
# =============================================================================

# Robot joint26
# NORM_PRESET="robot1_abs"
# NORM_PRESET="robot1_abs_single"  # 单头部相机；stats 与 robot1_abs 共用

# Robot joint16-gripper absolute
# NORM_PRESET="robot1_gripper_abs"

# Robot EEF30
# NORM_PRESET="robot2_abs"
# NORM_PRESET="robot2_rel_shared"
# NORM_PRESET="robot2_rel_per_step"
# NORM_PRESET="robot2_rel_hybrid"
# 上述任一 Robot preset 末尾可添加 "_single"，例如：
# NORM_PRESET="robot2_rel_shared_single"

# Robot EEF20-gripper
# NORM_PRESET="robot2_gripper_abs"
NORM_PRESET="${NORM_PRESET:-robot2_gripper_rel_shared}"
# NORM_PRESET="robot2_gripper_rel_per_step"
# NORM_PRESET="robot2_gripper_rel_hybrid"

# Human EEF30（同时适用于 mode3/4/5/6）
# NORM_PRESET="human_abs"
# NORM_PRESET="human_abs_progress"  # 为 absolute 进度对齐混训计算 Human 统计
# NORM_PRESET="human_rel_shared"
# NORM_PRESET="human_rel_shared_progress"
# NORM_PRESET="human_rel_per_step"
# NORM_PRESET="human_rel_per_step_progress"
# NORM_PRESET="human_rel_hybrid"
# NORM_PRESET="human_rel_hybrid_progress"

# Human EEF20-gripper（同时适用于 mode3/4/5/6）
# NORM_PRESET="human_gripper_abs"
# NORM_PRESET="human_gripper_abs_progress"
# NORM_PRESET="human_gripper_rel_shared"
# NORM_PRESET="human_gripper_rel_shared_progress"
# NORM_PRESET="human_gripper_rel_per_step"
# NORM_PRESET="human_gripper_rel_per_step_progress"
# NORM_PRESET="human_gripper_rel_hybrid"
# NORM_PRESET="human_gripper_rel_hybrid_progress"

# Human EEF-only 监督域（源数据可为 EEF18 或 EEF20-gripper；20D 的最后 2D 会被忽略）
# NORM_PRESET="human_eef_only_abs"
# NORM_PRESET="human_eef_only_abs_progress"
# NORM_PRESET="human_eef_only_rel_shared"
# NORM_PRESET="human_eef_only_rel_shared_progress"
# NORM_PRESET="human_eef_only_rel_per_step"
# NORM_PRESET="human_eef_only_rel_per_step_progress"
# NORM_PRESET="human_eef_only_rel_hybrid"  # 无尾部时等价于 EEF18 per-step
# NORM_PRESET="human_eef_only_rel_hybrid_progress"
# 使用 gripper Robot 作为进度参考时：
# NORM_PRESET="human_eef_only_robot_gripper_abs_progress"
# NORM_PRESET="human_eef_only_robot_gripper_rel_shared_progress"
# NORM_PRESET="human_eef_only_robot_gripper_rel_per_step_progress"
# NORM_PRESET="human_eef_only_robot_gripper_rel_hybrid_progress"

# 数据集路径：归一化结果会写入所选数据集的 meta/openpi_assets。
ROBOT_JOINT_DATASET="${ROBOT_JOINT_DATASET:-/home/zh/w_ego_collect/IL/openpi/dataset/G1_Brainco_Abc_fold_clothes_7_7/G1_Brainco_Abc_fold_clothes_7_7}"
ROBOT_GRIPPER_JOINT_DATASET="${ROBOT_GRIPPER_JOINT_DATASET:-/home/zh/w_ego_collect/IL/openpi/dataset/clothers_3eps}"
ROBOT_EEF_DATASET="${ROBOT_EEF_DATASET:-/home/zh/w_ego_collect/IL/openpi/dataset/G1_Brainco_Abc_fold_clothes_7_7/G1_Brainco_Abc_fold_clothes_7_7_curated_eef30_columns_grouped}"
HUMAN_DATASET="${HUMAN_DATASET:-/home/zh/w_ego_collect/IL/openpi/dataset/pick_bottle_put_in_box_have_state}"
ROBOT_GRIPPER_EEF_DATASET="${ROBOT_GRIPPER_EEF_DATASET:-/home/zh/w_ego_collect/IL/openpi/dataset/clothers_3eps_eef20_columns_grouped}"
HUMAN_GRIPPER_DATASET="${HUMAN_GRIPPER_DATASET:-/home/zh/w_ego_collect/IL/openpi/dataset/pick_bottle_put_in_box_have_state_gripper_eef20}"
# 默认复用带二指爪的 Human 数据；EEF-only preset 会在归一化前丢弃 [18:20]。
HUMAN_EEF_ONLY_DATASET="${HUMAN_EEF_ONLY_DATASET:-${HUMAN_GRIPPER_DATASET}}"
# Ego 源模式与 OpenPI Human mode3/4/5/6 无关。当前旧默认数据是 Mode1；
# 新 Mode2 必须带 meta/action_semantics.json，旧相对-action Mode2 会被拒绝。
HUMAN_DATASET_MODE="${HUMAN_DATASET_MODE:-mode1}"
# Mode-aware canonicalization is shared by Human-only and mixed training:
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

# 正式统计参数，通常不用改。
RESERVOIR_SIZE=8192
# 所有 preset 都不划验证集，统计使用全部 episode。
ROBOT_VALIDATION_EPISODES=0
HUMAN_VALIDATION_EPISODES=0

# =============================================================================
# 下面无需修改。
# =============================================================================

if [[ $# -ne 0 ]]; then
  echo "本脚本不需要命令行参数；请在文件顶部取消注释一个 NORM_PRESET。" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "NORM_PRESET=${NORM_PRESET} HUMAN_DATASET_MODE=${HUMAN_DATASET_MODE} HUMAN_INPUT_FRAME=${HUMAN_INPUT_FRAME}"
exec "${PYTHON_BIN}" scripts/prepare_unitree_norm_stats.py \
  --norm-preset "${NORM_PRESET}" \
  --human-dataset-mode "${HUMAN_DATASET_MODE}" \
  --human-input-frame "${HUMAN_INPUT_FRAME}" \
  --robot-joint-dataset "${ROBOT_JOINT_DATASET}" \
  --robot-gripper-joint-dataset "${ROBOT_GRIPPER_JOINT_DATASET}" \
  --robot-eef-dataset "${ROBOT_EEF_DATASET}" \
  --human-dataset "${HUMAN_DATASET}" \
  --human-eef-only-dataset "${HUMAN_EEF_ONLY_DATASET}" \
  --robot-gripper-eef-dataset "${ROBOT_GRIPPER_EEF_DATASET}" \
  --human-gripper-dataset "${HUMAN_GRIPPER_DATASET}" \
  --robot-task-contains "${ROBOT_TASK_CONTAINS}" \
  --robot-validation-episodes "${ROBOT_VALIDATION_EPISODES}" \
  --human-validation-episodes "${HUMAN_VALIDATION_EPISODES}" \
  --reservoir-size "${RESERVOIR_SIZE}"
