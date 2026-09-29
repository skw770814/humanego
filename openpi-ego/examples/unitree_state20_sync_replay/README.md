# Episode 20 state synchronous replay

本目录只读取：

```text
dataset/pick_bottle_put_in_box_have_state/data/chunk-000/episode_000020.parquet
```

使用字段是 `observation.state`，不使用 Parquet 的 `action`。30 维 state 排列为：

```text
左 EEF xyz + rotation前两列    9
右 EEF xyz + rotation前两列    9
左 BrainCo                     6
右 BrainCo                     6
```

脚本复用原来的 `EEFPolicyAdapter`、`SynchronousStrategy`、`G1Kinematics` 和
`UnitreeRobotInterface`，不修改 `examples/unitree_inference`。

Parquet 将旋转矩阵前两列保存为 `[column0_xyz, column1_xyz]`。现有 IK 模块的接口使用同样的前两列，
但内存排列是 `3x2.reshape(6)`；脚本进入 IK 前只调整排列，不改变旋转含义。

默认 chunk 大小为 50，因此 239 帧分成：

```text
[0:50] [50:100] [100:150] [150:200] [200:239]
```

每个 chunk 会一次性完成 IK 和安全校验，然后通过机器人接口逐帧以 30 Hz 下发，这与同步推理的
“等待一个 chunk 推理完成，再执行该 chunk”一致。

先进行完整离线 IK 模拟，不连接机器人：

```bash
examples/unitree_state20_sync_replay/run.sh
```

真机执行：

```bash
examples/unitree_state20_sync_replay/run.sh --execute
```

指定 chunk、Unitree 网卡和首帧过渡时间：

```bash
examples/unitree_state20_sync_replay/run.sh --execute \
  --chunk-size 50 \
  --network-interface eth0 \
  --transition-seconds 5
```

真机连接后先读取当前 26 维状态，用现有 IK 解算数据首帧，并平滑过渡到首帧目标。完成两次人工确认后，
才开始同步 chunk 回放。Ctrl-C 可中止。
