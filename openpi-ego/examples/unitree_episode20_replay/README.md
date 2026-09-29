# Unitree episode 20 replay

本目录独立读取 `dataset/exports/episode_000020_g1_replay_state.npz`，并复用
`examples/unitree_inference/robot_interface.py` 控制 `unitree_g1_brainco`。不修改原推理目录。

26 维动作顺序固定为：

```text
[0:7]   左臂 7 关节
[7:14]  右臂 7 关节
[14:20] 左 BrainCo：thumb_flex, thumb_rot, index, middle, ring, pinky
[20:26] 右 BrainCo：thumb_flex, thumb_rot, index, middle, ring, pinky
```

先执行离线校验，不会连接机器人：

```bash
examples/unitree_episode20_replay/run.sh
```

确认离线校验通过、机器人急停和周围环境准备完毕后，才使用真机回放：

```bash
examples/unitree_episode20_replay/run.sh --execute
```

指定 Unitree 网卡或延长首帧过渡时间：

```bash
examples/unitree_episode20_replay/run.sh --execute \
  --network-interface eth0 \
  --transition-seconds 5
```

真机模式连接后会读取当前 26 维状态，要求输入固定确认语，随后用 smoothstep 平滑移动到首帧；
到达首帧后还需再次按 Enter，才会按原始 30 Hz 回放 239 帧。Ctrl-C 可中止回放。
