# LeRobot 数据筛选器

这是一个完全在本机运行的浏览器 GUI，用来逐回合查看 LeRobot v2.1 数据，并把人工保留的回合导出为一个新的、可直接加载的 LeRobot 数据集。

## 启动

在 openpi 仓库根目录执行：

```bash
.venv/bin/python scripts/lerobot_curator/serve.py \
  dataset/gripper_clothers/clothers_data
```

程序默认打开 <http://127.0.0.1:8765>。也可以查看已经拆出的 300 条数据：

```bash
.venv/bin/python scripts/lerobot_curator/serve.py \
  dataset/gripper_clothers/clothers_data
```

远程服务器上建议关闭自动打开浏览器，再用 SSH 转发端口：

```bash
.venv/bin/python scripts/lerobot_curator/serve.py DATASET_PATH --no-browser
ssh -L 8765:127.0.0.1:8765 USER@SERVER
```

## 操作

- `空格`：播放或暂停四路同步视频。
- `←` / `→`：上一个或下一个回合。
- `K`：保留当前回合。
- `X`：排除当前回合。
- `U`：重置为未处理。
- 倍速菜单支持 `0.25×`、`0.5×`、`1×`、`1.5×`、`2×` 和 `4×`。
- 双击任意相机画面可以全屏。
- 可以按筛选状态过滤列表，也可以直接输入 Episode ID 跳转。
- 备注会在停止输入约半秒后自动保存。

### 在线设置裁剪起点

把视频拖动（或用 `,` / `.` 逐帧微调）到希望作为开头的那一帧，然后：

- `S`：把当前播放位置设为该回合的裁剪起点（导出时删除此前的所有帧）。
- `G`：跳到已设的起点预览。
- `D`：清除起点（等于不裁）。

起点与 keep/reject 相互独立：只有**既 keep 又设了起点**的回合才会被裁剪；keep 但没设起点表示保留整条（删 0 帧）。设了起点的回合会在列表里显示 `✂N` 标记，头部“裁剪起点”也会显示“删除前 N 帧”。

筛选记录默认保存在数据集下的 `.lerobot-curator/selection.json`（含每条的 `start_frame`）。原始 parquet、视频和标准 `meta` 文件不会被修改，重新启动 GUI 后会恢复上次进度。

## 导出数据集

点击右上角“导出已保留数据”，填写一个不存在的输出目录。导出器会：

1. 只收集标记为“保留”的回合，并按原始 Episode ID 排序。
2. 按每条的裁剪起点删除开头帧，并**重编码对齐四路视频**（默认 H.264 多进程并行；可选 AV1 保持原编码但更慢），保证图像、state、action 三者时间一致。
3. 将 episode、task 和全局 frame index 重新编号，重写 parquet 的 `episode_index`、`index`、`frame_index`、`timestamp`、`task_index`。
4. 更新 `episodes.jsonl`、`episodes_stats.jsonl`、`tasks.jsonl`、`info.json` 和 `stats.json`；数值列统计精确重算，图像统计沿用原值。
5. 在 `meta/trim_manifest.json` 保存新旧 Episode ID 对照与每条删除的帧数，并在输出目录旁写出 `<输出名>.trim_spec.csv`，便于用 `scripts/build_trimmed_lerobot_subset.py` 完全复现同一份裁剪。

导出直接复用独立工具 `scripts/build_trimmed_lerobot_subset.py` 的裁剪流水线，因此 GUI 与命令行产出完全一致。因为要重编码，导出耗时取决于回合数与 CPU 核数（例如 50 条×4 路在 32 核上约 2 分钟）。

## 流畅性设计

- 视频直接交给浏览器原生播放器和硬件解码，不经过 Python 逐帧传输。
- 四路画面会自动缩放到浏览器的剩余高度，播放和筛选控件始终保留在同一屏内。
- 服务端实现 HTTP Range，拖动进度条时只读取所需的视频区间。
- 同一回合只加载当前四路视频，Episode 列表只渲染当前位置附近的条目。
- 多路同步以第一路视频为主时钟，仅在漂移超过 120 ms 时校正其他相机。
- 导出运行在独立的 HTTP 工作线程中，不会阻塞前端播放控制。
