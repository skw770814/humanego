# 新海右臂 XRPipe 推理部署

推理相关文件均在 `openpi-ego/` 内。训练代码、训练数据及归一化统计保持原样；新的机器人读写使用原有新海RPC接口的副本。

## 需要复制什么

只复制整个 `openpi-ego/` 文件夹即可带走推理源码、DINO/SAM2权重、原RPC接口副本、pipeline几何与可视化源码，以及SAM2 1.1.0源码。进入复制后的 `openpi-ego/` 目录运行下文命令。训练得到的checkpoint尚不在本文件夹内；复制前将完整checkpoint（模型权重、`assets/runtime_manifest.json`、`assets/<asset_id>/runtime_manifest.json`和`norm_stats.json`）放进`openpi-ego/checkpoints/`，这样它也随文件夹一起迁移。外参默认写入`openpi-ego/extrinsics.json`。

新增入口位于 `scripts/`，共享代码位于 `src/openpi/xrpipe_xinhai/`，配置、权重及依赖快照分别位于 `configs/xrpipe_xinhai/`、`models/xrpipe_xinhai/`和`third_party/xrpipe_source/`。`scripts/robot_client_xinhai_ws.py`为原有接口的副本，只调用右臂TCP/夹爪方法。

`models/xrpipe_xinhai/` 中已经从 test 的缓存**解引用复制**了 DINO tiny、配套分词器/处理器及 SAM2 tiny，共约808MB；没有下载新模型。默认离线加载。来源、快照ID及SHA256在 `SOURCES.json`。不要换成SAM2.1 large配置搭配tiny权重。

## 环境

从复制后的 `openpi-ego/` 目录运行，下例用`PYTHONPATH=src`显式选择本文件夹的openpi代码。服务端使用原有openpi训练/推理环境；机器人客户端使用能同时导入原有RPC接口与pipeline感知模块的环境。建议分机器或分GPU运行策略与感知。

客户端环境需安装第三方Python包（它们不属于需要额外复制的源码目录）：Python 3.11+、numpy、scipy、Pillow、OpenCV、matplotlib、PyYAML、h5py、pyarrow、tyro、zerorpc、websockets（含sync客户端）、openpi-client，以及torch/torchvision/transformers/Hydra环境。SAM2源码已随 `third_party/xrpipe_source/` 复制，运行时直接从该目录导入。`xrrel`现有导入链会用到h5py/pyarrow，尽管这里不读取训练集。

本地实权重预热通过的版本：torch 2.5.1+cu121、torchvision 0.20.1+cu121、transformers 4.57.6、numpy 2.2.6、scipy 1.17.1、OpenCV 5.0.0、hydra-core 1.3.7。打包的SAM2来自原 `test/pipeline/.venv` 的1.1.0版本；它的video state含`output_dict_per_obj`、`frames_tracked_per_obj`。不要复制整个虚拟环境到不同操作系统；在远端使用同版本源码/依赖。依赖环境仍需安装所列Python包；其他SAM2状态布局会在预热时明确拒绝，而非静默用单帧分割代替跟踪。

`openpi-client` 可从本文件夹的 `packages/openpi-client` 安装；服务端用本文件夹的openpi，不用其他版本同名包。权重加载不需要网络。服务器只应部署在可信机器人局域网，WebSocket未添加鉴权。

## 局部轴配置与回退

客户端和单帧测试现在默认读取 `configs/xrpipe_xinhai/right_gripper_train_aligned.json`。
原 `right_gripper.json` 和原几何转换函数保留。新配置是待实机验证的轴对齐候选：
`T_E_M_new = T_E_M_original @ Q`，`Q=diag(-1,1,-1,1)`，保持中点位置和Y抓取轴，
同时反向X、Z，并交换五关键点中的拇指/食指角色。训练代码及训练S转换没有改变。
同一配置用于state、动作还原、反馈锁存与可视化；局部平移分量也随轴表达变化。

在任一客户端或单帧测试命令中显式选择新配置：

```bash
--tcp-geometry configs/xrpipe_xinhai/right_gripper_train_aligned.json
```

回退原轴配置：

```bash
--tcp-geometry configs/xrpipe_xinhai/right_gripper.json
```

切换后重新启动客户端或测试脚本，使跟踪和夹持锁存重新初始化；策略服务端无需切换。
每轮 `snapshot.npz` 的metadata及可视化 `report.json` 保存实际使用的完整geometry配置。
回放旧配置采集的快照时必须显式传入原 `right_gripper.json`；
不同几何配置会触发 `Replay calibration mismatch`，不会自动重解释旧数据。

先用单帧检查验证中点、Y轴和实体两指对应，再用新state重新请求策略做dry-run。
旧state得到的actions不能当作新配置下模型的预测结果。候选轴映射仍需实机几何验证。

## 先做单帧检查（不需要策略权重）

```bash
PYTHONPATH=src python scripts/test_xrpipe_xinhai_observation.py \
  --live \
  --contract configs/xrpipe_xinhai/observation_contract.json \
  --robot-ip 172.16.0.30 --robot-port 4242 \
  --object-prompts 'green type' 'black plate' \
  --output-dir outputs/observation_check
```

默认`extrinsics.json`尚未填写时，单帧入口仅检查RGB、深度、分割和相机系物体位姿，`complete_state=false`；不伪造中点或19D state。默认提示词按当前训练物体顺序obj1/obj2读取，检测措辞可以改变，顺序不可变。诊断契约对应当前 `ego_all`，digest=`89f5be459788`；正式推理必须换成实际checkpoint的契约。

默认读取 `openpi-ego/extrinsics.json`。该文件已提供结构，`T_B_C`目前为`null`；拿到外参后直接填入4×4矩阵即可。也可以保留默认文件、改用：

```bash
  --extrinsics /path/to/extrinsics.json
```

外参约定 `p_B = T_B_C @ p_C`，C是D405 **RGB光学系**，B是实际TCP收发控制参考系，平移单位米。未填写的`T_B_C`会阻止动作推理入口执行。默认模板参考名`torso_link4`必须与真实控制链路核对；不根据RPC消息header自行转换。相机已拆下固定放置，绝不使用URDF的腕部相机安装外参。相机、底盘、躯干保持标定时状态。

输出每个时间戳一个目录：

- `rgb.png`、`depth_valid.png`、`segmentation.png`、`geometry.png`、`camera_scene_3d.png`。
- `snapshot.npz`：解码后的无损RGB、米制深度、TCP、反馈开度、采样时间/标定信息。
- `derived.npz`：掩码、点云、物体位姿、中点位姿、raw state和训练归一化前state（后两项需要外参）。
- `report.json`：深度有效比例、每物体点数与位姿质量、合法矩阵检查、投影范围、阶段耗时、反馈及配置。

轴色与原 `test/pipeline --vis` 一致：X红、Y绿、Z蓝，直接复用它的字体、箭头和文字绘制；投影使用D405内参与畸变，复用打包的pipeline绘图函数。检查中点是否落在实际两指末端中心、物体掩码是否正确、轴朝向是否合理。至少换几个右臂姿态重复验证；数值自洽不代表外参或实际接触点标定正确。

回放同一处理链路：

```bash
PYTHONPATH=src python scripts/test_xrpipe_xinhai_observation.py \
  --replay outputs/observation_check/TIMESTAMP/snapshot.npz \
  --contract configs/xrpipe_xinhai/observation_contract.json \
  --extrinsics /path/to/extrinsics.json \
  --output-dir outputs/replay_check
```

若原快照没有外参，回放也不传外参。回放要求内参、几何和外参与采集时相同，防止无意混用标定。`--replay`也接受含多个快照的目录，按时间戳目录名顺序处理，用于跨帧跟踪和锁存检查。单张快照只能验证首次PCA几何；模型浮点差异可能导致分割有微小变化。

## 策略服务与dry-run

服务端：

```bash
PYTHONPATH=src python scripts/serve_policy_xrpipe_xinhai.py \
  --checkpoint-dir checkpoints/YOUR_RUN/10000 \
  --port 5000 \
  --default-prompt 'pick up the small type and put it into the black plate'
```

启动会校验checkpoint契约和stats，加载并预热策略；服务端不再做state坐标转换，输出已经反归一化。训练manifest里旧的`deployment_supported=false`不会被改写；新部署协议用独立`inference_adapter`标识并校验。

客户端（默认dry-run，仅一轮）：

```bash
PYTHONPATH=src python scripts/robot_client_xrpipe_xinhai_ws.py \
  --contract checkpoints/YOUR_RUN/10000/assets/runtime_manifest.json \
  --server ws://POLICY_MACHINE:5000 \
  --robot-ip 172.16.0.30 --robot-port 4242 \
  --task 'pick up the small type and put it into the black plate' \
  --object-prompts 'green type' 'black plate' \
  --chunk-exec-steps 20 --fps 30 \
  --dry-run --max-cycles 1
```

确认图像几何和dry-run目标之后，才使用 `--no-dry-run --geometry-verified`。推荐首次真机只执行`--chunk-exec-steps 1 --max-cycles 1`，人工准备好急停。`--max-cycles 0`表示持续运行。

TCP收发完全复用旧接口；只调用`move_ee_right()`及`open_gripper_right()`，不调用左臂、IK、关节控制。保持你已有的**右臂**RelaxedIK链路运行；本程序不会启动或改写ROS节点。

每轮会保存快照、`actions.npz`（反归一化动作和还原后的目标TCP/夹爪）、`policy.json`和`execution.json`（含执行期实际TCP/夹爪反馈；序列回放会读取这些反馈恢复帧间锁存事件）。每轮采集前至少用0.2秒刷新实际夹爪反馈。失败后不自动重试、不继续发送余下目标。停止发送不是硬件急停，也不能撤销已经发送的目标。

## 与训练对齐的关键点

1. 模型输入19维：两个 `T_M_O` 的 `[xyz,R[:,0],R[:,1]]` 加实际反馈二值爪。RGB唯一有效图像为camera0/base_0_rgb，两腕图像mask=false，深度不传给策略。预处理与归一化由原openpi模型适配完成。
2. 当前代码的训练state做 `S @ T_M_O @ S`，S=diag(1,1,-1,1)。新增适配层严格只做一次；动作标签来自同一帧reference的相对SE(3)，不是相邻预测动作相减。50步全相对当前观测anchor，不用上一帧、也不逐步累加。
3. `right_gripper.json` 的 `T_E_M` 特意定义为**源五关键点轴的物理类比**：X为finger2/thumb到finger1/index（近似+E.y），Y沿指向末端（近似-E.z），Z叉乘（近似-E.x）；配置保留了URDF指根微小X偏置，直接用现有五关键点函数算轴。原点由原装URDF指节z=-0.03689与指端mesh最远端z=-0.04146503推得名义中点z=-0.07835503米；并非现场接触垫标定。对称开合时中点不随开度变化，需实机投影验证。
4. 因本适配层明确用了上述**源轴**，模型canonical增量需先还原为 `delta_source=S @ delta_model @ S`，随后才是 `T_B_E_target=T_B_E_anchor @ T_E_M @ delta_source @ inv(T_E_M)`。这是训练基变换的逆，不是第二次世界坐标镜像。不能省略它，也不能自行再对xyz或四元数改符号。数值测试直接执行未修改的训练变换定义并验证往返。
5. 改FPS按物理时间线性插值平移、SLERP旋转、保持式采样爪；起点为单位增量/反馈爪。要求steps≤50且steps/fps≤50/30，30Hz精确取前缀，不插值。
6. 模型爪≥0.5为闭合→发送0；否则张开→发送80。state绝不使用目标爪值。在线反馈开度固定端点0/80、阈值.70/.60、至少5次反馈且持续5/30秒确认、12/30秒驻留；与训练整段分位数标定、非因果中值/回填有不可避免差异，已明确保留为因果实现。
7. PCA、参考点云ICP、恢复及平滑调用pipeline现有函数；门控失败直接停止本轮，不冒用旧位姿发送动作。锁存规则按物体顺序、距中点<5cm、同刻最多一个；闭合后手推，张开恢复视觉。执行期读取实际TCP/爪维护锁存，锁存物体允许遮挡。新增锁存仅使用最近0.5秒的物体估计，避免绑住陈旧物体。
8. 感知只在每轮新RGBD推进一次，因此在线采样间隔通常不是训练的30Hz连续视觉帧；ICP位移门按时间差折算训练帧数（最多5倍），平滑为因果版本。长时间遮挡和快速运动仍须用现场序列验证。

## 延迟与安全门

DINO/SAM2在启动时一次加载、预热并清空测试跟踪状态；控制循环不加载权重、不启动子进程。SAM2持久化逐帧传播并保留有界历史；首次/丢跟由DINO按固定槽位重检。不把每轮图像重复发送给策略。

新观测必须时间戳递增且晚于本轮采集开始，RGBD必须精确同步；重复帧不会更新SAM2。深度按16UC1处理字节序、行stride、毫米→米、无效0及D405畸变。机器人与客户端时钟须同步（NTP/ROS时间），默认5秒观测年龄限制覆盖感知、推理及执行。TCP没有硬件时间戳，当前只能用采集前后读数门控（默认5mm/2°）；**无法替代硬件级RGBD/TCP同步**，采集时手臂应静止。

默认预测相对anchor位移≤15cm、旋转≤60°，单次目标跳变≤3cm/10°；参数可调，但首次实机应收紧。执行超一周期立即取消余下chunk，不追赶补发。RGBD获取频率无需等于执行FPS；但每步反馈/控制RPC能否达到30Hz，必须现场测量。旧RPC接口本身的阻塞时延不由本适配层消除，已经阻塞的调用无法即时撤销。

## 自动测试

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_xrpipe_xinhai.py' -v
```

无需机器人、策略权重或DINO/SAM2。覆盖训练数学定义对齐、五关键点轴、固定anchor、FPS重采样、二值爪、深度步幅/字节序、采集与回放state一致、只读观测、锁存优先级与解除、超时取消以及服务端不重复变换。

可选实权重离线多帧测试：

```bash
PYTHONPATH=src python tests/smoke_xrpipe_perception.py \
  tests/fixtures/xrpipe_observation/00050.jpg \
  tests/fixtures/xrpipe_observation/00051.jpg \
  tests/fixtures/xrpipe_observation/00052.jpg
```

迁入`openpi-ego/`后已通过12项离线测试，以及真实tiny权重的加载/预热和3帧连续检测跟踪（第一帧DINO，后两帧SAM2持续跟踪，模型对象保持同一实例）。本地SAM2未编译`_C`时会警告跳过小孔后处理，仍可执行，与当前pipeline环境一致。

本地使用打包的DINO/SAM2权重、SAM2源码和三张打包的测试图片完成了连续检测跟踪；没有运行真实机器人、真实外参或策略checkpoint闭环；这些必须按单帧可视化→多姿态→序列回放→dry-run→单步真机的顺序在远端验收。

