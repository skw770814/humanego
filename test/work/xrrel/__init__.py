"""Step3: 右手末端 (两指尖中点系) 相对 step2 SAM2 分割物体的位姿。

链路 (每一段的出处都写在对应模块的文件头里):

    out/seg_<stem>/masks/obj1/*.png          step2 的 2D 分割 (只读输入)
    CameraRecord_<stem>.mp4 的左半 + 右半      双目原始帧 (左半 = eye0)
      -> 双目深度   xrrel/stereo.py      直接调 ego_relation_policy 的 stereo_depth.py
      -> 物体点云   xrrel/lift.py        mask + 深度反投影 (HumanEgo DepthLifter.py:112)
      -> 物体位姿   xrrel/objectpose.py  帧50 PCA 定向 + 逐帧 ICP 鲁棒刚体拟合 + 门控/平滑/锁存
      -> 相对位姿   xrrel/relation.py    T_object_right_midpoint = inv(T_cam0_object) @ T_cam0_midpoint
      -> 可视化     xrrel/render.py      step1 的 5 关键点 + step2 的分割 + 逐帧实时相对位姿

坐标系约定 (整条链路只用一个相机系, 不混):

    "相机0" = **左眼 (eye0)**。物体深度是把校正后的左眼深度映回**原左图像素**得到的
    (stereo_depth._restore_original_left_depth), 两眼已平行 ⇒ R_left ≈ I、depth_scale ≈ 1,
    所以深度的 Z 就是沿**左眼光轴**的量, 点云天然活在左眼相机系 —— 与
    `xrhand.camera.Projector` 在 eye=0 下输出的 `p_cam` 是同一个系。
    因此手的中点也必须用 **half=0** 那条外参链送进相机0 (见 xrrel/relation.py)。

产物落在 out/rel_<stem>/ (depth/ pose/ relation/ render/), out/seg_<stem>/ 只读。
必须用 work/.venv 的解释器跑 (cv2 只在里面): tools/rel_run.sh。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PKG = Path(__file__).resolve().parent  # test/work/xrrel
WORK = PKG.parent  # test/work
DATA = WORK.parent  # test/  (CameraRecord_*.mp4 / trackingData_*.txt)
OUT = WORK / "out"


def _find_ego_repo() -> Path:
    """定位 ego_relation_policy —— 它的 stereo_depth.py 是我们直接调用的深度骨架。

    用「找得到 src/ego_relation/s2_object_relations/stereo_depth.py」当判据, 而不是
    硬编码某台机器的路径。
    """
    marker = Path("src") / "ego_relation" / "s2_object_relations" / "stereo_depth.py"
    for candidate in (DATA / "ego_relation_policy", DATA.parent / "ego_relation_policy"):
        if (candidate / marker).is_file():
            return candidate
    raise FileNotFoundError(
        "找不到 ego_relation_policy (需要 <repo>/src/ego_relation/s2_object_relations/"
        f"stereo_depth.py), 找过: {DATA / 'ego_relation_policy'}, "
        f"{DATA.parent / 'ego_relation_policy'}"
    )


EGO_REPO = _find_ego_repo()
EGO_SRC = EGO_REPO / "src"
EGO_CONFIG = EGO_REPO / "configs" / "default.yaml"

# 双目视频布局 (见 xrhand/camera.py): 2160x810 = 左半 eye0 + 右半 eye1
EYE_W, EYE_H = 1080, 810
FULL_W, FULL_H = 2160, 810

INSTANCE_ID = "obj1"
# 物体实例 id 的规范顺序 (obj1..objN), 与 `xrseg.common.INSTANCE_IDS` 同一个表。
# 只有一只手 (右手), 所以 state 里物体块的顺序就是这里的顺序。
INSTANCE_IDS = ("obj1", "obj2", "obj3")

# 单眼内参: 用 xrhand.camera 的标称值 (header 里的 fx=1373.668 是因为取内参时
# width 传了双眼总宽 2160 而翻倍, 推导见 xrhand/camera.py:1-24)。xrrel/adapter.py
# 里对着 header 做断言, 防止再被这个两倍坑到。
EYE_F = 686.868
EYE_CX = 539.5
EYE_CY = 404.5


@dataclass(frozen=True)
class RelPaths:
    """一条采集的 step3 产物目录。"""

    stem: str
    outdir: Path
    # step2 的 2D 分割来源。None = work 自己的 `out/seg_<stem>`（原有行为）；
    # pipeline 会传自己的 seg 目录，避免依赖/污染 work 的产物。
    segdir: Path | None = None

    @classmethod
    def for_stem(cls, stem: str, outdir: str | Path = OUT,
                 segdir: str | Path | None = None) -> "RelPaths":
        return cls(stem=stem, outdir=Path(outdir) / f"rel_{stem}",
                   segdir=Path(segdir) if segdir is not None else None)

    # ---- 输入 ----
    @property
    def mp4(self) -> Path:
        return DATA / f"CameraRecord_{self.stem}.mp4"

    @property
    def tracking(self) -> Path:
        return DATA / f"trackingData_{self.stem}.txt"

    @property
    def seg_dir(self) -> Path:
        return self.segdir if self.segdir is not None else OUT / f"seg_{self.stem}"

    @property
    def seg_report(self) -> Path:
        return self.seg_dir / "report.json"

    def seg_category(self, instance_id: str = INSTANCE_ID) -> str:
        """step2 用的文本提示 (report.json 的 `stage_image.objects[<id>].prompt`)。

        只当**标签**用 (写进 npz 的 categories / 报告), 不参与任何计算 —— 免得硬编码
        "earphone case" 而这种采集实际用的提示是 "small white earbud case"。
        多物体时 `stage_image.objects` 逐物体一份; 老产物 (或只有 obj1) 退回顶层
        `stage_image.prompt` —— 那个本来就是 obj1 的提示。
        """
        import json

        if self.seg_report.is_file():
            payload = json.loads(self.seg_report.read_text(encoding="utf-8"))
            stage = payload.get("stage_image", {})
            entry = (stage.get("objects") or {}).get(str(instance_id)) or {}
            prompt = entry.get("prompt")
            if not prompt and str(instance_id) == INSTANCE_ID:
                prompt = stage.get("prompt")
            if prompt:
                return str(prompt)
        return str(instance_id)

    def seg_start_frame(self) -> int | None:
        """SAM2 的起始帧 = `report.json` 的 `stage_video.initial_frame` (111342 是 50)。

        拿不到就退回 `metrics.npz` 里第一个 `areas > 0` 的帧; 都没有则 None
        (调用方决定是报错还是从 0 开始)。
        """
        import json

        import numpy as np

        if self.seg_report.is_file():
            payload = json.loads(self.seg_report.read_text(encoding="utf-8"))
            value = payload.get("stage_video", {}).get("initial_frame")
            if value is not None:
                return int(value)
        if self.seg_metrics.is_file():
            with np.load(self.seg_metrics) as archive:
                areas = archive["areas"]
            # 多物体时 areas 是 (帧, 物体); SAM2 在提示帧之前对所有物体都写全零 mask,
            # 所以"第一个有面积的帧"应当对**任一**物体取 —— 逐物体取第 0 个会漏掉
            # "obj1 在提示帧没框到、obj2 框到了"这种情形。
            areas = areas.any(axis=1) if areas.ndim > 1 else areas
            nonzero = np.nonzero(areas > 0)[0]
            if nonzero.size:
                return int(nonzero[0])
        return None

    @property
    def seg_metrics(self) -> Path:
        return self.seg_dir / "metrics.npz"

    def seg_mask(self, frame: int, instance_id: str = INSTANCE_ID) -> Path:
        return self.seg_dir / "masks" / str(instance_id) / f"{frame:05d}.png"

    def seg_masks(self, frame: int) -> list[Path]:
        """这一帧**所有**物体的 mask 路径 (按 `masks/` 下实际存在的目录)。

        给 `xrrel/step3.py` 用: 它要把物体区域从手部 mask 里抠掉, 多物体时只抠 obj1
        会让第二个物体被当成手一起修掉。
        """
        if not self.seg_dir.is_dir():
            return []
        names = sorted(p.name for p in (self.seg_dir / "masks").glob("*") if p.is_dir())
        known = [i for i in INSTANCE_IDS if i in names]
        return [self.seg_mask(frame, i) for i in known + [n for n in names if n not in INSTANCE_IDS]]

    # ---- 产物 ----
    @property
    def depth_dir(self) -> Path:
        return self.outdir / "depth"

    def depth(self, frame: int) -> Path:
        return self.depth_dir / f"{frame:05d}.png"

    @property
    def pose_dir(self) -> Path:
        return self.outdir / "pose"

    @property
    def clouds_npz(self) -> Path:
        return self.pose_dir / "clouds.npz"

    def clouds_npz_for(self, instance_id: str = INSTANCE_ID) -> Path:
        """多物体时逐物体一份点云; obj1 用原来的文件名 (`clouds.npz`), 其余带后缀。

        文件名不改是为了「N=1 时产物逐字节相同」: `clouds_npz` 与 `clouds_npz_for('obj1')`
        指向同一个文件。
        """
        if str(instance_id) == INSTANCE_ID:
            return self.clouds_npz
        return self.pose_dir / f"clouds_{instance_id}.npz"

    @property
    def object_npz(self) -> Path:
        return self.pose_dir / "T_camera0_object.npz"

    @property
    def relation_dir(self) -> Path:
        return self.outdir / "relation"

    @property
    def relation_npz(self) -> Path:
        return self.relation_dir / f"rel_{self.stem}.npz"

    @property
    def render_dir(self) -> Path:
        return self.outdir / "render"

    @property
    def video(self) -> Path:
        return self.render_dir / f"rel_{self.stem}.mp4"

    def still(self, frame: int) -> Path:
        return self.render_dir / f"still_f{frame:04d}.png"

    @property
    def qa_json(self) -> Path:
        return self.outdir / "stereo_qa.json"

    @property
    def report_json(self) -> Path:
        return self.outdir / "report.json"

    def ensure(self) -> None:
        for d in (self.outdir, self.depth_dir, self.pose_dir, self.relation_dir, self.render_dir):
            d.mkdir(parents=True, exist_ok=True)
