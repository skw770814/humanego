"""每段采集在 pipeline 里的输入/产物路径。

`PipePaths` 也按鸭子类型充当 `xrrel.step3._humanego_mask_and_lama` 的 `paths` 实参 ——
那个函数只用到 `paths.mp4` 与 `paths.seg_mask(i)` 两个成员, 所以这里提供同名的两个,
不必复用 `xrrel.RelPaths` (后者的 `seg_dir` 是 `test/pipeline/out/seg_<stem>`, 与本 pipeline
自己的产物目录不是一回事)。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import DATA, INSTANCE_ID, INSTANCE_IDS, OUT, WORK


@dataclass(frozen=True)
class PipePaths:
    """一段采集的输入与四个 step 的产物目录。"""

    stem: str
    outdir: Path

    @classmethod
    def for_stem(cls, stem: str, outdir: str | Path = OUT) -> "PipePaths":
        return cls(stem=stem, outdir=Path(outdir) / stem)

    # ---- 输入 (test/ 下的采集) ----
    @property
    def mp4(self) -> Path:
        return DATA / f"CameraRecord_{self.stem}.mp4"

    @property
    def tracking(self) -> Path:
        return DATA / f"trackingData_{self.stem}.txt"

    # ---- step1: 对齐 + 指尖中点 ----
    @property
    def step1_dir(self) -> Path:
        return self.outdir / "step1"

    @property
    def reel_npz(self) -> Path:
        return self.step1_dir / "reel.npz"

    @property
    def reel_json(self) -> Path:
        return self.step1_dir / "reel.json"

    # ---- step2: 物体标定 + state/action ----
    @property
    def step2_dir(self) -> Path:
        return self.outdir / "step2"

    @property
    def seg_dir(self) -> Path:
        """step2 自己那份 SAM2 mask。

        目录名 `seg_<stem>` 与 `xrseg.common.SegPaths.for_stem(stem, step2_dir)` 完全
        一致, 所以驱动 `tools/seg_object.py` 的阶段时不用做任何路径映射; 同时它也是
        `_humanego_mask_and_lama` 用来挡掉物体区域的 mask 来源。
        """
        return self.step2_dir / f"seg_{self.stem}"

    def seg_mask(self, frame: int, instance_id: str = INSTANCE_ID) -> Path:
        return self.seg_dir / "masks" / str(instance_id) / f"{frame:05d}.png"

    def seg_masks(self, frame: int) -> list[Path]:
        """这一帧**所有**物体的 mask 路径 (按 `masks/` 下实际存在的目录)。

        `xrrel.step3._humanego_mask_and_lama` 靠它把每个物体从手部 mask 里抠掉 ——
        多物体时只抠 obj1 会让其余物体被当成手一起修掉。
        """
        if not self.seg_dir.is_dir():
            return []
        names = sorted(p.name for p in (self.seg_dir / "masks").glob("*") if p.is_dir())
        known = [i for i in INSTANCE_IDS if i in names]
        return [self.seg_mask(frame, i) for i in known + [n for n in names if n not in INSTANCE_IDS]]

    @property
    def seg_json(self) -> Path:
        return self.seg_dir / "seg.json"

    @property
    def objectpose_dir(self) -> Path:
        return self.step2_dir / "pose"

    @property
    def relation_dir(self) -> Path:
        return self.step2_dir / "relation"

    @property
    def state_action_npz(self) -> Path:
        return self.step2_dir / "state_action.npz"

    @property
    def state_action_json(self) -> Path:
        return self.step2_dir / "state_action.json"

    # ---- step3: mask + 修复 + 夹爪 ----
    @property
    def step3_dir(self) -> Path:
        return self.outdir / "step3"

    @property
    def observation_dir(self) -> Path:
        return self.step3_dir / "observation"

    def observation(self, frame: int) -> Path:
        return self.observation_dir / f"{frame:05d}.png"

    @property
    def observation_json(self) -> Path:
        return self.step3_dir / "observation.json"

    @property
    def work(self) -> Path:
        return WORK

    def ensure(self, *steps: str) -> None:
        for step in steps:
            (self.outdir / step).mkdir(parents=True, exist_ok=True)
