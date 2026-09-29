"""生成 work/out/calib.json。

`extra_t` (相机原点相对 head 原点的平移) 现在**三个分量都有值**, 来源是
`/home/skw/git/ego` 那套外参参考系的换算, 不是拟合。

依据链 (逐条可复核):
  1. 旋转不是误差来源: 示意图 (874ea505701a55344fe8b7ca20137dff.jpg) 读出绕 head X 转
     180°, 与 header 外参独立算出的完全一致。见 out/head_cam_origin.md §2。
  2. 外参平移的**参考原点** ≠ head 原点: 外参给出的相机对中点在 head X = −0.05068 m,
     而基线仅 0.06407 m、两相机相对旋转 0.000000° (一个左右对称的立体对, 中点不该偏出
     5 cm)。见 out/head_cam_origin.md §3。横向那一半已由像素证据独立验证: 两个不同深度
     (0.379 / 0.517 m) 靠同一个平移落到手上, 位移按 1/Z 变化 => 三维平移, 不是像素偏移。见 §4。
  3. **另外两个分量来自 ego 参考系**: /home/skw/git/ego 的 9 个 HDF5 (跨三个采集批次)
     外参逐位相同 => 它记的也是硬件常数。与我们 header 对比:
         基线   0.064068 vs 0.064247 m  (差 0.28%)
         相对旋转 0.000000°             (两套一致)
         两套参考系差一个正交、det=+1 的基变换 B ≈ rotz(+90°);
         用 B 推原点位移时**左右相机自洽到 0.22 mm**, 用 Bᵀ 则是 128.31 mm
     => 两套 header 是**同一个刚体对的两种记法**, ego 参考系里的相机位置就是我们的。
     设备同为 PICO 4 Ultra, 相机相对 head 的安装是硬件常数, 不是个体差异。
  4. 眼别归属已用**视差符号**独立判过 (左半幅 = 左侧相机, dx ≈ −60~−66 px, 5 帧 2 采集一致),
     排除了"两半幅需要反向修正"的假设 => 这个平移是**两半幅共用**的。见 §3.5。

    python3 tools/make_calib.py            # 写 work/out/calib.json
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORK)

from xrhand.camera import CameraParams, rotz, RZ_DEVICE_DEG  # noqa: E402
from xrhand.io_tracking import load  # noqa: E402

OUT = os.path.join(WORK, "out")
TXT = os.path.join(os.path.dirname(WORK), "trackingData_20260920_111300.txt")

# head 系 (X 右 / Y 上 / Z 后) 的平移修正量, 来自 /home/skw/git/ego 外参参考系的换算。
# 来源: ego 的外参说相机对中点在它那套参考系里是 (0, −1.8, +69.0) mm (前后取向前为正),
# 而我们的 header 说 (在 head 系) 是 (−50.7, −20.7, −16.6) mm => 差值即下面三个分量。
DX_HEAD = 0.0505     # 横向 (与原值 0.05068 差 0.2 mm, 即 ego 的中点横向 −0.2 mm)
DY_HEAD = 0.0189     # 上下  ← 本次新增
DZ_HEAD = -0.0524    # 前后  ← 本次新增 (−Z = 向前)

# ego 参考系下, 相机对中点在 head 系 (右/上/后, m) 的位置 —— 施加修正后的自检目标。
# 注: "前后 +6.90 cm (向前)" 在 Z 向后的 head 系里就是 Z = −0.0690。
EGO_MID = np.array([-0.0002, -0.0018, -0.0690])


def head_to_extra_t(dx: float, dy: float, dz: float) -> np.ndarray:
    """head 系位移 -> device 系的 extra_t。

    camera.py: p_dev = (p_dev − extra_t) @ extra_Rᵀ => 相机原点落在 p_dev = t + extra_t,
    head 系里就是 M(t + extra_t) = Mt + M·extra_t。要 M·extra_t = (dx,dy,dz)_head,
    而 head->dev 的旋转是 Rz(+90°), 即 Rz(−90°)·(dx,dy,dz) = (dy, −dx, dz)。
    """
    return np.array([dy, -dx, dz], dtype=float)


def cam_origin_head(E: np.ndarray, eye: int) -> np.ndarray:
    """相机原点 in head 系 (右/上/后, m)。"""
    H2D = rotz(-RZ_DEVICE_DEG)          # head->dev (行向量)
    return H2D @ E[eye][:3, 3]          # 列向量形式: (H2D @ t) == D2H.T @ t


def main() -> int:
    header, _ = load(TXT)
    E = header.extrinsics

    before = [(cam_origin_head(E, e)) for e in (0, 1)]
    mid_before = (before[0] + before[1]) / 2.0

    d_head = np.array([DX_HEAD, DY_HEAD, DZ_HEAD])
    extra_t = head_to_extra_t(*d_head)
    p = CameraParams(extra_t=extra_t)

    # 自检: 施加修正后, 相机对中点应落在 ego 参考系给出的位置 (容差 1 mm)
    mid_after = mid_before + d_head
    err = np.abs(mid_after - EGO_MID)
    print(f"相机对中点 head 系  修正前 {np.round(mid_before, 5)}")
    print(f"                    修正后 {np.round(mid_after, 5)}")
    print(f"  ego 参考系目标            {np.round(EGO_MID, 5)}   "
          f"(最大分量残差 {err.max() * 1000:.3f} mm)")
    assert err.max() < 1e-3, "修正后中点和 ego 参考系对不上, 三个分量或符号有错"

    # 自检: 修正后两相机横向必须左右对称, 否则基线方向判错
    after = [b + d_head for b in before]
    print(f"  两相机横向 {after[0][0]:+.4f} / {after[1][0]:+.4f} m   "
          f"基线 {after[1][0] - after[0][0]:+.5f} m")
    assert abs(after[0][0] + after[1][0]) < 1e-3, "修正后相机对横向不对称"

    # 像素效应预告 (画面里骨架该往哪边移)
    Zm = 0.40
    du = -p.eff_f * DX_HEAD / Zm
    dv = p.eff_f * DY_HEAD / Zm
    print(f"预期像素效应 @ Z=0.40 m: Δu = {du:+.1f} px (相机右移 -> 物像左移), "
          f"Δv = {dv:+.1f} px (相机上移 -> 物像下移)")
    assert du < 0 and dv > 0, "像素效应方向不对"

    doc = {
        "generated_by": "tools/make_calib.py",
        "stage": "interim-ModelA-ego-frame-transfer",
        "supersedes": "interim-ModelA-translation-only (只有横向那一个分量)",
        "superseded_by": None,
        "provenance": {
            "why_translation_not_pixel_offset": (
                "同一 +0.0507 m 在两个不同深度 (0.379 m / 0.517 m) 都让骨架落到手上; "
                "像素位移随 1/Z 变化 => 三维平移。二维常数 (Δcx,Δcy) 无法同时拟合两个深度。"
            ),
            "why_this_magnitude": (
                "三个分量来自 /home/skw/git/ego 外参参考系的**换算**, 不是拟合: 它的 9 个 HDF5 "
                "跨三个采集批次外参逐位相同 => 记的是硬件常数; 与我们 header 相比基线差 0.28% "
                "(0.064068 vs 0.064247 m)、两相机相对旋转差 0.000000°, 两套参考系之间差一个 "
                "正交且 det=+1 的基变换 B≈rotz(+90°), 用 B 推原点位移时左右相机自洽到 0.22 mm "
                "(用 Bᵀ 则 128.31 mm) => 两套 header 是同一个刚体对的两种记法。设备同为 "
                "PICO 4 Ultra, 相机相对 head 的安装是硬件常数, 不是个体差异。"
                "横向分量另有像素证据独立印证 (两个深度都靠同一个平移落到手上)。"
            ),
            "forward_component_explains_scale_residual": (
                "Δz = −5.24 cm (相机前移) 把 0.379 m 处的深度降到 ~0.327 m, 投影放大 ~16%, "
                "方向与量级都对得上此前记的『投影手比实际手小约 10%』。"
            ),
            "vertical_component_is_the_untested_one": (
                "Δy = +1.89 cm 没有任何独立症状支持, 也未被独立验证。它的可观察后果是画面内容 "
                "下移约 34 px @0.40 m。上一轮只修横向后骨架在垂直方向看着是对的 —— 若那是对的, "
                "这一分量就是多余的。这是本次唯一可能被目视证伪的分量。"
            ),
            "eye_assignment_verified": (
                "视差符号判据 (带能失败的自检): 5 帧 / 2 组采集 "
                "dx ≈ −60~−66 px 恒为负 => u_L > u_R => 左半幅 = 左侧相机; "
                "两半幅共用同一个方向的修正。"
            ),
            "rotation_already_correct": (
                "示意图读出绕 head X 转 180°, 与外参独立算出的 cam_X=+head_X, "
                "cam_Y=−head_Y, cam_Z=−head_Z 完全一致 => extra_R 保持单位阵。"
            ),
            "not_yet_determined": [
                "标定参数在两段采集上是否一致 (Step 3 §4.5)",
                "『投影手比实际手小约 10%』假定由本次的 Δz = −5.24 cm 解释; 若渲染后仍偏小, "
                "剩余部分要记到 f 上 (两者靠 1/Z 依赖区分: Z 偏差缩放随深度变, f 偏差不变)",
                "『上下 +1.89 cm』无独立症状支持, 施加后画面内容应下移约 34 px @0.40 m —— "
                "若渲染出来垂直方向反而偏了, 撤掉这一分量即可, 另两项不受影响",
                "腕部略过冲",
            ],
        },
        "evidence_files": [
            "out/head_cam_origin.md",
        ],
        "params": p.to_json(),
    }

    os.makedirs(OUT, exist_ok=True)
    dst = os.path.join(OUT, "calib.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    print(f"\n-> {dst}")
    print(f"   extra_R = 单位阵, extra_t = {np.round(p.extra_t, 5)} (device 系)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
