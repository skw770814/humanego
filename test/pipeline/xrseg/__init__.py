"""xrseg —— 「耳机壳」分割链路 (ego_relation_policy step2 的 DINO→SAM2 复刻)。

本包的代码都是从 ego_relation_policy 复制过来再适配的, 出处如下 (上游目录只读, 不改):

    xrseg/dinosam.py        <- third_party/humanego_runtime/preprocess/DINOSAM.py
    xrseg/sam2_video.py     <- src/ego_relation/s2_object_relations/sam2_video.py
    xrseg/utils/*.py        <- third_party/humanego_runtime/utils/
    xrseg/cfg/DINOSAM.yaml  <- third_party/humanego_runtime/cfg/preprocess/base/DINOSAM.yaml

上游是 Project Aria 的单目流水线, 本仓库是 PICO 4 Ultra 双目 2160x810: 只取
**左半 eye0** 当单目帧喂进去, 保证与 step2 的相机假设一致 (眼别判定见
tools/eye_order_check.py 与 out/head_cam_origin.md)。

每一处偏离上游的地方, 在文件里都用 "适配:" 注释标出来了。
"""
