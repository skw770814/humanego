"""Optional real-weight, offline multi-frame smoke test (no robot, no policy)."""
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import cv2
import numpy as np

from openpi.xrpipe_xinhai.config import ObservationArgs
from openpi.xrpipe_xinhai.perception import ResidentPerception


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("frames", nargs="+", help="Existing pipeline RGB frames, in temporal order")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="outputs/xrpipe_perception_smoke")
    args = parser.parse_args()
    config = ObservationArgs(contract="configs/xrpipe_xinhai/observation_contract.json", perception_device=args.device)
    perception = ResidentPerception(config, ("green type", "black plate"))
    model_ids = (id(perception.dino), id(perception.video))
    perception.warmup()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(args.frames):
        bgr = cv2.imread(frame)
        if bgr is None:
            raise FileNotFoundError(frame)
        rgb = cv2.cvtColor(cv2.resize(bgr, (640, 480)), cv2.COLOR_BGR2RGB)
        start = time.perf_counter()
        result = perception.process(rgb, index + 1)
        overlay = rgb.astype(float)
        for mask, color in zip(result["masks"], ([80, 190, 60], [70, 100, 230])):
            overlay[mask] = overlay[mask] * .5 + np.asarray(color) * .5
        cv2.imwrite(str(output / f"{index:05d}.png"), overlay.astype(np.uint8)[..., ::-1])
        print(dict(frame=frame, redetected=result["redetected"], pixels=result["masks"].sum(axis=(1, 2)).tolist(),
                   seconds=time.perf_counter() - start), flush=True)
        assert model_ids == (id(perception.dino), id(perception.video))
    perception.reset()
    assert model_ids == (id(perception.dino), id(perception.video))
    print("PASS: real weights loaded once, warmup/reset reuse models, sequential tracking completed")


if __name__ == "__main__":
    main()
