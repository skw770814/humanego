"""Read-only live snapshot or replay; no policy and no control commands."""
from __future__ import annotations

import dataclasses
from pathlib import Path

from openpi.xrpipe_xinhai.config import ROOT, ObservationArgs, contract, prompts_for, read_json
from openpi.xrpipe_xinhai.observation import ObservationPipeline, RobotReader
from openpi.xrpipe_xinhai.perception import ResidentPerception
from openpi.xrpipe_xinhai.visualization import visualize


@dataclasses.dataclass
class Args(ObservationArgs):
    live: bool = False
    replay: str | None = None  # snapshot.npz, or directory of chronologically sorted snapshots


def main(args):
    if args.live == (args.replay is not None):
        raise ValueError("Choose exactly one: --live or --replay PATH")
    if args.extrinsics == str(ROOT / "extrinsics.json") and read_json(args.extrinsics).get("T_B_C") is None:
        args.extrinsics = None
        print("Default extrinsics.json is uncalibrated: producing camera/object diagnostics only.", flush=True)
    manifest = contract(args.contract)
    prompts = prompts_for(args, manifest)
    perception = ResidentPerception(args, prompts)
    perception.warmup()
    pipeline = ObservationPipeline(args, perception)
    reader = RobotReader(args) if args.live else None
    try:
        if args.live:
            observation = pipeline.capture_observation(reader)
            path = Path(args.output_dir) / observation["metadata"]["observation_id"]
            visualize(observation, path, prompts)
            print(f"Saved {path}; complete_state={observation['state'] is not None}")
        else:
            source = Path(args.replay)
            paths = [source] if source.is_file() else sorted(source.rglob("snapshot.npz"))
            if not paths:
                raise FileNotFoundError("No replay snapshots found")
            previous_execution = None
            for source in paths:
                if previous_execution is not None:
                    pipeline.replay_feedback(previous_execution)
                observation = pipeline.capture_observation(replay=source)
                path = Path(args.output_dir) / observation["metadata"]["observation_id"]
                if source.resolve().parent == path.resolve():
                    raise ValueError("Replay output must not overwrite input snapshots")
                visualize(observation, path, prompts)
                print(f"Saved replay {path}")
                previous_execution = source.parent / "execution.json"
    finally:
        if reader:
            reader.close()


if __name__ == "__main__":
    import tyro
    main(tyro.cli(Args))
