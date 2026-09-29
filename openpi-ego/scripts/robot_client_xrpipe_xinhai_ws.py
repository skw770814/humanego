"""Synchronous observe -> infer -> fixed-anchor right-arm chunk execution."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from openpi.xrpipe_xinhai.config import ADAPTER, ObservationArgs, contract, prompts_for, validate_contract
from openpi.xrpipe_xinhai.geometry import pose7, targets, validate_schedule
from openpi.xrpipe_xinhai.observation import ObservationPipeline, RobotController, RobotReader, save_snapshot
from openpi.xrpipe_xinhai.perception import ResidentPerception


@dataclasses.dataclass
class Args(ObservationArgs):
    server: str = "ws://127.0.0.1:5000"
    task: str = ""
    chunk_exec_steps: int = 20
    fps: float = 30.
    dry_run: bool = True  # --no-dry-run is required to enable commands
    geometry_verified: bool = False  # explicit acknowledgement AFTER multi-pose visual verification
    max_cycles: int = 1  # 0 = continuous until interrupted/failure
    policy_timeout: float = 30.
    max_chunk_translation: float = .15
    max_chunk_rotation_deg: float = 60.
    max_step_translation: float = .03
    max_step_rotation_deg: float = 10.


def execute_chunk(reader, pipeline, observation, poses, grips, args, clock=time.monotonic, sleep=time.sleep):
    """Never integrates deltas, resends observations, catches up or retries a failed command."""
    sent = []
    feedback_events = []
    attempted = None
    start = clock()
    try:
        for j, (pose, gripper) in enumerate(zip(poses, grips)):
            deadline = start + (j + 1) / args.fps
            wait = deadline - clock()
            if wait > 0:
                sleep(wait)
            if clock() - deadline > 1 / args.fps:
                raise TimeoutError("Execution missed a period; no burst catch-up")
            if clock() - observation["received_monotonic"] > args.max_observation_age:
                raise TimeoutError("Observation expired during execution")
            # Feedback drives gripper filter and latch, never the target label.
            feedback = pipeline.poll_feedback(reader)
            if feedback is not None:
                tcp, width, closed = feedback
                feedback_events.append(dict(tcp=np.asarray(tcp).tolist(), width=float(width),
                                            closed=None if closed is None else bool(closed), time_ns=time.time_ns()))
            if clock() - deadline > 1 / args.fps:
                raise TimeoutError("Feedback RPC exceeded execution budget")
            if not args.dry_run:
                attempted = j
                reader.move(pose, gripper)
            sent.append(dict(index=j, seconds=clock() - start, dry_run=args.dry_run))
            if clock() - deadline > 1 / args.fps:
                raise TimeoutError("Command RPC exceeded execution budget; remaining chunk cancelled")
    finally:
        path = Path(args.output_dir) / observation["metadata"]["observation_id"] / "execution.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(completed=len(sent), requested=len(poses), last_attempted=attempted, events=sent, feedback=feedback_events), indent=2))
    return sent


def check_targets(poses, anchor, args):
    matrices = [pose7(anchor), *[pose7(p) for p in poses]]
    for before, after in zip(matrices, matrices[1:]):
        if (np.linalg.norm(after[:3, 3] - before[:3, 3]) > args.max_step_translation
                or np.rad2deg(Rotation.from_matrix(before[:3, :3].T @ after[:3, :3]).magnitude()) > args.max_step_rotation_deg):
            raise ValueError("Target step exceeds configured translation/rotation safety limit")


def main(args):
    from openpi_client import msgpack_numpy
    from websockets.sync.client import connect

    validate_schedule(args.chunk_exec_steps, args.fps)
    if args.max_cycles < 0 or args.max_observation_age <= 0 or args.policy_timeout <= 0:
        raise ValueError("Invalid cycle count/timeout")
    for value in (args.max_chunk_translation, args.max_chunk_rotation_deg, args.max_step_translation, args.max_step_rotation_deg):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Motion limits must be finite and positive")
    if not args.extrinsics:
        raise ValueError("Inference requires --extrinsics; use single-frame test for camera-only diagnostics")
    if not args.dry_run and not args.geometry_verified:
        raise ValueError("Live execution requires --geometry-verified after visual/multi-pose verification")
    manifest = contract(args.contract)
    prompts = prompts_for(args, manifest)
    perception = ResidentPerception(args, prompts)
    perception.warmup()
    pipeline = ObservationPipeline(args, perception)
    if pipeline.tbc is None:
        raise ValueError("openpi/extrinsics.json still has no T_B_C; provide measured extrinsics before inference")
    reader = RobotReader(args) if args.dry_run else RobotController(args)
    packer = msgpack_numpy.Packer()
    try:
        with connect(args.server, compression=None, max_size=32 * 1024 * 1024, open_timeout=args.policy_timeout) as ws:
            metadata = msgpack_numpy.unpackb(ws.recv(timeout=args.policy_timeout))
            validate_contract(metadata)
            if (metadata.get("inference_adapter") != ADAPTER
                    or metadata["dataset_contract"] != manifest["dataset_contract"]
                    or metadata["asset_id"] != manifest["asset_id"]):
                raise ValueError("Client/server contract or normalization asset mismatch")
            cycle = 0
            while args.max_cycles == 0 or cycle < args.max_cycles:
                observation = pipeline.capture_observation(reader)
                info = observation["metadata"]
                output = Path(args.output_dir) / info["observation_id"]
                if output.exists():
                    raise FileExistsError(f"Refusing to overwrite prior run {output}")
                save_snapshot(observation, output)
                request = dict(rgb=observation["rgb"], state=observation["state"], task=args.task,
                               observation_id=info["observation_id"], stamp_ns=info["stamp_ns"],
                               adapter=ADAPTER, contract_digest=manifest["dataset_contract"]["digest"])
                start = time.monotonic()
                ws.send(packer.pack(request))
                wire = ws.recv(timeout=args.policy_timeout)
                if isinstance(wire, str):
                    raise RuntimeError(f"Policy error: {wire}")
                response = msgpack_numpy.unpackb(wire)
                if (response.get("observation_id") != info["observation_id"]
                        or response.get("stamp_ns") != info["stamp_ns"] or response.get("adapter") != ADAPTER):
                    raise ValueError("Response does not belong to the current observation/adapter")
                if time.monotonic() - observation["received_monotonic"] > args.max_observation_age:
                    raise TimeoutError("Perception/policy result is too old; no targets sent")
                poses, grips = targets(pose7(observation["tcp"]), pipeline.tem, response["actions"],
                                       args.chunk_exec_steps, args.fps, info["closed"],
                                       args.max_chunk_translation, args.max_chunk_rotation_deg)
                check_targets(poses, observation["tcp"], args)
                current = pose7(reader.read_tcp())
                anchor = pose7(observation["tcp"])
                if (np.linalg.norm(current[:3, 3] - anchor[:3, 3]) > args.max_tcp_bracket_translation
                        or np.rad2deg(Rotation.from_matrix(anchor[:3, :3].T @ current[:3, :3]).magnitude()) > args.max_tcp_bracket_rotation_deg):
                    raise RuntimeError("TCP changed during inference; fixed anchor invalid, no targets sent")
                np.savez_compressed(output / "actions.npz", actions=response["actions"], targets=poses, gripper=grips)
                (output / "policy.json").write_text(json.dumps(dict(round_trip_seconds=time.monotonic()-start,
                    policy_seconds=response.get("policy_seconds"), task=args.task, object_prompts=prompts,
                    fps=args.fps, exec_steps=args.chunk_exec_steps, dry_run=args.dry_run), indent=2))
                execute_chunk(reader, pipeline, observation, poses, grips, args)
                cycle += 1
                print(f"Cycle {cycle}: {args.chunk_exec_steps} targets, dry_run={args.dry_run}, saved {output}", flush=True)
    finally:
        reader.close()


if __name__ == "__main__":
    import tyro
    main(tyro.cli(Args))
