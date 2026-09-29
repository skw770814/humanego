#!/usr/bin/env python3
"""Serve a manifest-validated G1-D BrainCo or EEF20-gripper checkpoint."""

from __future__ import annotations

import argparse
import logging
import pathlib
import socket

from openpi.training import unitree_eval_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--default-prompt")
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    if not (checkpoint / "params").is_dir():
        raise FileNotFoundError(f"Checkpoint params directory not found: {checkpoint / 'params'}")
    config = unitree_eval_config.build_eval_config(checkpoint)
    robot = config.policy_metadata["eval_component"]

    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server

    policy = policy_config.create_trained_policy(
        config,
        checkpoint,
        sample_kwargs={"rtc_action_dim": int(robot["dimension"])},
        default_prompt=args.default_prompt,
    )
    metadata = {**config.policy_metadata, "checkpoint_dir": str(checkpoint)}
    logging.info(
        "Serving %s (%s, %sD, %s camera) on %s:%d; execute chunks at the recorded 30 Hz timebase",
        robot["name"],
        config.policy_metadata["action_representation"],
        robot["dimension"],
        config.policy_metadata["robot_camera_mode"],
        socket.gethostname(),
        args.port,
    )
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
