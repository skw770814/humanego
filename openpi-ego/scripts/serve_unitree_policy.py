from __future__ import annotations

import argparse
import logging
import pathlib
import socket

from openpi.training import unitree_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve a configured Unitree OpenPI policy")
    parser.add_argument("--config")
    parser.add_argument("--checkpoint", type=pathlib.Path)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--default-prompt")
    parser.add_argument("--list", action="store_true")
    return parser


def _print_configs() -> None:
    for spec in unitree_config.UNITREE_SPECS.values():
        status = "ready" if spec.checkpoint_dir.is_dir() else "missing"
        print(f"{spec.name:40s} {status:7s} {spec.checkpoint_dir}")


def main() -> None:
    args = _parser().parse_args()
    if args.list:
        _print_configs()
        return
    if not args.config:
        raise ValueError("--config is required unless --list is used")

    spec = unitree_config.get_spec(args.config)
    checkpoint_dir = (args.checkpoint or spec.checkpoint_dir).resolve()
    norm_path = unitree_config.checkpoint_norm_stats_path(spec, checkpoint_dir)
    if not (checkpoint_dir / "params").is_dir():
        raise FileNotFoundError(f"Checkpoint params directory not found: {checkpoint_dir / 'params'}")
    if not norm_path.is_file():
        raise FileNotFoundError(f"Checkpoint normalization stats not found: {norm_path}")

    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server

    train_config = unitree_config.get_train_config(spec.name)
    policy = policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        sample_kwargs={"rtc_action_dim": spec.model_dim},
        default_prompt=args.default_prompt,
    )
    metadata = {**spec.metadata, "checkpoint_dir": str(checkpoint_dir)}
    hostname = socket.gethostname()
    logging.info("Serving %s on %s:%d", spec.name, hostname, args.port)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
