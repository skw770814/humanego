"""OpenPi WebSocket inference server for XinHai bi-manual robot.

Loads a trained pi05_xinhai checkpoint and serves actions via WebSocket.
The robot client sends observations with native keys (camera_head_left,
camera_wrist_left, ...); the server repacks them into the format expected
by XinHaiInputs before running inference.

Usage:
    uv run scripts/serve_policy_xinhai.py \
        --checkpoint-dir ./checkpoints/pi05_xinhai/my_run/1000 \
        --port 5000

    # With default prompt:
    uv run scripts/serve_policy_xinhai.py \
        --checkpoint-dir ./checkpoints/pi05_xinhai/my_run/1000 \
        --default-prompt "fold Tshirt" \
        --port 5000
"""

import dataclasses
import logging
import socket

import tyro

from openpi import transforms
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    """Arguments for the XinHai policy server."""

    # Training config name.
    config: str = "pi05_xinhai"

    # Checkpoint directory (e.g., "checkpoints/pi05_xinhai/my_run/1000").
    checkpoint_dir: str

    # Port to serve the policy on.
    port: int = 5000

    # Default prompt to inject if "prompt" is not present in the observation.
    default_prompt: str | None = None

    # Record the policy's behavior to disk for debugging.
    record: bool = False


def main(args: Args) -> None:
    train_config = _config.get_config(args.config)

    # Inference repack: robot native keys → XinHaiInputs expected keys.
    #
    # Robot sends (from R1_PRO_Interface.get_observation()):
    #   observation.state, camera_head_left, camera_wrist_left, camera_wrist_right
    #
    # XinHaiInputs reads (after RepackTransform in xinhai_config.py):
    #   observation/image, observation/wrist_image_left,
    #   observation/wrist_image_right, observation/state
    repack = transforms.Group(
        inputs=[
            transforms.RepackTransform(
                {
                    "observation/image": "camera_head_left",
                    "observation/wrist_image_left": "camera_wrist_left",
                    "observation/wrist_image_right": "camera_wrist_right",
                    "observation/state": "observation.state",
                }
            )
        ]
    )

    policy = _policy_config.create_trained_policy(
        train_config,
        args.checkpoint_dir,
        repack_transforms=repack,
        default_prompt=args.default_prompt,
    )

    policy_metadata = policy.metadata

    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
