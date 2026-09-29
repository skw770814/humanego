"""WebSocket robot client for XinHai bi-manual robot with OpenPi policy.

Connects to serve_policy_xinhai.py via WebSocket, sends observations from
the R1_PRO hardware, and executes received action chunks.

Usage:
    uv run scripts/robot_client_xinhai_ws.py \
        --server ws://172.16.0.100:5000 \
        --task "fold Tshirt" \
        --fps 30
"""

import asyncio
import dataclasses
import time

import cv2
import numpy as np
import tyro
import zerorpc
from openpi_client import msgpack_numpy
import websockets


# ==========================================
# 机器人硬件接口层 (复用自 robot_client_r1pro_PI.py)
# ==========================================
class R1ProInterface:
    """Hardware interface for the XinHai R1_PRO bi-manual robot."""

    def __init__(self, ip: str = "172.16.0.30", port: int = 4242):
        self._server = zerorpc.Client(heartbeat=20)
        self._server.connect(f"tcp://{ip}:{port}")

    def close(self):
        self._server.close()

    # --- Grippers ---

    def open_gripper_right(self, width: float):
        self._server.move_right_gripper(float(width))

    def open_gripper_left(self, width: float):
        self._server.move_left_gripper(float(width))

    def get_gripper_state_right(self) -> float:
        x = np.array(self._server.get_right_gripper(), dtype=np.float32)
        return float(x.reshape(-1)[0])

    def get_gripper_state_left(self) -> float:
        x = np.array(self._server.get_left_gripper(), dtype=np.float32)
        return float(x.reshape(-1)[0])

    # --- End-effector poses ---

    def get_ee_pose_right(self) -> np.ndarray:
        return np.array(self._server.get_right_ee_pose(), dtype=np.float32)

    def get_ee_pose_left(self) -> np.ndarray:
        return np.array(self._server.get_left_ee_pose(), dtype=np.float32)

    def move_ee_right(self, pose: np.ndarray):
        self._server.move_right_arm(np.asarray(pose, dtype=np.float32).tolist())

    def move_ee_left(self, pose: np.ndarray):
        self._server.move_left_arm(np.asarray(pose, dtype=np.float32).tolist())

    # --- Cameras ---

    def get_head_image(self) -> np.ndarray:
        return self._decode_image(self._server.get_left_head_image())

    def get_wrist_left_image(self) -> np.ndarray:
        return self._decode_image(self._server.get_left_wrist_image())

    def get_wrist_right_image(self) -> np.ndarray:
        return self._decode_image(self._server.get_right_wrist_image())

    @staticmethod
    def _decode_image(img_data) -> np.ndarray:
        nparr = np.frombuffer(img_data["data"], np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return cv2.resize(frame_rgb, (224, 224))

    # --- Full observation ---

    def get_observation(self) -> dict[str, np.ndarray]:
        """Return one frame in the format expected by serve_policy_xinhai.py.

        Returns:
            dict with keys:
                observation.state:  16D [L_EE(7), L_gripper(1), R_EE(7), R_gripper(1)]
                camera_head_left:    (224, 224, 3) uint8
                camera_wrist_left:   (224, 224, 3) uint8
                camera_wrist_right:  (224, 224, 3) uint8
        """
        state = np.concatenate(
            [
                self.get_ee_pose_left(),
                [self.get_gripper_state_left()],
                self.get_ee_pose_right(),
                [self.get_gripper_state_right()],
            ]
        )
        return {
            "observation.state": state,
            "camera_head_left": self.get_head_image(),
            "camera_wrist_left": self.get_wrist_left_image(),
            "camera_wrist_right": self.get_wrist_right_image(),
        }


# ==========================================
# 客户端配置
# ==========================================
@dataclasses.dataclass
class Args:
    # WebSocket server address (e.g., ws://172.16.0.100:5000).
    server: str = "ws://127.0.0.1:5000"

    # Task description sent as prompt to the model.
    task: str = "fold Tshirt"

    # Robot zerorpc IP.
    robot_ip: str = "172.16.0.30"

    # Robot zerorpc port.
    robot_port: int = 4242

    # Target control frequency (Hz).
    fps: int = 30

    # Number of action steps to take from each chunk before requesting
    # a new chunk. Set to 1 to request a new chunk every step, or higher
    # to reuse the same chunk for multiple steps (lower latency but stale actions).
    chunk_exec_steps: int = 1


def main(args: Args) -> None:
    robot = R1ProInterface(ip=args.robot_ip, port=args.robot_port)
    environment_dt = 1.0 / args.fps

    async def run():
        print(f"Connecting to policy server: {args.server}")
        async with websockets.connect(args.server, max_size=None) as ws:
            # Receive metadata from server.
            metadata = msgpack_numpy.unpackb(await ws.recv())
            print(f"Connected. Server metadata: {list(metadata.keys())}")

            chunk_buffer = []
            chunk_step = 0

            while True:
                loop_start = time.perf_counter()

                # --- Get new action chunk if needed ---
                if chunk_step >= len(chunk_buffer):
                    obs = robot.get_observation()
                    obs["prompt"] = args.task

                    await ws.send(msgpack_numpy.packb(obs))
                    response = msgpack_numpy.unpackb(await ws.recv())

                    actions = response["actions"]  # (action_horizon, 16)
                    chunk_buffer = [actions[i] for i in range(actions.shape[0])]
                    chunk_step = 0

                    timing = response.get("server_timing", {})
                    print(
                        f"Inference: {timing.get('infer_ms', 0):.1f}ms | "
                        f"Chunk size: {len(chunk_buffer)}"
                    )

                # --- Execute one step ---
                action = chunk_buffer[chunk_step]
                robot.move_ee_left(action[0:7])
                robot.open_gripper_left(float(action[7]))
                robot.move_ee_right(action[8:15])
                robot.open_gripper_right(float(action[15]))
                chunk_step += 1

                # --- Sleep to maintain target FPS ---
                elapsed = time.perf_counter() - loop_start
                await asyncio.sleep(max(0, environment_dt - elapsed))

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        robot.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
