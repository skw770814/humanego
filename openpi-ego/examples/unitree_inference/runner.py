from __future__ import annotations

import argparse
import logging
import time

import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from policy_adapter import ROBOT_DIMS
from policy_adapter import adapt_policy
from robot_interface import UnitreeRobotInterface
from strategies import AsyncStrategy
from strategies import NaiveAsyncBuffer
from strategies import SynchronousStrategy
from strategies import TemporalEnsemblingBuffer
from strategies import TemporalSmoothingBuffer


def _make_parser(mode: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"OpenPI Unitree {mode} inference")
    parser.add_argument("--host", default="192.168.10.31")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--robot-type",
        choices=("auto", "unitree_g1_dex1", "unitree_g1_brainco"),
        default="auto",
    )
    parser.add_argument("--urdf-path", default=None)
    parser.add_argument("--network-interface", default=None)
    parser.add_argument("--dt", type=float, default=1 / 30)   #设置控制频率的参数
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-steps", type=int, default=10_000_000)

    if mode in {"sync", "rtc"}:
        parser.add_argument("--chunk-size", type=int, default=50)
    if mode != "sync":
        parser.add_argument("--inference-hz", type=float, default=1.0)
    if mode == "temporal_ensembling":
        parser.add_argument("--exp-weight-m", type=float, default=0.01)
    if mode in {"temporal_smoothing", "rtc"}:
        parser.add_argument("--max-latency-steps", type=int, default=8)
        parser.add_argument("--min-smooth-steps", type=int, default=10)
    if mode == "rtc":
        parser.add_argument("--rtc-execute-horizon", type=int, default=None)
    return parser



def _make_strategy(mode: str, args: argparse.Namespace, policy):
    if mode == "sync":   #同步  ok
        return SynchronousStrategy(policy, chunk_size=args.chunk_size)
    if mode == "async":  #异步   ok
        return AsyncStrategy(policy, NaiveAsyncBuffer(), inference_hz=args.inference_hz)
    if mode == "temporal_ensembling":  #时间集成
        return AsyncStrategy(
            policy,
            TemporalEnsemblingBuffer(exp_weight_m=args.exp_weight_m),
            inference_hz=args.inference_hz,
        )
    if mode == "temporal_smoothing":   #异步平滑
        return AsyncStrategy(
            policy,
            TemporalSmoothingBuffer(args.max_latency_steps, args.min_smooth_steps),
            inference_hz=args.inference_hz,
        )
    if mode == "rtc":  #rtc
        return AsyncStrategy(
            policy,
            TemporalSmoothingBuffer(args.max_latency_steps, args.min_smooth_steps),
            inference_hz=args.inference_hz,
            rtc=True,
            execute_horizon=args.rtc_execute_horizon or args.chunk_size,
            control_hz=1.0 / args.dt,
        )
    raise ValueError(f"Unsupported inference mode: {mode}")


def run(mode: str) -> None:
    args = _make_parser(mode).parse_args()  #加载参数
    #一些参数的判定
    if args.dt <= 0:
        raise ValueError("dt must be positive")
    if args.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if hasattr(args, "chunk_size") and args.chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    #与polciy建立联系
    websocket_policy = WebsocketClientPolicy(host=args.host, port=args.port)
    metadata = websocket_policy.get_server_metadata() #得到服务端的数据
    logging.info("Server metadata: %s", metadata)
    robot_type = metadata["robot_type"] if args.robot_type == "auto" else args.robot_type
    if robot_type not in ROBOT_DIMS:
        raise ValueError(f"Unsupported robot type from server metadata: {robot_type}")
    policy = adapt_policy(websocket_policy, metadata, robot_type, args.urdf_path)
    robot = UnitreeRobotInterface(robot_type, args.dt, args.network_interface) #机器人端
    strategy = _make_strategy(mode, args, policy)

    try:
        robot.connect()  #连接机器人
        input("Robot connected. Press Enter to start inference...")
        state_dim: int | None = None
        for step in range(args.max_steps):  #设定的执行最大的步数，这个非常的长
            observation = robot.get_observation(args.prompt)
            state_dim = state_dim or int(np.asarray(observation["state"]).size)
            strategy.update_observation(observation)  

            while not strategy.has_action():  #每次都会判断还有没有动作
                time.sleep(0.001)

            action = np.asarray(strategy.pop_action())  #一步一步的输出
            if action.ndim != 1 or action.size != state_dim:
                raise ValueError(f"Action shape {action.shape} does not match robot state dimension {state_dim}")
            #print(action)
            robot.step(action)
            if (step + 1) % 50 == 0:
                logging.info("Executed %d steps", step + 1)
    except KeyboardInterrupt:
        logging.info("Inference interrupted")
    finally:
        strategy.close()
        reset = getattr(policy, "reset", None)
        if callable(reset):
            reset()
        robot.close()


def main(mode: str) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    run(mode)
