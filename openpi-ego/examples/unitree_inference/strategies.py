from __future__ import annotations

from collections import deque
import threading
import time
from typing import Protocol

import numpy as np


class Policy(Protocol):
    def infer(self, observation: dict) -> dict: ...


def _extract_actions(result: dict) -> np.ndarray:
    if not isinstance(result, dict) or "actions" not in result:
        raise ValueError("Policy response must contain an 'actions' field")
    actions = np.asarray(result["actions"], dtype=np.float64)
    if actions.ndim != 2 or actions.shape[0] == 0:
        raise ValueError(f"Expected actions with shape [horizon, action_dim], got {actions.shape}")
    if not np.all(np.isfinite(actions)):
        raise ValueError("Policy returned a non-finite action")
    return actions


class SynchronousStrategy:
    def __init__(self, policy: Policy, chunk_size: int) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self._policy = policy
        self._chunk_size = chunk_size
        self._chunk: np.ndarray | None = None
        self._index = 0
    
    #根新观测，chunk为空或者执行的chunk快执行完了
    def update_observation(self, observation: dict) -> None:
        if self._chunk is None or self._index >= min(self._chunk_size, len(self._chunk)):
            self._chunk = _extract_actions(self._policy.infer(observation))  #根据观测得到chunk
            self._index = 0

    def has_action(self) -> bool:  #是否还有动作
        return self._chunk is not None and self._index < len(self._chunk)

    def pop_action(self) -> np.ndarray:
        if not self.has_action():  #没有动作了
            raise RuntimeError("No synchronous action is available")
        assert self._chunk is not None
        action = self._chunk[self._index].copy()
        self._index += 1  #去除当前一步出来
        return action

    def close(self) -> None:
        pass


class NaiveAsyncBuffer:
    """Use the newest chunk and skip the steps spent waiting for inference."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._chunk: np.ndarray | None = None
        self._chunk_start_t = 0
        self._global_t = 0
        self._last_action: np.ndarray | None = None

    def add_chunk(self, actions: np.ndarray, start_timestep: int) -> None:
        with self._lock:
            skip_steps = min(max(0, self._global_t - start_timestep), len(actions) - 1)
            self._chunk = actions.copy()
            self._chunk_start_t = self._global_t - skip_steps

    def has_action(self) -> bool:
        with self._lock:
            if self._chunk is None:
                return self._last_action is not None
            return self._global_t - self._chunk_start_t < len(self._chunk) or self._last_action is not None

    def pop_action(self) -> np.ndarray | None:
        with self._lock:
            if self._chunk is None:
                action = self._last_action
            else:
                index = max(0, self._global_t - self._chunk_start_t)
                action = self._chunk[index] if index < len(self._chunk) else self._last_action
            self._global_t += 1
            if action is not None:
                self._last_action = np.asarray(action).copy()
                return self._last_action.copy()
            return None

    def current_timestep(self) -> int:
        with self._lock:
            return self._global_t   #全局步数


class TemporalEnsemblingBuffer:
    """Aggregate every chunk that predicts the current global timestep."""

    def __init__(self, exp_weight_m: float) -> None:
        if exp_weight_m < 0:
            raise ValueError("exp_weight_m must be non-negative")
        self._exp_weight_m = exp_weight_m
        self._lock = threading.Lock()
        self._predictions: dict[int, list[tuple[int, np.ndarray]]] = {}
        self._current_t = 0
        self._inference_count = 0
        self._last_action: np.ndarray | None = None

    def add_chunk(self, actions: np.ndarray, start_timestep: int) -> None:
        with self._lock:
            inference_index = self._inference_count
            self._inference_count += 1
            for offset, action in enumerate(actions):
                timestep = start_timestep + offset
                self._predictions.setdefault(timestep, []).append((inference_index, action.copy()))
            for timestep in tuple(self._predictions):
                if timestep < max(0, self._current_t - 10):
                    del self._predictions[timestep]

    def has_action(self) -> bool:
        with self._lock:
            return bool(self._predictions.get(self._current_t)) or self._last_action is not None

    def pop_action(self) -> np.ndarray | None:
        with self._lock:
            predictions = sorted(self._predictions.get(self._current_t, ()), key=lambda item: item[0])
            if predictions:
                actions = np.stack([item[1] for item in predictions])
                weights = np.exp(-self._exp_weight_m * np.arange(len(actions), dtype=np.float64))
                weights /= weights.sum()
                self._last_action = np.sum(actions * weights[:, None], axis=0)
            action = None if self._last_action is None else self._last_action.copy()
            self._current_t += 1
            return action

    def current_timestep(self) -> int:
        with self._lock:
            return self._current_t


class TemporalSmoothingBuffer:
    """Linearly blend the unexecuted overlap when a new chunk arrives."""

    def __init__(self, max_latency_steps: int, min_smooth_steps: int) -> None:
        if max_latency_steps < 0:
            raise ValueError("max_latency_steps must be non-negative")
        if min_smooth_steps <= 0:
            raise ValueError("min_smooth_steps must be positive")
        self._max_latency_steps = max_latency_steps
        self._min_smooth_steps = min_smooth_steps
        self._lock = threading.Lock()
        self._chunk: deque[np.ndarray] = deque()
        self._steps_since_update = 0
        self._last_action: np.ndarray | None = None
        self._global_t = 0

    def add_chunk(self, actions: np.ndarray, start_timestep: int) -> None:
        del start_timestep
        with self._lock:
            drop_count = min(self._steps_since_update, self._max_latency_steps)
            if drop_count >= len(actions):
                return
            new_actions = [action.copy() for action in actions[drop_count:]]

            if self._chunk:
                old_actions = list(self._chunk)
                if len(old_actions) < self._min_smooth_steps:
                    old_actions.extend(old_actions[-1].copy() for _ in range(self._min_smooth_steps - len(old_actions)))
            elif self._last_action is not None:
                old_actions = [self._last_action.copy() for _ in range(self._min_smooth_steps)]
                self._last_action = None
            else:
                self._chunk = deque(new_actions)
                self._steps_since_update = 0
                return

            overlap = min(len(old_actions), len(new_actions))
            if overlap == 0:
                combined = new_actions
            else:
                old_actions = old_actions[: len(new_actions)]
                old_weights = np.ones(1) if overlap == 1 else np.linspace(1.0, 0.0, overlap)
                combined = [
                    old_weights[index] * old_actions[index] + (1.0 - old_weights[index]) * new_actions[index]
                    for index in range(overlap)
                ]
                combined.extend(new_actions[overlap:])
            self._chunk = deque(np.asarray(action).copy() for action in combined)
            self._steps_since_update = 0

    def has_action(self) -> bool:
        with self._lock:
            return bool(self._chunk)

    def pop_action(self) -> np.ndarray | None:
        with self._lock:
            if not self._chunk:
                return None
            if len(self._chunk) == 1:
                self._last_action = self._chunk[0].copy()
            action = self._chunk.popleft()
            self._steps_since_update += 1
            self._global_t += 1
            return action

    def current_timestep(self) -> int:
        with self._lock:
            return self._global_t


class AsyncStrategy:
    def __init__(
        self,
        policy: Policy,
        action_buffer: NaiveAsyncBuffer | TemporalEnsemblingBuffer | TemporalSmoothingBuffer,
        inference_hz: float,
        *,
        rtc: bool = False,
        execute_horizon: int = 0,
        control_hz: float = 0,
    ) -> None:
        if inference_hz <= 0:
            raise ValueError("inference_hz must be positive")
        if rtc and (execute_horizon <= 0 or control_hz <= 0):
            raise ValueError("RTC requires positive execute_horizon and control_hz")
        self._policy = policy
        self._buffer = action_buffer
        self._period = 1.0 / inference_hz
        self._rtc = rtc
        self._execute_horizon = execute_horizon
        self._control_hz = control_hz
        self._condition = threading.Condition()
        self._observation: dict | None = None
        self._stopping = False
        self._error: BaseException | None = None
        self._previous_chunk: np.ndarray | None = None
        self._delays: deque[float] = deque(maxlen=20)
        self._thread = threading.Thread(target=self._inference_loop, name="openpi-inference", daemon=True)
        self._thread.start()

    def update_observation(self, observation: dict) -> None:
        with self._condition:
            self._observation = observation
            self._condition.notify()

    def has_action(self) -> bool:  
        self._raise_worker_error()
        return self._buffer.has_action()   #直接判断这个buffer是否有动作

    def pop_action(self) -> np.ndarray:   #取动作
        self._raise_worker_error()
        action = self._buffer.pop_action()
        if action is None:
            raise RuntimeError("No asynchronous action is available")
        return action

    def close(self) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        self._thread.join(timeout=2.0)

    def _raise_worker_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("Inference worker stopped") from self._error

    def _inference_loop(self) -> None:
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._observation is not None or self._stopping)
                    if self._stopping:
                        return
                    observation = self._observation
                assert observation is not None

                start_timestep = self._buffer.current_timestep()
                request = dict(observation)
                if self._rtc:
                    request["execute_horizon"] = self._execute_horizon
                    request["enable_rtc"] = True
                    request["inference_delay"] = self._predicted_delay_steps()
                    if self._previous_chunk is not None:
                        request["prev_action_chunk"] = self._previous_chunk.tolist()

                start_time = time.monotonic()
                actions = _extract_actions(self._policy.infer(request))
                elapsed = time.monotonic() - start_time
                print(elapsed)
                if self._rtc:
                    self._delays.append(elapsed)
                    self._previous_chunk = actions.copy()
                self._buffer.add_chunk(actions, start_timestep)

                with self._condition:
                    deadline = start_time + self._period
                    while not self._stopping:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        self._condition.wait(timeout=remaining)
                    if self._stopping:
                        return
        except BaseException as error:
            self._error = error

    def _predicted_delay_steps(self) -> int:
        if not self._delays:
            return 0
        return max(0, round(float(np.median(self._delays)) * self._control_hz))
