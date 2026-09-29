from __future__ import annotations

import numpy as np
from PIL import Image
from unitree_deploy.real_unitree_env import make_real_env


class UnitreeRobotInterface:
    """Thin OpenPI adapter around unitree-deploy's real robot environment."""

    def __init__(
        self,
        robot_type: str,
        dt: float,
        network_interface: str | None = None,
    ) -> None:
        self._env = make_real_env(
            robot_type=robot_type,
            dt=dt,
            network_interface=network_interface,
        )
        self._connected = False

    def connect(self) -> None:
        self._env.connect()
        self._connected = True

    def get_observation(self, prompt: str) -> dict:
        observation = self._env.get_observation()
        images = dict(observation["images"])

        if "cam_left_high" in images:
            images["cam_high"] = images.pop("cam_left_high")
        images.pop("cam_right_high", None)

        processed_images: dict[str, np.ndarray] = {}
        for name, image in images.items():
            if "_depth" in name:
                continue
            resized = Image.fromarray(np.asarray(image)).resize((224, 224), Image.Resampling.BILINEAR)
            processed_images[name] = np.asarray(resized).transpose(2, 0, 1)

        if "cam_high" not in processed_images:
            raise KeyError(f"Unitree observation has no cam_high image; received {tuple(processed_images)}")

        return {
            "state": np.asarray(observation["qpos"]),
            "images": processed_images,
            "prompt": prompt,
        }

    def step(self, action: np.ndarray) -> None:
        self._env.step(np.asarray(action))

    def close(self) -> None:
        if self._connected:
            self._env.close()
            self._connected = False
