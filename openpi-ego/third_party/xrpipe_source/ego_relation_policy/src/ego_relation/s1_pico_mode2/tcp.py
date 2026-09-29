from __future__ import annotations

import numpy as np


TCP_TO_INWARD_PALM = {
    "left": np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
        dtype=np.float64,
    ),
    "right": np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    ),
}


def palm_pose_to_tcp(palm_pose: np.ndarray, side: str) -> np.ndarray:
    """Apply the fixed side-specific palm-to-TCP axis convention."""
    if side not in TCP_TO_INWARD_PALM:
        raise ValueError(f"side must be left or right, got {side!r}")
    pose = np.asarray(palm_pose, dtype=np.float64)
    if pose.shape[-2:] != (4, 4):
        raise ValueError(f"palm pose must end with (4,4), got {pose.shape}")
    result = pose.copy()
    result[..., :3, :3] = pose[..., :3, :3] @ TCP_TO_INWARD_PALM[side].T
    return result
