"""Explicit coordinate adapters; all physical transforms use metres and column vectors."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

S = np.diag([1., 1., -1., 1.])


def se3(value):
    t = np.asarray(value, dtype=np.float64)
    if (t.shape != (4, 4) or not np.isfinite(t).all()
            or not np.allclose(t[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-5)):
        raise ValueError("Invalid proper SE(3) matrix")
    return t


def pose7(value):
    p = np.asarray(value, dtype=float)
    if p.shape != (7,) or not np.isfinite(p).all() or abs(np.linalg.norm(p[3:]) - 1) > .01:
        raise ValueError("TCP must be finite xyz + unit quaternion xyzw")
    t = np.eye(4)
    t[:3, :3] = Rotation.from_quat(p[3:]).as_matrix()
    t[:3, 3] = p[:3]
    return t


def to_pose7(t):
    t = se3(t)
    return np.r_[t[:3, 3], Rotation.from_matrix(t[:3, :3]).as_quat()]


def vec9(t):
    t = se3(t)
    return np.r_[t[:3, 3], t[:3, 0], t[:3, 1]]


def from_vec9(v):
    v = np.asarray(v, dtype=float)
    if v.shape != (9,) or not np.isfinite(v).all():
        raise ValueError("Expected finite xyz + grouped-column Rot6D")
    x, y = v[3:6].copy(), v[6:9].copy()
    if np.linalg.norm(x) < 1e-6:
        raise ValueError("Degenerate rotation column 1")
    x /= np.linalg.norm(x)
    y -= x * (x @ y)
    if np.linalg.norm(y) < 1e-6:
        raise ValueError("Degenerate rotation column 2")
    y /= np.linalg.norm(y)
    t = np.eye(4)
    t[:3, :3] = np.column_stack([x, y, np.cross(x, y)])
    t[:3, 3] = v[:3]
    return se3(t)


def relation_state(t_c_m, objects, closed):
    """Return stored/raw and canonical pre-normalization training state.

    T_E_M denotes the physical analogue of the *source* five-keypoint frame.
    Therefore the training reflection is applied here exactly once and undone
    on model deltas in targets(). It is NOT a robot/world extrinsic rotation.
    """
    relations = [np.linalg.inv(se3(t_c_m)) @ se3(t) for t in objects]
    raw = np.r_[np.concatenate([vec9(t) for t in relations]), float(closed)]
    canonical = np.r_[np.concatenate([vec9(S @ t @ S) for t in relations]), float(closed)]
    return raw.astype(np.float32), canonical.astype(np.float32)


def validate_schedule(steps, fps, horizon=50, train_fps=30.):
    if not np.isfinite(fps) or fps <= 0 or not 1 <= steps <= horizon:
        raise ValueError("steps must be 1..50 and fps finite/positive")
    if steps / fps > horizon / train_fps + 1e-9:
        raise ValueError("Requested execution time exceeds prediction horizon; extrapolation forbidden")


def resample(actions, steps, fps, closed):
    a = np.asarray(actions, dtype=float)
    validate_schedule(steps, fps)
    if a.shape != (50, 10) or not np.isfinite(a).all():
        raise ValueError("Policy must return finite, unnormalized (50,10) actions")
    ts = np.stack([from_vec9(v[:9]) for v in a])  # validate ENTIRE chunk before any send
    if fps == 30:
        return ts[:steps], a[:steps, 9] >= .5
    times = np.arange(51) / 30.
    query = np.arange(1, steps + 1) / fps
    all_t = np.concatenate([np.eye(4)[None], ts])
    out = np.tile(np.eye(4), (steps, 1, 1))
    for axis in range(3):
        out[:, axis, 3] = np.interp(query, times, all_t[:, axis, 3])
    out[:, :3, :3] = Slerp(times, Rotation.from_matrix(all_t[:, :3, :3]))(query).as_matrix()
    index = np.searchsorted(times, query + 1e-10, side="right") - 1
    grip = np.r_[float(closed), a[:, 9]][index] >= .5
    return out, grip


def targets(anchor_e, t_e_m, actions, steps, fps, closed,
            max_translation=.15, max_rotation_deg=60.):
    delta, grips = resample(actions, steps, fps, closed)
    # Undo training canonicalization, not a second camera/world conversion.
    delta = S @ delta @ S
    if (np.linalg.norm(delta[:, :3, 3], axis=1).max() > max_translation
            or np.rad2deg(Rotation.from_matrix(delta[:, :3, :3]).magnitude()).max() > max_rotation_deg):
        raise ValueError("Predicted chunk exceeds configured anchor-relative motion limits")
    anchor = se3(anchor_e) @ se3(t_e_m)
    result = anchor @ delta @ np.linalg.inv(t_e_m)
    return np.stack([to_pose7(t) for t in result]), np.where(grips, 0., 80.)

