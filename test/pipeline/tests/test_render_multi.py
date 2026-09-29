from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("cv2", reason="xrrel.lift imports OpenCV")

from xrrel.render import (
    REL_PANEL_H,
    _object_rgb,
    _with_object_axis,
    draw_rel_panel,
    draw_relative,
    video_height,
)


def _pose(x: float, y: float, z: float) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = [x, y, z]
    return result


def _fields(instance_id: str, relative: np.ndarray | None) -> dict:
    return {
        "instance_id": instance_id,
        "T_object_midpoint": relative,
        "rpy": np.zeros(3),
        "distance": 0.2 if relative is not None else float("nan"),
        "latched": False,
        "observed": relative is not None,
        "closed": False,
        "status": "OBSERVED" if relative is not None else "NO-OBS",
    }


def test_two_objects_share_one_video_canvas() -> None:
    assert video_height(2) == 1324
    assert video_height(3) == 1492

    image = Image.new("RGB", (2160, 988), (0, 0, 0))
    midpoint = _pose(0.0, 0.0, 1.0)
    object1 = _pose(-0.20, 0.0, 1.0)
    object2 = _pose(+0.20, 0.0, 1.0)
    relative1 = np.linalg.inv(object1) @ midpoint
    relative2 = np.linalg.inv(object2) @ midpoint

    first = draw_relative(
        image,
        frame=0,
        mask_cloud=np.array([[-0.20, 0.0, 1.0]]),
        T_camera0_object=object1,
        T_camera0_midpoint=midpoint,
        T_object_midpoint=relative1,
        fields=_fields("obj1", relative1),
        instance_id="obj1",
        object_color=_object_rgb("obj1"),
        valid=True,
        draw_text=False,
    )
    both = draw_relative(
        first,
        frame=0,
        mask_cloud=np.array([[+0.20, 0.0, 1.0]]),
        T_camera0_object=object2,
        T_camera0_midpoint=midpoint,
        T_object_midpoint=relative2,
        fields=_fields("obj2", relative2),
        instance_id="obj2",
        object_color=_object_rgb("obj2"),
        valid=True,
        draw_text=False,
    )
    before = np.asarray(first)
    after = np.asarray(both)
    assert np.count_nonzero(after != before) > 0
    assert np.any(np.all(after == np.asarray(_object_rgb("obj1")), axis=2))
    assert np.any(np.all(after == np.asarray(_object_rgb("obj2")), axis=2))


def test_invalid_object_does_not_draw_fake_tcp() -> None:
    image = Image.new("RGB", (2160, 988), (0, 0, 0))
    output = draw_relative(
        image,
        frame=0,
        mask_cloud=np.zeros((0, 3)),
        T_camera0_object=np.eye(4),
        T_camera0_midpoint=np.eye(4),
        T_object_midpoint=None,
        fields=_fields("obj1", None),
        instance_id="obj1",
        valid=False,
        draw_text=False,
    )
    np.testing.assert_array_equal(np.asarray(output), np.asarray(image))


def test_legacy_single_object_shapes_and_per_object_panels() -> None:
    poses = _with_object_axis(np.zeros((5, 4, 4)), 4, "poses")
    flags = _with_object_axis(np.zeros(5, dtype=bool), 2, "flags")
    assert poses.shape == (5, 1, 4, 4)
    assert flags.shape == (5, 1)

    relative = np.repeat(np.eye(4)[None], 5, axis=0)
    panel = draw_rel_panel(
        2160,
        frame=2,
        relative=relative,
        valid=np.ones(5, dtype=bool),
        latched=np.zeros(5, dtype=bool),
        observed=np.ones(5, dtype=bool),
        distance=np.full(5, 0.2),
        instance_id="obj2",
    )
    assert panel.size == (2160, REL_PANEL_H)
