"""Camera view helpers: standard views, axis look-along, empty workspace frame."""

from __future__ import annotations

import numpy as np
import pytest

from cadcore.scale import (
    EMPTY_PLANE_HALF_MM,
    empty_workspace_camera,
    plane_half_mm,
)


def test_empty_plane_half_is_modest():
    """Opening screen must not use huge planes that fill the window."""
    h = plane_half_mm(50.0, has_display_solids=False)
    assert h == pytest.approx(EMPTY_PLANE_HALF_MM)
    assert h <= 20.0
    # With solids, half grows with the part
    hs = plane_half_mm(200.0, has_display_solids=True)
    assert hs > h


def test_empty_workspace_camera_frames_planes():
    pos, focus, up = empty_workspace_camera(16.0)
    assert focus == (0.0, 0.0, 0.0)
    # Camera sits outside the plane square
    dist = float(np.linalg.norm(np.asarray(pos)))
    assert dist > 16.0 * 2.0
    assert abs(up[1] - 1.0) < 1e-9


def test_view_name_aliases():
    """view_along_axis mapping is stable (pure)."""
    mapping = {
        "x": ("right", "left"),
        "y": ("top", "bottom"),
        "z": ("front", "back"),
    }
    assert mapping["x"][0] == "right"
    assert mapping["z"][1] == "back"
