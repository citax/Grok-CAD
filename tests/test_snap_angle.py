"""Angular ortho snap: near-ortho snaps; free angles are preserved."""

from __future__ import annotations

import math

import pytest

from app.sketch_mode import SNAP_ANGLE_DEG, SketchController, SketchTool
from cadcore.sketch import PlaneFrame, Sketch


def _ctrl() -> SketchController:
    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    ctrl = SketchController(sk)
    ctrl.set_tool(SketchTool.LINE)
    ctrl.on_press((0.0, 0.0))
    return ctrl


@pytest.mark.parametrize("deg", [15, 20, 30, 45, 60, 75])
def test_free_angle_not_snapped(deg: float):
    ctrl = _ctrl()
    L = 5.3
    u = L * math.cos(math.radians(deg))
    v = L * math.sin(math.radians(deg))
    sn = ctrl.snap((u, v), drawing=True)
    assert sn.kind not in ("h", "v"), f"deg={deg} forced ortho kind={sn.kind}"
    assert abs(sn.uv[0] - u) < 1e-9
    assert abs(sn.uv[1] - v) < 1e-9


@pytest.mark.parametrize("deg", [0, 1, 3, 5, 6.5])
def test_near_horizontal_snaps(deg: float):
    ctrl = _ctrl()
    L = 5.3
    u = L * math.cos(math.radians(deg))
    v = L * math.sin(math.radians(deg))
    sn = ctrl.snap((u, v), drawing=True)
    assert sn.kind == "h", f"deg={deg} expected h got {sn.kind}"
    assert abs(sn.uv[1]) < 1e-9
    assert abs(abs(sn.uv[0]) - L) < 1e-6


@pytest.mark.parametrize("deg", [90, 88, 85, 84])
def test_near_vertical_snaps(deg: float):
    ctrl = _ctrl()
    L = 5.3
    u = L * math.cos(math.radians(deg))
    v = L * math.sin(math.radians(deg))
    sn = ctrl.snap((u, v), drawing=True)
    assert sn.kind == "v", f"deg={deg} expected v got {sn.kind}"
    assert abs(sn.uv[0]) < 1e-9


def test_angle_just_outside_band_is_free():
    """±7° band — 8° must remain free."""
    assert SNAP_ANGLE_DEG == 7.0
    ctrl = _ctrl()
    L = 5.3
    deg = SNAP_ANGLE_DEG + 1.0
    u = L * math.cos(math.radians(deg))
    v = L * math.sin(math.radians(deg))
    sn = ctrl.snap((u, v), drawing=True)
    assert sn.kind not in ("h", "v")
    assert abs(sn.uv[0] - u) < 1e-9
