"""Inline length entry while drawing a line."""

from __future__ import annotations

import math

from app.sketch_mode import SketchController, SketchTool
from cadcore.sketch import LineEntity, PlaneFrame, Sketch, line_length
from cadcore.units import Unit, to_mm


def test_commit_line_length_along_direction():
    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    ctrl = SketchController(sk)
    ctrl.set_tool(SketchTool.LINE)
    ctrl.on_press((0.0, 0.0))
    # Aim 30°
    ctrl.preview_uv = (math.cos(math.radians(30)), math.sin(math.radians(30)))
    msg = ctrl.commit_line_length(10.0)
    assert msg == "Line"
    ent = sk.entities[-1]
    assert isinstance(ent, LineEntity)
    assert abs(line_length(ent) - 10.0) < 1e-9
    ang = math.degrees(math.atan2(ent.p1[1] - ent.p0[1], ent.p1[0] - ent.p0[0]))
    assert abs(ang - 30.0) < 1e-6


def test_commit_line_length_default_u_when_no_preview():
    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    ctrl = SketchController(sk)
    ctrl.set_tool(SketchTool.LINE)
    ctrl.on_press((1.0, 2.0))
    ctrl.preview_uv = None
    msg = ctrl.commit_line_length(7.5)
    assert msg == "Line"
    ent = sk.entities[-1]
    assert abs(line_length(ent) - 7.5) < 1e-9
    assert abs(ent.p1[0] - 8.5) < 1e-9
    assert abs(ent.p1[1] - 2.0) < 1e-9


def test_commit_uses_display_unit_mm_value():
    # Controller takes mm; conversion is caller's job
    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    ctrl = SketchController(sk)
    ctrl.set_tool(SketchTool.LINE)
    ctrl.on_press((0, 0))
    ctrl.preview_uv = (1, 0)
    mm = to_mm(2.0, Unit.CM)  # 20 mm
    assert ctrl.commit_line_length(mm) == "Line"
    assert abs(line_length(sk.entities[-1]) - 20.0) < 1e-9


def test_commit_requires_drawing_line():
    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    ctrl = SketchController(sk)
    ctrl.set_tool(SketchTool.LINE)
    assert ctrl.commit_line_length(5.0) is None  # no first point
    ctrl.set_tool(SketchTool.RECTANGLE)
    ctrl.on_press((0, 0))
    assert ctrl.commit_line_length(5.0) is None
