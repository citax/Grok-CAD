"""Hover entity shows handles; body drag translates; hover pick is cheap."""

from __future__ import annotations

from app.sketch_mode import SketchController, SketchTool
from cadcore.sketch import PlaneFrame, Sketch


def test_hover_entity_body_sets_hover_entity_id():
    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    ln = sk.add_line((0.0, 0.0), (10.0, 0.0))
    ctrl = SketchController(sk)
    ctrl.set_tool(SketchTool.SELECT)
    # Hover mid-body (not endpoint — endpoints are handles)
    ctrl.on_move((5.0, 0.02))
    assert ctrl.hover_entity_id == ln.id
    # May or may not hit mid handle depending on tol; entity must be set either way
    assert ctrl.hover_handle is None or ctrl.hover_handle.entity_id == ln.id


def test_hover_clears_over_empty_space():
    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    sk.add_line((0.0, 0.0), (2.0, 0.0))
    ctrl = SketchController(sk)
    ctrl.set_tool(SketchTool.SELECT)
    ctrl.on_move((1.0, 0.0))
    assert ctrl.hover_entity_id is not None
    ctrl.on_move((50.0, 50.0))
    assert ctrl.hover_entity_id is None
    assert ctrl.hover_handle is None


def test_body_drag_translates_line():
    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    ln = sk.add_line((0.0, 0.0), (10.0, 0.0))
    ctrl = SketchController(sk)
    ctrl.set_tool(SketchTool.SELECT)
    msg = ctrl.on_press((3.0, 0.0))
    assert msg and msg.startswith("Drag body")
    ctrl.on_move((3.0, 3.0))
    ctrl.on_release((3.0, 3.0))
    # Whole line moved up by ~3
    assert abs(ln.p0[1] - 3.0) < 0.05
    assert abs(ln.p1[1] - 3.0) < 0.05
    assert abs(ln.p0[0] - 0.0) < 0.05
    assert abs(ln.p1[0] - 10.0) < 0.05


def test_endpoint_drag_still_resizes():
    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    ln = sk.add_line((0.0, 0.0), (10.0, 0.0))
    ctrl = SketchController(sk)
    ctrl.set_tool(SketchTool.SELECT)
    msg = ctrl.on_press((10.0, 0.0))
    assert msg and msg.startswith("Drag")
    assert ctrl.drag is not None
    assert ctrl.drag.handle_name == "p1"
    ctrl.on_move((14.0, 0.0))
    ctrl.on_release((14.0, 0.0))
    assert abs(ln.p1[0] - 14.0) < 0.05
    assert abs(ln.p0[0] - 0.0) < 0.05
