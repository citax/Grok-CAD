"""Polyline chaining + pixel-based endpoint snap."""

from __future__ import annotations

import pytest

from app.sketch_mode import SketchController, SketchTool
from cadcore.document import Document, FeatureType
from cadcore.profiles import find_closed_line_loops
from cadcore.sketch import LineEntity, PlaneFrame, Sketch, line_length


def test_line_tool_chains_from_last_endpoint():
    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    ctrl = SketchController(sk)
    ctrl.set_tool(SketchTool.LINE)
    ctrl.set_snap_world_tol(0.2)
    assert ctrl.on_press((0.0, 0.0))  # start
    assert ctrl.on_press((1.0, 0.0)) == "Line"
    assert len(sk.entities) == 1
    assert ctrl.is_drawing()
    assert ctrl.draw is not None
    # Next segment starts at exact last endpoint
    assert ctrl.draw.points[0] == (1.0, 0.0)
    assert ctrl.draw.chain_start == (0.0, 0.0)
    assert ctrl.on_press((1.0, 1.0)) == "Line"
    assert len(sk.entities) == 2
    e0, e1 = sk.entities[0], sk.entities[1]
    assert isinstance(e0, LineEntity) and isinstance(e1, LineEntity)
    assert e0.p1 == e1.p0 == (1.0, 0.0)


def test_auto_close_chain_forms_line_loop():
    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    ctrl = SketchController(sk)
    ctrl.set_tool(SketchTool.LINE)
    ctrl.set_snap_world_tol(0.25)
    ctrl.on_press((0.0, 0.0))
    ctrl.on_press((2.0, 0.0))
    ctrl.on_press((2.0, 2.0))
    ctrl.on_press((0.0, 2.0))
    # snap onto start
    msg = ctrl.on_press((0.05, 0.05))  # within tol of origin start
    assert msg == "LineClosed"
    assert not ctrl.is_drawing()
    assert len(sk.entities) == 4
    # exact shared start
    last = sk.entities[-1]
    assert isinstance(last, LineEntity)
    assert last.p1 == (0.0, 0.0)
    loops = find_closed_line_loops(sk)
    assert len(loops) == 1
    assert abs(loops[0].area() - 4.0) < 1e-9


def test_closed_chain_extrudes_watertight():
    doc = Document()
    doc.seed_reference_planes()
    front = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(front.id)
    sk = skf.sketch
    ctrl = SketchController(sk)
    ctrl.set_tool(SketchTool.LINE)
    ctrl.set_snap_world_tol(0.2)
    L = 3.0
    ctrl.on_press((0.0, 0.0))
    ctrl.on_press((L, 0.0))
    ctrl.on_press((L, L))
    ctrl.on_press((0.0, L))
    ctrl.on_press((0.0, 0.0))
    assert len(sk.entities) == 4
    feat = doc.create_extrude(skf.id, 2.0)
    mesh = doc.evaluate_feature(feat.id)
    assert mesh is not None and mesh.is_watertight()
    expect = L * L * 2.0
    assert abs(mesh.volume() - expect) / expect < 0.01


def test_pixel_based_snap_picks_endpoint_when_world_tol_large():
    """At zoomed-out scale, large world tol (from pixels) still hits endpoints over grid."""
    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    sk.add_line((0.0, 0.0), (5.0, 0.0))
    ctrl = SketchController(sk)
    ctrl.set_tool(SketchTool.LINE)
    # Simulate zoomed out: 1 px ≈ 0.5 world → 14 px ≈ 7 world units
    ctrl.set_snap_world_tol(7.0)
    # Click near endpoint but not exact, and nearer to a grid point
    sn = ctrl.snap((0.4, 0.3), drawing=True)
    assert sn.kind in ("point", "origin")
    assert sn.uv == (0.0, 0.0)


def test_point_snap_preferred_over_grid():
    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    sk.add_line((0.3, 0.3), (1.0, 0.3))  # endpoint off-grid
    ctrl = SketchController(sk)
    ctrl.set_snap_world_tol(0.5)
    sn = ctrl.snap((0.35, 0.28), drawing=False)
    assert sn.kind == "point"
    assert abs(sn.uv[0] - 0.3) < 1e-12


def test_enter_ends_open_chain():
    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    ctrl = SketchController(sk)
    ctrl.set_tool(SketchTool.LINE)
    ctrl.on_press((0, 0))
    ctrl.on_press((1, 0))
    assert ctrl.in_line_chain()
    msg = ctrl.confirm_current()  # mid-chain, one point only → end
    assert msg and msg.startswith("ChainEnd")
    assert not ctrl.is_drawing()
