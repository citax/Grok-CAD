"""Arc entity: draw/reshape, radius dim, tangent, closed extrude.

Uses Document + SketchController (geometry path the viewport calls).
Does not start MainWindow/Qt mouse (OpenGL fails headless here).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from app.sketch_mode import DragState, SketchController, SketchTool
from cadcore.constraints import (
    ConstraintKind,
    SketchConstraint,
    add_constraint,
    constraint_residual,
    solve_sketch,
)
from cadcore.document import Document, FeatureType
from cadcore.profiles import find_closed_line_loops
from cadcore.project_io import load_document, save_document
from cadcore.sketch import ArcEntity, HandleKind, LineEntity


def _doc_sketch():
    doc = Document()
    doc.seed_reference_planes()
    front = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(front.id)
    return doc, skf, skf.sketch


def _drag(ctrl, eid, handle, to_uv):
    ent = ctrl.sketch.find_entity(eid)
    if handle == "mid":
        start = ent.mid_uv()
        kind = HandleKind.MIDPOINT
    elif handle == "center":
        start = ent.center
        kind = HandleKind.CENTER
    elif handle == "p0":
        start = ent.p0() if isinstance(ent, ArcEntity) else ent.p0
        kind = HandleKind.ENDPOINT
    elif handle == "p1":
        start = ent.p1() if isinstance(ent, ArcEntity) else ent.p1
        kind = HandleKind.ENDPOINT
    else:
        start = getattr(ent, handle)
        kind = HandleKind.ENDPOINT
    ctrl.tool = SketchTool.SELECT
    ctrl.drag = DragState(eid, handle, kind, start)
    ctrl._apply_drag(to_uv)
    ctrl.drag = None


def _kink(line: LineEntity, arc: ArcEntity, end: str) -> float:
    """|sin| of angle between line and arc tangent at join (0 = smooth)."""
    d = np.array([line.p1[0] - line.p0[0], line.p1[1] - line.p0[1]], dtype=np.float64)
    nd = float(np.linalg.norm(d))
    d = d / nd
    t = arc.tangent_at_start() if end == "p0" else arc.tangent_at_end()
    return abs(float(d[0] * t[1] - d[1] * t[0]))


def test_arc_three_point_and_drag_moves_mid():
    doc, skf, sk = _doc_sketch()
    arc = sk.add_arc((0, 0), (5, 5), (10, 0))
    assert isinstance(arc, ArcEntity)
    assert arc.radius > 1e-6
    mid_before = arc.mid_uv()
    ctrl = SketchController(sk)
    _drag(ctrl, arc.id, "mid", (5, 8))
    mid_after = arc.mid_uv()
    assert mid_before != mid_after
    # Endpoints should still exist as handles
    assert abs(arc.p0()[0] - 0) < 2.0  # may move slightly with mid rebuild


def test_radius_dimension_holds_after_drag():
    doc, skf, sk = _doc_sketch()
    arc = sk.add_arc((0, 0), (4, 3), (8, 0))
    r0 = float(arc.radius)
    doc.apply_sketch_dimension(skf.id, arc.id, "radius", 12.0)
    assert arc.radius == pytest.approx(12.0)
    ctrl = SketchController(sk)
    p1_before = tuple(arc.p1())
    _drag(ctrl, arc.id, "p1", (15, 6))
    p1_after = tuple(arc.p1())
    assert p1_before != p1_after
    assert arc.radius == pytest.approx(12.0, abs=1e-3)


def test_tangent_survives_drag():
    doc, skf, sk = _doc_sketch()
    ln = sk.add_line((0, 0), (10, 0))
    arc = sk.add_arc((10, 0), (14, 4), (18, 0))
    add_constraint(
        sk,
        SketchConstraint(
            id=-1,
            kind=ConstraintKind.COINCIDENT,
            e0=ln.id,
            h0="p1",
            e1=arc.id,
            h1="p0",
        ),
    )
    add_constraint(
        sk,
        SketchConstraint(
            id=-1, kind=ConstraintKind.TANGENT, e0=ln.id, e1=arc.id, h1="p0"
        ),
    )
    k0 = _kink(ln, arc, "p0")
    assert k0 < 1e-3, f"kink after apply {k0}"
    ctrl = SketchController(sk)
    p1b = tuple(arc.p1())
    _drag(ctrl, arc.id, "p1", (16, 9))
    assert tuple(arc.p1()) != p1b
    k1 = _kink(ln, arc, "p0")
    assert k1 < 0.05, f"kink after drag {k1}"


def test_closed_line_arc_profile_extrudes():
    doc, skf, sk = _doc_sketch()
    # U-shape with arc bottom: left, top, right, arc base
    sk.add_line((0, 0), (0, 10))
    sk.add_line((0, 10), (20, 10))
    sk.add_line((20, 10), (20, 0))
    sk.add_arc((20, 0), (10, -6), (0, 0))
    loops = find_closed_line_loops(sk)
    assert len(loops) == 1
    assert loops[0].area() > 100
    ex = doc.create_extrude(skf.id, 5.0)
    mesh = doc.evaluate_feature(ex.id)
    assert mesh is not None
    assert mesh.is_watertight()
    assert mesh.volume() > 500


def test_arc_save_reload():
    doc, skf, sk = _doc_sketch()
    arc = sk.add_arc((1, 1), (3, 4), (5, 1))
    r = arc.radius
    doc.apply_sketch_dimension(skf.id, arc.id, "radius", r)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "arc.gcad"
        save_document(doc, path)
        loaded = load_document(path)
    sk2 = next(f for f in loaded.features if f.type is FeatureType.SKETCH).sketch
    arcs = [e for e in sk2.entities if isinstance(e, ArcEntity)]
    assert len(arcs) == 1
    assert arcs[0].radius == pytest.approx(r, abs=1e-6)
    assert any(d.role == "radius" for d in sk2.dimensions)


def test_arc_tool_three_clicks():
    doc, skf, sk = _doc_sketch()
    ctrl = SketchController(sk)
    ctrl.set_tool(SketchTool.ARC)
    msg1 = ctrl.on_press((0.0, 0.0))
    assert msg1 is not None and ctrl.draw is not None and len(ctrl.draw.points) == 1
    msg2 = ctrl.on_press((5.0, 4.0))
    # Second click places on-arc point; arc not finished until third click
    assert ctrl.draw is not None and len(ctrl.draw.points) == 2
    assert msg2 is None
    msg3 = ctrl.on_press((10.0, 0.0))
    assert msg3 == "Arc"
    arcs = [e for e in sk.entities if isinstance(e, ArcEntity)]
    assert len(arcs) == 1
