"""Driving dimensions: promises that hold; angle keeps shared corners.

These tests exercise Document.apply_sketch_dimension and SketchController._apply_drag
(the geometry path the viewport calls after a handle move). They do **not** start
MainWindow or send Qt mouse events — that requires a working OpenGL/VTK display.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from app.sketch_mode import DragState, SketchController, SketchTool
from cadcore.document import Document, FeatureType
from cadcore.constraints import max_residual
from cadcore.project_io import load_document, save_document
from cadcore.sketch import (
    HandleKind,
    find_shared_line_endpoints,
    line_angle_degrees_oriented,
    line_length,
)


def _front_sketch():
    doc = Document()
    doc.seed_reference_planes()
    front = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(front.id)
    return doc, skf, skf.sketch


def _apply_handle_drag(ctrl: SketchController, eid: int, handle: str, to_uv):
    """Apply a handle drag the way SketchController does after a mouse move.

    This is **not** a Qt press/move/release sequence — it calls _apply_drag
    directly (same method on_move/on_release use once a DragState exists).
    """
    ent = ctrl.sketch.find_entity(eid)
    if handle == "rim":
        start = ent.rim_point()
        kind = HandleKind.RIM
    elif handle == "center":
        start = ent.center
        kind = HandleKind.CENTER
    else:
        start = getattr(ent, handle)
        kind = HandleKind.ENDPOINT
    ctrl.tool = SketchTool.SELECT
    ctrl.drag = DragState(eid, handle, kind, start)
    ctrl._apply_drag(to_uv)
    ctrl.drag = None
    return start


def _min_endpoint_distance(a, b) -> float:
    best = 1e9
    for pa in (a.p0, a.p1):
        for pb in (b.p0, b.p1):
            best = min(best, float(np.hypot(pa[0] - pb[0], pa[1] - pb[1])))
    return best


def _acute_angle(a, b) -> float:
    ang = line_angle_degrees_oriented(a, b)
    return min(ang, 180.0 - ang)


def test_length_dim_survives_drag_and_leaves_others_alone():
    doc, skf, sk = _front_sketch()
    a = sk.add_line((0, 0), (10, 0))
    other = sk.add_line((0, 5), (8, 5))
    other_before = (other.p0, other.p1)
    L0 = line_length(a)
    doc.apply_sketch_dimension(skf.id, a.id, "length", 45.0)
    assert line_length(a) == pytest.approx(45.0)
    # Other line must be byte-identical
    assert other.p0 == other_before[0] and other.p1 == other_before[1]
    ctrl = SketchController(sk)
    p1_before = (float(a.p1[0]), float(a.p1[1]))
    _apply_handle_drag(ctrl, a.id, "p1", (30.0, 20.0))
    p1_after = (float(a.p1[0]), float(a.p1[1]))
    assert p1_before != p1_after  # drag moved the free end
    assert line_length(a) == pytest.approx(45.0, abs=1e-4)
    assert other.p0 == other_before[0] and other.p1 == other_before[1]


def test_diameter_dim_survives_center_drag_leaves_lines_alone():
    doc, skf, sk = _front_sketch()
    ln = sk.add_line((0, 0), (10, 0))
    ln_before = (ln.p0, ln.p1)
    c = sk.add_circle((0, 0), 5.0)
    doc.apply_sketch_dimension(skf.id, c.id, "diameter", 20.0)
    assert c.radius == pytest.approx(10.0)
    assert ln.p0 == ln_before[0] and ln.p1 == ln_before[1]
    ctrl = SketchController(sk)
    center_before = (float(c.center[0]), float(c.center[1]))
    _apply_handle_drag(ctrl, c.id, "center", (12.0, 8.0))
    center_after = (float(c.center[0]), float(c.center[1]))
    assert center_before != center_after
    assert abs(center_after[0] - 12.0) < 0.5 and abs(center_after[1] - 8.0) < 0.5
    assert c.radius * 2 == pytest.approx(20.0, abs=1e-4)
    assert ln.p0 == ln_before[0] and ln.p1 == ln_before[1]


def test_angle_shared_corner_stays_closed_after_apply_and_drag():
    doc, skf, sk = _front_sketch()
    a = sk.add_line((0, 0), (10, 0))
    b = sk.add_line((0, 0), (5, 5))
    d0 = _min_endpoint_distance(a, b)
    assert d0 == pytest.approx(0.0, abs=1e-9)
    ang0 = _acute_angle(a, b)
    doc.apply_sketch_dimension(skf.id, a.id, "angle", 30.0, entity_b_id=b.id)
    d1 = _min_endpoint_distance(a, b)
    ang1 = _acute_angle(a, b)
    assert d1 == pytest.approx(0.0, abs=1e-6), f"corner opened after apply: {d1}"
    assert ang1 == pytest.approx(30.0, abs=0.05)
    assert find_shared_line_endpoints(a, b) is not None
    ctrl = SketchController(sk)
    p1_before = (float(b.p1[0]), float(b.p1[1]))
    _apply_handle_drag(ctrl, b.id, "p1", (8.0, 12.0))
    p1_after = (float(b.p1[0]), float(b.p1[1]))
    assert p1_before != p1_after
    d2 = _min_endpoint_distance(a, b)
    ang2 = _acute_angle(a, b)
    assert d2 == pytest.approx(0.0, abs=1e-5), f"corner opened after drag: {d2}"
    assert ang2 == pytest.approx(30.0, abs=0.1)


def test_angle_disjoint_lines_no_false_corner():
    doc, skf, sk = _front_sketch()
    c = sk.add_line((20, 0), (30, 0))
    d = sk.add_line((20, 10), (28, 14))
    assert find_shared_line_endpoints(c, d) is None
    doc.apply_sketch_dimension(skf.id, c.id, "angle", 45.0, entity_b_id=d.id)
    assert _acute_angle(c, d) == pytest.approx(45.0, abs=0.1)
    # Still not sharing a vertex
    assert find_shared_line_endpoints(c, d) is None
    ctrl = SketchController(sk)
    p1_before = (float(d.p1[0]), float(d.p1[1]))
    _apply_handle_drag(ctrl, d.id, "p1", (35.0, 20.0))
    assert (float(d.p1[0]), float(d.p1[1])) != p1_before
    assert _acute_angle(c, d) == pytest.approx(45.0, abs=0.15)


def test_dimension_conflict_refused_leaves_geometry():
    doc, skf, sk = _front_sketch()
    a = sk.add_line((0, 0), (10, 0))
    doc.apply_sketch_dimension(skf.id, a.id, "length", 20.0)
    from cadcore.constraints import ConstraintKind, SketchConstraint, add_constraint

    add_constraint(sk, SketchConstraint(-1, ConstraintKind.FIX, e0=a.id, h0="p0"))
    add_constraint(sk, SketchConstraint(-1, ConstraintKind.FIX, e0=a.id, h0="p1"))
    before = (tuple(a.p0), tuple(a.p1), len(sk.dimensions))
    with pytest.raises(ValueError, match="conflict|cannot"):
        doc.apply_sketch_dimension(skf.id, a.id, "length", 100.0)
    a2 = sk.find_entity(a.id)
    assert (tuple(a2.p0), tuple(a2.p1)) == (before[0], before[1])
    assert len(sk.dimensions) == before[2]


def test_angle_save_reopen_corner_and_angle_hold():
    doc, skf, sk = _front_sketch()
    a = sk.add_line((0, 0), (10, 0))
    b = sk.add_line((0, 0), (4, 4))
    doc.apply_sketch_dimension(skf.id, a.id, "angle", 30.0, entity_b_id=b.id)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ang.gcad"
        save_document(doc, path)
        loaded = load_document(path)
    skf2 = next(f for f in loaded.features if f.type is FeatureType.SKETCH)
    sk2 = skf2.sketch
    a2 = sk2.find_entity(a.id)
    b2 = sk2.find_entity(b.id)
    assert _min_endpoint_distance(a2, b2) == pytest.approx(0.0, abs=1e-6)
    ctrl = SketchController(sk2)
    _apply_handle_drag(ctrl, b2.id, "p1", (9.0, 15.0))
    assert _min_endpoint_distance(a2, b2) == pytest.approx(0.0, abs=1e-5)
    assert _acute_angle(a2, b2) == pytest.approx(30.0, abs=0.15)
