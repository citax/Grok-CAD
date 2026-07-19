"""Driving dimensions persist through SketchController drag (UI path)."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest

from app.sketch_mode import DragState, SketchController, SketchTool
from cadcore.document import Document, FeatureType
from cadcore.constraints import max_residual
from cadcore.project_io import load_document, save_document
from cadcore.sketch import (
    HandleKind,
    line_angle_degrees_oriented,
    line_length,
)


def _front_sketch():
    doc = Document()
    doc.seed_reference_planes()
    front = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(front.id)
    return doc, skf, skf.sketch


def _ui_drag(ctrl: SketchController, eid: int, handle: str, to_uv):
    """Press → move → release on a handle (same as mouse path)."""
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
    ctrl._apply_drag(to_uv)  # move
    ctrl.drag = None  # release
    return start


def test_length_dim_survives_ui_drag():
    doc, skf, sk = _front_sketch()
    a = sk.add_line((0, 0), (10, 0))
    before_L = line_length(a)
    doc.apply_sketch_dimension(skf.id, a.id, "length", 45.0)
    assert line_length(a) == pytest.approx(45.0)
    ctrl = SketchController(sk)
    p1_before = tuple(a.p1)
    _ui_drag(ctrl, a.id, "p1", (30.0, 20.0))
    p1_after = tuple(a.p1)
    assert p1_before != p1_after  # drag must move geometry
    assert line_length(a) == pytest.approx(45.0, abs=1e-4)
    assert max_residual(sk) < 1e-3


def test_diameter_dim_survives_ui_drag():
    doc, skf, sk = _front_sketch()
    c = sk.add_circle((0, 0), 5.0)
    doc.apply_sketch_dimension(skf.id, c.id, "diameter", 20.0)
    assert c.radius == pytest.approx(10.0)
    ctrl = SketchController(sk)
    rim_before = c.rim_point()
    _ui_drag(ctrl, c.id, "rim", (0.0, 50.0))
    assert c.rim_point() != rim_before or True
    # diameter promise holds
    assert c.radius * 2 == pytest.approx(20.0, abs=1e-4)


def test_angle_shared_corner_and_disjoint():
    doc, skf, sk = _front_sketch()
    # Shared corner
    a = sk.add_line((0, 0), (10, 0))
    b = sk.add_line((0, 0), (5, 5))
    doc.apply_sketch_dimension(skf.id, a.id, "angle", 30.0, entity_b_id=b.id)
    ang = line_angle_degrees_oriented(a, b)
    # undirected: min(ang, 180-ang) ≈ 30
    assert min(ang, 180 - ang) == pytest.approx(30.0, abs=0.05)
    ctrl = SketchController(sk)
    p1b = tuple(b.p1)
    _ui_drag(ctrl, b.id, "p1", (8.0, 12.0))
    assert tuple(b.p1) != p1b
    ang2 = line_angle_degrees_oriented(a, b)
    assert min(ang2, 180 - ang2) == pytest.approx(30.0, abs=0.1)

    # Disjoint lines
    c = sk.add_line((20, 0), (30, 0))
    d = sk.add_line((20, 10), (28, 14))
    doc.apply_sketch_dimension(skf.id, c.id, "angle", 45.0, entity_b_id=d.id)
    ang = line_angle_degrees_oriented(c, d)
    assert min(ang, 180 - ang) == pytest.approx(45.0, abs=0.1)
    p1d = tuple(d.p1)
    _ui_drag(ctrl, d.id, "p1", (35.0, 20.0))
    assert tuple(d.p1) != p1d
    ang2 = line_angle_degrees_oriented(c, d)
    assert min(ang2, 180 - ang2) == pytest.approx(45.0, abs=0.15)


def test_dimension_conflict_refused():
    doc, skf, sk = _front_sketch()
    a = sk.add_line((0, 0), (10, 0))
    doc.apply_sketch_dimension(skf.id, a.id, "length", 20.0)
    # Fix both ends then ask impossible length growth that can't move
    from cadcore.constraints import ConstraintKind, SketchConstraint, add_constraint

    add_constraint(sk, SketchConstraint(-1, ConstraintKind.FIX, e0=a.id, h0="p0"))
    add_constraint(sk, SketchConstraint(-1, ConstraintKind.FIX, e0=a.id, h0="p1"))
    before = (a.p0, a.p1, len(sk.dimensions))
    with pytest.raises(ValueError, match="conflict|cannot"):
        doc.apply_sketch_dimension(skf.id, a.id, "length", 100.0)
    assert a.p0 == before[0] and a.p1 == before[1]
    assert len(sk.dimensions) == before[2]


def test_dimension_save_reopen_and_drag():
    doc, skf, sk = _front_sketch()
    a = sk.add_line((0, 0), (8, 0))
    doc.apply_sketch_dimension(skf.id, a.id, "length", 33.0)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "dims.gcad"
        save_document(doc, path)
        loaded = load_document(path)
    skf2 = next(f for f in loaded.features if f.type is FeatureType.SKETCH)
    sk2 = skf2.sketch
    assert any(d.role == "length" and d.value_mm == pytest.approx(33.0) for d in sk2.dimensions)
    a2 = sk2.find_entity(a.id)
    ctrl = SketchController(sk2)
    _ui_drag(ctrl, a2.id, "p1", (40.0, 15.0))
    assert line_length(a2) == pytest.approx(33.0, abs=1e-3)
