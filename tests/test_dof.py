"""Motion-based DOF: fully defined = cannot move (not constraint counting)."""

from __future__ import annotations

import numpy as np
import pytest

from app.sketch_mode import DragState, SketchController, SketchTool
from cadcore.constraints import ConstraintKind, SketchConstraint, add_constraint, solve_sketch
from cadcore.document import Document, FeatureType
from cadcore.dof import entity_dof_status, format_dof_status_line, worst_constraint_label
from cadcore.sketch import HandleKind


def _front():
    doc = Document()
    doc.seed_reference_planes()
    front = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(front.id)
    return doc, skf, skf.sketch


def _drag(ctrl, eid, handle, start, to, kind=HandleKind.ENDPOINT):
    ctrl.tool = SketchTool.SELECT
    ctrl.drag = DragState(eid, handle, kind, start)
    ctrl._apply_drag(to)
    ctrl.drag = None


def test_bare_rect_under_and_moves():
    doc, skf, sk = _front()
    r = sk.add_rectangle((0, 0), (20, 10))
    assert entity_dof_status(sk, sk.find_entity(r.id)) == "under"
    ctrl = SketchController(sk)
    s = sk.find_entity(r.id).corners()[2]
    _drag(ctrl, r.id, "c2", s, (30, 20), HandleKind.CORNER)
    e = sk.find_entity(r.id).corners()[2]
    assert np.hypot(e[0] - s[0], e[1] - s[1]) > 1.0


def test_fully_defined_rect_locked():
    doc, skf, sk = _front()
    r = sk.add_rectangle((0, 0), (20, 10))
    rid = r.id
    add_constraint(sk, SketchConstraint(id=-1, kind=ConstraintKind.FIX, e0=rid, h0="c0"))
    doc.apply_sketch_dimension(skf.id, rid, "width", 20.0)
    doc.apply_sketch_dimension(skf.id, rid, "height", 10.0)
    assert entity_dof_status(sk, sk.find_entity(rid)) == "well"
    assert "Fully defined" in format_dof_status_line(sk)
    ctrl = SketchController(sk)
    s = sk.find_entity(rid).corners()[2]
    _drag(ctrl, rid, "c2", s, (50, 50), HandleKind.CORNER)
    e = sk.find_entity(rid).corners()[2]
    assert np.hypot(e[0] - s[0], e[1] - s[1]) < 0.05


def test_redundant_parallel_still_under():
    doc, skf, sk = _front()
    a = sk.add_line((0, 0), (10, 0))
    b = sk.add_line((0, 5), (8, 5))
    add_constraint(sk, SketchConstraint(id=-1, kind=ConstraintKind.HORIZONTAL, e0=a.id))
    add_constraint(sk, SketchConstraint(id=-1, kind=ConstraintKind.HORIZONTAL, e0=b.id))
    add_constraint(sk, SketchConstraint(id=-1, kind=ConstraintKind.PARALLEL, e0=a.id, e1=b.id))
    # Three relations, but still free to translate → must be under, not well
    assert entity_dof_status(sk, sk.find_entity(a.id)) == "under"
    assert entity_dof_status(sk, sk.find_entity(b.id)) == "under"
    ctrl = SketchController(sk)
    mid = sk.find_entity(a.id).midpoint()
    _drag(ctrl, a.id, "mid", mid, (mid[0], mid[1] + 5), HandleKind.MIDPOINT)
    assert abs(sk.find_entity(a.id).midpoint()[1] - (mid[1] + 5)) < 0.5


def test_overdefined_reports_guilty_dim():
    doc, skf, sk = _front()
    ln = sk.add_line((0, 0), (10, 0))
    add_constraint(sk, SketchConstraint(id=-1, kind=ConstraintKind.FIX, e0=ln.id, h0="p0"))
    add_constraint(sk, SketchConstraint(id=-1, kind=ConstraintKind.FIX, e0=ln.id, h0="p1"))
    sk.add_or_update_dimension(ln.id, "length", 20.0)
    solve_sketch(sk, max_iters=30)
    assert entity_dof_status(sk, sk.find_entity(ln.id)) == "over"
    g = worst_constraint_label(sk)
    assert g is not None and "length" in g
