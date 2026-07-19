"""Persistent sketch constraints: survive drag, conflict, save/reopen."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from cadcore.constraints import (
    ConstraintKind,
    SketchConstraint,
    add_constraint,
    all_satisfied,
    constraint_residual,
    max_residual,
    solve_sketch,
)
from cadcore.document import Document, FeatureType
from cadcore.project_io import load_document, save_document
from cadcore.sketch import LineEntity, line_length, snapshot_sketch_contents


def _line_angle_deg(ent: LineEntity) -> float:
    du = ent.p1[0] - ent.p0[0]
    dv = ent.p1[1] - ent.p0[1]
    return math.degrees(math.atan2(dv, du))


def _drag(sk, eid: int, handle: str, uv):
    ent = sk.find_entity(eid)
    ent.set_handle(handle, uv)
    solve_sketch(sk, drag=(eid, handle, uv))


def test_horizontal_survives_drag():
    sk = __import__("cadcore.sketch", fromlist=["Sketch"]).Sketch()
    a = sk.add_line((0, 0), (10, 3))
    add_constraint(sk, SketchConstraint(id=-1, kind=ConstraintKind.HORIZONTAL, e0=a.id))
    assert abs(a.p0[1] - a.p1[1]) < 1e-6
    _drag(sk, a.id, "p1", (25.0, 40.0))
    assert abs(a.p0[1] - a.p1[1]) < 1e-5, "horizontal must hold after drag"
    assert constraint_residual(
        sk, sk.constraints[0]
    ) < 1e-5


def test_vertical_survives_drag():
    sk = __import__("cadcore.sketch", fromlist=["Sketch"]).Sketch()
    a = sk.add_line((1, 1), (4, 12))
    add_constraint(sk, SketchConstraint(id=-1, kind=ConstraintKind.VERTICAL, e0=a.id))
    _drag(sk, a.id, "p1", (30.0, 5.0))
    assert abs(a.p0[0] - a.p1[0]) < 1e-5


def test_equal_survives_drag():
    sk = __import__("cadcore.sketch", fromlist=["Sketch"]).Sketch()
    a = sk.add_line((0, 0), (10, 0))
    b = sk.add_line((0, 5), (4, 5))
    add_constraint(
        sk, SketchConstraint(id=-1, kind=ConstraintKind.EQUAL, e0=a.id, e1=b.id)
    )
    assert line_length(a) == pytest.approx(line_length(b), abs=1e-6)
    _drag(sk, a.id, "p1", (22.0, 0.0))
    assert line_length(a) == pytest.approx(line_length(b), abs=1e-4)


def test_perpendicular_survives_drag():
    sk = __import__("cadcore.sketch", fromlist=["Sketch"]).Sketch()
    a = sk.add_line((0, 0), (10, 0))
    b = sk.add_line((0, 0), (5, 5))
    add_constraint(
        sk,
        SketchConstraint(id=-1, kind=ConstraintKind.PERPENDICULAR, e0=a.id, e1=b.id),
    )
    d0 = np.array([a.p1[0] - a.p0[0], a.p1[1] - a.p0[1]])
    d1 = np.array([b.p1[0] - b.p0[0], b.p1[1] - b.p0[1]])
    assert abs(float(np.dot(d0, d1))) / (np.linalg.norm(d0) * np.linalg.norm(d1)) < 1e-5
    _drag(sk, b.id, "p1", (8.0, 12.0))
    d0 = np.array([a.p1[0] - a.p0[0], a.p1[1] - a.p0[1]])
    d1 = np.array([b.p1[0] - b.p0[0], b.p1[1] - b.p0[1]])
    cos = abs(float(np.dot(d0, d1))) / (np.linalg.norm(d0) * np.linalg.norm(d1) + 1e-15)
    assert cos < 1e-4, f"still perpendicular after drag, cos={cos}"


def test_parallel_survives_drag():
    sk = __import__("cadcore.sketch", fromlist=["Sketch"]).Sketch()
    a = sk.add_line((0, 0), (10, 0))
    b = sk.add_line((0, 5), (6, 2))
    add_constraint(
        sk, SketchConstraint(id=-1, kind=ConstraintKind.PARALLEL, e0=a.id, e1=b.id)
    )
    _drag(sk, b.id, "p1", (15.0, 9.0))
    d0 = np.array([a.p1[0] - a.p0[0], a.p1[1] - a.p0[1]])
    d1 = np.array([b.p1[0] - b.p0[0], b.p1[1] - b.p0[1]])
    cross = abs(float(d0[0] * d1[1] - d0[1] * d1[0]))
    n = np.linalg.norm(d0) * np.linalg.norm(d1)
    assert cross / (n + 1e-15) < 1e-4


def test_coincident_survives_drag():
    sk = __import__("cadcore.sketch", fromlist=["Sketch"]).Sketch()
    a = sk.add_line((0, 0), (10, 0))
    b = sk.add_line((10, 0), (15, 8))
    add_constraint(
        sk,
        SketchConstraint(
            id=-1,
            kind=ConstraintKind.COINCIDENT,
            e0=a.id,
            h0="p1",
            e1=b.id,
            h1="p0",
        ),
    )
    assert a.p1 == pytest.approx(b.p0, abs=1e-6)
    _drag(sk, a.id, "p1", (12.0, 4.0))
    assert a.p1[0] == pytest.approx(b.p0[0], abs=1e-4)
    assert a.p1[1] == pytest.approx(b.p0[1], abs=1e-4)


def test_fix_holds_point():
    sk = __import__("cadcore.sketch", fromlist=["Sketch"]).Sketch()
    a = sk.add_line((3, 4), (10, 4))
    add_constraint(
        sk, SketchConstraint(id=-1, kind=ConstraintKind.FIX, e0=a.id, h0="p0")
    )
    fixed = (3.0, 4.0)
    _drag(sk, a.id, "p0", (50.0, 50.0))
    assert a.p0[0] == pytest.approx(fixed[0], abs=1e-5)
    assert a.p0[1] == pytest.approx(fixed[1], abs=1e-5)
    # Free end can still move
    _drag(sk, a.id, "p1", (20.0, 4.0))
    assert a.p0[0] == pytest.approx(fixed[0], abs=1e-5)
    assert abs(a.p1[0] - 20.0) < 1.0  # roughly follows


def test_conflict_h_and_v_leaves_unchanged():
    sk = __import__("cadcore.sketch", fromlist=["Sketch"]).Sketch()
    a = sk.add_line((0, 0), (10, 2))
    add_constraint(sk, SketchConstraint(id=-1, kind=ConstraintKind.HORIZONTAL, e0=a.id))
    before = snapshot_sketch_contents(sk)
    with pytest.raises(ValueError, match="already Horizontal|Vertical|conflict"):
        add_constraint(sk, SketchConstraint(id=-1, kind=ConstraintKind.VERTICAL, e0=a.id))
    after = snapshot_sketch_contents(sk)
    assert before["entities"] == after["entities"]
    assert len(sk.constraints) == 1
    assert sk.find_entity(a.id).p0[1] == pytest.approx(sk.find_entity(a.id).p1[1])


def test_underconstrained_still_works():
    """Partial constraints are fine — free DOF remain movable."""
    sk = __import__("cadcore.sketch", fromlist=["Sketch"]).Sketch()
    a = sk.add_line((0, 0), (10, 0))
    b = sk.add_line((0, 5), (3, 8))  # unconstrained
    add_constraint(sk, SketchConstraint(id=-1, kind=ConstraintKind.HORIZONTAL, e0=a.id))
    _drag(sk, b.id, "p1", (9.0, 1.0))
    assert b.p1[0] == pytest.approx(9.0, abs=1e-3)
    assert abs(a.p0[1] - a.p1[1]) < 1e-5


def test_save_reopen_constraints_hold_after_drag():
    doc = Document()
    doc.seed_reference_planes()
    front = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(front.id)
    sk = skf.sketch
    a = sk.add_line((0, 0), (10, 1))
    b = sk.add_line((0, 4), (5, 4))
    add_constraint(sk, SketchConstraint(id=-1, kind=ConstraintKind.HORIZONTAL, e0=a.id))
    add_constraint(
        sk, SketchConstraint(id=-1, kind=ConstraintKind.EQUAL, e0=a.id, e1=b.id)
    )
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "cstr.gcad"
        save_document(doc, path)
        loaded = load_document(path)
    sk2 = next(f for f in loaded.features if f.type is FeatureType.SKETCH).sketch
    assert len(sk2.constraints) == 2
    la = sk2.find_entity(a.id)
    lb = sk2.find_entity(b.id)
    _drag(sk2, la.id, "p1", (18.0, 7.0))
    assert abs(la.p0[1] - la.p1[1]) < 1e-4
    assert line_length(la) == pytest.approx(line_length(lb), abs=1e-3)
    assert all_satisfied(sk2)
