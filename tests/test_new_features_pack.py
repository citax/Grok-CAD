"""Edit sketch rebuild, chamfer, patterns, mirror, offset plane, sketch ops."""

from __future__ import annotations

import numpy as np
import pytest

from cadcore.constraints import (
    ConstraintKind,
    SketchConstraint,
    add_constraint,
    solve_sketch,
)
from cadcore.document import Document, FeatureType
from cadcore.edge_fillet import extract_convex_edges
from cadcore.sketch import PlaneFrame, Sketch, ArcEntity
from cadcore.sketch_ops import (
    entity_dof_status,
    offset_line,
    toggle_construction,
    trim_line_at,
)


def _box_extrude(doc=None, size=20.0, depth=10.0):
    doc = doc or Document()
    doc.seed_reference_planes()
    front = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(front.id)
    skf.sketch.add_rectangle((0, 0), (size, size))
    ex = doc.create_extrude(skf.id, depth)
    return doc, skf, ex


def test_edit_sketch_volume_updates_after_geometry_change():
    doc, skf, ex = _box_extrude()
    v0 = doc.evaluate_feature(ex.id).volume()
    from cadcore.sketch import set_rect_width

    set_rect_width(skf.sketch.entities[0], 40.0)
    v1 = doc.evaluate_feature(ex.id).volume()
    assert v1 == pytest.approx(v0 * 2.0, rel=1e-6)


def test_edge_chamfer_reduces_volume():
    doc, skf, ex = _box_extrude()
    body = doc.evaluate_feature(ex.id)
    v0 = body.volume()
    edges = extract_convex_edges(body.vertices, body.faces)
    key = edges[0].key()
    ch = doc.create_edge_chamfer(ex.id, [key], 1.5)
    m = doc.evaluate_feature(ch.id)
    assert m is not None and m.is_watertight()
    assert m.volume() < v0
    assert ch.id in doc.evaluate_display_solids()
    assert ex.id not in doc.evaluate_display_solids()


def test_linear_pattern_triples_volume():
    doc, skf, ex = _box_extrude(size=10, depth=5)
    v0 = doc.evaluate_feature(ex.id).volume()
    lp = doc.create_linear_pattern(ex.id, 3, 15.0, 0.0, 0.0)
    m = doc.evaluate_feature(lp.id)
    assert m.volume() == pytest.approx(3 * v0, rel=1e-3)


def test_circular_pattern_four_instances():
    doc, skf, ex = _box_extrude(size=5, depth=3)
    # Place box away from origin so rotation about Z doesn't fully overlap
    # box is 0..5 in UV on front → world x,y; extrude z
    cp = doc.create_circular_pattern(ex.id, 4, total_angle_deg=360.0)
    m = doc.evaluate_feature(cp.id)
    assert m is not None and m.is_watertight()
    assert m.volume() > doc.evaluate_feature(ex.id).volume() * 2.5


def test_mirror_about_front_doubles_volume():
    doc, skf, ex = _box_extrude(size=10, depth=5)
    front = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    v0 = doc.evaluate_feature(ex.id).volume()
    mir = doc.create_mirror(ex.id, front.id)
    m = doc.evaluate_feature(mir.id)
    assert m.volume() == pytest.approx(2 * v0, rel=1e-3)


def test_offset_plane_frame():
    doc = Document()
    doc.seed_reference_planes()
    front = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    op = doc.create_offset_plane(front.id, 12.0)
    fr = doc.resolve_plane_frame(op)
    assert fr.origin[2] == pytest.approx(12.0)
    skf = doc.create_sketch_on_plane(op.id)
    assert skf is not None
    assert skf.sketch.frame.origin[2] == pytest.approx(12.0)


def test_construction_excluded_from_profiles():
    from cadcore.profiles import find_closed_line_loops

    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    a = sk.add_line((0, 0), (10, 0))
    b = sk.add_line((10, 0), (10, 10))
    c = sk.add_line((10, 10), (0, 10))
    d = sk.add_line((0, 10), (0, 0))
    assert len(find_closed_line_loops(sk)) == 1
    toggle_construction([a, b, c, d])
    assert len(find_closed_line_loops(sk)) == 0


def test_trim_and_offset_line():
    from cadcore.sketch_ops import trim_entity_at

    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    ln = sk.add_line((0, 0), (10, 0))
    sk.add_line((6, -2), (6, 2))
    assert trim_entity_at(sk, ln, (8.5, 0.0))
    ends = sorted([ln.p0[0], ln.p1[0]])
    assert ends[0] == pytest.approx(0.0)
    assert ends[1] == pytest.approx(6.0)  # cut at intersection, not cursor
    off = offset_line(sk.add_line((0, 0), (10, 0)), 2.0)
    assert off.p0[1] == pytest.approx(2.0)


def test_sw_trim_middle_splits_entity():
    from cadcore.sketch_ops import trim_entity_at

    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    h = sk.add_line((0, 0), (10, 0))
    sk.add_line((3, -1), (3, 1))
    sk.add_line((6, -1), (6, 1))
    n0 = len(sk.entities)
    assert trim_entity_at(sk, h, (5.0, 0.0))
    assert len(sk.entities) == n0 + 1
    horiz = [
        e
        for e in sk.entities
        if isinstance(e, type(h)) and abs(e.p0[1]) < 1e-9 and abs(e.p1[1]) < 1e-9
    ]
    assert len(horiz) == 2


def test_sw_extend_to_intersection():
    from cadcore.sketch_ops import extend_entity_at

    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    h = sk.add_line((-8, 0), (0, 0))
    sk.add_line((8, -2), (8, 2))
    assert extend_entity_at(sk, h, (-0.5, 0.0))
    ends = sorted([h.p0[0], h.p1[0]])
    assert ends[0] == pytest.approx(-8.0)
    assert ends[1] == pytest.approx(8.0)


def test_equal_radius_constraint():
    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    c1 = sk.add_circle((0, 0), 3)
    c2 = sk.add_circle((10, 0), 7)
    add_constraint(
        sk,
        SketchConstraint(
            id=-1, kind=ConstraintKind.EQUAL_RADIUS, e0=c1.id, e1=c2.id
        ),
    )
    assert c1.radius == pytest.approx(c2.radius)


def test_arc_chord_dimension():
    doc, skf, _ = _box_extrude()
    # use empty sketch
    skf2 = doc.create_sketch_on_plane(
        next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT).id
    )
    arc = skf2.sketch.add_arc((0, 0), (5, 4), (10, 0))
    chord0 = float(np.hypot(arc.p1()[0] - arc.p0()[0], arc.p1()[1] - arc.p0()[1]))
    doc.apply_sketch_dimension(skf2.id, arc.id, "chord", chord0 * 0.9)
    chord1 = float(np.hypot(arc.p1()[0] - arc.p0()[0], arc.p1()[1] - arc.p0()[1]))
    assert chord1 == pytest.approx(chord0 * 0.9, rel=1e-3)


def test_spline_add_and_sample():
    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    sp = sk.add_spline([(0, 0), (5, 3), (10, 0), (15, 2)])
    pts = sp.sample_uv()
    assert len(pts) >= 4
    assert entity_dof_status(sk, sp) in ("under", "well", "over")


def test_feature_param_edit_extrude():
    doc, skf, ex = _box_extrude()
    v0 = doc.evaluate_feature(ex.id).volume()
    assert doc.update_feature_params(ex.id, depth=20.0)
    assert doc.evaluate_feature(ex.id).volume() == pytest.approx(v0 * 2.0, rel=1e-6)
