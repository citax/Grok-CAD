"""Extrude Reverse Direction: material on ±normal, same volume, watertight."""

from __future__ import annotations

import numpy as np
import pytest

from cadcore.document import Document, FeatureType
from cadcore.mesh import extrude_profile
from cadcore.sketch import CircleEntity, EntityKind, PlaneFrame, RectEntity


def _front_doc():
    doc = Document()
    doc.seed_reference_planes()
    plane = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(plane.id)
    return doc, skf


def _assert_sides(mesh_fwd, mesh_rev, *, n: np.ndarray, dist: float):
    """fwd on +n, rev on −n; volumes equal; boxes meet only at the plane."""
    n = np.asarray(n, float)
    n = n / np.linalg.norm(n)
    vf = mesh_fwd.vertices
    vr = mesh_rev.vertices
    # Project onto normal
    pf = vf @ n
    pr = vr @ n
    # Forward: material in [0, dist] (up to numeric eps)
    assert pf.min() >= -1e-6, pf.min()
    assert pf.max() <= dist + 1e-6, pf.max()
    assert pf.max() > dist * 0.5
    # Reversed: material in [-dist, 0]
    assert pr.max() <= 1e-6, pr.max()
    assert pr.min() >= -dist - 1e-6, pr.min()
    assert pr.min() < -dist * 0.5
    # Volumes match
    assert abs(mesh_fwd.volume() - mesh_rev.volume()) / max(abs(mesh_fwd.volume()), 1e-9) < 1e-6
    # Interiors do not overlap: open intervals (0,dist) and (-dist,0)
    # Shared plane face is OK — interior samples should not cross
    assert pf.min() > -1e-6 and pr.max() < 1e-6
    assert mesh_fwd.is_watertight() and mesh_rev.is_watertight()


def test_reverse_rectangle_front_plane():
    frame = PlaneFrame.from_plane_type("PLANE_FRONT")
    rect = RectEntity(id=1, kind=EntityKind.RECTANGLE, c0=(0, 0), c1=(2, 2))
    d = 2.0
    mf = extrude_profile(rect, d, frame, reversed=False)
    mr = extrude_profile(rect, d, frame, reversed=True)
    _assert_sides(mf, mr, n=frame.normal, dist=d)
    assert abs(mf.volume() - 8.0) < 1e-6


def test_reverse_circle_front_plane():
    frame = PlaneFrame.from_plane_type("PLANE_FRONT")
    circ = CircleEntity(id=1, kind=EntityKind.CIRCLE, center=(0, 0), radius=1.0)
    d = 1.5
    mf = extrude_profile(circ, d, frame, segments=48, reversed=False)
    mr = extrude_profile(circ, d, frame, segments=48, reversed=True)
    _assert_sides(mf, mr, n=frame.normal, dist=d)


def test_reverse_line_loop():
    from cadcore.profiles import ClosedLineLoop

    frame = PlaneFrame.from_plane_type("PLANE_FRONT")
    loop = ClosedLineLoop(
        vertices=((0.0, 0.0), (3.0, 0.0), (3.0, 1.0), (0.0, 1.0)),
        line_ids=(1, 2, 3, 4),
        id=-1,
    )
    d = 2.0
    mf = extrude_profile(loop, d, frame, reversed=False)
    mr = extrude_profile(loop, d, frame, reversed=True)
    _assert_sides(mf, mr, n=frame.normal, dist=d)
    assert abs(mf.volume() - 6.0) < 1e-5


def test_reverse_with_nested_hole():
    frame = PlaneFrame.from_plane_type("PLANE_FRONT")
    outer = RectEntity(id=1, kind=EntityKind.RECTANGLE, c0=(0, 0), c1=(4, 4))
    hole = CircleEntity(id=2, kind=EntityKind.CIRCLE, center=(2, 2), radius=0.5)
    d = 2.0
    mf = extrude_profile(outer, d, frame, holes=[hole], segments=48, reversed=False)
    mr = extrude_profile(outer, d, frame, holes=[hole], segments=48, reversed=True)
    _assert_sides(mf, mr, n=frame.normal, dist=d)
    # volume ≈ (16 - π*0.25) * 2
    exact = (16.0 - np.pi * 0.25) * d
    assert abs(mf.volume() - exact) / exact < 0.02


def test_document_create_extrude_reversed_and_default():
    doc, skf = _front_doc()
    skf.sketch.add_rectangle((0, 0), (2, 2))
    fwd = doc.create_extrude(skf.id, 2.0)
    assert fwd.reversed is False
    # New sketch for reverse (same plane)
    plane = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    sk2 = doc.create_sketch_on_plane(plane.id)
    sk2.sketch.add_rectangle((0, 0), (2, 2))
    rev = doc.create_extrude(sk2.id, 2.0, reversed=True)
    assert rev.reversed is True
    mf = doc.evaluate_feature(fwd.id)
    mr = doc.evaluate_feature(rev.id)
    assert mf is not None and mr is not None
    n = skf.sketch.frame.normal
    _assert_sides(mf, mr, n=n, dist=2.0)


def test_update_feature_params_reversed_undo():
    doc, skf = _front_doc()
    skf.sketch.add_rectangle((0, 0), (2, 2))
    ex = doc.create_extrude(skf.id, 2.0)
    m0 = doc.evaluate_feature(ex.id)
    assert m0 is not None
    z0 = m0.vertices[:, 2]
    assert z0.min() >= -1e-6 and z0.max() > 1.0
    assert doc.update_feature_params(ex.id, reversed=True)
    assert ex.reversed is True
    m1 = doc.evaluate_feature(ex.id)
    assert m1 is not None
    z1 = m1.vertices[:, 2]
    assert z1.max() <= 1e-6 and z1.min() < -1.0
    assert abs(m0.volume() - m1.volume()) < 1e-6
    assert doc.undo()
    assert ex.reversed is False
    m2 = doc.evaluate_feature(ex.id)
    assert m2 is not None
    assert m2.vertices[:, 2].min() >= -1e-6


def test_negative_depth_still_rejected():
    frame = PlaneFrame.from_plane_type("PLANE_FRONT")
    rect = RectEntity(id=1, kind=EntityKind.RECTANGLE, c0=(0, 0), c1=(1, 1))
    with pytest.raises(ValueError, match="positive"):
        extrude_profile(rect, -1.0, frame)
    doc, skf = _front_doc()
    skf.sketch.add_rectangle((0, 0), (1, 1))
    with pytest.raises(ValueError, match="positive"):
        doc.create_extrude(skf.id, -1.0)
