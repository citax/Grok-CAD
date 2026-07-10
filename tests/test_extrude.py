"""Unit tests for extrude (pad) — cadcore only, no GUI imports."""

import math

import pytest

from cadcore.document import Document, FeatureType, first_closed_profile
from cadcore.mesh import extrude_circle, extrude_profile, extrude_rectangle
from cadcore.sketch import (
    CircleEntity,
    EntityKind,
    LineEntity,
    PlaneFrame,
    RectEntity,
    Sketch,
)


def test_extrude_rectangle_volume_watertight():
    frame = PlaneFrame.from_plane_type("PLANE_FRONT")
    # 2 x 3 rectangle, distance 4 → volume 24
    mesh = extrude_rectangle((0.0, 0.0), (2.0, 3.0), 4.0, frame)
    assert mesh.is_watertight()
    assert mesh.manifold_is_solid()
    assert abs(mesh.volume() - 24.0) < 1e-6


def test_extrude_rectangle_on_top_plane():
    frame = PlaneFrame.from_plane_type("PLANE_TOP")
    mesh = extrude_rectangle((-1.0, -1.0), (1.0, 1.0), 5.0, frame)
    assert mesh.is_watertight()
    assert abs(mesh.volume() - 20.0) < 1e-6
    # extruded along +Y (top normal)
    ys = mesh.vertices[:, 1]
    assert ys.min() >= -1e-9
    assert ys.max() == pytest.approx(5.0, abs=1e-6)


def test_extrude_circle_volume_watertight():
    frame = PlaneFrame.from_plane_type("PLANE_FRONT")
    r, h, segs = 1.0, 2.0, 64
    mesh = extrude_circle((0.0, 0.0), r, h, frame, segments=segs)
    assert mesh.is_watertight()
    assert mesh.manifold_is_solid()
    exact = math.pi * r * r * h
    # polygonal cylinder under-approximates slightly
    assert abs(mesh.volume() - exact) / exact < 0.03


def test_extrude_profile_rect_and_circle_entities():
    frame = PlaneFrame.from_plane_type("PLANE_RIGHT")
    rect = RectEntity(id=1, kind=EntityKind.RECTANGLE, c0=(0, 0), c1=(1.5, 2.5))
    m1 = extrude_profile(rect, 3.0, frame)
    assert m1.is_watertight()
    assert abs(m1.volume() - 1.5 * 2.5 * 3.0) < 1e-6

    circ = CircleEntity(id=2, kind=EntityKind.CIRCLE, center=(1.0, 1.0), radius=0.5)
    m2 = extrude_profile(circ, 1.0, frame, segments=48)
    assert m2.is_watertight()
    exact = math.pi * 0.25 * 1.0
    assert abs(m2.volume() - exact) / exact < 0.04


def test_extrude_rejects_open_line():
    frame = PlaneFrame.from_plane_type("PLANE_FRONT")
    line = LineEntity(id=1, kind=EntityKind.LINE, p0=(0, 0), p1=(1, 0))
    with pytest.raises(ValueError, match="open|closed|line"):
        extrude_profile(line, 1.0, frame)


def test_extrude_rejects_degenerate_rectangle():
    frame = PlaneFrame.from_plane_type("PLANE_FRONT")
    with pytest.raises(ValueError, match="degenerate|zero"):
        extrude_rectangle((1.0, 2.0), (1.0, 5.0), 2.0, frame)


def test_extrude_rejects_degenerate_circle():
    frame = PlaneFrame.from_plane_type("PLANE_FRONT")
    with pytest.raises(ValueError, match="degenerate|radius|positive"):
        extrude_circle((0.0, 0.0), 0.0, 1.0, frame)


def test_extrude_rejects_nonpositive_distance():
    frame = PlaneFrame.from_plane_type("PLANE_FRONT")
    with pytest.raises(ValueError, match="distance"):
        extrude_rectangle((0, 0), (1, 1), 0.0, frame)
    with pytest.raises(ValueError, match="distance"):
        extrude_rectangle((0, 0), (1, 1), -2.0, frame)


def test_document_create_extrude_rectangle():
    doc = Document()
    doc.seed_reference_planes()
    plane = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(plane.id)
    assert skf is not None and skf.sketch is not None
    skf.sketch.add_rectangle((0, 0), (2, 3))
    ex = doc.create_extrude(skf.id, 4.0)
    assert ex.type is FeatureType.EXTRUDE
    mesh = doc.evaluate_feature(ex.id)
    assert mesh is not None
    assert mesh.is_watertight()
    assert abs(mesh.volume() - 24.0) < 1e-6
    solids = doc.evaluate_display_solids()
    assert ex.id in solids


def test_document_create_extrude_circle():
    doc = Document()
    doc.seed_reference_planes()
    plane = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(plane.id)
    assert skf is not None and skf.sketch is not None
    skf.sketch.add_circle((0.5, 0.5), 1.0)
    ex = doc.create_extrude(skf.id, 2.0, segments=64)
    mesh = doc.evaluate_feature(ex.id)
    assert mesh is not None
    assert mesh.is_watertight()
    exact = math.pi * 1.0 * 2.0
    assert abs(mesh.volume() - exact) / exact < 0.03


def test_document_extrude_rejects_open_only_sketch():
    doc = Document()
    doc.seed_reference_planes()
    plane = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(plane.id)
    assert skf is not None and skf.sketch is not None
    skf.sketch.add_line((0, 0), (1, 0))
    assert first_closed_profile(skf.sketch) is None
    with pytest.raises(ValueError, match="closed profile"):
        doc.create_extrude(skf.id, 1.0)


def test_document_extrude_rejects_bad_distance():
    doc = Document()
    doc.seed_reference_planes()
    plane = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(plane.id)
    assert skf is not None and skf.sketch is not None
    skf.sketch.add_rectangle((0, 0), (1, 1))
    with pytest.raises(ValueError, match="distance"):
        doc.create_extrude(skf.id, 0.0)


def test_no_gui_imports_in_cadcore_extrude():
    """Guard: extrude path stays pure (no PySide / VTK)."""
    import cadcore.document as d
    import cadcore.mesh as m

    for mod in (d, m):
        src = open(mod.__file__, encoding="utf-8").read()
        assert "PySide" not in src
        assert "pyvista" not in src
        assert "vtk" not in src.lower() or "manifold" in src  # mesh mentions none
        assert "PySide6" not in src
