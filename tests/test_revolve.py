"""Unit tests for revolve (revolution) — cadcore only, no GUI imports."""

import math

import pytest

from cadcore.document import Document, FeatureType
from cadcore.mesh import revolve_circle, revolve_profile, revolve_rectangle
from cadcore.sketch import (
    CircleEntity,
    EntityKind,
    LineEntity,
    PlaneFrame,
    RectEntity,
)


def test_revolve_rectangle_pappus_volume_watertight():
    """Rectangle offset from V-axis, full 360° → Pappus theorem volume."""
    frame = PlaneFrame.from_plane_type("PLANE_FRONT")
    # u=1..2, v=0..1 → area=1, centroid radius=1.5 about V-axis
    mesh = revolve_rectangle(
        (1.0, 0.0),
        (2.0, 1.0),
        frame,
        axis_origin=(0.0, 0.0),
        axis_direction=(0.0, 1.0),
        angle_degrees=360.0,
        segments=96,
    )
    assert mesh.is_watertight()
    assert mesh.manifold_is_solid()
    area = 1.0
    r_bar = 1.5
    pappus = area * 2.0 * math.pi * r_bar  # ≈ 9.4248
    assert abs(mesh.volume() - pappus) / pappus < 0.02


def test_revolve_rectangle_half_angle():
    frame = PlaneFrame.from_plane_type("PLANE_FRONT")
    full = revolve_rectangle((1, 0), (2, 1), frame, angle_degrees=360.0, segments=64)
    half = revolve_rectangle((1, 0), (2, 1), frame, angle_degrees=180.0, segments=64)
    assert half.is_watertight()
    assert abs(half.volume() - 0.5 * full.volume()) / full.volume() < 0.03


def test_revolve_circle_torus_like_watertight():
    """Circle in UV revolved about V-axis → torus (Pappus)."""
    frame = PlaneFrame.from_plane_type("PLANE_FRONT")
    # tube radius 0.25, center at u=1.5
    R, r = 1.5, 0.25
    mesh = revolve_circle(
        (R, 0.0),
        r,
        frame,
        angle_degrees=360.0,
        segments=64,
        profile_segments=48,
    )
    assert mesh.is_watertight()
    assert mesh.manifold_is_solid()
    # torus volume = 2 π² R r²
    exact = 2.0 * math.pi**2 * R * r * r
    assert abs(mesh.volume() - exact) / exact < 0.05


def test_revolve_profile_entity():
    frame = PlaneFrame.from_plane_type("PLANE_TOP")
    rect = RectEntity(id=1, kind=EntityKind.RECTANGLE, c0=(0.5, -0.5), c1=(1.5, 0.5))
    mesh = revolve_profile(rect, frame, angle_degrees=360.0, segments=48)
    assert mesh.is_watertight()
    area = 1.0 * 1.0
    r_bar = 1.0  # centroid u=1
    pappus = area * 2.0 * math.pi * r_bar
    assert abs(mesh.volume() - pappus) / pappus < 0.03


def test_revolve_rejects_axis_crossing():
    frame = PlaneFrame.from_plane_type("PLANE_FRONT")
    with pytest.raises(ValueError, match="cross"):
        revolve_rectangle((-1.0, 0.0), (1.0, 1.0), frame, angle_degrees=360.0)


def test_revolve_rejects_zero_angle():
    frame = PlaneFrame.from_plane_type("PLANE_FRONT")
    with pytest.raises(ValueError, match="angle"):
        revolve_rectangle((1.0, 0.0), (2.0, 1.0), frame, angle_degrees=0.0)
    with pytest.raises(ValueError, match="angle"):
        revolve_rectangle((1.0, 0.0), (2.0, 1.0), frame, angle_degrees=-90.0)


def test_revolve_rejects_open_line():
    frame = PlaneFrame.from_plane_type("PLANE_FRONT")
    line = LineEntity(id=1, kind=EntityKind.LINE, p0=(1, 0), p1=(2, 0))
    with pytest.raises(ValueError, match="open|closed|line"):
        revolve_profile(line, frame)


def test_revolve_rejects_zero_area():
    frame = PlaneFrame.from_plane_type("PLANE_FRONT")
    with pytest.raises(ValueError, match="degenerate|zero"):
        revolve_rectangle((1.0, 0.0), (1.0, 2.0), frame)


def test_document_create_revolve():
    doc = Document()
    doc.seed_reference_planes()
    plane = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(plane.id)
    assert skf is not None and skf.sketch is not None
    skf.sketch.add_rectangle((1.0, 0.0), (2.0, 1.0))
    rev = doc.create_revolve(skf.id, angle_degrees=360.0, segments=64)
    assert rev.type is FeatureType.REVOLVE
    mesh = doc.evaluate_feature(rev.id)
    assert mesh is not None
    assert mesh.is_watertight()
    pappus = 1.0 * 2.0 * math.pi * 1.5
    assert abs(mesh.volume() - pappus) / pappus < 0.02
    assert rev.id in doc.evaluate_display_solids()


def test_document_revolve_rejects_crossing():
    doc = Document()
    doc.seed_reference_planes()
    plane = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(plane.id)
    assert skf is not None and skf.sketch is not None
    skf.sketch.add_rectangle((-1.0, 0.0), (1.0, 1.0))
    with pytest.raises(ValueError, match="cross"):
        doc.create_revolve(skf.id)


def test_document_revolve_rejects_zero_angle():
    doc = Document()
    doc.seed_reference_planes()
    plane = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(plane.id)
    assert skf is not None and skf.sketch is not None
    skf.sketch.add_rectangle((1.0, 0.0), (2.0, 1.0))
    with pytest.raises(ValueError, match="angle"):
        doc.create_revolve(skf.id, angle_degrees=0.0)


def test_no_gui_imports_in_cadcore_revolve():
    import cadcore.document as d
    import cadcore.mesh as m

    for mod in (d, m):
        src = open(mod.__file__, encoding="utf-8").read()
        assert "PySide" not in src
        assert "pyvista" not in src
        assert "PySide6" not in src
