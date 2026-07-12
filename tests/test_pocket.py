"""Analytic unit tests for through-hole pocket + extrude."""

import math
from pathlib import Path

import pytest

from cadcore.document import Document, FeatureType
from cadcore.mesh import (
    extrude_pocketed_profile,
    profile_with_hole,
    read_stl_binary,
    write_stl_binary,
)
from cadcore.sketch import EntityKind, LineEntity, PlaneFrame, RectEntity


def analytic_pocket_volume(L: float, r: float, h: float) -> float:
    return (L * L - math.pi * r * r) * h


def test_pocket_square_volume_analytic():
    frame = PlaneFrame.from_plane_type("PLANE_FRONT")
    L, r, h = 10.0, 2.0, 5.0
    rect = RectEntity(id=1, kind=EntityKind.RECTANGLE, c0=(0, 0), c1=(L, L))
    mesh = extrude_pocketed_profile(
        rect, h, frame, (L * 0.5, L * 0.5), r, segments=64
    )
    assert mesh.is_watertight()
    assert mesh.manifold_is_solid()
    exact = analytic_pocket_volume(L, r, h)
    rel = abs(mesh.volume() - exact) / exact
    assert rel < 0.01, f"vol={mesh.volume()} exact={exact} rel={rel}"


def test_pocket_watertight():
    frame = PlaneFrame.from_plane_type("PLANE_FRONT")
    rect = RectEntity(id=1, kind=EntityKind.RECTANGLE, c0=(-2, -2), c1=(2, 2))
    mesh = extrude_pocketed_profile(rect, 1.0, frame, (0.0, 0.0), 0.75, segments=32)
    assert mesh.is_watertight()
    assert mesh.manifold_is_solid()


def test_pocket_stl_roundtrip(tmp_path: Path):
    frame = PlaneFrame.from_plane_type("PLANE_FRONT")
    L, r, h = 6.0, 1.0, 2.0
    rect = RectEntity(id=1, kind=EntityKind.RECTANGLE, c0=(0, 0), c1=(L, L))
    mesh = extrude_pocketed_profile(
        rect, h, frame, (L * 0.5, L * 0.5), r, segments=48
    )
    path = tmp_path / "pocket.stl"
    write_stl_binary(mesh, path)
    loaded = read_stl_binary(path)
    assert len(loaded.faces) == len(mesh.faces)
    assert abs(loaded.volume() - mesh.volume()) / max(abs(mesh.volume()), 1e-9) < 1e-3


def test_pocket_rejects_nonpositive_hole_radius():
    rect = RectEntity(id=1, kind=EntityKind.RECTANGLE, c0=(0, 0), c1=(4, 4))
    with pytest.raises(ValueError, match=r"hole radius.*positive|hole_radius <= 0"):
        profile_with_hole(rect, (2, 2), 0.0, segments=16)
    with pytest.raises(ValueError, match=r"hole radius.*positive|hole_radius <= 0"):
        profile_with_hole(rect, (2, 2), -1.0, segments=16)


def test_pocket_rejects_hole_outside_bounds():
    rect = RectEntity(id=1, kind=EntityKind.RECTANGLE, c0=(0, 0), c1=(4, 4))
    # center near edge so circle extends outside
    with pytest.raises(ValueError, match=r"outside the profile bounds"):
        profile_with_hole(rect, (3.5, 2.0), 1.0, segments=16)
    # completely outside
    with pytest.raises(ValueError, match=r"outside the profile bounds"):
        profile_with_hole(rect, (10.0, 10.0), 0.5, segments=16)


def test_pocket_rejects_hole_touching_edge():
    rect = RectEntity(id=1, kind=EntityKind.RECTANGLE, c0=(0, 0), c1=(4, 4))
    # circle of r=1 centered at (1, 2) touches x=0
    with pytest.raises(ValueError, match=r"non-manifold|touches an edge"):
        profile_with_hole(rect, (1.0, 2.0), 1.0, segments=16)


def test_document_create_pocket():
    doc = Document()
    doc.seed_reference_planes()
    plane = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(plane.id)
    assert skf and skf.sketch
    L, r, h = 8.0, 1.5, 3.0
    skf.sketch.add_rectangle((0, 0), (L, L))
    feat = doc.create_pocket(
        skf.id, h, r, (L * 0.5, L * 0.5), segments=48
    )
    assert feat.type is FeatureType.POCKET
    mesh = doc.evaluate_feature(feat.id)
    assert mesh is not None and mesh.is_watertight()
    exact = analytic_pocket_volume(L, r, h)
    assert abs(mesh.volume() - exact) / exact < 0.01
    assert feat.id in doc.evaluate_display_solids()


def test_document_pocket_rejects_bad_hole():
    doc = Document()
    doc.seed_reference_planes()
    plane = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(plane.id)
    assert skf and skf.sketch
    skf.sketch.add_rectangle((0, 0), (4, 4))
    with pytest.raises(ValueError, match=r"hole radius"):
        doc.create_pocket(skf.id, 1.0, 0.0, (2, 2))
    with pytest.raises(ValueError, match=r"outside|bounds"):
        doc.create_pocket(skf.id, 1.0, 1.0, (3.5, 2.0))
