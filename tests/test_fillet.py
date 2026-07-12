"""Analytic unit tests for constant-radius profile fillet + extrude."""

import math
from pathlib import Path

import pytest

from cadcore.document import Document, FeatureType
from cadcore.mesh import (
    extrude_filleted_profile,
    fillet_profile,
    read_stl_binary,
    write_stl_binary,
)
from cadcore.sketch import (
    EntityKind,
    LineEntity,
    PlaneFrame,
    RectEntity,
)


def analytic_filleted_square_volume(L: float, r: float, h: float) -> float:
    """Volume of square side L, corner radius r, extruded height h.

    Area = L² − (4−π)r²  (each corner replaces r² with a quarter-disk πr²/4).
    """
    return (L * L - (4.0 - math.pi) * r * r) * h


def test_filleted_square_extrude_volume_analytic():
    frame = PlaneFrame.from_plane_type("PLANE_FRONT")
    L, r, h = 4.0, 0.5, 3.0
    rect = RectEntity(id=1, kind=EntityKind.RECTANGLE, c0=(0.0, 0.0), c1=(L, L))
    mesh = extrude_filleted_profile(rect, h, frame, r, segments=64)
    assert mesh.is_watertight()
    assert mesh.manifold_is_solid()
    exact = analytic_filleted_square_volume(L, r, h)
    rel = abs(mesh.volume() - exact) / exact
    assert rel < 0.01, f"volume={mesh.volume()} exact={exact} rel_err={rel}"


def test_filleted_square_extrude_watertight():
    frame = PlaneFrame.from_plane_type("PLANE_FRONT")
    rect = RectEntity(id=1, kind=EntityKind.RECTANGLE, c0=(0, 0), c1=(2, 2))
    mesh = extrude_filleted_profile(rect, 1.5, frame, 0.25, segments=32)
    assert mesh.is_watertight()
    assert mesh.manifold_is_solid()


def test_fillet_stl_roundtrip(tmp_path: Path):
    frame = PlaneFrame.from_plane_type("PLANE_FRONT")
    L, r, h = 3.0, 0.4, 2.0
    rect = RectEntity(id=1, kind=EntityKind.RECTANGLE, c0=(0, 0), c1=(L, L))
    mesh = extrude_filleted_profile(rect, h, frame, r, segments=48)
    path = tmp_path / "fillet.stl"
    write_stl_binary(mesh, path)
    loaded = read_stl_binary(path)
    assert len(loaded.faces) == len(mesh.faces)
    assert abs(loaded.volume() - mesh.volume()) / max(abs(mesh.volume()), 1e-9) < 1e-3


def test_fillet_rejects_nonpositive_radius():
    rect = RectEntity(id=1, kind=EntityKind.RECTANGLE, c0=(0, 0), c1=(2, 2))
    with pytest.raises(ValueError, match=r"radius.*positive|radius <= 0"):
        fillet_profile(rect, 0.0, segments=16)
    with pytest.raises(ValueError, match=r"radius.*positive|radius <= 0"):
        fillet_profile(rect, -0.5, segments=16)


def test_fillet_rejects_radius_too_large():
    rect = RectEntity(id=1, kind=EntityKind.RECTANGLE, c0=(0, 0), c1=(2, 2))
    # half-side = 1.0; r >= 1.0 is too large
    with pytest.raises(ValueError, match=r"too large|self-intersection"):
        fillet_profile(rect, 1.0, segments=16)
    with pytest.raises(ValueError, match=r"too large|self-intersection"):
        fillet_profile(rect, 1.5, segments=16)


def test_fillet_rejects_open_profile():
    line = LineEntity(id=1, kind=EntityKind.LINE, p0=(0, 0), p1=(1, 0))
    with pytest.raises(ValueError, match=r"open profile"):
        fillet_profile(line, 0.1, segments=16)


def test_document_create_fillet_volume():
    doc = Document()
    doc.seed_reference_planes()
    plane = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(plane.id)
    assert skf and skf.sketch
    L, r, h = 4.0, 0.5, 2.0
    skf.sketch.add_rectangle((0, 0), (L, L))
    feat = doc.create_fillet(skf.id, h, r, segments=64)
    assert feat.type is FeatureType.FILLET
    mesh = doc.evaluate_feature(feat.id)
    assert mesh is not None
    assert mesh.is_watertight()
    exact = analytic_filleted_square_volume(L, r, h)
    assert abs(mesh.volume() - exact) / exact < 0.01
    assert feat.id in doc.evaluate_display_solids()


def test_document_fillet_rejects_bad_radius():
    doc = Document()
    doc.seed_reference_planes()
    plane = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(plane.id)
    assert skf and skf.sketch
    skf.sketch.add_rectangle((0, 0), (2, 2))
    with pytest.raises(ValueError, match=r"radius"):
        doc.create_fillet(skf.id, 1.0, 0.0)
    with pytest.raises(ValueError, match=r"too large|self-intersection"):
        doc.create_fillet(skf.id, 1.0, 1.5)


def test_no_gui_imports_in_fillet_path():
    import cadcore.document as d
    import cadcore.mesh as m

    for mod in (d, m):
        src = open(mod.__file__, encoding="utf-8").read()
        assert "PySide" not in src
        assert "pyvista" not in src
