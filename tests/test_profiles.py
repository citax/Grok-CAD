"""Nested / disjoint closed-profile resolution for extrude."""

import math
from pathlib import Path

import pytest

from cadcore.document import (
    Document,
    FeatureType,
    resolve_profiles,
)
from cadcore.mesh import extrude_profile, read_stl_binary, write_stl_binary
from cadcore.sketch import PlaneFrame, Sketch


def test_nested_square_circle_extrude_volume():
    frame = PlaneFrame.from_plane_type("PLANE_FRONT")
    sk = Sketch(frame=frame)
    L, r, h = 10.0, 2.0, 4.0
    outer = sk.add_rectangle((0, 0), (L, L))
    sk.add_circle((L * 0.5, L * 0.5), r)
    resolved = resolve_profiles(sk)
    assert resolved.outer.id == outer.id
    assert len(resolved.holes) == 1
    mesh = extrude_profile(
        resolved.outer, h, frame, segments=64, holes=resolved.holes
    )
    assert mesh.is_watertight()
    exact = (L * L - math.pi * r * r) * h
    rel = abs(mesh.volume() - exact) / exact
    assert rel < 0.01, f"vol={mesh.volume()} exact={exact} rel={rel}"


def test_nested_document_create_extrude():
    doc = Document()
    doc.seed_reference_planes()
    plane = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(plane.id)
    assert skf and skf.sketch
    L, r, h = 8.0, 1.5, 3.0
    skf.sketch.add_rectangle((0, 0), (L, L))
    skf.sketch.add_circle((L * 0.5, L * 0.5), r)
    feat = doc.create_extrude(skf.id, h, segments=48)
    mesh = doc.evaluate_feature(feat.id)
    assert mesh is not None and mesh.is_watertight()
    exact = (L * L - math.pi * r * r) * h
    assert abs(mesh.volume() - exact) / exact < 0.01


def test_nested_stl_roundtrip(tmp_path: Path):
    frame = PlaneFrame.from_plane_type("PLANE_FRONT")
    sk = Sketch(frame=frame)
    L, r, h = 6.0, 1.0, 2.0
    sk.add_rectangle((0, 0), (L, L))
    sk.add_circle((3, 3), r)
    res = resolve_profiles(sk)
    mesh = extrude_profile(res.outer, h, frame, segments=48, holes=res.holes)
    path = tmp_path / "nested.stl"
    write_stl_binary(mesh, path)
    loaded = read_stl_binary(path)
    assert len(loaded.faces) == len(mesh.faces)
    assert abs(loaded.volume() - mesh.volume()) / max(abs(mesh.volume()), 1e-9) < 1e-3


def test_disjoint_profiles_raise():
    sk = Sketch()
    sk.add_rectangle((0, 0), (2, 2))
    sk.add_rectangle((5, 5), (7, 7))
    with pytest.raises(ValueError, match=r"ambiguous|disjoint"):
        resolve_profiles(sk)


def test_single_profile_unchanged():
    frame = PlaneFrame.from_plane_type("PLANE_FRONT")
    sk = Sketch(frame=frame)
    sk.add_rectangle((0, 0), (3, 2))
    res = resolve_profiles(sk)
    assert len(res.holes) == 0
    mesh = extrude_profile(res.outer, 1.5, frame)
    assert abs(mesh.volume() - 9.0) < 1e-6
    assert mesh.is_watertight()


def test_document_single_profile_compat():
    doc = Document()
    doc.seed_reference_planes()
    plane = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(plane.id)
    skf.sketch.add_rectangle((0, 0), (2, 3))
    feat = doc.create_extrude(skf.id, 4.0)
    mesh = doc.evaluate_feature(feat.id)
    assert mesh is not None
    assert abs(mesh.volume() - 24.0) < 1e-6
