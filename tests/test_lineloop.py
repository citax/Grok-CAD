"""Closed line-segment loops as extrude profiles."""

from __future__ import annotations

import pytest

from cadcore.document import Document, FeatureType, resolve_profiles
from cadcore.mesh import extrude_profile
from cadcore.profiles import find_closed_line_loops, list_closed_profiles
from cadcore.sketch import PlaneFrame, Sketch


def _square_lines(sk: Sketch, L: float = 2.0) -> None:
    sk.add_line((0.0, 0.0), (L, 0.0))
    sk.add_line((L, 0.0), (L, L))
    sk.add_line((L, L), (0.0, L))
    sk.add_line((0.0, L), (0.0, 0.0))


def test_four_line_square_loop_detected():
    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    _square_lines(sk, 2.0)
    loops = find_closed_line_loops(sk)
    assert len(loops) == 1
    assert abs(loops[0].area() - 4.0) < 1e-9


def test_four_line_square_extrudes_volume_watertight():
    frame = PlaneFrame.from_plane_type("PLANE_FRONT")
    sk = Sketch(frame=frame)
    L, h = 2.0, 3.0
    _square_lines(sk, L)
    loop = find_closed_line_loops(sk)[0]
    mesh = extrude_profile(loop, h, frame)
    assert mesh.is_watertight()
    expect = L * L * h
    assert abs(mesh.volume() - expect) / expect < 0.01


def test_document_create_extrude_line_loop():
    doc = Document()
    doc.seed_reference_planes()
    front = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(front.id)
    _square_lines(skf.sketch, 1.5)
    feat = doc.create_extrude(skf.id, 2.0)
    mesh = doc.evaluate_feature(feat.id)
    assert mesh is not None and mesh.is_watertight()
    expect = 1.5 * 1.5 * 2.0
    assert abs(mesh.volume() - expect) / expect < 0.01


def test_open_chain_raises():
    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    sk.add_line((0, 0), (1, 0))
    sk.add_line((1, 0), (2, 1))
    with pytest.raises(ValueError, match="open chain"):
        list_closed_profiles(sk)


def test_self_intersecting_loop_raises():
    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    # Bow-tie / figure-eight-ish crossed quad
    sk.add_line((0, 0), (1, 1))
    sk.add_line((1, 1), (0, 1))
    sk.add_line((0, 1), (1, 0))
    sk.add_line((1, 0), (0, 0))
    with pytest.raises(ValueError, match="self-intersect"):
        find_closed_line_loops(sk)


def test_branching_raises():
    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    sk.add_line((0, 0), (1, 0))
    sk.add_line((1, 0), (2, 0))
    sk.add_line((1, 0), (1, 1))  # T-junction degree 3
    with pytest.raises(ValueError, match="branching"):
        find_closed_line_loops(sk)


def test_resolve_profiles_line_loop_outer():
    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    _square_lines(sk, 3.0)
    r = resolve_profiles(sk)
    from cadcore.profiles import ClosedLineLoop

    assert isinstance(r.outer, ClosedLineLoop)
    assert abs(r.outer.area() - 9.0) < 1e-9
    assert r.holes == []


def test_rect_still_preferred_when_present():
    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    sk.add_rectangle((0, 0), (2, 2))
    r = resolve_profiles(sk)
    from cadcore.sketch import RectEntity

    assert isinstance(r.outer, RectEntity)
