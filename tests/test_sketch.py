import numpy as np
import pytest

from cadcore.document import Document, FeatureType
from cadcore.sketch import (
    CircleEntity,
    LineEntity,
    PlaneFrame,
    RectEntity,
    Sketch,
)


def test_plane_frame_front_roundtrip():
    fr = PlaneFrame.from_plane_type("PLANE_FRONT")
    for uv in [(0, 0), (1.5, -2.0), (3, 4)]:
        w = fr.to_world(uv)
        back = fr.to_local(w)
        assert abs(back[0] - uv[0]) < 1e-12
        assert abs(back[1] - uv[1]) < 1e-12
        # Front: world z ~ 0
        assert abs(w[2]) < 1e-12


def test_plane_frame_top_right_roundtrip():
    for name in ("PLANE_TOP", "PLANE_RIGHT"):
        fr = PlaneFrame.from_plane_type(name)
        uv = (1.25, -0.5)
        w = fr.to_world(uv)
        back = fr.to_local(w)
        assert abs(back[0] - uv[0]) < 1e-12
        assert abs(back[1] - uv[1]) < 1e-12


def test_ray_intersect_front():
    fr = PlaneFrame.from_plane_type("PLANE_FRONT")
    hit = fr.ray_intersect((0, 0, 5), (0, 0, -1))
    assert hit is not None
    assert np.allclose(hit, [0, 0, 0])
    uv = fr.to_local(hit)
    assert abs(uv[0]) < 1e-12 and abs(uv[1]) < 1e-12


def test_line_handles_and_midpoint_move():
    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    ln = sk.add_line((0, 0), (2, 0))
    hs = {h.name: h for h in ln.handles()}
    assert "p0" in hs and "p1" in hs and "mid" in hs
    assert abs(hs["mid"].uv[0] - 1.0) < 1e-12
    ln.set_handle("mid", (3, 1))
    assert abs(ln.midpoint()[0] - 3) < 1e-12
    assert abs(ln.midpoint()[1] - 1) < 1e-12
    # Endpoints moved by same delta
    assert abs(ln.p0[0] - 2) < 1e-12 and abs(ln.p0[1] - 1) < 1e-12
    assert abs(ln.p1[0] - 4) < 1e-12 and abs(ln.p1[1] - 1) < 1e-12


def test_circle_rim_handle():
    sk = Sketch()
    c = sk.add_circle((0, 0), 1.0)
    c.set_handle("rim", (0, 2))
    assert abs(c.radius - 2.0) < 1e-12
    c.set_handle("center", (1, 1))
    assert c.center == (1.0, 1.0)


def test_rect_corners():
    sk = Sketch()
    r = sk.add_rectangle((0, 0), (2, 1))
    cs = r.corners()
    assert len(cs) == 4
    assert (0.0, 0.0) in cs and (2.0, 1.0) in cs


def test_document_create_sketch():
    doc = Document()
    doc.seed_reference_planes()
    plane = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(plane.id)
    assert skf is not None
    assert skf.type is FeatureType.SKETCH
    assert skf.sketch is not None
    assert skf.sketch.plane_feature_id == plane.id
    skf.sketch.add_line((0, 0), (1, 0))
    assert len(skf.sketch.entities) == 1
