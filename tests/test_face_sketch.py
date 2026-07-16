"""Sketch on solid face: plane frame, extrude direction, document API."""

from __future__ import annotations

import numpy as np
import pytest

from cadcore.document import Document, FeatureType, is_solid_feature
from cadcore.faces import closest_triangle, plane_frame_from_face, triangle_normal
from cadcore.mesh import make_box
from cadcore.sketch import PlaneFrame


def test_is_solid_feature():
    assert is_solid_feature(FeatureType.EXTRUDE)
    assert not is_solid_feature(FeatureType.PLANE_FRONT)
    assert not is_solid_feature(FeatureType.SKETCH)


def test_plane_frame_outward_on_box_top():
    # make_box is axis-aligned; top face is +Y for standard box at origin?
    m = make_box(40.0, 20.0, 10.0)
    # Pick the vertex-max face along each axis by sampling high-Y points
    ymax = float(m.vertices[:, 1].max())
    # Find a triangle near ymax
    pick = np.array([20.0, ymax, 5.0])
    fr = plane_frame_from_face(m.vertices, m.faces, pick)
    # Normal should point outward (+Y-ish for a top face) or at least away from centroid
    centroid = m.vertices.mean(axis=0)
    assert float(np.dot(fr.normal, fr.origin - centroid)) > 0
    # Pick point projects onto plane
    dist = abs(float(np.dot(fr.normal, pick - fr.origin)))
    assert dist < 1e-6


def test_front_extrude_top_face_sketch_and_boss():
    """Real workflow: sketch rect → extrude → sketch on top face → boss up."""
    doc = Document()
    doc.seed_reference_planes()
    front = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    sk1 = doc.create_sketch_on_plane(front.id)
    assert sk1 is not None and sk1.sketch is not None
    sk1.sketch.add_rectangle((0.0, 0.0), (40.0, 30.0))
    ex1 = doc.create_extrude(sk1.id, 15.0)
    mesh1 = doc.evaluate_feature(ex1.id)
    assert mesh1 is not None and mesh1.is_watertight()
    # Top of front-plane extrude is +Z
    zmax = float(mesh1.vertices[:, 2].max())
    assert zmax == pytest.approx(15.0, abs=1e-5)
    pick = np.array([20.0, 15.0, zmax])
    fr = plane_frame_from_face(mesh1.vertices, mesh1.faces, pick)
    assert abs(float(fr.normal[2]) - 1.0) < 1e-6
    assert abs(float(fr.origin[2]) - zmax) < 1e-5

    sk2 = doc.create_sketch_on_face(ex1.id, fr)
    assert sk2 is not None and sk2.sketch is not None
    assert sk2.plane_id == ex1.id
    # UV on face: small boss footprint
    sk2.sketch.add_rectangle((-5.0, -5.0), (5.0, 5.0))
    ex2 = doc.create_extrude(sk2.id, 10.0)
    mesh2 = doc.evaluate_feature(ex2.id)
    assert mesh2 is not None
    zs = mesh2.vertices[:, 2]
    assert zs.min() == pytest.approx(15.0, abs=1e-4)
    assert zs.max() == pytest.approx(25.0, abs=1e-4)
    # Volume of 10×10×10 boss
    assert abs(mesh2.volume() - 1000.0) < 1e-3


def test_create_sketch_on_face_rejects_plane():
    doc = Document()
    doc.seed_reference_planes()
    fr = PlaneFrame.from_plane_type("PLANE_FRONT")
    assert doc.create_sketch_on_face(doc.features[0].id, fr) is None


def test_closest_triangle_and_normal():
    m = make_box(10, 10, 10)
    i = closest_triangle(m.vertices, m.faces, [5, 5, 10])
    assert i >= 0
    f = m.faces[i]
    n = triangle_normal(m.vertices[f[0]], m.vertices[f[1]], m.vertices[f[2]])
    assert abs(float(np.linalg.norm(n)) - 1.0) < 1e-9
