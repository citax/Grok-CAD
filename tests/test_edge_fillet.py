"""Solid edge fillet: volume drop, watertight, undo, save/load, failure."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from cadcore.document import Document, FeatureType, is_solid_feature
from cadcore.edge_fillet import extract_convex_edges
from cadcore.project_io import load_document, save_document


def _box_extrude(doc: Document, size: float = 30.0, depth: float = 30.0):
    plane = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    sk = doc.create_sketch_on_plane(plane.id)
    sk.sketch.add_rectangle((0.0, 0.0), (size, size))
    ex = doc.create_extrude(sk.id, depth)
    return ex


def test_edge_fillet_reduces_volume_and_stays_watertight():
    doc = Document()
    doc.seed_reference_planes()
    ex = _box_extrude(doc)
    body = doc.evaluate_feature(ex.id)
    assert body is not None
    v0 = body.volume()
    assert body.is_watertight()
    edges = extract_convex_edges(body.vertices, body.faces)
    assert len(edges) >= 1
    key = edges[0].key()
    r = 3.0
    fl = doc.create_edge_fillet(ex.id, [key], r)
    assert fl.type is FeatureType.EDGE_FILLET
    out = doc.evaluate_feature(fl.id)
    assert out is not None
    assert out.is_watertight()
    v1 = out.volume()
    assert v1 < v0 - 1.0  # removed material
    # ~ L r^2 (1 - pi/4) for one 90° edge of length depth
    expected = 30.0 * r * r * (1.0 - np.pi / 4.0)
    assert abs((v0 - v1) - expected) < 5.0  # segment approximation


def test_edge_fillet_radius_edit_and_undo():
    doc = Document()
    doc.seed_reference_planes()
    ex = _box_extrude(doc)
    body = doc.evaluate_feature(ex.id)
    key = extract_convex_edges(body.vertices, body.faces)[0].key()
    fl = doc.create_edge_fillet(ex.id, [key], 2.0)
    v_r2 = doc.evaluate_feature(fl.id).volume()
    assert doc.update_feature_params(fl.id, radius=4.0)
    v_r4 = doc.evaluate_feature(fl.id).volume()
    assert v_r4 < v_r2  # larger radius removes more
    assert doc.undo()  # undo radius edit
    assert fl.radius == pytest.approx(2.0)
    assert doc.evaluate_feature(fl.id).volume() == pytest.approx(v_r2, rel=1e-5)
    assert doc.undo()  # undo create
    assert doc.find(fl.id) is None
    # parent solid restored in display evaluation
    solids = doc.evaluate_display_solids()
    assert ex.id in solids


def test_edge_fillet_save_reload():
    doc = Document()
    doc.seed_reference_planes()
    ex = _box_extrude(doc)
    body = doc.evaluate_feature(ex.id)
    key = extract_convex_edges(body.vertices, body.faces)[0].key()
    fl = doc.create_edge_fillet(ex.id, [key], 2.5)
    v0 = doc.evaluate_feature(fl.id).volume()
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "fillet_part.gcad"
        save_document(doc, path)
        loaded = load_document(path)
    fl2 = next(f for f in loaded.features if f.type is FeatureType.EDGE_FILLET)
    assert fl2.radius == pytest.approx(2.5)
    assert fl2.edge_keys == [key]
    assert fl2.operand_a == ex.id
    mesh = loaded.evaluate_feature(fl2.id)
    assert mesh is not None and mesh.is_watertight()
    assert mesh.volume() == pytest.approx(v0, rel=1e-4)
    # still editable
    assert loaded.update_feature_params(fl2.id, radius=3.0)
    assert loaded.evaluate_feature(fl2.id).volume() < v0


def test_edge_fillet_failure_leaves_part_unchanged():
    doc = Document()
    doc.seed_reference_planes()
    ex = _box_extrude(doc)
    body = doc.evaluate_feature(ex.id)
    v0 = body.volume()
    n_feat = len(doc.features)
    key = extract_convex_edges(body.vertices, body.faces)[0].key()
    with pytest.raises(ValueError, match="too large|cannot|invalid|not"):
        doc.create_edge_fillet(ex.id, [key], 100.0)
    assert len(doc.features) == n_feat
    assert doc.evaluate_feature(ex.id).volume() == pytest.approx(v0)
    # empty selection
    with pytest.raises(ValueError, match="at least one edge"):
        doc.create_edge_fillet(ex.id, [], 2.0)
    assert len(doc.features) == n_feat


def test_edge_fillet_display_hides_parent():
    doc = Document()
    doc.seed_reference_planes()
    ex = _box_extrude(doc)
    body = doc.evaluate_feature(ex.id)
    key = extract_convex_edges(body.vertices, body.faces)[0].key()
    fl = doc.create_edge_fillet(ex.id, [key], 2.0)
    solids = doc.evaluate_display_solids()
    assert fl.id in solids
    assert ex.id not in solids
    assert is_solid_feature(fl.type)
