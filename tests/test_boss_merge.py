"""Boss-extrude merges into the parent solid (one watertight body)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from cadcore.document import Document, FeatureType, resolve_profiles
from cadcore.faces import plane_frame_from_face
from cadcore.mesh import BooleanOp, boolean_op, extrude_profile
from cadcore.project_io import load_document, save_document


def _base_box(doc: Document, w=40.0, h=30.0, d=15.0):
    front = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    sk = doc.create_sketch_on_plane(front.id)
    sk.sketch.add_rectangle((0.0, 0.0), (w, h))
    ex = doc.create_extrude(sk.id, d)
    return ex, w * h * d


def test_boss_merge_single_body_and_volume():
    doc = Document()
    doc.seed_reference_planes()
    ex1, v_base = _base_box(doc)
    m1 = doc.evaluate_feature(ex1.id)
    assert m1 is not None and m1.is_watertight()
    zmax = float(m1.vertices[:, 2].max())
    fr = plane_frame_from_face(m1.vertices, m1.faces, [20.0, 15.0, zmax])
    sk2 = doc.create_sketch_on_face(ex1.id, fr)
    sk2.sketch.add_rectangle((-5.0, -5.0), (5.0, 5.0))
    ex2 = doc.create_extrude(sk2.id, 10.0)
    assert ex2.operand_b == ex1.id
    m2 = doc.evaluate_feature(ex2.id)
    assert m2 is not None and m2.is_watertight()
    v_boss = 10.0 * 10.0 * 10.0
    assert m2.volume() == pytest.approx(v_base + v_boss, abs=0.05)
    display = doc.evaluate_display_solids()
    assert list(display.keys()) == [ex2.id]


def test_boss_overlap_volume_not_double_counted():
    """Reverse boss into the solid: tool fully inside base → union vol == base.

    That is the double-counting trap: V_union must be V_base + V_tool - V_overlap
    with V_overlap ≈ V_tool (contained), not V_base + V_tool.
    """
    doc = Document()
    doc.seed_reference_planes()
    ex1, v_base = _base_box(doc, w=40.0, h=30.0, d=20.0)
    m1 = doc.evaluate_feature(ex1.id)
    zmax = float(m1.vertices[:, 2].max())
    fr = plane_frame_from_face(m1.vertices, m1.faces, [20.0, 15.0, zmax])
    sk2 = doc.create_sketch_on_face(ex1.id, fr)
    # 12×12 footprint fully on the face, reverse 8 mm into the 20 mm thick base
    sk2.sketch.add_rectangle((-6.0, -6.0), (6.0, 6.0))
    depth = 8.0
    ex2 = doc.create_extrude(sk2.id, depth, reversed=True)
    assert ex2.operand_b == ex1.id
    m2 = doc.evaluate_feature(ex2.id)
    assert m2 is not None and m2.is_watertight()

    # Tool volume alone (same profile, reverse) without merge
    resolved = resolve_profiles(sk2.sketch)
    tool = extrude_profile(
        resolved.outer, depth, sk2.sketch.frame, reversed=True
    )
    v_tool = tool.volume()
    assert v_tool == pytest.approx(12.0 * 12.0 * depth, abs=0.05)

    # Intersection of base with tool ≈ tool (fully contained)
    inter = boolean_op(m1, tool, BooleanOp.INTERSECTION)
    v_overlap = inter.volume()
    assert v_overlap == pytest.approx(v_tool, abs=0.5)

    v_union = m2.volume()
    plain_sum = v_base + v_tool
    # Must NOT equal the plain sum
    assert v_union < plain_sum - 1.0
    # Must match inclusion-exclusion
    expected = v_base + v_tool - v_overlap
    assert v_union == pytest.approx(expected, abs=0.5)
    # Fully contained boss: union ≈ base
    assert v_union == pytest.approx(v_base, abs=0.5)


def test_boss_depth_edit_undo_save_reload():
    doc = Document()
    doc.seed_reference_planes()
    ex1, v_base = _base_box(doc)
    m1 = doc.evaluate_feature(ex1.id)
    zmax = float(m1.vertices[:, 2].max())
    fr = plane_frame_from_face(m1.vertices, m1.faces, [20.0, 15.0, zmax])
    sk2 = doc.create_sketch_on_face(ex1.id, fr)
    sk2.sketch.add_rectangle((-4.0, -4.0), (4.0, 4.0))
    ex2 = doc.create_extrude(sk2.id, 6.0)
    v6 = doc.evaluate_feature(ex2.id).volume()
    assert v6 == pytest.approx(v_base + 8.0 * 8.0 * 6.0, abs=0.05)

    assert doc.update_feature_params(ex2.id, depth=12.0)
    v12 = doc.evaluate_feature(ex2.id).volume()
    assert v12 == pytest.approx(v_base + 8.0 * 8.0 * 12.0, abs=0.05)
    assert v12 > v6

    assert doc.undo()
    assert ex2.depth == pytest.approx(6.0)
    assert doc.evaluate_feature(ex2.id).volume() == pytest.approx(v6, abs=0.05)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "boss_merge.gcad"
        save_document(doc, path)
        loaded = load_document(path)
    ex2l = next(f for f in loaded.features if f.name == ex2.name)
    assert ex2l.operand_b == ex1.id
    mesh = loaded.evaluate_feature(ex2l.id)
    assert mesh is not None and mesh.is_watertight()
    assert mesh.volume() == pytest.approx(v6, abs=0.05)
    display = loaded.evaluate_display_solids()
    assert ex2l.id in display
    assert len(display) == 1


def test_base_extrude_on_plane_stays_standalone():
    """Sketch on reference plane still creates a base solid (no merge)."""
    doc = Document()
    doc.seed_reference_planes()
    ex, v = _base_box(doc)
    assert ex.operand_b == -1
    assert doc.evaluate_feature(ex.id).volume() == pytest.approx(v, abs=1e-6)
