"""Cut-Extrude, absorbed sketches, nested feature tree."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from cadcore.document import Document, FeatureType
from cadcore.project_io import load_document, save_document
from cadcore.sketch import PlaneFrame


def _box_with_top_cut(depth: float = 8.0):
    doc = Document()
    doc.seed_reference_planes()
    front = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    sk = doc.create_sketch_on_plane(front.id)
    sk.sketch.add_rectangle((0, 0), (40, 30))
    ex = doc.create_extrude(sk.id, 20.0)
    frame = PlaneFrame(
        origin=np.array([0.0, 0.0, 20.0]),
        u_axis=np.array([1.0, 0.0, 0.0]),
        v_axis=np.array([0.0, 1.0, 0.0]),
        normal=np.array([0.0, 0.0, 1.0]),
    )
    skc = doc.create_sketch_on_face(ex.id, frame)
    skc.sketch.add_circle((20.0, 15.0), 6.0)
    cut = doc.create_cut_extrude(skc.id, ex.id, depth)
    return doc, ex, cut, sk, skc


def test_cut_reduces_volume_and_stays_watertight():
    doc, ex, cut, _sk, _skc = _box_with_top_cut(8.0)
    v0 = doc.evaluate_feature(ex.id).volume()
    m1 = doc.evaluate_feature(cut.id)
    assert m1 is not None and m1.is_watertight()
    assert m1.volume() < v0 - 100.0


def test_cut_depth_edit_undo_and_save_roundtrip(tmp_path: Path):
    doc, ex, cut, _sk, _skc = _box_with_top_cut(8.0)
    v8 = doc.evaluate_feature(cut.id).volume()
    assert doc.update_feature_params(cut.id, depth=15.0)
    v15 = doc.evaluate_feature(cut.id).volume()
    assert v15 < v8 - 50.0
    assert doc.undo()
    assert abs(doc.evaluate_feature(cut.id).volume() - v8) < 1e-2
    path = tmp_path / "cut.gcad"
    save_document(doc, path)
    loaded = load_document(path)
    c2 = next(f for f in loaded.features if f.type is FeatureType.CUT_EXTRUDE)
    m = loaded.evaluate_feature(c2.id)
    assert m is not None and m.is_watertight()
    assert abs(m.volume() - v8) < 1e-2
    assert loaded.update_feature_params(c2.id, depth=12.0)
    assert loaded.evaluate_feature(c2.id).volume() < v8


def test_absorbed_sketch_map_and_display_hides_target():
    doc, ex, cut, sk, skc = _box_with_top_cut(8.0)
    amap = doc.absorbed_sketch_map()
    assert sk.id in amap and amap[sk.id] == ex.id
    assert skc.id in amap and amap[skc.id] == cut.id
    # Display solids: original extrude hidden, cut shown
    disp = doc.evaluate_display_solids()
    assert ex.id not in disp
    assert cut.id in disp


def test_tree_nests_absorbed_sketches():
    """Tree building logic: sketch is child of Extrude, not a top-level sibling.

    Does not construct MainWindow (VTK) — exercises the same nesting rules
    used by MainWindow._refresh_tree.
    """
    doc = Document()
    doc.seed_reference_planes()
    front = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    sk = doc.create_sketch_on_plane(front.id)
    sk.sketch.add_rectangle((0, 0), (10, 10))
    ex = doc.create_extrude(sk.id, 5.0)
    free_sk = doc.create_sketch_on_plane(front.id)  # unused sketch stays top-level

    absorbed = doc.absorbed_sketch_map()
    assert absorbed.get(sk.id) == ex.id
    assert free_sk.id not in absorbed

    # Top-level feature ids (SolidWorks non-flat tree)
    top_level = []
    for f in doc.features:
        if f.type is FeatureType.SKETCH and f.id in absorbed:
            continue
        if f.type.name.startswith("PLANE"):
            continue
        top_level.append(f)
    names = [f.name for f in top_level]
    assert any("Extrude" in n for n in names)
    assert free_sk.name in names
    assert sk.name not in names  # absorbed under Extrude
    # Children under extrude
    kids = [sid for sid, cid in absorbed.items() if cid == ex.id]
    assert sk.id in kids
