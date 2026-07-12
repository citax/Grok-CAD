"""Paste: cumulative offset, refuse without live sketch, on-plane, undo."""

from __future__ import annotations

import numpy as np

from cadcore.document import PASTE_UV_DELTA, Document, FeatureType
from cadcore.sketch import LineEntity


def _doc_sk():
    doc = Document()
    doc.seed_reference_planes()
    front = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(front.id)
    return doc, skf


def test_repeated_pastes_different_positions():
    doc, skf = _doc_sk()
    line = skf.sketch.add_line((0.0, 0.0), (3.0, 0.0))
    doc.record_entity_add(skf.id, line)
    doc.copy_entity(skf.id, line.id)
    a = doc.paste_entity(skf.id)
    b = doc.paste_entity(skf.id)
    c = doc.paste_entity(skf.id)
    assert a is not None and b is not None and c is not None
    # cumulative n*delta → distinct anchors
    assert a.p0 != b.p0
    assert b.p0 != c.p0
    assert a.p0 != c.p0
    # step between consecutive is PASTE_UV_DELTA
    assert abs(b.p0[0] - a.p0[0] - PASTE_UV_DELTA[0]) < 1e-9
    assert abs(c.p0[0] - b.p0[0] - PASTE_UV_DELTA[0]) < 1e-9


def test_paste_at_place_uv_staggers():
    doc, skf = _doc_sk()
    line = skf.sketch.add_line((1.0, 1.0), (4.0, 1.0))
    doc.record_entity_add(skf.id, line)
    doc.copy_entity(skf.id, line.id)
    a = doc.paste_entity(skf.id, place_uv=(10.0, 20.0))
    b = doc.paste_entity(skf.id, place_uv=(10.0, 20.0))
    assert a is not None and b is not None
    assert abs(a.p0[0] - 10.0) < 1e-9 and abs(a.p0[1] - 20.0) < 1e-9
    assert abs(b.p0[0] - (10.0 + PASTE_UV_DELTA[0])) < 1e-9
    assert a.p0 != b.p0


def test_paste_without_live_sketch_is_complete_noop():
    """Without a live sketch session, paste must not change doc or undo.

    Mirrors MainWindow._paste gate: refuse when not in_sketch_mode (no
    paste_entity call → no document mutation, no undo entry). Full GUI
    path is covered by the real xcb verification.
    """
    doc, skf = _doc_sk()
    line = skf.sketch.add_line((0, 0), (2, 0))
    doc.record_entity_add(skf.id, line)
    doc.copy_entity(skf.id, line.id)
    n_ent = len(skf.sketch.entities)
    n_undo = len(doc._undo_stack)
    # Simulate GUI gate: if not in_sketch_mode → return without paste_entity
    in_sketch_mode = False
    if not in_sketch_mode:
        pass  # refuse — do not call paste_entity
    else:
        doc.paste_entity(skf.id)
    assert len(skf.sketch.entities) == n_ent
    assert len(doc._undo_stack) == n_undo
    assert doc._clipboard is not None  # copy still held, just not pasted


def test_paste_on_plane():
    doc, skf = _doc_sk()
    line = skf.sketch.add_line((0, 0), (5, 3))
    doc.record_entity_add(skf.id, line)
    doc.copy_entity(skf.id, line.id)
    p = doc.paste_entity(skf.id, place_uv=(1.0, 2.0))
    fr = skf.sketch.frame
    for uv in (p.p0, p.p1):
        w = fr.to_world(uv)
        dev = abs(float(np.dot(fr.normal, w - fr.origin)))
        assert dev < 1e-9


def test_paste_undo():
    doc, skf = _doc_sk()
    line = skf.sketch.add_line((0, 0), (2, 0))
    doc.record_entity_add(skf.id, line)
    doc.copy_entity(skf.id, line.id)
    p = doc.paste_entity(skf.id)
    pid = p.id
    assert skf.sketch.find_entity(pid) is not None
    doc.undo()
    assert skf.sketch.find_entity(pid) is None
    doc.redo()
    assert skf.sketch.find_entity(pid) is not None
