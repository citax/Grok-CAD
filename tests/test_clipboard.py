"""Clipboard cut/copy/paste for sketch entities."""

from __future__ import annotations

from cadcore.document import PASTE_UV_DELTA, Document, FeatureType
from cadcore.sketch import LineEntity


def _front_sketch():
    doc = Document()
    doc.seed_reference_planes()
    front = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(front.id)
    return doc, skf


def test_copy_paste_offsets_and_new_id():
    doc, skf = _front_sketch()
    sk = skf.sketch
    line = sk.add_line((1.0, 2.0), (4.0, 2.0))
    doc.record_entity_add(skf.id, line)
    assert doc.copy_entity(skf.id, line.id)
    pasted = doc.paste_entity(skf.id)
    assert pasted is not None
    assert isinstance(pasted, LineEntity)
    assert pasted.id != line.id
    # first paste uses 1 * PASTE_UV_DELTA
    assert abs(pasted.p0[0] - (1.0 + PASTE_UV_DELTA[0])) < 1e-9
    assert abs(pasted.p0[1] - (2.0 + PASTE_UV_DELTA[1])) < 1e-9
    assert abs(pasted.p1[0] - (4.0 + PASTE_UV_DELTA[0])) < 1e-9


def test_paste_is_undoable():
    doc, skf = _front_sketch()
    sk = skf.sketch
    line = sk.add_line((0, 0), (3, 0))
    doc.record_entity_add(skf.id, line)
    doc.copy_entity(skf.id, line.id)
    pasted = doc.paste_entity(skf.id)
    pid = pasted.id
    assert sk.find_entity(pid) is not None
    doc.undo()
    assert sk.find_entity(pid) is None
    doc.redo()
    assert sk.find_entity(pid) is not None


def test_cut_copies_and_deletes():
    doc, skf = _front_sketch()
    sk = skf.sketch
    line = sk.add_line((0, 0), (2, 1))
    doc.record_entity_add(skf.id, line)
    eid = line.id
    assert doc.cut_entity(skf.id, eid)
    assert sk.find_entity(eid) is None
    pasted = doc.paste_entity(skf.id)
    assert pasted is not None
    assert sk.find_entity(eid) is None
    assert sk.find_entity(pasted.id) is not None


def test_paste_empty_clipboard_noop():
    doc, skf = _front_sketch()
    assert doc.paste_entity(skf.id) is None


def test_paste_on_plane_world():
    import numpy as np

    doc, skf = _front_sketch()
    sk = skf.sketch
    line = sk.add_line((0, 0), (5, 3))
    doc.record_entity_add(skf.id, line)
    doc.copy_entity(skf.id, line.id)
    p = doc.paste_entity(skf.id)
    fr = sk.frame
    for uv in (p.p0, p.p1):
        w = fr.to_world(uv)
        dev = abs(float(np.dot(fr.normal, np.asarray(w) - fr.origin)))
        assert dev < 1e-9
