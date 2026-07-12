"""Document undo/redo stack — empty no-op, redo cleared on new action."""

from __future__ import annotations

from cadcore.document import Document, FeatureType
from cadcore.sketch import LineEntity, snapshot_entity


def _front_sketch():
    doc = Document()
    doc.seed_reference_planes()
    front = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(front.id)
    return doc, skf


def test_empty_stack_undo_redo_noop():
    doc = Document()
    doc.seed_reference_planes()
    assert doc.undo() is False
    assert doc.redo() is False
    assert not doc.can_undo()
    assert not doc.can_redo()


def test_entity_add_undo_redo_roundtrip():
    doc, skf = _front_sketch()
    sk = skf.sketch
    line = sk.add_line((0.0, 0.0), (10.0, 0.0))
    doc.record_entity_add(skf.id, line)
    assert sk.find_entity(line.id) is not None
    assert doc.undo() is True
    assert sk.find_entity(line.id) is None
    assert doc.redo() is True
    restored = sk.find_entity(line.id)
    assert restored is not None
    assert isinstance(restored, LineEntity)
    assert restored.p0 == (0.0, 0.0) and restored.p1 == (10.0, 0.0)


def test_new_action_clears_redo():
    doc, skf = _front_sketch()
    sk = skf.sketch
    a = sk.add_line((0, 0), (1, 0))
    doc.record_entity_add(skf.id, a)
    doc.undo()
    assert doc.can_redo()
    b = sk.add_line((0, 1), (1, 1))
    doc.record_entity_add(skf.id, b)
    assert not doc.can_redo()
    assert sk.find_entity(a.id) is None  # stayed undone
    assert sk.find_entity(b.id) is not None


def test_entity_delete_undo():
    doc, skf = _front_sketch()
    sk = skf.sketch
    line = sk.add_line((0, 0), (2, 0))
    doc.record_entity_add(skf.id, line)
    assert doc.delete_entity(skf.id, line.id)
    assert sk.find_entity(line.id) is None
    doc.undo()
    assert sk.find_entity(line.id) is not None


def test_entity_move_undo():
    doc, skf = _front_sketch()
    sk = skf.sketch
    line = sk.add_line((0, 0), (4, 0))
    doc.record_entity_add(skf.id, line)
    before = snapshot_entity(line)
    line.p1 = (8.0, 0.0)
    after = snapshot_entity(line)
    doc.record_entity_move(skf.id, before, after)
    doc.undo()
    assert line.p1 == (4.0, 0.0)
    doc.redo()
    assert line.p1 == (8.0, 0.0)


def test_feature_delete_undoable():
    doc, skf = _front_sketch()
    fid = skf.id
    assert doc.delete_feature_undoable(fid)
    assert doc.find(fid) is None
    doc.undo()
    assert doc.find(fid) is not None
