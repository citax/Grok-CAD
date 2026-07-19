"""History integrity: dimensions, fillet sketch restore, cascade delete."""

from __future__ import annotations

from cadcore.document import Document, FeatureType
from cadcore.sketch import LineEntity, RectEntity


def _front_sketch():
    doc = Document()
    doc.seed_reference_planes()
    front = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(front.id)
    return doc, skf


def test_multi_delete_clears_dimensions():
    doc, skf = _front_sketch()
    sk = skf.sketch
    a = sk.add_line((0, 0), (10, 0))
    b = sk.add_line((0, 5), (10, 5))
    doc.record_entity_add(skf.id, a)
    doc.record_entity_add(skf.id, b)
    doc.apply_sketch_dimension(skf.id, a.id, "length", 12.0)
    doc.apply_sketch_dimension(skf.id, b.id, "length", 8.0)
    assert len(sk.dimensions) == 2
    n = doc.delete_entities(skf.id, [a.id, b.id])
    assert n == 2
    assert len(sk.entities) == 0
    assert len(sk.dimensions) == 0


def test_single_delete_undo_restores_dimensions():
    doc, skf = _front_sketch()
    sk = skf.sketch
    a = sk.add_line((0, 0), (10, 0))
    doc.record_entity_add(skf.id, a)
    doc.apply_sketch_dimension(skf.id, a.id, "length", 15.0)
    assert len(sk.dimensions) == 1
    doc.delete_entity(skf.id, a.id)
    assert len(sk.dimensions) == 0
    doc.undo()
    assert sk.find_entity(a.id) is not None
    assert len(sk.dimensions) == 1
    assert sk.dimensions[0].value_mm == 15.0


def test_dimension_apply_undo_restores_value_and_geometry():
    doc, skf = _front_sketch()
    sk = skf.sketch
    line = sk.add_line((0, 0), (10, 0))
    lid = line.id
    doc.record_entity_add(skf.id, line)
    doc.apply_sketch_dimension(skf.id, lid, "length", 20.0)
    line = sk.find_entity(lid)
    assert abs(line.p1[0] - line.p0[0]) == 20.0
    assert sk.dimensions[0].value_mm == 20.0
    doc.undo()
    line = sk.find_entity(lid)
    assert abs(line.p1[0] - line.p0[0]) == 10.0
    # First apply created the dim — undo removes it (no prior dim)
    assert len(sk.dimensions) == 0
    doc.redo()
    line = sk.find_entity(lid)
    assert abs(line.p1[0] - line.p0[0]) == 20.0
    assert sk.dimensions[0].value_mm == 20.0
    # Second apply then undo restores previous dim value
    doc.apply_sketch_dimension(skf.id, lid, "length", 30.0)
    assert sk.dimensions[0].value_mm == 30.0
    doc.undo()
    line = sk.find_entity(lid)
    assert abs(line.p1[0] - line.p0[0]) == 20.0
    assert sk.dimensions[0].value_mm == 20.0


def test_fillet_undo_restores_sharp_sketch():
    doc, skf = _front_sketch()
    sk = skf.sketch
    r = sk.add_rectangle((0, 0), (20, 10))
    doc.record_entity_add(skf.id, r)
    assert isinstance(sk.entities[0], RectEntity)
    ff = doc.create_fillet(skf.id, 5.0, 2.0)
    assert doc.find(ff.id) is not None
    assert all(isinstance(e, LineEntity) for e in sk.entities)
    assert len(sk.entities) > 1
    doc.undo()
    assert doc.find(ff.id) is None
    assert len(sk.entities) == 1
    assert isinstance(sk.entities[0], RectEntity)
    assert sk.entities[0].c0 == (0.0, 0.0)
    assert sk.entities[0].c1 == (20.0, 10.0)


def test_delete_sketch_cascades_to_extrude():
    doc, skf = _front_sketch()
    sk = skf.sketch
    sk.add_rectangle((0, 0), (10, 10))
    ex = doc.create_extrude(skf.id, 5.0)
    assert doc.find(ex.id) is not None
    assert doc.evaluate_feature(ex.id) is not None
    assert doc.delete_feature_undoable(skf.id)
    assert doc.find(skf.id) is None
    assert doc.find(ex.id) is None
    # Undo restores both
    doc.undo()
    assert doc.find(skf.id) is not None
    assert doc.find(ex.id) is not None
    assert doc.evaluate_feature(ex.id) is not None
