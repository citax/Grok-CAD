"""PropertyManager: feature param edit + undo."""

from __future__ import annotations

from cadcore.document import Document, FeatureType


def test_update_extrude_depth_undoable():
    doc = Document()
    doc.seed_reference_planes()
    plane = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(plane.id)
    skf.sketch.add_rectangle((0, 0), (2, 2))
    ex = doc.create_extrude(skf.id, 1.0)
    assert abs(ex.depth - 1.0) < 1e-12
    assert doc.update_feature_params(ex.id, depth=3.5)
    assert abs(ex.depth - 3.5) < 1e-12
    assert doc.undo()
    assert abs(ex.depth - 1.0) < 1e-12
    assert doc.redo()
    assert abs(ex.depth - 3.5) < 1e-12


def test_update_fillet_radius_rebuilds_sketch():
    doc = Document()
    doc.seed_reference_planes()
    plane = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(plane.id)
    skf.sketch.add_rectangle((0, 0), (4, 4))
    fil = doc.create_fillet(skf.id, 1.0, 0.3, segments=24)
    n0 = len(skf.sketch.entities)
    assert doc.update_feature_params(fil.id, radius=0.6)
    assert abs(fil.radius - 0.6) < 1e-12
    # Sketch rebuilt from source sharp poly
    assert len(skf.sketch.entities) >= n0 - 5  # still a closed loop polyline
    mesh = doc.evaluate_feature(fil.id)
    assert mesh is not None and mesh.is_watertight()
