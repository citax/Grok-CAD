"""Project save/load: document round-trip (parametric state)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cadcore.document import Document, FeatureType
from cadcore.project_io import (
    ProjectIOError,
    document_from_dict,
    document_to_dict,
    load_document,
    save_document,
)
from cadcore.units import Unit


def _build_sample_part() -> Document:
    doc = Document()
    doc.seed_reference_planes()
    front = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(front.id)
    assert skf is not None and skf.sketch is not None
    sk = skf.sketch
    rect = sk.add_rectangle((0.0, 0.0), (40.0, 20.0))
    doc.record_entity_add(skf.id, rect)
    doc.apply_sketch_dimension(skf.id, rect.id, "width", 50.0)
    doc.apply_sketch_dimension(skf.id, rect.id, "height", 25.0)
    circ = sk.add_circle((12.0, 12.0), 4.0)
    doc.record_entity_add(skf.id, circ)
    doc.apply_sketch_dimension(skf.id, circ.id, "diameter", 10.0)
    ex = doc.create_extrude(skf.id, 15.0, reversed=True)
    # Edit after creation
    assert doc.update_feature_params(ex.id, depth=22.0)
    # Second sketch + revolve profile offset from axis
    sk2 = doc.create_sketch_on_plane(front.id)
    assert sk2 is not None and sk2.sketch is not None
    r2 = sk2.sketch.add_rectangle((10.0, 0.0), (18.0, 8.0))
    doc.record_entity_add(sk2.id, r2)
    doc.create_revolve(sk2.id, angle_degrees=180.0)
    doc.set_display_unit(Unit.INCH)
    doc.name = "SamplePart"
    return doc


def test_dict_roundtrip_preserves_tree_and_params():
    doc = _build_sample_part()
    data = document_to_dict(doc)
    loaded = document_from_dict(data)
    assert loaded.name == "SamplePart"
    assert loaded.display_unit is Unit.INCH
    assert len(loaded.features) == len(doc.features)
    types = [f.type for f in loaded.features]
    assert FeatureType.EXTRUDE in types
    assert FeatureType.REVOLVE in types
    assert sum(1 for t in types if t is FeatureType.SKETCH) == 2
    ex = next(f for f in loaded.features if f.type is FeatureType.EXTRUDE)
    assert ex.depth == pytest.approx(22.0)
    assert ex.reversed is True
    skf = loaded.find(ex.operand_a)
    assert skf is not None and skf.sketch is not None
    assert len(skf.sketch.dimensions) == 3
    # Solids evaluate
    m = loaded.evaluate_feature(ex.id)
    assert m is not None and not m.empty
    m0 = doc.evaluate_feature(ex.id)
    assert abs(m.volume() - m0.volume()) < 1e-4 * max(abs(m0.volume()), 1.0)


def test_file_roundtrip(tmp_path: Path):
    doc = _build_sample_part()
    path = tmp_path / "part.gcad"
    save_document(doc, path)
    assert path.is_file()
    assert doc.dirty is False
    loaded = load_document(path)
    assert loaded.dirty is False
    assert loaded.name == "part"
    ex = next(f for f in loaded.features if f.type is FeatureType.EXTRUDE)
    assert ex.depth == pytest.approx(22.0)
    # Editable after load
    assert loaded.update_feature_params(ex.id, depth=30.0)
    assert loaded.can_undo()
    loaded.undo()
    assert ex.depth == pytest.approx(22.0)
    loaded.redo()
    assert ex.depth == pytest.approx(30.0)


def test_reject_bad_format(tmp_path: Path):
    p = tmp_path / "bad.gcad"
    p.write_text('{"format": "nope", "version": 1}', encoding="utf-8")
    with pytest.raises(ProjectIOError):
        load_document(p)


def test_face_sketch_frame_survives():
    """Sketch on a solid face keeps its world frame through serialize."""
    doc = Document()
    doc.seed_reference_planes()
    front = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(front.id)
    skf.sketch.add_rectangle((0, 0), (10, 10))
    doc.create_extrude(skf.id, 5.0)
    # Fake face frame offset along +Z of a solid
    from cadcore.sketch import PlaneFrame

    frame = PlaneFrame(
        origin=np.array([0.0, 0.0, 5.0]),
        u_axis=np.array([1.0, 0.0, 0.0]),
        v_axis=np.array([0.0, 1.0, 0.0]),
        normal=np.array([0.0, 0.0, 1.0]),
    )
    solid = next(f for f in doc.features if f.type is FeatureType.EXTRUDE)
    sk_face = doc.create_sketch_on_face(solid.id, frame)
    assert sk_face is not None
    data = document_to_dict(doc)
    loaded = document_from_dict(data)
    sk2 = next(
        f
        for f in loaded.features
        if f.type is FeatureType.SKETCH and f.plane_id == solid.id and f.id == sk_face.id
    )
    assert np.allclose(sk2.sketch.frame.origin, [0, 0, 5])
    assert np.allclose(sk2.sketch.frame.normal, [0, 0, 1])
