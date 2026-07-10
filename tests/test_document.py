from cadcore.document import Document, Feature, FeatureType, is_reference_plane


def test_seed_planes():
    doc = Document()
    doc.seed_reference_planes()
    assert len(doc.features) == 3
    assert all(is_reference_plane(f.type) for f in doc.features)
    assert not doc.remove_feature(doc.features[0].id)


def test_add_box_evaluate():
    doc = Document()
    doc.seed_reference_planes()
    fid = doc.add_feature(Feature(type=FeatureType.BOX, width=2, height=2, depth=2))
    mesh = doc.evaluate_feature(fid)
    assert mesh is not None
    assert abs(mesh.volume() - 8.0) < 1e-9
