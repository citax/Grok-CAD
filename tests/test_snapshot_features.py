"""snapshot_features must copy every Feature field (structural)."""

from __future__ import annotations

from dataclasses import fields

import numpy as np

from app.workers import (
    evaluate_solids_snapshot,
    feature_fingerprint,
    snapshot_feature,
    snapshot_features,
)
from cadcore.document import Document, Feature, FeatureType
from cadcore.sketch import PlaneFrame, Sketch


def _nondefault_feature() -> Feature:
    """Feature with every field set away from its dataclass default."""
    sk = Sketch(
        name="SnapSketch",
        plane_feature_id=99,
        frame=PlaneFrame.from_plane_type("PLANE_FRONT"),
    )
    sk.add_rectangle((0.0, 0.0), (2.0, 2.0))
    return Feature(
        id=42,
        name="NonDefault",
        type=FeatureType.EXTRUDE,
        width=3.0,
        height=4.0,
        depth=5.0,
        radius=0.77,
        segments=48,
        rings=7,
        operand_a=11,
        operand_b=12,
        translation=(0.1, 0.2, 0.3),
        plane_id=3,
        sketch=sk,
        profile_entity_id=7,
        reversed=True,
        revolve_angle=123.0,
        axis_origin=(0.5, 0.25),
        axis_direction=(0.0, -1.0),
        hole_center_u=0.3,
        hole_center_v=0.4,
        source_profile_uv=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
        visible=False,
        suppressed=True,
    )


def test_snapshot_feature_copies_all_fields():
    src = _nondefault_feature()
    snap = snapshot_feature(src)
    for fld in fields(Feature):
        a = getattr(src, fld.name)
        b = getattr(snap, fld.name)
        if fld.name == "sketch":
            assert b is not None and a is not None
            assert b is not a
            assert b.name == a.name
            assert len(b.entities) == len(a.entities)
        elif fld.name == "source_profile_uv":
            assert b == a and b is not a
        elif isinstance(a, tuple):
            assert b == a
        else:
            assert b == a, f"field {fld.name}: {a!r} != {b!r}"


def test_snapshot_features_includes_reversed():
    doc = Document()
    doc.seed_reference_planes()
    plane = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(plane.id)
    skf.sketch.add_rectangle((0, 0), (2, 2))
    ex = doc.create_extrude(skf.id, 2.0, reversed=True)
    snaps = snapshot_features(doc)
    snap_ex = next(s for s in snaps if s.id == ex.id)
    assert snap_ex.reversed is True
    assert snap_ex.depth == 2.0


def test_snapshot_path_geometry_differs_when_reversed():
    """Regression: reverse must change fingerprint AND mesh via worker path."""
    doc = Document()
    doc.seed_reference_planes()
    plane = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(plane.id)
    skf.sketch.add_rectangle((0, 0), (2, 2))
    ex = doc.create_extrude(skf.id, 2.0, reversed=False)

    feats_fwd = snapshot_features(doc)
    res_fwd = evaluate_solids_snapshot(feats_fwd)
    assert ex.id in res_fwd
    v_fwd, _f_fwd, fp_fwd = res_fwd[ex.id]

    assert doc.update_feature_params(ex.id, reversed=True)
    feats_rev = snapshot_features(doc)
    snap_ex = next(s for s in feats_rev if s.id == ex.id)
    assert snap_ex.reversed is True
    res_rev = evaluate_solids_snapshot(feats_rev)
    v_rev, _f_rev, fp_rev = res_rev[ex.id]

    assert fp_fwd != fp_rev
    # Geometry sides: FRONT normal +Z
    assert float(v_fwd[:, 2].min()) >= -1e-6
    assert float(v_fwd[:, 2].max()) > 1.0
    assert float(v_rev[:, 2].max()) <= 1e-6
    assert float(v_rev[:, 2].min()) < -1.0
    # volumes match
    # approximate via bbox * is wrong; just check z extents swapped sign
    assert abs(float(v_fwd[:, 2].max()) + float(v_rev[:, 2].min())) < 1e-5


def test_source_profile_uv_copied_in_snapshot():
    doc = Document()
    doc.seed_reference_planes()
    plane = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(plane.id)
    skf.sketch.add_rectangle((0, 0), (4, 4))
    fil = doc.create_fillet(skf.id, 1.0, 0.3, segments=24)
    assert fil.source_profile_uv
    snap = next(s for s in snapshot_features(doc) if s.id == fil.id)
    assert snap.source_profile_uv == list(fil.source_profile_uv)
    assert snap.source_profile_uv is not fil.source_profile_uv
