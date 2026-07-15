"""PropertyManager: real PropertyPanel widget (not just Document layer)."""

from __future__ import annotations

import os
import sys

import pytest

# Headless-friendly default for pure widget tests (no VTK)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app.property_panel import PropertyPanel
from cadcore.document import Document, FeatureType
from cadcore.fillet2d import fillet_closed_polygon
from cadcore.sketch import LineEntity


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def _doc_with_all_solids():
    doc = Document()
    doc.seed_reference_planes()
    plane = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)

    sk_ex = doc.create_sketch_on_plane(plane.id)
    sk_ex.sketch.add_rectangle((0, 0), (2, 2))
    extrude = doc.create_extrude(sk_ex.id, 1.0)

    sk_fil = doc.create_sketch_on_plane(plane.id)
    sk_fil.sketch.add_rectangle((0, 0), (4, 4))
    fillet = doc.create_fillet(sk_fil.id, 1.0, 0.3, segments=24)

    sk_rev = doc.create_sketch_on_plane(plane.id)
    sk_rev.sketch.add_rectangle((0.5, 0.0), (1.5, 1.0))
    revolve = doc.create_revolve(sk_rev.id, angle_degrees=180.0)

    sk_pok = doc.create_sketch_on_plane(plane.id)
    sk_pok.sketch.add_rectangle((0, 0), (3, 3))
    pocket = doc.create_pocket(sk_pok.id, 1.0, 0.4, hole_center=(1.5, 1.5))

    return doc, extrude, fillet, revolve, pocket, sk_fil


def test_show_feature_twice_no_crash():
    """Second show_feature must not touch deleted C++ spin boxes."""
    _app()
    doc, extrude, fillet, revolve, pocket, _ = _doc_with_all_solids()
    p = PropertyPanel()
    p.set_document(doc)
    for feat in (extrude, fillet, revolve, pocket):
        p.show_feature(feat)
        p.show_feature(feat)  # would crash with removeRow + reuse
        assert "depth" in p._editors or "angle" in p._editors or "radius" in p._editors


def test_type_switching_no_crash():
    _app()
    doc, extrude, fillet, revolve, pocket, _ = _doc_with_all_solids()
    p = PropertyPanel()
    p.set_document(doc)
    order = [extrude, fillet, extrude, revolve, pocket, fillet, extrude]
    for feat in order:
        p.show_feature(feat)
    # Final form is Extrude
    assert abs(float(p._editors["depth"].value()) - 1.0) < 1e-9


def test_apply_button_changes_extrude_depth_and_volume():
    _app()
    doc = Document()
    doc.seed_reference_planes()
    plane = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(plane.id)
    skf.sketch.add_rectangle((0, 0), (2, 2))  # area 4
    ex = doc.create_extrude(skf.id, 1.0)
    mesh0 = doc.evaluate_feature(ex.id)
    assert mesh0 is not None
    assert abs(mesh0.volume() - 4.0) < 0.05

    p = PropertyPanel()
    p.set_document(doc)
    p.show_feature(ex)
    p._editors["depth"].setValue(3.5)
    p.btn_apply.click()  # real button signal, not _on_apply()
    assert abs(ex.depth - 3.5) < 1e-12
    mesh1 = doc.evaluate_feature(ex.id)
    assert mesh1 is not None
    assert abs(mesh1.volume() - 14.0) < 0.1


def test_apply_reverse_direction_moves_solid():
    from PySide6.QtWidgets import QCheckBox

    _app()
    doc = Document()
    doc.seed_reference_planes()
    plane = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(plane.id)
    skf.sketch.add_rectangle((0, 0), (2, 2))
    ex = doc.create_extrude(skf.id, 2.0)
    m0 = doc.evaluate_feature(ex.id)
    assert m0 is not None and m0.vertices[:, 2].min() >= -1e-6

    p = PropertyPanel()
    p.set_document(doc)
    p.show_feature(ex)
    cb = p._editors["reversed"]
    assert isinstance(cb, QCheckBox)
    cb.setChecked(True)
    p.btn_apply.click()
    assert ex.reversed is True
    m1 = doc.evaluate_feature(ex.id)
    assert m1 is not None
    assert m1.vertices[:, 2].max() <= 1e-6
    assert m1.vertices[:, 2].min() < -1.0
    assert abs(m0.volume() - m1.volume()) < 1e-6


def test_apply_fillet_radius_via_button():
    _app()
    doc = Document()
    doc.seed_reference_planes()
    plane = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(plane.id)
    skf.sketch.add_rectangle((0, 0), (4, 4))
    fil = doc.create_fillet(skf.id, 1.0, 0.3, segments=24)
    p = PropertyPanel()
    p.set_document(doc)
    p.show_feature(fil)
    p._editors["radius"].setValue(0.6)
    p.btn_apply.click()
    assert abs(fil.radius - 0.6) < 1e-12


def test_apply_revolve_and_pocket_via_button():
    _app()
    doc, _ex, _fil, revolve, pocket, _ = _doc_with_all_solids()
    p = PropertyPanel()
    p.set_document(doc)
    p.show_feature(revolve)
    p._editors["angle"].setValue(90.0)
    p.btn_apply.click()
    assert abs(revolve.revolve_angle - 90.0) < 1e-12

    p.show_feature(pocket)
    p._editors["radius"].setValue(0.55)
    p._editors["depth"].setValue(2.0)
    p.btn_apply.click()
    assert abs(pocket.radius - 0.55) < 1e-12
    assert abs(pocket.depth - 2.0) < 1e-12


def test_show_sketch_line_twice_and_apply():
    _app()
    doc = Document()
    doc.seed_reference_planes()
    plane = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(plane.id)
    line = skf.sketch.add_line((0, 0), (2, 0))
    p = PropertyPanel()
    p.set_document(doc)
    p.show_sketch_line(skf.id, line)
    p.show_sketch_line(skf.id, line)
    p._editors["line_len"].setValue(5.0)
    p.btn_apply.click()
    ent = skf.sketch.find_entity(line.id)
    assert isinstance(ent, LineEntity)
    assert abs(ent.p1[0] - 5.0) < 1e-9


def test_clear_then_show_again():
    _app()
    doc, extrude, fillet, *_ = _doc_with_all_solids()
    p = PropertyPanel()
    p.set_document(doc)
    p.show_feature(extrude)
    p.clear()
    p.show_feature(fillet)
    p.show_feature(extrude)
    assert "depth" in p._editors


def test_update_extrude_depth_undoable_document_layer():
    """Document-layer undo still works (used by MainWindow Undo)."""
    doc = Document()
    doc.seed_reference_planes()
    plane = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(plane.id)
    skf.sketch.add_rectangle((0, 0), (2, 2))
    ex = doc.create_extrude(skf.id, 1.0)
    assert doc.update_feature_params(ex.id, depth=3.5)
    assert abs(ex.depth - 3.5) < 1e-12
    assert doc.undo()
    assert abs(ex.depth - 1.0) < 1e-12
    assert doc.redo()
    assert abs(ex.depth - 3.5) < 1e-12


def test_fillet_radius_change_entity_count_exact():
    """After radius change, entity count equals closed polyline length."""
    doc = Document()
    doc.seed_reference_planes()
    plane = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(plane.id)
    skf.sketch.add_rectangle((0, 0), (4, 4))
    fil = doc.create_fillet(skf.id, 1.0, 0.3, segments=24)
    # create_fillet used arc_segments=max(6, 24//4)=6
    n_create = len(skf.sketch.entities)
    expected_create = len(
        fillet_closed_polygon(fil.source_profile_uv, 0.3, arc_segments=6)
    )
    assert n_create == expected_create

    assert doc.update_feature_params(fil.id, radius=0.6)
    expected = len(fillet_closed_polygon(fil.source_profile_uv, 0.6, arc_segments=6))
    assert len(skf.sketch.entities) == expected
    mesh = doc.evaluate_feature(fil.id)
    assert mesh is not None and mesh.is_watertight()
