"""Feature command PM: select → set → OK; cancel; failure leaves part unchanged."""

from __future__ import annotations

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app.property_panel import PropertyPanel
from cadcore.document import Document, FeatureType


def _app():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def test_command_ok_creates_extrude_without_dialogs():
    _app()
    doc = Document()
    doc.seed_reference_planes()
    plane = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    sk = doc.create_sketch_on_plane(plane.id)
    sk.sketch.add_rectangle((0, 0), (10, 10))
    n0 = len(doc.features)

    p = PropertyPanel()
    p.set_document(doc)
    p.show_command(
        "extrude",
        title="Extrude",
        selection_text=f"Sketch: {sk.name}",
        ready=True,
        defaults={"depth": 5.0, "reversed": False},
    )
    p._editors["depth"].setText("7")
    params = p.read_command_params()
    assert params["depth"] == pytest.approx(7.0)
    feat = doc.create_extrude(sk.id, params["depth"], reversed=params["reversed"])
    assert len(doc.features) == n0 + 1
    assert feat.depth == pytest.approx(7.0)
    assert doc.undo()
    assert doc.find(feat.id) is None


def test_fillet_failure_does_not_mutate_sketch():
    """OK on a circle profile must fail cleanly — no half-changed sketch."""
    doc = Document()
    doc.seed_reference_planes()
    plane = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    sk = doc.create_sketch_on_plane(plane.id)
    sk.sketch.add_circle((0, 0), 5.0)
    n_ent = len(sk.sketch.entities)
    n_feat = len(doc.features)
    with pytest.raises(ValueError, match="no corners"):
        doc.create_fillet(sk.id, 5.0, 1.0)
    assert len(sk.sketch.entities) == n_ent
    assert len(doc.features) == n_feat


def test_cancel_command_signal():
    _app()
    p = PropertyPanel()
    cancelled = []
    p.command_cancel.connect(lambda: cancelled.append(1))
    p.show_command("fillet", title="Fillet", selection_text="…", ready=False)
    assert not p.btn_cancel.isHidden()
    p.btn_cancel.click()
    assert cancelled == [1]
