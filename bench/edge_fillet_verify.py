#!/usr/bin/env python3
"""Unattended verification: solid edge fillet + PM text + clean process exit.

Must exit on its own with a process exit code (no human, no blocking dialogs).
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

# Headless + unattended before Qt import
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["GROK_CAD_UNATTENDED"] = "1"
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

OUT = Path(__file__).resolve().parent / "_ui_shots"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    print("[edge_fillet_verify] start", flush=True)
    try:
        import numpy as np
        from PySide6.QtWidgets import QApplication

        from app.theme import apply_theme
        from app.mainwindow import MainWindow
        from app.property_panel import PropertyPanel
        from cadcore.document import Document, FeatureType
        from cadcore.edge_fillet import extract_convex_edges
        from cadcore.project_io import load_document, save_document
    except Exception as exc:
        print(f"[edge_fillet_verify] import failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app)

    # ----- 1) Core solid fillet on a real part -----
    doc = Document()
    doc.seed_reference_planes()
    plane = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    sk = doc.create_sketch_on_plane(plane.id)
    sk.sketch.add_rectangle((0.0, 0.0), (30.0, 30.0))
    ex = doc.create_extrude(sk.id, 30.0)
    body0 = doc.evaluate_feature(ex.id)
    assert body0 is not None and body0.is_watertight()
    v0 = body0.volume()
    print(f"[edge_fillet_verify] volume before fillet: {v0:.6f}", flush=True)
    edges = extract_convex_edges(body0.vertices, body0.faces)
    assert len(edges) >= 4, f"expected box edges, got {len(edges)}"
    key = edges[0].key()
    fl = doc.create_edge_fillet(ex.id, [key], 3.0)
    body1 = doc.evaluate_feature(fl.id)
    assert body1 is not None and body1.is_watertight()
    v1 = body1.volume()
    print(
        f"[edge_fillet_verify] volume after r=3: {v1:.6f}  delta={v0 - v1:.6f}",
        flush=True,
    )
    assert v1 < v0 - 10.0, "fillet must remove material"
    assert doc.update_feature_params(fl.id, radius=5.0)
    body2 = doc.evaluate_feature(fl.id)
    v2 = body2.volume()
    print(
        f"[edge_fillet_verify] volume after r=5: {v2:.6f}  delta={v0 - v2:.6f}",
        flush=True,
    )
    assert v2 < v1, "larger radius must remove more material"
    assert doc.undo()  # radius back to 3
    assert fl.radius == 3.0
    # failure path
    n_feat = len(doc.features)
    try:
        doc.create_edge_fillet(ex.id, [key], 50.0)
        print("[edge_fillet_verify] FAIL: oversized radius should raise", flush=True)
        return 2
    except ValueError as exc:
        print(f"[edge_fillet_verify] failure message (good): {exc}", flush=True)
    assert len(doc.features) == n_feat
    assert doc.evaluate_feature(ex.id).volume() == v0  # parent unchanged
    # save / reopen
    path = OUT / "edge_fillet_part.gcad"
    save_document(doc, path)
    loaded = load_document(path)
    fl2 = next(f for f in loaded.features if f.type is FeatureType.EDGE_FILLET)
    assert fl2.radius == 3.0 and fl2.edge_keys == [key]
    m = loaded.evaluate_feature(fl2.id)
    assert m is not None and m.is_watertight()
    print(
        f"[edge_fillet_verify] reloaded fillet vol={m.volume():.6f} editable",
        flush=True,
    )
    assert loaded.update_feature_params(fl2.id, radius=4.0)

    # ----- 2) PropertyManager stays compact; text still fully readable -----
    p = PropertyPanel()
    p.resize(PropertyPanel.PREFERRED_WIDTH, 400)
    p.show()
    app.processEvents()
    assert p.maximumWidth() <= PropertyPanel.MAX_WIDTH
    assert p.sizeHint().width() <= PropertyPanel.PREFERRED_WIDTH + 20
    print(
        f"[edge_fillet_verify] PM sizeHint={p.sizeHint().width()}x{p.sizeHint().height()} "
        f"maxW={p.maximumWidth()}",
        flush=True,
    )
    for state, kwargs in (
        ("empty", None),
        (
            "command_fillet_waiting",
            dict(
                command="fillet",
                title="Fillet",
                selection_text="Click edges on a solid to fillet…",
                ready=False,
            ),
        ),
        (
            "command_fillet_ready",
            dict(
                command="fillet",
                title="Fillet",
                selection_text="Solid: Extrude1\n2 edges selected (last L=30 mm)",
                ready=True,
            ),
        ),
        (
            "command_extrude",
            dict(
                command="extrude",
                title="Extrude",
                selection_text="Select a sketch with a closed profile…",
                ready=False,
            ),
        ),
        (
            "command_cut",
            dict(
                command="cut",
                title="Cut-Extrude",
                selection_text="Sketch: Sketch1\nSolid: Extrude1",
                ready=True,
            ),
        ),
    ):
        if state == "empty":
            p.show_empty()
        else:
            p.show_command(**kwargs)
        app.processEvents()
        sel = p._selection_label
        hint = p._hint
        # Compact caps — panel must not balloon
        assert hint.maximumHeight() <= 72, f"{state}: hint maxH={hint.maximumHeight()}"
        assert hint.minimumHeight() <= 64, f"{state}: hint minH={hint.minimumHeight()}"
        if not sel.isHidden():
            assert sel.maximumHeight() <= 80, f"{state}: sel maxH={sel.maximumHeight()}"
            assert sel.minimumHeight() <= 72, f"{state}: sel minH={sel.minimumHeight()}"
            print(
                f"[edge_fillet_verify] PM {state}: selection "
                f"minH={sel.minimumHeight()} maxH={sel.maximumHeight()}",
                flush=True,
            )
        print(
            f"[edge_fillet_verify] PM {state}: hint "
            f"minH={hint.minimumHeight()} maxH={hint.maximumHeight()}",
            flush=True,
        )
    p.close()

    # ----- 3) Unattended discard (no modal) — real MainWindow method -----
    assert MainWindow._is_unattended() is True

    class StubWin:
        def __init__(self, dirty_doc: Document):
            self.doc = dirty_doc

        _is_unattended = staticmethod(MainWindow._is_unattended)
        _confirm_discard_if_dirty = MainWindow._confirm_discard_if_dirty
        _file_save = lambda self: False  # should never be called unattended

    dirty = Document()
    dirty.seed_reference_planes()
    pl = next(f for f in dirty.features if f.type is FeatureType.PLANE_FRONT)
    dirty.create_sketch_on_plane(pl.id)
    assert dirty.dirty
    stub = StubWin(dirty)
    # If this blocked on a dialog, the process would hang and never print PASS.
    ok = stub._confirm_discard_if_dirty()
    assert ok is True, "unattended dirty close must discard without dialog"
    print(
        "[edge_fillet_verify] unattended dirty discard returned True (no dialog)",
        flush=True,
    )

    # Also exercise closeEvent logic path without constructing VTK viewport:
    # closeEvent calls _confirm_discard_if_dirty then accept/ignore.
    class FakeEvent:
        def __init__(self):
            self.accepted = None

        def accept(self):
            self.accepted = True

        def ignore(self):
            self.accepted = False

    class CloseStub(StubWin):
        closeEvent = MainWindow.closeEvent

    ev = FakeEvent()
    CloseStub(dirty).closeEvent(ev)
    assert ev.accepted is True
    print("[edge_fillet_verify] closeEvent accepted (unattended dirty)", flush=True)

    # Quit Qt event loop cleanly if any
    app.quit()
    print("[edge_fillet_verify] PASS", flush=True)
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception:
        traceback.print_exc()
        code = 99
    print(f"[edge_fillet_verify] exit_code={code}", flush=True)
    raise SystemExit(code)
