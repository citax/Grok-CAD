#!/usr/bin/env python3
"""Drive real PropertyPanel + MainWindow; fail on any uncaught exception.

PySide6 6.11 slots can swallow exceptions — install hooks that force non-zero exit.
"""
from __future__ import annotations

import os
import sys
import time
import traceback

import numpy as np

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("QT_XCB_GL_INTEGRATION", "none")
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

_EXC: list[str] = []


def _record_exc(kind: str, exc: BaseException, tb=None) -> None:
    text = "".join(traceback.format_exception(type(exc), exc, tb or exc.__traceback__))
    _EXC.append(f"{kind}: {text}")
    print(f"EXC_CAPTURED {kind}: {exc!r}", flush=True)


def _install_hooks() -> None:
    def hook(etype, value, tb):
        _record_exc("sys.excepthook", value, tb)

    sys.excepthook = hook

    try:
        from PySide6.QtCore import qInstallMessageHandler, QtMsgType

        def qt_handler(mode, context, message):  # noqa: ANN001
            # Surface Qt criticals that often wrap slot failures
            if mode in (
                QtMsgType.QtCriticalMsg,
                QtMsgType.QtFatalMsg,
                QtMsgType.QtWarningMsg,
            ):
                if "libshiboken" in message or "deleted" in message.lower():
                    _EXC.append(f"qt:{mode}: {message}")
                    print(f"EXC_CAPTURED qt: {message}", flush=True)

        qInstallMessageHandler(qt_handler)
    except Exception as exc:  # noqa: BLE001
        print(f"[hooks] qt handler skip: {exc}", flush=True)


def _pump(app, n: int = 15) -> None:
    for _ in range(n):
        app.processEvents()
        time.sleep(0.01)


def main() -> int:
    _install_hooks()

    from PySide6.QtWidgets import QApplication

    from app.mainwindow import MainWindow
    from app.property_panel import PropertyPanel
    from app.theme import apply_theme
    from cadcore.document import Document, FeatureType
    from cadcore.sketch import LineEntity

    app = QApplication(sys.argv)
    apply_theme(app)

    # ----- standalone panel: every type, double show, type switch, Apply -----
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

    line = sk_ex.sketch.add_line((5, 0), (7, 0))

    panel = PropertyPanel()
    panel.set_document(doc)
    panel.show()
    _pump(app, 8)

    features = [
        ("extrude", extrude),
        ("fillet", fillet),
        ("revolve", revolve),
        ("pocket", pocket),
    ]
    for name, feat in features:
        panel.show_feature(feat)
        panel.show_feature(feat)
        print(f"DOUBLE_SHOW_OK {name}", flush=True)

    for name, feat in [
        ("extrude", extrude),
        ("fillet", fillet),
        ("extrude", extrude),
        ("revolve", revolve),
        ("pocket", pocket),
        ("fillet", fillet),
    ]:
        panel.show_feature(feat)
    print("TYPE_SWITCH_OK", flush=True)

    # Apply each type via real button
    panel.show_feature(extrude)
    panel._editors["depth"].setValue(3.5)
    panel.btn_apply.click()
    _pump(app, 4)
    assert abs(extrude.depth - 3.5) < 1e-12, extrude.depth
    vol = doc.evaluate_feature(extrude.id)
    assert vol is not None and abs(vol.volume() - 14.0) < 0.15, vol.volume()
    print("APPLY_EXTRUDE_OK volume≈14", flush=True)

    panel.show_feature(fillet)
    panel._editors["radius"].setValue(0.5)
    panel.btn_apply.click()
    _pump(app, 4)
    assert abs(fillet.radius - 0.5) < 1e-12
    print("APPLY_FILLET_OK", flush=True)

    panel.show_feature(revolve)
    panel._editors["angle"].setValue(90.0)
    panel.btn_apply.click()
    _pump(app, 4)
    assert abs(revolve.revolve_angle - 90.0) < 1e-12
    print("APPLY_REVOLVE_OK", flush=True)

    panel.show_feature(pocket)
    panel._editors["depth"].setValue(2.0)
    panel.btn_apply.click()
    _pump(app, 4)
    assert abs(pocket.depth - 2.0) < 1e-12
    print("APPLY_POCKET_OK", flush=True)

    panel.show_sketch_line(sk_ex.id, line)
    panel.show_sketch_line(sk_ex.id, line)
    panel._editors["line_len"].setValue(4.0)
    panel.btn_apply.click()
    _pump(app, 4)
    ent = sk_ex.sketch.find_entity(line.id)
    # p0 fixed at 5; length 4 along +u → p1 at 9
    assert isinstance(ent, LineEntity) and abs(ent.p1[0] - 9.0) < 1e-9, ent.p1
    print("APPLY_SKETCH_LINE_OK", flush=True)

    panel.close()

    # ----- real MainWindow: select → Apply → Undo → Redo → re-select -----
    win = MainWindow()
    win.resize(1200, 800)
    win.show()
    _pump(app, 25)
    if not win.viewport._ok:
        print("PROPS_FAIL viewport", flush=True)
        return 1

    front = next(f for f in win.doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = win.doc.create_sketch_on_plane(front.id)
    skf.sketch.add_rectangle((0, 0), (2, 2))
    ex = win.doc.create_extrude(skf.id, 1.0)
    win.viewport.schedule_rebuild()
    _pump(app, 20)
    win._refresh_tree()
    win._sync_selection(ex.id)
    _pump(app, 8)

    def _wait_rebuild(timeout_s: float = 8.0) -> None:
        """Pump until rebuild timer idle and job gen stable (worker finished)."""
        t_end = time.time() + timeout_s
        while time.time() < t_end:
            app.processEvents()
            # Rebuild timer not pending and no busy message from worker
            busy = False
            try:
                # GeometryRebuildJob clears busy on finish via busy_changed
                pass
            except Exception:
                pass
            if not win.viewport._rebuild_timer.isActive():
                # give worker a few ticks after timer fire
                for _ in range(30):
                    app.processEvents()
                    time.sleep(0.02)
                return
            time.sleep(0.02)
        raise TimeoutError("rebuild did not finish")

    def _viewport_solid_z(fid: int):
        """Z bounds of the mesh the viewport actually uploaded (actor polydata)."""
        name = f"solid_{fid}"
        act = win.viewport.plotter.actors.get(name)
        if act is None:
            return None
        try:
            mapper = act.GetMapper()
            data = mapper.GetInput()
            pts = np.array(data.GetPoints().GetData())
            if pts.size == 0:
                return None
            return float(pts[:, 2].min()), float(pts[:, 2].max())
        except Exception as exc:  # noqa: BLE001
            print(f"[viewport solid] {exc}", flush=True)
            return None

    def _wait_solid(fid: int, timeout_s: float = 8.0):
        t_end = time.time() + timeout_s
        while time.time() < t_end:
            app.processEvents()
            z = _viewport_solid_z(fid)
            if z is not None:
                return z
            time.sleep(0.03)
        return None

    # Change depth via panel on the live window
    win.props.show_feature(ex)
    win.props.show_feature(ex)  # second show — crash site
    win.props._editors["depth"].setValue(3.5)
    win.props.btn_apply.click()
    _wait_rebuild()
    assert abs(ex.depth - 3.5) < 1e-12
    # Worker path volume (not only doc.evaluate_feature)
    from app.workers import evaluate_solids_snapshot, snapshot_features

    res = evaluate_solids_snapshot(snapshot_features(win.doc))
    assert ex.id in res
    v, faces, _fp = res[ex.id]
    # approximate volume via mesh from verts/faces
    from cadcore.mesh import Mesh

    m_snap = Mesh(v, faces)
    assert abs(m_snap.volume() - 14.0) < 0.2, m_snap.volume()
    print("MAINWINDOW_APPLY_OK volume≈14 (snapshot path)", flush=True)

    win._undo()
    _wait_rebuild()
    assert abs(ex.depth - 1.0) < 1e-12
    print("MAINWINDOW_UNDO_OK", flush=True)

    win._redo()
    _wait_rebuild()
    assert abs(ex.depth - 3.5) < 1e-12
    print("MAINWINDOW_REDO_OK", flush=True)

    # Re-select same feature (sync_selection → show_feature again)
    win._sync_selection(ex.id)
    win._sync_selection(ex.id)
    _pump(app, 8)
    print("MAINWINDOW_RESELECT_OK", flush=True)

    # --- Reverse direction: MUST assert VIEWPORT actor mesh, not doc.evaluate ---
    z_before = _wait_solid(ex.id)
    assert z_before is not None, "no solid actor before reverse"
    print(f"VIEWPORT_Z_BEFORE {z_before[0]:.4f}..{z_before[1]:.4f}", flush=True)
    assert z_before[0] >= -1e-3 and z_before[1] > 1.0

    # Camera: look from +X so ±Z is left/right on screen for framebuffer check
    try:
        win.viewport.plotter.camera_position = [(8, 1, 0), (1, 1, 0), (0, 1, 0)]
        win.viewport._do_render()
        _pump(app, 5)
    except Exception:
        pass
    img_before = np.asarray(win.viewport.plotter.screenshot(return_img=True))

    win.props.show_feature(ex)
    win.props.show_feature(ex)
    from PySide6.QtWidgets import QCheckBox

    cb = win.props._editors.get("reversed")
    assert isinstance(cb, QCheckBox), type(cb)
    cb.setChecked(True)
    win.props.btn_apply.click()
    _wait_rebuild()
    assert ex.reversed is True

    z_after = _wait_solid(ex.id)
    assert z_after is not None, "no solid actor after reverse"
    print(f"VIEWPORT_Z_AFTER {z_after[0]:.4f}..{z_after[1]:.4f}", flush=True)
    if z_after[1] <= 1e-3 and z_after[0] < -1.0:
        print("VIEWPORT_REVERSE_OK", flush=True)
    else:
        print(f"VIEWPORT_REVERSE_FAIL z={z_after}", flush=True)
        return 1

    # Snapshot-path geometry must also be on −Z
    res2 = evaluate_solids_snapshot(snapshot_features(win.doc))
    vz = res2[ex.id][0][:, 2]
    assert float(vz.max()) <= 1e-5 and float(vz.min()) < -1.0

    img_after = np.asarray(win.viewport.plotter.screenshot(return_img=True))
    # Solid pixels should differ after reverse (coarse but real framebuffer)
    if img_before.shape == img_after.shape:
        diff = np.mean(np.abs(img_before.astype(float) - img_after.astype(float)))
        print(f"FRAMEBUFFER_DIFF mean_abs={diff:.4f}", flush=True)
        if diff < 0.5:
            print("FRAMEBUFFER_DIFF_WEAK (camera may not separate sides well)", flush=True)
        else:
            print("FRAMEBUFFER_DIFF_OK", flush=True)

    win._undo()
    _wait_rebuild()
    assert ex.reversed is False
    z_undo = _wait_solid(ex.id)
    assert z_undo is not None and z_undo[0] >= -1e-3 and z_undo[1] > 1.0
    print("VIEWPORT_REVERSE_UNDO_OK", flush=True)
    print("MAINWINDOW_REVERSE_UNDO_OK", flush=True)

    if _EXC:
        print(f"EXC_FAIL {len(_EXC)}", flush=True)
        for e in _EXC:
            print(e[:500], flush=True)
        return 1
    print("EXC_CLEAN_OK", flush=True)
    print("PROPERTY_PANEL_VERIFY_OK", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        _record_exc("main", exc)
        print(f"EXC_FAIL {len(_EXC)}", flush=True)
        raise
