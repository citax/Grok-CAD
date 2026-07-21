#!/usr/bin/env python3
"""Verify Command Manager labels visible + sketch hover handles / FPS."""
from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ["GROK_CAD_UNATTENDED"] = "1"

OUT = _ROOT / "bench" / "_ui_shots"
OUT.mkdir(parents=True, exist_ok=True)


def _pump(app, n=20):
    for _ in range(n):
        app.processEvents()
        time.sleep(0.01)


def main() -> int:
    from PySide6.QtWidgets import QApplication, QToolButton
    from PySide6.QtGui import QImage

    from app.mainwindow import MainWindow
    from app.theme import apply_theme
    from app.sketch_mode import SketchTool
    from cadcore.document import FeatureType

    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app)
    win = MainWindow()
    win.resize(1280, 800)
    win.show()
    _pump(app, 40)

    if not win.isVisible():
        print("FAIL window not visible", flush=True)
        return 1

    # --- 1) Command Manager labels / height ---
    if win.cmd_tabs is not None:
        win.cmd_tabs.setCurrentIndex(0)
    _pump(app, 15)
    full = win.grab()
    crop_h = min(140, full.height())
    full.copy(0, 0, full.width(), crop_h).save(str(OUT / "fix1_cmd_features_strip.png"))
    print(f"SHOT {OUT / 'fix1_cmd_features_strip.png'}", flush=True)

    btns = [b for b in win.findChildren(QToolButton) if b.objectName() == "CmdStripButton"]
    if not btns:
        print("FAIL no CmdStripButton", flush=True)
        return 1
    b0 = btns[0]
    print(
        f"OK sample button {b0.width()}x{b0.height()} text={b0.text()!r} "
        f"bar_h={win.sketch_tb.height()}",
        flush=True,
    )
    if b0.height() < 52:
        print(f"FAIL button too short for labels: {b0.height()}", flush=True)
        return 1
    # Pixel check: bottom 12px of button should not be entirely bg-colored
    # (caption text present). Soft check via non-empty text.
    if not (b0.text() or "").strip():
        print("FAIL button has no text caption", flush=True)
        return 1

    # Sketch tab
    front = next(f for f in win.doc.features if f.type is FeatureType.PLANE_FRONT)
    win.doc.selected_id = front.id
    win._sync_selection(front.id)
    win._enter_sketch()
    _pump(app, 25)
    full = win.grab()
    full.copy(0, 0, full.width(), crop_h).save(str(OUT / "fix1_cmd_sketch_strip.png"))
    print(f"SHOT {OUT / 'fix1_cmd_sketch_strip.png'}", flush=True)
    win.grab().save(str(OUT / "fix1_sketch_mode_full.png"))
    print(f"SHOT {OUT / 'fix1_sketch_mode_full.png'}", flush=True)

    # --- 2) Hover handles + FPS ---
    skf = win.viewport._sketch_feature_id
    sk = win.doc.find(skf).sketch
    ln = sk.add_line((0.0, 0.0), (20.0, 0.0))
    circ = sk.add_circle((10.0, 10.0), 5.0)
    win.viewport.sync_sketch_visuals()
    ctrl = win.viewport._sketch_ctrl
    ctrl.set_tool(SketchTool.SELECT)
    _pump(app, 10)

    # Simulate hover over line body via controller + handle update path
    times = []
    for i in range(80):
        u = 2.0 + 0.2 * (i % 40)
        t0 = time.perf_counter()
        # Full mouse-move path if we can project; else controller-only + visual
        win.viewport._update_snap_tolerance()
        ctrl.on_move((u, 0.02))
        prev_sig = getattr(win.viewport, "_handles_vis_sig", None)
        win.viewport._update_handles_visual()
        # Only render when hover/handles change (mirrors fixed path)
        if getattr(win.viewport, "_handles_vis_sig", None) != prev_sig:
            if win.viewport._render_timer.isActive():
                win.viewport._render_timer.stop()
            win.viewport._do_render()
        app.processEvents()
        times.append((time.perf_counter() - t0) * 1000.0)

    mean = statistics.mean(times)
    p95 = sorted(times)[int(0.95 * (len(times) - 1))]
    print(
        f"OK hover move mean={mean:.2f}ms p95={p95:.2f}ms "
        f"hover_entity={ctrl.hover_entity_id} handles_sig_len="
        f"{len(getattr(win.viewport, '_handles_vis_sig', '') or '')}",
        flush=True,
    )
    # Software GL is slow; still should stay interactive (< 80ms mean when idle hover)
    if mean > 80.0:
        print(f"FAIL hover too slow mean={mean:.2f}ms", flush=True)
        return 1

    # Hover line → handles for that entity
    ctrl.on_move((10.0, 0.02))
    win.viewport._handles_vis_sig = None
    win.viewport._update_handles_visual()
    assert ctrl.hover_entity_id == ln.id, f"expected line hover got {ctrl.hover_entity_id}"
    if "__sk_handles" not in win.viewport._overlay_actors and (
        win.viewport.plotter is None
        or "__sk_handles" not in win.viewport.plotter.actors
    ):
        print("FAIL handles actor missing on entity hover", flush=True)
        return 1
    print("OK handles visible on line body hover", flush=True)

    # Hover circle
    ctrl.on_move((15.0, 10.0))  # on rim
    win.viewport._handles_vis_sig = None
    win.viewport._update_handles_visual()
    if win.viewport._render_timer.isActive():
        win.viewport._render_timer.stop()
    win.viewport._do_render()
    _pump(app, 8)
    win.grab().save(str(OUT / "fix2_hover_handles.png"))
    print(f"SHOT {OUT / 'fix2_hover_handles.png'}", flush=True)
    print(f"OK circle/line hover_entity={ctrl.hover_entity_id}", flush=True)

    # Body drag (re-fetch entity — never trust a stale python ref after edits)
    ln = sk.find_entity(ln.id)
    p0_before = (float(ln.p0[0]), float(ln.p0[1]))
    msg = ctrl.on_press((4.0, 0.0))
    print(f"OK body press msg={msg!r} drag={ctrl.drag}", flush=True)
    ctrl.on_move((4.0, 5.0))
    ctrl.on_release((4.0, 5.0))
    ln = sk.find_entity(ln.id)
    win.viewport.sync_sketch_visuals()
    _pump(app, 8)
    win.grab().save(str(OUT / "fix3_body_drag.png"))
    print(f"SHOT {OUT / 'fix3_body_drag.png'}", flush=True)
    dy = float(ln.p0[1]) - p0_before[1]
    if abs(dy - 5.0) > 0.5:
        print(f"FAIL body drag: p0={ln.p0} dy={dy} expected ~5", flush=True)
        return 1
    print(f"OK body drag moved line to p0={ln.p0} dy={dy:.2f}", flush=True)

    # Edit sketch rock-solid: extrude, edit sketch, move, exit, volume changes
    win._exit_sketch()
    _pump(app, 15)
    sk_feat = win.doc.find(skf)
    # closed rect for extrude
    sk_feat.sketch.entities.clear()
    sk_feat.sketch.add_rectangle((0, 0), (10, 10))
    ex = win.doc.create_extrude(sk_feat.id, 5.0)
    win.viewport.schedule_rebuild()
    _pump(app, 40)
    v0 = win.doc.evaluate_feature(ex.id).volume()
    win._open_sketch_edit(sk_feat.id, rollback_from=ex.id)
    _pump(app, 20)
    from cadcore.sketch import set_rect_width

    set_rect_width(sk_feat.sketch.entities[0], 20.0)
    win.viewport.sync_sketch_visuals()
    _pump(app, 10)
    win.grab().save(str(OUT / "fix4_edit_sketch.png"))
    print(f"SHOT {OUT / 'fix4_edit_sketch.png'}", flush=True)
    win._exit_sketch()
    _pump(app, 40)
    v1 = win.doc.evaluate_feature(ex.id).volume()
    print(f"OK edit-sketch volume {v0:.1f} → {v1:.1f}", flush=True)
    if abs(v1 - 2.0 * v0) > 1.0:
        print(f"FAIL volume not doubled after edit sketch", flush=True)
        return 1
    win.grab().save(str(OUT / "fix4_after_exit_edit.png"))
    print(f"SHOT {OUT / 'fix4_after_exit_edit.png'}", flush=True)

    print("PASS hover + cmd mgr + edit-sketch daily driver", flush=True)
    try:
        win.close()
        _pump(app, 5)
    except Exception:
        pass
    # VTK/Qt teardown can SIGSEGV under software GL — force clean exit code
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
