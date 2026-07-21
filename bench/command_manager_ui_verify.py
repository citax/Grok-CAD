#!/usr/bin/env python3
"""Screenshot the redesigned Command Manager (Features / Sketch tabs)."""
from __future__ import annotations

import os
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


def _grab_widget(w, path: Path):
    pix = w.grab()
    pix.save(str(path))
    print(f"SHOT {path}  size={pix.width()}x{pix.height()}", flush=True)
    return path


def main() -> int:
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QRect

    from app.mainwindow import MainWindow
    from app.theme import apply_theme
    from cadcore.document import FeatureType

    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app)
    win = MainWindow()
    win.resize(1280, 800)
    win.show()
    _pump(app, 35)

    if not win.isVisible():
        print("FAIL window not visible", flush=True)
        return 1

    # 1) Features tab default
    if win.cmd_tabs is not None:
        win.cmd_tabs.setCurrentIndex(0)
    _pump(app, 15)
    _grab_widget(win, OUT / "cmd_mgr_features_tab.png")
    # Crop top command manager region
    full = win.grab()
    # Top ~110 px of window (menu + cmd manager)
    crop = full.copy(0, 0, full.width(), min(120, full.height()))
    crop.save(str(OUT / "cmd_mgr_features_strip.png"))
    print(f"SHOT {OUT / 'cmd_mgr_features_strip.png'}", flush=True)

    # Measure button sizes
    btns = win.findChildren(__import__("PySide6.QtWidgets", fromlist=["QToolButton"]).QToolButton)
    cmd_btns = [b for b in btns if b.objectName() in ("CmdStripButton", "CmdIconButton")]
    if cmd_btns:
        b0 = cmd_btns[0]
        print(
            f"OK sample CmdStripButton size={b0.width()}x{b0.height()} "
            f"icon={b0.iconSize().width()}x{b0.iconSize().height()} "
            f"count={len(cmd_btns)}",
            flush=True,
        )
        if b0.width() > 70 or b0.height() > 68:
            print(f"FAIL buttons still large: {b0.width()}x{b0.height()}", flush=True)
            return 1
        # Labels under icons need vertical room (was 44px — captions clipped)
        if b0.height() < 52:
            print(f"FAIL buttons too short for labels: {b0.height()}", flush=True)
            return 1
        print(
            f"OK buttons fit captions ({b0.width()}x{b0.height()}, expect ~52x58)",
            flush=True,
        )
    else:
        print("FAIL no CmdStripButton found", flush=True)
        return 1

    # 2) Enter sketch → Sketch tab
    front = next(f for f in win.doc.features if f.type is FeatureType.PLANE_FRONT)
    win.doc.selected_id = front.id
    win._sync_selection(front.id)
    win._enter_sketch()
    _pump(app, 25)
    if win.cmd_tabs is not None:
        print(f"OK sketch tab index={win.cmd_tabs.currentIndex()} (expect {win._sketch_tab_index})", flush=True)
    crop = win.grab().copy(0, 0, win.width(), min(120, win.height()))
    crop.save(str(OUT / "cmd_mgr_sketch_strip.png"))
    print(f"SHOT {OUT / 'cmd_mgr_sketch_strip.png'}", flush=True)
    _grab_widget(win, OUT / "cmd_mgr_sketch_full.png")

    # 3) Evaluate tab
    if win.cmd_tabs is not None:
        win.cmd_tabs.setCurrentIndex(2)
    _pump(app, 10)
    crop = win.grab().copy(0, 0, win.width(), min(120, win.height()))
    crop.save(str(OUT / "cmd_mgr_evaluate_strip.png"))
    print(f"SHOT {OUT / 'cmd_mgr_evaluate_strip.png'}", flush=True)

    # Tab bar height / overall command manager height
    bar = win.sketch_tb
    print(f"OK CommandManagerBar height={bar.height()} (target ~90–110)", flush=True)
    if bar.height() > 120:
        print(f"FAIL command manager too tall: {bar.height()}", flush=True)
        return 1
    if bar.height() < 70:
        print(
            f"FAIL command manager too short (labels may clip): {bar.height()}",
            flush=True,
        )
        return 1

    win._exit_sketch()
    _pump(app, 10)
    win.close()
    _pump(app, 5)
    print("PASS command manager UI redesign", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
