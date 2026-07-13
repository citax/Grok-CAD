#!/usr/bin/env python3
"""Measure sketch-mode per-move render cost at N entities (real MainWindow)."""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("QT_XCB_GL_INTEGRATION", "none")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run")
    ap.add_argument("--entities", type=int, default=300)
    ap.add_argument("--moves", type=int, default=40)
    ap.add_argument("--platform", default="xcb")
    args = ap.parse_args()
    os.environ["QT_QPA_PLATFORM"] = args.platform

    from PySide6.QtWidgets import QApplication

    from app.mainwindow import MainWindow
    from app.sketch_mode import SketchTool
    from app.theme import apply_theme
    from cadcore.document import FeatureType

    app = QApplication(sys.argv)
    apply_theme(app)
    win = MainWindow()
    win.resize(1280, 800)
    win.show()
    for _ in range(20):
        app.processEvents()
        time.sleep(0.01)
    if not win.viewport._ok:
        print(f"DRAW_LAG_FAIL tag={args.tag}", flush=True)
        return 1

    front = next(f for f in win.doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = win.doc.create_sketch_on_plane(front.id)
    # seed many entities
    sk = skf.sketch
    cols = int(args.entities ** 0.5) + 1
    for i in range(args.entities):
        r, c = divmod(i, cols)
        sk.add_line((c * 0.25, r * 0.25), (c * 0.25 + 0.15, r * 0.25))
    win.viewport.enter_sketch(skf.id)
    for _ in range(15):
        app.processEvents()
        time.sleep(0.01)
    ctrl = win.viewport._sketch_ctrl
    ctrl.set_tool(SketchTool.LINE)
    # start a draw so LOD path activates (after fix)
    ctrl.on_press((0.0, -2.0))
    win.viewport._begin_draw_lod() if hasattr(win.viewport, "_begin_draw_lod") else None

    times = []
    for i in range(args.moves):
        u = 0.05 * i
        v = -2.0 + 0.02 * i
        t0 = time.perf_counter()
        # Simulate move: snap update + preview + render (real path)
        ctrl.on_move((u, v))
        win.viewport._update_preview_visual()
        if win.viewport._render_timer.isActive():
            win.viewport._render_timer.stop()
        win.viewport._do_render()
        app.processEvents()
        times.append((time.perf_counter() - t0) * 1000.0)

    mean = statistics.mean(times)
    med = statistics.median(times)
    p95 = sorted(times)[int(0.95 * (len(times) - 1))]
    lod = getattr(win.viewport, "_draw_lod_active", None)
    print(
        f"DRAW_LAG tag={args.tag} n={args.entities} moves={args.moves} "
        f"mean={mean:.2f}ms median={med:.2f}ms p95={p95:.2f}ms "
        f"draw_lod={lod}",
        flush=True,
    )
    print(
        f"DRAW_LAG_DONE tag={args.tag} n={args.entities} mean={mean:.2f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
