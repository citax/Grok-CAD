#!/usr/bin/env python3
"""Real MainWindow ablation bench: pure render + maximize + interactive drag.

Reads GROK_PERF_METHOD from the environment (none|a|b|c|d|e|final).
"""
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
    ap.add_argument("--method", default=os.environ.get("GROK_PERF_METHOD", "none"))
    ap.add_argument("--entities", type=int, default=300)
    ap.add_argument("--tag", default="")
    ap.add_argument("--platform", default="xcb")
    args = ap.parse_args()
    os.environ["GROK_PERF_METHOD"] = args.method
    os.environ["QT_QPA_PLATFORM"] = args.platform
    tag = args.tag or args.method

    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QEvent, QPoint, Qt
    from PySide6.QtGui import QMouseEvent

    from app.mainwindow import MainWindow
    from app.theme import apply_theme
    from cadcore.document import FeatureType

    app = QApplication(sys.argv)
    apply_theme(app)
    win = MainWindow()
    win.show()
    for _ in range(20):
        app.processEvents()
        time.sleep(0.01)
    if not win.viewport._ok:
        print(f"ABLATION_FAIL method={tag} viewport", flush=True)
        return 1
    print(
        f"ABLATION start method={tag} perf={getattr(win.viewport, '_perf_method', '?')} "
        f"entities={args.entities}",
        flush=True,
    )

    front = next(f for f in win.doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = win.doc.create_sketch_on_plane(front.id)
    sk = skf.sketch
    cols = int(args.entities ** 0.5) + 1
    for i in range(args.entities):
        r, c = divmod(i, cols)
        sk.add_line((c * 0.3, r * 0.3), (c * 0.3 + 0.2, r * 0.3))
    win.viewport.refresh_sketches()
    for _ in range(10):
        app.processEvents()
        time.sleep(0.01)

    results = {}
    for w, h in ((1920, 1080), (2560, 1440)):
        win.resize(w, h)
        for _ in range(12):
            app.processEvents()
            time.sleep(0.015)
        if win.viewport._render_timer.isActive():
            win.viewport._render_timer.stop()
        win.viewport._do_render()
        app.processEvents()
        times = []
        for _ in range(6):
            t0 = time.perf_counter()
            win.viewport._do_render()
            app.processEvents()
            times.append((time.perf_counter() - t0) * 1000.0)
        results[f"r{w}"] = statistics.mean(times)
        print(
            f"ABLATION method={tag} n={args.entities} size={w}x{h} "
            f"pure_render_mean={results[f'r{w}']:.2f}ms",
            flush=True,
        )

    # Simulated maximize
    max_t = []
    for wh in ((1280, 800), (1920, 1080), (2560, 1440), (1920, 1080)):
        t0 = time.perf_counter()
        win.resize(*wh)
        win.viewport._resizing = True
        win.viewport._request_render()
        for _ in range(8):
            app.processEvents()
            time.sleep(0.012)
        win.viewport._resizing = False
        if hasattr(win.viewport, "_end_resize_coalesce"):
            try:
                win.viewport._end_resize_coalesce()
            except Exception:
                pass
        if win.viewport._render_timer.isActive():
            win.viewport._render_timer.stop()
        win.viewport._do_render()
        app.processEvents()
        max_t.append((time.perf_counter() - t0) * 1000.0)
    results["maximize"] = statistics.mean(max_t)
    print(
        f"ABLATION method={tag} n={args.entities} maximize_mean={results['maximize']:.2f}ms",
        flush=True,
    )

    # Interactive drag simulation at 1920×1080
    win.resize(1920, 1080)
    for _ in range(10):
        app.processEvents()
        time.sleep(0.01)
    inter = win.viewport.plotter.interactor if win.viewport.plotter else None
    drag_times = []
    if inter is not None:
        win.viewport._begin_interaction_lod()
        for i in range(12):
            t0 = time.perf_counter()
            # nudge camera
            try:
                cam = win.viewport.plotter.camera
                pos = list(cam.GetPosition())
                pos[0] += 0.05 * ((-1) ** i)
                cam.SetPosition(*pos)
            except Exception:
                pass
            win.viewport._do_render()
            app.processEvents()
            drag_times.append((time.perf_counter() - t0) * 1000.0)
        win.viewport._end_interaction_lod()
        app.processEvents()
    if drag_times:
        mean_drag = statistics.mean(drag_times)
        fps = 1000.0 / mean_drag if mean_drag > 1e-6 else 0.0
        results["drag_ms"] = mean_drag
        results["drag_fps"] = fps
        print(
            f"ABLATION method={tag} n={args.entities} drag_mean={mean_drag:.2f}ms "
            f"eff_fps={fps:.1f}",
            flush=True,
        )

    print(
        f"ABLATION_DONE method={tag} n={args.entities} "
        f"r1920={results.get('r1920', 0):.2f} r2560={results.get('r2560', 0):.2f} "
        f"maximize={results.get('maximize', 0):.2f} "
        f"drag_ms={results.get('drag_ms', 0):.2f} fps={results.get('drag_fps', 0):.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
