#!/usr/bin/env python3
"""Large-window viewport render timing via REAL MainWindow (git before/after).

Measures pure plotter.render() at 1920×1080 and 2560×1440 plus simulated maximize.
Does NOT toggle flags — run on two code trees (worktree checkout).
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
    ap.add_argument("--tag", default="run")
    ap.add_argument("--platform", default="xcb")
    args = ap.parse_args()
    os.environ["QT_QPA_PLATFORM"] = args.platform

    from PySide6.QtWidgets import QApplication

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
        print("VIEWPORT_LARGE_BENCH_FAIL: no viewport", flush=True)
        return 1

    # Seed a modest scene so fill-rate dominates at large sizes
    front = next(f for f in win.doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = win.doc.create_sketch_on_plane(front.id)
    sk = skf.sketch
    # closed square loop for realism
    sk.add_line((0, 0), (2, 0))
    sk.add_line((2, 0), (2, 2))
    sk.add_line((2, 2), (0, 2))
    sk.add_line((0, 2), (0, 0))
    try:
        win.doc.create_extrude(skf.id, 1.0)
        win.viewport.schedule_rebuild()
        for _ in range(30):
            app.processEvents()
            time.sleep(0.02)
    except Exception as exc:
        print(f"note: extrude seed skipped: {exc}", flush=True)

    results = {}
    for w, h in ((1920, 1080), (2560, 1440)):
        win.resize(w, h)
        for _ in range(15):
            app.processEvents()
            time.sleep(0.02)
        if win.viewport._render_timer.isActive():
            win.viewport._render_timer.stop()
        win.viewport._do_render()
        app.processEvents()
        times = []
        for _ in range(8):
            t0 = time.perf_counter()
            win.viewport._do_render()
            app.processEvents()
            times.append((time.perf_counter() - t0) * 1000.0)
        results[(w, h)] = statistics.mean(times)
        print(
            f"LARGE tag={args.tag} size={w}x{h} pure_render_mean={results[(w, h)]:.2f}ms "
            f"median={statistics.median(times):.2f}ms",
            flush=True,
        )

    # Simulated maximize sequence
    max_times = []
    for wh in ((1280, 800), (1920, 1080), (2560, 1440), (1920, 1080), (960, 720)):
        t0 = time.perf_counter()
        win.resize(*wh)
        win.viewport._resizing = True
        win.viewport._request_render()
        for _ in range(10):
            app.processEvents()
            time.sleep(0.015)
        win.viewport._resizing = False
        if win.viewport._render_timer.isActive():
            win.viewport._render_timer.stop()
        win.viewport._do_render()
        app.processEvents()
        max_times.append((time.perf_counter() - t0) * 1000.0)
    print(
        f"LARGE tag={args.tag} maximize_seq mean={statistics.mean(max_times):.2f}ms "
        f"median={statistics.median(max_times):.2f}ms",
        flush=True,
    )

    # Chrome presence probes
    actors = win.viewport.plotter.actors if win.viewport.plotter else {}
    has_bounds = any("bounds" in k.lower() or k.startswith("Axis") for k in actors)
    # show_bounds often names actors like 'CubeAxes' etc.
    cube = [k for k in actors if "cube" in k.lower() or "bounds" in k.lower() or "Axis" in k]
    print(
        f"LARGE tag={args.tag} actors_sample={list(actors.keys())[:12]} "
        f"bounds_like={cube}",
        flush=True,
    )
    print(
        f"VIEWPORT_LARGE_BENCH_DONE tag={args.tag} "
        f"r1920={results[(1920, 1080)]:.2f} r2560={results[(2560, 1440)]:.2f} "
        f"maximize={statistics.mean(max_times):.2f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
