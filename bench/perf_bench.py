#!/usr/bin/env python3
"""Real viewport refresh/render timing for N entities + resize + edit cycles.

Uses the REAL MainWindow / Viewport plotter (no synthetic Plotter).
Measures wall time of sync_sketch_visuals + _do_render for 50/200/500 entities,
simulates maximize/resize, exercises add→paste→undo, checks ghosts + on-plane.
Also reports pure _do_render cost (interactive path).
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import numpy as np

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("QT_XCB_GL_INTEGRATION", "none")


def max_plane_dev(sk) -> float:
    fr = sk.frame
    n = np.asarray(fr.normal, float)
    o = np.asarray(fr.origin, float)
    worst = 0.0
    from cadcore.sketch import LineEntity

    for e in sk.entities:
        if not isinstance(e, LineEntity):
            continue
        for uv in (e.p0, e.p1):
            w = fr.to_world(uv)
            worst = max(worst, abs(float(np.dot(n, w - o))))
    return worst


def force_render(vp, app, n=1):
    if vp._render_timer.isActive():
        vp._render_timer.stop()
    for _ in range(n):
        vp._do_render()
        app.processEvents()


def measure_sync_render(vp, app, repeats=5):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        vp.sync_sketch_visuals()
        if vp._render_timer.isActive():
            vp._render_timer.stop()
        vp._do_render()
        app.processEvents()
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


def measure_pure_render(vp, app, repeats=6):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        vp._do_render()
        app.processEvents()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.mean(times)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default=os.environ.get("QT_QPA_PLATFORM", "xcb"))
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()
    os.environ["QT_QPA_PLATFORM"] = args.platform

    from PySide6.QtWidgets import QApplication

    from app.mainwindow import MainWindow
    from app.theme import apply_theme
    from cadcore.document import FeatureType

    app = QApplication(sys.argv)
    apply_theme(app)
    win = MainWindow()
    win.resize(960, 720)
    win.show()
    for _ in range(20):
        app.processEvents()
        time.sleep(0.01)
    if not win.viewport._ok:
        print("PERF_BENCH_FAIL: viewport not ok", flush=True)
        return 1

    front = next(f for f in win.doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = win.doc.create_sketch_on_plane(front.id)
    win.viewport.enter_sketch(skf.id)
    app.processEvents()
    ctrl = win.viewport._sketch_ctrl
    sk = ctrl.sketch
    results = {}

    def seed_entities(n: int):
        for e in list(sk.entities):
            sk.remove_entity(e.id)
        win.viewport.sync_sketch_visuals()
        cols = int(np.ceil(np.sqrt(n)))
        for i in range(n):
            r, c = divmod(i, cols)
            p0 = (c * 0.4, r * 0.4)
            p1 = (p0[0] + 0.3, p0[1])
            e = sk.add_line(p0, p1)
            win.doc.record_entity_add(skf.id, e)
        win.viewport.sync_sketch_visuals()
        force_render(win.viewport, app, 2)

    for n in (50, 200, 500):
        seed_entities(n)
        stats = measure_sync_render(win.viewport, app, repeats=5)
        pure = measure_pure_render(win.viewport, app, repeats=6)
        results[n] = stats
        results[f"pure_{n}"] = pure
        print(
            f"PERF n={n} sync+render mean={stats['mean_ms']:.2f}ms "
            f"median={stats['median_ms']:.2f}ms pure_render={pure:.2f}ms "
            f"tag={args.tag}",
            flush=True,
        )
        sk_e = [k for k in win.viewport._overlay_actors if k.startswith("sk_e_")]
        if len(sk_e) != n:
            print(f"PERF_BENCH_FAIL: actors have={len(sk_e)} want={n}", flush=True)
            return 1
        if max_plane_dev(sk) >= 1e-6:
            print("PERF_BENCH_FAIL: off-plane", flush=True)
            return 1

    seed_entities(200)
    resize_times = []
    for w, h in ((1280, 800), (1920, 1080), (800, 600), (1600, 900), (960, 720)):
        t0 = time.perf_counter()
        win.resize(w, h)
        win.viewport._resizing = True
        win.viewport._request_render()
        for _ in range(10):
            app.processEvents()
            time.sleep(0.01)
        win.viewport._resizing = False
        if win.viewport._render_timer.isActive():
            win.viewport._render_timer.stop()
        win.viewport._do_render()
        app.processEvents()
        resize_times.append((time.perf_counter() - t0) * 1000.0)
    print(
        f"PERF resize/maximize mean={statistics.mean(resize_times):.2f}ms "
        f"median={statistics.median(resize_times):.2f}ms tag={args.tag}",
        flush=True,
    )

    seed_entities(10)
    e0 = sk.entities[0]
    win.doc.copy_entity(skf.id, e0.id)
    before = len(sk.entities)
    p1 = win.doc.paste_entity(skf.id, place_uv=(50.0, 50.0))
    p2 = win.doc.paste_entity(skf.id, place_uv=(50.0, 50.0))
    win.viewport.sync_sketch_visuals()
    force_render(win.viewport, app)
    assert p1 is not None and p2 is not None and p1.p0 != p2.p0
    win.doc.undo()
    win.viewport.sync_sketch_visuals()
    force_render(win.viewport, app)
    ghost = f"sk_e_{p2.id}"
    if ghost in win.viewport._overlay_actors:
        print(f"PERF_BENCH_FAIL: ghost after undo {ghost}", flush=True)
        return 1
    print(
        f"PERF edit_cycle ok pastes_distinct=True undo_no_ghost=True "
        f"entities {before}->{len(sk.entities)} tag={args.tag}",
        flush=True,
    )

    print(
        f"PERF_BENCH_DONE tag={args.tag} "
        f"n50={results[50]['mean_ms']:.2f}/{results['pure_50']:.2f} "
        f"n200={results[200]['mean_ms']:.2f}/{results['pure_200']:.2f} "
        f"n500={results[500]['mean_ms']:.2f}/{results['pure_500']:.2f} "
        f"resize={statistics.mean(resize_times):.2f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
