#!/usr/bin/env python3
"""Measure sketch-mode per-move render cost at large window sizes (real MainWindow).

Benchmarks at 1920×1080 and 2560×1440 — the fill-rate-bound resolutions where
draw-time jank actually shows up under software GL.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("QT_XCB_GL_INTEGRATION", "none")


def _run_at_size(win, app, *, w: int, h: int, entities: int, moves: int, tag: str) -> dict:
    win.resize(w, h)
    for _ in range(15):
        app.processEvents()
        time.sleep(0.012)
    # Force render at full size first
    if win.viewport._render_timer.isActive():
        win.viewport._render_timer.stop()
    win.viewport._do_render()
    app.processEvents()

    from app.sketch_mode import SketchTool
    from cadcore.document import FeatureType

    # Fresh sketch each size so entity count is clean
    front = next(f for f in win.doc.features if f.type is FeatureType.PLANE_FRONT)
    # Reuse or create sketch
    skf = None
    for f in win.doc.features:
        if f.type is FeatureType.SKETCH and f.sketch is not None:
            skf = f
            break
    if skf is None:
        skf = win.doc.create_sketch_on_plane(front.id)
    sk = skf.sketch
    # pad entities to target
    while len(sk.entities) < entities:
        i = len(sk.entities)
        cols = int(entities ** 0.5) + 1
        r, c = divmod(i, cols)
        sk.add_line((c * 0.25, r * 0.25), (c * 0.25 + 0.15, r * 0.25))

    if not win.viewport.in_sketch_mode:
        win.viewport.enter_sketch(skf.id)
    for _ in range(12):
        app.processEvents()
        time.sleep(0.01)

    ctrl = win.viewport._sketch_ctrl
    assert ctrl is not None
    ctrl.set_tool(SketchTool.LINE)
    ctrl.on_press((0.0, -2.0))
    if hasattr(win.viewport, "_begin_draw_lod"):
        win.viewport._begin_draw_lod()

    # Report actual render size after LOD (should be ~half if draw LOD applied)
    try:
        rw = win.viewport.plotter.render_window.GetSize()
        rsz = (int(rw[0]), int(rw[1]))
    except Exception:
        rsz = (0, 0)

    times = []
    for i in range(moves):
        u = 0.05 * i
        v = -2.0 + 0.02 * i
        t0 = time.perf_counter()
        ctrl.on_move((u, v))
        win.viewport._update_preview_visual()
        if win.viewport._render_timer.isActive():
            win.viewport._render_timer.stop()
        win.viewport._do_render()
        app.processEvents()
        times.append((time.perf_counter() - t0) * 1000.0)

    if hasattr(win.viewport, "_end_draw_lod"):
        win.viewport._end_draw_lod()
    if win.viewport.in_sketch_mode:
        # leave chain idle
        if ctrl.is_drawing():
            ctrl.end_line_chain() if hasattr(ctrl, "end_line_chain") else ctrl.cancel_drawing()

    mean = statistics.mean(times)
    med = statistics.median(times)
    p95 = sorted(times)[int(0.95 * (len(times) - 1))]
    lod = getattr(win.viewport, "_draw_lod_active", None)
    print(
        f"DRAW_LAG tag={tag} size={w}x{h} render_size={rsz[0]}x{rsz[1]} "
        f"n={entities} moves={moves} mean={mean:.2f}ms median={med:.2f}ms "
        f"p95={p95:.2f}ms draw_lod_was_active_during={rsz[0] < w if w else '?'}",
        flush=True,
    )
    return {
        "w": w,
        "h": h,
        "mean": mean,
        "median": med,
        "p95": p95,
        "render_size": rsz,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run")
    ap.add_argument("--entities", type=int, default=300)
    ap.add_argument("--moves", type=int, default=40)
    ap.add_argument("--platform", default="xcb")
    ap.add_argument(
        "--sizes",
        default="1920x1080,2560x1440",
        help="comma-separated WxH list",
    )
    args = ap.parse_args()
    os.environ["QT_QPA_PLATFORM"] = args.platform

    from PySide6.QtWidgets import QApplication

    from app.mainwindow import MainWindow
    from app.theme import apply_theme

    app = QApplication(sys.argv)
    apply_theme(app)
    win = MainWindow()
    win.show()
    for _ in range(20):
        app.processEvents()
        time.sleep(0.01)
    if not win.viewport._ok:
        print(f"DRAW_LAG_FAIL tag={args.tag}", flush=True)
        return 1

    results = []
    for part in args.sizes.split(","):
        part = part.strip().lower()
        if "x" not in part:
            continue
        ws, hs = part.split("x", 1)
        w, h = int(ws), int(hs)
        results.append(
            _run_at_size(
                win,
                app,
                w=w,
                h=h,
                entities=args.entities,
                moves=args.moves,
                tag=args.tag,
            )
        )

    # Summary line
    bits = " ".join(f"{r['w']}x{r['h']}={r['mean']:.2f}ms" for r in results)
    print(f"DRAW_LAG_DONE tag={args.tag} n={args.entities} {bits}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
