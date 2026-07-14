#!/usr/bin/env python3
"""Hard-mode sketch performance: many entities + honest per-move VTK renders.

VTK preview path (layer-1 overlay actor + one light render per mouse-move).
Does NOT chase 0 renders/move. Reports median move ms and renders/move.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("QT_XCB_GL_INTEGRATION", "none")


def _pad_entities(sk, n: int) -> None:
    cols = max(1, int(n**0.5) + 1)
    while len(sk.entities) < n:
        i = len(sk.entities)
        r, c = divmod(i, cols)
        sk.add_line((c * 0.22, r * 0.22), (c * 0.22 + 0.12, r * 0.22 + 0.04))
        if len(sk.entities) < n:
            sk.add_circle((c * 0.22 + 0.06, r * 0.22 + 0.10), 0.04)
        if len(sk.entities) < n:
            sk.add_rectangle(
                (c * 0.22 - 0.02, r * 0.22 - 0.02),
                (c * 0.22 + 0.08, r * 0.22 + 0.06),
            )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="hard")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--entities", type=int, default=400)
    ap.add_argument("--sketches", type=int, default=3)
    ap.add_argument("--moves", type=int, default=50)
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
    win.resize(args.width, args.height)
    win.show()
    for _ in range(25):
        app.processEvents()
        time.sleep(0.01)
    if not win.viewport._ok:
        print(f"HARD_MODE_FAIL tag={args.tag} reason=viewport", flush=True)
        return 1

    doc = win.doc
    planes = [
        f
        for f in doc.features
        if f.type
        in (FeatureType.PLANE_FRONT, FeatureType.PLANE_TOP, FeatureType.PLANE_RIGHT)
    ]
    active_skf = None
    for i in range(max(1, args.sketches)):
        plane = planes[i % len(planes)] if planes else None
        if plane is None:
            break
        skf = doc.create_sketch_on_plane(plane.id)
        _pad_entities(skf.sketch, args.entities // max(1, args.sketches))
        if active_skf is None:
            active_skf = skf

    win.viewport.set_document(doc)
    for _ in range(15):
        app.processEvents()
        time.sleep(0.01)
    if win.viewport._render_timer.isActive():
        win.viewport._render_timer.stop()
    win.viewport._do_render()

    assert active_skf is not None
    win.viewport.enter_sketch(active_skf.id)
    for _ in range(12):
        app.processEvents()
        time.sleep(0.01)

    cam = win.viewport.plotter.camera
    parallel = bool(cam.GetParallelProjection())
    no_rubber = not hasattr(win.viewport, "_rubber") or win.viewport.__dict__.get(
        "_rubber"
    ) is None

    ctrl = win.viewport._sketch_ctrl
    assert ctrl is not None
    ctrl.set_tool(SketchTool.LINE)
    win.viewport.reset_render_stats()

    iw = win.viewport.plotter.interactor
    cx = max(20, iw.width() // 2)
    cy = max(20, iw.height() // 2)

    win.viewport._sketch_mouse_press(cx, cy)
    app.processEvents()

    move_ms: list[float] = []
    for i in range(args.moves):
        t0 = time.perf_counter()
        win.viewport._sketch_mouse_move(cx + 2 * i, cy - i)
        app.processEvents()
        move_ms.append((time.perf_counter() - t0) * 1000.0)

    # Preview actor must exist on overlay after stroke moves
    preview_actor = (
        "__sk_preview" in getattr(win.viewport, "_overlay_actors", {})
        or (win.viewport.plotter and "__sk_preview" in win.viewport.plotter.actors)
    )
    n_renders, render_ms = win.viewport.render_stats()
    mean = statistics.mean(move_ms) if move_ms else 0.0
    med = statistics.median(move_ms) if move_ms else 0.0
    p95 = sorted(move_ms)[int(0.95 * (len(move_ms) - 1))] if move_ms else 0.0
    renders_per_move = n_renders / max(1, args.moves)
    ms_per_render = render_ms / max(1, n_renders)

    win.viewport.exit_sketch()
    for _ in range(8):
        app.processEvents()
        time.sleep(0.01)
    parallel_after = bool(cam.GetParallelProjection())

    print(
        f"HARD_MODE tag={args.tag} size={args.width}x{args.height} "
        f"entities={args.entities} sketches={args.sketches} moves={args.moves} "
        f"no_qt_rubber={int(no_rubber)} parallel_in={int(parallel)} "
        f"parallel_out={int(not parallel_after)} preview_actor={int(preview_actor)} "
        f"mean_move={mean:.2f}ms median={med:.2f}ms p95={p95:.2f}ms "
        f"stroke_renders={n_renders} renders/move={renders_per_move:.3f} "
        f"ms/render={ms_per_render:.2f} render_ms_sum={render_ms:.1f}",
        flush=True,
    )

    # Honest path: ~1 render per move is expected and correct
    ok = (
        no_rubber
        and parallel
        and (not parallel_after)
        and preview_actor
        and renders_per_move >= 0.8  # must re-render for VTK preview
        and med < 80.0  # soft-GL budget with LOD (not zero-render fantasy)
    )
    if ok:
        print(
            f"HARD_MODE_OK tag={args.tag} VTK_PREVIEW_OK 2D_OK 3D_EXIT_OK "
            f"median_move={med:.2f}ms ms/render={ms_per_render:.2f}",
            flush=True,
        )
        return 0
    print(
        f"HARD_MODE_FAIL tag={args.tag} no_rubber={no_rubber} parallel={parallel} "
        f"exit3d={not parallel_after} preview_actor={preview_actor} "
        f"renders/move={renders_per_move:.3f} med={med:.2f}",
        flush=True,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
