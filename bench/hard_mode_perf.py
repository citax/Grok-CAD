#!/usr/bin/env python3
"""Hard-mode sketch performance: many renderable features + live stroke monitor.

Builds a dense scene (planes + many sketch entities + extruded solids when
possible), enters sketch, and measures per-move cost for line preview under
software GL. Prints HARD_MODE_OK / HARD_MODE_FAIL tokens for CI-style logs.
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
    ap.add_argument("--extrude", action="store_true", help="also create extruded solids")
    args = ap.parse_args()
    os.environ["QT_QPA_PLATFORM"] = args.platform
    os.environ.setdefault("GROK_DRAW_PREVIEW", "qt")

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
    planes = [f for f in doc.features if f.type in (
        FeatureType.PLANE_FRONT, FeatureType.PLANE_TOP, FeatureType.PLANE_RIGHT
    )]
    active_skf = None
    for i in range(max(1, args.sketches)):
        plane = planes[i % len(planes)] if planes else None
        if plane is None:
            break
        skf = doc.create_sketch_on_plane(plane.id)
        _pad_entities(skf.sketch, args.entities // max(1, args.sketches))
        if active_skf is None:
            active_skf = skf
        if args.extrude and skf.sketch is not None:
            # closed rectangle for extrude if API exists
            try:
                skf.sketch.add_rectangle((-1.5 + i * 0.1, -1.0), (-0.5 + i * 0.1, 0.0))
                # Prefer document extrude helper when present
                if hasattr(doc, "add_extrude"):
                    doc.add_extrude(skf.id, depth=0.5)
                elif hasattr(doc, "create_extrude"):
                    doc.create_extrude(skf.id, 0.5)
            except Exception as exc:  # noqa: BLE001
                print(f"[hard] extrude skip: {exc}", flush=True)

    # Rebuild scene with all closed sketches
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

    # Parallel / 2D checks
    cam = win.viewport.plotter.camera
    parallel = bool(cam.GetParallelProjection())
    rubber = win.viewport._rubber
    sibling_ok = rubber is not None and rubber.parentWidget() is win.viewport

    ctrl = win.viewport._sketch_ctrl
    assert ctrl is not None
    ctrl.set_tool(SketchTool.LINE)
    win.viewport.reset_render_stats()

    iw = win.viewport.plotter.interactor
    cx = max(20, iw.width() // 2)
    cy = max(20, iw.height() // 2)

    # First click → start cue must populate rubber segments
    win.viewport._sketch_mouse_press(cx, cy)
    app.processEvents()
    after_click_segs = list(rubber._segments) if rubber else []
    after_click_visible = bool(rubber and rubber.isVisible())

    move_ms: list[float] = []
    for i in range(args.moves):
        t0 = time.perf_counter()
        win.viewport._sketch_mouse_move(cx + 2 * i, cy - i)
        app.processEvents()
        move_ms.append((time.perf_counter() - t0) * 1000.0)

    after_move_segs = list(rubber._segments) if rubber else []
    n_renders, render_ms = win.viewport.render_stats()
    mean = statistics.mean(move_ms) if move_ms else 0.0
    med = statistics.median(move_ms) if move_ms else 0.0
    p95 = sorted(move_ms)[int(0.95 * (len(move_ms) - 1))] if move_ms else 0.0

    # Exit to 3D
    win.viewport.exit_sketch()
    for _ in range(8):
        app.processEvents()
        time.sleep(0.01)
    parallel_after = bool(cam.GetParallelProjection())

    preview_ok = after_click_visible and len(after_click_segs) > 0
    stroke_ok = len(after_move_segs) > 0 and after_move_segs != after_click_segs
    # Stroke should not full-render every move (qt path)
    renders_per_move = n_renders / max(1, args.moves)

    print(
        f"HARD_MODE tag={args.tag} size={args.width}x{args.height} "
        f"entities={args.entities} sketches={args.sketches} moves={args.moves} "
        f"sibling_ok={int(sibling_ok)} parallel_in={int(parallel)} "
        f"parallel_out={int(not parallel_after)} "
        f"click_preview={int(preview_ok)} stroke_preview={int(stroke_ok)} "
        f"mean_move={mean:.2f}ms median={med:.2f}ms p95={p95:.2f}ms "
        f"stroke_renders={n_renders} renders/move={renders_per_move:.3f} "
        f"render_ms_sum={render_ms:.1f}",
        flush=True,
    )

    ok = (
        sibling_ok
        and parallel
        and (not parallel_after)
        and preview_ok
        and stroke_ok
        and renders_per_move < 0.25  # freeze only; not per-move full renders
        and med < 25.0  # soft-GL budget for Qt rubber-band path
    )
    if ok:
        print(
            f"HARD_MODE_OK tag={args.tag} PREVIEW_OK 2D_OK 3D_EXIT_OK "
            f"median_move={med:.2f}ms",
            flush=True,
        )
        return 0
    print(
        f"HARD_MODE_FAIL tag={args.tag} sibling={sibling_ok} parallel={parallel} "
        f"exit3d={not parallel_after} click_preview={preview_ok} "
        f"stroke_preview={stroke_ok} renders/move={renders_per_move:.3f} med={med:.2f}",
        flush=True,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
