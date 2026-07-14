#!/usr/bin/env python3
"""Profile draw-time VTK Render() calls vs stroke cost (real MainWindow)."""
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
    ap.add_argument("--width", type=int, default=2560)
    ap.add_argument("--height", type=int, default=1440)
    ap.add_argument("--entities", type=int, default=300)
    ap.add_argument("--moves", type=int, default=40)
    ap.add_argument("--platform", default="xcb")
    ap.add_argument(
        "--mode",
        default="auto",
        choices=("auto", "vtk", "qt"),
        help="force preview path: vtk=full re-render, qt=rubber-band only, auto=app default",
    )
    args = ap.parse_args()
    os.environ["QT_QPA_PLATFORM"] = args.platform
    if args.mode == "vtk":
        os.environ["GROK_DRAW_PREVIEW"] = "vtk"
    elif args.mode == "qt":
        os.environ["GROK_DRAW_PREVIEW"] = "qt"

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
        print(f"PROFILE_FAIL tag={args.tag}", flush=True)
        return 1

    # Instrument plotter.render
    render_times: list[float] = []
    orig = win.viewport._do_render

    def counting_render():
        t0 = time.perf_counter()
        orig()
        render_times.append((time.perf_counter() - t0) * 1000.0)

    win.viewport._do_render = counting_render  # type: ignore

    front = next(f for f in win.doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = win.doc.create_sketch_on_plane(front.id)
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
    # Start stroke
    ctrl.on_press((0.0, -2.0))
    if hasattr(win.viewport, "_begin_draw_lod"):
        win.viewport._begin_draw_lod()
    # One baseline freeze render (expected)
    render_times.clear()
    if win.viewport._render_timer.isActive():
        win.viewport._render_timer.stop()
    win.viewport._do_render()
    app.processEvents()
    freeze_renders = len(render_times)
    freeze_ms = sum(render_times)

    # Simulate stroke moves through real mouse-move path
    render_times.clear()
    t_stroke0 = time.perf_counter()
    for i in range(args.moves):
        # Widget coords roughly center-ish (not critical — UV path used below)
        # Call real handler with fabricated display coords via internal path
        u = 0.05 * i
        v = -2.0 + 0.02 * i
        # Prefer public mouse path if available; else simulate as app does
        win.viewport._update_snap_tolerance()
        ctrl.on_move((u, v))
        # Real app path during draw
        if getattr(win.viewport, "_draw_preview_qt", False) or (
            os.environ.get("GROK_DRAW_PREVIEW", "qt") == "qt"
            and hasattr(win.viewport, "_update_preview_qt")
        ):
            if hasattr(win.viewport, "_update_preview_qt"):
                win.viewport._update_preview_qt()
            # no VTK render requested
        else:
            if hasattr(win.viewport, "_update_preview_visual"):
                win.viewport._update_preview_visual()
            if win.viewport._render_timer.isActive():
                win.viewport._render_timer.stop()
            win.viewport._do_render()
            app.processEvents()
        # Also exercise the real _sketch_mouse_move path if coords available
        app.processEvents()
    stroke_ms = (time.perf_counter() - t_stroke0) * 1000.0
    n_renders = len(render_times)
    ms_per_render = statistics.mean(render_times) if render_times else 0.0

    # Commit one segment + full render
    render_times.clear()
    ctrl.on_press((2.0, -1.0))
    if win.viewport._render_timer.isActive():
        win.viewport._render_timer.stop()
    win.viewport._do_render()
    app.processEvents()
    commit_renders = len(render_times)

    print(
        f"PROFILE tag={args.tag} mode={args.mode} size={args.width}x{args.height} "
        f"n={args.entities} moves={args.moves} "
        f"freeze_renders={freeze_renders} freeze_ms={freeze_ms:.2f} "
        f"stroke_renders={n_renders} ms_per_render={ms_per_render:.2f} "
        f"ms_stroke={stroke_ms:.2f} renders_per_move={n_renders / max(1, args.moves):.3f} "
        f"commit_renders={commit_renders}",
        flush=True,
    )
    print(
        f"{args.tag.upper()}_DRAW = renders/stroke={n_renders} "
        f"ms/render={ms_per_render:.2f} ms/stroke={stroke_ms:.2f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
