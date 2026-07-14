#!/usr/bin/env python3
"""Profile draw-time VTK Render() cost (real MainWindow, VTK preview path)."""
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
        print(f"PROFILE_FAIL tag={args.tag}", flush=True)
        return 1

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
    cols = int(args.entities**0.5) + 1
    for i in range(args.entities):
        r, c = divmod(i, cols)
        sk.add_line((c * 0.25, r * 0.25), (c * 0.25 + 0.15, r * 0.25))
    win.viewport.enter_sketch(skf.id)
    for _ in range(15):
        app.processEvents()
        time.sleep(0.01)

    ctrl = win.viewport._sketch_ctrl
    ctrl.set_tool(SketchTool.LINE)
    iw = win.viewport.plotter.interactor
    cx, cy = iw.width() // 2, iw.height() // 2

    # Start stroke via real mouse path
    render_times.clear()
    win.viewport._sketch_mouse_press(cx, cy)
    app.processEvents()
    freeze_renders = len(render_times)
    freeze_ms = sum(render_times)

    render_times.clear()
    t_stroke0 = time.perf_counter()
    for i in range(args.moves):
        win.viewport._sketch_mouse_move(cx + 2 * i, cy - i)
        app.processEvents()
    stroke_ms = (time.perf_counter() - t_stroke0) * 1000.0
    n_renders = len(render_times)
    ms_per_render = statistics.mean(render_times) if render_times else 0.0

    render_times.clear()
    win.viewport._sketch_mouse_press(cx + 80, cy - 40)
    app.processEvents()
    commit_renders = len(render_times)

    print(
        f"PROFILE tag={args.tag} mode=vtk size={args.width}x{args.height} "
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
