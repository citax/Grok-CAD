#!/usr/bin/env python3
"""Fullscreen frametime: real MainWindow plotter.render at 2560×1440."""
from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import time

os.environ.setdefault("QT_XCB_GL_INTEGRATION", "none")


def gl_renderer() -> str:
    try:
        out = subprocess.check_output(
            ["bash", "-lc", "glxinfo 2>/dev/null | grep -i 'OpenGL renderer' | head -1"],
            text=True,
            timeout=10,
        ).strip()
        return out or "(glxinfo empty)"
    except Exception as exc:
        return f"(glxinfo failed: {exc})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--width", type=int, default=2560)
    ap.add_argument("--height", type=int, default=1440)
    ap.add_argument("--platform", default="xcb")
    args = ap.parse_args()
    os.environ["QT_QPA_PLATFORM"] = args.platform

    print(f"GPU_PROBE tag={args.tag} {gl_renderer()}", flush=True)
    print(
        f"GPU_PROBE tag={args.tag} LIBGL_ALWAYS_SOFTWARE={os.environ.get('LIBGL_ALWAYS_SOFTWARE', '<unset>')}",
        flush=True,
    )

    from PySide6.QtWidgets import QApplication

    from app.mainwindow import MainWindow
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
        print(f"GPU_TEST_FAIL tag={args.tag} viewport", flush=True)
        return 1

    # Seed some geometry
    front = next(f for f in win.doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = win.doc.create_sketch_on_plane(front.id)
    for i in range(100):
        skf.sketch.add_line((i * 0.1, 0), (i * 0.1 + 0.05, 1))
    try:
        win.doc.create_extrude(skf.id, 1.0)
        win.viewport.schedule_rebuild()
        for _ in range(30):
            app.processEvents()
            time.sleep(0.02)
    except Exception:
        win.viewport.refresh_sketches()

    if win.viewport._render_timer.isActive():
        win.viewport._render_timer.stop()
    # warm
    for _ in range(3):
        win.viewport._do_render()
        app.processEvents()

    times = []
    for _ in range(args.frames):
        t0 = time.perf_counter()
        win.viewport._do_render()
        app.processEvents()
        times.append((time.perf_counter() - t0) * 1000.0)
    mean = statistics.mean(times)
    med = statistics.median(times)
    # Also read VTK/OpenGL string from app if available
    app_gl = getattr(win.viewport, "gl_renderer", "") or ""
    print(
        f"GPU_FRAME tag={args.tag} size={args.width}x{args.height} "
        f"mean={mean:.2f}ms median={med:.2f}ms app_gl={app_gl!r}",
        flush=True,
    )
    # Prefer VTK-reported renderer (glxinfo often missing in minimal WSLg)
    ren_proof = app_gl or gl_renderer()
    print(
        f"GPU_TEST_LINE tag={args.tag} renderer={ren_proof} ms/frame={mean:.2f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
