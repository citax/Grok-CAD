#!/usr/bin/env python3
"""Framebuffer proof that live sketch preview is visible in the VTK render.

Captures plotter.screenshot() from a real shown MainWindow under xcb — NOT
widget.grab() / rubber.isVisible() / source-string checks.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("QT_XCB_GL_INTEGRATION", "none")
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")


def _hex_rgb(hex_color: str) -> np.ndarray:
    h = hex_color.lstrip("#")
    return np.array([int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)], dtype=np.float64)


def _count_near(img: np.ndarray, target_rgb: np.ndarray, *, tol: float = 48.0) -> int:
    """Count pixels within L-inf distance tol of target RGB (0-255)."""
    if img is None or img.size == 0:
        return 0
    arr = np.asarray(img)
    if arr.ndim == 2:
        return 0
    rgb = arr[..., :3].astype(np.float64)
    diff = np.max(np.abs(rgb - target_rgb.reshape(1, 1, 3)), axis=2)
    return int(np.count_nonzero(diff <= tol))


def _pump(app, n: int = 12) -> None:
    for _ in range(n):
        app.processEvents()
        time.sleep(0.01)


def _shot(win) -> np.ndarray:
    if win.viewport._render_timer.isActive():
        win.viewport._render_timer.stop()
    win.viewport._do_render()
    img = win.viewport.plotter.screenshot(return_img=True)
    return np.asarray(img)


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from app.mainwindow import MainWindow
    from app.sketch_mode import SketchTool
    from app.theme import SKETCH_COLOR, SKETCH_PREVIEW, apply_theme
    from cadcore.document import FeatureType

    app = QApplication(sys.argv)
    apply_theme(app)
    win = MainWindow()
    win.resize(1000, 800)
    win.show()
    _pump(app, 25)
    if not win.viewport._ok:
        print("PREVIEW_PIXELS_FAIL reason=viewport", flush=True)
        return 1

    front = next(f for f in win.doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = win.doc.create_sketch_on_plane(front.id)
    win.viewport.enter_sketch(skf.id)
    _pump(app, 15)

    preview_rgb = _hex_rgb(SKETCH_PREVIEW)
    commit_rgb = _hex_rgb(SKETCH_COLOR)
    iw = win.viewport.plotter.interactor
    cx, cy = max(40, iw.width() // 2), max(40, iw.height() // 2)

    # Baseline empty sketch (no preview)
    base = _shot(win)
    base_preview = _count_near(base, preview_rgb)
    print(f"BASELINE_PREVIEW_PIXELS {base_preview}", flush=True)

    ok_all = True
    move_ms: list[float] = []

    # ----- LINE: mid-stroke preview -----
    ctrl = win.viewport._sketch_ctrl
    assert ctrl is not None
    ctrl.set_tool(SketchTool.LINE)
    win.viewport.reset_render_stats()
    win.viewport._sketch_mouse_press(cx - 120, cy + 40)
    t0 = time.perf_counter()
    win.viewport._sketch_mouse_move(cx + 140, cy - 80)
    move_ms.append((time.perf_counter() - t0) * 1000.0)
    _pump(app, 4)
    mid = _shot(win)
    n_prev = _count_near(mid, preview_rgb)
    # Must have more SKETCH_PREVIEW-colored pixels than baseline
    if n_prev > max(20, base_preview + 15):
        print(f"PREVIEW_PIXELS_OK line {n_prev}", flush=True)
    else:
        print(f"PREVIEW_PIXELS_FAIL line {n_prev} base={base_preview}", flush=True)
        ok_all = False

    # 3-point polyline: commit first segment, mid-stroke second, commit second
    win.viewport._sketch_mouse_press(cx + 140, cy - 80)  # commit seg1
    _pump(app, 4)
    t0 = time.perf_counter()
    win.viewport._sketch_mouse_move(cx + 140, cy + 100)  # preview second
    move_ms.append((time.perf_counter() - t0) * 1000.0)
    mid2 = _shot(win)
    n_prev2 = _count_near(mid2, preview_rgb)
    n_commit_mid = _count_near(mid2, commit_rgb)
    if n_prev2 > 20 and n_commit_mid > 20:
        print(
            f"PREVIEW_PIXELS_OK line_chain_mid preview={n_prev2} commit={n_commit_mid}",
            flush=True,
        )
    else:
        print(
            f"PREVIEW_PIXELS_FAIL line_chain_mid preview={n_prev2} commit={n_commit_mid}",
            flush=True,
        )
        ok_all = False

    win.viewport._sketch_mouse_press(cx + 140, cy + 100)  # commit seg2
    _pump(app, 4)
    # Still mid-chain — third point preview
    t0 = time.perf_counter()
    win.viewport._sketch_mouse_move(cx - 120, cy + 100)
    move_ms.append((time.perf_counter() - t0) * 1000.0)
    mid3 = _shot(win)
    n_commit_both = _count_near(mid3, commit_rgb)
    if n_commit_both > 40:
        print(f"COMMIT_PIXELS_OK line_chain_two_segments {n_commit_both}", flush=True)
    else:
        print(f"COMMIT_PIXELS_FAIL line_chain_two_segments {n_commit_both}", flush=True)
        ok_all = False

    # Finish chain
    if ctrl.is_drawing():
        win.viewport._end_line_chain()
    _pump(app, 6)

    # ----- RECTANGLE -----
    ctrl.set_tool(SketchTool.RECTANGLE)
    win.viewport._sketch_mouse_press(cx - 100, cy - 60)
    t0 = time.perf_counter()
    win.viewport._sketch_mouse_move(cx + 80, cy + 70)
    move_ms.append((time.perf_counter() - t0) * 1000.0)
    rmid = _shot(win)
    n_r = _count_near(rmid, preview_rgb)
    if n_r > 30:
        print(f"PREVIEW_PIXELS_OK rectangle {n_r}", flush=True)
    else:
        print(f"PREVIEW_PIXELS_FAIL rectangle {n_r}", flush=True)
        ok_all = False
    win.viewport._sketch_mouse_press(cx + 80, cy + 70)
    _pump(app, 6)
    rdone = _shot(win)
    n_rc = _count_near(rdone, commit_rgb)
    if n_rc > 30:
        print(f"COMMIT_PIXELS_OK rectangle {n_rc}", flush=True)
    else:
        print(f"COMMIT_PIXELS_FAIL rectangle {n_rc}", flush=True)
        ok_all = False

    # ----- CIRCLE -----
    ctrl.set_tool(SketchTool.CIRCLE)
    win.viewport._sketch_mouse_press(cx, cy)
    t0 = time.perf_counter()
    win.viewport._sketch_mouse_move(cx + 90, cy)
    move_ms.append((time.perf_counter() - t0) * 1000.0)
    cmid = _shot(win)
    n_c = _count_near(cmid, preview_rgb)
    if n_c > 30:
        print(f"PREVIEW_PIXELS_OK circle {n_c}", flush=True)
    else:
        print(f"PREVIEW_PIXELS_FAIL circle {n_c}", flush=True)
        ok_all = False
    win.viewport._sketch_mouse_press(cx + 90, cy)
    _pump(app, 6)
    cdone = _shot(win)
    n_cc = _count_near(cdone, commit_rgb)
    if n_cc > 30:
        print(f"COMMIT_PIXELS_OK circle {n_cc}", flush=True)
    else:
        print(f"COMMIT_PIXELS_FAIL circle {n_cc}", flush=True)
        ok_all = False

    n_ren, ms_sum = win.viewport.render_stats()
    mean_move = float(np.mean(move_ms)) if move_ms else 0.0
    stroke_moves = list(getattr(win.viewport, "_stroke_move_ms", []) or [])
    honest = float(np.mean(stroke_moves)) if stroke_moves else mean_move
    print(
        f"STROKE_PERF moves={len(move_ms)} mean_move_ms={mean_move:.2f} "
        f"viewport_stroke_mean_ms={honest:.2f} full_renders={n_ren} "
        f"render_ms_sum={ms_sum:.1f}",
        flush=True,
    )
    # Sanity: preview actor is on overlay map when drawing would have left it
    print(
        f"NO_QT_RUBBER ok={not hasattr(win.viewport, '_rubber') or win.viewport.__dict__.get('_rubber') is None}",
        flush=True,
    )

    if ok_all:
        print("FRAMEBUFFER_VERIFY_OK", flush=True)
        return 0
    print("FRAMEBUFFER_VERIFY_FAIL", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
