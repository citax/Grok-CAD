#!/usr/bin/env python3
"""2560×1440 stroke/LOD A-B + preview + box-select timing (real MainWindow).

Measures:
  - A/B: stroke with planes hidden (draw-LOD) vs planes forced visible
  - Line stroke per-move median/first/last
  - Preview pixels mid-stroke
  - Box window/crossing 25-move drag timing + SELBOX pixels
"""
from __future__ import annotations

import os
import statistics
import sys
import time

import numpy as np

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("QT_XCB_GL_INTEGRATION", "none")
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

W, H = 2560, 1440
N_MOVES = 25


def _hex_rgb(hex_color: str) -> np.ndarray:
    h = hex_color.lstrip("#")
    return np.array([int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)], dtype=np.float64)


def _count_near(img, target_rgb, *, tol=48.0) -> int:
    arr = np.asarray(img)
    if arr.ndim < 3:
        return 0
    rgb = arr[..., :3].astype(np.float64)
    diff = np.max(np.abs(rgb - target_rgb.reshape(1, 1, 3)), axis=2)
    return int(np.count_nonzero(diff <= tol))


def _pump(app, n=12):
    for _ in range(n):
        app.processEvents()
        time.sleep(0.01)


def _shot(win):
    if win.viewport._render_timer.isActive():
        win.viewport._render_timer.stop()
    win.viewport._do_render()
    return np.asarray(win.viewport.plotter.screenshot(return_img=True))


def _summ(times):
    return {
        "median": statistics.median(times) if times else 0.0,
        "max": max(times) if times else 0.0,
        "first": times[0] if times else 0.0,
        "last": times[-1] if times else 0.0,
        "mean": statistics.mean(times) if times else 0.0,
        "n": len(times),
    }


def _timed_moves(vp, x0, y0, x1, y1, n=N_MOVES, *, shift=False, press=True):
    """vp is a Viewport (not MainWindow)."""
    if press:
        vp._sketch_mouse_press(x0, y0, shift=shift)
    times = []
    for i in range(1, n + 1):
        t = i / float(n)
        x = x0 + (x1 - x0) * t
        y = y0 + (y1 - y0) * t
        t0 = time.perf_counter()
        vp._sketch_mouse_move(x, y, shift=shift)
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from app.mainwindow import MainWindow
    from app.sketch_mode import SketchTool
    from app.theme import (
        SEL_BOX_CROSSING,
        SEL_BOX_WINDOW,
        SKETCH_PREVIEW,
        SKETCH_SELECTED,
        apply_theme,
    )
    from cadcore.document import FeatureType

    app = QApplication(sys.argv)
    apply_theme(app)
    win = MainWindow()
    win.resize(W, H)
    win.show()
    _pump(app, 30)
    if not win.viewport._ok:
        print("STROKE_2560_FAIL viewport", flush=True)
        return 1

    front = next(f for f in win.doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = win.doc.create_sketch_on_plane(front.id)
    sk = skf.sketch
    for i in range(60):
        r, c = divmod(i, 10)
        sk.add_line((c * 0.3 - 1.5, r * 0.3 - 1.5), (c * 0.3 - 1.35, r * 0.3 - 1.5))
    line_in = sk.add_line((0.4, 0.4), (1.6, 0.4))
    line_partial = sk.add_line((1.5, 1.2), (3.0, 1.2))
    circ_out = sk.add_circle((4.0, 3.0), 0.35)

    win.viewport.enter_sketch(skf.id)
    _pump(app, 15)
    vp = win.viewport
    ctrl = vp._sketch_ctrl
    assert ctrl is not None
    iw = vp.plotter.interactor
    cx, cy = iw.width() // 2, iw.height() // 2
    ok = True

    # ----- A/B pure-render: plane hide (what LOD now does) vs planes visible -----
    # Isolate VTK Render cost — same method as bench/draw_lod_profile.py
    def med_render(n=12):
        ts = []
        for _ in range(n):
            if vp._render_timer.isActive():
                vp._render_timer.stop()
            t0 = time.perf_counter()
            vp._do_render()
            ts.append((time.perf_counter() - t0) * 1000.0)
        return statistics.median(ts)

    ctrl.set_tool(SketchTool.LINE)
    vp._sketch_mouse_press(cx - 200, cy + 80)
    vp._begin_draw_lod()  # hides planes
    assert vp._draw_lod_active
    for _ in range(3):
        med_render(3)
    pure_hid = med_render(15)
    vp._set_draw_planes_visible(True)
    pure_vis = med_render(15)
    pure_ratio = pure_hid / pure_vis if pure_vis > 1e-6 else 0.0
    print(
        f"AB_PURE_RENDER size={W}x{H} planes_hidden={pure_hid:.2f}ms "
        f"planes_visible={pure_vis:.2f}ms ratio={pure_ratio:.3f} "
        f"saved={pure_vis - pure_hid:+.2f}ms",
        flush=True,
    )
    # Full mouse-move path with LOD (planes hidden) — includes preview rebuild
    vp._set_draw_planes_visible(False)
    times_on = _timed_moves(
        vp, cx - 200, cy + 80, cx + 200, cy - 80, press=False
    )
    s_on = _summ(times_on)
    print(
        f"STROKE_PERF lod_on n={s_on['n']} median={s_on['median']:.2f}ms "
        f"first={s_on['first']:.2f}ms last={s_on['last']:.2f}ms max={s_on['max']:.2f}ms",
        flush=True,
    )
    # Pure-render ratio is the LOD justification; ~0.5–0.7 expected at 2560
    if pure_ratio <= 0.75 and (pure_vis - pure_hid) >= 5.0:
        print(f"AB_STROKE_LOD_OK pure_ratio={pure_ratio:.3f}", flush=True)
    else:
        print(
            f"AB_STROKE_LOD_WEAK pure_ratio={pure_ratio:.3f} "
            f"(profile expected ~0.53 @2560)",
            flush=True,
        )
        # Not a hard fail of the feature — still report honestly
        # ok stays True; numbers are the verdict

    # Preview pixels while still mid-stroke with LOD on
    vp._set_draw_planes_visible(False)
    vp._sketch_mouse_move(cx + 200, cy - 80)
    img = _shot(win)
    n_prev = _count_near(img, _hex_rgb(SKETCH_PREVIEW))
    if n_prev > 30:
        print(f"PREVIEW_PIXELS_OK line {n_prev}", flush=True)
    else:
        print(f"PREVIEW_PIXELS_FAIL line {n_prev}", flush=True)
        ok = False

    # Rect + circle quick mid-stroke pixel checks
    vp._end_draw_lod()
    if ctrl.is_drawing():
        ctrl.end_line_chain() if hasattr(ctrl, "end_line_chain") else ctrl.cancel_drawing()
    ctrl.set_tool(SketchTool.RECTANGLE)
    vp._sketch_mouse_press(cx - 80, cy - 60)
    vp._sketch_mouse_move(cx + 100, cy + 80)
    n_r = _count_near(_shot(win), _hex_rgb(SKETCH_PREVIEW))
    print(
        f"{'PREVIEW_PIXELS_OK' if n_r > 30 else 'PREVIEW_PIXELS_FAIL'} rectangle {n_r}",
        flush=True,
    )
    if n_r <= 30:
        ok = False
    vp._sketch_mouse_press(cx + 100, cy + 80)
    _pump(app, 3)

    ctrl.set_tool(SketchTool.CIRCLE)
    vp._sketch_mouse_press(cx, cy)
    vp._sketch_mouse_move(cx + 90, cy)
    n_c = _count_near(_shot(win), _hex_rgb(SKETCH_PREVIEW))
    print(
        f"{'PREVIEW_PIXELS_OK' if n_c > 30 else 'PREVIEW_PIXELS_FAIL'} circle {n_c}",
        flush=True,
    )
    if n_c <= 30:
        ok = False
    vp._end_draw_lod()
    ctrl.cancel_drawing()

    # ----- Box select both directions @ 2560 (explicit display corners) -----
    ctrl.cancel_drawing()
    vp._end_draw_lod()
    ctrl.clear_selection()
    vp._clear_selbox()
    ctrl.set_tool(SketchTool.SELECT)
    # Explicit L→R / R→L in display space (avoid UV-map flakiness after clutter)
    p0 = (cx - 220, cy + 140)
    p1 = (cx + 220, cy - 140)
    assert p0[0] < p1[0]

    base = _shot(win)
    base_win = _count_near(base, _hex_rgb(SEL_BOX_WINDOW))
    base_cross = _count_near(base, _hex_rgb(SEL_BOX_CROSSING))

    win_times = _timed_moves(vp, p0[0], p0[1], p1[0], p1[1], press=True)
    assert ctrl.box_select is not None and ctrl.box_select.is_window
    n_win = _count_near(_shot(win), _hex_rgb(SEL_BOX_WINDOW))
    sw = _summ(win_times)
    print(
        f"{'SELBOX_PIXELS_OK' if n_win > max(20, base_win+10) else 'SELBOX_PIXELS_FAIL'} "
        f"window {n_win}",
        flush=True,
    )
    if n_win <= max(20, base_win + 10):
        ok = False
    print(
        f"BOX_PERF window n={sw['n']} median={sw['median']:.2f}ms "
        f"first={sw['first']:.2f}ms last={sw['last']:.2f}ms max={sw['max']:.2f}ms",
        flush=True,
    )
    vp._sketch_mouse_release(p1[0], p1[1])
    _pump(app, 3)

    # id sets (UV math — independent of display path)
    win_ids = ctrl.entities_in_box(0.0, 0.0, 2.0, 2.0, window=True)
    cross_ids = ctrl.entities_in_box(0.0, 0.0, 2.0, 2.0, window=False)
    print(f"WINDOW_IDS {sorted(win_ids)}", flush=True)
    print(f"CROSSING_IDS {sorted(cross_ids)}", flush=True)
    if win_ids != cross_ids and line_in.id in win_ids and line_partial.id in cross_ids:
        print("ID_SETS_OK window_vs_crossing differ as expected", flush=True)
    else:
        print("ID_SETS_FAIL", flush=True)
        ok = False

    # crossing R→L
    press_r = (p1[0], p0[1])
    rel_l = (p0[0], p1[1])
    cross_times = _timed_moves(vp, press_r[0], press_r[1], rel_l[0], rel_l[1], press=True)
    assert ctrl.box_select is not None and not ctrl.box_select.is_window
    n_cross = _count_near(_shot(win), _hex_rgb(SEL_BOX_CROSSING))
    sc = _summ(cross_times)
    print(
        f"{'SELBOX_PIXELS_OK' if n_cross > max(20, base_cross+10) else 'SELBOX_PIXELS_FAIL'} "
        f"crossing {n_cross}",
        flush=True,
    )
    if n_cross <= max(20, base_cross + 10):
        ok = False
    print(
        f"BOX_PERF crossing n={sc['n']} median={sc['median']:.2f}ms "
        f"first={sc['first']:.2f}ms last={sc['last']:.2f}ms max={sc['max']:.2f}ms",
        flush=True,
    )
    growth = sc["last"] / sc["first"] if sc["first"] > 1e-6 else 999.0
    ratio_box = sc["median"] / sw["median"] if sw["median"] > 1e-6 else 999.0
    print(
        f"BOX_PERF ratio_crossing/window={ratio_box:.2f} crossing_last/first={growth:.2f}",
        flush=True,
    )
    if ratio_box <= 1.5 and growth <= 2.0:
        print("BOX_PERF_OK", flush=True)
    else:
        print(f"BOX_PERF_FAIL ratio={ratio_box:.2f} growth={growth:.2f}", flush=True)
        ok = False
    vp._sketch_mouse_release(rel_l[0], rel_l[1])

    # selected pixels after multi-select
    ctrl.set_selection(win_ids)
    vp._rebuild_all_sketch_entities()
    n_sel = _count_near(_shot(win), _hex_rgb(SKETCH_SELECTED))
    print(
        f"{'SELECTED_PIXELS_OK' if n_sel > 15 else 'SELECTED_PIXELS_FAIL'} {n_sel}",
        flush=True,
    )
    if n_sel <= 15:
        ok = False

    if ok:
        print("STROKE_2560_OK", flush=True)
        return 0
    print("STROKE_2560_FAIL", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
