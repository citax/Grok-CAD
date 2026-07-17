#!/usr/bin/env python3
"""Honest verification: pale cube, DPI-safe clicks, Space, opening screen.

Rules:
  * Mouse path failures are FAIL — never fall back to apply_cube_view/pick_region.
  * Cube must not be a black blob (luminance of cube corner region).
  * Coordinate conversion must work for DPR=1 and DPR=2 (multi-monitor HiDPI).
  * Break-on-purpose for Space, DPI, and cube colour.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("QT_XCB_GL_INTEGRATION", "none")
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("GROK_THEME", "light")

OUT = _ROOT / "bench" / "fixtures" / "ui_space_cube"
OUT.mkdir(parents=True, exist_ok=True)

_FAILS: list[str] = []
_OKS: list[str] = []


def _ok(m: str) -> None:
    _OKS.append(m)
    print(f"OK  {m}", flush=True)


def _fail(m: str) -> None:
    _FAILS.append(m)
    print(f"FAIL {m}", flush=True)


def _pump(app, n=18):
    for _ in range(n):
        app.processEvents()
        time.sleep(0.01)


def _shot(vp, name: str):
    path = OUT / name
    try:
        if vp._render_timer.isActive():
            vp._render_timer.stop()
        vp._do_render()
        img = np.asarray(vp.plotter.screenshot(return_img=True))
        vp.plotter.screenshot(str(path))
        print(f"SHOT {path}", flush=True)
        return img
    except Exception as exc:
        _fail(f"shot {name}: {exc}")
        return None


def _cam(vp):
    return vp.camera_snapshot()


def _delta(a, b):
    if not a or not b:
        return 0.0
    return float(np.linalg.norm(np.asarray(a["position"]) - np.asarray(b["position"])))


def main() -> int:
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QMenu

    from app.display_coords import in_normalized_viewport, qt_to_normalized
    from app.mainwindow import MainWindow
    from app.theme import apply_theme
    from app.view_cube import build_chamfered_cube, color_for_region, face_label_text
    from cadcore.document import FeatureType, is_reference_plane

    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app)
    win = MainWindow()
    win.resize(1280, 800)
    win.show()
    win.raise_()
    win.activateWindow()
    _pump(app, 30)
    vp = win.viewport
    iw = vp.plotter.interactor

    # ------------------------------------------------------------------
    # OPENING SCREEN
    # ------------------------------------------------------------------
    print("\n=== OPENING SCREEN ===", flush=True)
    if win.doc.selected_id >= 0:
        f = win.doc.find(win.doc.selected_id)
        if f is not None and is_reference_plane(f.type):
            _fail(f"opening selects plane {f.name}")
        else:
            _ok(f"opening selection id={win.doc.selected_id}")
    else:
        _ok("opening: no feature selected")
    half = float(vp._plane_half)
    (_ok if half <= 20 else _fail)(f"plane half={half:g}")
    bad = []
    for name, act in list(vp.plotter.actors.items()):
        if name.startswith("plane_") and act.GetVisibility():
            if act.GetProperty().GetOpacity() > 0.15:
                bad.append((name, act.GetProperty().GetOpacity()))
    (_ok if not bad else _fail)("planes low opacity" if not bad else f"opacity {bad}")
    img0 = _shot(vp, "open_empty.png")
    try:
        win.grab().save(str(OUT / "open_chrome.png"))
        print(f"SHOT {OUT / 'open_chrome.png'}", flush=True)
    except Exception:
        pass

    # ------------------------------------------------------------------
    # CUBE LOOK — not black (user judges picture; we gate black)
    # ------------------------------------------------------------------
    print("\n=== CUBE LOOK ===", flush=True)
    for lab in ("face:+x", "face:+y", "face:+z"):
        r, g, b = color_for_region(lab)
        mean = (r + g + b) / 3.0
        if mean < 0.75:
            _fail(f"{lab} colour too dark mean={mean:.2f}")
        else:
            _ok(f"{lab} pale mean={mean:.2f} label={face_label_text(lab)}")
    if vp._view_cube is None or not getattr(vp._view_cube, "_actors", None):
        _fail("view cube missing")
    else:
        prop = vp._view_cube._actors[0].GetProperty()
        if prop.GetLighting():
            _fail("cube lighting ON — causes black faces on soft-GL")
        else:
            _ok("cube lighting OFF (readable fills)")
        nlab = len(getattr(vp._view_cube, "_label_actors", []) or [])
        if nlab < 6:
            _fail(f"face labels missing (have {nlab})")
        else:
            _ok(f"face labels present n={nlab}")
            # At least one label should be visible (front-facing) after sync
            vis = 0
            for item in vp._view_cube._label_actors:
                ta = item[1] if isinstance(item, tuple) else item
                try:
                    if ta.GetVisibility():
                        vis += 1
                except Exception:
                    pass
            if vis < 1:
                _fail("no front-facing face labels visible")
            else:
                _ok(f"front-facing labels visible n={vis}")

    # Sample cube pad of screenshot. Mean alone is soft (light main scene shows
    # through PreserveColorBuffer). Dark-pixel fraction catches a black blob:
    # pale cube ≈ 0–2% dark; black faces ≈ 40%+ dark.
    def _cube_pad(img):
        h, w = img.shape[:2]
        x0, x1 = int(w * 0.80), int(w * 0.995)
        y0, y1 = int(h * (1.0 - 0.995)), int(h * (1.0 - 0.80))
        crop = img[y0:y1, x0:x1, :3].astype(np.float64)
        lum = crop.mean(axis=2)
        return crop, lum, (y0, y1, x0, x1)

    DARK_FRAC_FAIL = 0.20  # ≥ this → black-blob fail

    if img0 is not None:
        crop, lum, (y0, y1, x0, x1) = _cube_pad(img0)
        mean = float(lum.mean())
        dark_frac = float((lum < 40).mean())
        if dark_frac >= DARK_FRAC_FAIL or mean < 60:
            _fail(
                f"cube corner is a black blob mean={mean:.1f} dark_frac={dark_frac:.2f}"
            )
        else:
            _ok(f"cube corner pale mean={mean:.1f} dark_frac={dark_frac:.3f}")
        try:
            from PIL import Image

            Image.fromarray(img0[y0:y1, x0:x1, :3].astype(np.uint8)).save(
                str(OUT / "open_empty_cube_close.png")
            )
            print(f"SHOT {OUT / 'open_empty_cube_close.png'}", flush=True)
        except Exception as exc:
            _fail(f"cube close crop: {exc}")
            _shot(vp, "open_empty_cube_close.png")
    else:
        _shot(vp, "open_empty_cube_close.png")

    # Break-on-purpose: force black actor colours. Gate must go red (dark_frac high).
    print("\n=== CUBE LOOK BREAK-ON-PURPOSE ===", flush=True)
    cube = vp._view_cube
    saved_colors = []
    try:
        for act in cube._actors:
            c = act.GetProperty().GetColor()
            saved_colors.append(tuple(c))
            act.GetProperty().SetColor(0.02, 0.02, 0.02)
        for item in cube._label_actors:
            ta = item[1] if isinstance(item, tuple) else item
            ta.VisibilityOff()
        broken = _shot(vp, "_break_black_cube.png")
        try:
            (OUT / "_break_black_cube.png").unlink(missing_ok=True)
        except Exception:
            pass
        if broken is not None:
            _, blum, _ = _cube_pad(broken)
            bmean = float(blum.mean())
            bdark = float((blum < 40).mean())
            if bdark >= DARK_FRAC_FAIL:
                _ok(
                    f"cube colour break: forced black → dark_frac={bdark:.2f} "
                    f"mean={bmean:.1f} fails gate"
                )
            else:
                _fail(
                    f"cube colour break soft: forced black dark_frac={bdark:.2f} "
                    f"(threshold {DARK_FRAC_FAIL})"
                )
        else:
            _fail("cube colour break: no screenshot")
    finally:
        for act, c in zip(cube._actors, saved_colors):
            act.GetProperty().SetColor(*c)
        cube.sync_orientation()
        _pump(app, 4)
        vp._do_render()

    # ------------------------------------------------------------------
    # SPACE
    # ------------------------------------------------------------------
    print("\n=== SPACE BAR ===", flush=True)
    try:
        iw.setFocus()
    except Exception:
        win.setFocus()
    _pump(app, 4)
    win._view_menu = None
    QTest.keyClick(iw, Qt.Key.Key_Space)
    _pump(app, 12)
    menu = getattr(win, "_view_menu", None)
    if menu is None:
        QTest.keyClick(win, Qt.Key.Key_Space)
        _pump(app, 12)
        menu = getattr(win, "_view_menu", None)
    if menu is None or not menu.actions():
        _fail("Space: no menu after real keyClick")
    else:
        _ok(f"Space: menu with {len(menu.actions())} actions")
        menu.close()
    win._space_shortcut.setEnabled(False)
    win._view_menu = None
    QTest.keyClick(iw, Qt.Key.Key_Space)
    QTest.keyClick(win, Qt.Key.Key_Space)
    _pump(app, 8)
    if getattr(win, "_view_menu", None) is not None:
        _fail("Space break: menu still appears with shortcut off")
        win._view_menu.close()
    else:
        _ok("Space break: no menu when shortcut disabled")
    win._space_shortcut.setEnabled(True)

    # ------------------------------------------------------------------
    # DPI / multi-monitor coordinate honesty
    # ------------------------------------------------------------------
    print("\n=== DPI HIT-TEST (working vs broken conversion) ===", flush=True)
    from app.display_coords import in_normalized_viewport as invp

    class FakeIW:
        def __init__(self, w, h, dpr=1.0):
            self._w, self._h, self._dpr = w, h, dpr

        def width(self):
            return self._w

        def height(self):
            return self._h

        def devicePixelRatioF(self):
            return self._dpr

    # Logical click in cube pad on a 2× screen
    iw2 = FakeIW(800, 600, 2.0)
    # cube pad x 0.90 → logical x = 720
    ok_new = invp(720, 40, iw2, vp._view_cube.VIEWPORT)
    # Old bug: divide by device width 1600
    nx_old = 720 / 1600
    ok_old = 0.80 <= nx_old <= 0.995
    if ok_new and not ok_old:
        _ok("DPR=2: logical hit works; device-width division misses (covers dead screen)")
    else:
        _fail(f"DPR gate soft new={ok_new} old={ok_old}")

    # Real triad/cube region hit with Qt size (not window_size)
    ww, hh = iw.width(), iw.height()
    # bottom-left triad centre
    tx = 0.065 * ww
    ty = (1.0 - 0.065) * hh
    if invp(tx, ty, iw, vp._axes_viewport):
        _ok("triad pad hit-test via logical size")
    else:
        _fail("triad pad miss with logical size")
    cx = 0.90 * ww
    cy = (1.0 - 0.90) * hh
    if invp(cx, cy, iw, vp._view_cube.VIEWPORT):
        _ok("cube pad hit-test via logical size")
    else:
        _fail("cube pad miss with logical size")

    # Break-on-purpose: classic bug divides Qt logical coords by device size.
    # On a simulated DPR=2 screen the cube pad centre must MISS — sharp check.
    def _broken_invp(qx, qy, interactor, viewport):
        try:
            dw = float(interactor.width()) * float(interactor.devicePixelRatioF())
            dh = float(interactor.height()) * float(interactor.devicePixelRatioF())
        except Exception:
            dw, dh = float(interactor.width()), float(interactor.height())
        if dw < 1 or dh < 1:
            return False
        nx = float(qx) / dw
        ny = 1.0 - float(qy) / dh
        vx0, vy0, vx1, vy1 = viewport
        return vx0 <= nx <= vx1 and vy0 <= ny <= vy1

    dead = _broken_invp(720, 40, iw2, vp._view_cube.VIEWPORT)
    live = invp(720, 40, iw2, vp._view_cube.VIEWPORT)
    if live and not dead:
        _ok("DPI break: broken device-size hit misses pad; fixed helper hits")
    else:
        _fail(f"DPI break soft live={live} dead={dead}")

    # ------------------------------------------------------------------
    # CUBE MOUSE — no fallback
    # ------------------------------------------------------------------
    print("\n=== CUBE MOUSE (no fallback) ===", flush=True)
    cube = vp._view_cube

    def mouse_click(x, y):
        pos = QPointF(x, y)
        press = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            pos,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        release = QMouseEvent(
            QMouseEvent.Type.MouseButtonRelease,
            pos,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        app.sendEvent(iw, press)
        _pump(app, 3)
        app.sendEvent(iw, release)
        _pump(app, 10)

    def mouse_drag(x0, y0, x1, y1, steps=10):
        app.sendEvent(
            iw,
            QMouseEvent(
                QMouseEvent.Type.MouseButtonPress,
                QPointF(x0, y0),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        _pump(app, 2)
        for i in range(1, steps + 1):
            t = i / steps
            app.sendEvent(
                iw,
                QMouseEvent(
                    QMouseEvent.Type.MouseMove,
                    QPointF(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t),
                    Qt.MouseButton.NoButton,
                    Qt.MouseButton.LeftButton,
                    Qt.KeyboardModifier.NoModifier,
                ),
            )
            _pump(app, 1)
        app.sendEvent(
            iw,
            QMouseEvent(
                QMouseEvent.Type.MouseButtonRelease,
                QPointF(x1, y1),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        _pump(app, 10)

    def cube_xy(u, v):
        vx0, vy0, vx1, vy1 = cube.VIEWPORT
        nx = vx0 + u * (vx1 - vx0)
        ny = vy0 + v * (vy1 - vy0)
        return nx * ww, (1.0 - ny) * hh

    # Face pick via mouse only
    vp.set_view("iso")
    _pump(app, 6)
    before = _cam(vp)
    x, y = cube_xy(0.55, 0.55)
    mouse_click(x, y)
    d = _delta(before, _cam(vp))
    if d > 0.5:
        _ok(f"cube mouse face click moved camera Δ={d:.2f}")
    else:
        _fail(f"cube mouse face click did nothing Δ={d:.3f} (no fallback)")
    _shot(vp, "cube_after_mouse_face.png")

    # Drag
    vp.set_view("iso")
    _pump(app, 6)
    before = _cam(vp)
    x0, y0 = cube_xy(0.5, 0.5)
    x1, y1 = cube_xy(0.9, 0.4)
    mouse_drag(x0, y0, x1, y1)
    d = _delta(before, _cam(vp))
    if d > 0.5:
        _ok(f"cube mouse drag orbit Δ={d:.2f}")
    else:
        _fail(f"cube mouse drag did nothing Δ={d:.3f} (no fallback)")
    _shot(vp, "cube_after_drag.png")

    # Triad mouse click
    print("\n=== TRIAD MOUSE ===", flush=True)
    vp.set_view("iso")
    _pump(app, 6)
    before = _cam(vp)
    # Send through the same filter path as a user click
    ax0, ay0, ax1, ay1 = vp._axes_viewport
    tx = (ax0 + ax1) * 0.5 * ww
    # click toward right of pad for +X-ish
    tx = (ax0 + 0.85 * (ax1 - ax0)) * ww
    ty = (1.0 - (ay0 + 0.5 * (ay1 - ay0))) * hh
    mouse_click(tx, ty)
    d = _delta(before, _cam(vp))
    if d > 0.5:
        _ok(f"triad mouse click moved camera Δ={d:.2f}")
    else:
        _fail(f"triad mouse click did nothing Δ={d:.3f} (no fallback)")
    _shot(vp, "after_triad_mouse.png")

    # Face API still checked for +X vs -X (separate from mouse; documents mapping)
    print("\n=== FACE MAPPING (+X vs -X) ===", flush=True)
    from app.view_cube_widget import apply_cube_view

    apply_cube_view(vp, "right")
    _pump(app, 4)
    pr = np.asarray(_cam(vp)["position"])
    apply_cube_view(vp, "left")
    _pump(app, 4)
    pl = np.asarray(_cam(vp)["position"])
    if float(np.linalg.norm(pr - pl)) < 1.0:
        _fail("+X/-X same pose")
    else:
        _ok(f"+X vs -X differ by {np.linalg.norm(pr-pl):.1f}")

    # ------------------------------------------------------------------
    # Parts
    # ------------------------------------------------------------------
    print("\n=== PARTS ===", flush=True)

    def build(size, depth):
        front = next(f for f in win.doc.features if f.type is FeatureType.PLANE_FRONT)
        sk = win.doc.create_sketch_on_plane(front.id)
        sk.sketch.add_rectangle((0, 0), (size, size * 0.7))
        return win.doc.create_extrude(sk.id, depth)

    tiny = build(4.0, 2.5)
    vp.schedule_rebuild()
    t0 = time.time()
    while time.time() - t0 < 10:
        _pump(app, 4)
        if tiny.id in vp._solid_fps:
            break
    vp.set_selected_id(tiny.id)
    vp.set_view("iso")
    _pump(app, 10)
    img = _shot(vp, "tiny_part.png")
    (_ok if img is not None and float(img[..., :3].std()) > 8 else _fail)("tiny part")

    large = build(200.0, 80.0)
    vp.schedule_rebuild()
    t0 = time.time()
    while time.time() - t0 < 10:
        _pump(app, 4)
        if large.id in vp._solid_fps:
            break
    vp.set_selected_id(large.id)
    vp.set_view("iso")
    _pump(app, 10)
    img = _shot(vp, "large_part.png")
    (_ok if img is not None and float(img[..., :3].std()) > 8 else _fail)("large part")

    front = next(f for f in win.doc.features if f.type is FeatureType.PLANE_FRONT)
    sk = win.doc.create_sketch_on_plane(front.id)
    vp.enter_sketch(sk.id)
    _pump(app, 12)
    _shot(vp, "sketch_mode.png")
    (_ok if vp.in_sketch_mode else _fail)("sketch mode")
    vp.exit_sketch()
    try:
        win.grab().save(str(OUT / "toolbar_chrome.png"))
        print(f"SHOT {OUT / 'toolbar_chrome.png'}", flush=True)
        _ok("toolbar chrome")
    except Exception as exc:
        _fail(f"toolbar: {exc}")

    (OUT / "MANIFEST.txt").write_text(
        f"passed={len(_OKS)} failed={len(_FAILS)}\n"
        + "\n".join(f"OK {m}" for m in _OKS)
        + "\n"
        + "\n".join(f"FAIL {m}" for m in _FAILS)
        + "\n",
        encoding="utf-8",
    )
    print(f"\n=== SUMMARY passed={len(_OKS)} failed={len(_FAILS)} ===", flush=True)
    for f in _FAILS:
        print(f"  FAIL: {f}", flush=True)
    return 1 if _FAILS else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
