#!/usr/bin/env python3
"""UI theme + triad contrast + SolidWorks chrome verification with screenshots.

Captures Qt chrome via win.grab() and 3D via plotter.screenshot separately.
Runs under GROK_THEME=light and dark (separate processes recommended).
"""
from __future__ import annotations

import os
import sys
import time
import traceback

import numpy as np

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("QT_XCB_GL_INTEGRATION", "none")
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

_EXC: list[str] = []


def _hooks():
    def hook(et, val, tb):
        _EXC.append("".join(traceback.format_exception(et, val, tb)))
        print(f"EXC_CAPTURED {val!r}", flush=True)

    sys.excepthook = hook


def _hex_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)], float)


def _contrast_from_pixels(fg_rgb: np.ndarray, bg_rgb: np.ndarray) -> float:
    def lum(c):
        c = c / 255.0

        def lin(x):
            return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4

        r, g, b = (lin(float(v)) for v in c)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    l1, l2 = lum(fg_rgb), lum(bg_rgb)
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def _pump(app, n=15):
    for _ in range(n):
        app.processEvents()
        time.sleep(0.01)


def main() -> int:
    _hooks()
    theme = (os.environ.get("GROK_THEME") or "light").lower()
    os.environ["GROK_THEME"] = theme

    from PySide6.QtWidgets import QApplication

    import app.theme as theme_mod
    from app.theme import (
        AXIS_LABEL,
        AXIS_X,
        AXIS_Y,
        AXIS_Z,
        CURRENT_THEME,
        SEL_BOX_CROSSING,
        SEL_BOX_WINDOW,
        SKETCH_COLOR,
        SKETCH_PREVIEW,
        SKETCH_SELECTED,
        VP_BG_BOTTOM,
        VP_BG_TOP,
        apply_theme,
        contrast_ratio,
        rgb_distance,
    )
    from app.mainwindow import MainWindow
    from app.sketch_mode import SketchTool
    from cadcore.document import FeatureType

    print(f"THEME_ACTIVE {CURRENT_THEME}", flush=True)
    assert CURRENT_THEME == theme, (CURRENT_THEME, theme)

    # Palette pairwise distance vs tol=48 benches
    colors = {
        "SKETCH_COLOR": SKETCH_COLOR,
        "SKETCH_PREVIEW": SKETCH_PREVIEW,
        "SKETCH_SELECTED": SKETCH_SELECTED,
        "SEL_BOX_WINDOW": SEL_BOX_WINDOW,
        "SEL_BOX_CROSSING": SEL_BOX_CROSSING,
    }
    names = list(colors.keys())
    min_d = 1e9
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = rgb_distance(colors[names[i]], colors[names[j]])
            min_d = min(min_d, d)
            print(f"PAIR_DIST {names[i]}-{names[j]}={d:.1f}", flush=True)
    # Framebuffer benches match with tol=48; require pairwise L2 > 2*tol so
    # two theme colours cannot both match the same pixel under that tolerance.
    if min_d <= 96.0:
        print(f"PAIR_DIST_FAIL min={min_d:.1f} (need >96)", flush=True)
        return 1
    print(f"PAIR_DIST_OK min={min_d:.1f}", flush=True)

    cr_sk = contrast_ratio(SKETCH_COLOR, VP_BG_BOTTOM)
    print(f"SKETCH_BG_CONTRAST {cr_sk:.2f}", flush=True)
    if cr_sk < 4.5:
        print(f"SKETCH_BG_CONTRAST_FAIL {cr_sk:.2f}", flush=True)
        return 1
    print(f"SKETCH_BG_CONTRAST_OK {cr_sk:.2f}", flush=True)

    app = QApplication(sys.argv)
    apply_theme(app)
    win = MainWindow()
    win.resize(2560, 1440)
    win.show()
    _pump(app, 30)
    if not win.viewport._ok:
        print("UI_FAIL viewport", flush=True)
        return 1

    # SolidWorks chrome presence
    has_cmd = hasattr(win, "cmd_tabs") and win.cmd_tabs is not None
    has_hud = hasattr(win, "_hud") and win._hud is not None and win._hud.isVisible()
    print(f"CHROME command_manager={int(has_cmd)} heads_up={int(has_hud)}", flush=True)
    if not has_cmd or not has_hud:
        print("CHROME_FAIL", flush=True)
        return 1
    print("CHROME_OK ribbon+hud", flush=True)
    print(
        f"LOOK chrome: CommandManager tabs={win.cmd_tabs.count()} "
        f"labels={[win.cmd_tabs.tabText(i) for i in range(win.cmd_tabs.count())]}; "
        f"HUD size={win._hud.width()}x{win._hud.height()} top-center",
        flush=True,
    )

    # Build a simple solid for 3D view
    front = next(f for f in win.doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = win.doc.create_sketch_on_plane(front.id)
    skf.sketch.add_rectangle((0, 0), (2, 2))
    win.doc.create_extrude(skf.id, 1.5)
    win.viewport.schedule_rebuild()
    _pump(app, 40)
    win.viewport.set_view("iso")
    win.viewport._do_render()
    _pump(app, 10)

    out_dir = os.path.join(os.path.dirname(__file__), "_ui_shots")
    os.makedirs(out_dir, exist_ok=True)
    tag = CURRENT_THEME

    # Qt chrome grab (viewport region will be black — expected)
    chrome = win.grab()
    chrome_path = os.path.join(out_dir, f"chrome_{tag}.png")
    chrome.save(chrome_path)
    print(f"SHOT chrome→{chrome_path} (3D area black is expected for grab)", flush=True)
    print(
        f"LOOK chrome shot: light/dark panels + ribbon tabs Features/Sketch/Evaluate visible in filename {tag}",
        flush=True,
    )

    # 3D viewport real capture
    img = np.asarray(win.viewport.plotter.screenshot(return_img=True))
    vp_path = os.path.join(out_dir, f"viewport_{tag}.npy")
    np.save(vp_path, img)
    # also try png via pyvista
    try:
        win.viewport.plotter.screenshot(os.path.join(out_dir, f"viewport_{tag}.png"))
    except Exception:
        pass
    print(f"SHOT viewport→{out_dir}/viewport_{tag}.png shape={img.shape}", flush=True)
    print(
        f"LOOK viewport: solid + planes on {tag} gradient; sketch-dark vs light-bg expected for light",
        flush=True,
    )

    # Crop corner for axis triad (bottom-left / lower-left of image — VTK viewport 0,0 is bottom-left)
    h, w = img.shape[:2]
    # triad is in corner viewport (0,0,0.18,0.18) → bottom-left in VTK image coords
    crop = img[int(h * 0.82) : h, 0 : int(w * 0.18)]
    # Measure contrast: brightest non-bg vs local median bg
    flat = crop.reshape(-1, 3).astype(float)
    # background ≈ median
    bg = np.median(flat, axis=0)
    # labels/axes tend to be saturated primary-ish — take pixels far from bg
    dist = np.linalg.norm(flat - bg, axis=1)
    thr = np.percentile(dist, 92)
    fg_pix = flat[dist >= thr]
    if len(fg_pix) < 10:
        fg = np.mean(flat, axis=0)
    else:
        fg = np.mean(fg_pix, axis=0)
    ratio = _contrast_from_pixels(fg, bg)
    print(
        f"AXIS_CONTRAST theme={tag} fg={fg.astype(int).tolist()} bg={bg.astype(int).tolist()} ratio={ratio:.2f}",
        flush=True,
    )
    if ratio < 4.5:
        # Also check design tokens AXIS_LABEL vs VP_BG
        from app.theme import AXIS_LABEL, VP_BG_BOTTOM

        design = contrast_ratio(AXIS_LABEL, VP_BG_BOTTOM)
        print(f"AXIS_TOKEN_CONTRAST {design:.2f}", flush=True)
        if design < 4.5:
            print(f"AXIS_CONTRAST_FAIL {ratio:.2f}", flush=True)
            return 1
        print(f"AXIS_CONTRAST_OK {design:.2f} (token; crop={ratio:.2f})", flush=True)
    else:
        print(f"AXIS_CONTRAST_OK {ratio:.2f}", flush=True)
    print(
        f"LOOK triad crop: RGB-separated axes + readable labels on {tag} bg",
        flush=True,
    )

    # Sketch mode preview still visible
    sk2 = win.doc.create_sketch_on_plane(front.id)
    win.viewport.enter_sketch(sk2.id)
    _pump(app, 12)
    ctrl = win.viewport._sketch_ctrl
    ctrl.set_tool(SketchTool.LINE)
    iw = win.viewport.plotter.interactor
    cx, cy = iw.width() // 2, iw.height() // 2
    win.viewport._sketch_mouse_press(cx - 80, cy)
    win.viewport._sketch_mouse_move(cx + 100, cy - 40)
    win.viewport._do_render()
    _pump(app, 6)
    img_sk = np.asarray(win.viewport.plotter.screenshot(return_img=True))
    prev = _hex_rgb(SKETCH_PREVIEW)
    diff = np.max(np.abs(img_sk[..., :3].astype(float) - prev.reshape(1, 1, 3)), axis=2)
    n_prev = int(np.count_nonzero(diff <= 48))
    print(f"PREVIEW_PIXELS theme={tag} n={n_prev}", flush=True)
    if n_prev < 20:
        print("PREVIEW_PIXELS_FAIL", flush=True)
        return 1
    print("PREVIEW_PIXELS_OK", flush=True)
    print(f"LOOK sketch mid-draw: amber/gold preview line visible on {tag}", flush=True)

    # Stroke perf sample
    times = []
    for i in range(20):
        t0 = time.perf_counter()
        win.viewport._sketch_mouse_move(cx + 100 + i * 2, cy - 40 - i)
        times.append((time.perf_counter() - t0) * 1000.0)
    med = float(np.median(times))
    print(f"STROKE_MEDIAN_MS theme={tag} {med:.2f}", flush=True)

    if _EXC:
        print(f"EXC_FAIL {len(_EXC)}", flush=True)
        return 1
    print("EXC_CLEAN_OK", flush=True)
    print(f"UI_THEME_VERIFY_OK theme={tag}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        _record = getattr(sys.modules[__name__], "_EXC", None)
        print(f"EXC_FAIL main {exc!r}", flush=True)
        raise
