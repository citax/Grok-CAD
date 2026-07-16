#!/usr/bin/env python3
"""Framebuffer proof: click closed regions → fill highlight → extrude exact picks.

Uses real screen positions (WorldToDisplay), real mouse-path handlers, and
plotter.screenshot for fill + solid evidence. No list-picker.
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
os.environ.setdefault("GROK_THEME", os.environ.get("GROK_THEME", "light"))


def _pump(app, n=20):
    for _ in range(n):
        app.processEvents()
        time.sleep(0.01)


def _hex_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)], float)


def _uv_to_display(vp, uv) -> tuple[float, float]:
    fr = vp._sketch_ctrl.sketch.frame
    w = fr.to_world(uv)
    ren = vp.plotter.renderer
    ren.SetWorldPoint(float(w[0]), float(w[1]), float(w[2]), 1.0)
    ren.WorldToDisplay()
    d = ren.GetDisplayPoint()
    h = vp.plotter.interactor.height()
    # VTK display y-up → Qt y-down
    return float(d[0]), float(h - d[1])


def _click(vp, uv, *, ctrl: bool = False, shift: bool = False):
    x, y = _uv_to_display(vp, uv)
    vp._sketch_mouse_press(x, y, shift=shift, ctrl=ctrl)
    vp._sketch_mouse_release(x, y, shift=shift, ctrl=ctrl)
    vp._do_render()


def _count_color(img: np.ndarray, hex_color: str, tol: float = 48.0) -> int:
    c = _hex_rgb(hex_color)
    d = np.max(np.abs(img[..., :3].astype(float) - c.reshape(1, 1, 3)), axis=2)
    return int(np.count_nonzero(d <= tol))


def _count_fill_delta(before: np.ndarray, after: np.ndarray, hex_color: str) -> int:
    """Pixels that changed and move toward PROFILE_FILL (handles translucent blend)."""
    c = _hex_rgb(hex_color)
    b = before[..., :3].astype(float)
    a = after[..., :3].astype(float)
    changed = np.max(np.abs(a - b), axis=2) > 18.0
    # Closer to fill token after the click than before
    dist_b = np.linalg.norm(b - c.reshape(1, 1, 3), axis=2)
    dist_a = np.linalg.norm(a - c.reshape(1, 1, 3), axis=2)
    toward = dist_a + 8.0 < dist_b
    return int(np.count_nonzero(changed & toward))


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from app.theme import PROFILE_FILL, apply_theme, CURRENT_THEME
    from app.mainwindow import MainWindow
    from app.sketch_mode import SketchTool
    from cadcore.document import FeatureType
    from cadcore.profiles import list_closed_profiles, profile_area

    print(f"THEME {CURRENT_THEME} PROFILE_FILL={PROFILE_FILL}", flush=True)
    app = QApplication(sys.argv)
    apply_theme(app)
    win = MainWindow()
    win.resize(1600, 1000)
    win.show()
    _pump(app, 30)
    if not win.viewport._ok:
        print("FAIL viewport", flush=True)
        return 1

    front = next(f for f in win.doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = win.doc.create_sketch_on_plane(front.id)
    # Two disjoint rectangles: left large (2x2), right small (1x1)
    # Keep both regions near the origin so both stay in the default sketch frustum
    skf.sketch.add_rectangle((0.0, 0.0), (2.0, 2.0))  # area 4, center (1,1)
    skf.sketch.add_rectangle((2.5, 0.0), (3.5, 1.0))  # area 1, center (3.0, 0.5)
    profs = list_closed_profiles(skf.sketch)
    assert len(profs) == 2, profs
    by_area = sorted(profs, key=profile_area)
    small, large = by_area[0], by_area[1]
    print(
        f"PROFILES small_id={small.id} area={profile_area(small):.3f} "
        f"large_id={large.id} area={profile_area(large):.3f}",
        flush=True,
    )

    win.viewport.enter_sketch(skf.id)
    _pump(app, 20)
    win.viewport.set_sketch_tool(SketchTool.SELECT)
    win.viewport._do_render()
    _pump(app, 10)

    # --- empty: no fill ---
    img0 = np.asarray(win.viewport.plotter.screenshot(return_img=True))
    print(f"FILL_BASELINE shape={img0.shape}", flush=True)

    # Click center of large rect
    _click(win.viewport, (1.0, 1.0), ctrl=False)
    _pump(app, 8)
    img1 = np.asarray(win.viewport.plotter.screenshot(return_img=True))
    n1 = _count_fill_delta(img0, img1, PROFILE_FILL)
    sel = win.viewport.selected_profile_ids()
    print(f"FILL_AFTER_LARGE_CLICK n_delta={n1} sel={sel}", flush=True)
    if large.id not in sel:
        print(f"FAIL selection missing large id; got {sel}", flush=True)
        return 1
    if n1 < 200:
        print(
            f"FAIL fill not visible on real framebuffer (delta toward "
            f"{PROFILE_FILL} n={n1} < 200)",
            flush=True,
        )
        return 1
    # Save evidence crop
    try:
        from PIL import Image

        Image.fromarray(img1).save(
            os.path.join(str(_ROOT), "bench", "_ui_shots", "profile_fill_large.png")
        )
    except Exception:
        pass
    print("FILL_OK large region highlight on framebuffer", flush=True)

    # Empty click clears
    _click(win.viewport, (-2.0, -2.0), ctrl=False)
    _pump(app, 6)
    img_c = np.asarray(win.viewport.plotter.screenshot(return_img=True))
    nc = _count_fill_delta(img0, img_c, PROFILE_FILL)
    sel_c = win.viewport.selected_profile_ids()
    print(f"FILL_AFTER_CLEAR n_delta={nc} sel={sel_c}", flush=True)
    if sel_c:
        print("FAIL empty click did not clear profile selection", flush=True)
        return 1
    if nc > n1 * 0.25:
        print(f"FAIL fill still strong after clear ({nc} vs peak {n1})", flush=True)
        return 1
    print("CLEAR_OK", flush=True)

    # Small-only fill (prove the other region can be highlighted alone)
    _click(win.viewport, (3.0, 0.5), ctrl=False)
    _pump(app, 6)
    img_sm = np.asarray(win.viewport.plotter.screenshot(return_img=True))
    n_sm = _count_fill_delta(img0, img_sm, PROFILE_FILL)
    sel_sm = win.viewport.selected_profile_ids()
    print(f"FILL_SMALL_ONLY n_delta={n_sm} sel={sel_sm}", flush=True)
    if sel_sm != {small.id}:
        print(f"FAIL small-only selection {sel_sm}", flush=True)
        return 1
    if n_sm < 40:
        print(f"FAIL small region fill not visible (n={n_sm})", flush=True)
        return 1
    if n_sm >= n1:
        print(
            f"FAIL small fill should be smaller on screen than large ({n_sm} >= {n1})",
            flush=True,
        )
        return 1
    print("FILL_SMALL_OK", flush=True)

    # Ctrl multi: large then small — both selected; fill still on screen
    _click(win.viewport, (1.0, 1.0), ctrl=False)
    _click(win.viewport, (3.0, 0.5), ctrl=True)
    _pump(app, 8)
    img2 = np.asarray(win.viewport.plotter.screenshot(return_img=True))
    n2 = _count_fill_delta(img0, img2, PROFILE_FILL)
    sel2 = win.viewport.selected_profile_ids()
    print(f"FILL_MULTI n_delta={n2} sel={sel2}", flush=True)
    if large.id not in sel2 or small.id not in sel2:
        print(f"FAIL multi-select incomplete: {sel2}", flush=True)
        return 1
    if n2 < n_sm:
        print(f"FAIL multi fill vanished ({n2} < small-only {n_sm})", flush=True)
        return 1
    print("MULTI_FILL_OK", flush=True)

    # --- Extrude only the SMALL region (single pick) ---
    _click(win.viewport, (-2.0, -2.0))  # clear
    _click(win.viewport, (3.0, 0.5))  # small only
    _pump(app, 6)
    sel_s = win.viewport.selected_profile_ids()
    assert sel_s == {small.id}, sel_s
    # Same resolution path as Extrude UI (after distance dialog)
    pids = win._resolve_profile_ids_for_command(skf.sketch)
    print(f"RESOLVE_FOR_EXTRUDE_SMALL pids={pids}", flush=True)
    if pids != [small.id]:
        print(f"FAIL resolve ids for small pick: {pids}", flush=True)
        return 1
    win.viewport.exit_sketch()
    h = 2.0
    f_small = win.doc.create_extrude(skf.id, h, profile_entity_id=int(pids[0]))
    win.viewport.schedule_rebuild()
    _pump(app, 50)
    m_small = win.doc.evaluate_feature(f_small.id)
    if m_small is None or not m_small.is_watertight():
        print("FAIL small solid missing/not watertight", flush=True)
        return 1
    vol_s = float(m_small.volume())
    expect_s = 1.0 * h
    print(f"SOLID_SMALL vol={vol_s:.4f} expect={expect_s:.4f}", flush=True)
    if abs(vol_s - expect_s) / expect_s > 0.02:
        print("FAIL volume mismatch for small-only extrude", flush=True)
        return 1
    # Only one extrude solid
    extrudes = [f for f in win.doc.features if f.type is FeatureType.EXTRUDE]
    if len(extrudes) != 1:
        print(f"FAIL expected 1 extrude, got {len(extrudes)}", flush=True)
        return 1
    print("EXTRUDE_SMALL_OK", flush=True)

    # Framebuffer: solid present
    img_sol = np.asarray(win.viewport.plotter.screenshot(return_img=True))
    if int(img_sol.max()) < 10:
        print("FAIL solid view black", flush=True)
        return 1

    # --- Second sketch: multi-select both, two solids ---
    skf2 = win.doc.create_sketch_on_plane(front.id)
    skf2.sketch.add_rectangle((0.0, 0.0), (2.0, 2.0))
    skf2.sketch.add_rectangle((2.5, 0.0), (3.5, 1.0))
    win.viewport.enter_sketch(skf2.id)
    _pump(app, 15)
    win.viewport.set_sketch_tool(SketchTool.SELECT)
    win.viewport._do_render()
    _pump(app, 8)
    profs2 = list_closed_profiles(skf2.sketch)
    by2 = sorted(profs2, key=profile_area)
    s2, l2 = by2[0], by2[1]
    _click(win.viewport, (1.0, 1.0), ctrl=False)
    _click(win.viewport, (3.0, 0.5), ctrl=True)
    pids2 = win._resolve_profile_ids_for_command(skf2.sketch)
    print(f"RESOLVE_MULTI pids={pids2} sel={win.viewport.selected_profile_ids()}", flush=True)
    if set(pids2) != {s2.id, l2.id}:
        print(f"FAIL multi resolve {pids2} vs {[s2.id, l2.id]}", flush=True)
        return 1
    win.viewport.exit_sketch()
    before = {f.id for f in win.doc.features if f.type is FeatureType.EXTRUDE}
    created = []
    for pid in pids2:
        created.append(win.doc.create_extrude(skf2.id, 1.5, profile_entity_id=int(pid)))
    win.viewport.schedule_rebuild()
    _pump(app, 50)
    after = [f for f in win.doc.features if f.type is FeatureType.EXTRUDE and f.id not in before]
    if len(after) != 2:
        print(f"FAIL expected 2 new extrudes, got {len(after)}", flush=True)
        return 1
    vols = sorted(float(win.doc.evaluate_feature(f.id).volume()) for f in after)
    print(f"SOLID_MULTI vols={vols}", flush=True)
    # 1*1.5=1.5 and 4*1.5=6.0
    if abs(vols[0] - 1.5) / 1.5 > 0.02 or abs(vols[1] - 6.0) / 6.0 > 0.02:
        print("FAIL multi extrude volumes wrong", flush=True)
        return 1
    print("EXTRUDE_MULTI_OK", flush=True)

    # Ambiguous without selection must not silently resolve
    skf3 = win.doc.create_sketch_on_plane(front.id)
    skf3.sketch.add_rectangle((0, 0), (1, 1))
    skf3.sketch.add_rectangle((3, 0), (4, 1))
    win.viewport.enter_sketch(skf3.id)
    _pump(app, 10)
    # clear any selection
    if win.viewport._sketch_ctrl:
        win.viewport._sketch_ctrl.clear_selection()
    pids_none = win._resolve_profile_ids_for_command(skf3.sketch)
    print(f"RESOLVE_NO_PICK pids={pids_none}", flush=True)
    if pids_none is not None:
        print("FAIL ambiguous sketch without pick should return None", flush=True)
        return 1
    print("NO_GUESS_OK", flush=True)

    # Single profile auto
    skf4 = win.doc.create_sketch_on_plane(front.id)
    skf4.sketch.add_rectangle((0, 0), (2, 3))
    pids_one = win._resolve_profile_ids_for_command(skf4.sketch)
    print(f"RESOLVE_SINGLE pids={pids_one}", flush=True)
    if pids_one != [-1]:
        print("FAIL single profile should auto [-1]", flush=True)
        return 1
    print("SINGLE_AUTO_OK", flush=True)

    print("PROFILE_PICK_FRAMEBUFFER_OK", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"EXC_FAIL {exc!r}", flush=True)
        traceback.print_exc()
        raise SystemExit(1)
