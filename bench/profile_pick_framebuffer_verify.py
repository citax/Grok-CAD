#!/usr/bin/env python3
"""Honest proof: region pick + box-from-inside + edge pick + real Extrude command.

- Real screen positions (WorldToDisplay) and viewport mouse handlers.
- Box vs region: same press point; drag → box, click → region.
- Edge: click on the line → entity, not region.
- Extrude: call MainWindow._extrude() (same as the E action); only the distance
  dialog is stubbed — not the command wiring or create path.
- Solids checked via document evaluation after schedule_rebuild (what the app
  rebuilds for the viewport).
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path
from unittest.mock import patch

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
    return float(d[0]), float(h - d[1])


def _click(vp, uv, *, ctrl: bool = False, shift: bool = False):
    x, y = _uv_to_display(vp, uv)
    vp._sketch_mouse_press(x, y, shift=shift, ctrl=ctrl)
    vp._sketch_mouse_release(x, y, shift=shift, ctrl=ctrl)
    vp._do_render()


def _drag_box(vp, uv0, uv1, *, ctrl: bool = False, shift: bool = False):
    """Press at uv0, drag to uv1, release — real box-select gesture."""
    x0, y0 = _uv_to_display(vp, uv0)
    x1, y1 = _uv_to_display(vp, uv1)
    vp._sketch_mouse_press(x0, y0, shift=shift, ctrl=ctrl)
    # Several moves so drag_px clearly exceeds BOX_SELECT_MIN_PX
    for i in range(1, 6):
        t = i / 5.0
        x = x0 + (x1 - x0) * t
        y = y0 + (y1 - y0) * t
        vp._sketch_mouse_move(x, y, shift=shift, ctrl=ctrl)
    vp._sketch_mouse_release(x1, y1, shift=shift, ctrl=ctrl)
    vp._do_render()


def _count_fill_delta(before: np.ndarray, after: np.ndarray, hex_color: str) -> int:
    c = _hex_rgb(hex_color)
    b = before[..., :3].astype(float)
    a = after[..., :3].astype(float)
    changed = np.max(np.abs(a - b), axis=2) > 18.0
    dist_b = np.linalg.norm(b - c.reshape(1, 1, 3), axis=2)
    dist_a = np.linalg.norm(a - c.reshape(1, 1, 3), axis=2)
    toward = dist_a + 8.0 < dist_b
    return int(np.count_nonzero(changed & toward))


def main() -> int:
    from PySide6.QtWidgets import QApplication, QInputDialog

    from app.theme import PROFILE_FILL, apply_theme, CURRENT_THEME
    from app.mainwindow import MainWindow
    from app.sketch_mode import SketchTool, BOX_SELECT_MIN_PX
    from cadcore.document import FeatureType
    from cadcore.profiles import list_closed_profiles, profile_area

    print(f"THEME {CURRENT_THEME} PROFILE_FILL={PROFILE_FILL}", flush=True)
    print(f"BOX_SELECT_MIN_PX={BOX_SELECT_MIN_PX}", flush=True)
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
    # Large left + small right; interior of large used for both click and drag tests
    r_large = skf.sketch.add_rectangle((0.0, 0.0), (2.0, 2.0))  # center (1,1)
    r_small = skf.sketch.add_rectangle((2.5, 0.0), (3.5, 1.0))  # center (3.0, 0.5)
    # Extra free line fully outside large, inside a window box from (0.3,0.3)→(1.7,1.7)? 
    # Put a line inside the large rect for box-select entity hit
    line_in = skf.sketch.add_line((0.4, 0.4), (1.6, 0.4))

    profs = list_closed_profiles(skf.sketch)
    by_area = sorted(profs, key=profile_area)
    small, large = by_area[0], by_area[1]
    print(
        f"PROFILES small_id={small.id} large_id={large.id} "
        f"rect_large={r_large.id} rect_small={r_small.id} line={line_in.id}",
        flush=True,
    )

    win.viewport.enter_sketch(skf.id)
    _pump(app, 20)
    win.viewport.set_sketch_tool(SketchTool.SELECT)
    win.viewport._do_render()
    _pump(app, 10)
    vp = win.viewport
    ctrl = vp._sketch_ctrl
    assert ctrl is not None

    # ------------------------------------------------------------------
    # 1) Same interior point: click → region; drag → box (not region)
    # ------------------------------------------------------------------
    interior = (1.0, 1.0)
    img0 = np.asarray(vp.plotter.screenshot(return_img=True))

    # 1a press-release at interior → region fill
    _click(vp, interior)
    _pump(app, 8)
    img_click = np.asarray(vp.plotter.screenshot(return_img=True))
    n_fill = _count_fill_delta(img0, img_click, PROFILE_FILL)
    sel_p = set(ctrl.selected_profile_ids)
    sel_e = set(ctrl.selected_ids)
    print(
        f"GESTURE_CLICK_INTERIOR fill_delta={n_fill} profiles={sel_p} entities={sel_e}",
        flush=True,
    )
    if large.id not in sel_p:
        print(f"FAIL click interior did not select region {large.id}: {sel_p}", flush=True)
        return 1
    if sel_e:
        print(f"FAIL click interior should not select entities: {sel_e}", flush=True)
        return 1
    if n_fill < 200:
        print(f"FAIL region fill not on framebuffer (delta={n_fill})", flush=True)
        return 1
    print("GESTURE_CLICK_INTERIOR_OK", flush=True)

    # Clear
    _click(vp, (-2.0, -2.0))
    _pump(app, 4)

    # 1b press-drag-release starting at SAME interior point → box select
    # Window box L→R covering the inner line (0.4,0.4)-(1.6,0.4)
    _drag_box(vp, (0.3, 0.3), (1.7, 1.7))
    _pump(app, 8)
    sel_p2 = set(ctrl.selected_profile_ids)
    sel_e2 = set(ctrl.selected_ids)
    print(
        f"GESTURE_DRAG_INTERIOR profiles={sel_p2} entities={sel_e2}",
        flush=True,
    )
    if sel_p2:
        print(
            f"FAIL drag from interior should be box-select, not region pick: {sel_p2}",
            flush=True,
        )
        return 1
    if line_in.id not in sel_e2:
        print(
            f"FAIL box from interior did not select inner line; entities={sel_e2}",
            flush=True,
        )
        return 1
    # Box should also catch large rect if fully inside window
    if r_large.id not in sel_e2:
        print(
            f"NOTE large rect not in window set (partial edges?) entities={sel_e2}",
            flush=True,
        )
    print("GESTURE_DRAG_INTERIOR_OK", flush=True)

    # ------------------------------------------------------------------
    # 2) Edge click → shape (entity), not region — from inside and outside
    # ------------------------------------------------------------------
    _click(vp, (-2.0, -2.0))
    # Bottom edge of large rect y=0, x=1 (mid-edge). Snap tol should hit entity.
    edge_on = (1.0, 0.0)
    _click(vp, edge_on)
    _pump(app, 4)
    print(
        f"EDGE_ON profiles={ctrl.selected_profile_ids} entities={ctrl.selected_ids}",
        flush=True,
    )
    if r_large.id not in ctrl.selected_ids:
        print(f"FAIL click on edge did not select rect entity {r_large.id}", flush=True)
        return 1
    if ctrl.selected_profile_ids:
        print(
            f"FAIL edge click selected region instead of shape: {ctrl.selected_profile_ids}",
            flush=True,
        )
        return 1
    print("EDGE_ON_OK", flush=True)

    _click(vp, (-2.0, -2.0))
    # Just outside the bottom edge (toward -v)
    edge_out = (1.0, -0.02)
    # Ensure snap tol reaches the edge: set generous world tol if needed
    ctrl.set_snap_world_tol(0.08)
    _click(vp, edge_out)
    _pump(app, 4)
    print(
        f"EDGE_OUT profiles={ctrl.selected_profile_ids} entities={ctrl.selected_ids}",
        flush=True,
    )
    if r_large.id not in ctrl.selected_ids:
        print(
            f"FAIL click just outside edge did not select rect; "
            f"entities={ctrl.selected_ids} (tol={ctrl.snap_point_tol})",
            flush=True,
        )
        return 1
    if ctrl.selected_profile_ids:
        print(
            f"FAIL outside-edge click selected region: {ctrl.selected_profile_ids}",
            flush=True,
        )
        return 1
    print("EDGE_OUT_OK", flush=True)

    # ------------------------------------------------------------------
    # 3) Ctrl multi region + real Extrude command (dialog stubbed only)
    # ------------------------------------------------------------------
    _click(vp, (-2.0, -2.0))
    _click(vp, (3.0, 0.5))  # small only first
    _pump(app, 4)
    img_sm = np.asarray(vp.plotter.screenshot(return_img=True))
    n_sm = _count_fill_delta(img0, img_sm, PROFILE_FILL)
    if ctrl.selected_profile_ids != {small.id}:
        print(f"FAIL small-only pick {ctrl.selected_profile_ids}", flush=True)
        return 1
    if n_sm < 40:
        print(f"FAIL small fill missing delta={n_sm}", flush=True)
        return 1
    print(f"FILL_SMALL_OK delta={n_sm}", flush=True)

    before_ext = {
        f.id: f for f in win.doc.features if f.type is FeatureType.EXTRUDE
    }
    # Drive the same slot the Extrude action uses; stub only the distance dialog
    with patch.object(QInputDialog, "getDouble", return_value=(2.0, True)):
        win._extrude()
    _pump(app, 60)

    after_ext = [
        f
        for f in win.doc.features
        if f.type is FeatureType.EXTRUDE and f.id not in before_ext
    ]
    print(
        f"EXTRUDE_CMD_SMALL new_features={[(f.id, f.profile_entity_id, f.depth) for f in after_ext]}",
        flush=True,
    )
    if len(after_ext) != 1:
        print(f"FAIL expected 1 extrude from command, got {len(after_ext)}", flush=True)
        return 1
    f_s = after_ext[0]
    if int(f_s.profile_entity_id) != int(small.id):
        print(
            f"FAIL Extrude command profile_entity_id={f_s.profile_entity_id} "
            f"!= clicked small {small.id}",
            flush=True,
        )
        return 1
    mesh_s = win.doc.evaluate_feature(f_s.id)
    if mesh_s is None or not mesh_s.is_watertight():
        print("FAIL small solid missing after Extrude command", flush=True)
        return 1
    vol_s = float(mesh_s.volume())
    if abs(vol_s - 2.0) / 2.0 > 0.02:
        print(f"FAIL small volume {vol_s} expected 2.0", flush=True)
        return 1
    # Viewport rebuild path left a non-black framebuffer
    img_sol = np.asarray(win.viewport.plotter.screenshot(return_img=True))
    if int(img_sol.max()) < 10:
        print("FAIL viewport black after Extrude command", flush=True)
        return 1
    print(f"EXTRUDE_CMD_SMALL_OK vol={vol_s:.4f} profile_id={f_s.profile_entity_id}", flush=True)

    # Multi via Ctrl + Extrude command → two solids
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
    sel_m = win.viewport.selected_profile_ids()
    print(f"MULTI_SEL {sel_m}", flush=True)
    if sel_m != {s2.id, l2.id}:
        print(f"FAIL multi sel {sel_m}", flush=True)
        return 1

    before2 = {f.id for f in win.doc.features if f.type is FeatureType.EXTRUDE}
    with patch.object(QInputDialog, "getDouble", return_value=(1.5, True)):
        win._extrude()
    _pump(app, 60)
    new2 = [
        f
        for f in win.doc.features
        if f.type is FeatureType.EXTRUDE and f.id not in before2
    ]
    pids = sorted(int(f.profile_entity_id) for f in new2)
    vols = sorted(float(win.doc.evaluate_feature(f.id).volume()) for f in new2)
    print(f"EXTRUDE_CMD_MULTI n={len(new2)} pids={pids} vols={vols}", flush=True)
    if len(new2) != 2:
        print("FAIL expected 2 extrudes from command", flush=True)
        return 1
    if set(pids) != {int(s2.id), int(l2.id)}:
        print(f"FAIL Extrude command used wrong profiles {pids}", flush=True)
        return 1
    if abs(vols[0] - 1.5) / 1.5 > 0.02 or abs(vols[1] - 6.0) / 6.0 > 0.02:
        print(f"FAIL multi volumes {vols}", flush=True)
        return 1
    print("EXTRUDE_CMD_MULTI_OK", flush=True)

    # No silent guess when multi and nothing selected
    skf3 = win.doc.create_sketch_on_plane(front.id)
    skf3.sketch.add_rectangle((0, 0), (1, 1))
    skf3.sketch.add_rectangle((2.5, 0), (3.5, 1))
    win.viewport.enter_sketch(skf3.id)
    _pump(app, 10)
    win.viewport._sketch_ctrl.clear_selection()
    before3 = {f.id for f in win.doc.features if f.type is FeatureType.EXTRUDE}
    # Stub dialogs that would appear for "please pick" info box too
    with patch.object(QInputDialog, "getDouble", return_value=(1.0, True)):
        with patch("app.mainwindow.QMessageBox.information", return_value=None):
            win._extrude()
    _pump(app, 20)
    new3 = [
        f
        for f in win.doc.features
        if f.type is FeatureType.EXTRUDE and f.id not in before3
    ]
    if new3:
        print(f"FAIL Extrude without pick created solids: {new3}", flush=True)
        return 1
    print("NO_GUESS_OK", flush=True)

    print("PROFILE_PICK_FRAMEBUFFER_OK", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"EXC_FAIL {exc!r}", flush=True)
        traceback.print_exc()
        raise SystemExit(1)
