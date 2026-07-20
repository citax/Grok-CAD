#!/usr/bin/env python3
"""Prove motion-based under/well/over colors with real MainWindow + drags.

Requires working Qt GUI (xcb + libxcb-cursor). Screenshots → bench/_ui_shots/dof_*.png
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ["GROK_CAD_UNATTENDED"] = "1"

OUT = _ROOT / "bench" / "_ui_shots"
OUT.mkdir(parents=True, exist_ok=True)

_FAILS: list[str] = []
_OKS: list[str] = []
_STATUS: list[str] = []


def _ok(m: str) -> None:
    _OKS.append(m)
    print(f"OK  {m}", flush=True)


def _fail(m: str) -> None:
    _FAILS.append(m)
    print(f"FAIL {m}", flush=True)


def _pump(app, n=20):
    for _ in range(n):
        app.processEvents()
        time.sleep(0.01)


def _shot(win, name: str):
    path = OUT / name
    try:
        win.viewport.plotter.screenshot(str(path))
        print(f"SHOT {path}", flush=True)
        return path
    except Exception as exc:
        try:
            win.grab().save(str(path))
            print(f"SHOT_QT {path}", flush=True)
            return path
        except Exception as exc2:
            print(f"SHOT_FAIL {name}: {exc} / {exc2}", flush=True)
            return None


def _status(win) -> str:
    msg = win.statusBar().currentMessage()
    _STATUS.append(msg)
    print(f"  STATUS: {msg}", flush=True)
    return msg


def _drag_handle(ctrl, eid, handle, start, to):
    from app.sketch_mode import DragState, SketchTool
    from cadcore.sketch import HandleKind

    kind = HandleKind.CORNER if str(handle).startswith("c") else HandleKind.ENDPOINT
    if handle == "mid":
        kind = HandleKind.MIDPOINT
    ctrl.tool = SketchTool.SELECT
    ctrl.drag = DragState(eid, handle, kind, start)
    ctrl._apply_drag(to)
    ctrl.drag = None


def main() -> int:
    print("[dof_color_verify] start", flush=True)
    import numpy as np
    from PySide6.QtWidgets import QApplication

    from app.mainwindow import MainWindow
    from app.theme import apply_theme
    from cadcore.constraints import (
        ConstraintKind,
        SketchConstraint,
        add_constraint,
        solve_sketch,
    )
    from cadcore.document import FeatureType
    from cadcore.dof import entity_dof_status, format_dof_status_line, worst_constraint_label
    from cadcore.sketch import LineEntity, RectEntity

    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app)
    win = MainWindow()
    win.resize(1100, 800)
    win.show()
    _pump(app, 30)
    print(f"  platform={app.platformName()} visible={win.isVisible()}", flush=True)
    if not win.isVisible():
        _fail("MainWindow not visible")
        return 1

    def fresh_sketch():
        win.doc = __import__("cadcore.document", fromlist=["Document"]).Document()
        win.doc.seed_reference_planes()
        win.viewport.set_document(win.doc)
        win._refresh_tree()
        front = next(f for f in win.doc.features if f.type is FeatureType.PLANE_FRONT)
        win.doc.selected_id = front.id
        win._enter_sketch()
        _pump(app, 15)
        return win.viewport._sketch_ctrl, win.viewport._sketch_feature_id

    # ==================================================================
    # 1) Bare rectangle = blue, drag moves
    # ==================================================================
    print("\n=== 1. Bare rectangle (blue, movable) ===", flush=True)
    ctrl, sid = fresh_sketch()
    sk = ctrl.sketch
    r = sk.add_rectangle((0, 0), (20, 10))
    rid = r.id
    win.viewport.sync_sketch_visuals()
    _pump(app, 15)
    st = entity_dof_status(sk, sk.find_entity(rid))
    sb = _status(win)
    print(f"  color-status={st}", flush=True)
    r = sk.find_entity(rid)
    s = r.corners()[2]
    _drag_handle(ctrl, rid, "c2", s, (30, 20))
    win.viewport.sync_sketch_visuals()
    _pump(app, 15)
    r = sk.find_entity(rid)
    e = r.corners()[2]
    d1 = float(np.hypot(e[0] - s[0], e[1] - s[1]))
    print(f"  drag distance = {d1:.4f} mm  {s} -> {e}", flush=True)
    if st == "under" and d1 > 1.0:
        _ok(f"1 under-defined blue, drag moved {d1:.4f} mm")
    else:
        _fail(f"1 st={st} d={d1}")
    _shot(win, "dof_1_under_rect.png")
    win._exit_sketch()
    _pump(app, 10)

    # ==================================================================
    # 2) Fix + width + height = black, drag 0
    # ==================================================================
    print("\n=== 2. Fully defined rect (black, immovable) ===", flush=True)
    ctrl, sid = fresh_sketch()
    sk = ctrl.sketch
    r = sk.add_rectangle((0, 0), (20, 10))
    rid = r.id
    add_constraint(sk, SketchConstraint(id=-1, kind=ConstraintKind.FIX, e0=rid, h0="c0"))
    win.doc.record_sketch_contents  # noqa: keep sketch dirty ok
    # record constraint properly
    from cadcore.sketch import snapshot_sketch_contents

    before = snapshot_sketch_contents(sk)
    # constraint already added; snapshot after
    after = snapshot_sketch_contents(sk)
    win.doc.record_sketch_contents(sid, before, after)
    win.doc.apply_sketch_dimension(sid, rid, "width", 20.0)
    win.doc.apply_sketch_dimension(sid, rid, "height", 10.0)
    win.viewport.sync_sketch_visuals()
    _pump(app, 15)
    st = entity_dof_status(sk, sk.find_entity(rid))
    sb = _status(win)
    print(f"  color-status={st}", flush=True)
    r = sk.find_entity(rid)
    s = r.corners()[2]
    _drag_handle(ctrl, rid, "c2", s, (50, 50))
    win.viewport.sync_sketch_visuals()
    _pump(app, 15)
    r = sk.find_entity(rid)
    e = r.corners()[2]
    d2 = float(np.hypot(e[0] - s[0], e[1] - s[1]))
    print(f"  drag distance = {d2:.6f} mm  {s} -> {e}", flush=True)
    if st == "well" and d2 < 0.05:
        _ok(f"2 fully defined black, drag moved {d2:.6f} mm (≈0)")
    else:
        _fail(f"2 st={st} d={d2}")
    _shot(win, "dof_2_fully_defined.png")
    win._exit_sketch()
    _pump(app, 10)

    # ==================================================================
    # 3) Remove height → blue again, drag moves
    # ==================================================================
    print("\n=== 3. Remove height dim → under again ===", flush=True)
    ctrl, sid = fresh_sketch()
    sk = ctrl.sketch
    r = sk.add_rectangle((0, 0), (20, 10))
    rid = r.id
    add_constraint(sk, SketchConstraint(id=-1, kind=ConstraintKind.FIX, e0=rid, h0="c0"))
    win.doc.apply_sketch_dimension(sid, rid, "width", 20.0)
    win.doc.apply_sketch_dimension(sid, rid, "height", 10.0)
    # drop height
    before = snapshot_sketch_contents(sk)
    sk.dimensions = [d for d in sk.dimensions if d.role != "height"]
    solve_sketch(sk)
    after = snapshot_sketch_contents(sk)
    win.doc.record_sketch_contents(sid, before, after)
    win.viewport.sync_sketch_visuals()
    _pump(app, 15)
    st = entity_dof_status(sk, sk.find_entity(rid))
    sb = _status(win)
    print(f"  color-status={st}", flush=True)
    r = sk.find_entity(rid)
    s = r.corners()[2]
    _drag_handle(ctrl, rid, "c2", s, (s[0], s[1] + 15))
    win.viewport.sync_sketch_visuals()
    _pump(app, 15)
    r = sk.find_entity(rid)
    e = r.corners()[2]
    d3 = float(abs(e[1] - s[1]))
    print(f"  drag |Δv| = {d3:.4f} mm", flush=True)
    if st == "under" and d3 > 1.0:
        _ok(f"3 under again, drag |Δv|={d3:.4f} mm")
    else:
        _fail(f"3 st={st} d={d3}")
    _shot(win, "dof_3_under_again.png")
    win._exit_sketch()
    _pump(app, 10)

    # ==================================================================
    # 4) Redundant parallel — still blue, still moves
    # ==================================================================
    print("\n=== 4. H+H+Parallel still under (not counting) ===", flush=True)
    ctrl, sid = fresh_sketch()
    sk = ctrl.sketch
    a = sk.add_line((0, 0), (10, 0))
    b = sk.add_line((0, 5), (8, 5))
    for c in (
        SketchConstraint(id=-1, kind=ConstraintKind.HORIZONTAL, e0=a.id),
        SketchConstraint(id=-1, kind=ConstraintKind.HORIZONTAL, e0=b.id),
        SketchConstraint(id=-1, kind=ConstraintKind.PARALLEL, e0=a.id, e1=b.id),
    ):
        add_constraint(sk, c)
    win.viewport.sync_sketch_visuals()
    _pump(app, 15)
    sa = entity_dof_status(sk, sk.find_entity(a.id))
    sb_ = entity_dof_status(sk, sk.find_entity(b.id))
    sb = _status(win)
    print(f"  a={sa} b={sb_}", flush=True)
    a = sk.find_entity(a.id)
    mid = a.midpoint()
    _drag_handle(ctrl, a.id, "mid", mid, (mid[0], mid[1] + 7))
    win.viewport.sync_sketch_visuals()
    _pump(app, 15)
    a = sk.find_entity(a.id)
    d4 = float(abs(a.midpoint()[1] - mid[1]))
    print(f"  mid drag |Δv| = {d4:.4f} mm  p0={a.p0} p1={a.p1}", flush=True)
    if sa == "under" and sb_ == "under" and d4 > 1.0:
        _ok(f"4 still blue despite 3 relations, mid drag {d4:.4f} mm")
    else:
        _fail(f"4 sa={sa} sb={sb_} d={d4}")
    _shot(win, "dof_4_redundant_parallel.png")
    win._exit_sketch()
    _pump(app, 10)

    # ==================================================================
    # 5) Over-defined — red + guilty
    # ==================================================================
    print("\n=== 5. Over-defined (red + guilty) ===", flush=True)
    ctrl, sid = fresh_sketch()
    sk = ctrl.sketch
    ln = sk.add_line((0, 0), (10, 0))
    lid = ln.id
    add_constraint(sk, SketchConstraint(id=-1, kind=ConstraintKind.FIX, e0=lid, h0="p0"))
    add_constraint(sk, SketchConstraint(id=-1, kind=ConstraintKind.FIX, e0=lid, h0="p1"))
    # Force conflicting length promise (fixed ends length=10, dim wants 20)
    sk.add_or_update_dimension(lid, "length", 20.0)
    solve_sketch(sk, max_iters=30)
    win.viewport.sync_sketch_visuals()
    _pump(app, 15)
    st = entity_dof_status(sk, sk.find_entity(lid))
    guilty = worst_constraint_label(sk)
    sb = _status(win)
    print(f"  color-status={st}", flush=True)
    print(f"  guilty={guilty}", flush=True)
    if st == "over" and guilty and "length" in guilty:
        _ok(f"5 over-defined red; guilty: {guilty}")
    else:
        _fail(f"5 st={st} guilty={guilty}")
    _shot(win, "dof_5_overdefined.png")
    win._exit_sketch()
    _pump(app, 10)

    print("\n=== SUMMARY ===", flush=True)
    print(f"PASS {len(_OKS)}  FAIL {len(_FAILS)}", flush=True)
    for f in _FAILS:
        print(f"  - {f}", flush=True)
    print("Status bar lines:", flush=True)
    for s in _STATUS:
        print(f"  · {s}", flush=True)
    print(f"Screenshots: {OUT}/dof_*.png", flush=True)
    try:
        win.close()
    except Exception:
        pass
    _pump(app, 5)
    return 1 if _FAILS else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
