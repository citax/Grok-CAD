#!/usr/bin/env python3
"""SolidWorks-style trim/extend: numeric checks + real MainWindow screenshots.

Uses the same SketchController tools the UI ribbon calls. Screenshots go to
bench/_ui_shots/trim_*.png.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

# Prefer real display for screenshots when available
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ["GROK_CAD_UNATTENDED"] = "1"
# Keep DISPLAY if present so we can screenshot a real window
if not os.environ.get("DISPLAY"):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

OUT = _ROOT / "bench" / "_ui_shots"
OUT.mkdir(parents=True, exist_ok=True)

_FAILS: list[str] = []
_OKS: list[str] = []


def _ok(m: str) -> None:
    _OKS.append(m)
    print(f"OK  {m}", flush=True)


def _fail(m: str) -> None:
    _FAILS.append(m)
    print(f"FAIL {m}", flush=True)


def _pump(app, n=15):
    for _ in range(n):
        app.processEvents()
        time.sleep(0.01)


def _shot(win, name: str):
    path = OUT / name
    try:
        # Prefer VTK framebuffer screenshot
        win.viewport.plotter.screenshot(str(path))
        print(f"SHOT {path}", flush=True)
        return path
    except Exception as exc:
        try:
            # Fallback: grab the Qt window
            pix = win.grab()
            pix.save(str(path))
            print(f"SHOT_QT {path}", flush=True)
            return path
        except Exception as exc2:
            print(f"SHOT_FAIL {name}: {exc} / {exc2}", flush=True)
            return None


def _enter_sketch_on_front(win):
    from cadcore.document import FeatureType

    front = next(f for f in win.doc.features if f.type is FeatureType.PLANE_FRONT)
    win.doc.selected_id = front.id
    win._sync_selection(front.id)
    win._enter_sketch()
    _pump(win._app if hasattr(win, "_app") else None, 5) if False else None
    return win.viewport._sketch_feature_id, win.viewport._sketch_ctrl


def main() -> int:
    print("[trim_extend_verify] start", flush=True)
    print(f"  DISPLAY={os.environ.get('DISPLAY')!r} QT_QPA={os.environ.get('QT_QPA_PLATFORM')!r}", flush=True)

    from PySide6.QtWidgets import QApplication

    from app.mainwindow import MainWindow
    from app.sketch_mode import SketchTool
    from app.theme import apply_theme
    from cadcore.document import FeatureType
    from cadcore.sketch import ArcEntity, CircleEntity, LineEntity
    from cadcore.sketch_ops import trim_entity_at, extend_entity_at, entity_point_at

    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app)
    win = MainWindow()
    win.resize(1100, 800)
    try:
        win.show()
    except Exception as exc:
        print(f"show() failed: {exc} — continuing headless", flush=True)
    _pump(app, 25)

    # ------------------------------------------------------------------
    # Scenario 1
    # ------------------------------------------------------------------
    print("\n=== 1. Trim horizontal at 8.5 → end at 6.0 ===", flush=True)
    win.doc = __import__("cadcore.document", fromlist=["Document"]).Document()
    win.doc.seed_reference_planes()
    win.viewport.set_document(win.doc)
    win._refresh_tree()
    front = next(f for f in win.doc.features if f.type is FeatureType.PLANE_FRONT)
    win.doc.selected_id = front.id
    win._sync_selection(front.id)
    win._enter_sketch()
    _pump(app, 15)
    ctrl = win.viewport._sketch_ctrl
    sk = ctrl.sketch
    h = sk.add_line((0, 0), (10, 0))
    sk.add_line((6, -3), (6, 3))
    win.viewport.sync_sketch_visuals()
    _pump(app, 10)
    # Real tool path
    ctrl.set_tool(SketchTool.TRIM)
    # Snapshot like viewport does
    from cadcore.sketch import snapshot_sketch_contents

    before = snapshot_sketch_contents(sk)
    msg = ctrl.on_press((8.5, 0.0))
    after = snapshot_sketch_contents(sk)
    win.doc.record_sketch_contents(win.viewport._sketch_feature_id, before, after)
    win.viewport.sync_sketch_visuals()
    _pump(app, 15)
    print(f"  tool msg: {msg}", flush=True)
    # find horizontal remnant
    horiz = [
        e
        for e in sk.entities
        if isinstance(e, LineEntity) and abs(e.p0[1]) < 1e-9 and abs(e.p1[1]) < 1e-9
    ]
    if not horiz:
        _fail("scenario1: no horizontal line left")
    else:
        ends = sorted([horiz[0].p0[0], horiz[0].p1[0]])
        print(f"  horizontal ends: {ends}", flush=True)
        if abs(ends[0] - 0.0) < 1e-6 and abs(ends[1] - 6.0) < 1e-6:
            _ok(f"scenario1 ends at 6.0 (not 8.5): {ends}")
        else:
            _fail(f"scenario1 wrong ends {ends}")
    _shot(win, "trim_1_end_at_6.png")
    win._exit_sketch()
    _pump(app, 10)

    # ------------------------------------------------------------------
    # Scenario 2
    # ------------------------------------------------------------------
    print("\n=== 2. Middle trim → two pieces, n+1 ===", flush=True)
    win.doc = __import__("cadcore.document", fromlist=["Document"]).Document()
    win.doc.seed_reference_planes()
    win.viewport.set_document(win.doc)
    win._refresh_tree()
    front = next(f for f in win.doc.features if f.type is FeatureType.PLANE_FRONT)
    win.doc.selected_id = front.id
    win._enter_sketch()
    _pump(app, 10)
    ctrl = win.viewport._sketch_ctrl
    sk = ctrl.sketch
    h = sk.add_line((0, 0), (10, 0))
    sk.add_line((3, -3), (3, 3))
    sk.add_line((6, -3), (6, 3))
    n0 = len(sk.entities)
    win.viewport.sync_sketch_visuals()
    _pump(app, 10)
    ctrl.set_tool(SketchTool.TRIM)
    before = snapshot_sketch_contents(sk)
    msg = ctrl.on_press((5.0, 0.0))
    after = snapshot_sketch_contents(sk)
    win.doc.record_sketch_contents(win.viewport._sketch_feature_id, before, after)
    win.viewport.sync_sketch_visuals()
    _pump(app, 15)
    n1 = len(sk.entities)
    print(f"  n {n0} → {n1}  msg={msg}", flush=True)
    horiz = [
        e
        for e in sk.entities
        if isinstance(e, LineEntity) and abs(e.p0[1]) < 1e-9 and abs(e.p1[1]) < 1e-9
    ]
    spans = sorted(tuple(sorted([e.p0[0], e.p1[0]])) for e in horiz)
    print(f"  horizontal spans: {spans}", flush=True)
    if n1 == n0 + 1 and len(horiz) == 2:
        _ok(f"scenario2 entity count {n0}→{n1}, spans={spans}")
    else:
        _fail(f"scenario2 n={n0}→{n1} horiz={len(horiz)} spans={spans}")
    _shot(win, "trim_2_middle_split.png")
    win._exit_sketch()
    _pump(app, 10)

    # ------------------------------------------------------------------
    # Scenario 3
    # ------------------------------------------------------------------
    print("\n=== 3. Extend -8..0 to x=8 ===", flush=True)
    win.doc = __import__("cadcore.document", fromlist=["Document"]).Document()
    win.doc.seed_reference_planes()
    win.viewport.set_document(win.doc)
    win._refresh_tree()
    front = next(f for f in win.doc.features if f.type is FeatureType.PLANE_FRONT)
    win.doc.selected_id = front.id
    win._enter_sketch()
    _pump(app, 10)
    ctrl = win.viewport._sketch_ctrl
    sk = ctrl.sketch
    h = sk.add_line((-8, 0), (0, 0))
    sk.add_line((8, -4), (8, 4))
    win.viewport.sync_sketch_visuals()
    _pump(app, 10)
    ctrl.set_tool(SketchTool.EXTEND)
    before = snapshot_sketch_contents(sk)
    msg = ctrl.on_press((-1.0, 0.0))
    after = snapshot_sketch_contents(sk)
    win.doc.record_sketch_contents(win.viewport._sketch_feature_id, before, after)
    win.viewport.sync_sketch_visuals()
    _pump(app, 15)
    ends = sorted([h.p0[0], h.p1[0]])
    print(f"  ends {ends} msg={msg}", flush=True)
    if abs(ends[0] + 8) < 1e-6 and abs(ends[1] - 8) < 1e-6:
        _ok(f"scenario3 extended to x=8.0: {ends}")
    else:
        _fail(f"scenario3 ends {ends}")
    _shot(win, "extend_3_to_8.png")
    win._exit_sketch()
    _pump(app, 10)

    # ------------------------------------------------------------------
    # Scenario 4a arc
    # ------------------------------------------------------------------
    print("\n=== 4a. Trim arc at right of vertical x=5 ===", flush=True)
    win.doc = __import__("cadcore.document", fromlist=["Document"]).Document()
    win.doc.seed_reference_planes()
    win.viewport.set_document(win.doc)
    win._refresh_tree()
    front = next(f for f in win.doc.features if f.type is FeatureType.PLANE_FRONT)
    win.doc.selected_id = front.id
    win._enter_sketch()
    _pump(app, 10)
    ctrl = win.viewport._sketch_ctrl
    sk = ctrl.sketch
    arc = sk.add_arc((0, 0), (5, 5), (10, 0))
    sk.add_line((5, -2), (5, 8))
    r_before = arc.radius
    p0b, p1b = arc.p0(), arc.p1()
    pt = entity_point_at(arc, 0.75)
    win.viewport.sync_sketch_visuals()
    _pump(app, 10)
    ctrl.set_tool(SketchTool.TRIM)
    before = snapshot_sketch_contents(sk)
    msg = ctrl.on_press((float(pt[0]), float(pt[1])))
    after = snapshot_sketch_contents(sk)
    win.doc.record_sketch_contents(win.viewport._sketch_feature_id, before, after)
    win.viewport.sync_sketch_visuals()
    _pump(app, 15)
    arcs = [e for e in sk.entities if isinstance(e, ArcEntity)]
    print(f"  before p0={p0b} p1={p1b} r={r_before:.4f}", flush=True)
    print(f"  click={pt} msg={msg} n_arcs={len(arcs)}", flush=True)
    for a in arcs:
        print(f"  arc p0={a.p0()} p1={a.p1()}", flush=True)
    end_xs = []
    for a in arcs:
        end_xs += [a.p0()[0], a.p1()[0]]
    if arcs and any(abs(x - 5.0) < 0.05 for x in end_xs):
        _ok(f"scenario4a arc trimmed to x≈5 ends={end_xs}")
    else:
        _fail(f"scenario4a arcs={len(arcs)} ends={end_xs}")
    _shot(win, "trim_4a_arc.png")
    win._exit_sketch()
    _pump(app, 10)

    # ------------------------------------------------------------------
    # Scenario 4b circle
    # ------------------------------------------------------------------
    print("\n=== 4b. Trim circle right of two verticals ===", flush=True)
    win.doc = __import__("cadcore.document", fromlist=["Document"]).Document()
    win.doc.seed_reference_planes()
    win.viewport.set_document(win.doc)
    win._refresh_tree()
    front = next(f for f in win.doc.features if f.type is FeatureType.PLANE_FRONT)
    win.doc.selected_id = front.id
    win._enter_sketch()
    _pump(app, 10)
    ctrl = win.viewport._sketch_ctrl
    sk = ctrl.sketch
    circ = sk.add_circle((0, 0), 5.0)
    sk.add_line((-1, -8), (-1, 8))
    sk.add_line((1, -8), (1, 8))
    n0 = len(sk.entities)
    win.viewport.sync_sketch_visuals()
    _pump(app, 10)
    ctrl.set_tool(SketchTool.TRIM)
    before = snapshot_sketch_contents(sk)
    msg = ctrl.on_press((5.0, 0.0))
    after = snapshot_sketch_contents(sk)
    win.doc.record_sketch_contents(win.viewport._sketch_feature_id, before, after)
    win.viewport.sync_sketch_visuals()
    _pump(app, 15)
    n1 = len(sk.entities)
    n_circ = sum(1 for e in sk.entities if isinstance(e, CircleEntity))
    n_arc = sum(1 for e in sk.entities if isinstance(e, ArcEntity))
    print(f"  n {n0}→{n1} circles={n_circ} arcs={n_arc} msg={msg}", flush=True)
    for e in sk.entities:
        if isinstance(e, ArcEntity):
            print(f"  arc p0={e.p0()} p1={e.p1()}", flush=True)
    if n_circ == 0 and n_arc >= 1:
        _ok(f"scenario4b circle→{n_arc} arc(s), n {n0}→{n1}")
    else:
        _fail(f"scenario4b circ={n_circ} arc={n_arc}")
    _shot(win, "trim_4b_circle.png")
    win._exit_sketch()
    _pump(app, 10)

    print("\n=== SUMMARY ===", flush=True)
    print(f"PASS {len(_OKS)}  FAIL {len(_FAILS)}", flush=True)
    for f in _FAILS:
        print(f"  - {f}", flush=True)
    print(f"Screenshots in {OUT}", flush=True)
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
