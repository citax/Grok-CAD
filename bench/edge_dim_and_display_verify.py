#!/usr/bin/env python3
"""Verify: clicked rectangle edge is the one dimensioned; measure via VTK actors.

Display path (from source, not assumed):

  Document
    → Viewport.schedule_rebuild()
    → GeometryRebuildJob(gen, snapshot_features(self._doc))   # app/viewport.py
    → evaluate_solids_snapshot(features)                       # app/workers.py
    → _on_job_finished → _apply_solid_results(results)
    → plotter.add_mesh(..., name=f"solid_{fid}")
    → VTK composites that actor into the framebuffer
    → plotter.screenshot / the pixels you see

  solid_display_vertices(fid) reads mapper input of that actor.
  Document.evaluate_feature is a *separate* in-process evaluation used for
  export/export-like checks — it does not upload to VTK.

Sketch strokes: _upsert_entity_actor → sk_e_{id} on overlay → same render.
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
os.environ.setdefault("GROK_THEME", "light")

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


def _pump(app, n=20):
    for _ in range(n):
        app.processEvents()
        time.sleep(0.01)


def _shot(vp, name: str):
    path = OUT / name
    try:
        if vp._render_timer.isActive():
            vp._render_timer.stop()
        vp._do_render()
        vp.plotter.screenshot(str(path))
        print(f"SHOT {path}", flush=True)
    except Exception as exc:
        print(f"SHOT_FAIL {name}: {exc}", flush=True)


def _wait_solid_actor(vp, fid, app, timeout=10.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        _pump(app, 5)
        if vp.solid_display_vertices(fid) is not None:
            return True
    return False


def _prove_display_path() -> None:
    print("\n=== DISPLAY PATH PROOF (source quotes) ===", flush=True)
    vp = (_ROOT / "app" / "viewport.py").read_text()
    wr = (_ROOT / "app" / "workers.py").read_text()
    proofs = [
        (
            "rebuild starts worker with snapshot_features(doc)",
            "GeometryRebuildJob(gen, snapshot_features(self._doc))" in vp
            or "GeometryRebuildJob(gen, snapshot_features" in vp,
        ),
        (
            "job result lands in _apply_solid_results",
            "self._apply_solid_results(results)" in vp,
        ),
        (
            "solid mesh uploaded as named actor solid_{fid}",
            "solid_{fid}" in vp and "add_mesh" in vp,
        ),
        (
            "worker builds verts/faces via evaluate_solids_snapshot",
            "def evaluate_solids_snapshot" in wr
            and "results[f.id] = (" in wr,
        ),
        (
            "solid_display_vertices reads actor.GetMapper().GetInput()",
            "def solid_display_vertices" in vp and "GetMapper()" in vp,
        ),
    ]
    for label, ok in proofs:
        (_ok if ok else _fail)(f"path: {label}")
    print(
        "Screen solids = actors filled by _apply_solid_results. "
        "evaluate_feature is NOT that upload.",
        flush=True,
    )


def main() -> int:
    from PySide6.QtWidgets import QApplication, QInputDialog

    from app.mainwindow import MainWindow
    from app.sketch_mode import SketchTool
    from app.theme import apply_theme
    from cadcore.document import FeatureType
    from cadcore.sketch import rect_height, rect_width

    _prove_display_path()

    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app)
    win = MainWindow()
    win.resize(1280, 800)
    win.show()
    _pump(app, 30)

    print("\n=== EDGE DIMENSION (Smart Dim tool, role from click) ===", flush=True)
    front = next(f for f in win.doc.features if f.type is FeatureType.PLANE_FRONT)
    win.doc.selected_id = front.id
    win._sync_selection(front.id)
    win._enter_sketch()
    _pump(app, 15)
    sid = win.viewport._sketch_feature_id
    sk = win.viewport._sketch_ctrl.sketch

    rect = sk.add_rectangle((0.0, 0.0), (30.0, 20.0))
    win.doc.record_entity_add(sid, rect)
    win.viewport.sync_sketch_visuals()
    _pump(app, 10)

    # Same as ribbon: select Smart Dimension tool
    win._on_sketch_tool(SketchTool.DIMENSION)
    assert win.viewport._sketch_ctrl.tool is SketchTool.DIMENSION

    iw = win.viewport.plotter.interactor

    def find_xy(target_uv, tol=1.25):
        best = None
        best_d = 1e9
        w, h = iw.width(), iw.height()
        for yy in range(4, h - 4, 2):
            for xx in range(4, w - 4, 2):
                uv = win.viewport._mouse_to_uv(xx, yy)
                if uv is None:
                    continue
                d = abs(uv[0] - target_uv[0]) + abs(uv[1] - target_uv[1])
                if d < best_d:
                    best_d = d
                    best = (xx, yy, uv, best_d)
        if best is None or best[3] > tol:
            return None
        return best[0], best[1], best[2]

    def edge_mid(name: str):
        u0, u1 = sorted([rect.c0[0], rect.c1[0]])
        v0, v1 = sorted([rect.c0[1], rect.c1[1]])
        return {
            "bottom": ((u0 + u1) * 0.5, v0),
            "top": ((u0 + u1) * 0.5, v1),
            "left": (u0, (v0 + v1) * 0.5),
            "right": (u1, (v0 + v1) * 0.5),
        }[name]

    # bottom→width 40, top→width 55, left→height 25, right→height 18
    cases = [
        ("bottom", "width", 40.0),
        ("top", "width", 55.0),
        ("left", "height", 25.0),
        ("right", "height", 18.0),
    ]

    for edge_name, expect_role, value in cases:
        mid = edge_mid(edge_name)
        hit = find_xy(mid)
        received: list[tuple[int, str]] = []

        def _cap(eid: int, role: str, _recv=received) -> None:
            _recv.append((int(eid), str(role)))

        # Steal the signal so we observe what the tool decides, then apply ourselves
        try:
            win.viewport.dimension_requested.disconnect()
        except Exception:
            pass
        win.viewport.dimension_requested.connect(_cap)

        used_path = "mouse"
        if hit is not None:
            xx, yy, uv = hit
            print(
                f"  EDGE {edge_name} mid={mid} display=({xx},{yy}) mapped_uv={uv}",
                flush=True,
            )
            win.viewport._sketch_mouse_press(float(xx), float(yy), shift=False)
        else:
            used_path = "controller_uv"
            print(f"  EDGE {edge_name} mid={mid} (no pixel hit; ctrl.on_press UV)", flush=True)
            msg = win.viewport._sketch_ctrl.on_press(mid)
            if msg and msg.startswith("DimPick:"):
                parts = msg.split(":")
                received.append((int(parts[1]), parts[2]))
            else:
                _fail(f"{edge_name}: no DimPick ({msg!r})")
                continue

        if not received:
            # mouse path: signal should have fired; if not, read status from ctrl
            msg = None
            _fail(f"{edge_name}: dimension_requested did not fire via {used_path}")
            continue

        eid, role = received[0]
        if role != expect_role:
            _fail(f"{edge_name}: tool role={role!r} expected {expect_role!r} ({used_path})")
        else:
            _ok(f"{edge_name}: tool role={role} via {used_path} (not handed in)")

        # Type the value through the real MainWindow command (dialog stub only)
        with patch.object(QInputDialog, "getDouble", return_value=(value, True)):
            win._on_dimension_requested(eid, role)

        got = rect_width(rect) if expect_role == "width" else rect_height(rect)
        other = rect_height(rect) if expect_role == "width" else rect_width(rect)
        if abs(got - value) > 1e-6:
            _fail(f"{edge_name}: {expect_role}={got} after typing {value}")
        else:
            _ok(f"{edge_name}: {expect_role}→{got:g} (other={other:g})")

        # Actor path for the sketch stroke; grow/fit view so later edges stay on-screen
        win.viewport.sync_sketch_visuals()
        try:
            from cadcore.scale import sketch_entity_uv_extent, sketch_parallel_scale

            ext = sketch_entity_uv_extent(sk.entities)
            # Cover max corner distance from origin (rect grows only in +u/+v)
            u0, u1 = sorted([rect.c0[0], rect.c1[0]])
            v0, v1 = sorted([rect.c0[1], rect.c1[1]])
            cover = max(abs(u0), abs(u1), abs(v0), abs(v1), ext) * 1.25
            cam = win.viewport.plotter.camera
            if bool(cam.GetParallelProjection()):
                cam.SetParallelScale(max(float(cam.GetParallelScale()), cover, 20.0))
            win.viewport._draw_sketch_overlay()
        except Exception as exc:
            print(f"  fit_view: {exc}", flush=True)
        _pump(app, 5)
        pts = win.viewport.sketch_entity_display_points(rect.id)
        if pts is None or len(pts) < 2:
            _fail(f"{edge_name}: missing sk_e_{rect.id} actor points")
        else:
            fr = sk.frame
            us = [float(np.dot(p - fr.origin, fr.u_axis)) for p in pts]
            vs = [float(np.dot(p - fr.origin, fr.v_axis)) for p in pts]
            w_disp, h_disp = max(us) - min(us), max(vs) - min(vs)
            target = value
            measured = w_disp if expect_role == "width" else h_disp
            if abs(measured - target) > 0.08:
                _fail(
                    f"{edge_name}: actor {expect_role}={measured:.3f} != {target} "
                    f"(u={w_disp:.3f} v={h_disp:.3f})"
                )
            else:
                _ok(
                    f"{edge_name}: actor {expect_role}={measured:.2f} "
                    f"from sk_e_{rect.id} mapper"
                )

    # Restore MainWindow signal hookup
    try:
        win.viewport.dimension_requested.disconnect()
    except Exception:
        pass
    win.viewport.dimension_requested.connect(win._on_dimension_requested)

    if abs(rect_width(rect) - 55.0) > 1e-6 or abs(rect_height(rect) - 18.0) > 1e-6:
        _fail(f"final rect {rect_width(rect)}×{rect_height(rect)} expected 55×18")
    else:
        _ok(f"final rect {rect_width(rect):g}×{rect_height(rect):g}")

    _shot(win.viewport, "edge_dim_four_edges.png")

    # ----- Solid via display actors -----
    print("\n=== SOLID VIA DISPLAY ACTOR (not evaluate_feature) ===", flush=True)
    win.viewport.exit_sketch()
    _pump(app, 5)
    solid = win.doc.create_extrude(sid, 10.0, profile_entity_id=rect.id)
    win.viewport.schedule_rebuild()
    if not _wait_solid_actor(win.viewport, solid.id, app):
        _fail("solid actor never appeared")
    else:
        _ok(f"actor solid_{solid.id} present after GeometryRebuildJob")

    verts = win.viewport.solid_display_vertices(solid.id)
    if verts is None:
        _fail("solid_display_vertices is None")
    else:
        dx = float(verts[:, 0].max() - verts[:, 0].min())
        dy = float(verts[:, 1].max() - verts[:, 1].min())
        dz = float(verts[:, 2].max() - verts[:, 2].min())
        print(f"  ACTOR_AABB dx={dx:.4f} dy={dy:.4f} dz={dz:.4f} n={len(verts)}", flush=True)
        (_ok if abs(dx - 55) < 0.1 else _fail)(f"actor X={dx:.3f} (want 55)")
        (_ok if abs(dy - 18) < 0.1 else _fail)(f"actor Y={dy:.3f} (want 18)")
        (_ok if abs(dz - 10) < 0.1 else _fail)(f"actor Z={dz:.3f} (want 10)")

    win.viewport.set_view("iso")
    _pump(app, 8)
    _shot(win.viewport, "edge_dim_extrude_actor.png")

    print("\n=== SUMMARY ===", flush=True)
    print(f"passed={len(_OKS)} failed={len(_FAILS)}", flush=True)
    for f in _FAILS:
        print(f"  FAIL: {f}", flush=True)
    return 1 if _FAILS else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
