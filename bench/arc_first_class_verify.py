#!/usr/bin/env python3
"""Unattended verification: arc as a first-class sketch entity.

Exercises the exact same code paths that real mouse actions use
(SketchController.on_press, _apply_drag, set_tool, add_constraint, etc.).

Every check can fail.  Exit code 0 = all passed, 1 = something failed.

Must exit on its own (GROK_CAD_UNATTENDED=1, no blocking dialogs).
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["GROK_CAD_UNATTENDED"] = "1"
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
# Suppress VTK noise
os.environ.pop("DISPLAY", None)
os.environ.pop("WAYLAND_DISPLAY", None)

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

PASS = 0
FAIL = 1


def _kink(line, arc, end: str) -> float:
    """| sin(angle) | between line direction and arc tangent at join.  0 = smooth."""
    import numpy as np

    d = np.array(
        [line.p1[0] - line.p0[0], line.p1[1] - line.p0[1]], dtype=np.float64
    )
    nd = float(np.linalg.norm(d))
    if nd < 1e-12:
        return 1.0
    d /= nd
    t = arc.tangent_at_start() if end == "p0" else arc.tangent_at_end()
    return abs(float(d[0] * t[1] - d[1] * t[0]))


def main() -> int:  # noqa: C901
    print("[arc_verify] start", flush=True)
    try:
        import numpy as np

        from app.sketch_mode import DragState, HandleKind, SketchController, SketchTool
        from cadcore.constraints import (
            ConstraintKind,
            SketchConstraint,
            add_constraint,
            solve_sketch,
        )
        from cadcore.document import Document, FeatureType
        from cadcore.profiles import find_closed_line_loops
        from cadcore.project_io import load_document, save_document
        from cadcore.sketch import ArcEntity, LineEntity
    except Exception as exc:
        print(f"[arc_verify] import failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return FAIL

    def doc_sketch():
        doc = Document()
        doc.seed_reference_planes()
        front = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
        skf = doc.create_sketch_on_plane(front.id)
        return doc, skf, skf.sketch

    errors = []

    def check(label: str, cond: bool, detail: str = ""):
        tag = "PASS" if cond else "FAIL"
        msg = f"  [{tag}] {label}"
        if detail:
            msg += f"  ({detail})"
        print(msg, flush=True)
        if not cond:
            errors.append(label)

    # ==================================================================
    # 1.  Arc drawing via the three-click tool
    # ==================================================================
    print("\n=== 1. Arc Drawing (three-click tool) ===", flush=True)
    doc, skf, sk = doc_sketch()
    ctrl = SketchController(sk)
    ctrl.set_tool(SketchTool.ARC)

    msg1 = ctrl.on_press((0.0, 0.0))
    check("click-1 starts draw", ctrl.draw is not None and len(ctrl.draw.points) == 1)

    msg2 = ctrl.on_press((5.0, 4.0))
    check(
        "click-2 records on-arc point, arc not finished",
        ctrl.draw is not None and len(ctrl.draw.points) == 2 and msg2 is None,
    )

    msg3 = ctrl.on_press((10.0, 0.0))
    check("click-3 finishes arc", msg3 == "Arc" and ctrl.draw is None)

    arcs = [e for e in sk.entities if isinstance(e, ArcEntity)]
    check("one ArcEntity in sketch", len(arcs) == 1)
    arc = arcs[0]
    check("arc has positive radius", arc.radius > 1e-6, f"r={arc.radius:.6f}")
    check(
        "p0 near (0,0)",
        np.hypot(arc.p0()[0] - 0, arc.p0()[1] - 0) < 0.01,
        f"p0={arc.p0()}",
    )
    check(
        "p1 near (10,0)",
        np.hypot(arc.p1()[0] - 10, arc.p1()[1] - 0) < 0.01,
        f"p1={arc.p1()}",
    )
    print(
        f"  Arc created: center={arc.center}, r={arc.radius:.4f}, "
        f"p0={arc.p0()}, p1={arc.p1()}, mid={arc.mid_uv()}",
        flush=True,
    )

    # ==================================================================
    # 2.  Drag reshaping — mid, p0, p1, center  (coordinates before/after)
    # ==================================================================
    print("\n=== 2. Arc Drag Reshaping ===", flush=True)
    ctrl.tool = SketchTool.SELECT

    # -- mid drag --
    mid_b = arc.mid_uv()
    ctrl.drag = DragState(arc.id, "mid", HandleKind.MIDPOINT, mid_b)
    ctrl._apply_drag((5.0, 8.0))
    ctrl.drag = None
    mid_a = arc.mid_uv()
    d_mid = np.hypot(mid_a[0] - mid_b[0], mid_a[1] - mid_b[1])
    print(f"  mid drag: before={mid_b} → after={mid_a}  Δ={d_mid:.4f}", flush=True)
    check("mid handle drag moves point", d_mid > 0.5, f"Δ={d_mid:.4f}")

    # -- p1 drag --
    p1_b = arc.p1()
    ctrl.drag = DragState(arc.id, "p1", HandleKind.ENDPOINT, p1_b)
    ctrl._apply_drag((12.0, 2.0))
    ctrl.drag = None
    p1_a = arc.p1()
    d_p1 = np.hypot(p1_a[0] - p1_b[0], p1_a[1] - p1_b[1])
    print(f"  p1 drag: before={p1_b} → after={p1_a}  Δ={d_p1:.4f}", flush=True)
    check("p1 handle drag moves point", d_p1 > 0.5, f"Δ={d_p1:.4f}")

    # -- p0 drag --
    p0_b = arc.p0()
    ctrl.drag = DragState(arc.id, "p0", HandleKind.ENDPOINT, p0_b)
    ctrl._apply_drag((-2.0, 3.0))
    ctrl.drag = None
    p0_a = arc.p0()
    d_p0 = np.hypot(p0_a[0] - p0_b[0], p0_a[1] - p0_b[1])
    print(f"  p0 drag: before={p0_b} → after={p0_a}  Δ={d_p0:.4f}", flush=True)
    check("p0 handle drag moves point", d_p0 > 0.5, f"Δ={d_p0:.4f}")

    # -- center drag --
    c_b = arc.center
    ctrl.drag = DragState(arc.id, "center", HandleKind.CENTER, c_b)
    ctrl._apply_drag((c_b[0] + 5, c_b[1] + 5))
    ctrl.drag = None
    c_a = arc.center
    d_c = np.hypot(c_a[0] - c_b[0], c_a[1] - c_b[1])
    print(f"  center drag: before={c_b} → after={c_a}  Δ={d_c:.4f}", flush=True)
    check("center handle drag moves point", d_c > 0.5, f"Δ={d_c:.4f}")

    # -- deletion --
    sk.remove_entity(arc.id)
    check(
        "arc deleted",
        len([e for e in sk.entities if isinstance(e, ArcEntity)]) == 0,
    )

    # ==================================================================
    # 3.  Radius dimension applied to an arc, survives drag
    # ==================================================================
    print("\n=== 3. Radius Dimension on Arc ===", flush=True)
    doc, skf, sk = doc_sketch()
    arc = sk.add_arc((0, 0), (4, 3), (8, 0))
    r_native = arc.radius
    TARGET_R = 12.0
    doc.apply_sketch_dimension(skf.id, arc.id, "radius", TARGET_R)
    check(
        f"radius set to {TARGET_R}",
        abs(arc.radius - TARGET_R) < 1e-6,
        f"actual={arc.radius:.6f}",
    )
    print(f"  Measured radius after apply: {arc.radius:.6f}", flush=True)

    ctrl = SketchController(sk)
    p1_b = arc.p1()
    ctrl.drag = DragState(arc.id, "p1", HandleKind.ENDPOINT, p1_b)
    ctrl._apply_drag((15.0, 6.0))
    ctrl.drag = None
    p1_a = arc.p1()
    d_p1 = np.hypot(p1_a[0] - p1_b[0], p1_a[1] - p1_b[1])
    print(
        f"  After drag p1: before={p1_b} → after={p1_a}  Δ={d_p1:.4f}",
        flush=True,
    )
    print(f"  Measured radius after drag: {arc.radius:.6f}", flush=True)
    check("drag moved p1", d_p1 > 0.5, f"Δ={d_p1:.4f}")
    check(
        "radius held through drag",
        abs(arc.radius - TARGET_R) < 0.01,
        f"actual={arc.radius:.6f}",
    )

    # ==================================================================
    # 4.  Tangent constraint: smooth join arc ↔ line
    # ==================================================================
    print("\n=== 4. Tangent Constraint (arc ↔ line) ===", flush=True)
    doc, skf, sk = doc_sketch()
    ln = sk.add_line((0, 0), (10, 0))
    arc = sk.add_arc((10, 0), (14, 4), (18, 0))

    k_before = _kink(ln, arc, "p0")
    print(f"  Kink before constraints: {k_before:.6f}", flush=True)

    add_constraint(
        sk,
        SketchConstraint(
            id=-1,
            kind=ConstraintKind.COINCIDENT,
            e0=ln.id, h0="p1",
            e1=arc.id, h1="p0",
        ),
    )
    add_constraint(
        sk,
        SketchConstraint(
            id=-1,
            kind=ConstraintKind.TANGENT,
            e0=ln.id, e1=arc.id, h1="p0",
        ),
    )
    k_after = _kink(ln, arc, "p0")
    print(f"  Kink after tangent applied: {k_after:.6f}", flush=True)
    check("tangent makes kink zero", k_after < 1e-3, f"kink={k_after:.6f}")

    ctrl = SketchController(sk)

    # Drag arc free end
    p1_b = arc.p1()
    ctrl.drag = DragState(arc.id, "p1", HandleKind.ENDPOINT, p1_b)
    ctrl._apply_drag((16, 9))
    ctrl.drag = None
    k_drag_arc = _kink(ln, arc, "p0")
    d1 = np.hypot(arc.p1()[0] - p1_b[0], arc.p1()[1] - p1_b[1])
    print(
        f"  After drag arc p1: Δ={d1:.4f}, kink={k_drag_arc:.6f}", flush=True
    )
    check("arc p1 drag moves point", d1 > 0.5, f"Δ={d1:.4f}")
    check(
        "tangent survives arc drag",
        k_drag_arc < 0.05,
        f"kink={k_drag_arc:.6f}",
    )

    # Drag line p1 (the shared join)
    lp1_b = ln.p1
    ctrl.drag = DragState(ln.id, "p1", HandleKind.ENDPOINT, lp1_b)
    ctrl._apply_drag((8, 3))
    ctrl.drag = None
    k_drag_ln = _kink(ln, arc, "p0")
    d2 = np.hypot(ln.p1[0] - lp1_b[0], ln.p1[1] - lp1_b[1])
    print(
        f"  After drag line p1: Δ={d2:.4f}, kink={k_drag_ln:.6f}", flush=True
    )
    check("line p1 drag moves point", d2 > 0.5, f"Δ={d2:.4f}")
    check(
        "tangent survives line drag",
        k_drag_ln < 0.1,
        f"kink={k_drag_ln:.6f}",
    )

    # ==================================================================
    # 5.  Auto-tangent: arc started right at a line end
    # ==================================================================
    print("\n=== 5. Auto-Tangent (arc placed on line end) ===", flush=True)
    doc, skf, sk = doc_sketch()
    ln = sk.add_line((0, 0), (10, 0))
    ctrl = SketchController(sk)
    ctrl.set_tool(SketchTool.ARC)
    ctrl.on_press((10, 0))  # start at line p1
    ctrl.on_press((14, 4))
    msg = ctrl.on_press((18, 0))
    check("arc created via tool", msg == "Arc")
    arc = [e for e in sk.entities if isinstance(e, ArcEntity)][0]

    coin = [c for c in sk.constraints if c.kind is ConstraintKind.COINCIDENT]
    tang = [c for c in sk.constraints if c.kind is ConstraintKind.TANGENT]
    check("coincident auto-added", len(coin) >= 1, f"count={len(coin)}")
    check("tangent auto-added", len(tang) >= 1, f"count={len(tang)}")

    k_auto = _kink(ln, arc, "p0")
    print(f"  Auto-tangent kink: {k_auto:.6f}", flush=True)
    check("auto-tangent kink is zero", k_auto < 0.05, f"kink={k_auto:.6f}")

    # Drag arc free end — tangent should survive
    p1_b = arc.p1()
    ctrl.tool = SketchTool.SELECT
    ctrl.drag = DragState(arc.id, "p1", HandleKind.ENDPOINT, p1_b)
    ctrl._apply_drag((20, 5))
    ctrl.drag = None
    k_auto2 = _kink(ln, arc, "p0")
    d_auto = np.hypot(arc.p1()[0] - p1_b[0], arc.p1()[1] - p1_b[1])
    print(
        f"  After drag: Δ={d_auto:.4f}, kink={k_auto2:.6f}", flush=True
    )
    check(
        "auto-tangent survives drag",
        k_auto2 < 0.05,
        f"kink={k_auto2:.6f}",
    )

    # ==================================================================
    # 6.  Closed line+arc profile → extrude to watertight solid
    # ==================================================================
    print("\n=== 6. Closed Line+Arc Profile → Extrude ===", flush=True)
    doc, skf, sk = doc_sketch()
    # U-shape: three walls + arc bottom
    sk.add_line((0, 0), (0, 10))
    sk.add_line((0, 10), (20, 10))
    sk.add_line((20, 10), (20, 0))
    sk.add_arc((20, 0), (10, -6), (0, 0))

    loops = find_closed_line_loops(sk)
    check("one closed loop found", len(loops) == 1, f"count={len(loops)}")
    area = loops[0].area() if loops else 0
    check("loop area > 100", area > 100, f"area={area:.2f}")
    print(f"  Profile: {len(loops[0].line_ids)} edges, area={area:.2f}", flush=True)

    ex = doc.create_extrude(skf.id, 5.0)
    mesh = doc.evaluate_feature(ex.id)
    check("extrude produced a mesh", mesh is not None)
    if mesh is not None:
        vol = mesh.volume()
        wt = mesh.is_watertight()
        nv = mesh.vertices.shape[0]
        nf = mesh.faces.shape[0]
        print(f"  Mesh: {nv} verts, {nf} faces", flush=True)
        print(f"  Volume: {vol:.2f}", flush=True)
        print(f"  Watertight: {wt}", flush=True)
        check("solid is watertight (no holes)", wt)
        check("volume > 500", vol > 500, f"vol={vol:.2f}")

    # ==================================================================
    # 7.  Save / reload round-trip
    # ==================================================================
    print("\n=== 7. Arc Save & Reload (.gcad) ===", flush=True)
    doc, skf, sk = doc_sketch()
    arc = sk.add_arc((1, 1), (3, 4), (5, 1))
    r_orig = arc.radius
    a0_orig, a1_orig, ccw_orig = arc.a0, arc.a1, arc.ccw
    center_orig = arc.center
    doc.apply_sketch_dimension(skf.id, arc.id, "radius", r_orig)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "arc_verify.gcad"
        save_document(doc, path)
        loaded = load_document(path)

    sk_l = next(
        f for f in loaded.features if f.type is FeatureType.SKETCH
    ).sketch
    arcs_l = [e for e in sk_l.entities if isinstance(e, ArcEntity)]
    check("arc survives save/load", len(arcs_l) == 1)
    if arcs_l:
        a = arcs_l[0]
        check(
            "radius preserved",
            abs(a.radius - r_orig) < 1e-6,
            f"{a.radius:.6f} vs {r_orig:.6f}",
        )
        check("a0 preserved", abs(a.a0 - a0_orig) < 1e-6)
        check("a1 preserved", abs(a.a1 - a1_orig) < 1e-6)
        check("ccw preserved", a.ccw == ccw_orig)
        check(
            "center preserved",
            abs(a.center[0] - center_orig[0]) < 1e-6
            and abs(a.center[1] - center_orig[1]) < 1e-6,
        )
        dims = [d for d in sk_l.dimensions if d.role == "radius"]
        check("radius dimension preserved", len(dims) >= 1)

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * 60, flush=True)
    if errors:
        print(f"FAILED {len(errors)} check(s):", flush=True)
        for e in errors:
            print(f"  ✗ {e}", flush=True)
        return FAIL
    print("ALL CHECKS PASSED", flush=True)
    return PASS


if __name__ == "__main__":
    raise SystemExit(main())
