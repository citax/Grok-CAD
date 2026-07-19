#!/usr/bin/env python3
"""Verify driving dimensions via Document + SketchController geometry path.

Honesty notes (read before trusting a PASS):
  * This does **not** start MainWindow or send Qt mouse/toolbar events. In this
    environment MainWindow+VTK fails to open a window (X11/BadWindow). What is
    tested is the same geometry stack the viewport uses after a dialog OK
    (Document.apply_sketch_dimension) and after a handle move
    (SketchController._apply_drag → solve_sketch).
  * The PNG is a matplotlib plot of the sketch data for human inspection of
    the corner — it is **not** a capture of the CAD window.
  * Every numeric check below can fail if geometry is wrong.

Exit 0 only if all assertions pass.
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["GROK_CAD_UNATTENDED"] = "1"

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
OUT = Path(__file__).resolve().parent / "_ui_shots"
OUT.mkdir(parents=True, exist_ok=True)


def _acute(a, b):
    from cadcore.sketch import line_angle_degrees_oriented

    ang = line_angle_degrees_oriented(a, b)
    return min(ang, 180.0 - ang)


def _corner_dist(a, b) -> float:
    import numpy as np

    best = 1e9
    for pa in (a.p0, a.p1):
        for pb in (b.p0, b.p1):
            best = min(best, float(np.hypot(pa[0] - pb[0], pa[1] - pb[1])))
    return best


def _handle_drag(ctrl, eid, handle, to_uv):
    """Call SketchController._apply_drag (not Qt events)."""
    from app.sketch_mode import DragState, SketchTool
    from cadcore.sketch import HandleKind

    ent = ctrl.sketch.find_entity(eid)
    if handle == "center":
        start, kind = ent.center, HandleKind.CENTER
    elif handle == "rim":
        start, kind = ent.rim_point(), HandleKind.RIM
    else:
        start, kind = getattr(ent, handle), HandleKind.ENDPOINT
    ctrl.tool = SketchTool.SELECT
    ctrl.drag = DragState(eid, handle, kind, start)
    ctrl._apply_drag(to_uv)
    ctrl.drag = None


def main() -> int:
    print("[driving_dims_verify] start", flush=True)
    print(
        "[driving_dims_verify] NOTE: not driving MainWindow/Qt mouse; "
        "testing Document.apply_sketch_dimension + SketchController._apply_drag",
        flush=True,
    )
    from app.sketch_mode import SketchController
    from cadcore.document import Document, FeatureType
    from cadcore.sketch import line_length
    from cadcore.project_io import save_document, load_document

    doc = Document()
    doc.seed_reference_planes()
    front = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(front.id)
    sk = skf.sketch
    # Geometry as drawn (two lines from one corner, etc.)
    a = sk.add_line((0.0, 0.0), (12.0, 0.0))
    b = sk.add_line((0.0, 0.0), (6.0, 6.0))
    c = sk.add_line((25.0, 0.0), (35.0, 2.0))
    d = sk.add_line((25.0, 8.0), (32.0, 12.0))
    circ = sk.add_circle((50.0, 10.0), 4.0)
    other = sk.add_line((0.0, -10.0), (5.0, -10.0))
    other_snap = (tuple(other.p0), tuple(other.p1))
    ctrl = SketchController(sk)

    # --- LENGTH ---
    L0 = line_length(a)
    typed = 45.0
    doc.apply_sketch_dimension(skf.id, a.id, "length", typed)
    L1 = line_length(a)
    assert abs(L1 - typed) < 1e-6, f"length after apply {L1} != {typed}"
    assert (tuple(other.p0), tuple(other.p1)) == other_snap, "length dim moved unrelated line"
    p1_before = (float(a.p1[0]), float(a.p1[1]))
    _handle_drag(ctrl, a.id, "p1", (20.0, 15.0))
    p1_after = (float(a.p1[0]), float(a.p1[1]))
    L2 = line_length(a)
    assert p1_before != p1_after, "length drag did not move endpoint"
    assert abs(L2 - typed) < 1e-3, f"length after drag {L2}"
    print(
        f"LENGTH before={L0:.4f} typed={typed} after_apply={L1:.4f} "
        f"p1 {p1_before}→{p1_after} after_drag={L2:.4f}",
        flush=True,
    )

    # --- ANGLE shared corner ---
    d_before = _corner_dist(a, b)
    assert d_before < 1e-9, f"setup: corner not closed ({d_before})"
    ang0 = _acute(a, b)
    typed_a = 30.0
    doc.apply_sketch_dimension(skf.id, a.id, "angle", typed_a, entity_b_id=b.id)
    d_after_apply = _corner_dist(a, b)
    ang1 = _acute(a, b)
    assert d_after_apply < 1e-6, f"ANGLE opened corner after apply: dist={d_after_apply}"
    assert abs(ang1 - typed_a) < 0.05, f"angle after apply {ang1}"
    bp1_before = (float(b.p1[0]), float(b.p1[1]))
    _handle_drag(ctrl, b.id, "p1", (10.0, 18.0))
    bp1_after = (float(b.p1[0]), float(b.p1[1]))
    d_after_drag = _corner_dist(a, b)
    ang2 = _acute(a, b)
    assert bp1_before != bp1_after, "angle drag did not move free end"
    assert d_after_drag < 1e-5, f"ANGLE opened corner after drag: dist={d_after_drag}"
    assert abs(ang2 - typed_a) < 0.15, f"angle after drag {ang2}"
    print(
        f"ANGLE(shared) before={ang0:.4f} typed={typed_a} after_apply={ang1:.4f} "
        f"corner dist before/after_apply/after_drag="
        f"{d_before:.3e}/{d_after_apply:.3e}/{d_after_drag:.3e} "
        f"p1 {bp1_before}→{bp1_after} after_drag={ang2:.4f}",
        flush=True,
    )

    # --- ANGLE disjoint ---
    ang0 = _acute(c, d)
    typed_a2 = 40.0
    doc.apply_sketch_dimension(skf.id, c.id, "angle", typed_a2, entity_b_id=d.id)
    ang1 = _acute(c, d)
    dp1_before = (float(d.p1[0]), float(d.p1[1]))
    _handle_drag(ctrl, d.id, "p1", (40.0, 20.0))
    dp1_after = (float(d.p1[0]), float(d.p1[1]))
    ang2 = _acute(c, d)
    assert dp1_before != dp1_after
    assert abs(ang1 - typed_a2) < 0.15 and abs(ang2 - typed_a2) < 0.2
    print(
        f"ANGLE(disjoint) before={ang0:.4f} typed={typed_a2} after_apply={ang1:.4f} "
        f"p1 {dp1_before}→{dp1_after} after_drag={ang2:.4f}",
        flush=True,
    )

    # --- DIAMETER ---
    d0 = circ.radius * 2
    typed_d = 18.0
    line_before = (tuple(other.p0), tuple(other.p1))
    doc.apply_sketch_dimension(skf.id, circ.id, "diameter", typed_d)
    d1 = circ.radius * 2
    assert abs(d1 - typed_d) < 1e-6
    assert (tuple(other.p0), tuple(other.p1)) == line_before
    center_before = (float(circ.center[0]), float(circ.center[1]))
    _handle_drag(ctrl, circ.id, "center", (60.0, 25.0))
    center_after = (float(circ.center[0]), float(circ.center[1]))
    d2 = circ.radius * 2
    assert center_before != center_after
    assert abs(d2 - typed_d) < 1e-3
    print(
        f"DIAMETER before={d0:.4f} typed={typed_d} after_apply={d1:.4f} "
        f"center {center_before}→{center_after} after_drag={d2:.4f}",
        flush=True,
    )

    # Conflict
    from cadcore.constraints import ConstraintKind, SketchConstraint, add_constraint

    add_constraint(sk, SketchConstraint(-1, ConstraintKind.FIX, e0=a.id, h0="p0"))
    add_constraint(sk, SketchConstraint(-1, ConstraintKind.FIX, e0=a.id, h0="p1"))
    try:
        doc.apply_sketch_dimension(skf.id, a.id, "length", 200.0)
        print("FAIL: expected conflict", flush=True)
        return 2
    except ValueError as exc:
        print(f"CONFLICT message: {exc}", flush=True)

    # Save / reopen
    path = OUT / "driving_dims.gcad"
    save_document(doc, path)
    loaded = load_document(path)
    skf2 = next(f for f in loaded.features if f.type is FeatureType.SKETCH)
    a2 = skf2.sketch.find_entity(a.id)
    b2 = skf2.sketch.find_entity(b.id)
    ctrl2 = SketchController(skf2.sketch)
    _handle_drag(ctrl2, b2.id, "p1", (5.0, 9.0))
    assert _corner_dist(a2, b2) < 1e-5
    assert abs(_acute(a2, b2) - 30.0) < 0.2
    print(
        f"REOPEN angle after drag={_acute(a2, b2):.4f} corner={_corner_dist(a2, b2):.3e}",
        flush=True,
    )

    # Matplotlib figure of sketch data (NOT a window screenshot)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle as MplCircle
        from cadcore.sketch import dimension_anchor_uv

        fig, ax = plt.subplots(figsize=(10, 6))
        for e in sk.entities:
            if e.__class__.__name__ == "LineEntity":
                ax.plot([e.p0[0], e.p1[0]], [e.p0[1], e.p1[1]], "b-o", lw=2, ms=4)
            elif e.__class__.__name__ == "CircleEntity":
                ax.add_patch(
                    MplCircle(e.center, e.radius, fill=False, color="b", lw=2)
                )
        for dim in sk.dimensions:
            ent = sk.find_entity(dim.entity_id)
            if ent is None:
                continue
            ent_b = sk.find_entity(dim.entity_b_id) if dim.role == "angle" else None
            uv = dimension_anchor_uv(ent, dim.role, ent_b=ent_b)
            label = (
                f"{dim.value_mm:g}°"
                if dim.role == "angle"
                else (
                    f"⌀{dim.value_mm:g}"
                    if dim.role == "diameter"
                    else f"{dim.value_mm:g}"
                )
            )
            ax.annotate(
                label,
                xy=uv,
                fontsize=11,
                color="darkred",
                bbox=dict(boxstyle="round,pad=0.2", fc="wheat", alpha=0.9),
            )
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(
            "Sketch data plot (not CAD window) — corner at origin must stay closed"
        )
        shot = OUT / "driving_dims_sketch.png"
        fig.savefig(shot, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"DATA_PLOT {shot} (not a program screenshot)", flush=True)
    except Exception as exc:
        print(f"PLOT_FAIL {exc}", flush=True)
        return 3

    print("[driving_dims_verify] PASS", flush=True)
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception:
        traceback.print_exc()
        code = 99
    print(f"[driving_dims_verify] exit_code={code}", flush=True)
    raise SystemExit(code)
