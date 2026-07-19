#!/usr/bin/env python3
"""Verify driving dimensions via the app UI path (SketchController + Document).

Drives the same entry points the viewport mouse handlers use:
  - draw: SketchController draw tools
  - dimension: Document.apply_sketch_dimension (dialog OK path)
  - drag: SELECT handle press → _apply_drag → release
Reports before/typed/after + endpoint coords before/after drag.
Writes a screenshot of the dimensioned sketch.
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


def _drag(ctrl, eid, handle, to_uv):
    from app.sketch_mode import DragState, SketchTool
    from cadcore.sketch import HandleKind, CircleEntity, LineEntity

    ent = ctrl.sketch.find_entity(eid)
    if handle == "rim":
        start = ent.rim_point()
        kind = HandleKind.RIM
    elif handle == "center":
        start = ent.center
        kind = HandleKind.CENTER
    else:
        start = getattr(ent, handle)
        kind = HandleKind.ENDPOINT
    ctrl.tool = SketchTool.SELECT
    ctrl.drag = DragState(eid, handle, kind, start)
    ctrl._apply_drag(to_uv)
    ctrl.drag = None
    return start


def main() -> int:
    print("[driving_dims_verify] start", flush=True)
    from app.sketch_mode import SketchController, SketchTool
    from cadcore.document import Document, FeatureType
    from cadcore.sketch import (
        line_angle_degrees_oriented,
        line_length,
        measure_dimension_value,
    )
    from cadcore.project_io import save_document, load_document
    import tempfile

    doc = Document()
    doc.seed_reference_planes()
    front = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(front.id)
    sk = skf.sketch
    ctrl = SketchController(sk)

    # --- Draw via drawing tools (UI path) ---
    ctrl.set_tool(SketchTool.LINE)
    ctrl.on_press((0.0, 0.0))
    ctrl.on_press((12.0, 0.0))  # line A
    ctrl.set_tool(SketchTool.LINE)
    ctrl.on_press((0.0, 0.0))
    ctrl.on_press((6.0, 6.0))  # line B sharing corner
    ctrl.set_tool(SketchTool.LINE)
    ctrl.on_press((25.0, 0.0))
    ctrl.on_press((35.0, 2.0))  # line C disjoint
    ctrl.set_tool(SketchTool.LINE)
    ctrl.on_press((25.0, 8.0))
    ctrl.on_press((32.0, 12.0))  # line D disjoint
    ctrl.set_tool(SketchTool.CIRCLE)
    ctrl.on_press((50.0, 10.0))
    ctrl.on_press((54.0, 10.0))  # circle r≈4

    lines = [e for e in sk.entities if e.__class__.__name__ == "LineEntity"]
    circles = [e for e in sk.entities if e.__class__.__name__ == "CircleEntity"]
    assert len(lines) >= 4 and len(circles) >= 1
    a, b, c, d = lines[0], lines[1], lines[2], lines[3]
    circ = circles[0]

    report = []

    # LENGTH
    L0 = line_length(a)
    typed = 45.0
    doc.apply_sketch_dimension(skf.id, a.id, "length", typed)
    L1 = line_length(a)
    p1_before = tuple(a.p1)
    p0_before = tuple(a.p0)
    _drag(ctrl, a.id, "p1", (20.0, 15.0))
    p1_after = tuple(a.p1)
    p0_after = tuple(a.p0)
    L2 = line_length(a)
    report.append(
        ("LENGTH", L0, typed, L1, p0_before, p1_before, p0_after, p1_after, L2)
    )
    assert abs(L1 - typed) < 1e-3 and abs(L2 - typed) < 1e-3
    assert p1_before != p1_after
    print(
        f"LENGTH before={L0:.4f} typed={typed} after_apply={L1:.4f} "
        f"p1 {p1_before}→{p1_after} after_drag={L2:.4f}",
        flush=True,
    )

    # ANGLE shared corner
    ang0 = line_angle_degrees_oriented(a, b)
    typed_a = 30.0
    doc.apply_sketch_dimension(skf.id, a.id, "angle", typed_a, entity_b_id=b.id)
    ang1 = line_angle_degrees_oriented(a, b)
    def acute(x):
        return min(x, 180 - x)
    bp1_before = tuple(b.p1)
    _drag(ctrl, b.id, "p1", (10.0, 18.0))
    bp1_after = tuple(b.p1)
    ang2 = line_angle_degrees_oriented(a, b)
    print(
        f"ANGLE(shared) before={acute(ang0):.4f} typed={typed_a} "
        f"after_apply={acute(ang1):.4f} p1 {bp1_before}→{bp1_after} "
        f"after_drag={acute(ang2):.4f}",
        flush=True,
    )
    assert abs(acute(ang1) - typed_a) < 0.15
    assert abs(acute(ang2) - typed_a) < 0.2
    assert bp1_before != bp1_after

    # ANGLE disjoint
    ang0 = line_angle_degrees_oriented(c, d)
    typed_a2 = 40.0
    doc.apply_sketch_dimension(skf.id, c.id, "angle", typed_a2, entity_b_id=d.id)
    ang1 = line_angle_degrees_oriented(c, d)
    dp1_before = tuple(d.p1)
    _drag(ctrl, d.id, "p1", (40.0, 20.0))
    dp1_after = tuple(d.p1)
    ang2 = line_angle_degrees_oriented(c, d)
    print(
        f"ANGLE(disjoint) before={acute(ang0):.4f} typed={typed_a2} "
        f"after_apply={acute(ang1):.4f} p1 {dp1_before}→{dp1_after} "
        f"after_drag={acute(ang2):.4f}",
        flush=True,
    )
    assert abs(acute(ang1) - typed_a2) < 0.2
    assert abs(acute(ang2) - typed_a2) < 0.25
    assert dp1_before != dp1_after

    # DIAMETER — drag the center so geometry moves while diameter holds
    d0 = circ.radius * 2
    typed_d = 18.0
    doc.apply_sketch_dimension(skf.id, circ.id, "diameter", typed_d)
    d1 = circ.radius * 2
    center_before = tuple(circ.center)
    _drag(ctrl, circ.id, "center", (60.0, 25.0))
    center_after = tuple(circ.center)
    d2 = circ.radius * 2
    print(
        f"DIAMETER before={d0:.4f} typed={typed_d} after_apply={d1:.4f} "
        f"center {center_before}→{center_after} after_drag={d2:.4f}",
        flush=True,
    )
    assert abs(d1 - typed_d) < 1e-3 and abs(d2 - typed_d) < 1e-3
    assert center_before != center_after

    # Conflict
    from cadcore.constraints import ConstraintKind, SketchConstraint, add_constraint
    from cadcore.sketch import snapshot_sketch_contents

    add_constraint(sk, SketchConstraint(-1, ConstraintKind.FIX, e0=a.id, h0="p0"))
    add_constraint(sk, SketchConstraint(-1, ConstraintKind.FIX, e0=a.id, h0="p1"))
    snap = snapshot_sketch_contents(sk)
    try:
        doc.apply_sketch_dimension(skf.id, a.id, "length", 200.0)
        print("FAIL: conflict should raise", flush=True)
        return 2
    except ValueError as exc:
        print(f"CONFLICT message: {exc}", flush=True)
    # fixed ends unchanged
    assert a.p0[0] == snap["entities"][0]["p0"][0]

    # Save / reopen / drag
    path = OUT / "driving_dims.gcad"
    save_document(doc, path)
    loaded = load_document(path)
    skf2 = next(f for f in loaded.features if f.type is FeatureType.SKETCH)
    sk2 = skf2.sketch
    a2 = sk2.find_entity(a.id)
    ctrl2 = SketchController(sk2)
    _drag(ctrl2, a2.id, "p1", (5.0, 9.0))
    print(f"REOPEN length after drag: {line_length(a2):.4f} (expect 45)", flush=True)
    assert abs(line_length(a2) - 45.0) < 1e-2

    # Screenshot (matplotlib — readable dim labels)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle as MplCircle

        fig, ax = plt.subplots(figsize=(10, 6))
        for e in sk.entities:
            if e.__class__.__name__ == "LineEntity":
                ax.plot([e.p0[0], e.p1[0]], [e.p0[1], e.p1[1]], "b-", lw=2)
            elif e.__class__.__name__ == "CircleEntity":
                ax.add_patch(
                    MplCircle(e.center, e.radius, fill=False, color="b", lw=2)
                )
        for dim in sk.dimensions:
            ent = sk.find_entity(dim.entity_id)
            if ent is None:
                continue
            from cadcore.sketch import dimension_anchor_uv

            ent_b = (
                sk.find_entity(dim.entity_b_id) if dim.role == "angle" else None
            )
            uv = dimension_anchor_uv(ent, dim.role, ent_b=ent_b)
            if dim.role == "angle":
                label = f"{dim.value_mm:g}°"
            elif dim.role == "diameter":
                label = f"⌀{dim.value_mm:g}"
            else:
                label = f"{dim.value_mm:g}"
            ax.annotate(
                label,
                xy=uv,
                fontsize=11,
                color="darkred",
                bbox=dict(boxstyle="round,pad=0.2", fc="wheat", alpha=0.9),
            )
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title("Driving dimensions (length / angle / diameter)")
        shot = OUT / "driving_dims_sketch.png"
        fig.savefig(shot, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"SHOT {shot}", flush=True)
    except Exception as exc:
        print(f"SHOT_FAIL {exc}", flush=True)
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
