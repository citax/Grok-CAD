"""Sketch editing ops: trim, extend, offset, convert, DOF status."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from cadcore.sketch import (
    ArcEntity,
    CircleEntity,
    LineEntity,
    RectEntity,
    Sketch,
    SketchEntity,
    Vec2,
    arc_from_three_points,
)


def is_construction(ent: SketchEntity) -> bool:
    return bool(getattr(ent, "construction", False))


def set_construction(ent: SketchEntity, on: bool = True) -> None:
    ent.construction = bool(on)


def toggle_construction(ents: Sequence[SketchEntity]) -> int:
    n = 0
    for e in ents:
        e.construction = not bool(getattr(e, "construction", False))
        n += 1
    return n


def offset_line(ln: LineEntity, distance: float) -> LineEntity:
    """Return a new line parallel to ``ln`` offset by signed ``distance`` (left of p0→p1)."""
    d = np.array([ln.p1[0] - ln.p0[0], ln.p1[1] - ln.p0[1]], dtype=np.float64)
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        raise ValueError("cannot offset zero-length line")
    # left normal
    nx, ny = -d[1] / n, d[0] / n
    du, dv = nx * float(distance), ny * float(distance)
    return LineEntity(
        id=-1,
        kind=ln.kind,
        p0=(ln.p0[0] + du, ln.p0[1] + dv),
        p1=(ln.p1[0] + du, ln.p1[1] + dv),
        construction=bool(getattr(ln, "construction", False)),
    )


def offset_circle(c: CircleEntity, distance: float) -> CircleEntity:
    r = float(c.radius) + float(distance)
    if r <= 1e-9:
        raise ValueError("offset would make radius non-positive")
    return CircleEntity(
        id=-1,
        kind=c.kind,
        center=c.center,
        radius=r,
        construction=bool(getattr(c, "construction", False)),
    )


def extend_line_to_point(ln: LineEntity, uv: Vec2, *, free_end: str = "p1") -> None:
    """Extend/shrink the free end along the line direction toward the projection of uv."""
    p0 = np.array(ln.p0, dtype=np.float64)
    p1 = np.array(ln.p1, dtype=np.float64)
    d = p1 - p0
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        return
    dhat = d / n
    pt = np.array([float(uv[0]), float(uv[1])], dtype=np.float64)
    if free_end == "p0":
        # project onto ray from p1 along -dhat
        t = float(np.dot(pt - p1, -dhat))
        ln.p0 = (float(p1[0] - dhat[0] * t), float(p1[1] - dhat[1] * t))
    else:
        t = float(np.dot(pt - p0, dhat))
        ln.p1 = (float(p0[0] + dhat[0] * t), float(p0[1] + dhat[1] * t))


def trim_line_at(ln: LineEntity, uv: Vec2, keep_side: str = "near") -> None:
    """Trim line at the closest point to uv, keeping the longer remaining part by default."""
    p0 = np.array(ln.p0, dtype=np.float64)
    p1 = np.array(ln.p1, dtype=np.float64)
    d = p1 - p0
    n2 = float(np.dot(d, d))
    if n2 < 1e-24:
        return
    pt = np.array([float(uv[0]), float(uv[1])], dtype=np.float64)
    t = float(np.clip(np.dot(pt - p0, d) / n2, 0.0, 1.0))
    cut = p0 + t * d
    # Keep the side with more length unless keep_side says otherwise
    if t < 0.5:
        # closer to p0 — drop p0 side, keep cut→p1
        ln.p0 = (float(cut[0]), float(cut[1]))
    else:
        ln.p1 = (float(cut[0]), float(cut[1]))


def entity_dof_status(sk: Sketch, ent: SketchEntity) -> str:
    """Heuristic DOF status: under / well / over.

    Not a full DOF solver — good enough for SolidWorks-like black/blue/red cues.
    """
    from cadcore.constraints import constraint_residual, dimension_residual

    # Count constraints/dims that touch this entity
    n_c = 0
    max_r = 0.0
    for c in sk.constraints or []:
        if int(c.e0) == ent.id or int(getattr(c, "e1", -1)) == ent.id:
            n_c += 1
            try:
                max_r = max(max_r, float(constraint_residual(sk, c)))
            except Exception:
                pass
    n_d = 0
    for d in sk.dimensions or []:
        if int(d.entity_id) == ent.id or int(getattr(d, "entity_b_id", -1)) == ent.id:
            n_d += 1
            try:
                max_r = max(max_r, float(dimension_residual(sk, d)))
            except Exception:
                pass
    if max_r > 0.05:
        return "over"  # conflict residual
    # crude expected constraint count by kind
    if isinstance(ent, LineEntity):
        need = 3  # position+orientation underconstrained freely
        have = n_c + n_d
        if have >= 4:
            return "well"
        if have == 0:
            return "under"
        return "under" if have < 3 else "well"
    if isinstance(ent, (CircleEntity, ArcEntity)):
        have = n_c + n_d
        if have >= 3:
            return "well"
        return "under" if have < 2 else "well"
    if isinstance(ent, RectEntity):
        have = n_c + n_d
        return "well" if have >= 2 else "under"
    return "under"


def convert_face_edges_to_sketch(
    sk: Sketch,
    edges_uv: Sequence[Tuple[Vec2, Vec2]],
    *,
    construction: bool = True,
) -> List[SketchEntity]:
    """Add line entities from face-edge UV pairs (convert entities)."""
    out: List[SketchEntity] = []
    for p0, p1 in edges_uv:
        ln = sk.add_line(p0, p1)
        ln.construction = bool(construction)
        out.append(ln)
    return out
