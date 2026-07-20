"""Persistent geometric sketch constraints + iterative solver.

Relationships are promises the sketch keeps after drags. Partially constrained
sketches are valid; conflicting adds are refused with the sketch unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from cadcore.sketch import (
    ArcEntity,
    CircleEntity,
    LineEntity,
    RectEntity,
    Sketch,
    SketchEntity,
    Vec2,
    apply_entity_snapshot,
    line_length,
    snapshot_entity,
)

PointRef = Tuple[int, str]  # (entity_id, handle_name)
TOL = 1e-6
SOLVE_ITERS = 48


class ConstraintKind(Enum):
    COINCIDENT = auto()  # two points stay together
    HORIZONTAL = auto()  # line stays horizontal
    VERTICAL = auto()  # line stays vertical
    PARALLEL = auto()  # two lines stay parallel
    PERPENDICULAR = auto()  # two lines stay perpendicular
    EQUAL = auto()  # two lines keep equal length
    FIX = auto()  # point locked in place
    TANGENT = auto()  # line and arc meet smoothly (h0 = arc end "p0"/"p1")
    MIDPOINT = auto()  # point is midpoint of line (e0=line, e1=point-ent, h1)
    CONCENTRIC = auto()  # two circles/arcs share center
    COLLINEAR = auto()  # two lines collinear
    SYMMETRIC = auto()  # e0=mirror line; e1=subject; h0/h1 handles on e1
    EQUAL_RADIUS = auto()  # two arcs/circles equal radius


@dataclass
class SketchConstraint:
    id: int
    kind: ConstraintKind
    e0: int = -1
    h0: str = ""  # point handle for coincident/fix; unused for line-only
    e1: int = -1
    h1: str = ""
    # FIX: locked coordinates in sketch UV
    u: float = 0.0
    v: float = 0.0

    def label(self) -> str:
        return {
            ConstraintKind.COINCIDENT: "Coincident",
            ConstraintKind.HORIZONTAL: "Horizontal",
            ConstraintKind.VERTICAL: "Vertical",
            ConstraintKind.PARALLEL: "Parallel",
            ConstraintKind.PERPENDICULAR: "Perpendicular",
            ConstraintKind.EQUAL: "Equal",
            ConstraintKind.FIX: "Fix",
            ConstraintKind.TANGENT: "Tangent",
        }[self.kind]


def snapshot_constraint(c: SketchConstraint) -> dict:
    return {
        "id": int(c.id),
        "kind": c.kind.name,
        "e0": int(c.e0),
        "h0": str(c.h0),
        "e1": int(c.e1),
        "h1": str(c.h1),
        "u": float(c.u),
        "v": float(c.v),
    }


def restore_constraint(data: dict) -> SketchConstraint:
    kind_raw = data.get("kind", "HORIZONTAL")
    try:
        kind = ConstraintKind[str(kind_raw)]
    except KeyError:
        kind = ConstraintKind.HORIZONTAL
    return SketchConstraint(
        id=int(data.get("id", 0)),
        kind=kind,
        e0=int(data.get("e0", -1)),
        h0=str(data.get("h0", "")),
        e1=int(data.get("e1", -1)),
        h1=str(data.get("h1", "")),
        u=float(data.get("u", 0.0)),
        v=float(data.get("v", 0.0)),
    )


# ---------------------------------------------------------------------------
# Geometry access
# ---------------------------------------------------------------------------


def _get_point(sk: Sketch, eid: int, handle: str) -> Optional[np.ndarray]:
    ent = sk.find_entity(eid)
    if ent is None:
        return None
    if isinstance(ent, LineEntity):
        if handle == "p0":
            return np.array(ent.p0, dtype=np.float64)
        if handle == "p1":
            return np.array(ent.p1, dtype=np.float64)
        if handle == "mid":
            return np.array(ent.midpoint(), dtype=np.float64)
    if isinstance(ent, CircleEntity):
        if handle == "center":
            return np.array(ent.center, dtype=np.float64)
        if handle == "rim":
            return np.array(ent.rim_point(), dtype=np.float64)
    if isinstance(ent, ArcEntity):
        if handle == "p0":
            return np.array(ent.p0(), dtype=np.float64)
        if handle == "p1":
            return np.array(ent.p1(), dtype=np.float64)
        if handle == "mid":
            return np.array(ent.mid_uv(), dtype=np.float64)
        if handle == "center":
            return np.array(ent.center, dtype=np.float64)
    if isinstance(ent, RectEntity) and handle.startswith("c") and handle[1:].isdigit():
        corners = ent.corners()
        idx = int(handle[1])
        if 0 <= idx < 4:
            return np.array(corners[idx], dtype=np.float64)
    return None


def _set_point(sk: Sketch, eid: int, handle: str, uv: Sequence[float]) -> None:
    ent = sk.find_entity(eid)
    if ent is None:
        return
    p = (float(uv[0]), float(uv[1]))
    if isinstance(ent, LineEntity):
        if handle in ("p0", "p1"):
            ent.set_handle(handle, p)
        elif handle == "mid":
            ent.set_handle("mid", p)
    elif isinstance(ent, CircleEntity):
        if handle in ("center", "rim"):
            ent.set_handle(handle, p)
    elif isinstance(ent, ArcEntity):
        ent.set_handle(handle, p)
    elif isinstance(ent, RectEntity) and handle.startswith("c"):
        ent.set_handle(handle, p)


def _line_dir(sk: Sketch, eid: int) -> Optional[np.ndarray]:
    ent = sk.find_entity(eid)
    if not isinstance(ent, LineEntity):
        return None
    d = np.array([ent.p1[0] - ent.p0[0], ent.p1[1] - ent.p0[1]], dtype=np.float64)
    return d


def _fixed_points(sk: Sketch) -> Dict[PointRef, np.ndarray]:
    out: Dict[PointRef, np.ndarray] = {}
    for c in sk.constraints:
        if c.kind is ConstraintKind.FIX and c.e0 >= 0 and c.h0:
            out[(int(c.e0), str(c.h0))] = np.array([c.u, c.v], dtype=np.float64)
    return out


def _is_fixed(fixed: Dict[PointRef, np.ndarray], ref: PointRef) -> bool:
    return ref in fixed


# ---------------------------------------------------------------------------
# Residuals (for validation / testing)
# ---------------------------------------------------------------------------


def constraint_residual(sk: Sketch, c: SketchConstraint) -> float:
    """Scalar residual ≥ 0; ~0 means satisfied."""
    if c.kind is ConstraintKind.COINCIDENT:
        a = _get_point(sk, c.e0, c.h0)
        b = _get_point(sk, c.e1, c.h1)
        if a is None or b is None:
            return 1e9
        return float(np.linalg.norm(a - b))
    if c.kind is ConstraintKind.HORIZONTAL:
        d = _line_dir(sk, c.e0)
        if d is None:
            return 1e9
        return abs(float(d[1]))
    if c.kind is ConstraintKind.VERTICAL:
        d = _line_dir(sk, c.e0)
        if d is None:
            return 1e9
        return abs(float(d[0]))
    if c.kind is ConstraintKind.PARALLEL:
        d0 = _line_dir(sk, c.e0)
        d1 = _line_dir(sk, c.e1)
        if d0 is None or d1 is None:
            return 1e9
        # |cross| / (|d0||d1|) * scale — absolute cross normalized
        n0 = float(np.linalg.norm(d0))
        n1 = float(np.linalg.norm(d1))
        if n0 < 1e-12 or n1 < 1e-12:
            return 0.0
        return abs(float(d0[0] * d1[1] - d0[1] * d1[0])) / (n0 * n1) * max(n0, n1)
    if c.kind is ConstraintKind.PERPENDICULAR:
        d0 = _line_dir(sk, c.e0)
        d1 = _line_dir(sk, c.e1)
        if d0 is None or d1 is None:
            return 1e9
        n0 = float(np.linalg.norm(d0))
        n1 = float(np.linalg.norm(d1))
        if n0 < 1e-12 or n1 < 1e-12:
            return 0.0
        return abs(float(np.dot(d0, d1))) / (n0 * n1) * max(n0, n1)
    if c.kind is ConstraintKind.EQUAL:
        e0 = sk.find_entity(c.e0)
        e1 = sk.find_entity(c.e1)
        if not isinstance(e0, LineEntity) or not isinstance(e1, LineEntity):
            return 1e9
        return abs(line_length(e0) - line_length(e1))
    if c.kind is ConstraintKind.FIX:
        p = _get_point(sk, c.e0, c.h0)
        if p is None:
            return 1e9
        return float(np.hypot(p[0] - c.u, p[1] - c.v))
    if c.kind is ConstraintKind.TANGENT:
        return _tangent_residual(sk, c)
    if c.kind is ConstraintKind.MIDPOINT:
        ln = sk.find_entity(int(c.e0))
        if not isinstance(ln, LineEntity):
            return 1e9
        mid = np.array(ln.midpoint(), dtype=np.float64)
        p = _get_point(sk, c.e1, c.h1 or c.h0)
        if p is None:
            return 1e9
        return float(np.linalg.norm(p - mid))
    if c.kind is ConstraintKind.CONCENTRIC:
        a = sk.find_entity(int(c.e0))
        b = sk.find_entity(int(c.e1))
        if not isinstance(a, (CircleEntity, ArcEntity)) or not isinstance(
            b, (CircleEntity, ArcEntity)
        ):
            return 1e9
        ca = np.array(a.center, dtype=np.float64)
        cb = np.array(b.center, dtype=np.float64)
        return float(np.linalg.norm(ca - cb))
    if c.kind is ConstraintKind.COLLINEAR:
        d0 = _line_dir(sk, c.e0)
        d1 = _line_dir(sk, c.e1)
        e0 = sk.find_entity(c.e0)
        e1 = sk.find_entity(c.e1)
        if (
            d0 is None
            or d1 is None
            or not isinstance(e0, LineEntity)
            or not isinstance(e1, LineEntity)
        ):
            return 1e9
        n0 = float(np.linalg.norm(d0))
        n1 = float(np.linalg.norm(d1))
        if n0 < 1e-12 or n1 < 1e-12:
            return 0.0
        # Parallel residual + point-on-line residual
        par = abs(float(d0[0] * d1[1] - d0[1] * d1[0])) / (n0 * n1) * max(n0, n1)
        # distance of e1.p0 to line e0
        p = np.array(e1.p0, dtype=np.float64)
        a = np.array(e0.p0, dtype=np.float64)
        dhat = d0 / n0
        dist = abs(float((p - a)[0] * dhat[1] - (p - a)[1] * dhat[0]))
        return par + dist
    if c.kind is ConstraintKind.SYMMETRIC:
        mir = sk.find_entity(int(c.e0))
        sub = sk.find_entity(int(c.e1))
        if not isinstance(mir, LineEntity):
            return 1e9
        h0 = str(c.h0 or "p0")
        h1 = str(c.h1 or "p1")
        pa = _get_point(sk, c.e1, h0)
        pb = _get_point(sk, c.e1, h1)
        if pa is None or pb is None:
            return 1e9
        # Reflect pa over mir; distance to pb
        mid = _reflect_point_over_line(pa, mir)
        return float(np.linalg.norm(mid - pb))
    if c.kind is ConstraintKind.EQUAL_RADIUS:
        a = sk.find_entity(int(c.e0))
        b = sk.find_entity(int(c.e1))
        if not isinstance(a, (CircleEntity, ArcEntity)) or not isinstance(
            b, (CircleEntity, ArcEntity)
        ):
            return 1e9
        return abs(float(a.radius) - float(b.radius))
    return 1e9


def _reflect_point_over_line(p: np.ndarray, ln: LineEntity) -> np.ndarray:
    a = np.array(ln.p0, dtype=np.float64)
    b = np.array(ln.p1, dtype=np.float64)
    d = b - a
    n2 = float(np.dot(d, d))
    if n2 < 1e-24:
        return p.copy()
    t = float(np.dot(p - a, d) / n2)
    proj = a + t * d
    return 2.0 * proj - p


def _tangent_residual(sk: Sketch, c: SketchConstraint) -> float:
    """|sin| of angle between line direction and arc tangent at join (0 = smooth)."""
    ln = sk.find_entity(int(c.e0))
    arc = sk.find_entity(int(c.e1))
    if not isinstance(ln, LineEntity) or not isinstance(arc, ArcEntity):
        # allow reversed storage
        ln = sk.find_entity(int(c.e1))
        arc = sk.find_entity(int(c.e0))
        h = str(c.h0) if c.h0 else str(c.h1)
    else:
        h = str(c.h1) if c.h1 else str(c.h0)
    if not isinstance(ln, LineEntity) or not isinstance(arc, ArcEntity):
        return 1e9
    d = np.array([ln.p1[0] - ln.p0[0], ln.p1[1] - ln.p0[1]], dtype=np.float64)
    nd = float(np.linalg.norm(d))
    if nd < 1e-12:
        return 0.0
    d = d / nd
    if h == "p0":
        t = arc.tangent_at_start()
    else:
        t = arc.tangent_at_end()
    # Parallel (either sense): |cross| small
    return abs(float(d[0] * t[1] - d[1] * t[0]))


def dimension_residual(sk: Sketch, dim) -> float:
    """How far a driving dimension is from its stored value."""
    from cadcore.sketch import (
        CircleEntity,
        LineEntity,
        line_angle_degrees_oriented,
        line_length,
        measure_dimension_value,
    )

    role = str(dim.role)
    ent = sk.find_entity(int(dim.entity_id))
    if ent is None:
        return 1e9
    if role == "angle":
        ent_b = sk.find_entity(int(getattr(dim, "entity_b_id", -1)))
        if not isinstance(ent, LineEntity) or not isinstance(ent_b, LineEntity):
            return 1e9
        # Smallest difference considering 180° periodicity of undirected lines
        meas = line_angle_degrees_oriented(ent, ent_b)
        target = float(dim.value_mm) % 180.0
        # undirected: angle and 180-angle are same line pair — use min to either
        d1 = abs(meas - target)
        d2 = abs(meas - (180.0 - target)) if target not in (0.0, 90.0) else d1
        # Also meas vs target when both represent same
        return float(min(d1, abs((180.0 - meas) - target), d2))
    try:
        if role == "diameter" and isinstance(ent, CircleEntity):
            return abs(float(ent.radius) * 2.0 - float(dim.value_mm))
        if role == "length" and isinstance(ent, LineEntity):
            return abs(line_length(ent) - float(dim.value_mm))
        return abs(measure_dimension_value(ent, role) - float(dim.value_mm))
    except ValueError:
        return 1e9


def max_residual(sk: Sketch) -> float:
    vals = []
    if sk.constraints:
        vals.extend(constraint_residual(sk, c) for c in sk.constraints)
    for d in getattr(sk, "dimensions", None) or []:
        vals.append(dimension_residual(sk, d))
    return max(vals) if vals else 0.0


def all_satisfied(sk: Sketch, *, tol: float = TOL * 50) -> bool:
    return max_residual(sk) <= tol


# ---------------------------------------------------------------------------
# Projection steps
# ---------------------------------------------------------------------------


def _project_coincident(sk: Sketch, c: SketchConstraint, fixed: Dict[PointRef, np.ndarray]) -> None:
    r0: PointRef = (int(c.e0), str(c.h0))
    r1: PointRef = (int(c.e1), str(c.h1))
    p0 = _get_point(sk, *r0)
    p1 = _get_point(sk, *r1)
    if p0 is None or p1 is None:
        return
    f0, f1 = _is_fixed(fixed, r0), _is_fixed(fixed, r1)
    if f0 and f1:
        return  # both fixed — conflict handled elsewhere
    if f0:
        _set_point(sk, *r1, p0)
        return
    if f1:
        _set_point(sk, *r0, p1)
        return
    mid = 0.5 * (p0 + p1)
    _set_point(sk, *r0, mid)
    _set_point(sk, *r1, mid)


def _project_horizontal(sk: Sketch, c: SketchConstraint, fixed: Dict[PointRef, np.ndarray]) -> None:
    ent = sk.find_entity(c.e0)
    if not isinstance(ent, LineEntity):
        return
    r0: PointRef = (ent.id, "p0")
    r1: PointRef = (ent.id, "p1")
    f0, f1 = _is_fixed(fixed, r0), _is_fixed(fixed, r1)
    if f0 and f1:
        return
    if f0:
        ent.p1 = (ent.p1[0], ent.p0[1])
    elif f1:
        ent.p0 = (ent.p0[0], ent.p1[1])
    else:
        v = 0.5 * (ent.p0[1] + ent.p1[1])
        ent.p0 = (ent.p0[0], v)
        ent.p1 = (ent.p1[0], v)


def _project_vertical(sk: Sketch, c: SketchConstraint, fixed: Dict[PointRef, np.ndarray]) -> None:
    ent = sk.find_entity(c.e0)
    if not isinstance(ent, LineEntity):
        return
    r0: PointRef = (ent.id, "p0")
    r1: PointRef = (ent.id, "p1")
    f0, f1 = _is_fixed(fixed, r0), _is_fixed(fixed, r1)
    if f0 and f1:
        return
    if f0:
        ent.p1 = (ent.p0[0], ent.p1[1])
    elif f1:
        ent.p0 = (ent.p1[0], ent.p0[1])
    else:
        u = 0.5 * (ent.p0[0] + ent.p1[0])
        ent.p0 = (u, ent.p0[1])
        ent.p1 = (u, ent.p1[1])


def _rotate_line_keep_end(
    ent: LineEntity,
    *,
    fixed_end: str,
    target_dir: np.ndarray,
) -> None:
    """Rotate free end around fixed end so direction matches target_dir (unit-ish)."""
    n = float(np.linalg.norm(target_dir))
    if n < 1e-12:
        return
    dhat = target_dir / n
    L = line_length(ent)
    if L < 1e-12:
        L = 1.0
    if fixed_end == "p0":
        ent.p1 = (ent.p0[0] + dhat[0] * L, ent.p0[1] + dhat[1] * L)
    else:
        ent.p0 = (ent.p1[0] - dhat[0] * L, ent.p1[1] - dhat[1] * L)


def _project_parallel(sk: Sketch, c: SketchConstraint, fixed: Dict[PointRef, np.ndarray]) -> None:
    e0 = sk.find_entity(c.e0)
    e1 = sk.find_entity(c.e1)
    if not isinstance(e0, LineEntity) or not isinstance(e1, LineEntity):
        return
    d0 = np.array([e0.p1[0] - e0.p0[0], e0.p1[1] - e0.p0[1]], dtype=np.float64)
    # Prefer rotating the freer line toward the more fixed one
    f0 = sum(1 for h in ("p0", "p1") if _is_fixed(fixed, (e0.id, h)))
    f1 = sum(1 for h in ("p0", "p1") if _is_fixed(fixed, (e1.id, h)))
    if f1 >= f0:
        # rotate e0 toward e1 dir
        d1 = np.array([e1.p1[0] - e1.p0[0], e1.p1[1] - e1.p0[1]], dtype=np.float64)
        _align_line_to_dir(e0, d1, fixed)
    else:
        _align_line_to_dir(e1, d0, fixed)


def _align_line_to_dir(
    ent: LineEntity, target_dir: np.ndarray, fixed: Dict[PointRef, np.ndarray]
) -> None:
    f0 = _is_fixed(fixed, (ent.id, "p0"))
    f1 = _is_fixed(fixed, (ent.id, "p1"))
    if f0 and f1:
        return
    n = float(np.linalg.norm(target_dir))
    if n < 1e-12:
        return
    dhat = target_dir / n
    # Preserve orientation sense roughly
    cur = np.array([ent.p1[0] - ent.p0[0], ent.p1[1] - ent.p0[1]], dtype=np.float64)
    if float(np.dot(cur, dhat)) < 0:
        dhat = -dhat
    L = line_length(ent)
    if L < 1e-12:
        L = 1.0
    if f0:
        ent.p1 = (ent.p0[0] + dhat[0] * L, ent.p0[1] + dhat[1] * L)
    elif f1:
        ent.p0 = (ent.p1[0] - dhat[0] * L, ent.p1[1] - dhat[1] * L)
    else:
        mid = ent.midpoint()
        half = 0.5 * L
        ent.p0 = (mid[0] - dhat[0] * half, mid[1] - dhat[1] * half)
        ent.p1 = (mid[0] + dhat[0] * half, mid[1] + dhat[1] * half)


def _project_perpendicular(
    sk: Sketch, c: SketchConstraint, fixed: Dict[PointRef, np.ndarray]
) -> None:
    e0 = sk.find_entity(c.e0)
    e1 = sk.find_entity(c.e1)
    if not isinstance(e0, LineEntity) or not isinstance(e1, LineEntity):
        return
    d0 = np.array([e0.p1[0] - e0.p0[0], e0.p1[1] - e0.p0[1]], dtype=np.float64)
    # Perp to d0 is (-dy, dx)
    perp = np.array([-d0[1], d0[0]], dtype=np.float64)
    f0 = sum(1 for h in ("p0", "p1") if _is_fixed(fixed, (e0.id, h)))
    f1 = sum(1 for h in ("p0", "p1") if _is_fixed(fixed, (e1.id, h)))
    if f1 > f0:
        # rotate e0 to be perp to e1
        d1 = np.array([e1.p1[0] - e1.p0[0], e1.p1[1] - e1.p0[1]], dtype=np.float64)
        _align_line_to_dir(e0, np.array([-d1[1], d1[0]], dtype=np.float64), fixed)
    else:
        _align_line_to_dir(e1, perp, fixed)


def _project_equal(
    sk: Sketch,
    c: SketchConstraint,
    fixed: Dict[PointRef, np.ndarray],
    *,
    prefer_entity: Optional[int] = None,
) -> None:
    e0 = sk.find_entity(c.e0)
    e1 = sk.find_entity(c.e1)
    if not isinstance(e0, LineEntity) or not isinstance(e1, LineEntity):
        return
    L0 = line_length(e0)
    L1 = line_length(e1)
    if L0 < 1e-12 and L1 < 1e-12:
        return
    # If the user is dragging one of the lines, that line's length is the driver
    if prefer_entity is not None:
        if int(prefer_entity) == int(e0.id):
            _scale_line_to_length(e1, L0 if L0 >= 1e-12 else L1, fixed)
            return
        if int(prefer_entity) == int(e1.id):
            _scale_line_to_length(e0, L1 if L1 >= 1e-12 else L0, fixed)
            return
    # Scale freer line toward the more fixed one's length
    f0 = sum(1 for h in ("p0", "p1") if _is_fixed(fixed, (e0.id, h)))
    f1 = sum(1 for h in ("p0", "p1") if _is_fixed(fixed, (e1.id, h)))
    if f1 >= f0:
        _scale_line_to_length(e0, L1 if L1 >= 1e-12 else L0, fixed)
    else:
        _scale_line_to_length(e1, L0 if L0 >= 1e-12 else L1, fixed)


def _scale_line_to_length(
    ent: LineEntity, length: float, fixed: Dict[PointRef, np.ndarray]
) -> None:
    L = max(1e-12, float(length))
    f0 = _is_fixed(fixed, (ent.id, "p0"))
    f1 = _is_fixed(fixed, (ent.id, "p1"))
    if f0 and f1:
        return
    d = np.array([ent.p1[0] - ent.p0[0], ent.p1[1] - ent.p0[1]], dtype=np.float64)
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        dhat = np.array([1.0, 0.0])
    else:
        dhat = d / n
    if f0:
        ent.p1 = (ent.p0[0] + dhat[0] * L, ent.p0[1] + dhat[1] * L)
    elif f1:
        ent.p0 = (ent.p1[0] - dhat[0] * L, ent.p1[1] - dhat[1] * L)
    else:
        mid = ent.midpoint()
        half = 0.5 * L
        ent.p0 = (mid[0] - dhat[0] * half, mid[1] - dhat[1] * half)
        ent.p1 = (mid[0] + dhat[0] * half, mid[1] + dhat[1] * half)


def _project_fix(sk: Sketch, c: SketchConstraint) -> None:
    _set_point(sk, c.e0, c.h0, (c.u, c.v))


def _project_dimensions(
    sk: Sketch,
    fixed: Dict[PointRef, np.ndarray],
    *,
    prefer_entity: Optional[int] = None,
    prefer_handle: Optional[str] = None,
) -> None:
    """Drive geometry to stored dimension values (length / diameter / angle)."""
    from cadcore.sketch import (
        CircleEntity,
        LineEntity,
        RectEntity,
        set_arc_radius,
        set_circle_diameter,
        set_line_pair_angle,
        set_rect_height,
        set_rect_width,
    )

    for dim in getattr(sk, "dimensions", None) or []:
        role = str(dim.role)
        ent = sk.find_entity(int(dim.entity_id))
        if ent is None:
            continue
        if role == "length" and isinstance(ent, LineEntity):
            # If user drags one end, keep the other as pivot
            if prefer_entity == ent.id and prefer_handle in ("p0", "p1"):
                pivot = "p0" if prefer_handle == "p1" else "p1"
                tmp_fixed = dict(fixed)
                tmp_fixed[(ent.id, pivot)] = _get_point(sk, ent.id, pivot)  # type: ignore[arg-type]
                _scale_line_to_length(ent, float(dim.value_mm), tmp_fixed)
            else:
                _scale_line_to_length(ent, float(dim.value_mm), fixed)
        elif role == "diameter" and isinstance(ent, CircleEntity):
            set_circle_diameter(ent, float(dim.value_mm))
        elif role == "radius" and isinstance(ent, ArcEntity):
            # Keep endpoints; only center/bulge moves (same as apply_dimension_value)
            try:
                set_arc_radius(ent, float(dim.value_mm))
            except ValueError:
                # Chord too long for stored R (e.g. after a drag) — leave geometry;
                # residual will report the conflict.
                pass
        elif role == "width" and isinstance(ent, RectEntity):
            set_rect_width(ent, float(dim.value_mm), free_side="max")
        elif role == "height" and isinstance(ent, RectEntity):
            set_rect_height(ent, float(dim.value_mm), free_side="max")
        elif role == "angle" and isinstance(ent, LineEntity):
            ent_b = sk.find_entity(int(getattr(dim, "entity_b_id", -1)))
            if not isinstance(ent_b, LineEntity):
                continue
            ph0 = str(getattr(dim, "pivot_h0", "") or "")
            ph1 = str(getattr(dim, "pivot_h1", "") or "")
            pivot = (ph0, ph1) if ph0 and ph1 else None
            # Prefer rotating the dragged line when possible; always about pivot
            if prefer_entity is not None and int(prefer_entity) == int(ent.id):
                # move a: set_line_pair_angle(b, a) with pivot swapped
                piv = (ph1, ph0) if pivot else None
                set_line_pair_angle(
                    ent_b, ent, float(dim.value_mm), move="b", pivot=piv
                )
            else:
                set_line_pair_angle(
                    ent, ent_b, float(dim.value_mm), move="b", pivot=pivot
                )


def _project_one(
    sk: Sketch,
    c: SketchConstraint,
    fixed: Dict[PointRef, np.ndarray],
    *,
    prefer_entity: Optional[int] = None,
) -> None:
    if c.kind is ConstraintKind.FIX:
        _project_fix(sk, c)
    elif c.kind is ConstraintKind.COINCIDENT:
        _project_coincident(sk, c, fixed)
    elif c.kind is ConstraintKind.HORIZONTAL:
        _project_horizontal(sk, c, fixed)
    elif c.kind is ConstraintKind.VERTICAL:
        _project_vertical(sk, c, fixed)
    elif c.kind is ConstraintKind.PARALLEL:
        _project_parallel(sk, c, fixed)
    elif c.kind is ConstraintKind.PERPENDICULAR:
        _project_perpendicular(sk, c, fixed)
    elif c.kind is ConstraintKind.EQUAL:
        _project_equal(sk, c, fixed, prefer_entity=prefer_entity)
    elif c.kind is ConstraintKind.TANGENT:
        _project_tangent(sk, c, fixed)
    elif c.kind is ConstraintKind.MIDPOINT:
        _project_midpoint(sk, c, fixed)
    elif c.kind is ConstraintKind.CONCENTRIC:
        _project_concentric(sk, c, fixed)
    elif c.kind is ConstraintKind.COLLINEAR:
        _project_collinear(sk, c, fixed)
    elif c.kind is ConstraintKind.SYMMETRIC:
        _project_symmetric(sk, c, fixed)
    elif c.kind is ConstraintKind.EQUAL_RADIUS:
        _project_equal_radius(sk, c, prefer_entity=prefer_entity)


def _project_midpoint(
    sk: Sketch, c: SketchConstraint, fixed: Dict[PointRef, np.ndarray]
) -> None:
    ln = sk.find_entity(int(c.e0))
    if not isinstance(ln, LineEntity):
        return
    mid = ln.midpoint()
    h = str(c.h1 or c.h0 or "p0")
    ref = (int(c.e1), h)
    if not _is_fixed(fixed, ref):
        _set_point(sk, c.e1, h, mid)


def _project_concentric(
    sk: Sketch, c: SketchConstraint, fixed: Dict[PointRef, np.ndarray]
) -> None:
    a = sk.find_entity(int(c.e0))
    b = sk.find_entity(int(c.e1))
    if not isinstance(a, (CircleEntity, ArcEntity)) or not isinstance(
        b, (CircleEntity, ArcEntity)
    ):
        return
    # Move b's center to a's (unless a center fixed — keep simple: average)
    ca = np.array(a.center, dtype=np.float64)
    cb = np.array(b.center, dtype=np.float64)
    mid = 0.5 * (ca + cb)
    a.center = (float(mid[0]), float(mid[1]))
    b.center = (float(mid[0]), float(mid[1]))


def _project_collinear(
    sk: Sketch, c: SketchConstraint, fixed: Dict[PointRef, np.ndarray]
) -> None:
    e0 = sk.find_entity(c.e0)
    e1 = sk.find_entity(c.e1)
    if not isinstance(e0, LineEntity) or not isinstance(e1, LineEntity):
        return
    # Make e1 parallel to e0 then project endpoints onto e0's line
    d0 = np.array([e0.p1[0] - e0.p0[0], e0.p1[1] - e0.p0[1]], dtype=np.float64)
    n0 = float(np.linalg.norm(d0))
    if n0 < 1e-12:
        return
    dhat = d0 / n0
    L = line_length(e1)
    mid = e1.midpoint()
    # Project mid onto e0 line
    a = np.array(e0.p0, dtype=np.float64)
    m = np.array(mid, dtype=np.float64)
    t = float(np.dot(m - a, dhat))
    mproj = a + t * dhat
    half = 0.5 * L
    e1.p0 = (float(mproj[0] - dhat[0] * half), float(mproj[1] - dhat[1] * half))
    e1.p1 = (float(mproj[0] + dhat[0] * half), float(mproj[1] + dhat[1] * half))


def _project_symmetric(
    sk: Sketch, c: SketchConstraint, fixed: Dict[PointRef, np.ndarray]
) -> None:
    mir = sk.find_entity(int(c.e0))
    if not isinstance(mir, LineEntity):
        return
    h0 = str(c.h0 or "p0")
    h1 = str(c.h1 or "p1")
    pa = _get_point(sk, c.e1, h0)
    if pa is None:
        return
    pb_target = _reflect_point_over_line(pa, mir)
    if not _is_fixed(fixed, (int(c.e1), h1)):
        _set_point(sk, c.e1, h1, pb_target)


def _project_equal_radius(
    sk: Sketch,
    c: SketchConstraint,
    *,
    prefer_entity: Optional[int] = None,
) -> None:
    a = sk.find_entity(int(c.e0))
    b = sk.find_entity(int(c.e1))
    if not isinstance(a, (CircleEntity, ArcEntity)) or not isinstance(
        b, (CircleEntity, ArcEntity)
    ):
        return
    # Prefer leaving prefer_entity alone
    if prefer_entity is not None and int(prefer_entity) == int(a.id):
        if isinstance(b, ArcEntity):
            from cadcore.sketch import set_arc_radius

            set_arc_radius(b, float(a.radius))
        else:
            b.radius = float(a.radius)
    else:
        if isinstance(a, ArcEntity):
            from cadcore.sketch import set_arc_radius

            set_arc_radius(a, float(b.radius))
        else:
            a.radius = float(b.radius)


def _project_tangent(
    sk: Sketch, c: SketchConstraint, fixed: Dict[PointRef, np.ndarray]
) -> None:
    """Place the arc center so the arc is tangent to the line at the join.

    Convention: ``e0`` = line, ``e1`` = arc, ``h1`` = arc end ``p0``/``p1``.
    """
    ln = sk.find_entity(int(c.e0))
    arc = sk.find_entity(int(c.e1))
    h_arc = str(c.h1 or "p0")
    if not isinstance(ln, LineEntity) or not isinstance(arc, ArcEntity):
        ln2 = sk.find_entity(int(c.e1))
        arc2 = sk.find_entity(int(c.e0))
        if isinstance(ln2, LineEntity) and isinstance(arc2, ArcEntity):
            ln, arc = ln2, arc2
            h_arc = str(c.h0 or "p0")
        else:
            return
    d = np.array([ln.p1[0] - ln.p0[0], ln.p1[1] - ln.p0[1]], dtype=np.float64)
    nd = float(np.linalg.norm(d))
    if nd < 1e-12:
        return
    dhat = d / nd
    # Capture ends before mutating the arc
    free_pt = arc.p1() if h_arc == "p0" else arc.p0()
    join = arc.p0() if h_arc == "p0" else arc.p1()
    lp0 = np.array(ln.p0, dtype=np.float64)
    lp1 = np.array(ln.p1, dtype=np.float64)
    jp = np.array(join, dtype=np.float64)
    # Snap join to nearest line endpoint (usual coincident case)
    if float(np.linalg.norm(jp - lp0)) <= float(np.linalg.norm(jp - lp1)):
        jp = lp0.copy()
    else:
        jp = lp1.copy()
    r = float(arc.radius)
    n1 = np.array([-dhat[1], dhat[0]], dtype=np.float64)
    cur_c = np.array(arc.center, dtype=np.float64)
    cand1 = jp + n1 * r
    cand2 = jp - n1 * r
    new_c = (
        cand1
        if float(np.linalg.norm(cand1 - cur_c))
        <= float(np.linalg.norm(cand2 - cur_c))
        else cand2
    )
    # Free end on circle about new center
    vo = np.array([free_pt[0] - new_c[0], free_pt[1] - new_c[1]], dtype=np.float64)
    no = float(np.linalg.norm(vo))
    if no < 1e-12:
        vo = dhat.copy()
        no = 1.0
    free_on = (
        float(new_c[0] + vo[0] / no * r),
        float(new_c[1] + vo[1] / no * r),
    )
    join_pt = (float(jp[0]), float(jp[1]))
    # Mid sample on the preferred side of the chord
    mid = (
        float(0.5 * (join_pt[0] + free_on[0]) + n1[0] * r * 0.25),
        float(0.5 * (join_pt[1] + free_on[1]) + n1[1] * r * 0.25),
    )
    vm = np.array([mid[0] - new_c[0], mid[1] - new_c[1]], dtype=np.float64)
    nm = float(np.linalg.norm(vm))
    if nm > 1e-12:
        mid = (float(new_c[0] + vm[0] / nm * r), float(new_c[1] + vm[1] / nm * r))
    from cadcore.sketch import arc_from_three_points

    if h_arc == "p0":
        built = arc_from_three_points(join_pt, mid, free_on)
    else:
        built = arc_from_three_points(free_on, mid, join_pt)
    if built is None:
        arc.center = (float(new_c[0]), float(new_c[1]))
        return
    _c, _r, a0, a1, ccw = built
    arc.center = (float(new_c[0]), float(new_c[1]))
    arc.radius = r
    arc.a0, arc.a1, arc.ccw = a0, a1, ccw



def solve_sketch(
    sk: Sketch,
    *,
    drag: Optional[Tuple[int, str, Vec2]] = None,
    max_iters: int = SOLVE_ITERS,
    tol: float = TOL * 50,
) -> float:
    """Iteratively project constraints + driving dimensions.

    Optional drag pin (eid, handle, uv). Returns final max residual.
    Underconstrained DOF stay free (drag moves them).
    """
    has_dims = bool(getattr(sk, "dimensions", None))
    if not sk.constraints and not has_dims and drag is None:
        return 0.0

    # Soft pin: repeatedly re-apply dragged handle toward target if free
    drag_ref: Optional[PointRef] = None
    drag_uv: Optional[np.ndarray] = None
    prefer_entity: Optional[int] = None
    prefer_handle: Optional[str] = None
    if drag is not None:
        drag_ref = (int(drag[0]), str(drag[1]))
        drag_uv = np.array([float(drag[2][0]), float(drag[2][1])], dtype=np.float64)
        prefer_entity = int(drag[0])
        prefer_handle = str(drag[1])
        # Initial move
        if drag_ref[1]:
            _set_point(sk, drag_ref[0], drag_ref[1], drag_uv)

    fixed = _fixed_points(sk)
    # Drag of a fixed point: keep fixed (ignore mouse for that point)
    if drag_ref is not None and _is_fixed(fixed, drag_ref):
        drag_ref = None
        drag_uv = None
        prefer_entity = None
        prefer_handle = None

    residual = max_residual(sk)
    for _ in range(max_iters):
        # Prefer FIX first so others work around anchors
        ordered = sorted(
            sk.constraints,
            key=lambda c: 0 if c.kind is ConstraintKind.FIX else 1,
        )
        for c in ordered:
            _project_one(sk, c, fixed, prefer_entity=prefer_entity)
        for c in sk.constraints:
            if c.kind is ConstraintKind.FIX:
                _project_fix(sk, c)
        # Soft drag before dimensions so length/angle can re-assert
        if drag_ref is not None and drag_uv is not None and not _is_fixed(fixed, drag_ref):
            _set_point(sk, drag_ref[0], drag_ref[1], drag_uv)
        _project_dimensions(
            sk, fixed, prefer_entity=prefer_entity, prefer_handle=prefer_handle
        )
        # Constraints again to absorb dimension moves
        for c in ordered:
            if c.kind is ConstraintKind.FIX:
                continue
            _project_one(sk, c, fixed, prefer_entity=prefer_entity)
        for c in sk.constraints:
            if c.kind is ConstraintKind.FIX:
                _project_fix(sk, c)
        # Final dimension assert (dimensions win over soft drag)
        _project_dimensions(
            sk, fixed, prefer_entity=prefer_entity, prefer_handle=prefer_handle
        )
        residual = max_residual(sk)
        if residual <= tol:
            break
    return residual


# ---------------------------------------------------------------------------
# Add / remove with conflict detection
# ---------------------------------------------------------------------------


def _logical_conflict(sk: Sketch, new: SketchConstraint) -> Optional[str]:
    """Static conflicts before geometry solve."""
    for c in sk.constraints:
        if c.kind is ConstraintKind.HORIZONTAL and new.kind is ConstraintKind.VERTICAL:
            if c.e0 == new.e0:
                return f"line {new.e0} is already Horizontal"
        if c.kind is ConstraintKind.VERTICAL and new.kind is ConstraintKind.HORIZONTAL:
            if c.e0 == new.e0:
                return f"line {new.e0} is already Vertical"
        if c.kind is ConstraintKind.PARALLEL and new.kind is ConstraintKind.PERPENDICULAR:
            if {c.e0, c.e1} == {new.e0, new.e1}:
                return f"lines {new.e0} and {new.e1} are already Parallel"
        if c.kind is ConstraintKind.PERPENDICULAR and new.kind is ConstraintKind.PARALLEL:
            if {c.e0, c.e1} == {new.e0, new.e1}:
                return f"lines {new.e0} and {new.e1} are already Perpendicular"
        # Duplicate
        if (
            c.kind is new.kind
            and c.e0 == new.e0
            and c.e1 == new.e1
            and c.h0 == new.h0
            and c.h1 == new.h1
        ):
            return f"{new.label()} is already applied"
        if (
            c.kind is new.kind
            and c.kind
            in (
                ConstraintKind.PARALLEL,
                ConstraintKind.PERPENDICULAR,
                ConstraintKind.EQUAL,
                ConstraintKind.COINCIDENT,
            )
            and {c.e0, c.e1} == {new.e0, new.e1}
            and (
                c.kind is not ConstraintKind.COINCIDENT
                or {(c.e0, c.h0), (c.e1, c.h1)} == {(new.e0, new.h0), (new.e1, new.h1)}
            )
        ):
            return f"{new.label()} is already applied"
    if new.kind is ConstraintKind.COINCIDENT:
        # Two distinct FIX points cannot be forced coincident if positions differ
        f0 = f1 = None
        for c in sk.constraints:
            if c.kind is ConstraintKind.FIX and c.e0 == new.e0 and c.h0 == new.h0:
                f0 = (c.u, c.v)
            if c.kind is ConstraintKind.FIX and c.e0 == new.e1 and c.h0 == new.h1:
                f1 = (c.u, c.v)
        if f0 is not None and f1 is not None:
            if abs(f0[0] - f1[0]) > TOL * 10 or abs(f0[1] - f1[1]) > TOL * 10:
                return (
                    f"cannot make coincident: both points are Fixed at different places "
                    f"({f0[0]:.4g},{f0[1]:.4g}) vs ({f1[0]:.4g},{f1[1]:.4g})"
                )
    return None


def _validate_refs(sk: Sketch, c: SketchConstraint) -> None:
    if c.kind is ConstraintKind.HORIZONTAL or c.kind is ConstraintKind.VERTICAL:
        ent = sk.find_entity(c.e0)
        if not isinstance(ent, LineEntity):
            raise ValueError(f"{c.label()} requires a line")
        return
    if c.kind is ConstraintKind.PARALLEL or c.kind is ConstraintKind.PERPENDICULAR or c.kind is ConstraintKind.EQUAL:
        e0 = sk.find_entity(c.e0)
        e1 = sk.find_entity(c.e1)
        if not isinstance(e0, LineEntity) or not isinstance(e1, LineEntity):
            raise ValueError(f"{c.label()} requires two lines")
        if c.e0 == c.e1:
            raise ValueError(f"{c.label()} needs two different lines")
        return
    if c.kind is ConstraintKind.COINCIDENT:
        if _get_point(sk, c.e0, c.h0) is None or _get_point(sk, c.e1, c.h1) is None:
            raise ValueError("Coincident requires two valid points")
        if c.e0 == c.e1 and c.h0 == c.h1:
            raise ValueError("Coincident needs two different points")
        return
    if c.kind is ConstraintKind.FIX:
        if _get_point(sk, c.e0, c.h0) is None:
            raise ValueError("Fix requires a valid point")
        return
    if c.kind is ConstraintKind.TANGENT:
        ln = sk.find_entity(c.e0)
        arc = sk.find_entity(c.e1)
        if not isinstance(ln, LineEntity) or not isinstance(arc, ArcEntity):
            raise ValueError("Tangent requires a line and an arc")
        if str(c.h1) not in ("p0", "p1"):
            raise ValueError("Tangent requires the arc end (p0 or p1)")
        return
    if c.kind is ConstraintKind.MIDPOINT:
        ln = sk.find_entity(c.e0)
        if not isinstance(ln, LineEntity):
            raise ValueError("Midpoint requires a line as e0")
        if _get_point(sk, c.e1, c.h1 or c.h0) is None:
            raise ValueError("Midpoint requires a valid point (e1, h1)")
        return
    if c.kind is ConstraintKind.CONCENTRIC:
        a = sk.find_entity(c.e0)
        b = sk.find_entity(c.e1)
        if not isinstance(a, (CircleEntity, ArcEntity)) or not isinstance(
            b, (CircleEntity, ArcEntity)
        ):
            raise ValueError("Concentric requires two circles or arcs")
        return
    if c.kind is ConstraintKind.COLLINEAR:
        e0 = sk.find_entity(c.e0)
        e1 = sk.find_entity(c.e1)
        if not isinstance(e0, LineEntity) or not isinstance(e1, LineEntity):
            raise ValueError("Collinear requires two lines")
        return
    if c.kind is ConstraintKind.SYMMETRIC:
        mir = sk.find_entity(c.e0)
        if not isinstance(mir, LineEntity):
            raise ValueError("Symmetric requires a mirror line as e0")
        if _get_point(sk, c.e1, c.h0 or "p0") is None:
            raise ValueError("Symmetric requires handles on subject entity e1")
        return
    if c.kind is ConstraintKind.EQUAL_RADIUS:
        a = sk.find_entity(c.e0)
        b = sk.find_entity(c.e1)
        if not isinstance(a, (CircleEntity, ArcEntity)) or not isinstance(
            b, (CircleEntity, ArcEntity)
        ):
            raise ValueError("Equal radius requires two circles or arcs")
        return
    raise ValueError("unknown constraint kind")


def add_constraint(sk: Sketch, c: SketchConstraint, *, tol: float = TOL * 80) -> SketchConstraint:
    """Add a constraint, solve, refuse on conflict (sketch unchanged).

    Mutates ``c.id`` and appends to ``sk.constraints`` on success.
    """
    _validate_refs(sk, c)
    msg = _logical_conflict(sk, c)
    if msg:
        raise ValueError(msg)

    # Snapshot geometry
    before = [snapshot_entity(e) for e in sk.entities]
    before_cons = list(sk.constraints)

    if c.kind is ConstraintKind.FIX:
        p = _get_point(sk, c.e0, c.h0)
        assert p is not None
        c.u, c.v = float(p[0]), float(p[1])

    c.id = int(sk._next_constraint_id)
    sk._next_constraint_id += 1
    sk.constraints.append(c)

    # Seed geometric nudge for H/V so solve starts near manifold
    if c.kind is ConstraintKind.HORIZONTAL:
        from cadcore.sketch import make_line_horizontal

        ent = sk.find_entity(c.e0)
        if isinstance(ent, LineEntity):
            make_line_horizontal(ent)
    elif c.kind is ConstraintKind.VERTICAL:
        from cadcore.sketch import make_line_vertical

        ent = sk.find_entity(c.e0)
        if isinstance(ent, LineEntity):
            make_line_vertical(ent)
    elif c.kind is ConstraintKind.EQUAL:
        from cadcore.sketch import make_lines_equal_length

        e0 = sk.find_entity(c.e0)
        e1 = sk.find_entity(c.e1)
        if isinstance(e0, LineEntity) and isinstance(e1, LineEntity):
            make_lines_equal_length(e0, e1)
    elif c.kind is ConstraintKind.COINCIDENT:
        _project_coincident(sk, c, _fixed_points(sk))
    elif c.kind is ConstraintKind.PARALLEL:
        _project_parallel(sk, c, _fixed_points(sk))
    elif c.kind is ConstraintKind.PERPENDICULAR:
        _project_perpendicular(sk, c, _fixed_points(sk))

    residual = solve_sketch(sk, max_iters=SOLVE_ITERS * 2, tol=tol)
    if residual > tol:
        # Restore
        for snap in before:
            ent = sk.find_entity(int(snap["id"]))
            if ent is not None:
                apply_entity_snapshot(ent, snap)
        sk.constraints = before_cons
        sk._next_constraint_id = max(
            [0] + [x.id for x in sk.constraints]
        ) + 1
        # Find worst old constraint if any for message
        raise ValueError(
            f"cannot apply {c.label()}: it conflicts with existing constraints "
            f"(residual {residual:.4g}). Sketch left unchanged."
        )
    return c


def remove_constraint(sk: Sketch, cid: int) -> bool:
    n = len(sk.constraints)
    sk.constraints = [c for c in sk.constraints if c.id != int(cid)]
    return len(sk.constraints) < n


def remove_constraints_for_entity(sk: Sketch, eid: int) -> int:
    before = len(sk.constraints)
    sk.constraints = [
        c
        for c in sk.constraints
        if int(c.e0) != int(eid) and int(c.e1) != int(eid)
    ]
    return before - len(sk.constraints)


def constraints_involving(sk: Sketch, eid: int) -> List[SketchConstraint]:
    return [c for c in sk.constraints if c.e0 == eid or c.e1 == eid]


def constraint_anchor_uv(sk: Sketch, c: SketchConstraint) -> Vec2:
    """Where to draw the constraint glyph."""
    if c.kind is ConstraintKind.FIX or c.kind is ConstraintKind.COINCIDENT:
        p = _get_point(sk, c.e0, c.h0)
        if p is not None:
            return (float(p[0]), float(p[1]))
    ent = sk.find_entity(c.e0)
    if isinstance(ent, LineEntity):
        mid = ent.midpoint()
        if c.kind in (
            ConstraintKind.PARALLEL,
            ConstraintKind.PERPENDICULAR,
            ConstraintKind.EQUAL,
        ):
            e1 = sk.find_entity(c.e1)
            if isinstance(e1, LineEntity):
                m1 = e1.midpoint()
                return (0.5 * (mid[0] + m1[0]), 0.5 * (mid[1] + m1[1]))
        return mid
    return (0.0, 0.0)
