"""Closed sketch profiles: rectangles, circles, and closed line-segment loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from cadcore.sketch import CircleEntity, LineEntity, RectEntity, Sketch, Vec2

ENDPOINT_TOL = 1e-6


@dataclass(frozen=True)
class ClosedLineLoop:
    """Virtual closed profile from ordered, connected LineEntity segments.

    ``id`` is synthetic (negative) so it never collides with real entity ids.
    ``vertices`` are UV polygon corners in order (first != last; loop is implicit).
    """

    vertices: Tuple[Vec2, ...]
    line_ids: Tuple[int, ...]
    id: int = -1

    def __post_init__(self) -> None:
        if self.id == -1 and self.line_ids:
            object.__setattr__(self, "id", -1_000_000 - int(min(self.line_ids)))

    def area(self) -> float:
        return abs(_shoelace(self.vertices))


def _shoelace(verts: Sequence[Vec2]) -> float:
    n = len(verts)
    if n < 3:
        return 0.0
    a = 0.0
    for i in range(n):
        u0, v0 = float(verts[i][0]), float(verts[i][1])
        u1, v1 = float(verts[(i + 1) % n][0]), float(verts[(i + 1) % n][1])
        a += u0 * v1 - u1 * v0
    return 0.5 * a


def _qkey(uv: Vec2, tol: float = ENDPOINT_TOL) -> Tuple[float, float]:
    return (round(float(uv[0]) / tol) * tol, round(float(uv[1]) / tol) * tol)


def _seg_intersect(a: Vec2, b: Vec2, c: Vec2, d: Vec2, *, tol: float = 1e-9) -> bool:
    """True if open segments ab and cd properly cross (not at shared endpoints)."""
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    cx, cy = float(c[0]), float(c[1])
    dx, dy = float(d[0]), float(d[1])

    def orient(ox, oy, px, py, qx, qy):
        return (px - ox) * (qy - oy) - (py - oy) * (qx - ox)

    o1 = orient(ax, ay, bx, by, cx, cy)
    o2 = orient(ax, ay, bx, by, dx, dy)
    o3 = orient(cx, cy, dx, dy, ax, ay)
    o4 = orient(cx, cy, dx, dy, bx, by)
    if ((o1 > tol and o2 < -tol) or (o1 < -tol and o2 > tol)) and (
        (o3 > tol and o4 < -tol) or (o3 < -tol and o4 > tol)
    ):
        return True
    return False


def _orient_ccw(verts: List[Vec2]) -> List[Vec2]:
    if _shoelace(verts) < 0:
        return list(reversed(verts))
    return verts


def find_closed_line_loops(
    sketch: Sketch,
    *,
    tol: float = ENDPOINT_TOL,
) -> List[ClosedLineLoop]:
    """Detect closed polyline loops from LineEntity segments.

    Raises:
        ValueError: branching (vertex degree > 2) or self-intersecting loop.
    Open chains are not returned (callers may raise via ``list_closed_profiles``).
    """
    lines = [e for e in sketch.entities if isinstance(e, LineEntity)]
    if not lines:
        return []

    # adj[key] -> list of (other_key, line_id, from_uv, to_uv)
    adj: Dict[Tuple[float, float], List[Tuple[Tuple[float, float], int, Vec2, Vec2]]] = {}
    for ln in lines:
        k0, k1 = _qkey(ln.p0, tol), _qkey(ln.p1, tol)
        if k0 == k1:
            continue
        adj.setdefault(k0, []).append((k1, ln.id, ln.p0, ln.p1))
        adj.setdefault(k1, []).append((k0, ln.id, ln.p1, ln.p0))

    for k, nbrs in adj.items():
        if len(nbrs) > 2:
            raise ValueError(
                "branching: a vertex is shared by more than 2 line segments "
                f"(degree={len(nbrs)} at {k})"
            )

    used: set[int] = set()
    loops: List[ClosedLineLoop] = []

    for start in list(adj.keys()):
        if len(adj[start]) != 2:
            continue
        if all(n[1] in used for n in adj[start]):
            continue

        # Start along one unused edge
        start_edge = next(n for n in adj[start] if n[1] not in used)
        verts: List[Vec2] = []
        lids: List[int] = []
        cur = start
        prev_lid: Optional[int] = None
        nkey, lid, from_uv, to_uv = start_edge
        verts.append((float(from_uv[0]), float(from_uv[1])))
        verts.append((float(to_uv[0]), float(to_uv[1])))
        lids.append(lid)
        used.add(lid)
        prev_lid = lid
        cur = nkey

        closed = False
        for _ in range(len(lines) + 1):
            if cur == start and len(lids) >= 3:
                closed = True
                break
            # next edge: not reverse of previous line
            options = [n for n in adj[cur] if n[1] != prev_lid]
            if not options:
                break
            # prefer unused
            options_u = [n for n in options if n[1] not in used]
            if not options_u:
                # only way is reverse or already used — fail unless at start
                break
            nkey, lid, from_uv, to_uv = options_u[0]
            verts.append((float(to_uv[0]), float(to_uv[1])))
            lids.append(lid)
            used.add(lid)
            prev_lid = lid
            cur = nkey

        if not closed or cur != start:
            for lid in lids:
                used.discard(lid)
            continue

        # Drop closing duplicate
        if len(verts) >= 2 and _qkey(verts[0], tol) == _qkey(verts[-1], tol):
            verts = verts[:-1]
        if len(verts) < 3:
            continue

        n = len(verts)
        for i in range(n):
            a, b = verts[i], verts[(i + 1) % n]
            for j in range(i + 1, n):
                if j == i or j == (i + 1) % n or i == (j + 1) % n:
                    continue
                # adjacent edges share a vertex — skip
                if abs(i - j) % n == 1 or abs(i - j) % n == n - 1:
                    continue
                c, d = verts[j], verts[(j + 1) % n]
                if _seg_intersect(a, b, c, d):
                    raise ValueError(
                        "self-intersecting loop: line segments cross each other"
                    )

        if abs(_shoelace(verts)) <= 1e-12:
            raise ValueError("degenerate closed loop: zero area")
        verts = _orient_ccw(verts)
        loops.append(ClosedLineLoop(vertices=tuple(verts), line_ids=tuple(lids)))

    return loops


def has_open_line_chain(sketch: Sketch, *, tol: float = ENDPOINT_TOL) -> bool:
    lines = [e for e in sketch.entities if isinstance(e, LineEntity)]
    if not lines:
        return False
    deg: Dict[Tuple[float, float], int] = {}
    for ln in lines:
        for p in (ln.p0, ln.p1):
            k = _qkey(p, tol)
            deg[k] = deg.get(k, 0) + 1
    return any(d == 1 for d in deg.values())


def is_closed_profile(entity: object) -> bool:
    if isinstance(entity, ClosedLineLoop):
        return entity.area() > 1e-12
    if isinstance(entity, RectEntity):
        u0, u1 = sorted([entity.c0[0], entity.c1[0]])
        v0, v1 = sorted([entity.c0[1], entity.c1[1]])
        return (u1 - u0) > 1e-12 and (v1 - v0) > 1e-12
    if isinstance(entity, CircleEntity):
        return entity.radius > 1e-12
    return False


def list_closed_profiles(sketch: Sketch) -> List[object]:
    """All closed profiles. Raises for branching/self-intersect/open-only lines."""
    closed: List[object] = [
        e for e in sketch.entities if is_closed_profile(e) and not isinstance(e, ClosedLineLoop)
    ]
    loops = find_closed_line_loops(sketch)
    closed.extend(loops)
    if closed:
        return closed
    lines = [e for e in sketch.entities if isinstance(e, LineEntity)]
    if lines:
        raise ValueError("open chain: line segments do not form a closed loop")
    raise ValueError(
        "sketch has no closed profile (rectangle, circle, or closed line loop)"
    )


def profile_polygon_uv(profile: object) -> List[Vec2]:
    if isinstance(profile, ClosedLineLoop):
        return list(profile.vertices)
    if isinstance(profile, RectEntity):
        u0, u1 = sorted([profile.c0[0], profile.c1[0]])
        v0, v1 = sorted([profile.c0[1], profile.c1[1]])
        return [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
    if isinstance(profile, CircleEntity):
        n = 32
        return [
            (
                profile.center[0] + profile.radius * float(np.cos(2 * np.pi * i / n)),
                profile.center[1] + profile.radius * float(np.sin(2 * np.pi * i / n)),
            )
            for i in range(n)
        ]
    raise TypeError(f"not a closed profile: {type(profile)!r}")


def point_in_profile(uv: Vec2, profile: object, *, margin: float = 0.0) -> bool:
    if isinstance(profile, RectEntity):
        u0, u1 = sorted([profile.c0[0], profile.c1[0]])
        v0, v1 = sorted([profile.c0[1], profile.c1[1]])
        return (u0 + margin <= uv[0] <= u1 - margin) and (
            v0 + margin <= uv[1] <= v1 - margin
        )
    if isinstance(profile, CircleEntity):
        d = float(np.hypot(uv[0] - profile.center[0], uv[1] - profile.center[1]))
        return d < profile.radius - margin
    if isinstance(profile, ClosedLineLoop):
        return _point_in_poly(uv, profile.vertices)
    return False


def _point_in_poly(uv: Vec2, verts: Sequence[Vec2]) -> bool:
    x, y = float(uv[0]), float(uv[1])
    n = len(verts)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(verts[i][0]), float(verts[i][1])
        xj, yj = float(verts[j][0]), float(verts[j][1])
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-30) + xi
        ):
            inside = not inside
        j = i
    return inside
