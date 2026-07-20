"""Closed sketch profiles: rectangles, circles, and closed line-segment loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from cadcore.sketch import ArcEntity, CircleEntity, LineEntity, RectEntity, Sketch, Vec2

ENDPOINT_TOL = 1e-6
# Samples per arc when building a closed polygon for area / extrude
ARC_POLY_SAMPLES = 24


@dataclass(frozen=True)
class ClosedLineLoop:
    """Virtual closed profile from ordered, connected line/arc segments.

    ``id`` is synthetic (negative) so it never collides with real entity ids.
    ``vertices`` are UV polygon corners in order (first != last; loop is implicit).
    Arcs are tessellated into polyline samples in ``vertices``.
    """

    vertices: Tuple[Vec2, ...]
    line_ids: Tuple[int, ...]  # entity ids of edges (lines and arcs)
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


def _edge_endpoints(e) -> Optional[Tuple[Vec2, Vec2]]:
    if isinstance(e, LineEntity):
        return e.p0, e.p1
    if isinstance(e, ArcEntity):
        return e.p0(), e.p1()
    return None


def _edge_polyline(e, *, from_uv: Vec2, to_uv: Vec2, n: int = ARC_POLY_SAMPLES) -> List[Vec2]:
    """Polyline for an edge traveled from ``from_uv`` to ``to_uv`` (exclude start)."""
    if isinstance(e, LineEntity):
        return [(float(to_uv[0]), float(to_uv[1]))]
    if isinstance(e, ArcEntity):
        samples = e.sample_uv(n)
        # sample_uv goes p0→p1; reverse if we travel p1→p0
        if _qkey(from_uv) == _qkey(e.p1()) and _qkey(to_uv) == _qkey(e.p0()):
            samples = list(reversed(samples))
        # drop first (already at from_uv)
        return [(float(p[0]), float(p[1])) for p in samples[1:]]
    return [(float(to_uv[0]), float(to_uv[1]))]


def find_closed_line_loops(
    sketch: Sketch,
    *,
    tol: float = ENDPOINT_TOL,
) -> List[ClosedLineLoop]:
    """Detect closed loops from LineEntity and ArcEntity edges.

    Raises:
        ValueError: branching (vertex degree > 2) or self-intersecting loop.
    Open chains are not returned (callers may raise via ``list_closed_profiles``).
    """
    edges = [
        e
        for e in sketch.entities
        if isinstance(e, (LineEntity, ArcEntity)) and not bool(
            getattr(e, "construction", False)
        )
    ]
    if not edges:
        return []

    # adj[key] -> list of (other_key, edge_id, from_uv, to_uv, entity)
    adj: Dict[
        Tuple[float, float],
        List[Tuple[Tuple[float, float], int, Vec2, Vec2, object]],
    ] = {}
    by_id = {e.id: e for e in edges}
    for e in edges:
        ends = _edge_endpoints(e)
        if ends is None:
            continue
        p0, p1 = ends
        k0, k1 = _qkey(p0, tol), _qkey(p1, tol)
        if k0 == k1:
            continue
        adj.setdefault(k0, []).append((k1, e.id, p0, p1, e))
        adj.setdefault(k1, []).append((k0, e.id, p1, p0, e))

    for k, nbrs in adj.items():
        if len(nbrs) > 2:
            raise ValueError(
                "branching: a vertex is shared by more than 2 segments "
                f"(degree={len(nbrs)} at {k})"
            )

    used: set[int] = set()
    loops: List[ClosedLineLoop] = []

    for start in list(adj.keys()):
        if len(adj[start]) != 2:
            continue
        if all(n[1] in used for n in adj[start]):
            continue

        start_edge = next(n for n in adj[start] if n[1] not in used)
        verts: List[Vec2] = []
        lids: List[int] = []
        nkey, lid, from_uv, to_uv, ent = start_edge
        verts.append((float(from_uv[0]), float(from_uv[1])))
        verts.extend(_edge_polyline(ent, from_uv=from_uv, to_uv=to_uv))
        lids.append(lid)
        used.add(lid)
        prev_lid = lid
        cur = nkey

        closed = False
        for _ in range(len(edges) + 1):
            if cur == start and len(lids) >= 2:
                # Need at least 2 edges for line+arc slot, 3 for pure lines
                if len(lids) >= 2:
                    closed = True
                    break
            options = [n for n in adj[cur] if n[1] != prev_lid]
            if not options:
                break
            options_u = [n for n in options if n[1] not in used]
            if not options_u:
                break
            nkey, lid, from_uv, to_uv, ent = options_u[0]
            verts.extend(_edge_polyline(ent, from_uv=from_uv, to_uv=to_uv))
            lids.append(lid)
            used.add(lid)
            prev_lid = lid
            cur = nkey

        if not closed or cur != start:
            for lid in lids:
                used.discard(lid)
            continue

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
                if abs(i - j) % n == 1 or abs(i - j) % n == n - 1:
                    continue
                c, d = verts[j], verts[(j + 1) % n]
                if _seg_intersect(a, b, c, d):
                    raise ValueError(
                        "self-intersecting loop: segments cross each other"
                    )

        if abs(_shoelace(verts)) <= 1e-12:
            raise ValueError("degenerate closed loop: zero area")
        verts = _orient_ccw(verts)
        loops.append(ClosedLineLoop(vertices=tuple(verts), line_ids=tuple(lids)))

    return loops


def has_open_line_chain(sketch: Sketch, *, tol: float = ENDPOINT_TOL) -> bool:
    edges = [
        e
        for e in sketch.entities
        if isinstance(e, (LineEntity, ArcEntity))
        and not bool(getattr(e, "construction", False))
        ]
    if not edges:
        return False
    deg: Dict[Tuple[float, float], int] = {}
    for e in edges:
        ends = _edge_endpoints(e)
        if ends is None:
            continue
        for p in ends:
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
    edges = [
        e
        for e in sketch.entities
        if isinstance(e, (LineEntity, ArcEntity))
        and not bool(getattr(e, "construction", False))
        ]
    if edges:
        raise ValueError("open chain: segments do not form a closed loop")
    raise ValueError(
        "sketch has no closed profile (rectangle, circle, or closed line/arc loop)"
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


def profile_area(profile: object) -> float:
    """Signed-area magnitude of a closed profile (for innermost pick)."""
    if isinstance(profile, ClosedLineLoop):
        return float(profile.area())
    if isinstance(profile, RectEntity):
        u0, u1 = sorted([profile.c0[0], profile.c1[0]])
        v0, v1 = sorted([profile.c0[1], profile.c1[1]])
        return abs((u1 - u0) * (v1 - v0))
    if isinstance(profile, CircleEntity):
        return float(np.pi * profile.radius * profile.radius)
    return 0.0


def pick_closed_profile_at(sketch: Sketch, uv: Vec2) -> Optional[object]:
    """Closed profile containing ``uv``, preferring the smallest (innermost).

    Returns None if the point is outside every closed region.
    """
    try:
        closed = list_closed_profiles(sketch)
    except ValueError:
        return None
    hits = [p for p in closed if point_in_profile(uv, p)]
    if not hits:
        return None
    return min(hits, key=profile_area)


def profile_by_id(sketch: Sketch, profile_id: int) -> Optional[object]:
    """Look up a closed profile (entity or virtual line-loop) by id."""
    try:
        closed = list_closed_profiles(sketch)
    except ValueError:
        return None
    pid = int(profile_id)
    for p in closed:
        if int(getattr(p, "id", -1)) == pid:
            return p
        if isinstance(p, ClosedLineLoop) and pid in p.line_ids:
            return p
    return None


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
