"""2D polygon corner fillet (SolidWorks-style sketch fillet geometry).

Replaces sharp corners with circular arcs so the corner vertex is removed.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

Vec2 = Tuple[float, float]


def _norm(x: float, y: float) -> Tuple[float, float]:
    n = float(np.hypot(x, y))
    if n < 1e-15:
        return (0.0, 0.0)
    return (x / n, y / n)


def fillet_closed_polygon(
    vertices: Sequence[Vec2],
    radius: float,
    *,
    arc_segments: int = 12,
) -> List[Vec2]:
    """Return a closed polygon with every convex corner rounded by ``radius``.

    Input vertices form a closed ring (first != last, or last may equal first).
    Output is open ring (first != last); caller closes by connecting last→first.
    Sharp corner vertices are **not** present in the output — arcs replace them.
    """
    r = float(radius)
    if not np.isfinite(r) or r <= 1e-12:
        raise ValueError("fillet radius must be positive")
    segs = max(2, int(arc_segments))

    pts = [(float(p[0]), float(p[1])) for p in vertices]
    if len(pts) >= 2 and abs(pts[0][0] - pts[-1][0]) < 1e-12 and abs(pts[0][1] - pts[-1][1]) < 1e-12:
        pts = pts[:-1]
    n = len(pts)
    if n < 3:
        raise ValueError("polygon needs at least 3 vertices")

    # Ensure CCW for consistent left-hand normals
    area = 0.0
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        area += x0 * y1 - x1 * y0
    if area < 0:
        pts = list(reversed(pts))

    out: List[Vec2] = []
    for i in range(n):
        p_prev = pts[(i - 1) % n]
        p = pts[i]
        p_next = pts[(i + 1) % n]
        v1 = _norm(p[0] - p_prev[0], p[1] - p_prev[1])  # incoming direction
        v2 = _norm(p_next[0] - p[0], p_next[1] - p[1])  # outgoing direction
        if v1 == (0.0, 0.0) or v2 == (0.0, 0.0):
            out.append(p)
            continue
        # Signed turn (CCW positive)
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        dot = max(-1.0, min(1.0, v1[0] * v2[0] + v1[1] * v2[1]))
        turn = float(np.arctan2(cross, dot))
        # Concave or nearly collinear: keep the vertex
        if turn <= 1e-6:
            out.append(p)
            continue
        half = 0.5 * turn
        # Trim distance along each edge from the corner
        tan_h = float(np.tan(half))
        if abs(tan_h) < 1e-12:
            out.append(p)
            continue
        dist = r / tan_h
        # Edge lengths — don't over-trim
        len_in = float(np.hypot(p[0] - p_prev[0], p[1] - p_prev[1]))
        len_out = float(np.hypot(p_next[0] - p[0], p_next[1] - p[1]))
        max_trim = 0.5 * min(len_in, len_out) - 1e-9
        if dist >= max_trim - 1e-12:
            raise ValueError(
                "fillet radius too large for the profile (self-intersection)"
            )
        t1 = (p[0] - v1[0] * dist, p[1] - v1[1] * dist)
        t2 = (p[0] + v2[0] * dist, p[1] + v2[1] * dist)
        # Inward normal (left of incoming): (-vy, vx)
        n1 = (-v1[1], v1[0])
        center = (t1[0] + n1[0] * r, t1[1] + n1[1] * r)
        # Arc angles from center: t1 → t2, sweeping CCW (inward for CCW poly)
        a0 = float(np.arctan2(t1[1] - center[1], t1[0] - center[0]))
        a1 = float(np.arctan2(t2[1] - center[1], t2[0] - center[0]))
        # Sweep positive CCW from a0 to a1
        da = a1 - a0
        while da <= 0:
            da += 2.0 * np.pi
        # For convex corner, expected sweep is (pi - turn)? Actually interior
        # supplement: for 90° turn, sweep should be 90° = pi/2 = turn.
        # Clamp to the smaller positive arc matching turn angle
        if abs(da - turn) > abs(da - (2 * np.pi - turn)) and turn < np.pi:
            # wrong direction — go the other way
            da = da - 2.0 * np.pi
        n_arc = max(2, int(round(segs * abs(da) / (0.5 * np.pi))))
        n_arc = max(2, min(n_arc, segs * 2))
        out.append(t1)
        for k in range(1, n_arc):
            t = k / float(n_arc)
            ang = a0 + da * t
            out.append(
                (center[0] + r * float(np.cos(ang)), center[1] + r * float(np.sin(ang)))
            )
        out.append(t2)
    # Dedup consecutive near-equal points
    cleaned: List[Vec2] = []
    for p in out:
        if cleaned and abs(p[0] - cleaned[-1][0]) < 1e-12 and abs(p[1] - cleaned[-1][1]) < 1e-12:
            continue
        cleaned.append((float(p[0]), float(p[1])))
    if (
        len(cleaned) >= 2
        and abs(cleaned[0][0] - cleaned[-1][0]) < 1e-12
        and abs(cleaned[0][1] - cleaned[-1][1]) < 1e-12
    ):
        cleaned = cleaned[:-1]
    if len(cleaned) < 3:
        raise ValueError("fillet produced degenerate polygon")
    return cleaned


def profile_to_polygon_uv(profile) -> List[Vec2]:
    """Extract UV ring (no repeated closing vertex) from a closed profile entity."""
    from cadcore.profiles import ClosedLineLoop
    from cadcore.sketch import CircleEntity, RectEntity

    if isinstance(profile, RectEntity):
        return list(profile.corners())
    if isinstance(profile, ClosedLineLoop):
        verts = list(profile.vertices)
        if (
            len(verts) >= 2
            and abs(verts[0][0] - verts[-1][0]) < 1e-12
            and abs(verts[0][1] - verts[-1][1]) < 1e-12
        ):
            verts = verts[:-1]
        return [(float(p[0]), float(p[1])) for p in verts]
    if isinstance(profile, CircleEntity):
        # Already smooth — return polyline approximation
        n = 48
        cx, cy, r = profile.center[0], profile.center[1], profile.radius
        return [
            (cx + r * float(np.cos(2 * np.pi * i / n)), cy + r * float(np.sin(2 * np.pi * i / n)))
            for i in range(n)
        ]
    if isinstance(profile, (list, tuple)) and profile and not hasattr(profile, "kind"):
        return [(float(p[0]), float(p[1])) for p in profile]  # type: ignore[union-attr]
    raise ValueError(f"unsupported profile for 2D fillet: {type(profile)!r}")
