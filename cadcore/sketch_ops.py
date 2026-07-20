"""Sketch editing ops: SolidWorks-style trim/extend, offset, convert, DOF.

Trim removes the *segment between intersections* that the user clicked — never
the raw cursor projection. Extend grows an open end until the next entity hit.
Lines, arcs, and circles are all supported.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from cadcore.sketch import (
    ArcEntity,
    CircleEntity,
    EntityKind,
    LineEntity,
    RectEntity,
    Sketch,
    SketchEntity,
    Vec2,
    arc_from_three_points,
)

TOL = 1e-9
PARAM_EPS = 1e-8


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Parametric helpers (t ∈ [0,1] along entity travel)
# ---------------------------------------------------------------------------


def _line_point(ln: LineEntity, t: float) -> np.ndarray:
    p0 = np.array(ln.p0, dtype=np.float64)
    p1 = np.array(ln.p1, dtype=np.float64)
    return p0 + float(t) * (p1 - p0)


def _line_t(ln: LineEntity, uv: Sequence[float]) -> float:
    p0 = np.array(ln.p0, dtype=np.float64)
    p1 = np.array(ln.p1, dtype=np.float64)
    d = p1 - p0
    n2 = float(np.dot(d, d))
    if n2 < 1e-24:
        return 0.0
    pt = np.array([float(uv[0]), float(uv[1])], dtype=np.float64)
    return float(np.dot(pt - p0, d) / n2)


def _circle_point(c: CircleEntity, t: float) -> np.ndarray:
    """t ∈ [0,1) around full circle from +U, CCW."""
    a = 2.0 * np.pi * float(t)
    return np.array(
        [
            c.center[0] + c.radius * float(np.cos(a)),
            c.center[1] + c.radius * float(np.sin(a)),
        ],
        dtype=np.float64,
    )


def _circle_t(c: CircleEntity, uv: Sequence[float]) -> float:
    a = float(np.atan2(float(uv[1]) - c.center[1], float(uv[0]) - c.center[0]))
    if a < 0:
        a += 2.0 * np.pi
    return a / (2.0 * np.pi)


def _arc_point(arc: ArcEntity, t: float) -> np.ndarray:
    """t ∈ [0,1] along arc from p0 to p1 in travel sense."""
    a = arc.a0 + float(t) * arc.sweep()
    return np.array(
        [
            arc.center[0] + arc.radius * float(np.cos(a)),
            arc.center[1] + arc.radius * float(np.sin(a)),
        ],
        dtype=np.float64,
    )


def _arc_t(arc: ArcEntity, uv: Sequence[float]) -> float:
    a = float(np.atan2(float(uv[1]) - arc.center[1], float(uv[0]) - arc.center[0]))
    sw = arc.sweep()
    # Delta from a0 in travel direction
    if arc.ccw:
        d = a - arc.a0
        while d < 0:
            d += 2.0 * np.pi
        while d >= 2.0 * np.pi:
            d -= 2.0 * np.pi
        if abs(sw) < 1e-15:
            return 0.0
        return float(np.clip(d / sw, 0.0, 1.0))
    d = arc.a0 - a
    while d < 0:
        d += 2.0 * np.pi
    while d >= 2.0 * np.pi:
        d -= 2.0 * np.pi
    if abs(sw) < 1e-15:
        return 0.0
    return float(np.clip(d / abs(sw), 0.0, 1.0))


def entity_point_at(ent: SketchEntity, t: float) -> np.ndarray:
    if isinstance(ent, LineEntity):
        return _line_point(ent, t)
    if isinstance(ent, CircleEntity):
        return _circle_point(ent, t % 1.0)
    if isinstance(ent, ArcEntity):
        return _arc_point(ent, t)
    raise TypeError(type(ent))


def entity_param_at(ent: SketchEntity, uv: Sequence[float]) -> float:
    if isinstance(ent, LineEntity):
        return float(np.clip(_line_t(ent, uv), 0.0, 1.0))
    if isinstance(ent, CircleEntity):
        return _circle_t(ent, uv) % 1.0
    if isinstance(ent, ArcEntity):
        return _arc_t(ent, uv)
    raise TypeError(type(ent))


# ---------------------------------------------------------------------------
# Intersections → parameters on target entity
# ---------------------------------------------------------------------------


def _seg_seg_t(
    a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray
) -> Optional[Tuple[float, float]]:
    """Infinite-line params; caller clips. Returns (t_a, t_b) or None if parallel."""
    da = a1 - a0
    db = b1 - b0
    cross = float(da[0] * db[1] - da[1] * db[0])
    if abs(cross) < 1e-14:
        return None
    diff = b0 - a0
    ta = float(diff[0] * db[1] - diff[1] * db[0]) / cross
    tb = float(diff[0] * da[1] - diff[1] * da[0]) / cross
    return ta, tb


def _line_circle_ts(
    p0: np.ndarray, p1: np.ndarray, center: np.ndarray, radius: float
) -> List[float]:
    """Parameters t on infinite line p0→p1 for circle intersections."""
    d = p1 - p0
    f = p0 - center
    a = float(np.dot(d, d))
    if a < 1e-24:
        return []
    b = 2.0 * float(np.dot(f, d))
    c = float(np.dot(f, f)) - float(radius) * float(radius)
    disc = b * b - 4 * a * c
    if disc < -1e-12:
        return []
    if disc < 0:
        disc = 0.0
    s = float(np.sqrt(disc))
    return [(-b - s) / (2 * a), (-b + s) / (2 * a)]


def _angle_on_arc(arc: ArcEntity, ang: float) -> bool:
    """True if absolute angle ``ang`` lies on the arc sweep (inclusive ends)."""
    sw = arc.sweep()
    if arc.ccw:
        d = ang - arc.a0
        while d < -1e-12:
            d += 2.0 * np.pi
        while d >= 2.0 * np.pi - 1e-12:
            d -= 2.0 * np.pi
        if d < 0:
            d = 0.0
        return -1e-9 <= d <= abs(sw) + 1e-9
    d = arc.a0 - ang
    while d < -1e-12:
        d += 2.0 * np.pi
    while d >= 2.0 * np.pi - 1e-12:
        d -= 2.0 * np.pi
    if d < 0:
        d = 0.0
    return -1e-9 <= d <= abs(sw) + 1e-9


def _point_on_line_seg(ln: LineEntity, pt: np.ndarray, *, tol: float = 1e-6) -> bool:
    t = _line_t(ln, pt)
    if t < -PARAM_EPS or t > 1.0 + PARAM_EPS:
        return False
    proj = _line_point(ln, float(np.clip(t, 0.0, 1.0)))
    return float(np.linalg.norm(pt - proj)) <= tol


def _intersections_on_target(
    target: SketchEntity, other: SketchEntity
) -> List[float]:
    """Return sorted unique param t on ``target`` where it meets ``other``."""
    out: List[float] = []

    def _add(t: float) -> None:
        if isinstance(target, CircleEntity):
            t = t % 1.0
            if t < 0:
                t += 1.0
        else:
            if t < -PARAM_EPS or t > 1.0 + PARAM_EPS:
                return
            t = float(np.clip(t, 0.0, 1.0))
        for u in out:
            if abs(u - t) < 1e-7 or (
                isinstance(target, CircleEntity)
                and (abs(u - t) < 1e-7 or abs(abs(u - t) - 1.0) < 1e-7)
            ):
                return
        out.append(t)

    # --- line target ---
    if isinstance(target, LineEntity):
        a0 = np.array(target.p0, dtype=np.float64)
        a1 = np.array(target.p1, dtype=np.float64)
        if isinstance(other, LineEntity):
            b0 = np.array(other.p0, dtype=np.float64)
            b1 = np.array(other.p1, dtype=np.float64)
            hit = _seg_seg_t(a0, a1, b0, b1)
            if hit is not None:
                ta, tb = hit
                if -PARAM_EPS <= ta <= 1.0 + PARAM_EPS and -PARAM_EPS <= tb <= 1.0 + PARAM_EPS:
                    _add(ta)
        elif isinstance(other, CircleEntity):
            for ta in _line_circle_ts(a0, a1, np.array(other.center, float), other.radius):
                if -PARAM_EPS <= ta <= 1.0 + PARAM_EPS:
                    _add(ta)
        elif isinstance(other, ArcEntity):
            for ta in _line_circle_ts(
                a0, a1, np.array(other.center, float), other.radius
            ):
                if -PARAM_EPS <= ta <= 1.0 + PARAM_EPS:
                    pt = _line_point(target, float(np.clip(ta, 0, 1)))
                    ang = float(np.atan2(pt[1] - other.center[1], pt[0] - other.center[0]))
                    if _angle_on_arc(other, ang):
                        _add(ta)
        elif isinstance(other, RectEntity):
            cs = other.corners()
            for i in range(4):
                b0 = np.array(cs[i], float)
                b1 = np.array(cs[(i + 1) % 4], float)
                hit = _seg_seg_t(a0, a1, b0, b1)
                if hit is not None:
                    ta, tb = hit
                    if -PARAM_EPS <= ta <= 1.0 + PARAM_EPS and -PARAM_EPS <= tb <= 1.0 + PARAM_EPS:
                        _add(ta)

    # --- circle target ---
    elif isinstance(target, CircleEntity):
        c = np.array(target.center, float)
        r = float(target.radius)
        if isinstance(other, LineEntity):
            b0 = np.array(other.p0, float)
            b1 = np.array(other.p1, float)
            for tb in _line_circle_ts(b0, b1, c, r):
                if -PARAM_EPS <= tb <= 1.0 + PARAM_EPS:
                    pt = b0 + float(np.clip(tb, 0, 1)) * (b1 - b0)
                    _add(_circle_t(target, pt))
        elif isinstance(other, CircleEntity):
            # two-circle intersections
            c2 = np.array(other.center, float)
            r2 = float(other.radius)
            dvec = c2 - c
            dist = float(np.linalg.norm(dvec))
            if dist < 1e-12 or dist > r + r2 + 1e-9 or dist < abs(r - r2) - 1e-9:
                pass
            else:
                a = (r * r - r2 * r2 + dist * dist) / (2 * dist)
                h2 = r * r - a * a
                if h2 >= -1e-12:
                    h = float(np.sqrt(max(0.0, h2)))
                    mid = c + a * dvec / dist
                    perp = np.array([-dvec[1], dvec[0]], float) / dist
                    for s in (-1.0, 1.0):
                        pt = mid + s * h * perp
                        _add(_circle_t(target, pt))
        elif isinstance(other, ArcEntity):
            # treat arc as circle then filter
            c2 = np.array(other.center, float)
            r2 = float(other.radius)
            dvec = c2 - c
            dist = float(np.linalg.norm(dvec))
            if not (dist < 1e-12 or dist > r + r2 + 1e-9 or dist < abs(r - r2) - 1e-9):
                a = (r * r - r2 * r2 + dist * dist) / (2 * dist)
                h2 = r * r - a * a
                if h2 >= -1e-12:
                    h = float(np.sqrt(max(0.0, h2)))
                    mid = c + a * dvec / dist
                    perp = np.array([-dvec[1], dvec[0]], float) / dist
                    for s in (-1.0, 1.0):
                        pt = mid + s * h * perp
                        ang = float(np.atan2(pt[1] - other.center[1], pt[0] - other.center[0]))
                        if _angle_on_arc(other, ang):
                            _add(_circle_t(target, pt))
        elif isinstance(other, RectEntity):
            cs = other.corners()
            for i in range(4):
                b0 = np.array(cs[i], float)
                b1 = np.array(cs[(i + 1) % 4], float)
                for tb in _line_circle_ts(b0, b1, c, r):
                    if -PARAM_EPS <= tb <= 1.0 + PARAM_EPS:
                        pt = b0 + float(np.clip(tb, 0, 1)) * (b1 - b0)
                        _add(_circle_t(target, pt))

    # --- arc target ---
    elif isinstance(target, ArcEntity):
        c = np.array(target.center, float)
        r = float(target.radius)
        if isinstance(other, LineEntity):
            b0 = np.array(other.p0, float)
            b1 = np.array(other.p1, float)
            for tb in _line_circle_ts(b0, b1, c, r):
                if -PARAM_EPS <= tb <= 1.0 + PARAM_EPS:
                    pt = b0 + float(np.clip(tb, 0, 1)) * (b1 - b0)
                    ang = float(np.atan2(pt[1] - c[1], pt[0] - c[0]))
                    if _angle_on_arc(target, ang):
                        _add(_arc_t(target, pt))
        elif isinstance(other, (CircleEntity, ArcEntity)):
            c2 = np.array(other.center, float)
            r2 = float(other.radius)
            dvec = c2 - c
            dist = float(np.linalg.norm(dvec))
            if not (dist < 1e-12 or dist > r + r2 + 1e-9 or dist < abs(r - r2) - 1e-9):
                a = (r * r - r2 * r2 + dist * dist) / (2 * dist)
                h2 = r * r - a * a
                if h2 >= -1e-12:
                    h = float(np.sqrt(max(0.0, h2)))
                    mid = c + a * dvec / dist
                    perp = np.array([-dvec[1], dvec[0]], float) / dist
                    for s in (-1.0, 1.0):
                        pt = mid + s * h * perp
                        ang = float(np.atan2(pt[1] - c[1], pt[0] - c[0]))
                        if not _angle_on_arc(target, ang):
                            continue
                        if isinstance(other, ArcEntity):
                            ang2 = float(
                                np.atan2(pt[1] - other.center[1], pt[0] - other.center[0])
                            )
                            if not _angle_on_arc(other, ang2):
                                continue
                        _add(_arc_t(target, pt))
        elif isinstance(other, LineEntity):
            pass  # handled above
        elif isinstance(other, RectEntity):
            cs = other.corners()
            for i in range(4):
                b0 = np.array(cs[i], float)
                b1 = np.array(cs[(i + 1) % 4], float)
                for tb in _line_circle_ts(b0, b1, c, r):
                    if -PARAM_EPS <= tb <= 1.0 + PARAM_EPS:
                        pt = b0 + float(np.clip(tb, 0, 1)) * (b1 - b0)
                        ang = float(np.atan2(pt[1] - c[1], pt[0] - c[0]))
                        if _angle_on_arc(target, ang):
                            _add(_arc_t(target, pt))

    out.sort()
    return out


def collect_cut_params(sk: Sketch, target: SketchEntity) -> List[float]:
    """All intersection parameters on target with every other entity."""
    cuts: List[float] = []
    for e in sk.entities:
        if e.id == target.id:
            continue
        for t in _intersections_on_target(target, e):
            # drop near-duplicates
            if any(abs(t - u) < 1e-7 for u in cuts):
                continue
            # for open entities, ignore pure endpoints as "cuts" for interval
            # splitting — they still bound the entity
            cuts.append(t)
    cuts.sort()
    return cuts


# ---------------------------------------------------------------------------
# Trim
# ---------------------------------------------------------------------------


def trim_entity_at(sk: Sketch, target: SketchEntity, uv: Vec2) -> bool:
    """SolidWorks-style trim: remove the segment under the click.

    Returns True if geometry changed.
    * End piece → shrink entity to the cut intersection.
    * Middle piece → replace with two entities (count +1).
    * Whole entity with no interior cuts → delete entity.
    """
    if not isinstance(target, (LineEntity, ArcEntity, CircleEntity)):
        return False
    t_click = entity_param_at(target, uv)
    cuts = collect_cut_params(sk, target)

    if isinstance(target, CircleEntity):
        return _trim_circle(sk, target, t_click, cuts)
    if isinstance(target, ArcEntity):
        return _trim_arc(sk, target, t_click, cuts)
    return _trim_line(sk, target, t_click, cuts)


def _trim_line(sk: Sketch, ln: LineEntity, t_click: float, cuts: List[float]) -> bool:
    # Interior cuts only (strictly between 0 and 1)
    interior = [t for t in cuts if PARAM_EPS < t < 1.0 - PARAM_EPS]
    bounds = [0.0] + interior + [1.0]
    # Unique bounds
    uniq: List[float] = []
    for t in bounds:
        if not uniq or abs(t - uniq[-1]) > 1e-7:
            uniq.append(t)
    if len(uniq) < 2:
        return False

    # Find interval containing click
    idx = None
    for i in range(len(uniq) - 1):
        lo, hi = uniq[i], uniq[i + 1]
        if lo - 1e-9 <= t_click <= hi + 1e-9:
            # prefer interior if on boundary of two
            if abs(t_click - lo) < 1e-7 and i > 0:
                # on cut: choose interval closer to original mid-of-interval
                # SolidWorks removes the segment you're "on"; pick smaller residual to mid
                mid_prev = 0.5 * (uniq[i - 1] + lo)
                mid_next = 0.5 * (lo + hi)
                idx = i - 1 if abs(t_click - mid_prev) <= abs(t_click - mid_next) else i
            else:
                idx = i
            break
    if idx is None:
        # nearest interval
        best = 0
        best_d = 1e9
        for i in range(len(uniq) - 1):
            mid = 0.5 * (uniq[i] + uniq[i + 1])
            d = abs(t_click - mid)
            if d < best_d:
                best_d = d
                best = i
        idx = best

    lo, hi = uniq[idx], uniq[idx + 1]
    # Degenerate interval
    if hi - lo < 1e-9:
        return False

    # No interior cuts → whole line is one piece: delete it
    if len(interior) == 0:
        sk.remove_entity(ln.id)
        sk.remove_dimensions_for_entity(ln.id)
        return True

    # End piece at start: keep [hi, 1]
    if abs(lo) < 1e-9 and hi < 1.0 - 1e-9:
        pt = _line_point(ln, hi)
        ln.p0 = (float(pt[0]), float(pt[1]))
        return True
    # End piece at end: keep [0, lo]
    if abs(hi - 1.0) < 1e-9 and lo > 1e-9:
        pt = _line_point(ln, lo)
        ln.p1 = (float(pt[0]), float(pt[1]))
        return True
    # Middle piece: keep [0, lo] and [hi, 1]
    p_lo = _line_point(ln, lo)
    p_hi = _line_point(ln, hi)
    p_end = tuple(ln.p1)
    ln.p1 = (float(p_lo[0]), float(p_lo[1]))
    # Second piece
    sk.add_line((float(p_hi[0]), float(p_hi[1])), p_end)
    # drop dims that no longer make sense on original
    return True


def _trim_arc(sk: Sketch, arc: ArcEntity, t_click: float, cuts: List[float]) -> bool:
    interior = [t for t in cuts if PARAM_EPS < t < 1.0 - PARAM_EPS]
    bounds = [0.0] + interior + [1.0]
    uniq: List[float] = []
    for t in bounds:
        if not uniq or abs(t - uniq[-1]) > 1e-7:
            uniq.append(t)
    if len(uniq) < 2:
        return False
    idx = None
    for i in range(len(uniq) - 1):
        lo, hi = uniq[i], uniq[i + 1]
        if lo - 1e-9 <= t_click <= hi + 1e-9:
            idx = i
            break
    if idx is None:
        idx = int(np.argmin([abs(t_click - 0.5 * (uniq[i] + uniq[i + 1])) for i in range(len(uniq) - 1)]))
    lo, hi = uniq[idx], uniq[idx + 1]
    if hi - lo < 1e-9:
        return False

    if len(interior) == 0:
        sk.remove_entity(arc.id)
        sk.remove_dimensions_for_entity(arc.id)
        return True

    def _arc_from_t_range(t0: float, t1: float) -> Optional[Tuple]:
        if t1 - t0 < 1e-9:
            return None
        p0 = _arc_point(arc, t0)
        mid = _arc_point(arc, 0.5 * (t0 + t1))
        p1 = _arc_point(arc, t1)
        return arc_from_three_points(
            (float(p0[0]), float(p0[1])),
            (float(mid[0]), float(mid[1])),
            (float(p1[0]), float(p1[1])),
        )

    # End at start: keep [hi, 1]
    if abs(lo) < 1e-9 and hi < 1.0 - 1e-9:
        built = _arc_from_t_range(hi, 1.0)
        if built is None:
            return False
        c, r, a0, a1, ccw = built
        arc.center, arc.radius, arc.a0, arc.a1, arc.ccw = c, r, a0, a1, ccw
        return True
    # End at end: keep [0, lo]
    if abs(hi - 1.0) < 1e-9 and lo > 1e-9:
        built = _arc_from_t_range(0.0, lo)
        if built is None:
            return False
        c, r, a0, a1, ccw = built
        arc.center, arc.radius, arc.a0, arc.a1, arc.ccw = c, r, a0, a1, ccw
        return True
    # Middle: keep [0,lo] and [hi,1]
    built_a = _arc_from_t_range(0.0, lo)
    built_b = _arc_from_t_range(hi, 1.0)
    if built_a is None or built_b is None:
        return False
    c, r, a0, a1, ccw = built_a
    arc.center, arc.radius, arc.a0, arc.a1, arc.ccw = c, r, a0, a1, ccw
    c2, r2, a02, a12, ccw2 = built_b
    sk.entities.append(
        ArcEntity(
            id=sk._next_entity_id,
            kind=EntityKind.ARC,
            center=c2,
            radius=r2,
            a0=a02,
            a1=a12,
            ccw=ccw2,
            construction=bool(getattr(arc, "construction", False)),
        )
    )
    sk._next_entity_id += 1
    return True


def _trim_circle(sk: Sketch, circ: CircleEntity, t_click: float, cuts: List[float]) -> bool:
    """Trim on a full circle: remove the arc sector under the click → becomes ArcEntity."""
    # Need at least 2 distinct cut angles
    uniq: List[float] = []
    for t in sorted(c % 1.0 for c in cuts):
        if t < 0:
            t += 1.0
        if not uniq or all(
            min(abs(t - u), 1.0 - abs(t - u)) > 1e-7 for u in uniq
        ):
            uniq.append(t)
    if len(uniq) < 2:
        # No proper scissors — delete whole circle (SW power-trim style when no bounds)
        sk.remove_entity(circ.id)
        sk.remove_dimensions_for_entity(circ.id)
        return True

    # Circular intervals between consecutive cuts (+ wrap)
    bounds = list(uniq)
    # Find sector containing t_click
    t = t_click % 1.0
    n = len(bounds)
    # Order bounds
    bounds.sort()
    sector = None  # (lo, hi) in unwrapped sense; hi may be lo+span wrapping
    for i in range(n):
        lo = bounds[i]
        hi = bounds[(i + 1) % n]
        if lo < hi:
            if lo - 1e-9 <= t <= hi + 1e-9:
                sector = (lo, hi, False)
                break
        else:
            # wrap
            if t >= lo - 1e-9 or t <= hi + 1e-9:
                sector = (lo, hi, True)
                break
    if sector is None:
        # nearest
        sector = (bounds[0], bounds[1 % n], bounds[0] > bounds[1 % n])

    lo, hi, wrap = sector
    # Remaining is complement of [lo,hi]
    # Result is one arc from hi to lo (the kept part). If only 2 cuts, one arc left.
    # If more cuts, SW typically only removes one sector and leaves the rest as one
    # open arc when 2 cuts, or multiple arcs when more cuts.
    # Simplest correct SW behavior with 2 cuts: circle → one arc (kept).
    # With >2 cuts and middle sector: leave multiple arcs.

    # Build list of kept intervals (all except the removed one)
    intervals = []
    for i in range(n):
        a = bounds[i]
        b = bounds[(i + 1) % n]
        if abs(a - lo) < 1e-7 and (
            (not wrap and abs(b - hi) < 1e-7)
            or (wrap and abs(b - hi) < 1e-7)
        ):
            continue  # removed
        intervals.append((a, b))

    # If the match failed, remove by index of sector
    if len(intervals) == n:
        # remove first matching by midpoint
        best_i = 0
        best_d = 1e9
        for i in range(n):
            a = bounds[i]
            b = bounds[(i + 1) % n]
            if a <= b:
                mid = 0.5 * (a + b)
            else:
                mid = (0.5 * (a + b + 1.0)) % 1.0
            d = min(abs(t - mid), 1.0 - abs(t - mid))
            if d < best_d:
                best_d = d
                best_i = i
        intervals = []
        for i in range(n):
            if i == best_i:
                continue
            intervals.append((bounds[i], bounds[(i + 1) % n]))

    if not intervals:
        sk.remove_entity(circ.id)
        sk.remove_dimensions_for_entity(circ.id)
        return True

    def _mk_arc(t0: float, t1: float) -> ArcEntity:
        # span from t0 to t1 CCW (may wrap)
        if t1 > t0 + 1e-12:
            p0 = _circle_point(circ, t0)
            mid = _circle_point(circ, 0.5 * (t0 + t1))
            p1 = _circle_point(circ, t1)
        else:
            # wrap: go t0 → 1 → 0 → t1
            span = (1.0 - t0) + t1
            p0 = _circle_point(circ, t0)
            mid_t = (t0 + 0.5 * span) % 1.0
            mid = _circle_point(circ, mid_t)
            p1 = _circle_point(circ, t1)
        built = arc_from_three_points(
            (float(p0[0]), float(p0[1])),
            (float(mid[0]), float(mid[1])),
            (float(p1[0]), float(p1[1])),
        )
        if built is None:
            # fallback angles
            a0 = 2 * np.pi * t0
            a1 = 2 * np.pi * t1
            return ArcEntity(
                id=-1,
                kind=EntityKind.ARC,
                center=circ.center,
                radius=circ.radius,
                a0=a0,
                a1=a1,
                ccw=True,
                construction=bool(getattr(circ, "construction", False)),
            )
        c, r, a0, a1, ccw = built
        return ArcEntity(
            id=-1,
            kind=EntityKind.ARC,
            center=c,
            radius=r,
            a0=a0,
            a1=a1,
            ccw=ccw,
            construction=bool(getattr(circ, "construction", False)),
        )

    # Replace circle with kept arc(s)
    sk.remove_entity(circ.id)
    sk.remove_dimensions_for_entity(circ.id)
    for t0, t1 in intervals:
        # skip zero-length
        span = (t1 - t0) if t1 > t0 else (1.0 - t0 + t1)
        if span < 1e-9:
            continue
        arc = _mk_arc(t0, t1)
        arc.id = sk._next_entity_id
        sk._next_entity_id += 1
        sk.entities.append(arc)
    return True


# ---------------------------------------------------------------------------
# Extend
# ---------------------------------------------------------------------------


def extend_entity_at(sk: Sketch, target: SketchEntity, uv: Vec2) -> bool:
    """Grow the open end until the next intersection with another entity.

    For lines/arcs: extend the free end (prefer end farther from click? SW
    extends the end you pick near — we pick the end that has a forward hit,
    preferring the end closer to the click when both can extend).
    Circles are closed → no extend.
    """
    if isinstance(target, CircleEntity):
        return False
    if isinstance(target, LineEntity):
        return _extend_line(sk, target, uv)
    if isinstance(target, ArcEntity):
        return _extend_arc(sk, target, uv)
    return False


def _ray_hits_on_line(
    sk: Sketch, origin: np.ndarray, direction: np.ndarray, exclude_id: int
) -> List[Tuple[float, np.ndarray]]:
    """Hits along ray origin + s*direction for s > 0, as (s, point)."""
    dhat = direction.astype(np.float64)
    n = float(np.linalg.norm(dhat))
    if n < 1e-12:
        return []
    dhat = dhat / n
    # Use a long segment for intersection tests
    far = origin + dhat * 1e6
    hits: List[Tuple[float, np.ndarray]] = []
    # Fake line entity along ray for reusing intersection code
    ray = LineEntity(
        id=-1,
        kind=EntityKind.LINE,
        p0=(float(origin[0]), float(origin[1])),
        p1=(float(far[0]), float(far[1])),
    )
    for e in sk.entities:
        if e.id == exclude_id:
            continue
        for t in _intersections_on_target(ray, e):
            if t < 1e-8:  # at origin
                continue
            pt = origin + t * (far - origin)
            s = float(np.dot(pt - origin, dhat))
            if s > 1e-8:
                hits.append((s, pt))
    hits.sort(key=lambda x: x[0])
    return hits


def _extend_line(sk: Sketch, ln: LineEntity, uv: Vec2) -> bool:
    p0 = np.array(ln.p0, dtype=np.float64)
    p1 = np.array(ln.p1, dtype=np.float64)
    d = p1 - p0
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        return False
    dhat = d / n
    click = np.array([float(uv[0]), float(uv[1])], dtype=np.float64)
    # Distance of click to each end
    dist0 = float(np.linalg.norm(click - p0))
    dist1 = float(np.linalg.norm(click - p1))

    # Candidates: extend p1 along +dhat, extend p0 along -dhat
    cand: List[Tuple[str, float, np.ndarray, float]] = []  # end, s, pt, click_dist
    for end, origin, direction, cdist in (
        ("p1", p1, dhat, dist1),
        ("p0", p0, -dhat, dist0),
    ):
        hits = _ray_hits_on_line(sk, origin, direction, ln.id)
        if hits:
            s, pt = hits[0]
            cand.append((end, s, pt, cdist))

    if not cand:
        return False
    # Prefer end closer to click among those that hit
    cand.sort(key=lambda x: x[3])
    end, _s, pt, _ = cand[0]
    if end == "p1":
        ln.p1 = (float(pt[0]), float(pt[1]))
    else:
        ln.p0 = (float(pt[0]), float(pt[1]))
    return True


def _extend_arc(sk: Sketch, arc: ArcEntity, uv: Vec2) -> bool:
    """Extend arc end by growing along the circle until next hit."""
    # Directions of increasing / decreasing sweep at each end
    # Travel from p0→p1; extend p1 further along travel, or p0 opposite travel
    c = np.array(arc.center, dtype=np.float64)
    r = float(arc.radius)
    click = np.array([float(uv[0]), float(uv[1])], dtype=np.float64)
    p0 = np.array(arc.p0(), dtype=np.float64)
    p1 = np.array(arc.p1(), dtype=np.float64)
    dist0 = float(np.linalg.norm(click - p0))
    dist1 = float(np.linalg.norm(click - p1))

    # Sample many angles outside current sweep on the same circle, find
    # intersections with other entities, pick nearest angle extension.
    def _hits_extending(from_end: str) -> Optional[Tuple[float, float]]:
        """Return (new_a0 or new_a1 angle, distance along arc) or None."""
        # Full circle other intersections as angles
        angles: List[float] = []
        phantom = CircleEntity(
            id=-1, kind=EntityKind.CIRCLE, center=arc.center, radius=arc.radius
        )
        for e in sk.entities:
            if e.id == arc.id:
                continue
            for t in _intersections_on_target(phantom, e):
                a = 2.0 * np.pi * (t % 1.0)
                angles.append(a)
        if not angles:
            return None
        # Current sweep as CCW range from a0
        # Normalize angles relative to start of extension
        if from_end == "p1":
            start = float(arc.a1)
            # walk in travel direction
            sense = 1.0 if arc.ccw else -1.0
        else:
            start = float(arc.a0)
            sense = -1.0 if arc.ccw else 1.0
        best = None
        best_d = 1e9
        for a in angles:
            # CCW delta from start to a
            dccw = a - start
            while dccw < 0:
                dccw += 2.0 * np.pi
            while dccw >= 2.0 * np.pi:
                dccw -= 2.0 * np.pi
            dcw = (2.0 * np.pi) - dccw
            if sense > 0:
                dist = dccw
            else:
                dist = dcw
            if dist < 1e-8 or dist > 2.0 * np.pi - 1e-8:
                continue
            # Don't stop inside existing sweep (except tiny)
            # For extend we only want outside current body
            if dist < best_d:
                best_d = dist
                best = a
        if best is None:
            return None
        return best, best_d

    candidates = []
    for end, cdist in (("p1", dist1), ("p0", dist0)):
        hit = _hits_extending(end)
        if hit is not None:
            candidates.append((end, hit[0], hit[1], cdist))
    if not candidates:
        return False
    candidates.sort(key=lambda x: x[3])
    end, ang, _dist, _ = candidates[0]
    if end == "p1":
        arc.a1 = float(ang)
    else:
        arc.a0 = float(ang)
    return True


# Legacy names used by older tests — redirect to intersection-aware API
def trim_line_at(ln: LineEntity, uv: Vec2, keep_side: str = "near") -> None:
    """Deprecated: cursor cut. Prefer ``trim_entity_at(sk, ln, uv)``."""
    p0 = np.array(ln.p0, dtype=np.float64)
    p1 = np.array(ln.p1, dtype=np.float64)
    d = p1 - p0
    n2 = float(np.dot(d, d))
    if n2 < 1e-24:
        return
    pt = np.array([float(uv[0]), float(uv[1])], dtype=np.float64)
    t = float(np.clip(np.dot(pt - p0, d) / n2, 0.0, 1.0))
    cut = p0 + t * d
    if t < 0.5:
        ln.p0 = (float(cut[0]), float(cut[1]))
    else:
        ln.p1 = (float(cut[0]), float(cut[1]))


def extend_line_to_point(ln: LineEntity, uv: Vec2, *, free_end: str = "p1") -> None:
    """Deprecated cursor extend. Prefer ``extend_entity_at``."""
    p0 = np.array(ln.p0, dtype=np.float64)
    p1 = np.array(ln.p1, dtype=np.float64)
    d = p1 - p0
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        return
    dhat = d / n
    pt = np.array([float(uv[0]), float(uv[1])], dtype=np.float64)
    if free_end == "p0":
        t = float(np.dot(pt - p1, -dhat))
        ln.p0 = (float(p1[0] - dhat[0] * t), float(p1[1] - dhat[1] * t))
    else:
        t = float(np.dot(pt - p0, dhat))
        ln.p1 = (float(p0[0] + dhat[0] * t), float(p0[1] + dhat[1] * t))


def entity_dof_status(sk: Sketch, ent: SketchEntity) -> str:
    """Motion-based DOF status (fully defined = cannot move). See cadcore.dof."""
    from cadcore.dof import entity_dof_status as _status

    return _status(sk, ent)


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
