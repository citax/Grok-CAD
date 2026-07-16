"""Scene / sketch scale helpers — axes, grid, and camera comfort at real sizes.

Internal units are millimetres. These helpers pick "nice" 1–2–5 steps so the
sketch grid stays readable whether the part is a few mm or half a metre.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

# Fallback when the document has no geometry yet (still a readable workspace).
DEFAULT_CHAR_MM = 50.0
# World axes / origin glyph: keep a usable minimum so empty scenes don't vanish.
MIN_AXIS_MM = 8.0
MIN_PLANE_HALF_MM = 5.0


def nice_step(target: float) -> float:
    """Nearest 1–2–5 decade step at or near ``target`` (always > 0)."""
    t = abs(float(target))
    if not math.isfinite(t) or t <= 1e-15:
        return 1.0
    exp = math.floor(math.log10(t))
    base = 10.0**exp
    mantissa = t / base
    if mantissa < 1.5:
        nice = 1.0
    elif mantissa < 3.5:
        nice = 2.0
    elif mantissa < 7.5:
        nice = 5.0
    else:
        nice = 10.0
    return float(nice * base)


def characteristic_length_from_bounds(
    bounds: Optional[Sequence[float]],
    *,
    default: float = DEFAULT_CHAR_MM,
) -> float:
    """Diagonal of an AABB ``(xmin,xmax,ymin,ymax,zmin,zmax)`` → characteristic length."""
    if bounds is None:
        return float(default)
    b = [float(x) for x in bounds]
    if len(b) != 6:
        return float(default)
    dx = abs(b[1] - b[0])
    dy = abs(b[3] - b[2])
    dz = abs(b[5] - b[4])
    diag = math.sqrt(dx * dx + dy * dy + dz * dz)
    if not math.isfinite(diag) or diag < 1e-9:
        # Degenerate / flat: use largest edge
        edge = max(dx, dy, dz)
        return float(default) if edge < 1e-9 else max(edge, 1.0)
    return max(diag, 1.0)


def characteristic_length_from_points(
    points: Iterable[Sequence[float]],
    *,
    default: float = DEFAULT_CHAR_MM,
) -> float:
    pts = np.asarray(list(points), dtype=np.float64).reshape(-1, 3)
    if pts.size == 0:
        return float(default)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    return characteristic_length_from_bounds(
        (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]), default=default
    )


def axis_length_mm(char_mm: float) -> float:
    """World-axis arm length so axes read clearly next to geometry."""
    c = max(float(char_mm), 1.0)
    # ~22% of characteristic size, clamped
    return max(MIN_AXIS_MM, min(c * 0.22, c * 0.5))


def plane_half_mm(char_mm: float) -> float:
    """Reference-plane half-extent (square from -h..+h)."""
    c = max(float(char_mm), 1.0)
    return max(MIN_PLANE_HALF_MM, c * 0.55)


def origin_glyph_sizes(char_mm: float) -> Tuple[float, float]:
    """Return (cross_half, ring_radius) for the origin marker."""
    L = axis_length_mm(char_mm)
    half = max(0.5, L * 0.07)
    ring = max(0.3, L * 0.045)
    return half, ring


def sketch_entity_uv_extent(entities) -> float:
    """Half-diagonal of entity AABB in UV (mm). 0 if empty."""
    us: list[float] = []
    vs: list[float] = []
    for e in entities or ():
        kind = getattr(e, "kind", None)
        name = getattr(kind, "name", "") or type(e).__name__
        if hasattr(e, "p0") and hasattr(e, "p1"):
            us.extend([e.p0[0], e.p1[0]])
            vs.extend([e.p0[1], e.p1[1]])
        elif hasattr(e, "c0") and hasattr(e, "c1"):
            us.extend([e.c0[0], e.c1[0]])
            vs.extend([e.c0[1], e.c1[1]])
        elif hasattr(e, "center") and hasattr(e, "radius"):
            r = float(e.radius)
            us.extend([e.center[0] - r, e.center[0] + r])
            vs.extend([e.center[1] - r, e.center[1] + r])
        else:
            _ = name  # unused
    if not us:
        return 0.0
    du = max(us) - min(us)
    dv = max(vs) - min(vs)
    return 0.5 * math.hypot(du, dv)


def sketch_grid_params(
    view_half_mm: float,
    *,
    entity_extent_mm: float = 0.0,
    min_half: float = 10.0,
) -> Tuple[float, float]:
    """Return ``(grid_half, grid_step)`` for a useful sketch grid.

    * ``view_half_mm`` — orthographic half-height (camera parallel scale).
    * Grid covers a bit more than the view so panning still shows lines.
    * Step is a nice 1–2–5 value yielding roughly 8–20 lines across half-width.
    """
    half = max(float(view_half_mm) * 1.15, float(entity_extent_mm) * 1.4, float(min_half))
    if not math.isfinite(half) or half <= 0:
        half = float(min_half)
    # Aim for ~10 major lines from centre to edge
    step = nice_step(half / 10.0)
    # Avoid microscopic or sparse grids
    n_lines = half / step if step > 0 else 0
    if n_lines > 24:
        step = nice_step(half / 12.0)
    elif n_lines < 4:
        step = nice_step(half / 6.0)
    step = max(step, 1e-6)
    # Snap half to an integer number of steps so edges look clean
    n = max(2, int(math.ceil(half / step)))
    half = n * step
    return float(half), float(step)


def sketch_parallel_scale(
    entity_extent_mm: float,
    *,
    default: float = 40.0,
    margin: float = 1.35,
) -> float:
    """Orthographic parallel scale (half-height) when entering sketch mode."""
    e = float(entity_extent_mm)
    if not math.isfinite(e) or e < 1e-6:
        return float(default)
    return max(float(default) * 0.25, e * float(margin), 5.0)
