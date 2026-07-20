"""Solid edge chamfer (equal-distance) via CSG — companion to edge_fillet.

Removes a square-section prism along a convex edge so both faces lose
``distance`` material (SolidWorks equal-distance chamfer for ~90° corners).
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from cadcore.edge_fillet import (
    SolidEdge,
    _box_corners,
    _norm,
    _v3,
    edges_from_keys,
    extract_convex_edges,
)
from cadcore.mesh import BooleanOp, Mesh, boolean_op


def chamfer_one_edge(body: Mesh, edge: SolidEdge, distance: float) -> Mesh:
    """Apply an equal-distance convex edge chamfer; raises ValueError if impossible."""
    d = float(distance)
    if not np.isfinite(d) or d <= 1e-12:
        raise ValueError("chamfer distance must be positive")
    p0, p1 = _v3(edge.p0), _v3(edge.p1)
    n0, n1 = _norm(_v3(edge.n0)), _norm(_v3(edge.n1))
    e = _norm(p1 - p0)
    length = float(np.linalg.norm(p1 - p0))
    if length < 1e-9:
        raise ValueError("edge is too short to chamfer")
    u = -n0
    v = -n1
    u = u - e * float(np.dot(u, e))
    v = v - e * float(np.dot(v, e))
    u, v = _norm(u), _norm(v)
    if float(np.linalg.norm(u)) < 0.5 or float(np.linalg.norm(v)) < 0.5:
        raise ValueError("edge normals are parallel to the edge (cannot chamfer)")
    ang = float(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0)))
    if ang < np.radians(15.0) or ang > np.radians(165.0):
        raise ValueError(
            f"edge angle {np.degrees(ang):.1f}° is not suitable for this chamfer"
        )
    bb = body.vertices.max(axis=0) - body.vertices.min(axis=0)
    max_d = 0.49 * float(min(bb[bb > 1e-9])) if np.any(bb > 1e-9) else d
    if d >= max_d - 1e-12:
        raise ValueError(
            f"chamfer distance {d:g} is too large for the part (max ~{max_d:g})"
        )
    prism = _box_corners(p0, u, v, e, d, length)
    try:
        result = boolean_op(body, prism, BooleanOp.DIFFERENCE)
    except Exception as exc:
        raise ValueError(f"chamfer boolean failed: {exc}") from exc
    if result.empty:
        raise ValueError("chamfer removed the entire solid")
    if not result.is_watertight():
        raise ValueError("chamfer result is not watertight")
    if abs(result.volume() - body.volume()) < 1e-6 * max(abs(body.volume()), 1.0):
        raise ValueError("chamfer did not change the solid (check edge selection)")
    if result.volume() >= body.volume() - 1e-9:
        raise ValueError("chamfer did not remove material")
    return result


def chamfer_edges(
    body: Mesh,
    edges: Sequence[SolidEdge],
    distance: float,
) -> Mesh:
    """Chamfer several edges sequentially. Fails entirely if any edge fails."""
    if not edges:
        raise ValueError("no edges selected to chamfer")
    out = body
    for e in edges:
        out = chamfer_one_edge(out, e, distance)
    return out


__all__ = [
    "SolidEdge",
    "chamfer_one_edge",
    "chamfer_edges",
    "edges_from_keys",
    "extract_convex_edges",
]
