"""Solid edge fillet via CSG (convex planar edges).

For each selected convex edge between two planar faces, remove the sharp
corner prism and leave a cylindrical blend — volume decreases by
approximately L·r²·(1 − π/4) on a 90° edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from cadcore.mesh import Mesh, BooleanOp, boolean_op

Vec3 = Tuple[float, float, float]
EDGE_KEY_DECIMALS = 5


@dataclass(frozen=True)
class SolidEdge:
    """A sharp edge of a solid, with outward normals of the two adjacent faces."""

    p0: Vec3
    p1: Vec3
    n0: Vec3  # outward normal of face 0
    n1: Vec3  # outward normal of face 1

    def key(self) -> str:
        a = tuple(round(float(x), EDGE_KEY_DECIMALS) for x in self.p0)
        b = tuple(round(float(x), EDGE_KEY_DECIMALS) for x in self.p1)
        if a > b:
            a, b = b, a
        return f"{a[0]},{a[1]},{a[2]}|{b[0]},{b[1]},{b[2]}"

    def length(self) -> float:
        return float(np.linalg.norm(np.subtract(self.p1, self.p0)))


def _v3(p) -> np.ndarray:
    return np.asarray(p, dtype=np.float64).reshape(3)


def _norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-15:
        return v * 0.0
    return v / n


def _tri_normal(v0, v1, v2) -> np.ndarray:
    n = np.cross(v1 - v0, v2 - v0)
    return _norm(n)


def extract_convex_edges(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    angle_tol_deg: float = 5.0,
) -> List[SolidEdge]:
    """Undirected mesh edges between faces meeting at a convex exterior angle."""
    verts = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    tris = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    if len(tris) == 0:
        return []

    # edge key (i,j) i<j -> list of (face_index, oriented as (i->j or not))
    from collections import defaultdict

    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    face_normals = []
    for fi, (a, b, c) in enumerate(tris):
        n = _tri_normal(verts[a], verts[b], verts[c])
        face_normals.append(n)
        for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            key = (u, v) if u < v else (v, u)
            edge_faces[key].append(fi)

    cos_colinear = float(np.cos(np.radians(angle_tol_deg)))
    edges: List[SolidEdge] = []
    seen: set[str] = set()
    centroid = verts.mean(axis=0)

    for (ia, ib), fis in edge_faces.items():
        if len(fis) != 2:
            continue
        f0, f1 = fis[0], fis[1]
        n0, n1 = face_normals[f0], face_normals[f1]
        # Nearly coplanar — not a sharp edge
        if abs(float(np.dot(n0, n1))) > cos_colinear:
            continue
        p0 = verts[ia]
        p1 = verts[ib]
        mid = 0.5 * (p0 + p1)
        # Convex exterior: both outward normals point away from solid centroid-ish
        # For a convex body, n0 and n1 both have positive component away from interior.
        # Interior is toward centroid from mid for a solid containing centroid.
        # Convex edge: the bisector of outward normals points outward.
        bis = _norm(n0 + n1)
        if float(np.dot(bis, mid - centroid)) < 0:
            # flip interpretation — still store outward normals as given by triangles
            pass
        # Dihedral: for convex solid, cross(n0,n1) aligns with edge direction
        e = _norm(p1 - p0)
        # Ensure n0, n1 are outward (dot with mid-centroid > 0 for faces near mid)
        # Sample face centroids
        def face_outward(fi, n):
            a, b, c = tris[fi]
            fc = (verts[a] + verts[b] + verts[c]) / 3.0
            if float(np.dot(n, fc - centroid)) < 0:
                return -n
            return n

        n0 = face_outward(f0, n0)
        n1 = face_outward(f1, n1)
        # Convex if the interior angle < 180°: outward normals diverge
        # (n0 + n1) points roughly outward
        if float(np.dot(_norm(n0 + n1), mid - centroid)) < 0:
            continue  # concave or flat-ish interior

        se = SolidEdge(
            p0=(float(p0[0]), float(p0[1]), float(p0[2])),
            p1=(float(p1[0]), float(p1[1]), float(p1[2])),
            n0=(float(n0[0]), float(n0[1]), float(n0[2])),
            n1=(float(n1[0]), float(n1[1]), float(n1[2])),
        )
        k = se.key()
        if k in seen:
            continue
        seen.add(k)
        edges.append(se)
    return edges


def pick_edge_near_point(
    edges: Sequence[SolidEdge],
    point: Sequence[float],
    *,
    max_dist: float,
) -> Optional[SolidEdge]:
    """Nearest edge to a 3D pick point, or None if farther than max_dist."""
    p = _v3(point)
    best = None
    best_d = float(max_dist)
    for e in edges:
        a, b = _v3(e.p0), _v3(e.p1)
        ab = b - a
        L2 = float(np.dot(ab, ab))
        if L2 < 1e-18:
            continue
        t = float(np.clip(np.dot(p - a, ab) / L2, 0.0, 1.0))
        q = a + t * ab
        d = float(np.linalg.norm(p - q))
        if d < best_d:
            best_d = d
            best = e
    return best


def _cylinder_along_segment(
    p0: np.ndarray,
    p1: np.ndarray,
    radius: float,
    *,
    segments: int = 32,
) -> Mesh:
    """Cylinder of given radius with axis from p0 to p1.

    manifold3d's ``Manifold.cylinder`` is aligned with **+Z** (height along Z
    from 0 to ``height`` when ``center=False``). Map local X/Y radial and
    local Z → edge direction.
    """
    from cadcore.mesh import Manifold, _status_ok

    if Manifold is None:
        raise RuntimeError("manifold3d is not installed")
    axis = p1 - p0
    height = float(np.linalg.norm(axis))
    if height < 1e-12:
        raise ValueError("degenerate edge (zero length)")
    direction = axis / height
    man = Manifold.cylinder(
        height, float(radius), float(radius), max(8, int(segments)), False
    )
    # Local Z = edge direction; X,Y form an orthonormal radial basis
    z = direction
    tmp = np.array([1.0, 0.0, 0.0]) if abs(float(z[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = _norm(np.cross(tmp, z))
    y = _norm(np.cross(z, x))
    # 3×4: columns are mapped basis vectors + origin (same as mesh._frame_transform)
    xf = [
        [float(x[0]), float(y[0]), float(z[0]), float(p0[0])],
        [float(x[1]), float(y[1]), float(z[1]), float(p0[1])],
        [float(x[2]), float(y[2]), float(z[2]), float(p0[2])],
    ]
    man = man.transform(xf)
    if not _status_ok(man.status()) or man.is_empty():
        raise RuntimeError(f"cylinder fillet tool failed: {man.status()}")
    return Mesh.from_manifold(man)


def _box_corners(origin: np.ndarray, u: np.ndarray, v: np.ndarray, e: np.ndarray, r: float, length: float) -> Mesh:
    """Axis-aligned in (u,v,e) frame: square [0,r]x[0,r] extruded along e for length."""
    # 8 corners of the prism
    pts = []
    for iu in (0.0, r):
        for iv in (0.0, r):
            for ie in (0.0, length):
                pts.append(origin + iu * u + iv * v + ie * e)
    pts = np.asarray(pts, dtype=np.float64)
    # Use convex hull via manifold hull of points... Manifold.hull_points or cube transform
    from cadcore.mesh import Manifold, Mesh as M, _status_ok

    # Build as unit cube [0,1]^3 then transform
    # cube(size, center=False) is [0,size] if not center
    cube = Manifold.cube([float(r), float(r), float(length)], False)
    # Map local X->u, Y->v, Z->e, origin
    xf = [
        [float(u[0]), float(v[0]), float(e[0]), float(origin[0])],
        [float(u[1]), float(v[1]), float(e[1]), float(origin[1])],
        [float(u[2]), float(v[2]), float(e[2]), float(origin[2])],
    ]
    cube = cube.transform(xf)
    if not _status_ok(cube.status()) or cube.is_empty():
        raise RuntimeError(f"corner prism failed: {cube.status()}")
    return Mesh.from_manifold(cube)


def fillet_one_edge(body: Mesh, edge: SolidEdge, radius: float, *, segments: int = 32) -> Mesh:
    """Apply a convex edge fillet of ``radius``; raises ValueError if impossible."""
    r = float(radius)
    if not np.isfinite(r) or r <= 1e-12:
        raise ValueError("fillet radius must be positive")
    p0, p1 = _v3(edge.p0), _v3(edge.p1)
    n0, n1 = _norm(_v3(edge.n0)), _norm(_v3(edge.n1))
    e = _norm(p1 - p0)
    length = float(np.linalg.norm(p1 - p0))
    if length < 1e-9:
        raise ValueError("edge is too short to fillet")
    # Local frame: u = -n0 (into material from face0), v = -n1
    # For outward normals, material is opposite outward direction near the face.
    # Corner prism sits in the exterior? Sharp corner of solid is where both
    # faces meet; material is on the side of -n0 and -n1 from the faces.
    # From edge point, into material along face0 is perpendicular to e and into body.
    # Outward n0: into material from edge along face = cross(e, n0) or -n0 projected...
    # For box edge at x=0,y=0 material x>=0,y>=0: faces x=0 (n0=-X), y=0 (n1=-Y).
    # Into material from edge: +X and +Y = -n0 and -n1.
    u = -n0
    v = -n1
    # Orthogonalize u,v in plane perp to e
    u = u - e * float(np.dot(u, e))
    v = v - e * float(np.dot(v, e))
    u, v = _norm(u), _norm(v)
    if float(np.linalg.norm(u)) < 0.5 or float(np.linalg.norm(v)) < 0.5:
        raise ValueError("edge normals are parallel to the edge (cannot fillet)")
    # Angle between u and v should be the exterior material corner (~90° for box)
    ang = float(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0)))
    if ang < np.radians(15.0) or ang > np.radians(165.0):
        raise ValueError(
            f"edge angle {np.degrees(ang):.1f}° is not suitable for this fillet "
            f"(need a clear exterior corner)"
        )
    # Max radius: half min adjacent edge-ish — use body bbox diagonal fraction
    bb = body.vertices.max(axis=0) - body.vertices.min(axis=0)
    max_r = 0.49 * float(min(bb[bb > 1e-9])) if np.any(bb > 1e-9) else r
    if r >= max_r - 1e-12:
        raise ValueError(
            f"fillet radius {r:g} is too large for the part (max ~{max_r:g})"
        )

    # Corner prism at edge, extending into material along u and v by r
    prism = _box_corners(p0, u, v, e, r, length)
    # Cylinder axis at p0 + r*u + r*v, along e
    axis0 = p0 + r * u + r * v
    axis1 = axis0 + e * length
    cyl = _cylinder_along_segment(axis0, axis1, r, segments=segments)
    # Sharp corner to remove = prism - cylinder
    try:
        corner = boolean_op(prism, cyl, BooleanOp.DIFFERENCE)
    except Exception as exc:
        raise ValueError(f"could not build fillet tool: {exc}") from exc
    if corner.empty:
        raise ValueError("fillet tool is empty (radius/edge invalid)")
    try:
        result = boolean_op(body, corner, BooleanOp.DIFFERENCE)
    except Exception as exc:
        raise ValueError(f"fillet boolean failed: {exc}") from exc
    if result.empty:
        raise ValueError("fillet removed the entire solid")
    if not result.is_watertight():
        raise ValueError("fillet result is not watertight")
    # Must remove some material
    if abs(result.volume() - body.volume()) < 1e-6 * max(abs(body.volume()), 1.0):
        raise ValueError("fillet did not change the solid (check edge selection)")
    if result.volume() >= body.volume() - 1e-9:
        # volume should decrease for exterior convex fillet
        raise ValueError("fillet did not remove material")
    return result


def fillet_edges(
    body: Mesh,
    edges: Sequence[SolidEdge],
    radius: float,
    *,
    segments: int = 32,
) -> Mesh:
    """Fillet several edges sequentially. Fails entirely if any edge fails."""
    if not edges:
        raise ValueError("no edges selected to fillet")
    out = body
    for e in edges:
        out = fillet_one_edge(out, e, radius, segments=segments)
    return out


def edges_from_keys(
    all_edges: Sequence[SolidEdge], keys: Sequence[str]
) -> List[SolidEdge]:
    by = {e.key(): e for e in all_edges}
    out = []
    for k in keys:
        if k not in by:
            raise ValueError(f"edge no longer found on solid: {k}")
        out.append(by[k])
    return out
