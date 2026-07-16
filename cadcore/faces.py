"""Planar face extraction from triangle meshes (for sketch-on-face)."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from cadcore.sketch import PlaneFrame

Vec3 = Tuple[float, float, float]


def _as_pts(vertices) -> np.ndarray:
    return np.asarray(vertices, dtype=np.float64).reshape(-1, 3)


def _as_faces(faces) -> np.ndarray:
    return np.asarray(faces, dtype=np.int32).reshape(-1, 3)


def triangle_normal(v0, v1, v2) -> np.ndarray:
    n = np.cross(np.asarray(v1) - np.asarray(v0), np.asarray(v2) - np.asarray(v0))
    ln = float(np.linalg.norm(n))
    if ln < 1e-15:
        return np.array([0.0, 0.0, 1.0])
    return n / ln


def closest_triangle(
    vertices: np.ndarray,
    faces: np.ndarray,
    point: Sequence[float],
) -> int:
    """Index of triangle whose centroid is closest to ``point``."""
    p = np.asarray(point, dtype=np.float64).reshape(3)
    v = _as_pts(vertices)
    f = _as_faces(faces)
    if len(f) == 0:
        return -1
    cents = (v[f[:, 0]] + v[f[:, 1]] + v[f[:, 2]]) / 3.0
    d2 = np.sum((cents - p) ** 2, axis=1)
    return int(np.argmin(d2))


def _build_edge_map(faces: np.ndarray) -> dict:
    """Undirected edge → list of face indices."""
    em: dict[tuple[int, int], list[int]] = {}
    for fi, (a, b, c) in enumerate(faces):
        for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            key = (u, v) if u < v else (v, u)
            em.setdefault(key, []).append(fi)
    return em


def coplanar_face_indices(
    vertices: np.ndarray,
    faces: np.ndarray,
    seed: int,
    *,
    angle_tol_deg: float = 2.0,
    dist_tol: Optional[float] = None,
) -> list[int]:
    """Flood-fill triangles coplanar and connected to ``seed``."""
    v = _as_pts(vertices)
    f = _as_faces(faces)
    if seed < 0 or seed >= len(f):
        return []
    n0 = triangle_normal(v[f[seed, 0]], v[f[seed, 1]], v[f[seed, 2]])
    p0 = v[f[seed, 0]]
    # Distance tolerance scales with triangle size if not given
    if dist_tol is None:
        e = float(np.linalg.norm(v[f[seed, 1]] - v[f[seed, 0]]))
        dist_tol = max(1e-6, e * 1e-4, 1e-4)
    cos_tol = float(np.cos(np.radians(angle_tol_deg)))
    em = _build_edge_map(f)
    seen = {seed}
    stack = [seed]
    while stack:
        fi = stack.pop()
        a, b, c = int(f[fi, 0]), int(f[fi, 1]), int(f[fi, 2])
        for u, vv in ((a, b), (b, c), (c, a)):
            key = (u, vv) if u < vv else (vv, u)
            for nb in em.get(key, ()):
                if nb in seen:
                    continue
                n1 = triangle_normal(v[f[nb, 0]], v[f[nb, 1]], v[f[nb, 2]])
                if abs(float(np.dot(n0, n1))) < cos_tol:
                    continue
                # Same plane: point of neighbour on seed plane
                q = v[f[nb, 0]]
                if abs(float(np.dot(n0, q - p0))) > dist_tol:
                    continue
                # Same orientation (not back-face of a thin wall)
                if float(np.dot(n0, n1)) < 0:
                    continue
                seen.add(nb)
                stack.append(nb)
    return sorted(seen)


def plane_frame_from_face(
    vertices,
    faces,
    pick_point: Sequence[float],
    *,
    cell_id: Optional[int] = None,
) -> PlaneFrame:
    """Build a ``PlaneFrame`` for the planar face under ``pick_point``.

    Normal is outward (away from the mesh centroid). Origin is the projection of
    the pick point onto the face plane (so the sketch sits on the surface).
    U/V are orthonormal, with U preferring world +X projected onto the plane.
    """
    v = _as_pts(vertices)
    f = _as_faces(faces)
    if len(f) == 0 or len(v) == 0:
        return PlaneFrame.from_plane_type("PLANE_FRONT")
    seed = int(cell_id) if cell_id is not None and 0 <= int(cell_id) < len(f) else closest_triangle(
        v, f, pick_point
    )
    if seed < 0:
        return PlaneFrame.from_plane_type("PLANE_FRONT")
    idxs = coplanar_face_indices(v, f, seed)
    if not idxs:
        idxs = [seed]
    # Average normal of region
    norms = []
    pts_face = []
    for fi in idxs:
        a, b, c = f[fi]
        norms.append(triangle_normal(v[a], v[b], v[c]))
        pts_face.extend([v[a], v[b], v[c]])
    n = np.mean(np.stack(norms, axis=0), axis=0)
    ln = float(np.linalg.norm(n))
    n = n / ln if ln > 1e-15 else np.array([0.0, 0.0, 1.0])
    # Outward: face centroid vs solid centroid
    face_pts = np.stack(pts_face, axis=0)
    face_c = face_pts.mean(axis=0)
    solid_c = v.mean(axis=0)
    if float(np.dot(n, face_c - solid_c)) < 0:
        n = -n
    # Origin = pick projected onto plane
    p = np.asarray(pick_point, dtype=np.float64).reshape(3)
    origin = p - n * float(np.dot(p - face_c, n))
    # Prefer projecting world X; fallback to Y
    x_ref = np.array([1.0, 0.0, 0.0])
    u = x_ref - n * float(np.dot(x_ref, n))
    if float(np.linalg.norm(u)) < 1e-8:
        y_ref = np.array([0.0, 1.0, 0.0])
        u = y_ref - n * float(np.dot(y_ref, n))
    u = u / float(np.linalg.norm(u))
    v_ax = np.cross(n, u)
    v_ax = v_ax / float(np.linalg.norm(v_ax))
    # Re-orthogonalize u = v × n wait: right-handed: u, v, n with v = n × u
    v_ax = np.cross(n, u)
    v_ax = v_ax / float(np.linalg.norm(v_ax))
    u = np.cross(v_ax, n)
    u = u / float(np.linalg.norm(u))
    return PlaneFrame(
        origin=origin.astype(np.float64),
        u_axis=u.astype(np.float64),
        v_axis=v_ax.astype(np.float64),
        normal=n.astype(np.float64),
    )


def face_bounds_uv(frame: PlaneFrame, vertices, faces, pick_point, *, cell_id=None):
    """Axis-aligned UV bounds of the coplanar face (for grid sizing)."""
    v = _as_pts(vertices)
    f = _as_faces(faces)
    seed = int(cell_id) if cell_id is not None and 0 <= int(cell_id) < len(f) else closest_triangle(
        v, f, pick_point
    )
    idxs = coplanar_face_indices(v, f, seed) if seed >= 0 else []
    uvs = []
    for fi in idxs:
        for vi in f[fi]:
            uvs.append(frame.to_local(v[int(vi)]))
    if not uvs:
        return (0.0, 0.0, 0.0, 0.0)
    us = [p[0] for p in uvs]
    vs = [p[1] for p in uvs]
    return (min(us), max(us), min(vs), max(vs))
