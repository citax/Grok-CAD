"""Triangle meshes and watertight CSG via manifold3d."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Sequence, Tuple, Union

import numpy as np

try:
    from manifold3d import CrossSection, Manifold, Mesh as ManifoldMesh
except ImportError:  # pragma: no cover
    CrossSection = None  # type: ignore
    Manifold = None  # type: ignore
    ManifoldMesh = None  # type: ignore

if TYPE_CHECKING:
    from cadcore.sketch import CircleEntity, PlaneFrame, RectEntity, SketchEntity


def _status_ok(st: object) -> bool:
    s = str(st)
    return s in ("Error.NoError", "NoError") or s.endswith("NoError")


@dataclass
class Mesh:
    """Indexed triangle mesh: Nx3 float vertices, Mx3 int triangles."""

    vertices: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float64))
    faces: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.int32))

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float64).reshape(-1, 3)
        self.faces = np.asarray(self.faces, dtype=np.int32).reshape(-1, 3)

    @property
    def empty(self) -> bool:
        return self.faces.size == 0

    def copy(self) -> "Mesh":
        return Mesh(self.vertices.copy(), self.faces.copy())

    def translate(self, offset: Tuple[float, float, float]) -> "Mesh":
        v = self.vertices + np.asarray(offset, dtype=np.float64)
        return Mesh(v, self.faces.copy())

    def volume(self) -> float:
        if self.empty:
            return 0.0
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        return float(np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2))) / 6.0)

    def surface_area(self) -> float:
        if self.empty:
            return 0.0
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        return float(0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).sum())

    def is_watertight(self) -> bool:
        """Closed 2-manifold: every undirected edge used twice with opposite winding."""
        if self.empty or len(self.faces) < 4:
            return False
        from collections import defaultdict

        edge_signed: dict[tuple[int, int], int] = defaultdict(int)
        edge_abs: dict[tuple[int, int], int] = defaultdict(int)
        n = len(self.vertices)
        for a, b, c in self.faces:
            ia, ib, ic = int(a), int(b), int(c)
            if not (0 <= ia < n and 0 <= ib < n and 0 <= ic < n):
                return False
            if ia == ib or ib == ic or ia == ic:
                return False
            for u, v in ((ia, ib), (ib, ic), (ic, ia)):
                key = (u, v) if u < v else (v, u)
                edge_abs[key] += 1
                edge_signed[key] += 1 if u < v else -1
        if any(c != 2 for c in edge_abs.values()):
            return False
        if any(c != 0 for c in edge_signed.values()):
            return False
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        if np.any(areas <= 1e-14):
            return False
        return True

    def to_manifold(self) -> "Manifold":
        if Manifold is None or ManifoldMesh is None:
            raise RuntimeError("manifold3d is not installed")
        mesh = ManifoldMesh(
            vert_properties=np.asarray(self.vertices, dtype=np.float32),
            tri_verts=np.asarray(self.faces, dtype=np.uint32),
        )
        man = Manifold(mesh)
        if not _status_ok(man.status()):
            raise RuntimeError(f"Invalid mesh for manifold: {man.status()}")
        return man

    @staticmethod
    def from_manifold(man: "Manifold") -> "Mesh":
        if not _status_ok(man.status()):
            raise RuntimeError(f"manifold status: {man.status()}")
        m = man.to_mesh()
        verts = np.asarray(m.vert_properties, dtype=np.float64)
        if verts.ndim == 2 and verts.shape[1] > 3:
            verts = verts[:, :3]
        faces = np.asarray(m.tri_verts, dtype=np.int32).reshape(-1, 3)
        out = Mesh(verts, faces)
        if out.volume() < 0:
            out.faces = out.faces[:, ::-1].copy()
        return out

    def manifold_is_solid(self) -> bool:
        """True if manifold3d accepts the mesh and reports NoError + genus defined."""
        try:
            man = self.to_manifold()
        except Exception:
            return False
        if not _status_ok(man.status()):
            return False
        if man.is_empty():
            return False
        # genus() is defined for closed manifolds
        try:
            _ = man.genus()
        except Exception:
            return False
        return True


class BooleanOp(Enum):
    UNION = auto()
    DIFFERENCE = auto()
    INTERSECTION = auto()


def boolean_op(a: Mesh, b: Mesh, op: BooleanOp) -> Mesh:
    """Watertight CSG via manifold3d (``+`` / ``-`` / ``^``)."""
    if Manifold is None:
        raise RuntimeError("manifold3d is not installed")
    ma = a.to_manifold()
    mb = b.to_manifold()
    if op is BooleanOp.UNION:
        result = ma + mb
    elif op is BooleanOp.DIFFERENCE:
        result = ma - mb
    elif op is BooleanOp.INTERSECTION:
        result = ma ^ mb
    else:
        raise ValueError(op)
    if not _status_ok(result.status()):
        raise RuntimeError(f"manifold3d boolean failed: {result.status()}")
    return Mesh.from_manifold(result)


def make_box(width: float, height: float, depth: float) -> Mesh:
    """Axis-aligned box centered at origin (manifold3d cube)."""
    if Manifold is None:
        raise RuntimeError("manifold3d is not installed")
    man = Manifold.cube([float(width), float(height), float(depth)], True)
    return Mesh.from_manifold(man)


def make_sphere(radius: float, segments: int = 32, rings: int = 16) -> Mesh:
    """UV sphere approx via manifold3d (circular segments ~ quality)."""
    if Manifold is None:
        raise RuntimeError("manifold3d is not installed")
    # manifold sphere(radius, circular_segments)
    segs = max(segments, rings * 2, 8)
    man = Manifold.sphere(float(radius), int(segs))
    return Mesh.from_manifold(man)


def make_cylinder(radius: float, height: float, segments: int = 32) -> Mesh:
    """Cylinder along +Y, centered at origin."""
    if Manifold is None:
        raise RuntimeError("manifold3d is not installed")
    # cylinder(height, radius_low, radius_high, circular_segments, center)
    man = Manifold.cylinder(
        float(height), float(radius), float(radius), max(3, int(segments)), True
    )
    return Mesh.from_manifold(man)


def _frame_transform(frame: "PlaneFrame") -> list:
    """3×4 matrix mapping (u, v, h) → origin + u·U + v·V + h·N."""
    u = np.asarray(frame.u_axis, dtype=np.float64).reshape(3)
    v = np.asarray(frame.v_axis, dtype=np.float64).reshape(3)
    n = np.asarray(frame.normal, dtype=np.float64).reshape(3)
    o = np.asarray(frame.origin, dtype=np.float64).reshape(3)
    return [
        [float(u[0]), float(v[0]), float(n[0]), float(o[0])],
        [float(u[1]), float(v[1]), float(n[1]), float(o[1])],
        [float(u[2]), float(v[2]), float(n[2]), float(o[2])],
    ]


def _rect_polygon_uv(c0: Sequence[float], c1: Sequence[float]) -> list:
    """Axis-aligned rectangle corners in CCW order (positive area for CrossSection)."""
    u0, u1 = sorted([float(c0[0]), float(c1[0])])
    v0, v1 = sorted([float(c0[1]), float(c1[1])])
    w = u1 - u0
    h = v1 - v0
    if w <= 1e-12 or h <= 1e-12:
        raise ValueError("degenerate rectangle: zero area")
    return [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]


def _cross_section_from_polygon(poly: Sequence[Tuple[float, float]]) -> "CrossSection":
    if CrossSection is None:
        raise RuntimeError("manifold3d is not installed")
    loop = [[float(p[0]), float(p[1])] for p in poly]
    cs = CrossSection([loop])
    if cs.is_empty() or abs(float(cs.area())) <= 1e-14:
        raise ValueError("degenerate profile: empty or zero-area cross-section")
    return cs


def extrude_polygon(
    polygon_uv: Sequence[Tuple[float, float]],
    distance: float,
    frame: "PlaneFrame",
) -> Mesh:
    """Extrude a closed 2D UV polygon along ``frame.normal`` by ``distance``.

    Uses manifold3d ``CrossSection`` + ``Manifold.extrude`` (not hand-rolled geometry).
    """
    if Manifold is None or CrossSection is None:
        raise RuntimeError("manifold3d is not installed")
    dist = float(distance)
    if not np.isfinite(dist) or dist <= 1e-12:
        raise ValueError("extrude distance must be a positive finite number")
    if len(polygon_uv) < 3:
        raise ValueError("polygon must have at least 3 vertices")
    cs = _cross_section_from_polygon(polygon_uv)
    man = Manifold.extrude(cs, dist)
    if not _status_ok(man.status()) or man.is_empty():
        raise RuntimeError(f"manifold extrude failed: {man.status()}")
    man = man.transform(_frame_transform(frame))
    mesh = Mesh.from_manifold(man)
    if not mesh.is_watertight():
        raise RuntimeError("extrude result is not watertight")
    return mesh


def extrude_rectangle(
    c0: Sequence[float],
    c1: Sequence[float],
    distance: float,
    frame: "PlaneFrame",
) -> Mesh:
    """Extrude an axis-aligned UV rectangle along the plane normal."""
    return extrude_polygon(_rect_polygon_uv(c0, c1), distance, frame)


def extrude_circle(
    center: Sequence[float],
    radius: float,
    distance: float,
    frame: "PlaneFrame",
    *,
    segments: int = 64,
) -> Mesh:
    """Extrude a UV circle along the plane normal (polygonal approx)."""
    if Manifold is None or CrossSection is None:
        raise RuntimeError("manifold3d is not installed")
    dist = float(distance)
    if not np.isfinite(dist) or dist <= 1e-12:
        raise ValueError("extrude distance must be a positive finite number")
    r = float(radius)
    if not np.isfinite(r) or r <= 1e-12:
        raise ValueError("degenerate circle: radius must be positive")
    segs = max(3, int(segments))
    cs = CrossSection.circle(r, segs)
    cx, cy = float(center[0]), float(center[1])
    if abs(cx) > 1e-15 or abs(cy) > 1e-15:
        cs = cs.translate([cx, cy])
    if cs.is_empty() or abs(float(cs.area())) <= 1e-14:
        raise ValueError("degenerate circle: empty cross-section")
    man = Manifold.extrude(cs, dist)
    if not _status_ok(man.status()) or man.is_empty():
        raise RuntimeError(f"manifold extrude failed: {man.status()}")
    man = man.transform(_frame_transform(frame))
    mesh = Mesh.from_manifold(man)
    if not mesh.is_watertight():
        raise RuntimeError("extrude result is not watertight")
    return mesh


def extrude_profile(
    profile: Union["RectEntity", "CircleEntity", "SketchEntity"],
    distance: float,
    frame: "PlaneFrame",
    *,
    segments: int = 64,
) -> Mesh:
    """Extrude a closed sketch profile (Rectangle or Circle) into a watertight solid.

    Open entities (e.g. lines) and degenerate geometry are rejected with ValueError.
    """
    from cadcore.sketch import CircleEntity, EntityKind, LineEntity, RectEntity

    if isinstance(profile, LineEntity) or getattr(profile, "kind", None) is EntityKind.LINE:
        raise ValueError("cannot extrude open profile: line is not a closed profile")
    if isinstance(profile, RectEntity):
        return extrude_rectangle(profile.c0, profile.c1, distance, frame)
    if isinstance(profile, CircleEntity):
        return extrude_circle(
            profile.center, profile.radius, distance, frame, segments=segments
        )
    raise ValueError(
        f"unsupported profile type for extrude: {type(profile).__name__}"
    )
