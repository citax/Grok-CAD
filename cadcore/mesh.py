"""Triangle meshes and watertight CSG via manifold3d."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from manifold3d import CrossSection, JoinType, Manifold, Mesh as ManifoldMesh
except ImportError:  # pragma: no cover
    CrossSection = None  # type: ignore
    JoinType = None  # type: ignore
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


def write_stl_binary(
    mesh: Mesh,
    path: Union[str, Path],
    *,
    header: str = "Grok CAD binary STL",
    require_watertight: bool = True,
) -> None:
    """Write a triangle mesh as binary STL.

    Rejects empty meshes. When ``require_watertight`` is True (default), also
    rejects non-watertight / non-solid meshes with a clear ValueError.
    """
    if mesh is None or mesh.empty:
        raise ValueError("cannot export empty mesh to STL")
    if require_watertight and not mesh.is_watertight():
        raise ValueError("cannot export non-watertight (open/non-solid) mesh to STL")

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    ntri = int(len(faces))
    if ntri <= 0:
        raise ValueError("cannot export empty mesh to STL")

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    lens = np.linalg.norm(normals, axis=1)
    # Avoid div-by-zero; zero-area faces should already fail watertight check
    good = lens > 1e-30
    normals = np.zeros_like(normals)
    normals[good] = (np.cross(v1 - v0, v2 - v0)[good]) / lens[good, None]

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    hdr = header.encode("ascii", errors="replace")[:80]
    hdr = hdr + b"\0" * (80 - len(hdr))

    # Pack facets efficiently: each facet is 12 floats + uint16 attribute
    facet_dtype = np.dtype(
        [
            ("n", "<f4", (3,)),
            ("v0", "<f4", (3,)),
            ("v1", "<f4", (3,)),
            ("v2", "<f4", (3,)),
            ("attr", "<u2"),
        ]
    )
    facets = np.empty(ntri, dtype=facet_dtype)
    facets["n"] = normals.astype(np.float32)
    facets["v0"] = v0.astype(np.float32)
    facets["v1"] = v1.astype(np.float32)
    facets["v2"] = v2.astype(np.float32)
    facets["attr"] = 0

    with out_path.open("wb") as fh:
        fh.write(hdr)
        fh.write(struct.pack("<I", ntri))
        fh.write(facets.tobytes())


def read_stl_binary(path: Union[str, Path]) -> Mesh:
    """Read a binary STL into an indexed Mesh (vertices may be duplicated)."""
    data = Path(path).read_bytes()
    if len(data) < 84:
        raise ValueError("STL file too small to be binary STL")
    # Binary STL: 80-byte header + uint32 count; ASCII starts with "solid"
    ntri = struct.unpack_from("<I", data, 80)[0]
    expected = 84 + ntri * 50
    if len(data) < expected:
        raise ValueError(
            f"STL truncated: header claims {ntri} triangles, need {expected} bytes, got {len(data)}"
        )
    facet_dtype = np.dtype(
        [
            ("n", "<f4", (3,)),
            ("v0", "<f4", (3,)),
            ("v1", "<f4", (3,)),
            ("v2", "<f4", (3,)),
            ("attr", "<u2"),
        ]
    )
    facets = np.frombuffer(data, dtype=facet_dtype, count=ntri, offset=84)
    # Build non-indexed then weld vertices (float32 STL quantisation)
    tri_verts = np.stack([facets["v0"], facets["v1"], facets["v2"]], axis=1)  # (T,3,3)
    flat = tri_verts.reshape(-1, 3).astype(np.float64)
    keys = np.round(flat, decimals=6)
    verts, inv = np.unique(keys, axis=0, return_inverse=True)
    faces = inv.reshape(-1, 3).astype(np.int32)
    return Mesh(verts, faces)


def write_stl(
    mesh: Mesh,
    path: Union[str, Path],
    *,
    binary: bool = True,
    require_watertight: bool = True,
) -> None:
    """Export mesh to STL (binary by default)."""
    if not binary:
        raise ValueError("ASCII STL is not supported; use binary=True")
    write_stl_binary(mesh, path, require_watertight=require_watertight)


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
    """Cylinder along +Z, centered at origin (manifold3d convention)."""
    if Manifold is None:
        raise RuntimeError("manifold3d is not installed")
    # cylinder(height, radius_low, radius_high, circular_segments, center)
    # Axis is +Z; center=True → symmetric about origin along Z.
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


def _profile_to_cross_section(
    profile: Union["RectEntity", "CircleEntity", "SketchEntity", Sequence[Tuple[float, float]]],
    *,
    circle_segments: int = 64,
) -> "CrossSection":
    """Build a CrossSection from a closed sketch entity, line-loop, or UV polygon."""
    from cadcore.profiles import ClosedLineLoop
    from cadcore.sketch import CircleEntity, EntityKind, LineEntity, RectEntity

    if isinstance(profile, ClosedLineLoop):
        return _cross_section_from_polygon(profile.vertices)

    if isinstance(profile, (list, tuple)) and profile and not hasattr(profile, "kind"):
        # Sequence of UV points
        pts = profile  # type: ignore[assignment]
        if len(pts) < 3:
            raise ValueError("open profile: polygon must have at least 3 vertices")
        return _cross_section_from_polygon(pts)  # type: ignore[arg-type]

    if isinstance(profile, LineEntity) or getattr(profile, "kind", None) is EntityKind.LINE:
        raise ValueError("open profile: line is not a closed profile")
    if isinstance(profile, RectEntity):
        return _cross_section_from_polygon(_rect_polygon_uv(profile.c0, profile.c1))
    if isinstance(profile, CircleEntity):
        if CrossSection is None:
            raise RuntimeError("manifold3d is not installed")
        r = float(profile.radius)
        if r <= 1e-12:
            raise ValueError("degenerate circle: radius must be positive")
        segs = max(3, int(circle_segments))
        cs = CrossSection.circle(r, segs)
        cx, cy = float(profile.center[0]), float(profile.center[1])
        if abs(cx) > 1e-15 or abs(cy) > 1e-15:
            cs = cs.translate([cx, cy])
        if cs.is_empty() or abs(float(cs.area())) <= 1e-14:
            raise ValueError("degenerate circle: empty cross-section")
        return cs
    raise ValueError(f"unsupported profile type for fillet: {type(profile).__name__}")


def _min_edge_length_uv(poly: Sequence[Tuple[float, float]]) -> float:
    n = len(poly)
    if n < 2:
        return 0.0
    m = float("inf")
    for i in range(n):
        x0, y0 = float(poly[i][0]), float(poly[i][1])
        x1, y1 = float(poly[(i + 1) % n][0]), float(poly[(i + 1) % n][1])
        m = min(m, float(np.hypot(x1 - x0, y1 - y0)))
    return m if np.isfinite(m) else 0.0


def fillet_profile(
    profile: Union["RectEntity", "CircleEntity", "SketchEntity", Sequence[Tuple[float, float]]],
    radius: float,
    segments: int = 32,
) -> "CrossSection":
    """Round every convex corner of a closed 2D profile by ``radius``.

    Uses manifold3d ``CrossSection.offset`` with ``JoinType.Round``: shrink by
    ``r`` then expand by ``r`` (dual offset). This is the dual-offset fillet
    (inset then outset); it preserves winding and reduces area by the missing
    corner sectors ``(4-π)r²`` on a square.

    Returns a manifold3d ``CrossSection`` ready for extrude/revolve.
    """
    if CrossSection is None or JoinType is None:
        raise RuntimeError("manifold3d is not installed")
    r = float(radius)
    if not np.isfinite(r) or r <= 1e-12:
        raise ValueError("fillet radius must be positive (radius <= 0 is invalid)")
    segs = max(3, int(segments))
    # Floor arc tessellation so coarse segs still stay within ~1% area on large fillets
    arc_segs = max(16, segs)

    cs = _profile_to_cross_section(profile, circle_segments=max(arc_segs, 32))
    area0 = float(cs.area())
    if area0 <= 1e-14:
        raise ValueError("degenerate profile: empty or zero-area cross-section")

    # Pre-check for rectangles: r must be < half min side
    from cadcore.sketch import RectEntity

    if isinstance(profile, RectEntity):
        u0, u1 = sorted([profile.c0[0], profile.c1[0]])
        v0, v1 = sorted([profile.c0[1], profile.c1[1]])
        half_min = 0.5 * min(u1 - u0, v1 - v0)
        if r >= half_min - 1e-12:
            raise ValueError(
                "fillet radius too large for the profile (self-intersection)"
            )
    elif isinstance(profile, (list, tuple)) and profile and not hasattr(profile, "kind"):
        min_e = _min_edge_length_uv(profile)  # type: ignore[arg-type]
        if min_e > 0 and r >= 0.5 * min_e - 1e-12:
            raise ValueError(
                "fillet radius too large for the profile (self-intersection)"
            )

    # Dual-offset fillet: inset by r (round), then outset by r (round).
    # (Outset-then-inset does not round convex exterior corners in Clipper2.)
    inset = cs.offset(-r, JoinType.Round, 2.0, arc_segs)
    if inset.is_empty() or abs(float(inset.area())) <= 1e-14:
        raise ValueError(
            "fillet radius too large for the profile (self-intersection)"
        )
    filleted = inset.offset(r, JoinType.Round, 2.0, arc_segs)
    if filleted.is_empty() or abs(float(filleted.area())) <= 1e-14:
        raise ValueError(
            "fillet radius too large for the profile (self-intersection)"
        )
    # Area should strictly decrease for a positive fillet on a polygonal profile
    # with corners (circles are already smooth — area nearly unchanged).
    area1 = float(filleted.area())
    if area1 > area0 + 1e-6 * max(area0, 1.0):
        raise ValueError(
            "fillet radius too large for the profile (self-intersection)"
        )
    return filleted


def extrude_filleted_profile(
    profile: Union["RectEntity", "CircleEntity", "SketchEntity", Sequence[Tuple[float, float]]],
    distance: float,
    frame: "PlaneFrame",
    radius: float,
    *,
    segments: int = 32,
) -> Mesh:
    """Fillet a closed profile by ``radius``, then extrude by ``distance`` along normal."""
    if Manifold is None:
        raise RuntimeError("manifold3d is not installed")
    dist = float(distance)
    if not np.isfinite(dist) or dist <= 1e-12:
        raise ValueError("extrude distance must be a positive finite number")
    cs = fillet_profile(profile, radius, segments=segments)
    man = Manifold.extrude(cs, dist)
    if not _status_ok(man.status()) or man.is_empty():
        raise RuntimeError(f"manifold extrude of filleted profile failed: {man.status()}")
    man = man.transform(_frame_transform(frame))
    mesh = Mesh.from_manifold(man)
    if not mesh.is_watertight():
        raise RuntimeError("filleted extrude result is not watertight")
    return mesh


def _profile_aabb(
    profile: Union["RectEntity", "CircleEntity", "SketchEntity", Sequence[Tuple[float, float]]],
) -> Tuple[float, float, float, float]:
    """Axis-aligned UV bounds (u0, v0, u1, v1) of a closed profile."""
    from cadcore.sketch import CircleEntity, RectEntity

    if isinstance(profile, RectEntity):
        u0, u1 = sorted([float(profile.c0[0]), float(profile.c1[0])])
        v0, v1 = sorted([float(profile.c0[1]), float(profile.c1[1])])
        return u0, v0, u1, v1
    if isinstance(profile, CircleEntity):
        cx, cy, r = float(profile.center[0]), float(profile.center[1]), float(profile.radius)
        return cx - r, cy - r, cx + r, cy + r
    if isinstance(profile, (list, tuple)) and profile and not hasattr(profile, "kind"):
        us = [float(p[0]) for p in profile]  # type: ignore[union-attr]
        vs = [float(p[1]) for p in profile]  # type: ignore[union-attr]
        return min(us), min(vs), max(us), max(vs)
    # Fallback via CrossSection bounds
    cs = _profile_to_cross_section(profile)
    b = cs.bounds()  # (minx, miny, maxx, maxy)
    return float(b[0]), float(b[1]), float(b[2]), float(b[3])


def profile_with_hole(
    profile: Union["RectEntity", "CircleEntity", "SketchEntity", Sequence[Tuple[float, float]]],
    hole_center: Sequence[float],
    hole_radius: float,
    segments: int = 32,
) -> "CrossSection":
    """Subtract a circular hole from a closed profile via CrossSection difference.

    Preserves outer winding; returns a manifold3d CrossSection with a hole contour.
    """
    if CrossSection is None:
        raise RuntimeError("manifold3d is not installed")
    hr = float(hole_radius)
    if not np.isfinite(hr) or hr <= 1e-12:
        raise ValueError("hole radius must be positive (hole_radius <= 0 is invalid)")
    cx, cy = float(hole_center[0]), float(hole_center[1])
    if not (np.isfinite(cx) and np.isfinite(cy)):
        raise ValueError("hole center must be finite")
    segs = max(3, int(segments))

    outer = _profile_to_cross_section(profile, circle_segments=max(segs, 32))
    area0 = float(outer.area())
    if area0 <= 1e-14:
        raise ValueError("degenerate profile: empty or zero-area cross-section")

    # Hole disk must lie strictly inside the profile AABB.
    # Touching an edge is rejected as non-manifold; any excursion outside as OOB.
    u0, v0, u1, v1 = _profile_aabb(profile)
    eps = 1e-9
    strictly_inside = (
        cx - hr > u0 + eps
        and cx + hr < u1 - eps
        and cy - hr > v0 + eps
        and cy + hr < v1 - eps
    )
    if not strictly_inside:
        part_outside = (
            cx - hr < u0 - eps
            or cx + hr > u1 + eps
            or cy - hr < v0 - eps
            or cy + hr > v1 + eps
        )
        if part_outside:
            raise ValueError("hole reaches outside the profile bounds")
        raise ValueError(
            "hole would make the profile non-manifold (touches an edge)"
        )

    hole = CrossSection.circle(hr, segs)
    if abs(cx) > 1e-15 or abs(cy) > 1e-15:
        hole = hole.translate([cx, cy])
    hole_area = float(hole.area())
    if hole.is_empty() or hole_area <= 1e-14:
        raise ValueError("degenerate hole: empty cross-section")

    result = outer - hole
    if result.is_empty():
        raise ValueError(
            "hole would make the profile non-manifold (empty difference)"
        )
    area1 = float(result.area())
    # Expected: outer area reduced by ~hole area; if almost unchanged, hole missed
    if area1 >= area0 - 1e-6 * max(area0, 1.0):
        raise ValueError("hole reaches outside the profile bounds")
    # If hole is only partially inside, area drop is less than full hole area
    expected = area0 - hole_area
    if area1 > expected + 0.05 * max(hole_area, 1.0):
        # Partial overlap — treat as out of bounds / invalid
        raise ValueError("hole reaches outside the profile bounds")
    # Contour count should be ≥2 (outer + hole) for a simple through-hole
    try:
        ncont = int(result.num_contour())
    except Exception:
        ncont = 0
    if ncont < 2:
        raise ValueError(
            "hole would make the profile non-manifold (missing hole contour)"
        )
    return result


def extrude_pocketed_profile(
    profile: Union["RectEntity", "CircleEntity", "SketchEntity", Sequence[Tuple[float, float]]],
    distance: float,
    frame: "PlaneFrame",
    hole_center: Sequence[float],
    hole_radius: float,
    *,
    segments: int = 32,
) -> Mesh:
    """Pocket a circular through-hole from a closed profile, then extrude."""
    if Manifold is None:
        raise RuntimeError("manifold3d is not installed")
    dist = float(distance)
    if not np.isfinite(dist) or dist <= 1e-12:
        raise ValueError("extrude distance must be a positive finite number")
    cs = profile_with_hole(profile, hole_center, hole_radius, segments=segments)
    man = Manifold.extrude(cs, dist)
    if not _status_ok(man.status()) or man.is_empty():
        raise RuntimeError(f"manifold extrude of pocketed profile failed: {man.status()}")
    man = man.transform(_frame_transform(frame))
    mesh = Mesh.from_manifold(man)
    if not mesh.is_watertight():
        raise RuntimeError("pocketed extrude result is not watertight")
    return mesh


def extrude_pocketed_filleted_profile(
    profile: Union["RectEntity", "CircleEntity", "SketchEntity", Sequence[Tuple[float, float]]],
    distance: float,
    frame: "PlaneFrame",
    *,
    fillet_radius: float,
    hole_center: Sequence[float],
    hole_radius: float,
    segments: int = 32,
) -> Mesh:
    """Fillet outer corners, subtract a circular hole, then extrude (canonical ref)."""
    if Manifold is None or CrossSection is None:
        raise RuntimeError("manifold3d is not installed")
    dist = float(distance)
    if not np.isfinite(dist) or dist <= 1e-12:
        raise ValueError("extrude distance must be a positive finite number")
    # Fillet outer profile first, then pocket the hole into the filleted CS
    filleted = fillet_profile(profile, fillet_radius, segments=segments)
    # Re-validate hole against original AABB (conservative) via profile_with_hole on original
    # but apply hole to the filleted section for geometry:
    hr = float(hole_radius)
    if not np.isfinite(hr) or hr <= 1e-12:
        raise ValueError("hole radius must be positive (hole_radius <= 0 is invalid)")
    # Containment check uses original profile AABB
    _ = profile_with_hole(profile, hole_center, hole_radius, segments=segments)
    segs = max(3, int(segments))
    cx, cy = float(hole_center[0]), float(hole_center[1])
    hole = CrossSection.circle(hr, segs)
    if abs(cx) > 1e-15 or abs(cy) > 1e-15:
        hole = hole.translate([cx, cy])
    pocketed = filleted - hole
    if pocketed.is_empty():
        raise ValueError("hole would make the profile non-manifold (empty difference)")
    man = Manifold.extrude(pocketed, dist)
    if not _status_ok(man.status()) or man.is_empty():
        raise RuntimeError(
            f"manifold extrude of pocketed-filleted profile failed: {man.status()}"
        )
    man = man.transform(_frame_transform(frame))
    mesh = Mesh.from_manifold(man)
    if not mesh.is_watertight():
        raise RuntimeError("pocketed-filleted extrude result is not watertight")
    return mesh


def _manifold_after_extrude(
    man: "Manifold",
    dist: float,
    frame: "PlaneFrame",
    *,
    reversed: bool = False,
) -> Mesh:
    """Apply reverse (local -Z shift) then map UVH → world via frame transform.

    Manifold.extrude builds height along +local Z in [0, dist]. Translating by
    ``-dist`` along Z before the frame transform places the solid on the
    opposite side of the sketch plane without changing |depth|.
    """
    if reversed:
        man = man.translate([0.0, 0.0, -float(dist)])
    man = man.transform(_frame_transform(frame))
    mesh = Mesh.from_manifold(man)
    if not mesh.is_watertight():
        raise RuntimeError("extrude result is not watertight")
    return mesh


def extrude_polygon(
    polygon_uv: Sequence[Tuple[float, float]],
    distance: float,
    frame: "PlaneFrame",
    *,
    reversed: bool = False,
) -> Mesh:
    """Extrude a closed 2D UV polygon along ``frame.normal`` by ``distance``.

    Uses manifold3d ``CrossSection`` + ``Manifold.extrude`` (not hand-rolled geometry).
    ``reversed=True`` pads along −normal (depth stays positive).
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
    return _manifold_after_extrude(man, dist, frame, reversed=reversed)


def extrude_rectangle(
    c0: Sequence[float],
    c1: Sequence[float],
    distance: float,
    frame: "PlaneFrame",
    *,
    reversed: bool = False,
) -> Mesh:
    """Extrude an axis-aligned UV rectangle along the plane normal."""
    return extrude_polygon(
        _rect_polygon_uv(c0, c1), distance, frame, reversed=reversed
    )


def extrude_circle(
    center: Sequence[float],
    radius: float,
    distance: float,
    frame: "PlaneFrame",
    *,
    segments: int = 64,
    reversed: bool = False,
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
    return _manifold_after_extrude(man, dist, frame, reversed=reversed)


def extrude_profile(
    profile: Union["RectEntity", "CircleEntity", "SketchEntity"],
    distance: float,
    frame: "PlaneFrame",
    *,
    segments: int = 64,
    holes: Optional[Sequence[Union["RectEntity", "CircleEntity", "SketchEntity"]]] = None,
    reversed: bool = False,
) -> Mesh:
    """Extrude a closed sketch profile (Rectangle or Circle) into a watertight solid.

    Optional ``holes`` are closed profiles nested inside the outer boundary; they
    are subtracted via CrossSection difference before extrude (through-holes).
    ``reversed=True`` pads along −plane normal; ``distance`` must stay positive.

    Open entities (e.g. lines) and degenerate geometry are rejected with ValueError.
    """
    from cadcore.profiles import ClosedLineLoop
    from cadcore.sketch import CircleEntity, EntityKind, LineEntity, RectEntity

    if isinstance(profile, LineEntity) or getattr(profile, "kind", None) is EntityKind.LINE:
        raise ValueError("cannot extrude open profile: line is not a closed profile")

    hole_list = list(holes) if holes else []
    if not hole_list:
        if isinstance(profile, RectEntity):
            return extrude_rectangle(
                profile.c0, profile.c1, distance, frame, reversed=reversed
            )
        if isinstance(profile, CircleEntity):
            return extrude_circle(
                profile.center,
                profile.radius,
                distance,
                frame,
                segments=segments,
                reversed=reversed,
            )
        if isinstance(profile, ClosedLineLoop):
            return extrude_polygon(
                profile.vertices, distance, frame, reversed=reversed
            )
        raise ValueError(
            f"unsupported profile type for extrude: {type(profile).__name__}"
        )

    # Nested holes: outer CS minus each hole CS, then extrude
    if Manifold is None or CrossSection is None:
        raise RuntimeError("manifold3d is not installed")
    dist = float(distance)
    if not np.isfinite(dist) or dist <= 1e-12:
        raise ValueError("extrude distance must be a positive finite number")
    segs = max(3, int(segments))
    cs = _profile_to_cross_section(profile, circle_segments=segs)
    for h in hole_list:
        if isinstance(h, LineEntity) or getattr(h, "kind", None) is EntityKind.LINE:
            raise ValueError("cannot use open profile as a hole")
        if isinstance(h, CircleEntity):
            hole_cs = CrossSection.circle(float(h.radius), segs)
            cx, cy = float(h.center[0]), float(h.center[1])
            if abs(cx) > 1e-15 or abs(cy) > 1e-15:
                hole_cs = hole_cs.translate([cx, cy])
        elif isinstance(h, RectEntity):
            hole_cs = _cross_section_from_polygon(_rect_polygon_uv(h.c0, h.c1))
        else:
            hole_cs = _profile_to_cross_section(h, circle_segments=segs)
        cs = cs - hole_cs
    if cs.is_empty() or abs(float(cs.area())) <= 1e-14:
        raise ValueError("extrude profile with holes produced empty cross-section")
    man = Manifold.extrude(cs, dist)
    if not _status_ok(man.status()) or man.is_empty():
        raise RuntimeError(f"manifold extrude failed: {man.status()}")
    return _manifold_after_extrude(man, dist, frame, reversed=reversed)


def _normalize_axis_2d(
    axis_origin: Sequence[float], axis_direction: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (origin, unit axis dir, unit radial basis) in UV."""
    o = np.asarray(axis_origin, dtype=np.float64).reshape(2)
    d = np.asarray(axis_direction, dtype=np.float64).reshape(2)
    nrm = float(np.linalg.norm(d))
    if nrm <= 1e-12:
        raise ValueError("revolve axis direction must be non-zero")
    d = d / nrm
    # 90° CW rotation of axis → radial basis so V-axis (0,1) maps to +U (1,0)
    radial = np.array([d[1], -d[0]], dtype=np.float64)
    return o, d, radial


def _uv_to_revolve_cs(
    points_uv: Sequence[Tuple[float, float]],
    axis_origin: Sequence[float],
    axis_direction: Sequence[float],
) -> Tuple[list, np.ndarray, np.ndarray, np.ndarray]:
    """Map UV polygon into manifold revolve CS: x=radial, y=along-axis.

    Rejects profiles that cross the axis (points on both sides).
    Flips radial basis if the profile lies on the negative-X side only.
    Ensures CCW winding for positive CrossSection area.
    """
    o, d, radial = _normalize_axis_2d(axis_origin, axis_direction)
    poly: list[Tuple[float, float]] = []
    for p in points_uv:
        uv = np.asarray(p, dtype=np.float64).reshape(2)
        along = float(np.dot(uv - o, d))
        rad = float(np.dot(uv - o, radial))
        poly.append((rad, along))
    rs = [p[0] for p in poly]
    r_min, r_max = min(rs), max(rs)
    if r_min < -1e-12 and r_max > 1e-12:
        raise ValueError("profile crosses the revolve axis")
    if r_max <= 1e-12:
        # Entirely on −radial side (or on axis): flip so manifold sees +X
        poly = [(-r, s) for r, s in poly]
        radial = -radial
        rs = [p[0] for p in poly]
        r_min, r_max = min(rs), max(rs)
    if r_max <= 1e-12:
        raise ValueError("profile is degenerate on the revolve axis (zero radius)")
    # Ensure CCW (positive area) for CrossSection
    area = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    if area < 0:
        poly = list(reversed(poly))
    return poly, o, d, radial


def _revolve_frame_transform(
    frame: "PlaneFrame",
    axis_origin_uv: np.ndarray,
    axis_dir_uv: np.ndarray,
    radial_uv: np.ndarray,
) -> list:
    """3×4 map of manifold revolve output (axis=+Z, radial in XY) → world.

    Manifold revolve: CS (x=radial, y=along-axis) → solid with axis along Z.
    World point = O + mx·R + my·N + mz·A where A,R are world images of UV axes.
    """
    u = np.asarray(frame.u_axis, dtype=np.float64).reshape(3)
    v = np.asarray(frame.v_axis, dtype=np.float64).reshape(3)
    n = np.asarray(frame.normal, dtype=np.float64).reshape(3)
    origin = np.asarray(frame.origin, dtype=np.float64).reshape(3)
    # Axis / radial directions in world (in-plane)
    axis_w = axis_dir_uv[0] * u + axis_dir_uv[1] * v
    radial_w = radial_uv[0] * u + radial_uv[1] * v
    # Origin of revolve CS in world (axis origin on plane)
    o_w = origin + axis_origin_uv[0] * u + axis_origin_uv[1] * v
    # Columns: radial, normal (out of plane), axis
    return [
        [float(radial_w[0]), float(n[0]), float(axis_w[0]), float(o_w[0])],
        [float(radial_w[1]), float(n[1]), float(axis_w[1]), float(o_w[1])],
        [float(radial_w[2]), float(n[2]), float(axis_w[2]), float(o_w[2])],
    ]


def revolve_polygon(
    polygon_uv: Sequence[Tuple[float, float]],
    frame: "PlaneFrame",
    *,
    axis_origin: Sequence[float] = (0.0, 0.0),
    axis_direction: Sequence[float] = (0.0, 1.0),
    angle_degrees: float = 360.0,
    segments: int = 64,
) -> Mesh:
    """Revolve a closed UV polygon about an in-plane axis into a watertight solid.

    Uses manifold3d ``CrossSection.revolve`` / ``Manifold.revolve`` (not hand-rolled).
    Default axis is the sketch V-axis through the origin.
    """
    if Manifold is None or CrossSection is None:
        raise RuntimeError("manifold3d is not installed")
    ang = float(angle_degrees)
    if not np.isfinite(ang) or ang <= 1e-12:
        raise ValueError("revolve angle must be a positive finite number (degrees)")
    if ang > 360.0 + 1e-9:
        raise ValueError("revolve angle must be at most 360 degrees")
    if len(polygon_uv) < 3:
        raise ValueError("polygon must have at least 3 vertices")
    poly_cs, o_uv, d_uv, radial_uv = _uv_to_revolve_cs(
        polygon_uv, axis_origin, axis_direction
    )
    cs = _cross_section_from_polygon(poly_cs)
    segs = max(3, int(segments))
    man = Manifold.revolve(cs, segs, ang)
    if not _status_ok(man.status()) or man.is_empty():
        raise RuntimeError(f"manifold revolve failed: {man.status()}")
    if float(man.volume()) <= 1e-14:
        raise ValueError("revolve produced empty solid (check angle/profile)")
    man = man.transform(_revolve_frame_transform(frame, o_uv, d_uv, radial_uv))
    mesh = Mesh.from_manifold(man)
    if not mesh.is_watertight():
        raise RuntimeError("revolve result is not watertight")
    return mesh


def revolve_rectangle(
    c0: Sequence[float],
    c1: Sequence[float],
    frame: "PlaneFrame",
    *,
    axis_origin: Sequence[float] = (0.0, 0.0),
    axis_direction: Sequence[float] = (0.0, 1.0),
    angle_degrees: float = 360.0,
    segments: int = 64,
) -> Mesh:
    """Revolve an axis-aligned UV rectangle about an in-plane axis."""
    return revolve_polygon(
        _rect_polygon_uv(c0, c1),
        frame,
        axis_origin=axis_origin,
        axis_direction=axis_direction,
        angle_degrees=angle_degrees,
        segments=segments,
    )


def revolve_circle(
    center: Sequence[float],
    radius: float,
    frame: "PlaneFrame",
    *,
    axis_origin: Sequence[float] = (0.0, 0.0),
    axis_direction: Sequence[float] = (0.0, 1.0),
    angle_degrees: float = 360.0,
    segments: int = 64,
    profile_segments: int = 48,
) -> Mesh:
    """Revolve a UV circle about an in-plane axis (polygonal profile)."""
    r = float(radius)
    if not np.isfinite(r) or r <= 1e-12:
        raise ValueError("degenerate circle: radius must be positive")
    cx, cy = float(center[0]), float(center[1])
    n = max(3, int(profile_segments))
    poly = [
        (cx + r * float(np.cos(2 * np.pi * i / n)), cy + r * float(np.sin(2 * np.pi * i / n)))
        for i in range(n)
    ]
    return revolve_polygon(
        poly,
        frame,
        axis_origin=axis_origin,
        axis_direction=axis_direction,
        angle_degrees=angle_degrees,
        segments=segments,
    )


def revolve_profile(
    profile: Union["RectEntity", "CircleEntity", "SketchEntity"],
    frame: "PlaneFrame",
    *,
    axis_origin: Sequence[float] = (0.0, 0.0),
    axis_direction: Sequence[float] = (0.0, 1.0),
    angle_degrees: float = 360.0,
    segments: int = 64,
) -> Mesh:
    """Revolve a closed sketch profile (Rectangle or Circle) into a watertight solid.

    Open entities, axis-crossing profiles, and non-positive angles are rejected.
    """
    from cadcore.profiles import ClosedLineLoop
    from cadcore.sketch import CircleEntity, EntityKind, LineEntity, RectEntity

    if isinstance(profile, LineEntity) or getattr(profile, "kind", None) is EntityKind.LINE:
        raise ValueError("cannot revolve open profile: line is not a closed profile")
    if isinstance(profile, RectEntity):
        return revolve_rectangle(
            profile.c0,
            profile.c1,
            frame,
            axis_origin=axis_origin,
            axis_direction=axis_direction,
            angle_degrees=angle_degrees,
            segments=segments,
        )
    if isinstance(profile, CircleEntity):
        return revolve_circle(
            profile.center,
            profile.radius,
            frame,
            axis_origin=axis_origin,
            axis_direction=axis_direction,
            angle_degrees=angle_degrees,
            segments=segments,
        )
    if isinstance(profile, ClosedLineLoop):
        return revolve_polygon(
            profile.vertices,
            frame,
            axis_origin=axis_origin,
            axis_direction=axis_direction,
            angle_degrees=angle_degrees,
            segments=segments,
        )
    raise ValueError(
        f"unsupported profile type for revolve: {type(profile).__name__}"
    )
