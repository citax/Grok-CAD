"""2D sketch model on a reference plane (pure Python, no GUI)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Sequence, Tuple

import numpy as np

Vec2 = Tuple[float, float]


@dataclass(frozen=True)
class PlaneFrame:
    """Orthonormal plane frame: p = origin + u * u_axis + v * v_axis."""

    origin: np.ndarray
    u_axis: np.ndarray
    v_axis: np.ndarray
    normal: np.ndarray

    @staticmethod
    def from_plane_type(plane_type_name: str) -> "PlaneFrame":
        o = np.zeros(3, dtype=np.float64)
        name = plane_type_name.upper()
        if "FRONT" in name or name == "XY":
            u = np.array([1.0, 0.0, 0.0])
            v = np.array([0.0, 1.0, 0.0])
            n = np.array([0.0, 0.0, 1.0])
        elif "TOP" in name or name == "XZ":
            u = np.array([1.0, 0.0, 0.0])
            v = np.array([0.0, 0.0, 1.0])
            n = np.array([0.0, 1.0, 0.0])
        elif "RIGHT" in name or name == "YZ":
            u = np.array([0.0, 0.0, 1.0])
            v = np.array([0.0, 1.0, 0.0])
            n = np.array([1.0, 0.0, 0.0])
        else:
            u = np.array([1.0, 0.0, 0.0])
            v = np.array([0.0, 1.0, 0.0])
            n = np.array([0.0, 0.0, 1.0])
        return PlaneFrame(o, u, v, n)

    def to_world(self, uv: Sequence[float]) -> np.ndarray:
        u, v = float(uv[0]), float(uv[1])
        return self.origin + u * self.u_axis + v * self.v_axis

    def to_local(self, xyz: Sequence[float]) -> Vec2:
        p = np.asarray(xyz, dtype=np.float64) - self.origin
        return (float(np.dot(p, self.u_axis)), float(np.dot(p, self.v_axis)))

    def ray_intersect(
        self, ray_origin: Sequence[float], ray_dir: Sequence[float]
    ) -> Optional[np.ndarray]:
        o = np.asarray(ray_origin, dtype=np.float64)
        d = np.asarray(ray_dir, dtype=np.float64)
        dn = float(np.dot(d, self.normal))
        if abs(dn) < 1e-12:
            return None
        t = float(np.dot(self.origin - o, self.normal) / dn)
        if t < 0:
            return None
        return o + t * d


class EntityKind(Enum):
    LINE = auto()
    RECTANGLE = auto()
    CIRCLE = auto()


class DimKind(Enum):
    """Driving sketch dimension kinds (SolidWorks-style smart dimensions)."""

    LINEAR = auto()  # line length, or rect width/height
    DIAMETER = auto()  # circle diameter
    ANGULAR = auto()  # angle between two lines (degrees in value_mm)


class HandleKind(Enum):
    ENDPOINT = auto()
    MIDPOINT = auto()
    CORNER = auto()
    CENTER = auto()
    RIM = auto()


@dataclass
class Handle:
    entity_id: int
    name: str
    kind: HandleKind
    uv: Vec2


@dataclass
class SketchEntity:
    id: int
    kind: EntityKind

    def handles(self) -> List[Handle]:
        raise NotImplementedError

    def translate(self, du: float, dv: float) -> None:
        raise NotImplementedError

    def set_handle(self, name: str, uv: Vec2) -> None:
        raise NotImplementedError


@dataclass
class LineEntity(SketchEntity):
    p0: Vec2 = (0.0, 0.0)
    p1: Vec2 = (1.0, 0.0)

    def __post_init__(self) -> None:
        self.kind = EntityKind.LINE
        self.p0 = (float(self.p0[0]), float(self.p0[1]))
        self.p1 = (float(self.p1[0]), float(self.p1[1]))

    def midpoint(self) -> Vec2:
        return ((self.p0[0] + self.p1[0]) * 0.5, (self.p0[1] + self.p1[1]) * 0.5)

    def handles(self) -> List[Handle]:
        return [
            Handle(self.id, "p0", HandleKind.ENDPOINT, self.p0),
            Handle(self.id, "p1", HandleKind.ENDPOINT, self.p1),
            Handle(self.id, "mid", HandleKind.MIDPOINT, self.midpoint()),
        ]

    def set_handle(self, name: str, uv: Vec2) -> None:
        if name == "p0":
            self.p0 = (float(uv[0]), float(uv[1]))
        elif name == "p1":
            self.p1 = (float(uv[0]), float(uv[1]))
        elif name == "mid":
            cur = self.midpoint()
            self.translate(uv[0] - cur[0], uv[1] - cur[1])

    def translate(self, du: float, dv: float) -> None:
        self.p0 = (self.p0[0] + du, self.p0[1] + dv)
        self.p1 = (self.p1[0] + du, self.p1[1] + dv)


@dataclass
class RectEntity(SketchEntity):
    c0: Vec2 = (0.0, 0.0)
    c1: Vec2 = (1.0, 1.0)

    def __post_init__(self) -> None:
        self.kind = EntityKind.RECTANGLE
        self.c0 = (float(self.c0[0]), float(self.c0[1]))
        self.c1 = (float(self.c1[0]), float(self.c1[1]))

    def corners(self) -> List[Vec2]:
        u0, u1 = sorted([self.c0[0], self.c1[0]])
        v0, v1 = sorted([self.c0[1], self.c1[1]])
        return [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]

    def handles(self) -> List[Handle]:
        return [Handle(self.id, f"c{i}", HandleKind.CORNER, c) for i, c in enumerate(self.corners())]

    def set_handle(self, name: str, uv: Vec2) -> None:
        idx = int(name[1]) if name.startswith("c") and name[1:].isdigit() else -1
        if idx < 0:
            return
        u0, u1 = sorted([self.c0[0], self.c1[0]])
        v0, v1 = sorted([self.c0[1], self.c1[1]])
        u, v = float(uv[0]), float(uv[1])
        if idx == 0:
            u0, v0 = u, v
        elif idx == 1:
            u1, v0 = u, v
        elif idx == 2:
            u1, v1 = u, v
        elif idx == 3:
            u0, v1 = u, v
        self.c0 = (u0, v0)
        self.c1 = (u1, v1)

    def translate(self, du: float, dv: float) -> None:
        self.c0 = (self.c0[0] + du, self.c0[1] + dv)
        self.c1 = (self.c1[0] + du, self.c1[1] + dv)


@dataclass
class CircleEntity(SketchEntity):
    center: Vec2 = (0.0, 0.0)
    radius: float = 1.0

    def __post_init__(self) -> None:
        self.kind = EntityKind.CIRCLE
        self.center = (float(self.center[0]), float(self.center[1]))
        self.radius = max(1e-9, float(self.radius))

    def rim_point(self) -> Vec2:
        return (self.center[0] + self.radius, self.center[1])

    def handles(self) -> List[Handle]:
        return [
            Handle(self.id, "center", HandleKind.CENTER, self.center),
            Handle(self.id, "rim", HandleKind.RIM, self.rim_point()),
        ]

    def set_handle(self, name: str, uv: Vec2) -> None:
        if name == "center":
            self.center = (float(uv[0]), float(uv[1]))
        elif name == "rim":
            du = uv[0] - self.center[0]
            dv = uv[1] - self.center[1]
            self.radius = max(1e-9, float(np.hypot(du, dv)))

    def translate(self, du: float, dv: float) -> None:
        self.center = (self.center[0] + du, self.center[1] + dv)


@dataclass
class SketchDimension:
    """Driving dimension: stored value owns geometry when applied.

    ``role``:
      * ``length`` — full line length (LineEntity), value in mm
      * ``width``  — |Δu| of a rectangle (RectEntity), value in mm
      * ``height`` — |Δv| of a rectangle (RectEntity), value in mm
      * ``diameter`` — 2·radius of a circle (CircleEntity), value in mm
      * ``angle`` — angle between two lines, value in **degrees** (stored in value_mm)

    ``entity_b_id`` is the second line for angle dimensions (−1 otherwise).
    """

    id: int
    kind: DimKind
    entity_id: int
    role: str = "length"
    value_mm: float = 0.0
    entity_b_id: int = -1
    # Angle: which endpoints form the corner (empty if lines do not share a vertex)
    pivot_h0: str = ""  # handle on entity_id
    pivot_h1: str = ""  # handle on entity_b_id


@dataclass
class Sketch:
    name: str = "Sketch"
    plane_feature_id: int = -1
    frame: PlaneFrame = field(default_factory=lambda: PlaneFrame.from_plane_type("FRONT"))
    entities: List[SketchEntity] = field(default_factory=list)
    dimensions: List[SketchDimension] = field(default_factory=list)
    # Persistent geometric relationships (see cadcore.constraints)
    constraints: list = field(default_factory=list)
    _next_entity_id: int = 1
    _next_dim_id: int = 1
    _next_constraint_id: int = 1

    def add_line(self, p0: Vec2, p1: Vec2) -> LineEntity:
        e = LineEntity(id=self._next_entity_id, kind=EntityKind.LINE, p0=p0, p1=p1)
        self._next_entity_id += 1
        self.entities.append(e)
        return e

    def add_rectangle(self, c0: Vec2, c1: Vec2) -> RectEntity:
        e = RectEntity(id=self._next_entity_id, kind=EntityKind.RECTANGLE, c0=c0, c1=c1)
        self._next_entity_id += 1
        self.entities.append(e)
        return e

    def add_circle(self, center: Vec2, radius: float) -> CircleEntity:
        e = CircleEntity(
            id=self._next_entity_id, kind=EntityKind.CIRCLE, center=center, radius=radius
        )
        self._next_entity_id += 1
        self.entities.append(e)
        return e

    def find_entity(self, eid: int) -> Optional[SketchEntity]:
        for e in self.entities:
            if e.id == eid:
                return e
        return None

    def remove_entity(self, eid: int) -> Optional[SketchEntity]:
        """Remove entity by id; return it if found."""
        for i, e in enumerate(self.entities):
            if e.id == eid:
                return self.entities.pop(i)
        return None

    def insert_entity(self, ent: SketchEntity, index: Optional[int] = None) -> None:
        """Insert an existing entity (keeps its id). Bumps _next_entity_id."""
        if self.find_entity(ent.id) is not None:
            return
        if index is None or index < 0 or index > len(self.entities):
            self.entities.append(ent)
        else:
            self.entities.insert(index, ent)
        self._next_entity_id = max(self._next_entity_id, int(ent.id) + 1)

    def all_handles(self) -> List[Handle]:
        hs: List[Handle] = []
        for e in self.entities:
            hs.extend(e.handles())
        return hs

    def snap_targets(self) -> List[Vec2]:
        pts: List[Vec2] = [(0.0, 0.0)]
        for e in self.entities:
            if isinstance(e, LineEntity):
                pts.extend([e.p0, e.p1])
            elif isinstance(e, RectEntity):
                pts.extend(e.corners())
            elif isinstance(e, CircleEntity):
                pts.append(e.center)
        return pts

    def unique_endpoints(self, *, tol: float = 1e-9) -> List[Vec2]:
        """Deduped entity endpoints / connection points (excludes bare origin unless used)."""
        raw: List[Vec2] = []
        for e in self.entities:
            if isinstance(e, LineEntity):
                raw.extend([e.p0, e.p1])
            elif isinstance(e, RectEntity):
                raw.extend(e.corners())
            elif isinstance(e, CircleEntity):
                raw.append(e.center)
        out: List[Vec2] = []
        for p in raw:
            if any(abs(p[0] - q[0]) <= tol and abs(p[1] - q[1]) <= tol for q in out):
                continue
            out.append((float(p[0]), float(p[1])))
        return out

    def shared_endpoints(self, *, tol: float = 1e-9) -> List[Vec2]:
        """Endpoints that appear on ≥2 entities (true junctions / connections)."""
        raw: List[Vec2] = []
        for e in self.entities:
            if isinstance(e, LineEntity):
                raw.extend([e.p0, e.p1])
            elif isinstance(e, RectEntity):
                raw.extend(e.corners())
            elif isinstance(e, CircleEntity):
                raw.append(e.center)
        clusters: List[List[Vec2]] = []
        for p in raw:
            placed = False
            for cl in clusters:
                q = cl[0]
                if abs(p[0] - q[0]) <= tol and abs(p[1] - q[1]) <= tol:
                    cl.append(p)
                    placed = True
                    break
            if not placed:
                clusters.append([p])
        out: List[Vec2] = []
        for cl in clusters:
            if len(cl) >= 2:
                u = sum(x[0] for x in cl) / len(cl)
                v = sum(x[1] for x in cl) / len(cl)
                out.append((float(u), float(v)))
        return out

    def find_dimension(self, did: int) -> Optional[SketchDimension]:
        for d in self.dimensions:
            if d.id == did:
                return d
        return None

    def dimensions_for_entity(self, eid: int) -> List[SketchDimension]:
        eid = int(eid)
        return [
            d
            for d in self.dimensions
            if int(d.entity_id) == eid or int(getattr(d, "entity_b_id", -1)) == eid
        ]

    def remove_dimensions_for_entity(self, eid: int) -> None:
        eid = int(eid)
        self.dimensions = [
            d
            for d in self.dimensions
            if int(d.entity_id) != eid and int(getattr(d, "entity_b_id", -1)) != eid
        ]

    def add_or_update_dimension(
        self,
        entity_id: int,
        role: str,
        value_mm: float,
        *,
        kind: Optional[DimKind] = None,
        entity_b_id: int = -1,
        pivot_h0: str = "",
        pivot_h1: str = "",
    ) -> SketchDimension:
        """Create or replace a driving dimension for (entity, role[, entity_b])."""
        eid = int(entity_id)
        ebid = int(entity_b_id)
        role = str(role)
        val = float(value_mm)
        if kind is None:
            if role == "diameter":
                kind = DimKind.DIAMETER
            elif role == "angle":
                kind = DimKind.ANGULAR
            else:
                kind = DimKind.LINEAR
        pivot_h0 = str(pivot_h0 or "")
        pivot_h1 = str(pivot_h1 or "")
        for d in self.dimensions:
            if (
                int(d.entity_id) == eid
                and str(d.role) == role
                and int(getattr(d, "entity_b_id", -1)) == ebid
            ):
                d.value_mm = val
                d.kind = kind
                d.entity_b_id = ebid
                if pivot_h0 or pivot_h1:
                    d.pivot_h0 = str(pivot_h0)
                    d.pivot_h1 = str(pivot_h1)
                return d
            # Angle is unordered pair
            if (
                role == "angle"
                and str(d.role) == "angle"
                and {int(d.entity_id), int(getattr(d, "entity_b_id", -1))}
                == {eid, ebid}
            ):
                d.value_mm = val
                d.kind = kind
                if pivot_h0 or pivot_h1:
                    # Map pivots if entity order matches stored order
                    if int(d.entity_id) == eid:
                        d.pivot_h0 = str(pivot_h0)
                        d.pivot_h1 = str(pivot_h1)
                    else:
                        d.pivot_h0 = str(pivot_h1)
                        d.pivot_h1 = str(pivot_h0)
                return d
        d = SketchDimension(
            id=self._next_dim_id,
            kind=kind,
            entity_id=eid,
            role=role,
            value_mm=val,
            entity_b_id=ebid,
            pivot_h0=str(pivot_h0 or ""),
            pivot_h1=str(pivot_h1 or ""),
        )
        self._next_dim_id += 1
        self.dimensions.append(d)
        return d


def line_length(ent: LineEntity) -> float:
    """World-UV length of a line (internal mm)."""
    return float(np.hypot(ent.p1[0] - ent.p0[0], ent.p1[1] - ent.p0[1]))


def set_line_length(ent: LineEntity, length: float, *, free_end: str = "p1") -> None:
    """Move free endpoint along the line direction so length becomes ``length`` (mm)."""
    L = max(1e-12, float(length))
    if free_end == "p0":
        fixed, free = ent.p1, ent.p0
        set_free = "p0"
    else:
        fixed, free = ent.p0, ent.p1
        set_free = "p1"
    du = free[0] - fixed[0]
    dv = free[1] - fixed[1]
    cur = float(np.hypot(du, dv))
    if cur < 1e-12:
        # Degenerate: extend along +u
        nu, nv = 1.0, 0.0
    else:
        nu, nv = du / cur, dv / cur
    new_free = (fixed[0] + nu * L, fixed[1] + nv * L)
    ent.set_handle(set_free, new_free)


def rect_width(ent: RectEntity) -> float:
    return abs(float(ent.c1[0]) - float(ent.c0[0]))


def rect_height(ent: RectEntity) -> float:
    return abs(float(ent.c1[1]) - float(ent.c0[1]))


def set_rect_width(ent: RectEntity, width: float, *, free_side: str = "max") -> None:
    """Set rectangle |Δu| to ``width`` (mm). ``free_side`` is 'min' or 'max' u-edge."""
    w = max(1e-12, float(width))
    u0, u1 = sorted([float(ent.c0[0]), float(ent.c1[0])])
    v0, v1 = sorted([float(ent.c0[1]), float(ent.c1[1])])
    if free_side == "min":
        u0 = u1 - w
    else:
        u1 = u0 + w
    ent.c0 = (u0, v0)
    ent.c1 = (u1, v1)


def set_rect_height(ent: RectEntity, height: float, *, free_side: str = "max") -> None:
    """Set rectangle |Δv| to ``height`` (mm)."""
    h = max(1e-12, float(height))
    u0, u1 = sorted([float(ent.c0[0]), float(ent.c1[0])])
    v0, v1 = sorted([float(ent.c0[1]), float(ent.c1[1])])
    if free_side == "min":
        v0 = v1 - h
    else:
        v1 = v0 + h
    ent.c0 = (u0, v0)
    ent.c1 = (u1, v1)


def set_circle_diameter(ent: CircleEntity, diameter: float) -> None:
    ent.radius = max(1e-12, float(diameter) * 0.5)


def line_direction(ent: LineEntity) -> np.ndarray:
    d = np.array(
        [ent.p1[0] - ent.p0[0], ent.p1[1] - ent.p0[1]], dtype=np.float64
    )
    return d


def line_angle_degrees(a: LineEntity, b: LineEntity) -> float:
    """Smaller angle between two line directions in degrees, range [0, 90].

    For driving dims we also support obtuse targets via apply; measurement
    reports the acute angle between undirected lines (SolidWorks-style).
    Use ``line_angle_degrees_oriented`` for 0–180 undirected interior.
    """
    return line_angle_degrees_oriented(a, b)


def line_angle_degrees_oriented(a: LineEntity, b: LineEntity) -> float:
    """Angle between direction vectors in degrees, range [0, 180]."""
    d0 = line_direction(a)
    d1 = line_direction(b)
    n0 = float(np.linalg.norm(d0))
    n1 = float(np.linalg.norm(d1))
    if n0 < 1e-12 or n1 < 1e-12:
        return 0.0
    c = float(np.clip(np.dot(d0, d1) / (n0 * n1), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def measure_dimension_value(
    ent: SketchEntity,
    role: str,
    *,
    ent_b: Optional[SketchEntity] = None,
) -> float:
    """Current geometric measure for a dimension role (mm, or degrees for angle)."""
    if isinstance(ent, LineEntity) and role == "length":
        return line_length(ent)
    if isinstance(ent, RectEntity) and role == "width":
        return rect_width(ent)
    if isinstance(ent, RectEntity) and role == "height":
        return rect_height(ent)
    if isinstance(ent, CircleEntity) and role == "diameter":
        return float(ent.radius) * 2.0
    if role == "angle" and isinstance(ent, LineEntity) and isinstance(ent_b, LineEntity):
        return line_angle_degrees_oriented(ent, ent_b)
    raise ValueError(f"cannot measure role={role!r} on {type(ent).__name__}")


def apply_dimension_value(
    ent: SketchEntity,
    role: str,
    value_mm: float,
    *,
    ent_b: Optional[SketchEntity] = None,
) -> None:
    """Drive geometry so the measured role equals ``value_mm`` (or degrees)."""
    val = float(value_mm)
    if not np.isfinite(val):
        raise ValueError("dimension value must be finite")
    if role != "angle" and val <= 1e-12:
        raise ValueError("dimension value must be a positive finite number")
    if isinstance(ent, LineEntity) and role == "length":
        set_line_length(ent, val, free_end="p1")
        return
    if isinstance(ent, RectEntity) and role == "width":
        set_rect_width(ent, val, free_side="max")
        return
    if isinstance(ent, RectEntity) and role == "height":
        set_rect_height(ent, val, free_side="max")
        return
    if isinstance(ent, CircleEntity) and role == "diameter":
        set_circle_diameter(ent, val)
        return
    if role == "angle" and isinstance(ent, LineEntity) and isinstance(ent_b, LineEntity):
        set_line_pair_angle(ent, ent_b, val)
        return
    raise ValueError(f"cannot apply role={role!r} on {type(ent).__name__}")


def find_shared_line_endpoints(
    a: LineEntity, b: LineEntity, *, tol: float = 1e-6
) -> Optional[Tuple[str, str]]:
    """If the lines meet at a corner, return (handle_on_a, handle_on_b), else None."""
    for ha, pa in (("p0", a.p0), ("p1", a.p1)):
        for hb, pb in (("p0", b.p0), ("p1", b.p1)):
            if abs(pa[0] - pb[0]) <= tol and abs(pa[1] - pb[1]) <= tol:
                return ha, hb
    return None


def _pick_direction_for_angle(
    ref_dir: np.ndarray, current_dir: np.ndarray, degrees: float
) -> np.ndarray:
    """Unit direction at ``degrees`` from ``ref_dir``, closest to ``current_dir``."""
    n0 = float(np.linalg.norm(ref_dir))
    if n0 < 1e-12:
        u0 = np.array([1.0, 0.0])
    else:
        u0 = ref_dir / n0
    target = float(degrees) % 180.0
    if target < 0:
        target += 180.0
    rad = np.radians(target)
    c, s = float(np.cos(rad)), float(np.sin(rad))
    cand1 = np.array([u0[0] * c - u0[1] * s, u0[0] * s + u0[1] * c])
    cand2 = np.array([u0[0] * c + u0[1] * s, -u0[0] * s + u0[1] * c])
    n1 = float(np.linalg.norm(current_dir))
    cur = current_dir / n1 if n1 > 1e-12 else cand1
    best = cand1
    best_score = -1e9
    for cand in (cand1, cand2, -cand1, -cand2):
        score = float(np.dot(cur, cand))
        if score > best_score:
            best_score = score
            best = cand
    return best


def set_line_pair_angle(
    a: LineEntity,
    b: LineEntity,
    degrees: float,
    *,
    move: str = "b",
    pivot: Optional[Tuple[str, str]] = None,
) -> None:
    """Set the angle between lines ``a`` and ``b`` to ``degrees`` (0–180).

    Rotates the line named by ``move`` (``'a'`` or ``'b'``).

    If the lines share a corner (or ``pivot=(handle_a, handle_b)`` is given),
    the moved line rotates about that shared endpoint so the corner stays put.
    Only if there is no shared corner does it fall back to rotating about the
    midpoint of the moved line.
    """
    if move not in ("a", "b"):
        move = "b"
    fixed_line, moved = (a, b) if move == "b" else (b, a)
    shared = pivot if pivot is not None else find_shared_line_endpoints(a, b)
    # Map shared handles onto moved/fixed
    pivot_handle_moved: Optional[str] = None
    if shared is not None:
        ha, hb = shared
        if move == "b":
            pivot_handle_moved = hb
            # Snap moved pivot to fixed line's shared end (preserve corner)
            fixed_pt = a.p0 if ha == "p0" else a.p1
            if hb == "p0":
                moved.p0 = (float(fixed_pt[0]), float(fixed_pt[1]))
            else:
                moved.p1 = (float(fixed_pt[0]), float(fixed_pt[1]))
        else:
            pivot_handle_moved = ha
            fixed_pt = b.p0 if hb == "p0" else b.p1
            if ha == "p0":
                moved.p0 = (float(fixed_pt[0]), float(fixed_pt[1]))
            else:
                moved.p1 = (float(fixed_pt[0]), float(fixed_pt[1]))

    ref = line_direction(fixed_line)
    cur = line_direction(moved)
    best = _pick_direction_for_angle(ref, cur, degrees)
    L = line_length(moved)
    if L < 1e-12:
        L = 1.0

    if pivot_handle_moved is not None:
        # Rotate about shared corner only — free end swings, pivot stays.
        # Line direction is always p1 - p0; pick sense of ``best`` that matches
        # the previous free-end side of the moved line.
        if pivot_handle_moved == "p0":
            pivot_pt = np.array(
                [moved.p0[0], moved.p0[1]], dtype=np.float64
            )
            free_cur = np.array(
                [moved.p1[0] - pivot_pt[0], moved.p1[1] - pivot_pt[1]],
                dtype=np.float64,
            )
            if float(np.dot(free_cur, best)) < 0:
                best = -best
            moved.p0 = (float(pivot_pt[0]), float(pivot_pt[1]))
            moved.p1 = (
                float(pivot_pt[0] + best[0] * L),
                float(pivot_pt[1] + best[1] * L),
            )
        else:
            pivot_pt = np.array(
                [moved.p1[0], moved.p1[1]], dtype=np.float64
            )
            free_cur = np.array(
                [moved.p0[0] - pivot_pt[0], moved.p0[1] - pivot_pt[1]],
                dtype=np.float64,
            )
            # p1 fixed, p0 = p1 - best*L  ⇒ direction p1-p0 = best
            if float(np.dot(free_cur, -best)) < 0:
                best = -best
            moved.p1 = (float(pivot_pt[0]), float(pivot_pt[1]))
            moved.p0 = (
                float(pivot_pt[0] - best[0] * L),
                float(pivot_pt[1] - best[1] * L),
            )
    else:
        # No shared corner: rotate about midpoint (disjoint lines only)
        mid = moved.midpoint()
        half = 0.5 * L
        moved.p0 = (mid[0] - best[0] * half, mid[1] - best[1] * half)
        moved.p1 = (mid[0] + best[0] * half, mid[1] + best[1] * half)


def infer_dimension_role(ent: SketchEntity, *, uv_hint: Optional[Vec2] = None) -> str:
    """Pick dimension role for Smart Dimension from the entity + click location.

    For a rectangle, the clicked **edge's length** is what the number sets:

    * Bottom / top (horizontal edges) → their length is the **width** (|Δu|)
    * Left / right (vertical edges) → their length is the **height** (|Δv|)

    ``uv_hint`` is the click in sketch UV; nearest edge wins.
    """
    if isinstance(ent, LineEntity):
        return "length"
    if isinstance(ent, CircleEntity):
        return "diameter"
    if isinstance(ent, RectEntity):
        if uv_hint is None:
            return "width"
        u0, u1 = sorted([float(ent.c0[0]), float(ent.c1[0])])
        v0, v1 = sorted([float(ent.c0[1]), float(ent.c1[1])])
        u, v = float(uv_hint[0]), float(uv_hint[1])
        # Distance to nearest vertical edge (left/right) vs horizontal (bottom/top)
        dist_to_vertical = min(abs(u - u0), abs(u - u1))
        dist_to_horizontal = min(abs(v - v0), abs(v - v1))
        # Near a horizontal edge → dimension that edge's length = width
        # Near a vertical edge → dimension that edge's length = height
        if dist_to_horizontal <= dist_to_vertical:
            return "width"
        return "height"
    raise ValueError(f"unsupported entity for dimension: {type(ent).__name__}")


def dimension_anchor_uv(
    ent: SketchEntity,
    role: str,
    *,
    ent_b: Optional[SketchEntity] = None,
) -> Vec2:
    """Label placement UV for a dimension."""
    if role == "angle" and isinstance(ent, LineEntity) and isinstance(ent_b, LineEntity):
        m0 = ent.midpoint()
        m1 = ent_b.midpoint()
        return (0.5 * (m0[0] + m1[0]), 0.5 * (m0[1] + m1[1]))
    if isinstance(ent, LineEntity):
        mid = ent.midpoint()
        # Offset slightly off the line for readability
        d = line_direction(ent)
        n = float(np.linalg.norm(d))
        if n > 1e-12:
            # perpendicular offset
            off = 0.08 * max(n, 1.0)
            return (mid[0] - d[1] / n * off, mid[1] + d[0] / n * off)
        return mid
    if isinstance(ent, RectEntity):
        u0, u1 = sorted([ent.c0[0], ent.c1[0]])
        v0, v1 = sorted([ent.c0[1], ent.c1[1]])
        cu, cv = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
        if role == "height":
            return (u1 + 0.05 * max(u1 - u0, 1.0), cv)
        return (cu, v0 - 0.05 * max(v1 - v0, 1.0))
    if isinstance(ent, CircleEntity):
        return (ent.center[0] + ent.radius, ent.center[1])
    return (0.0, 0.0)


def make_line_horizontal(ent: LineEntity) -> None:
    """Force line horizontal (same v), keeping midpoint and length."""
    mid = ent.midpoint()
    L = line_length(ent)
    half = 0.5 * L
    # Preserve general left→right sense of p0→p1 when possible
    if ent.p1[0] >= ent.p0[0]:
        ent.p0 = (mid[0] - half, mid[1])
        ent.p1 = (mid[0] + half, mid[1])
    else:
        ent.p0 = (mid[0] + half, mid[1])
        ent.p1 = (mid[0] - half, mid[1])


def make_line_vertical(ent: LineEntity) -> None:
    """Force line vertical (same u), keeping midpoint and length."""
    mid = ent.midpoint()
    L = line_length(ent)
    half = 0.5 * L
    if ent.p1[1] >= ent.p0[1]:
        ent.p0 = (mid[0], mid[1] - half)
        ent.p1 = (mid[0], mid[1] + half)
    else:
        ent.p0 = (mid[0], mid[1] + half)
        ent.p1 = (mid[0], mid[1] - half)


def make_lines_equal_length(source: LineEntity, target: LineEntity) -> None:
    """Set ``target`` length to match ``source`` (source unchanged)."""
    set_line_length(target, line_length(source), free_end="p1")


def snapshot_entity(ent: SketchEntity) -> dict:
    """Serializable snapshot of a sketch entity (for history / clipboard)."""
    if isinstance(ent, LineEntity):
        return {
            "kind": "line",
            "id": int(ent.id),
            "p0": (float(ent.p0[0]), float(ent.p0[1])),
            "p1": (float(ent.p1[0]), float(ent.p1[1])),
        }
    if isinstance(ent, RectEntity):
        return {
            "kind": "rect",
            "id": int(ent.id),
            "c0": (float(ent.c0[0]), float(ent.c0[1])),
            "c1": (float(ent.c1[0]), float(ent.c1[1])),
        }
    if isinstance(ent, CircleEntity):
        return {
            "kind": "circle",
            "id": int(ent.id),
            "center": (float(ent.center[0]), float(ent.center[1])),
            "radius": float(ent.radius),
        }
    raise TypeError(f"unsupported entity type {type(ent)!r}")


def snapshot_dimension(dim: SketchDimension) -> dict:
    """Serializable snapshot of a driving dimension."""
    kind = dim.kind if isinstance(dim.kind, DimKind) else DimKind.LINEAR
    return {
        "id": int(dim.id),
        "kind": kind.name if isinstance(kind, DimKind) else str(kind),
        "entity_id": int(dim.entity_id),
        "role": str(dim.role),
        "value_mm": float(dim.value_mm),
        "entity_b_id": int(getattr(dim, "entity_b_id", -1)),
        "pivot_h0": str(getattr(dim, "pivot_h0", "") or ""),
        "pivot_h1": str(getattr(dim, "pivot_h1", "") or ""),
    }


def restore_dimension(data: dict) -> SketchDimension:
    """Rebuild a SketchDimension from snapshot_dimension() output."""
    kind_raw = data.get("kind", "LINEAR")
    role = str(data.get("role", "length"))
    if isinstance(kind_raw, DimKind):
        kind = kind_raw
    else:
        try:
            kind = DimKind[str(kind_raw)]
        except KeyError:
            if role == "diameter":
                kind = DimKind.DIAMETER
            elif role == "angle":
                kind = DimKind.ANGULAR
            else:
                kind = DimKind.LINEAR
    return SketchDimension(
        id=int(data["id"]),
        kind=kind,
        entity_id=int(data["entity_id"]),
        role=role,
        value_mm=float(data.get("value_mm", 0.0)),
        entity_b_id=int(data.get("entity_b_id", -1)),
        pivot_h0=str(data.get("pivot_h0", "") or ""),
        pivot_h1=str(data.get("pivot_h1", "") or ""),
    )


def snapshot_sketch_contents(sk: Sketch) -> dict:
    """Full entity + dimension + constraint state for undo / bulk mutations."""
    from cadcore.constraints import snapshot_constraint

    return {
        "entities": [snapshot_entity(e) for e in sk.entities],
        "dimensions": [snapshot_dimension(d) for d in sk.dimensions],
        "constraints": [snapshot_constraint(c) for c in (sk.constraints or [])],
        "_next_entity_id": int(sk._next_entity_id),
        "_next_dim_id": int(sk._next_dim_id),
        "_next_constraint_id": int(getattr(sk, "_next_constraint_id", 1)),
    }


def restore_sketch_contents(sk: Sketch, data: dict) -> None:
    """Replace sketch entities/dimensions/constraints from snapshot_sketch_contents()."""
    from cadcore.constraints import restore_constraint

    sk.entities.clear()
    sk.dimensions.clear()
    sk.constraints = []
    for ed in data.get("entities") or []:
        sk.entities.append(restore_entity(ed))
    for dd in data.get("dimensions") or []:
        sk.dimensions.append(restore_dimension(dd))
    for cd in data.get("constraints") or []:
        sk.constraints.append(restore_constraint(cd))
    sk._next_entity_id = int(data.get("_next_entity_id", sk._next_entity_id))
    sk._next_dim_id = int(data.get("_next_dim_id", sk._next_dim_id))
    sk._next_constraint_id = int(
        data.get("_next_constraint_id", getattr(sk, "_next_constraint_id", 1))
    )
    if sk.constraints:
        sk._next_constraint_id = max(
            sk._next_constraint_id, max(c.id for c in sk.constraints) + 1
        )


def restore_entity(data: dict) -> SketchEntity:
    """Rebuild an entity from snapshot_entity() output."""
    kind = data["kind"]
    eid = int(data["id"])
    if kind == "line":
        return LineEntity(
            id=eid, kind=EntityKind.LINE, p0=tuple(data["p0"]), p1=tuple(data["p1"])  # type: ignore[arg-type]
        )
    if kind == "rect":
        return RectEntity(
            id=eid, kind=EntityKind.RECTANGLE, c0=tuple(data["c0"]), c1=tuple(data["c1"])  # type: ignore[arg-type]
        )
    if kind == "circle":
        return CircleEntity(
            id=eid,
            kind=EntityKind.CIRCLE,
            center=tuple(data["center"]),  # type: ignore[arg-type]
            radius=float(data["radius"]),
        )
    raise ValueError(f"unknown entity kind {kind!r}")


def offset_entity_data(data: dict, du: float, dv: float) -> dict:
    """Return a copy of entity snapshot translated by (du, dv)."""
    out = dict(data)
    if out["kind"] == "line":
        p0, p1 = out["p0"], out["p1"]
        out["p0"] = (p0[0] + du, p0[1] + dv)
        out["p1"] = (p1[0] + du, p1[1] + dv)
    elif out["kind"] == "rect":
        c0, c1 = out["c0"], out["c1"]
        out["c0"] = (c0[0] + du, c0[1] + dv)
        out["c1"] = (c1[0] + du, c1[1] + dv)
    elif out["kind"] == "circle":
        c = out["center"]
        out["center"] = (c[0] + du, c[1] + dv)
    return out


def apply_entity_snapshot(ent: SketchEntity, data: dict) -> None:
    """Overwrite geometry of ``ent`` from a snapshot (same id/kind)."""
    if isinstance(ent, LineEntity) and data["kind"] == "line":
        ent.p0 = (float(data["p0"][0]), float(data["p0"][1]))
        ent.p1 = (float(data["p1"][0]), float(data["p1"][1]))
    elif isinstance(ent, RectEntity) and data["kind"] == "rect":
        ent.c0 = (float(data["c0"][0]), float(data["c0"][1]))
        ent.c1 = (float(data["c1"][0]), float(data["c1"][1]))
    elif isinstance(ent, CircleEntity) and data["kind"] == "circle":
        ent.center = (float(data["center"][0]), float(data["center"][1]))
        ent.radius = max(1e-9, float(data["radius"]))
    else:
        raise ValueError("snapshot kind mismatch")
