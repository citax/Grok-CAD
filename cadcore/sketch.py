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
class Sketch:
    name: str = "Sketch"
    plane_feature_id: int = -1
    frame: PlaneFrame = field(default_factory=lambda: PlaneFrame.from_plane_type("FRONT"))
    entities: List[SketchEntity] = field(default_factory=list)
    _next_entity_id: int = 1

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
