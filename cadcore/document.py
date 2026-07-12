"""Document + feature history (no GUI)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np

from cadcore.mesh import (
    BooleanOp,
    Mesh,
    boolean_op,
    extrude_filleted_profile,
    extrude_profile,
    make_box,
    make_cylinder,
    make_sphere,
    revolve_profile,
)
from cadcore.sketch import (
    CircleEntity,
    LineEntity,
    PlaneFrame,
    RectEntity,
    Sketch,
    SketchEntity,
)


class FeatureType(Enum):
    PLANE_FRONT = auto()  # XY
    PLANE_TOP = auto()  # XZ
    PLANE_RIGHT = auto()  # YZ
    SKETCH = auto()
    EXTRUDE = auto()  # pad closed sketch profile along plane normal
    REVOLVE = auto()  # revolve closed sketch profile about in-plane axis
    FILLET = auto()  # fillet closed profile corners, then extrude
    # Kernel primitives (not primary UI path)
    BOX = auto()
    SPHERE = auto()
    CYLINDER = auto()
    BOOLEAN_UNION = auto()
    BOOLEAN_DIFFERENCE = auto()
    BOOLEAN_INTERSECTION = auto()


def feature_type_name(t: FeatureType) -> str:
    return {
        FeatureType.PLANE_FRONT: "Front Plane",
        FeatureType.PLANE_TOP: "Top Plane",
        FeatureType.PLANE_RIGHT: "Right Plane",
        FeatureType.SKETCH: "Sketch",
        FeatureType.EXTRUDE: "Extrude",
        FeatureType.REVOLVE: "Revolve",
        FeatureType.FILLET: "Fillet",
        FeatureType.BOX: "Box",
        FeatureType.SPHERE: "Sphere",
        FeatureType.CYLINDER: "Cylinder",
        FeatureType.BOOLEAN_UNION: "Union",
        FeatureType.BOOLEAN_DIFFERENCE: "Difference",
        FeatureType.BOOLEAN_INTERSECTION: "Intersection",
    }.get(t, "Unknown")


def is_reference_plane(t: FeatureType) -> bool:
    return t in (
        FeatureType.PLANE_FRONT,
        FeatureType.PLANE_TOP,
        FeatureType.PLANE_RIGHT,
    )


def is_boolean(t: FeatureType) -> bool:
    return t in (
        FeatureType.BOOLEAN_UNION,
        FeatureType.BOOLEAN_DIFFERENCE,
        FeatureType.BOOLEAN_INTERSECTION,
    )


def plane_frame_for_feature(f: "Feature") -> PlaneFrame:
    return PlaneFrame.from_plane_type(f.type.name)


def is_closed_profile(entity: SketchEntity) -> bool:
    """True for Rectangle / Circle sketch entities with positive area."""
    if isinstance(entity, RectEntity):
        u0, u1 = sorted([entity.c0[0], entity.c1[0]])
        v0, v1 = sorted([entity.c0[1], entity.c1[1]])
        return (u1 - u0) > 1e-12 and (v1 - v0) > 1e-12
    if isinstance(entity, CircleEntity):
        return entity.radius > 1e-12
    return False


def first_closed_profile(sketch: Sketch) -> Optional[SketchEntity]:
    for e in sketch.entities:
        if is_closed_profile(e):
            return e
    return None


def copy_sketch(sk: Optional[Sketch]) -> Optional[Sketch]:
    """Deep copy of a sketch (safe for worker threads)."""
    if sk is None:
        return None
    out = Sketch(
        name=sk.name,
        plane_feature_id=sk.plane_feature_id,
        frame=PlaneFrame(
            np.asarray(sk.frame.origin, dtype=np.float64).copy(),
            np.asarray(sk.frame.u_axis, dtype=np.float64).copy(),
            np.asarray(sk.frame.v_axis, dtype=np.float64).copy(),
            np.asarray(sk.frame.normal, dtype=np.float64).copy(),
        ),
        entities=[],
        _next_entity_id=sk._next_entity_id,
    )
    for e in sk.entities:
        if isinstance(e, LineEntity):
            out.entities.append(
                LineEntity(id=e.id, kind=e.kind, p0=tuple(e.p0), p1=tuple(e.p1))  # type: ignore[arg-type]
            )
        elif isinstance(e, RectEntity):
            out.entities.append(
                RectEntity(id=e.id, kind=e.kind, c0=tuple(e.c0), c1=tuple(e.c1))  # type: ignore[arg-type]
            )
        elif isinstance(e, CircleEntity):
            out.entities.append(
                CircleEntity(
                    id=e.id, kind=e.kind, center=tuple(e.center), radius=float(e.radius)  # type: ignore[arg-type]
                )
            )
    return out


def sketch_fingerprint(sk: Optional[Sketch]) -> str:
    if sk is None:
        return ""
    parts = [sk.name, str(sk.plane_feature_id)]
    o = sk.frame.origin
    n = sk.frame.normal
    parts.append(f"{o[0]:.6g},{o[1]:.6g},{o[2]:.6g}")
    parts.append(f"{n[0]:.6g},{n[1]:.6g},{n[2]:.6g}")
    for e in sk.entities:
        if isinstance(e, LineEntity):
            parts.append(f"L{e.id}:{e.p0[0]:.6g},{e.p0[1]:.6g},{e.p1[0]:.6g},{e.p1[1]:.6g}")
        elif isinstance(e, RectEntity):
            parts.append(f"R{e.id}:{e.c0[0]:.6g},{e.c0[1]:.6g},{e.c1[0]:.6g},{e.c1[1]:.6g}")
        elif isinstance(e, CircleEntity):
            parts.append(
                f"C{e.id}:{e.center[0]:.6g},{e.center[1]:.6g},{e.radius:.6g}"
            )
    return ";".join(parts)


@dataclass
class Feature:
    id: int = -1
    name: str = ""
    type: FeatureType = FeatureType.SKETCH
    # Solid params (kernel only)
    width: float = 1.0
    height: float = 1.0
    depth: float = 1.0
    radius: float = 0.5
    segments: int = 16
    rings: int = 10
    operand_a: int = -1
    operand_b: int = -1
    translation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Sketch linkage
    plane_id: int = -1
    sketch: Optional[Sketch] = None
    # Extrude (pad): operand_a = source sketch id, depth = distance,
    # profile_entity_id = closed entity (or -1 → first closed profile)
    profile_entity_id: int = -1
    # Revolve: operand_a = sketch id, revolve_angle in degrees (default 360),
    # axis in sketch UV (origin + direction); segments = angular resolution
    revolve_angle: float = 360.0
    axis_origin: Tuple[float, float] = (0.0, 0.0)
    axis_direction: Tuple[float, float] = (0.0, 1.0)  # default: sketch V-axis
    visible: bool = True
    suppressed: bool = False


@dataclass
class Document:
    name: str = "Untitled"
    features: List[Feature] = field(default_factory=list)
    selected_id: int = -1
    _next_id: int = 1
    _sketch_count: int = 0
    _extrude_count: int = 0
    _revolve_count: int = 0
    _fillet_count: int = 0

    def add_feature(self, f: Feature) -> int:
        f.id = self._next_id
        self._next_id += 1
        if not f.name:
            f.name = feature_type_name(f.type)
            if not is_reference_plane(f.type):
                f.name = f"{f.name} {f.id}"
        self.features.append(f)
        self.selected_id = f.id
        return f.id

    def find(self, fid: int) -> Optional[Feature]:
        for f in self.features:
            if f.id == fid:
                return f
        return None

    def remove_feature(self, fid: int) -> bool:
        f = self.find(fid)
        if f is None or is_reference_plane(f.type):
            return False
        self.features = [x for x in self.features if x.id != fid]
        if self.selected_id == fid:
            self.selected_id = self.features[0].id if self.features else -1
        return True

    def clear(self) -> None:
        self.features.clear()
        self._next_id = 1
        self.selected_id = -1
        self.name = "Untitled"
        self._sketch_count = 0
        self._extrude_count = 0
        self._revolve_count = 0
        self._fillet_count = 0

    def seed_reference_planes(self) -> None:
        have = {f.type for f in self.features}
        order = (
            (FeatureType.PLANE_FRONT, "Front Plane"),
            (FeatureType.PLANE_TOP, "Top Plane"),
            (FeatureType.PLANE_RIGHT, "Right Plane"),
        )
        for t, name in order:
            if t not in have:
                self.add_feature(Feature(type=t, name=name))
        for f in self.features:
            if f.type is FeatureType.PLANE_FRONT:
                self.selected_id = f.id
                break

    def create_sketch_on_plane(self, plane_id: int) -> Optional[Feature]:
        plane = self.find(plane_id)
        if plane is None or not is_reference_plane(plane.type):
            return None
        self._sketch_count += 1
        sketch = Sketch(
            name=f"Sketch{self._sketch_count}",
            plane_feature_id=plane_id,
            frame=plane_frame_for_feature(plane),
        )
        f = Feature(
            type=FeatureType.SKETCH,
            name=sketch.name,
            plane_id=plane_id,
            sketch=sketch,
        )
        self.add_feature(f)
        return f

    def create_extrude(
        self,
        sketch_id: int,
        distance: float,
        *,
        profile_entity_id: int = -1,
        segments: int = 64,
    ) -> Feature:
        """Pad a closed sketch profile (rectangle or circle) by ``distance``.

        Raises ValueError for missing sketch, open/degenerate profiles, or bad distance.
        """
        skf = self.find(sketch_id)
        if skf is None or skf.type is not FeatureType.SKETCH or skf.sketch is None:
            raise ValueError("extrude requires a valid sketch feature")
        sketch = skf.sketch
        if profile_entity_id >= 0:
            ent = sketch.find_entity(profile_entity_id)
            if ent is None:
                raise ValueError(f"sketch has no entity id={profile_entity_id}")
        else:
            ent = first_closed_profile(sketch)
            if ent is None:
                raise ValueError("sketch has no closed profile (rectangle or circle)")
            profile_entity_id = ent.id
        if not is_closed_profile(ent):
            raise ValueError("profile is not a closed rectangle/circle (or is degenerate)")
        dist = float(distance)
        if not np.isfinite(dist) or dist <= 1e-12:
            raise ValueError("extrude distance must be a positive finite number")
        # Validate by building once (also catches open types)
        _ = extrude_profile(ent, dist, sketch.frame, segments=int(segments))
        self._extrude_count += 1
        f = Feature(
            type=FeatureType.EXTRUDE,
            name=f"Extrude{self._extrude_count}",
            depth=dist,
            segments=int(segments),
            operand_a=sketch_id,
            profile_entity_id=int(profile_entity_id),
        )
        self.add_feature(f)
        return f

    def create_revolve(
        self,
        sketch_id: int,
        *,
        angle_degrees: float = 360.0,
        axis_origin: Tuple[float, float] = (0.0, 0.0),
        axis_direction: Tuple[float, float] = (0.0, 1.0),
        profile_entity_id: int = -1,
        segments: int = 64,
    ) -> Feature:
        """Revolve a closed sketch profile about an in-plane axis.

        Raises ValueError for missing sketch, open/degenerate/axis-crossing profiles,
        or non-positive angle.
        """
        skf = self.find(sketch_id)
        if skf is None or skf.type is not FeatureType.SKETCH or skf.sketch is None:
            raise ValueError("revolve requires a valid sketch feature")
        sketch = skf.sketch
        if profile_entity_id >= 0:
            ent = sketch.find_entity(profile_entity_id)
            if ent is None:
                raise ValueError(f"sketch has no entity id={profile_entity_id}")
        else:
            ent = first_closed_profile(sketch)
            if ent is None:
                raise ValueError("sketch has no closed profile (rectangle or circle)")
            profile_entity_id = ent.id
        if not is_closed_profile(ent):
            raise ValueError("profile is not a closed rectangle/circle (or is degenerate)")
        ang = float(angle_degrees)
        if not np.isfinite(ang) or ang <= 1e-12:
            raise ValueError("revolve angle must be a positive finite number (degrees)")
        ax_o = (float(axis_origin[0]), float(axis_origin[1]))
        ax_d = (float(axis_direction[0]), float(axis_direction[1]))
        # Validate by building once
        _ = revolve_profile(
            ent,
            sketch.frame,
            axis_origin=ax_o,
            axis_direction=ax_d,
            angle_degrees=ang,
            segments=int(segments),
        )
        self._revolve_count += 1
        f = Feature(
            type=FeatureType.REVOLVE,
            name=f"Revolve{self._revolve_count}",
            segments=int(segments),
            operand_a=sketch_id,
            profile_entity_id=int(profile_entity_id),
            revolve_angle=ang,
            axis_origin=ax_o,
            axis_direction=ax_d,
        )
        self.add_feature(f)
        return f

    def create_fillet(
        self,
        sketch_id: int,
        distance: float,
        radius: float,
        *,
        segments: int = 32,
        profile_entity_id: int = -1,
    ) -> Feature:
        """Fillet a closed sketch profile by ``radius``, then extrude by ``distance``.

        Raises ValueError for open profiles, non-positive radius, radius too large
        for the profile (self-intersection), or bad distance.
        """
        skf = self.find(sketch_id)
        if skf is None or skf.type is not FeatureType.SKETCH or skf.sketch is None:
            raise ValueError("fillet requires a valid sketch feature")
        sketch = skf.sketch
        if profile_entity_id >= 0:
            ent = sketch.find_entity(profile_entity_id)
            if ent is None:
                raise ValueError(f"sketch has no entity id={profile_entity_id}")
        else:
            ent = first_closed_profile(sketch)
            if ent is None:
                raise ValueError("sketch has no closed profile (rectangle or circle)")
            profile_entity_id = ent.id
        if not is_closed_profile(ent):
            raise ValueError("open profile: not a closed rectangle/circle")
        dist = float(distance)
        if not np.isfinite(dist) or dist <= 1e-12:
            raise ValueError("extrude distance must be a positive finite number")
        rad = float(radius)
        if not np.isfinite(rad) or rad <= 1e-12:
            raise ValueError("fillet radius must be positive (radius <= 0 is invalid)")
        # Validate by building once (catches r too large / open)
        _ = extrude_filleted_profile(
            ent,
            dist,
            sketch.frame,
            rad,
            segments=int(segments),
        )
        self._fillet_count += 1
        f = Feature(
            type=FeatureType.FILLET,
            name=f"Fillet{self._fillet_count}",
            depth=dist,
            radius=rad,
            segments=int(segments),
            operand_a=sketch_id,
            profile_entity_id=int(profile_entity_id),
        )
        self.add_feature(f)
        return f

    def evaluate_feature(self, fid: int) -> Optional[Mesh]:
        f = self.find(fid)
        if f is None or f.suppressed or is_reference_plane(f.type):
            return None
        if f.type is FeatureType.SKETCH:
            return None  # sketches are 2D curves, not solid meshes
        if f.type is FeatureType.EXTRUDE:
            skf = self.find(f.operand_a)
            if skf is None or skf.sketch is None:
                return None
            sketch = skf.sketch
            if f.profile_entity_id >= 0:
                ent = sketch.find_entity(f.profile_entity_id)
            else:
                ent = first_closed_profile(sketch)
            if ent is None:
                return None
            mesh = extrude_profile(
                ent, f.depth, sketch.frame, segments=max(3, int(f.segments))
            )
        elif f.type is FeatureType.FILLET:
            skf = self.find(f.operand_a)
            if skf is None or skf.sketch is None:
                return None
            sketch = skf.sketch
            if f.profile_entity_id >= 0:
                ent = sketch.find_entity(f.profile_entity_id)
            else:
                ent = first_closed_profile(sketch)
            if ent is None:
                return None
            mesh = extrude_filleted_profile(
                ent,
                f.depth,
                sketch.frame,
                f.radius,
                segments=max(3, int(f.segments)),
            )
        elif f.type is FeatureType.REVOLVE:
            skf = self.find(f.operand_a)
            if skf is None or skf.sketch is None:
                return None
            sketch = skf.sketch
            if f.profile_entity_id >= 0:
                ent = sketch.find_entity(f.profile_entity_id)
            else:
                ent = first_closed_profile(sketch)
            if ent is None:
                return None
            mesh = revolve_profile(
                ent,
                sketch.frame,
                axis_origin=f.axis_origin,
                axis_direction=f.axis_direction,
                angle_degrees=f.revolve_angle,
                segments=max(3, int(f.segments)),
            )
        elif f.type is FeatureType.BOX:
            mesh = make_box(f.width, f.height, f.depth)
        elif f.type is FeatureType.SPHERE:
            mesh = make_sphere(f.radius, f.segments, f.rings)
        elif f.type is FeatureType.CYLINDER:
            mesh = make_cylinder(f.radius, f.height, f.segments)
        elif is_boolean(f.type):
            ma = self.evaluate_feature(f.operand_a)
            mb = self.evaluate_feature(f.operand_b)
            if ma is None or mb is None:
                return None
            op = {
                FeatureType.BOOLEAN_UNION: BooleanOp.UNION,
                FeatureType.BOOLEAN_DIFFERENCE: BooleanOp.DIFFERENCE,
                FeatureType.BOOLEAN_INTERSECTION: BooleanOp.INTERSECTION,
            }[f.type]
            mesh = boolean_op(ma, mb, op)
        else:
            return None
        if f.translation != (0.0, 0.0, 0.0):
            mesh = mesh.translate(f.translation)
        return mesh

    def evaluate_display_solids(self) -> Dict[int, Mesh]:
        used = set()
        for f in self.features:
            if is_boolean(f.type):
                if f.operand_a >= 0:
                    used.add(f.operand_a)
                if f.operand_b >= 0:
                    used.add(f.operand_b)
        out: Dict[int, Mesh] = {}
        for f in self.features:
            if not f.visible or f.suppressed or is_reference_plane(f.type):
                continue
            if f.type is FeatureType.SKETCH:
                continue
            if f.id in used:
                continue
            m = self.evaluate_feature(f.id)
            if m is not None and not m.empty:
                out[f.id] = m
        return out
