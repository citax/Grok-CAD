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
    make_box,
    make_cylinder,
    make_sphere,
)
from cadcore.sketch import PlaneFrame, Sketch


class FeatureType(Enum):
    PLANE_FRONT = auto()  # XY
    PLANE_TOP = auto()  # XZ
    PLANE_RIGHT = auto()  # YZ
    SKETCH = auto()
    # Kept for kernel / future extrude path (not creatable from UI)
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
    visible: bool = True
    suppressed: bool = False


@dataclass
class Document:
    name: str = "Untitled"
    features: List[Feature] = field(default_factory=list)
    selected_id: int = -1
    _next_id: int = 1
    _sketch_count: int = 0

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

    def evaluate_feature(self, fid: int) -> Optional[Mesh]:
        f = self.find(fid)
        if f is None or f.suppressed or is_reference_plane(f.type):
            return None
        if f.type is FeatureType.SKETCH:
            return None  # sketches are 2D curves, not solid meshes yet
        if f.type is FeatureType.BOX:
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
