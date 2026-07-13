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
    extrude_pocketed_profile,
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
    apply_entity_snapshot,
    offset_entity_data,
    restore_entity,
    snapshot_entity,
)
from cadcore.units import Unit


class FeatureType(Enum):
    PLANE_FRONT = auto()  # XY
    PLANE_TOP = auto()  # XZ
    PLANE_RIGHT = auto()  # YZ
    SKETCH = auto()
    EXTRUDE = auto()  # pad closed sketch profile along plane normal
    REVOLVE = auto()  # revolve closed sketch profile about in-plane axis
    FILLET = auto()  # fillet closed profile corners, then extrude
    POCKET = auto()  # circular through-hole pocket then extrude
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
        FeatureType.POCKET: "Pocket",
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


from cadcore.profiles import (  # noqa: E402  — re-export after sketch imports
    ClosedLineLoop,
    is_closed_profile,
    list_closed_profiles,
    point_in_profile,
    profile_polygon_uv,
)


def first_closed_profile(sketch: Sketch) -> Optional[object]:
    """First rectangle/circle entity or detected closed line loop."""
    for e in sketch.entities:
        if is_closed_profile(e):
            return e
    try:
        from cadcore.profiles import find_closed_line_loops

        loops = find_closed_line_loops(sketch)
        return loops[0] if loops else None
    except ValueError:
        return None


def _profile_id(p: object) -> int:
    return int(getattr(p, "id", -1))


def _profile_contains(outer: object, inner: object) -> bool:
    """True if ``inner`` is strictly inside ``outer`` (both closed profiles)."""
    if not is_closed_profile(outer) or not is_closed_profile(inner):
        return False
    if outer is inner or _profile_id(outer) == _profile_id(inner):
        return False
    # Sample key points of inner
    if isinstance(inner, CircleEntity):
        pts = [inner.center]
        # also require full disk: center + margin handled via radius in point_in
        if isinstance(outer, RectEntity):
            return point_in_profile(inner.center, outer, margin=inner.radius + 1e-9)
        if isinstance(outer, CircleEntity):
            d = float(
                np.hypot(
                    inner.center[0] - outer.center[0],
                    inner.center[1] - outer.center[1],
                )
            )
            return d + inner.radius < outer.radius - 1e-9
        if isinstance(outer, ClosedLineLoop):
            return point_in_profile(inner.center, outer)
        return False
    if isinstance(inner, RectEntity):
        pts = inner.corners()
    elif isinstance(inner, ClosedLineLoop):
        pts = list(inner.vertices)
    else:
        return False
    return all(point_in_profile(c, outer, margin=1e-9) for c in pts)


@dataclass
class ResolvedProfiles:
    """Outer boundary + optional hole loops for extrude/pad."""

    outer: object  # SketchEntity | ClosedLineLoop
    holes: List[object] = field(default_factory=list)


def resolve_profiles(
    sketch: Sketch,
    *,
    preferred_outer_id: int = -1,
) -> ResolvedProfiles:
    """Resolve closed sketch entities / line-loops into outer + hole loops.

    - Single closed profile → outer only (backward compatible).
    - Nested (outer contains others) → outer + inners as holes.
    - Disjoint closed profiles (none contains the others) → ValueError
      (ambiguous; GUI should offer a picker).
    - Closed line-segment loops count as profiles (see cadcore.profiles).
    """
    closed = list_closed_profiles(sketch)
    if len(closed) == 1:
        return ResolvedProfiles(outer=closed[0], holes=[])

    # If user picked a specific outer, use it and treat contained profiles as holes
    if preferred_outer_id != -1:
        outer = next((e for e in closed if _profile_id(e) == preferred_outer_id), None)
        # Also: preferred may be a line id belonging to a loop
        if outer is None:
            for e in closed:
                if isinstance(e, ClosedLineLoop) and preferred_outer_id in e.line_ids:
                    outer = e
                    break
        if outer is None:
            raise ValueError(f"sketch has no closed profile id={preferred_outer_id}")
        holes = [
            e
            for e in closed
            if _profile_id(e) != _profile_id(outer) and _profile_contains(outer, e)
        ]
        others = [
            e
            for e in closed
            if _profile_id(e) != _profile_id(outer)
            and e not in holes
            and not _profile_contains(e, outer)
        ]
        if others:
            raise ValueError(
                "ambiguous profiles: multiple disjoint closed profiles; "
                "select which profile to extrude"
            )
        return ResolvedProfiles(outer=outer, holes=holes)

    # Find roots: not contained by any other closed profile
    roots: List[object] = []
    for e in closed:
        if any(
            _profile_contains(o, e)
            for o in closed
            if _profile_id(o) != _profile_id(e)
        ):
            continue
        roots.append(e)

    if len(roots) == 0:
        return ResolvedProfiles(outer=closed[0], holes=[])

    if len(roots) > 1:
        raise ValueError(
            "ambiguous profiles: multiple disjoint closed profiles; "
            "select which profile to extrude"
        )

    outer = roots[0]
    holes = [
        e
        for e in closed
        if _profile_id(e) != _profile_id(outer) and _profile_contains(outer, e)
    ]
    return ResolvedProfiles(outer=outer, holes=holes)


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
    # Pocket: radius = hole radius; hole_center_u/v = hole center in sketch UV
    hole_center_u: float = 0.0
    hole_center_v: float = 0.0
    visible: bool = True
    suppressed: bool = False


# ---------------------------------------------------------------------------
# Undo / redo command stack
# ---------------------------------------------------------------------------

PASTE_UV_DELTA: Tuple[float, float] = (5.0, 5.0)  # mm step between consecutive pastes


class HistoryCommand:
    """Base undoable command."""

    def undo(self, doc: "Document") -> None:
        raise NotImplementedError

    def redo(self, doc: "Document") -> None:
        raise NotImplementedError

    def description(self) -> str:
        return self.__class__.__name__


class EntityAddCommand(HistoryCommand):
    def __init__(self, sketch_id: int, entity_data: dict, index: Optional[int] = None) -> None:
        self.sketch_id = int(sketch_id)
        self.entity_data = dict(entity_data)
        self.index = index

    def redo(self, doc: "Document") -> None:
        sk = _sketch_of(doc, self.sketch_id)
        ent = restore_entity(self.entity_data)
        sk.insert_entity(ent, self.index)

    def undo(self, doc: "Document") -> None:
        sk = _sketch_of(doc, self.sketch_id)
        sk.remove_entity(int(self.entity_data["id"]))

    def description(self) -> str:
        return f"Add {self.entity_data.get('kind', 'entity')}"


class EntityDeleteCommand(HistoryCommand):
    def __init__(self, sketch_id: int, entity_data: dict, index: int) -> None:
        self.sketch_id = int(sketch_id)
        self.entity_data = dict(entity_data)
        self.index = int(index)

    def redo(self, doc: "Document") -> None:
        sk = _sketch_of(doc, self.sketch_id)
        sk.remove_entity(int(self.entity_data["id"]))

    def undo(self, doc: "Document") -> None:
        sk = _sketch_of(doc, self.sketch_id)
        ent = restore_entity(self.entity_data)
        sk.insert_entity(ent, self.index)

    def description(self) -> str:
        return f"Delete {self.entity_data.get('kind', 'entity')}"


class EntityMoveCommand(HistoryCommand):
    def __init__(self, sketch_id: int, before: dict, after: dict) -> None:
        self.sketch_id = int(sketch_id)
        self.before = dict(before)
        self.after = dict(after)
        assert before["id"] == after["id"]

    def redo(self, doc: "Document") -> None:
        ent = _entity_of(doc, self.sketch_id, int(self.after["id"]))
        apply_entity_snapshot(ent, self.after)

    def undo(self, doc: "Document") -> None:
        ent = _entity_of(doc, self.sketch_id, int(self.before["id"]))
        apply_entity_snapshot(ent, self.before)

    def description(self) -> str:
        return "Move entity"


class FeatureAddCommand(HistoryCommand):
    def __init__(self, feature: "Feature", index: Optional[int] = None) -> None:
        self.feature = feature
        self.index = index

    def redo(self, doc: "Document") -> None:
        if doc.find(self.feature.id) is not None:
            return
        if self.index is None or self.index < 0 or self.index > len(doc.features):
            doc.features.append(self.feature)
        else:
            doc.features.insert(self.index, self.feature)
        doc.selected_id = self.feature.id
        doc._next_id = max(doc._next_id, self.feature.id + 1)

    def undo(self, doc: "Document") -> None:
        doc.features = [x for x in doc.features if x.id != self.feature.id]
        if doc.selected_id == self.feature.id:
            doc.selected_id = doc.features[0].id if doc.features else -1

    def description(self) -> str:
        return f"Add feature {self.feature.name}"


class FeatureDeleteCommand(HistoryCommand):
    def __init__(self, feature: "Feature", index: int) -> None:
        self.feature = feature
        self.index = int(index)

    def redo(self, doc: "Document") -> None:
        doc.features = [x for x in doc.features if x.id != self.feature.id]
        if doc.selected_id == self.feature.id:
            doc.selected_id = doc.features[0].id if doc.features else -1

    def undo(self, doc: "Document") -> None:
        if doc.find(self.feature.id) is not None:
            return
        idx = max(0, min(self.index, len(doc.features)))
        doc.features.insert(idx, self.feature)
        doc.selected_id = self.feature.id

    def description(self) -> str:
        return f"Delete feature {self.feature.name}"


def _sketch_of(doc: "Document", sketch_id: int) -> Sketch:
    f = doc.find(sketch_id)
    if f is None or f.sketch is None:
        raise ValueError(f"no sketch feature id={sketch_id}")
    return f.sketch


def _entity_of(doc: "Document", sketch_id: int, eid: int) -> SketchEntity:
    sk = _sketch_of(doc, sketch_id)
    ent = sk.find_entity(eid)
    if ent is None:
        raise ValueError(f"no entity id={eid} in sketch {sketch_id}")
    return ent


def _entity_anchor_uv(data: dict) -> Tuple[float, float]:
    """Anchor UV for paste placement (line p0 / rect c0 / circle center)."""
    kind = data.get("kind")
    if kind == "line":
        p = data["p0"]
        return (float(p[0]), float(p[1]))
    if kind == "rect":
        p = data["c0"]
        return (float(p[0]), float(p[1]))
    if kind == "circle":
        p = data["center"]
        return (float(p[0]), float(p[1]))
    return (0.0, 0.0)


@dataclass
class Document:
    name: str = "Untitled"
    features: List[Feature] = field(default_factory=list)
    selected_id: int = -1
    display_unit: Unit = Unit.MM
    _next_id: int = 1
    _sketch_count: int = 0
    _extrude_count: int = 0
    _revolve_count: int = 0
    _fillet_count: int = 0
    _pocket_count: int = 0
    # History / clipboard
    _undo_stack: List[HistoryCommand] = field(default_factory=list, repr=False)
    _redo_stack: List[HistoryCommand] = field(default_factory=list, repr=False)
    _clipboard: Optional[dict] = field(default=None, repr=False)
    _paste_n: int = 0  # consecutive paste count (reset on copy); advances offset

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
        self._pocket_count = 0
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._clipboard = None
        self._paste_n = 0

    # ----- history -----
    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    def push_command(self, cmd: HistoryCommand, *, already_done: bool = True) -> None:
        """Record a command. If not already_done, redo() is executed first."""
        if not already_done:
            cmd.redo(self)
        self._undo_stack.append(cmd)
        self._redo_stack.clear()

    def undo(self) -> bool:
        """Undo last command. Empty stack → safe no-op (False)."""
        if not self._undo_stack:
            return False
        cmd = self._undo_stack.pop()
        cmd.undo(self)
        self._redo_stack.append(cmd)
        return True

    def redo(self) -> bool:
        """Redo last undone command. Empty stack → safe no-op (False)."""
        if not self._redo_stack:
            return False
        cmd = self._redo_stack.pop()
        cmd.redo(self)
        self._undo_stack.append(cmd)
        return True

    def record_entity_add(self, sketch_id: int, ent: SketchEntity) -> None:
        """Entity already in sketch; push undoable add (clears redo)."""
        sk = _sketch_of(self, sketch_id)
        idx = next((i for i, e in enumerate(sk.entities) if e.id == ent.id), len(sk.entities) - 1)
        self.push_command(EntityAddCommand(sketch_id, snapshot_entity(ent), idx))

    def delete_entity(self, sketch_id: int, eid: int) -> bool:
        """Delete sketch entity (undoable)."""
        sk = _sketch_of(self, sketch_id)
        idx = next((i for i, e in enumerate(sk.entities) if e.id == eid), -1)
        if idx < 0:
            return False
        data = snapshot_entity(sk.entities[idx])
        sk.remove_entity(eid)
        self.push_command(EntityDeleteCommand(sketch_id, data, idx))
        return True

    def record_entity_move(self, sketch_id: int, before: dict, after: dict) -> None:
        """Record geometry change if before != after."""
        if before == after:
            return
        self.push_command(EntityMoveCommand(sketch_id, before, after))

    def record_feature_add(self, f: Feature) -> None:
        """Feature already added; push undoable feature-add."""
        idx = next((i for i, x in enumerate(self.features) if x.id == f.id), len(self.features) - 1)
        self.push_command(FeatureAddCommand(f, idx))

    def delete_feature_undoable(self, fid: int) -> bool:
        """Delete non-plane feature and push undo command."""
        f = self.find(fid)
        if f is None or is_reference_plane(f.type):
            return False
        idx = next(i for i, x in enumerate(self.features) if x.id == fid)
        self.features = [x for x in self.features if x.id != fid]
        if self.selected_id == fid:
            self.selected_id = self.features[0].id if self.features else -1
        self.push_command(FeatureDeleteCommand(f, idx))
        return True

    # ----- clipboard -----
    def copy_entity(self, sketch_id: int, eid: int) -> bool:
        ent = _entity_of(self, sketch_id, eid)
        self._clipboard = snapshot_entity(ent)
        self._paste_n = 0  # next paste starts a new offset sequence
        return True

    def cut_entity(self, sketch_id: int, eid: int) -> bool:
        if not self.copy_entity(sketch_id, eid):
            return False
        return self.delete_entity(sketch_id, eid)

    def paste_entity(
        self,
        sketch_id: int,
        *,
        place_uv: Optional[Tuple[float, float]] = None,
    ) -> Optional[SketchEntity]:
        """Paste clipboard as a new entity (undoable).

        Offset advances cumulatively so repeated Ctrl+V never stacks on itself.
        If ``place_uv`` is given, the entity's anchor (p0 / c0 / center) is placed
        there, then staggered by ``(paste_n-1) * PASTE_UV_DELTA``.
        Without ``place_uv``, offset is ``paste_n * PASTE_UV_DELTA`` from the
        clipboard geometry.
        """
        if self._clipboard is None:
            return None
        sk = _sketch_of(self, sketch_id)
        self._paste_n += 1
        n = self._paste_n
        data = dict(self._clipboard)
        if place_uv is not None:
            anchor = _entity_anchor_uv(data)
            du = float(place_uv[0]) - anchor[0]
            dv = float(place_uv[1]) - anchor[1]
            # Stagger subsequent pastes at the same cursor so they never overlap
            du += (n - 1) * PASTE_UV_DELTA[0]
            dv += (n - 1) * PASTE_UV_DELTA[1]
            data = offset_entity_data(data, du, dv)
        else:
            data = offset_entity_data(
                data, n * PASTE_UV_DELTA[0], n * PASTE_UV_DELTA[1]
            )
        data["id"] = int(sk._next_entity_id)
        ent = restore_entity(data)
        sk.insert_entity(ent)
        self.push_command(EntityAddCommand(sketch_id, snapshot_entity(ent), len(sk.entities) - 1))
        return ent

    def set_display_unit(self, unit: Unit) -> None:
        self.display_unit = Unit(unit)

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
        self.record_feature_add(f)
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
        # Resolve outer + nested holes (or raise if ambiguous disjoint profiles)
        resolved = resolve_profiles(sketch, preferred_outer_id=profile_entity_id)
        ent = resolved.outer
        profile_entity_id = ent.id
        if not is_closed_profile(ent):
            raise ValueError("profile is not a closed rectangle/circle (or is degenerate)")
        dist = float(distance)
        if not np.isfinite(dist) or dist <= 1e-12:
            raise ValueError("extrude distance must be a positive finite number")
        # Validate by building once (also catches open types / holes)
        _ = extrude_profile(
            ent,
            dist,
            sketch.frame,
            segments=int(segments),
            holes=resolved.holes,
        )
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
        self.record_feature_add(f)
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
        self.record_feature_add(f)
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
        self.record_feature_add(f)
        return f

    def create_pocket(
        self,
        sketch_id: int,
        distance: float,
        hole_radius: float,
        hole_center: Tuple[float, float] = (0.0, 0.0),
        *,
        segments: int = 32,
        profile_entity_id: int = -1,
    ) -> Feature:
        """Cut a circular through-hole in a closed sketch profile, then extrude.

        Raises ValueError for bad hole radius, hole out of bounds, edge-touching
        (non-manifold) holes, or open profiles.
        """
        skf = self.find(sketch_id)
        if skf is None or skf.type is not FeatureType.SKETCH or skf.sketch is None:
            raise ValueError("pocket requires a valid sketch feature")
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
        hr = float(hole_radius)
        if not np.isfinite(hr) or hr <= 1e-12:
            raise ValueError("hole radius must be positive (hole_radius <= 0 is invalid)")
        hc = (float(hole_center[0]), float(hole_center[1]))
        _ = extrude_pocketed_profile(
            ent,
            dist,
            sketch.frame,
            hc,
            hr,
            segments=int(segments),
        )
        self._pocket_count += 1
        f = Feature(
            type=FeatureType.POCKET,
            name=f"Pocket{self._pocket_count}",
            depth=dist,
            radius=hr,
            segments=int(segments),
            operand_a=sketch_id,
            profile_entity_id=int(profile_entity_id),
            hole_center_u=hc[0],
            hole_center_v=hc[1],
        )
        self.add_feature(f)
        self.record_feature_add(f)
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
            try:
                resolved = resolve_profiles(
                    sketch, preferred_outer_id=f.profile_entity_id
                )
            except ValueError:
                return None
            mesh = extrude_profile(
                resolved.outer,
                f.depth,
                sketch.frame,
                segments=max(3, int(f.segments)),
                holes=resolved.holes,
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
        elif f.type is FeatureType.POCKET:
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
            mesh = extrude_pocketed_profile(
                ent,
                f.depth,
                sketch.frame,
                (f.hole_center_u, f.hole_center_v),
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
