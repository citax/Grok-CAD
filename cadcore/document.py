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
    DimKind,
    LineEntity,
    PlaneFrame,
    RectEntity,
    Sketch,
    SketchDimension,
    SketchEntity,
    apply_entity_snapshot,
    offset_entity_data,
    restore_dimension,
    restore_entity,
    restore_sketch_contents,
    snapshot_dimension,
    snapshot_entity,
    snapshot_sketch_contents,
)
from cadcore.units import Unit


class FeatureType(Enum):
    PLANE_FRONT = auto()  # XY
    PLANE_TOP = auto()  # XZ
    PLANE_RIGHT = auto()  # YZ
    SKETCH = auto()
    EXTRUDE = auto()  # pad closed sketch profile along plane normal
    REVOLVE = auto()  # revolve closed sketch profile about in-plane axis
    FILLET = auto()  # legacy: fillet closed profile corners, then extrude
    EDGE_FILLET = auto()  # SolidWorks-style: round edges on a solid
    POCKET = auto()  # circular through-hole pocket then extrude
    CUT_EXTRUDE = auto()  # SolidWorks Extruded Cut: remove material from a solid
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
        FeatureType.FILLET: "Fillet (sketch)",
        FeatureType.EDGE_FILLET: "Fillet",
        FeatureType.POCKET: "Pocket",
        FeatureType.CUT_EXTRUDE: "Cut-Extrude",
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


def is_sketch_consuming_feature(t: FeatureType) -> bool:
    """Features that absorb their source sketch (SolidWorks absorbed sketch)."""
    return t in (
        FeatureType.EXTRUDE,
        FeatureType.REVOLVE,
        FeatureType.FILLET,
        FeatureType.POCKET,
        FeatureType.CUT_EXTRUDE,
    )


def is_solid_feature(t: FeatureType) -> bool:
    """True if the feature can produce a triangle mesh with faces to sketch on."""
    return t in (
        FeatureType.EXTRUDE,
        FeatureType.REVOLVE,
        FeatureType.FILLET,
        FeatureType.EDGE_FILLET,
        FeatureType.POCKET,
        FeatureType.CUT_EXTRUDE,
        FeatureType.BOX,
        FeatureType.SPHERE,
        FeatureType.CYLINDER,
        FeatureType.BOOLEAN_UNION,
        FeatureType.BOOLEAN_DIFFERENCE,
        FeatureType.BOOLEAN_INTERSECTION,
    )


def plane_frame_for_feature(f: "Feature") -> PlaneFrame:
    if is_reference_plane(f.type):
        return PlaneFrame.from_plane_type(f.type.name)
    if f.sketch is not None:
        return f.sketch.frame
    return PlaneFrame.from_plane_type("PLANE_FRONT")


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
        # Contained profiles become holes; other disjoint profiles are ignored
        # (the user already chose which outer to extrude via the picker).
        holes = [
            e
            for e in closed
            if _profile_id(e) != _profile_id(outer) and _profile_contains(outer, e)
        ]
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
        dimensions=[],
        _next_entity_id=sk._next_entity_id,
        _next_dim_id=getattr(sk, "_next_dim_id", 1),
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
    for d in getattr(sk, "dimensions", None) or []:
        out.dimensions.append(
            SketchDimension(
                id=int(d.id),
                kind=d.kind if isinstance(d.kind, DimKind) else DimKind.LINEAR,
                entity_id=int(d.entity_id),
                role=str(d.role),
                value_mm=float(d.value_mm),
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
    # Extrude (pad): operand_a = source sketch id, depth = distance (always ≥ 0),
    # profile_entity_id = closed entity (or -1 → first closed profile),
    # reversed = pad along −plane normal (SolidWorks "Reverse Direction"),
    # operand_b = parent solid to merge into when sketch is on a solid face
    #            (−1 = base feature, no merge — standalone body)
    profile_entity_id: int = -1
    reversed: bool = False
    # Revolve: operand_a = sketch id, revolve_angle in degrees (default 360),
    # axis in sketch UV (origin + direction); segments = angular resolution
    revolve_angle: float = 360.0
    axis_origin: Tuple[float, float] = (0.0, 0.0)
    axis_direction: Tuple[float, float] = (0.0, 1.0)  # default: sketch V-axis
    # Pocket: radius = hole radius; hole_center_u/v = hole center in sketch UV
    hole_center_u: float = 0.0
    hole_center_v: float = 0.0
    # Cut-Extrude: operand_a = sketch, operand_b = solid being cut;
    # through_all = extend tool through the whole solid (ignores depth for tool size)
    through_all: bool = False
    # Fillet (sketch): sharp source polygon in sketch UV (parametric radius edits)
    # List of (u, v); not closed (first != last). Empty = resolve from sketch.
    source_profile_uv: List[Tuple[float, float]] = field(default_factory=list)
    # Edge fillet (solid): stable edge keys on the parent solid (operand_a)
    edge_keys: List[str] = field(default_factory=list)
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
    def __init__(
        self,
        sketch_id: int,
        entity_data: dict,
        index: int,
        dimensions: Optional[List[dict]] = None,
    ) -> None:
        self.sketch_id = int(sketch_id)
        self.entity_data = dict(entity_data)
        self.index = int(index)
        self.dimensions = [dict(d) for d in (dimensions or [])]

    def redo(self, doc: "Document") -> None:
        sk = _sketch_of(doc, self.sketch_id)
        eid = int(self.entity_data["id"])
        sk.remove_entity(eid)
        sk.remove_dimensions_for_entity(eid)

    def undo(self, doc: "Document") -> None:
        sk = _sketch_of(doc, self.sketch_id)
        ent = restore_entity(self.entity_data)
        sk.insert_entity(ent, self.index)
        for dd in self.dimensions:
            dim = restore_dimension(dd)
            # Avoid duplicate if already present
            if sk.find_dimension(dim.id) is None:
                sk.dimensions.append(dim)
                sk._next_dim_id = max(sk._next_dim_id, int(dim.id) + 1)

    def description(self) -> str:
        return f"Delete {self.entity_data.get('kind', 'entity')}"


class EntityMultiDeleteCommand(HistoryCommand):
    """Delete several sketch entities as ONE undo step.

    ``items`` is a list of (entity_data, index) captured before deletion, in
    ascending original-index order. ``dimensions`` are all driving dims that
    belonged to those entities (restored on undo).
    """

    def __init__(
        self,
        sketch_id: int,
        items: List[Tuple[dict, int]],
        dimensions: Optional[List[dict]] = None,
    ) -> None:
        self.sketch_id = int(sketch_id)
        self.items = [(dict(d), int(idx)) for d, idx in items]
        self.dimensions = [dict(d) for d in (dimensions or [])]

    def redo(self, doc: "Document") -> None:
        sk = _sketch_of(doc, self.sketch_id)
        eids = {int(data["id"]) for data, _idx in self.items}
        for data, _idx in self.items:
            sk.remove_entity(int(data["id"]))
        sk.dimensions = [d for d in sk.dimensions if int(d.entity_id) not in eids]

    def undo(self, doc: "Document") -> None:
        sk = _sketch_of(doc, self.sketch_id)
        for data, idx in sorted(self.items, key=lambda t: t[1]):
            ent = restore_entity(data)
            sk.insert_entity(ent, idx)
        for dd in self.dimensions:
            dim = restore_dimension(dd)
            if sk.find_dimension(dim.id) is None:
                sk.dimensions.append(dim)
                sk._next_dim_id = max(sk._next_dim_id, int(dim.id) + 1)

    def description(self) -> str:
        return f"Delete {len(self.items)} entities"


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


class DimensionApplyCommand(HistoryCommand):
    """Undoable driving-dimension apply: geometry + dimension record together."""

    def __init__(
        self,
        sketch_id: int,
        entity_before: dict,
        entity_after: dict,
        dim_before: Optional[dict],
        dim_after: dict,
    ) -> None:
        self.sketch_id = int(sketch_id)
        self.entity_before = dict(entity_before)
        self.entity_after = dict(entity_after)
        self.dim_before = dict(dim_before) if dim_before is not None else None
        self.dim_after = dict(dim_after)

    def _set_dim(self, sk: Sketch, data: Optional[dict], *, fallback_remove_id: int) -> None:
        role = str(self.dim_after.get("role", "length"))
        eid = int(self.entity_after["id"])
        # Drop any current dim for this entity/role
        sk.dimensions = [
            d
            for d in sk.dimensions
            if not (int(d.entity_id) == eid and str(d.role) == role)
        ]
        if data is None:
            return
        dim = restore_dimension(data)
        if sk.find_dimension(dim.id) is not None:
            # replace by id
            sk.dimensions = [d for d in sk.dimensions if d.id != dim.id]
        sk.dimensions.append(dim)
        sk._next_dim_id = max(sk._next_dim_id, int(dim.id) + 1)

    def redo(self, doc: "Document") -> None:
        sk = _sketch_of(doc, self.sketch_id)
        ent = _entity_of(doc, self.sketch_id, int(self.entity_after["id"]))
        apply_entity_snapshot(ent, self.entity_after)
        self._set_dim(sk, self.dim_after, fallback_remove_id=int(self.dim_after["id"]))

    def undo(self, doc: "Document") -> None:
        sk = _sketch_of(doc, self.sketch_id)
        ent = _entity_of(doc, self.sketch_id, int(self.entity_before["id"]))
        apply_entity_snapshot(ent, self.entity_before)
        self._set_dim(
            sk,
            self.dim_before,
            fallback_remove_id=int(self.dim_after["id"]),
        )

    def description(self) -> str:
        return "Apply dimension"


class FilletCreateCommand(HistoryCommand):
    """Add fillet feature and sketch mutation as one undoable step."""

    def __init__(
        self,
        feature: "Feature",
        sketch_id: int,
        sketch_before: dict,
        sketch_after: dict,
        index: Optional[int] = None,
    ) -> None:
        self.feature = feature
        self.sketch_id = int(sketch_id)
        self.sketch_before = dict(sketch_before)
        self.sketch_after = dict(sketch_after)
        self.index = index

    def redo(self, doc: "Document") -> None:
        sk = _sketch_of(doc, self.sketch_id)
        restore_sketch_contents(sk, self.sketch_after)
        if doc.find(self.feature.id) is None:
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
        sk = _sketch_of(doc, self.sketch_id)
        restore_sketch_contents(sk, self.sketch_before)

    def description(self) -> str:
        return f"Add fillet {self.feature.name}"


class FeatureMultiDeleteCommand(HistoryCommand):
    """Delete several features (e.g. sketch + dependents) as one undo step.

    ``items`` is (feature, index) in ascending original-index order.
    """

    def __init__(self, items: List[Tuple["Feature", int]]) -> None:
        self.items = list(items)

    def redo(self, doc: "Document") -> None:
        ids = {f.id for f, _ in self.items}
        doc.features = [x for x in doc.features if x.id not in ids]
        if doc.selected_id in ids:
            doc.selected_id = doc.features[0].id if doc.features else -1

    def undo(self, doc: "Document") -> None:
        for f, idx in sorted(self.items, key=lambda t: t[1]):
            if doc.find(f.id) is not None:
                continue
            i = max(0, min(idx, len(doc.features)))
            doc.features.insert(i, f)
            doc._next_id = max(doc._next_id, f.id + 1)
        if self.items:
            doc.selected_id = self.items[0][0].id

    def description(self) -> str:
        return f"Delete {len(self.items)} features"


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


class FeatureParamCommand(HistoryCommand):
    """Undoable edit of solid feature parameters (depth, radius, angle, …)."""

    def __init__(self, feature_id: int, before: dict, after: dict) -> None:
        self.feature_id = int(feature_id)
        self.before = dict(before)
        self.after = dict(after)

    def _apply(self, doc: "Document", data: dict) -> None:
        f = doc.find(self.feature_id)
        if f is None:
            return
        for k, v in data.items():
            if hasattr(f, k):
                setattr(f, k, v if k != "source_profile_uv" else list(v))
        # Keep sketch fillet polyline in sync with radius when undoing/redoing
        if (
            f.type is FeatureType.FILLET
            and f.source_profile_uv
            and len(f.source_profile_uv) >= 3
        ):
            from cadcore.fillet2d import fillet_closed_polygon

            skf = doc.find(f.operand_a)
            if skf is not None and skf.sketch is not None:
                try:
                    poly = fillet_closed_polygon(
                        f.source_profile_uv,
                        float(f.radius),
                        arc_segments=max(6, int(f.segments) // 4),
                    )
                    _clear_sketch_entities(skf.sketch)
                    _add_polyline_lines(skf.sketch, poly)
                except ValueError:
                    pass

    def undo(self, doc: "Document") -> None:
        self._apply(doc, self.before)

    def redo(self, doc: "Document") -> None:
        self._apply(doc, self.after)

    def description(self) -> str:
        return "Edit feature parameters"


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


def _clear_sketch_entities(sketch: Sketch) -> None:
    sketch.entities.clear()


def _add_polyline_lines(sketch: Sketch, poly: List[Tuple[float, float]]) -> None:
    """Add consecutive line segments around a closed UV ring (no repeated last)."""
    n = len(poly)
    if n < 2:
        return
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        sketch.add_line(
            (float(a[0]), float(a[1])),
            (float(b[0]), float(b[1])),
        )


def _replace_sketch_profile_with_polyline(
    sketch: Sketch, profile: object, poly: List[Tuple[float, float]]
) -> None:
    """Remove sharp profile geometry and insert filleted polyline as lines.

    Sharp corner vertices disappear from the sketch (SolidWorks sketch-fillet look).
    """
    from cadcore.profiles import ClosedLineLoop
    from cadcore.sketch import RectEntity

    if isinstance(profile, RectEntity):
        sketch.remove_entity(profile.id)
    elif isinstance(profile, ClosedLineLoop):
        for lid in getattr(profile, "line_ids", ()) or ():
            sketch.remove_entity(int(lid))
    else:
        # Fallback: clear everything if we can't identify members
        eid = getattr(profile, "id", None)
        if eid is not None and int(eid) >= 0:
            sketch.remove_entity(int(eid))
    _add_polyline_lines(sketch, poly)


@dataclass
class Document:
    name: str = "Untitled"
    features: List[Feature] = field(default_factory=list)
    selected_id: int = -1
    display_unit: Unit = Unit.MM
    # Unsaved changes since last Save / Open / New
    dirty: bool = False
    _next_id: int = 1
    _sketch_count: int = 0
    _extrude_count: int = 0
    _revolve_count: int = 0
    _fillet_count: int = 0
    _pocket_count: int = 0
    _cut_count: int = 0
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
        self.dirty = False
        self._sketch_count = 0
        self._extrude_count = 0
        self._revolve_count = 0
        self._fillet_count = 0
        self._pocket_count = 0
        self._cut_count = 0
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._clipboard = None
        self._paste_n = 0

    def mark_dirty(self) -> None:
        self.dirty = True

    def mark_clean(self) -> None:
        self.dirty = False

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
        self.dirty = True

    def undo(self) -> bool:
        """Undo last command. Empty stack → safe no-op (False)."""
        if not self._undo_stack:
            return False
        cmd = self._undo_stack.pop()
        cmd.undo(self)
        self._redo_stack.append(cmd)
        self.dirty = True
        return True

    def redo(self) -> bool:
        """Redo last undone command. Empty stack → safe no-op (False)."""
        if not self._redo_stack:
            return False
        cmd = self._redo_stack.pop()
        cmd.redo(self)
        self._undo_stack.append(cmd)
        self.dirty = True
        return True

    def record_entity_add(self, sketch_id: int, ent: SketchEntity) -> None:
        """Entity already in sketch; push undoable add (clears redo)."""
        sk = _sketch_of(self, sketch_id)
        idx = next((i for i, e in enumerate(sk.entities) if e.id == ent.id), len(sk.entities) - 1)
        self.push_command(EntityAddCommand(sketch_id, snapshot_entity(ent), idx))

    def delete_entity(self, sketch_id: int, eid: int) -> bool:
        """Delete sketch entity (undoable), restoring its dimensions on undo."""
        sk = _sketch_of(self, sketch_id)
        idx = next((i for i, e in enumerate(sk.entities) if e.id == eid), -1)
        if idx < 0:
            return False
        data = snapshot_entity(sk.entities[idx])
        dims = [snapshot_dimension(d) for d in sk.dimensions_for_entity(eid)]
        sk.remove_entity(eid)
        sk.remove_dimensions_for_entity(eid)
        self.push_command(EntityDeleteCommand(sketch_id, data, idx, dims))
        return True

    def delete_entities(self, sketch_id: int, eids: List[int]) -> int:
        """Delete multiple sketch entities as ONE undo step. Returns count deleted."""
        if not eids:
            return 0
        if len(eids) == 1:
            return 1 if self.delete_entity(sketch_id, int(eids[0])) else 0
        sk = _sketch_of(self, sketch_id)
        wanted = {int(e) for e in eids}
        items: List[Tuple[dict, int]] = []
        for i, e in enumerate(sk.entities):
            if e.id in wanted:
                items.append((snapshot_entity(e), i))
        if not items:
            return 0
        dims = [
            snapshot_dimension(d)
            for d in sk.dimensions
            if int(d.entity_id) in wanted
        ]
        for data, _idx in items:
            sk.remove_entity(int(data["id"]))
        sk.dimensions = [d for d in sk.dimensions if int(d.entity_id) not in wanted]
        self.push_command(EntityMultiDeleteCommand(sketch_id, items, dims))
        return len(items)

    def record_entity_move(self, sketch_id: int, before: dict, after: dict) -> None:
        """Record geometry change if before != after."""
        if before == after:
            return
        self.push_command(EntityMoveCommand(sketch_id, before, after))

    def record_feature_add(self, f: Feature) -> None:
        """Feature already added; push undoable feature-add."""
        idx = next((i for i, x in enumerate(self.features) if x.id == f.id), len(self.features) - 1)
        self.push_command(FeatureAddCommand(f, idx))

    def features_depending_on(self, fid: int) -> List[Feature]:
        """Direct dependents: solids whose operand is ``fid``, or sketches on that solid/plane."""
        fid = int(fid)
        out: List[Feature] = []
        for f in self.features:
            if f.id == fid:
                continue
            if int(f.operand_a) == fid or int(f.operand_b) == fid:
                out.append(f)
                continue
            # Sketch-on-face / sketch-on-plane parent link
            if f.type is FeatureType.SKETCH and int(f.plane_id) == fid:
                out.append(f)
        return out

    def collect_delete_set(self, fid: int) -> List[Feature]:
        """Feature plus all recursive dependents (children first for safe order)."""
        root = self.find(fid)
        if root is None:
            return []
        # BFS to gather the full subtree of dependents
        seen: Dict[int, Feature] = {root.id: root}
        queue = [root.id]
        while queue:
            cur = queue.pop(0)
            for dep in self.features_depending_on(cur):
                if dep.id not in seen:
                    seen[dep.id] = dep
                    queue.append(dep.id)
        # Order: dependents before their parents (deepest first) for redo delete
        # Simple approach: sort by reverse feature list position (later features first)
        order_idx = {f.id: i for i, f in enumerate(self.features)}
        return sorted(seen.values(), key=lambda f: -order_idx.get(f.id, 0))

    def delete_feature_undoable(self, fid: int) -> bool:
        """Delete non-plane feature and cascade dependents (one undo step)."""
        f = self.find(fid)
        if f is None or is_reference_plane(f.type):
            return False
        to_delete = self.collect_delete_set(fid)
        if not to_delete:
            return False
        # Capture original indices before mutation
        idx_map = {x.id: i for i, x in enumerate(self.features)}
        items: List[Tuple[Feature, int]] = [
            (feat, idx_map[feat.id]) for feat in to_delete if feat.id in idx_map
        ]
        # Store in ascending index order for stable multi-delete undo
        items.sort(key=lambda t: t[1])
        drop_ids = {feat.id for feat, _ in items}
        self.features = [x for x in self.features if x.id not in drop_ids]
        if self.selected_id in drop_ids:
            self.selected_id = self.features[0].id if self.features else -1
        if len(items) == 1:
            self.push_command(FeatureDeleteCommand(items[0][0], items[0][1]))
        else:
            self.push_command(FeatureMultiDeleteCommand(items))
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
        u = Unit(unit)
        if self.display_unit is not u and self.display_unit != u:
            self.dirty = True
        self.display_unit = u

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

    def create_sketch_on_face(
        self,
        solid_id: int,
        frame: PlaneFrame,
    ) -> Optional[Feature]:
        """Create a sketch whose UV plane sits on a solid face (``frame``).

        ``plane_id`` / ``plane_feature_id`` record the parent solid so the tree
        shows the relationship; geometry lives entirely in ``frame``.
        """
        solid = self.find(solid_id)
        if solid is None or not is_solid_feature(solid.type):
            return None
        # Normalize frame axes
        o = np.asarray(frame.origin, dtype=np.float64).reshape(3).copy()
        u = np.asarray(frame.u_axis, dtype=np.float64).reshape(3).copy()
        v = np.asarray(frame.v_axis, dtype=np.float64).reshape(3).copy()
        n = np.asarray(frame.normal, dtype=np.float64).reshape(3).copy()
        for vec, name in ((u, "u"), (v, "v"), (n, "n")):
            ln = float(np.linalg.norm(vec))
            if ln < 1e-12:
                raise ValueError(f"degenerate face frame ({name}-axis)")
            vec /= ln
        # Re-orthogonalize
        n = n / float(np.linalg.norm(n))
        u = u - n * float(np.dot(u, n))
        lu = float(np.linalg.norm(u))
        if lu < 1e-12:
            raise ValueError("degenerate face frame (u in normal)")
        u = u / lu
        v = np.cross(n, u)
        v = v / float(np.linalg.norm(v))
        fr = PlaneFrame(o, u, v, n)
        self._sketch_count += 1
        sketch = Sketch(
            name=f"Sketch{self._sketch_count}",
            plane_feature_id=solid_id,
            frame=fr,
        )
        f = Feature(
            type=FeatureType.SKETCH,
            name=sketch.name,
            plane_id=solid_id,
            sketch=sketch,
        )
        self.add_feature(f)
        self.record_feature_add(f)
        return f

    def apply_sketch_dimension(
        self,
        sketch_id: int,
        entity_id: int,
        role: str,
        value_mm: float,
    ) -> Optional[SketchDimension]:
        """Create/update a driving dimension and apply it to geometry (undoable)."""
        from cadcore.sketch import apply_dimension_value, measure_dimension_value

        sk = _sketch_of(self, sketch_id)
        ent = _entity_of(self, sketch_id, entity_id)
        before = snapshot_entity(ent)
        # Capture existing dimension for this (entity, role) if any
        dim_before: Optional[dict] = None
        for d in sk.dimensions_for_entity(entity_id):
            if str(d.role) == str(role):
                dim_before = snapshot_dimension(d)
                break
        apply_dimension_value(ent, role, value_mm)
        after = snapshot_entity(ent)
        # Store dimension at applied value (geometry is source of truth after apply)
        try:
            measured = measure_dimension_value(ent, role)
        except ValueError:
            measured = float(value_mm)
        dim = sk.add_or_update_dimension(entity_id, role, measured)
        dim_after = snapshot_dimension(dim)
        self.push_command(
            DimensionApplyCommand(sketch_id, before, after, dim_before, dim_after)
        )
        return dim

    def create_extrude(
        self,
        sketch_id: int,
        distance: float,
        *,
        profile_entity_id: int = -1,
        segments: int = 64,
        reversed: bool = False,
        merge_solid_id: Optional[int] = None,
    ) -> Feature:
        """Pad a closed sketch profile by ``distance`` along ±plane normal.

        ``distance`` is always a positive magnitude. ``reversed=True`` pads along
        the opposite of the sketch plane normal (SolidWorks Reverse Direction).

        When the sketch lives on a solid face (or ``merge_solid_id`` is given),
        the boss is **boolean-unioned** into that solid so the result is one
        continuous body (SolidWorks Boss-Extrude / merge result). Overlapping
        volume is counted once.
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
        rev = bool(reversed)
        # Parent solid for merge: explicit arg, else sketch-on-face plane_id
        merge_id = -1
        if merge_solid_id is not None:
            merge_id = int(merge_solid_id)
        elif int(skf.plane_id) >= 0:
            parent = self.find(int(skf.plane_id))
            if parent is not None and is_solid_feature(parent.type):
                merge_id = int(parent.id)
        if merge_id >= 0:
            parent = self.find(merge_id)
            if parent is None or not is_solid_feature(parent.type):
                raise ValueError("extrude merge target is not a valid solid")
            if merge_id == sketch_id:
                raise ValueError("extrude cannot merge into its own sketch")

        # Validate tool (and merge) before committing history
        tool = extrude_profile(
            ent,
            dist,
            sketch.frame,
            segments=int(segments),
            holes=resolved.holes,
            reversed=rev,
        )
        if merge_id >= 0:
            body = self.evaluate_feature(merge_id)
            if body is None or body.empty:
                raise ValueError("cannot merge extrude: parent solid has no mesh")
            try:
                merged = boolean_op(body, tool, BooleanOp.UNION)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"could not merge boss into solid: {exc}") from exc
            if merged.empty:
                raise ValueError("merged extrude is empty")
            if not merged.is_watertight():
                raise ValueError("merged extrude is not watertight")

        self._extrude_count += 1
        f = Feature(
            type=FeatureType.EXTRUDE,
            name=f"Extrude{self._extrude_count}",
            depth=dist,
            segments=int(segments),
            operand_a=sketch_id,
            operand_b=int(merge_id),
            profile_entity_id=int(profile_entity_id),
            reversed=rev,
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
        # profile_entity_id may be a ClosedLineLoop synthetic id (negative)
        resolved = resolve_profiles(sketch, preferred_outer_id=profile_entity_id)
        ent = resolved.outer
        profile_entity_id = _profile_id(ent)
        if not is_closed_profile(ent):
            raise ValueError("profile is not a closed rectangle/circle/line-loop")
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

        SolidWorks-style: sharp corner vertices are **removed from the sketch**
        (replaced by arc polylines). The solid uses the filleted outline.
        """
        from cadcore.fillet2d import fillet_closed_polygon, profile_to_polygon_uv
        from cadcore.profiles import ClosedLineLoop

        skf = self.find(sketch_id)
        if skf is None or skf.type is not FeatureType.SKETCH or skf.sketch is None:
            raise ValueError("fillet requires a valid sketch feature")
        sketch = skf.sketch
        resolved = resolve_profiles(sketch, preferred_outer_id=profile_entity_id)
        ent = resolved.outer
        profile_entity_id = _profile_id(ent)
        if not is_closed_profile(ent):
            raise ValueError("open profile: not a closed rectangle/circle/line-loop")
        # Circles are already smooth — corner fillet is a no-op on the sketch
        if getattr(ent, "kind", None) is not None:
            from cadcore.sketch import EntityKind

            if getattr(ent, "kind", None) is EntityKind.CIRCLE:
                raise ValueError("circle profile has no corners to fillet")
        dist = float(distance)
        if not np.isfinite(dist) or dist <= 1e-12:
            raise ValueError("extrude distance must be a positive finite number")
        rad = float(radius)
        if not np.isfinite(rad) or rad <= 1e-12:
            raise ValueError("fillet radius must be positive (radius <= 0 is invalid)")

        sharp = profile_to_polygon_uv(ent)
        # Validate solid build from sharp profile (dual-offset)
        _ = extrude_filleted_profile(
            sharp,
            dist,
            sketch.frame,
            rad,
            segments=int(segments),
        )
        # Snapshot sketch so undo restores sharp geometry
        sketch_before = snapshot_sketch_contents(sketch)
        # Sketch: delete sharp corners — replace profile with filleted polyline
        filleted_uv = fillet_closed_polygon(
            sharp, rad, arc_segments=max(6, int(segments) // 4)
        )
        _replace_sketch_profile_with_polyline(sketch, ent, filleted_uv)
        sketch_after = snapshot_sketch_contents(sketch)
        self._fillet_count += 1
        f = Feature(
            type=FeatureType.FILLET,
            name=f"Fillet{self._fillet_count}",
            depth=dist,
            radius=rad,
            segments=int(segments),
            operand_a=sketch_id,
            profile_entity_id=-1,  # profile is now the filleted line-loop
            source_profile_uv=[(float(p[0]), float(p[1])) for p in sharp],
        )
        self.add_feature(f)
        idx = next(
            (i for i, x in enumerate(self.features) if x.id == f.id),
            len(self.features) - 1,
        )
        self.push_command(
            FilletCreateCommand(f, sketch_id, sketch_before, sketch_after, idx)
        )
        return f

    def create_edge_fillet(
        self,
        solid_id: int,
        edge_keys: List[str],
        radius: float,
        *,
        segments: int = 32,
    ) -> Feature:
        """Round one or more edges on a solid (SolidWorks-style Fillet).

        Validates the boolean fillet fully before committing the feature so a
        failed radius / edge selection leaves the document unchanged.
        """
        from cadcore.edge_fillet import (
            edges_from_keys,
            extract_convex_edges,
            fillet_edges,
        )

        parent = self.find(solid_id)
        if parent is None or not is_solid_feature(parent.type):
            raise ValueError("fillet requires a valid solid feature")
        keys = [str(k) for k in (edge_keys or []) if str(k).strip()]
        if not keys:
            raise ValueError("select at least one edge on the solid to fillet")
        rad = float(radius)
        if not np.isfinite(rad) or rad <= 1e-12:
            raise ValueError("fillet radius must be positive")
        segs = max(8, int(segments))
        body = self.evaluate_feature(solid_id)
        if body is None or body.empty:
            raise ValueError("the selected solid has no mesh to fillet")
        try:
            all_edges = extract_convex_edges(body.vertices, body.faces)
            edges = edges_from_keys(all_edges, keys)
            # Full validate before history commit
            _ = fillet_edges(body, edges, rad, segments=segs)
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"could not apply fillet: {exc}") from exc

        self._fillet_count += 1
        f = Feature(
            type=FeatureType.EDGE_FILLET,
            name=f"Fillet{self._fillet_count}",
            radius=rad,
            segments=segs,
            operand_a=int(solid_id),
            edge_keys=list(keys),
        )
        self.add_feature(f)
        self.record_feature_add(f)
        return f

    def update_feature_params(self, fid: int, **params) -> bool:
        """Apply editable parameters on a solid feature (undoable) and re-sync sketch.

        Supported keys: depth, radius, segments, revolve_angle, hole_center_u,
        hole_center_v, reversed, name. Returns False if feature missing / no change.
        """
        f = self.find(fid)
        if f is None or is_reference_plane(f.type) or f.type is FeatureType.SKETCH:
            return False
        keys = (
            "depth",
            "radius",
            "segments",
            "revolve_angle",
            "hole_center_u",
            "hole_center_v",
            "reversed",
            "through_all",
            "name",
        )
        before = {k: getattr(f, k) for k in keys if hasattr(f, k)}
        before["source_profile_uv"] = list(f.source_profile_uv)
        changed = False
        for k, v in params.items():
            if k not in keys or not hasattr(f, k):
                continue
            if getattr(f, k) != v:
                setattr(f, k, v)
                changed = True
        if not changed:
            return False
        # Fillet radius change: rebuild sketch polyline from stored sharp profile
        if f.type is FeatureType.FILLET and "radius" in params and f.source_profile_uv:
            from cadcore.fillet2d import fillet_closed_polygon

            skf = self.find(f.operand_a)
            if skf is not None and skf.sketch is not None:
                try:
                    poly = fillet_closed_polygon(
                        f.source_profile_uv,
                        float(f.radius),
                        arc_segments=max(6, int(f.segments) // 4),
                    )
                    _clear_sketch_entities(skf.sketch)
                    _add_polyline_lines(skf.sketch, poly)
                except ValueError:
                    # Revert
                    for k, v in before.items():
                        if k != "source_profile_uv":
                            setattr(f, k, v)
                    f.source_profile_uv = list(before["source_profile_uv"])
                    raise
        # Edge fillet: re-validate radius/edges before committing the change
        if f.type is FeatureType.EDGE_FILLET and (
            "radius" in params or "segments" in params
        ):
            mesh = self.evaluate_feature(fid)
            if mesh is None or mesh.empty:
                for k, v in before.items():
                    if k != "source_profile_uv":
                        setattr(f, k, v)
                f.source_profile_uv = list(before["source_profile_uv"])
                raise ValueError(
                    "fillet radius cannot be applied to the selected edges "
                    "(geometry invalid for this radius)"
                )
        after = {k: getattr(f, k) for k in keys if hasattr(f, k)}
        after["source_profile_uv"] = list(f.source_profile_uv)
        self.push_command(FeatureParamCommand(fid, before, after))
        return True

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
        resolved = resolve_profiles(sketch, preferred_outer_id=profile_entity_id)
        ent = resolved.outer
        profile_entity_id = _profile_id(ent)
        if not is_closed_profile(ent):
            raise ValueError("open profile: not a closed rectangle/circle/line-loop")
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

    def absorbed_sketch_map(self) -> Dict[int, int]:
        """Map sketch_feature_id → consuming solid feature id (SolidWorks absorbed)."""
        out: Dict[int, int] = {}
        for f in self.features:
            if is_sketch_consuming_feature(f.type) and int(f.operand_a) >= 0:
                # First consumer wins (a sketch used by one feature in our model)
                if int(f.operand_a) not in out:
                    out[int(f.operand_a)] = int(f.id)
        return out

    def create_cut_extrude(
        self,
        sketch_id: int,
        target_solid_id: int,
        distance: float,
        *,
        profile_entity_id: int = -1,
        segments: int = 64,
        reversed: bool = False,
        through_all: bool = False,
    ) -> Feature:
        """SolidWorks Extruded Cut: subtract a sketched profile from a solid.

        ``operand_a`` = sketch, ``operand_b`` = solid being cut.
        ``through_all`` extends the tool through the target's bounding box.
        """
        skf = self.find(sketch_id)
        if skf is None or skf.type is not FeatureType.SKETCH or skf.sketch is None:
            raise ValueError("cut requires a valid sketch feature")
        target = self.find(target_solid_id)
        if target is None or not is_solid_feature(target.type):
            raise ValueError("cut requires a valid solid feature to cut into")
        if target.type is FeatureType.CUT_EXTRUDE and target.id == target_solid_id:
            pass  # can cut a previous cut result
        sketch = skf.sketch
        resolved = resolve_profiles(sketch, preferred_outer_id=profile_entity_id)
        ent = resolved.outer
        profile_entity_id = _profile_id(ent)
        if not is_closed_profile(ent):
            raise ValueError("cut profile is not a closed region")
        dist = float(distance)
        if not through_all and (not np.isfinite(dist) or dist <= 1e-12):
            raise ValueError("cut depth must be a positive finite number")
        # Validate by evaluating once
        body = self.evaluate_feature(target_solid_id)
        if body is None or body.empty:
            raise ValueError("target solid has no geometry to cut")
        tool_dist = dist
        if through_all:
            # Tool length: solid diagonal so it fully pierces from the sketch plane
            lo = body.vertices.min(axis=0)
            hi = body.vertices.max(axis=0)
            tool_dist = float(np.linalg.norm(hi - lo)) * 2.0 + 1.0
            tool_dist = max(tool_dist, 1.0)
        rev = bool(reversed)
        tool = extrude_profile(
            ent,
            tool_dist,
            sketch.frame,
            segments=int(segments),
            holes=resolved.holes,
            reversed=rev,
        )
        if through_all:
            tool_rev = extrude_profile(
                ent,
                tool_dist,
                sketch.frame,
                segments=int(segments),
                holes=resolved.holes,
                reversed=not rev,
            )
            tool = boolean_op(tool, tool_rev, BooleanOp.UNION)
        result = boolean_op(body, tool, BooleanOp.DIFFERENCE)
        # Face sketches use outward normals — first try often misses the solid.
        # Auto-flip the tool into the body when the cut removes no material.
        if (
            not through_all
            and result is not None
            and not result.empty
            and abs(result.volume() - body.volume()) < 1e-6 * max(abs(body.volume()), 1.0)
        ):
            rev = not rev
            tool = extrude_profile(
                ent,
                tool_dist,
                sketch.frame,
                segments=int(segments),
                holes=resolved.holes,
                reversed=rev,
            )
            result = boolean_op(body, tool, BooleanOp.DIFFERENCE)
        if result is None or result.empty:
            raise ValueError("cut removed the entire solid (empty result)")
        if abs(result.volume() - body.volume()) < 1e-6 * max(abs(body.volume()), 1.0):
            raise ValueError(
                "cut did not remove any material (check profile is on the solid "
                "and direction points into it)"
            )
        if not result.is_watertight():
            raise ValueError("cut result is not watertight")
        self._cut_count += 1
        f = Feature(
            type=FeatureType.CUT_EXTRUDE,
            name=f"Cut-Extrude{self._cut_count}",
            depth=float(dist) if not through_all else float(tool_dist),
            segments=int(segments),
            operand_a=sketch_id,
            operand_b=target_solid_id,
            profile_entity_id=int(profile_entity_id),
            reversed=rev,
            through_all=bool(through_all),
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
                reversed=bool(getattr(f, "reversed", False)),
            )
            # Sketch-on-face boss: union into parent solid (one continuous body)
            if int(getattr(f, "operand_b", -1)) >= 0:
                body = self.evaluate_feature(int(f.operand_b))
                if body is None or body.empty:
                    return None
                try:
                    mesh = boolean_op(body, mesh, BooleanOp.UNION)
                except Exception:
                    return None
                if mesh is None or mesh.empty:
                    return None
        elif f.type is FeatureType.FILLET:
            skf = self.find(f.operand_a)
            if skf is None or skf.sketch is None:
                return None
            sketch = skf.sketch
            # Prefer parametric sharp source (sketch already shows filleted edges)
            if f.source_profile_uv and len(f.source_profile_uv) >= 3:
                mesh = extrude_filleted_profile(
                    f.source_profile_uv,
                    f.depth,
                    sketch.frame,
                    f.radius,
                    segments=max(3, int(f.segments)),
                )
            else:
                try:
                    resolved = resolve_profiles(
                        sketch, preferred_outer_id=f.profile_entity_id
                    )
                except ValueError:
                    return None
                mesh = extrude_filleted_profile(
                    resolved.outer,
                    f.depth,
                    sketch.frame,
                    f.radius,
                    segments=max(3, int(f.segments)),
                )
        elif f.type is FeatureType.EDGE_FILLET:
            from cadcore.edge_fillet import (
                edges_from_keys,
                extract_convex_edges,
                fillet_edges,
            )

            body = self.evaluate_feature(f.operand_a)
            if body is None or body.empty:
                return None
            try:
                all_edges = extract_convex_edges(body.vertices, body.faces)
                edges = edges_from_keys(all_edges, f.edge_keys or [])
                mesh = fillet_edges(
                    body,
                    edges,
                    float(f.radius),
                    segments=max(8, int(f.segments)),
                )
            except Exception:
                return None
        elif f.type is FeatureType.POCKET:
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
            mesh = extrude_pocketed_profile(
                resolved.outer,
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
            try:
                resolved = resolve_profiles(
                    sketch, preferred_outer_id=f.profile_entity_id
                )
            except ValueError:
                return None
            mesh = revolve_profile(
                resolved.outer,
                sketch.frame,
                axis_origin=f.axis_origin,
                axis_direction=f.axis_direction,
                angle_degrees=f.revolve_angle,
                segments=max(3, int(f.segments)),
            )
        elif f.type is FeatureType.CUT_EXTRUDE:
            skf = self.find(f.operand_a)
            body = self.evaluate_feature(f.operand_b)
            if skf is None or skf.sketch is None or body is None or body.empty:
                return None
            sketch = skf.sketch
            try:
                resolved = resolve_profiles(
                    sketch, preferred_outer_id=f.profile_entity_id
                )
            except ValueError:
                return None
            tool_dist = float(f.depth)
            if bool(getattr(f, "through_all", False)):
                lo = body.vertices.min(axis=0)
                hi = body.vertices.max(axis=0)
                tool_dist = float(np.linalg.norm(hi - lo)) * 2.0 + 1.0
                tool_dist = max(tool_dist, 1.0)
            rev = bool(getattr(f, "reversed", False))
            tool = extrude_profile(
                resolved.outer,
                tool_dist,
                sketch.frame,
                segments=max(3, int(f.segments)),
                holes=resolved.holes,
                reversed=rev,
            )
            if bool(getattr(f, "through_all", False)):
                tool_rev = extrude_profile(
                    resolved.outer,
                    tool_dist,
                    sketch.frame,
                    segments=max(3, int(f.segments)),
                    holes=resolved.holes,
                    reversed=not rev,
                )
                tool = boolean_op(tool, tool_rev, BooleanOp.UNION)
            mesh = boolean_op(body, tool, BooleanOp.DIFFERENCE)
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
            # Cut consumes its target solid (like boolean difference operand)
            if f.type is FeatureType.CUT_EXTRUDE and f.operand_b >= 0:
                used.add(f.operand_b)
            # Boss-extrude merges into parent solid — show only the result
            if f.type is FeatureType.EXTRUDE and f.operand_b >= 0:
                used.add(f.operand_b)
            # Edge fillet replaces its parent solid in the display
            if f.type is FeatureType.EDGE_FILLET and f.operand_a >= 0:
                used.add(f.operand_a)
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
