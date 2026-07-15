"""Background geometry evaluation — never call VTK from here."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from cadcore.document import (
    Document,
    Feature,
    FeatureType,
    copy_sketch,
    first_closed_profile,
    is_boolean,
    is_reference_plane,
    resolve_profiles,
    sketch_fingerprint,
)
from cadcore.mesh import Mesh


def feature_fingerprint(f: Feature) -> str:
    """Stable key: if this changes, the solid mesh must be recomputed/re-uploaded."""
    parts = [
        str(f.id),
        f.type.name,
        f"{f.width:.6g}",
        f"{f.height:.6g}",
        f"{f.depth:.6g}",
        f"{f.radius:.6g}",
        str(f.segments),
        str(f.rings),
        str(f.operand_a),
        str(f.operand_b),
        str(f.profile_entity_id),
        f"{f.revolve_angle:.6g}",
        f"{f.axis_origin[0]:.6g},{f.axis_origin[1]:.6g}",
        f"{f.axis_direction[0]:.6g},{f.axis_direction[1]:.6g}",
        f"{f.hole_center_u:.6g},{f.hole_center_v:.6g}",
        f"{f.translation[0]:.6g},{f.translation[1]:.6g},{f.translation[2]:.6g}",
        str(int(getattr(f, "reversed", False))),
        str(int(f.visible)),
        str(int(f.suppressed)),
    ]
    if f.type in (
        FeatureType.EXTRUDE,
        FeatureType.REVOLVE,
        FeatureType.FILLET,
        FeatureType.POCKET,
        FeatureType.SKETCH,
    ):
        parts.append(sketch_fingerprint(f.sketch))
    return "|".join(parts)


def snapshot_features(doc: Document) -> List[Feature]:
    """Deep-ish copy of features safe to read on a worker thread."""
    out: List[Feature] = []
    for f in doc.features:
        out.append(
            Feature(
                id=f.id,
                name=f.name,
                type=f.type,
                width=f.width,
                height=f.height,
                depth=f.depth,
                radius=f.radius,
                segments=f.segments,
                rings=f.rings,
                operand_a=f.operand_a,
                operand_b=f.operand_b,
                translation=tuple(f.translation),  # type: ignore[arg-type]
                plane_id=f.plane_id,
                sketch=copy_sketch(f.sketch),
                profile_entity_id=f.profile_entity_id,
                revolve_angle=f.revolve_angle,
                axis_origin=tuple(f.axis_origin),  # type: ignore[arg-type]
                axis_direction=tuple(f.axis_direction),  # type: ignore[arg-type]
                hole_center_u=f.hole_center_u,
                hole_center_v=f.hole_center_v,
                visible=f.visible,
                suppressed=f.suppressed,
            )
        )
    return out


def evaluate_solids_snapshot(
    features: List[Feature],
) -> Dict[int, Tuple[np.ndarray, np.ndarray, str]]:
    """CPU-heavy: evaluate solid meshes for a frozen feature list.

    Returns feature_id -> (vertices Nx3, faces Mx3, fingerprint).
    """
    # Local evaluate mirroring Document.evaluate_feature without sharing Document
    by_id = {f.id: f for f in features}
    cache: Dict[int, Optional[Mesh]] = {}

    def eval_one(fid: int) -> Optional[Mesh]:
        if fid in cache:
            return cache[fid]
        f = by_id.get(fid)
        if f is None or f.suppressed or is_reference_plane(f.type) or f.type is FeatureType.SKETCH:
            cache[fid] = None
            return None
        from cadcore.mesh import (
            BooleanOp,
            boolean_op,
            extrude_filleted_profile,
            extrude_pocketed_profile,
            extrude_profile,
            make_box,
            make_cylinder,
            make_sphere,
            revolve_profile,
        )

        mesh: Optional[Mesh] = None
        if f.type is FeatureType.EXTRUDE:
            skf = by_id.get(f.operand_a)
            if skf is None or skf.sketch is None:
                cache[fid] = None
                return None
            sketch = skf.sketch
            try:
                resolved = resolve_profiles(
                    sketch, preferred_outer_id=f.profile_entity_id
                )
            except ValueError:
                cache[fid] = None
                return None
            mesh = extrude_profile(
                resolved.outer,
                f.depth,
                sketch.frame,
                segments=max(3, int(f.segments)),
                holes=resolved.holes,
                reversed=bool(getattr(f, "reversed", False)),
            )
        elif f.type is FeatureType.FILLET:
            skf = by_id.get(f.operand_a)
            if skf is None or skf.sketch is None:
                cache[fid] = None
                return None
            sketch = skf.sketch
            # Parametric sharp source (sketch already shows filleted polyline)
            if getattr(f, "source_profile_uv", None) and len(f.source_profile_uv) >= 3:
                mesh = extrude_filleted_profile(
                    f.source_profile_uv,
                    f.depth,
                    sketch.frame,
                    f.radius,
                    segments=max(3, int(f.segments)),
                )
            else:
                if f.profile_entity_id >= 0:
                    ent = sketch.find_entity(f.profile_entity_id)
                else:
                    ent = first_closed_profile(sketch)
                if ent is None:
                    cache[fid] = None
                    return None
                mesh = extrude_filleted_profile(
                    ent,
                    f.depth,
                    sketch.frame,
                    f.radius,
                    segments=max(3, int(f.segments)),
                )
        elif f.type is FeatureType.POCKET:
            skf = by_id.get(f.operand_a)
            if skf is None or skf.sketch is None:
                cache[fid] = None
                return None
            sketch = skf.sketch
            if f.profile_entity_id >= 0:
                ent = sketch.find_entity(f.profile_entity_id)
            else:
                ent = first_closed_profile(sketch)
            if ent is None:
                cache[fid] = None
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
            skf = by_id.get(f.operand_a)
            if skf is None or skf.sketch is None:
                cache[fid] = None
                return None
            sketch = skf.sketch
            if f.profile_entity_id >= 0:
                ent = sketch.find_entity(f.profile_entity_id)
            else:
                ent = first_closed_profile(sketch)
            if ent is None:
                cache[fid] = None
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
            ma = eval_one(f.operand_a)
            mb = eval_one(f.operand_b)
            if ma is None or mb is None:
                cache[fid] = None
                return None
            op = {
                FeatureType.BOOLEAN_UNION: BooleanOp.UNION,
                FeatureType.BOOLEAN_DIFFERENCE: BooleanOp.DIFFERENCE,
                FeatureType.BOOLEAN_INTERSECTION: BooleanOp.INTERSECTION,
            }[f.type]
            mesh = boolean_op(ma, mb, op)
        if mesh is not None and f.translation != (0.0, 0.0, 0.0):
            mesh = mesh.translate(f.translation)
        cache[fid] = mesh
        return mesh

    used = set()
    for f in features:
        if is_boolean(f.type):
            if f.operand_a >= 0:
                used.add(f.operand_a)
            if f.operand_b >= 0:
                used.add(f.operand_b)

    results: Dict[int, Tuple[np.ndarray, np.ndarray, str]] = {}
    for f in features:
        if not f.visible or f.suppressed or is_reference_plane(f.type) or f.type is FeatureType.SKETCH:
            continue
        if f.id in used:
            continue
        m = eval_one(f.id)
        if m is None or m.empty:
            continue
        # For sketch-based solids, include source sketch geometry in the fingerprint
        fp_feature = f
        if f.type in (
            FeatureType.EXTRUDE,
            FeatureType.REVOLVE,
            FeatureType.FILLET,
            FeatureType.POCKET,
        ):
            skf = by_id.get(f.operand_a)
            if skf is not None and skf.sketch is not None:
                # temporary view with sketch attached for fingerprint only
                fp_feature = Feature(
                    id=f.id,
                    name=f.name,
                    type=f.type,
                    width=f.width,
                    height=f.height,
                    depth=f.depth,
                    radius=f.radius,
                    segments=f.segments,
                    rings=f.rings,
                    operand_a=f.operand_a,
                    operand_b=f.operand_b,
                    translation=f.translation,
                    plane_id=f.plane_id,
                    sketch=skf.sketch,
                    profile_entity_id=f.profile_entity_id,
                    revolve_angle=f.revolve_angle,
                    axis_origin=f.axis_origin,
                    axis_direction=f.axis_direction,
                    hole_center_u=f.hole_center_u,
                    hole_center_v=f.hole_center_v,
                    visible=f.visible,
                    suppressed=f.suppressed,
                )
        results[f.id] = (
            np.ascontiguousarray(m.vertices, dtype=np.float64),
            np.ascontiguousarray(m.faces, dtype=np.int32),
            feature_fingerprint(fp_feature),
        )
    return results


class WorkerSignals(QObject):
    finished = Signal(int, object)  # generation, results dict
    failed = Signal(int, str)
    started = Signal(int)


class GeometryRebuildJob(QRunnable):
    def __init__(self, generation: int, features: List[Feature]) -> None:
        super().__init__()
        self.generation = generation
        self.features = features
        self.signals = WorkerSignals()
        self.setAutoDelete(True)

    @Slot()
    def run(self) -> None:
        try:
            self.signals.started.emit(self.generation)
            result = evaluate_solids_snapshot(self.features)
            self.signals.finished.emit(self.generation, result)
        except Exception as exc:  # noqa: BLE001
            self.signals.failed.emit(self.generation, str(exc))
