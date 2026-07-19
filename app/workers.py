"""Background geometry evaluation — never call VTK from here."""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Any, Dict, List, Optional, Tuple

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
from cadcore.sketch import Sketch


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
        str(int(bool(f.reversed))),
        str(int(bool(getattr(f, "through_all", False)))),
        str(int(f.visible)),
        str(int(f.suppressed)),
    ]
    # Fillet parametric source outline
    if f.source_profile_uv:
        parts.append(
            ";".join(f"{p[0]:.6g},{p[1]:.6g}" for p in f.source_profile_uv)
        )
    # Solid edge fillet keys
    if getattr(f, "edge_keys", None):
        parts.append("ek:" + ";".join(str(k) for k in f.edge_keys))
    if f.type in (
        FeatureType.EXTRUDE,
        FeatureType.REVOLVE,
        FeatureType.FILLET,
        FeatureType.POCKET,
        FeatureType.CUT_EXTRUDE,
        FeatureType.SKETCH,
    ):
        parts.append(sketch_fingerprint(f.sketch))
    return "|".join(parts)


def feature_fingerprint_with_deps(f: Feature, by_id: Dict[int, Feature]) -> str:
    """Fingerprint including parent solid fingerprints (for EDGE_FILLET / CUT)."""
    base = feature_fingerprint(f)
    deps: list[str] = []
    if f.type is FeatureType.EDGE_FILLET and int(f.operand_a) >= 0:
        parent = by_id.get(int(f.operand_a))
        if parent is not None:
            deps.append("pa:" + feature_fingerprint(parent))
    if f.type is FeatureType.CUT_EXTRUDE and int(f.operand_b) >= 0:
        parent = by_id.get(int(f.operand_b))
        if parent is not None:
            deps.append("pb:" + feature_fingerprint(parent))
    # Boss-extrude merges into parent solid
    if f.type is FeatureType.EXTRUDE and int(f.operand_b) >= 0:
        parent = by_id.get(int(f.operand_b))
        if parent is not None:
            deps.append("pb:" + feature_fingerprint(parent))
    if not deps:
        return base
    return base + "|" + "|".join(deps)


def _clone_feature_value(name: str, val: Any) -> Any:
    """Clone one Feature field for a worker-safe snapshot."""
    if name == "sketch":
        return copy_sketch(val)
    if name == "source_profile_uv":
        return [(float(p[0]), float(p[1])) for p in (val or [])]
    if name == "edge_keys":
        return [str(k) for k in (val or [])]
    if isinstance(val, tuple):
        return tuple(val)
    if isinstance(val, list):
        return list(val)
    if isinstance(val, np.ndarray):
        return val.copy()
    return val


def snapshot_feature(f: Feature) -> Feature:
    """Structural copy of one Feature from dataclasses.fields — no field can be forgotten."""
    kwargs = {
        fld.name: _clone_feature_value(fld.name, getattr(f, fld.name))
        for fld in fields(Feature)
    }
    return Feature(**kwargs)


def snapshot_features(doc: Document) -> List[Feature]:
    """Deep-ish copy of features safe to read on a worker thread.

    Derived from ``dataclasses.fields(Feature)`` so new Feature fields are
    always included (the previous hand-written list silently dropped
    ``reversed`` and ``source_profile_uv``).
    """
    return [snapshot_feature(f) for f in doc.features]


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
            if int(getattr(f, "operand_b", -1)) >= 0:
                body = eval_one(int(f.operand_b))
                if body is None or body.empty:
                    cache[fid] = None
                    return None
                try:
                    mesh = boolean_op(body, mesh, BooleanOp.UNION)
                except Exception:
                    cache[fid] = None
                    return None
                if mesh is None or mesh.empty:
                    cache[fid] = None
                    return None
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
        elif f.type is FeatureType.EDGE_FILLET:
            from cadcore.edge_fillet import (
                edges_from_keys,
                extract_convex_edges,
                fillet_edges,
            )

            body = eval_one(f.operand_a)
            if body is None or body.empty:
                cache[fid] = None
                return None
            try:
                all_edges = extract_convex_edges(body.vertices, body.faces)
                edges = edges_from_keys(all_edges, getattr(f, "edge_keys", None) or [])
                mesh = fillet_edges(
                    body,
                    edges,
                    float(f.radius),
                    segments=max(8, int(f.segments)),
                )
            except Exception:
                cache[fid] = None
                return None
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
        elif f.type is FeatureType.CUT_EXTRUDE:
            skf = by_id.get(f.operand_a)
            body = eval_one(f.operand_b)
            if skf is None or skf.sketch is None or body is None or body.empty:
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
        if f.type is FeatureType.CUT_EXTRUDE and f.operand_b >= 0:
            used.add(f.operand_b)
        if f.type is FeatureType.EXTRUDE and f.operand_b >= 0:
            used.add(f.operand_b)
        if f.type is FeatureType.EDGE_FILLET and f.operand_a >= 0:
            used.add(f.operand_a)

    results: Dict[int, Tuple[np.ndarray, np.ndarray, str]] = {}
    for f in features:
        if not f.visible or f.suppressed or is_reference_plane(f.type) or f.type is FeatureType.SKETCH:
            continue
        if f.id in used:
            continue
        m = eval_one(f.id)
        if m is None or m.empty:
            continue
        # Attach source sketch for fingerprint when the solid feature has none
        fp_feature = f
        if f.type in (
            FeatureType.EXTRUDE,
            FeatureType.REVOLVE,
            FeatureType.FILLET,
            FeatureType.POCKET,
            FeatureType.CUT_EXTRUDE,
        ):
            skf = by_id.get(f.operand_a)
            if skf is not None and skf.sketch is not None and f.sketch is None:
                fp_feature = replace(f, sketch=skf.sketch)
        results[f.id] = (
            np.ascontiguousarray(m.vertices, dtype=np.float64),
            np.ascontiguousarray(m.faces, dtype=np.int32),
            feature_fingerprint_with_deps(fp_feature, by_id),
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
