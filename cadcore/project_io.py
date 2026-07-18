"""Project file I/O: serialize / deserialize a Document to JSON (.gcad).

Saves the full parametric model (features, sketches, dimensions, counters).
Does **not** persist the undo/redo stacks or clipboard — after Open, history
starts empty (standard CAD behaviour). New edits after load are fully
undoable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from cadcore.document import Document, Feature, FeatureType
from cadcore.sketch import (
    DimKind,
    PlaneFrame,
    Sketch,
    restore_dimension,
    restore_entity,
    snapshot_dimension,
    snapshot_entity,
)
from cadcore.units import Unit

PROJECT_FORMAT = "grok-cad-project"
PROJECT_VERSION = 1
DEFAULT_EXTENSION = ".gcad"


class ProjectIOError(ValueError):
    """Invalid or unsupported project file."""


def _vec3(a) -> List[float]:
    v = np.asarray(a, dtype=np.float64).reshape(3)
    return [float(v[0]), float(v[1]), float(v[2])]


def _vec2(a) -> List[float]:
    return [float(a[0]), float(a[1])]


def serialize_plane_frame(frame: PlaneFrame) -> dict:
    return {
        "origin": _vec3(frame.origin),
        "u_axis": _vec3(frame.u_axis),
        "v_axis": _vec3(frame.v_axis),
        "normal": _vec3(frame.normal),
    }


def deserialize_plane_frame(data: dict) -> PlaneFrame:
    return PlaneFrame(
        origin=np.asarray(data["origin"], dtype=np.float64).reshape(3).copy(),
        u_axis=np.asarray(data["u_axis"], dtype=np.float64).reshape(3).copy(),
        v_axis=np.asarray(data["v_axis"], dtype=np.float64).reshape(3).copy(),
        normal=np.asarray(data["normal"], dtype=np.float64).reshape(3).copy(),
    )


def serialize_sketch(sk: Sketch) -> dict:
    return {
        "name": sk.name,
        "plane_feature_id": int(sk.plane_feature_id),
        "frame": serialize_plane_frame(sk.frame),
        "entities": [snapshot_entity(e) for e in sk.entities],
        "dimensions": [snapshot_dimension(d) for d in sk.dimensions],
        "next_entity_id": int(sk._next_entity_id),
        "next_dim_id": int(sk._next_dim_id),
    }


def deserialize_sketch(data: dict) -> Sketch:
    sk = Sketch(
        name=str(data.get("name", "Sketch")),
        plane_feature_id=int(data.get("plane_feature_id", -1)),
        frame=deserialize_plane_frame(data["frame"]),
        entities=[],
        dimensions=[],
        _next_entity_id=int(data.get("next_entity_id", 1)),
        _next_dim_id=int(data.get("next_dim_id", 1)),
    )
    for ed in data.get("entities") or []:
        # JSON may turn tuples into lists — restore_entity handles sequences
        ed = dict(ed)
        for key in ("p0", "p1", "c0", "c1", "center"):
            if key in ed and ed[key] is not None:
                ed[key] = tuple(float(x) for x in ed[key])
        sk.entities.append(restore_entity(ed))
    for dd in data.get("dimensions") or []:
        sk.dimensions.append(restore_dimension(dd))
    # Ensure counters cover restored ids
    if sk.entities:
        sk._next_entity_id = max(sk._next_entity_id, max(e.id for e in sk.entities) + 1)
    if sk.dimensions:
        sk._next_dim_id = max(sk._next_dim_id, max(d.id for d in sk.dimensions) + 1)
    return sk


def serialize_feature(f: Feature) -> dict:
    data: Dict[str, Any] = {
        "id": int(f.id),
        "name": str(f.name),
        "type": f.type.name,
        "width": float(f.width),
        "height": float(f.height),
        "depth": float(f.depth),
        "radius": float(f.radius),
        "segments": int(f.segments),
        "rings": int(f.rings),
        "operand_a": int(f.operand_a),
        "operand_b": int(f.operand_b),
        "translation": [float(x) for x in f.translation],
        "plane_id": int(f.plane_id),
        "profile_entity_id": int(f.profile_entity_id),
        "reversed": bool(f.reversed),
        "revolve_angle": float(f.revolve_angle),
        "axis_origin": _vec2(f.axis_origin),
        "axis_direction": _vec2(f.axis_direction),
        "hole_center_u": float(f.hole_center_u),
        "hole_center_v": float(f.hole_center_v),
        "source_profile_uv": [[float(p[0]), float(p[1])] for p in (f.source_profile_uv or [])],
        "visible": bool(f.visible),
        "suppressed": bool(f.suppressed),
        "sketch": serialize_sketch(f.sketch) if f.sketch is not None else None,
    }
    return data


def deserialize_feature(data: dict) -> Feature:
    try:
        ftype = FeatureType[str(data["type"])]
    except KeyError as exc:
        raise ProjectIOError(f"unknown feature type {data.get('type')!r}") from exc
    tr = data.get("translation") or [0.0, 0.0, 0.0]
    ao = data.get("axis_origin") or [0.0, 0.0]
    ad = data.get("axis_direction") or [0.0, 1.0]
    src = data.get("source_profile_uv") or []
    sketch = None
    if data.get("sketch") is not None:
        sketch = deserialize_sketch(data["sketch"])
    return Feature(
        id=int(data["id"]),
        name=str(data.get("name", "")),
        type=ftype,
        width=float(data.get("width", 1.0)),
        height=float(data.get("height", 1.0)),
        depth=float(data.get("depth", 1.0)),
        radius=float(data.get("radius", 0.5)),
        segments=int(data.get("segments", 16)),
        rings=int(data.get("rings", 10)),
        operand_a=int(data.get("operand_a", -1)),
        operand_b=int(data.get("operand_b", -1)),
        translation=(float(tr[0]), float(tr[1]), float(tr[2])),
        plane_id=int(data.get("plane_id", -1)),
        sketch=sketch,
        profile_entity_id=int(data.get("profile_entity_id", -1)),
        reversed=bool(data.get("reversed", False)),
        revolve_angle=float(data.get("revolve_angle", 360.0)),
        axis_origin=(float(ao[0]), float(ao[1])),
        axis_direction=(float(ad[0]), float(ad[1])),
        hole_center_u=float(data.get("hole_center_u", 0.0)),
        hole_center_v=float(data.get("hole_center_v", 0.0)),
        source_profile_uv=[(float(p[0]), float(p[1])) for p in src],
        visible=bool(data.get("visible", True)),
        suppressed=bool(data.get("suppressed", False)),
    )


def document_to_dict(doc: Document) -> dict:
    """Serialize document state (not undo/redo/clipboard)."""
    return {
        "format": PROJECT_FORMAT,
        "version": PROJECT_VERSION,
        "name": str(doc.name),
        "display_unit": doc.display_unit.value
        if isinstance(doc.display_unit, Unit)
        else str(doc.display_unit),
        "selected_id": int(doc.selected_id),
        "next_id": int(doc._next_id),
        "sketch_count": int(doc._sketch_count),
        "extrude_count": int(doc._extrude_count),
        "revolve_count": int(doc._revolve_count),
        "fillet_count": int(doc._fillet_count),
        "pocket_count": int(doc._pocket_count),
        "features": [serialize_feature(f) for f in doc.features],
    }


def document_from_dict(data: dict) -> Document:
    """Build a new Document from serialized state. Undo stacks start empty."""
    if not isinstance(data, dict):
        raise ProjectIOError("project root must be a JSON object")
    fmt = data.get("format")
    if fmt != PROJECT_FORMAT:
        raise ProjectIOError(
            f"not a Grok CAD project (format={fmt!r}, expected {PROJECT_FORMAT!r})"
        )
    ver = int(data.get("version", 0))
    if ver < 1 or ver > PROJECT_VERSION:
        raise ProjectIOError(f"unsupported project version {ver}")

    doc = Document()
    doc.name = str(data.get("name", "Untitled"))
    try:
        doc.display_unit = Unit(str(data.get("display_unit", "mm")))
    except ValueError:
        doc.display_unit = Unit.MM
    doc.selected_id = int(data.get("selected_id", -1))
    doc._next_id = int(data.get("next_id", 1))
    doc._sketch_count = int(data.get("sketch_count", 0))
    doc._extrude_count = int(data.get("extrude_count", 0))
    doc._revolve_count = int(data.get("revolve_count", 0))
    doc._fillet_count = int(data.get("fillet_count", 0))
    doc._pocket_count = int(data.get("pocket_count", 0))
    doc.features = []
    for fd in data.get("features") or []:
        doc.features.append(deserialize_feature(fd))
    # Sanity: next_id covers max feature id
    if doc.features:
        doc._next_id = max(doc._next_id, max(f.id for f in doc.features) + 1)
    if doc.selected_id >= 0 and doc.find(doc.selected_id) is None:
        doc.selected_id = doc.features[0].id if doc.features else -1
    doc._undo_stack.clear()
    doc._redo_stack.clear()
    doc._clipboard = None
    doc._paste_n = 0
    doc.dirty = False
    return doc


def save_document(doc: Document, path: Union[str, Path]) -> Path:
    """Write document to ``path`` (.gcad JSON). Marks document clean."""
    out = Path(path)
    if out.suffix.lower() not in (".gcad", ".json"):
        out = out.with_suffix(DEFAULT_EXTENSION)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Prefer file stem as the document name stored in the project
    doc.name = out.stem
    payload = document_to_dict(doc)
    text = json.dumps(payload, indent=2, sort_keys=False)
    out.write_text(text, encoding="utf-8")
    doc.dirty = False
    return out


def load_document(path: Union[str, Path]) -> Document:
    """Load a Document from a .gcad project file."""
    p = Path(path)
    if not p.is_file():
        raise ProjectIOError(f"file not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProjectIOError(f"invalid JSON: {exc}") from exc
    doc = document_from_dict(data)
    doc.name = p.stem
    doc.dirty = False
    return doc


def replace_document_contents(target: Document, source: Document) -> None:
    """Copy all model state from ``source`` into ``target`` (in-place swap)."""
    target.name = source.name
    target.features = list(source.features)
    target.selected_id = source.selected_id
    target.display_unit = source.display_unit
    target._next_id = source._next_id
    target._sketch_count = source._sketch_count
    target._extrude_count = source._extrude_count
    target._revolve_count = source._revolve_count
    target._fillet_count = source._fillet_count
    target._pocket_count = source._pocket_count
    target._undo_stack.clear()
    target._redo_stack.clear()
    target._clipboard = None
    target._paste_n = 0
    target.dirty = False
