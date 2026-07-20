"""SolidWorks-like sketch DOF: fully defined = nothing can move.

Colors and free-DOF counts come from *mobility probes*, not constraint
head-counts. A redundant parallel on two already-horizontal lines does not
make them black — they can still translate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from cadcore.constraints import (
    ConstraintKind,
    SketchConstraint,
    _get_point,
    constraint_residual,
    max_residual,
    solve_sketch,
)
from cadcore.sketch import (
    ArcEntity,
    CircleEntity,
    LineEntity,
    RectEntity,
    Sketch,
    SketchEntity,
    snapshot_sketch_contents,
    restore_sketch_contents,
)

# How far we try to drag a handle when probing (mm)
_PROBE_STEP = 1.0
# If the handle moves at least this fraction of the step, it is free
_FREE_FRAC = 0.2
# Residual above this (mm or dimensionless residual units) → over-defined
_OVER_TOL = 0.05


@dataclass
class DofReport:
    """Sketch-wide DOF summary for status bar / UI."""

    free_dof: int
    free_handles: int
    locked_handles: int
    over_entities: int
    under_entities: int
    well_entities: int
    guilty: Optional[str]  # human label of worst residual constraint


def _dof_handles(ent: SketchEntity) -> List[str]:
    """Independent-ish handles to probe for this entity."""
    if isinstance(ent, LineEntity):
        return ["p0", "p1"]
    if isinstance(ent, RectEntity):
        return ["c0", "c1", "c2", "c3"]
    if isinstance(ent, CircleEntity):
        return ["center", "rim"]  # rim = radius DOF via set_handle
    if isinstance(ent, ArcEntity):
        return ["p0", "p1", "center"]
    # spline etc.
    try:
        return [h.name for h in ent.handles() if h.name not in ("mid",)]
    except Exception:
        return []


def _probe_handle_free(sk: Sketch, eid: int, handle: str, step: float = _PROBE_STEP) -> bool:
    """True if this handle can still move while satisfying constraints."""
    before = snapshot_sketch_contents(sk)
    try:
        p = _get_point(sk, eid, handle)
        if p is None:
            # circle rim is not a _get_point handle — special case
            ent = sk.find_entity(eid)
            if isinstance(ent, CircleEntity) and handle == "rim":
                return _probe_circle_radius_free(sk, ent, step)
            return False
        for du, dv in ((step, 0.0), (0.0, step), (-step, 0.0)):
            restore_sketch_contents(sk, before)
            target = (float(p[0] + du), float(p[1] + dv))
            solve_sketch(sk, drag=(eid, handle, target), max_iters=36)
            p2 = _get_point(sk, eid, handle)
            if p2 is None:
                continue
            moved = float(np.linalg.norm(p2 - p))
            # Followed the drag a meaningful amount → free
            if moved >= step * _FREE_FRAC:
                return True
        return False
    finally:
        restore_sketch_contents(sk, before)


def _probe_circle_radius_free(sk: Sketch, circ: CircleEntity, step: float) -> bool:
    """Probe whether radius can change (rim handle)."""
    before = snapshot_sketch_contents(sk)
    try:
        r0 = float(circ.radius)
        rim = (circ.center[0] + r0 + step, circ.center[1])
        solve_sketch(sk, drag=(circ.id, "rim", rim), max_iters=36)
        # set_handle rim may not go through solve drag for circle — use set_handle path
        ent = sk.find_entity(circ.id)
        if not isinstance(ent, CircleEntity):
            return False
        # After solve, if radius still ~r0, locked; if changed, free
        # Also try direct: drag sim
        restore_sketch_contents(sk, before)
        ent = sk.find_entity(circ.id)
        assert isinstance(ent, CircleEntity)
        ent.set_handle("rim", rim)
        solve_sketch(sk, max_iters=36)
        ent2 = sk.find_entity(circ.id)
        if not isinstance(ent2, CircleEntity):
            return False
        return abs(float(ent2.radius) - r0) >= step * _FREE_FRAC
    finally:
        restore_sketch_contents(sk, before)


def _entity_overdefined(sk: Sketch, ent: SketchEntity) -> bool:
    """True if any constraint touching this entity has large residual."""
    for c in sk.constraints or []:
        if int(c.e0) != ent.id and int(getattr(c, "e1", -1)) != ent.id:
            continue
        try:
            if float(constraint_residual(sk, c)) > _OVER_TOL:
                return True
        except Exception:
            continue
    # Driving dims that can't be satisfied
    for d in getattr(sk, "dimensions", None) or []:
        if int(d.entity_id) != ent.id and int(getattr(d, "entity_b_id", -1)) != ent.id:
            continue
        from cadcore.constraints import dimension_residual

        try:
            if float(dimension_residual(sk, d)) > _OVER_TOL:
                return True
        except Exception:
            continue
    return False


def entity_dof_status(sk: Sketch, ent: SketchEntity) -> str:
    """Return ``under`` | ``well`` | ``over`` based on mobility, not counts.

    * over  — constraints on this entity conflict (residual)
    * well  — every structural handle is locked (cannot move under drag)
    * under — at least one handle can still move
    """
    if _entity_overdefined(sk, ent):
        return "over"
    handles = _dof_handles(ent)
    if not handles:
        return "under"
    free = 0
    for h in handles:
        if _probe_handle_free(sk, ent.id, h):
            free += 1
    if free == 0:
        return "well"
    return "under"


def free_dof_count(sk: Sketch) -> int:
    """Approximate free DOF: 2 per free point-handle + 1 for free circle radius.

    Coincident points are probed separately (may over-count slightly); good enough
    for a SolidWorks-like status readout.
    """
    n = 0
    for ent in sk.entities:
        if bool(getattr(ent, "construction", False)):
            continue
        for h in _dof_handles(ent):
            if h == "rim":
                if _probe_handle_free(sk, ent.id, h):
                    n += 1
            else:
                # each free handle ≈ up to 2 coords; count 2 if free, 0 if locked
                if _probe_handle_free(sk, ent.id, h):
                    n += 2
    return n


def worst_constraint_label(sk: Sketch) -> Optional[str]:
    """Human-readable label of the constraint/dim with largest residual, or None."""
    best_r = _OVER_TOL
    best: Optional[str] = None
    for c in sk.constraints or []:
        try:
            r = float(constraint_residual(sk, c))
        except Exception:
            continue
        if r > best_r:
            best_r = r
            kind = c.kind.name if hasattr(c.kind, "name") else str(c.kind)
            best = f"{kind} (#{c.id}) on entities {c.e0}" + (
                f",{c.e1}" if c.e1 >= 0 else ""
            )
    from cadcore.constraints import dimension_residual

    for d in getattr(sk, "dimensions", None) or []:
        try:
            r = float(dimension_residual(sk, d))
        except Exception:
            continue
        if r > best_r:
            best_r = r
            best = f"dimension {d.role} (#{d.id}) on entity {d.entity_id}"
    return best


def sketch_dof_report(sk: Sketch) -> DofReport:
    free_h = locked_h = 0
    under = well = over = 0
    for ent in sk.entities:
        if bool(getattr(ent, "construction", False)):
            continue
        st = entity_dof_status(sk, ent)
        if st == "over":
            over += 1
        elif st == "well":
            well += 1
        else:
            under += 1
        for h in _dof_handles(ent):
            if _probe_handle_free(sk, ent.id, h):
                free_h += 1
            else:
                locked_h += 1
    guilty = worst_constraint_label(sk) if over > 0 or max_residual(sk) > _OVER_TOL else None
    return DofReport(
        free_dof=free_dof_count(sk),
        free_handles=free_h,
        locked_handles=locked_h,
        over_entities=over,
        under_entities=under,
        well_entities=well,
        guilty=guilty,
    )


def format_dof_status_line(sk: Sketch) -> str:
    """One-line status like SolidWorks: free DOF + optional conflict."""
    rep = sketch_dof_report(sk)
    if rep.over_entities > 0 or rep.guilty:
        g = rep.guilty or "constraint conflict"
        return f"Over-defined · free DOF ≈ {rep.free_dof} · conflict: {g}"
    if rep.free_dof <= 0:
        return f"Fully defined · free DOF = 0 · {rep.well_entities} black entit(y/ies)"
    return (
        f"Under-defined · free DOF ≈ {rep.free_dof} · "
        f"{rep.under_entities} blue, {rep.well_entities} black"
    )
