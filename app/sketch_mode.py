"""Sketch mode controller: tools, snapping, handle editing, polyline chaining,
SolidWorks-style box multi-selection (window vs crossing).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Set, Tuple

import numpy as np

from cadcore.sketch import (
    ArcEntity,
    CircleEntity,
    EntityKind,
    Handle,
    HandleKind,
    LineEntity,
    RectEntity,
    Sketch,
    SketchEntity,
    Vec2,
)

# Default world-unit fallbacks (used when no pixel scale is supplied)
SNAP_GRID = 0.25
SNAP_POINT = 0.15
SNAP_ANGLE_DEG = 7.0
SNAP_HV = 0.12  # legacy
# Preferred screen-space snap radius (pixels) — converted to world via camera
SNAP_POINT_PX = 14.0
# Auto-close / chain-start snap uses the same point tolerance
# Box-select: drag shorter than this (display pixels) = click clear, not box
BOX_SELECT_MIN_PX = 3.0


class SketchTool(Enum):
    SELECT = auto()
    LINE = auto()
    RECTANGLE = auto()
    CIRCLE = auto()
    ARC = auto()
    SPLINE = auto()
    DIMENSION = auto()
    TRIM = auto()
    EXTEND = auto()
    OFFSET = auto()  # Smart Dimension — click entity, value drives geometry


@dataclass
class SnapResult:
    uv: Vec2
    kind: str  # "grid" | "point" | "origin" | "h" | "v" | "none" | "chain_start"
    ref: Optional[Vec2] = None


@dataclass
class DragState:
    entity_id: int
    handle_name: str
    handle_kind: HandleKind
    start_uv: Vec2
    start_entity_snapshot: object = None


@dataclass
class DrawState:
    tool: SketchTool
    points: List[Vec2] = field(default_factory=list)
    # Polyline chain: first point of the whole chain (for auto-close)
    chain_start: Optional[Vec2] = None
    # How many segments already committed in this chain
    chain_segments: int = 0


@dataclass
class BoxSelectState:
    """In-progress SolidWorks-style selection rectangle."""

    press_uv: Vec2
    press_xy: Tuple[float, float]  # widget display coords (x right, y down)
    current_uv: Vec2
    current_xy: Tuple[float, float]
    # Selection present when box started (for Shift-add)
    baseline_ids: Set[int] = field(default_factory=set)
    add_mode: bool = False  # Shift held at press (or release)

    @property
    def is_window(self) -> bool:
        """L→R (display x grows) = WINDOW; R→L = CROSSING."""
        return float(self.current_xy[0]) >= float(self.press_xy[0])

    @property
    def drag_px(self) -> float:
        dx = float(self.current_xy[0]) - float(self.press_xy[0])
        dy = float(self.current_xy[1]) - float(self.press_xy[1])
        return float(np.hypot(dx, dy))

    def uv_rect(self) -> Tuple[float, float, float, float]:
        """Normalized axis-aligned UV box: (u0, v0, u1, v1) with u0<=u1, v0<=v1."""
        u0 = min(float(self.press_uv[0]), float(self.current_uv[0]))
        u1 = max(float(self.press_uv[0]), float(self.current_uv[0]))
        v0 = min(float(self.press_uv[1]), float(self.current_uv[1]))
        v1 = max(float(self.press_uv[1]), float(self.current_uv[1]))
        return u0, v0, u1, v1


class SketchController:
    def __init__(self, sketch: Sketch) -> None:
        self.sketch = sketch
        self.tool = SketchTool.SELECT
        self.draw: Optional[DrawState] = None
        self.drag: Optional[DragState] = None
        self.box_select: Optional[BoxSelectState] = None
        self.hover_handle: Optional[Handle] = None
        # Multi-selection source of truth (entity geometry)
        self.selected_ids: Set[int] = set()
        # Closed-region (extrude profile) selection — ids may be negative (line loops)
        self.selected_profile_ids: Set[int] = set()
        self.preview_uv: Optional[Vec2] = None
        self.last_snap: SnapResult = SnapResult((0, 0), "none")
        # World-unit point snap radius; viewport overwrites from pixel scale
        self.snap_point_tol: float = SNAP_POINT
        self.snap_grid: float = SNAP_GRID
        self._dim_first_id: Optional[int] = None  # Smart Dimension two-click angle

    # --- selection property (compat for single-id consumers) ---
    @property
    def selected_entity_id(self) -> int:
        """If exactly one entity is selected return its id, else -1."""
        if len(self.selected_ids) == 1:
            return next(iter(self.selected_ids))
        return -1

    @selected_entity_id.setter
    def selected_entity_id(self, eid: int) -> None:
        """Replace selection with a single id, or clear when eid < 0."""
        if eid is None or int(eid) < 0:
            self.selected_ids.clear()
        else:
            self.selected_ids = {int(eid)}
            self.selected_profile_ids.clear()

    def clear_selection(self) -> None:
        self.selected_ids.clear()
        self.selected_profile_ids.clear()

    def set_selection(self, ids) -> None:
        self.selected_ids = {int(i) for i in ids if int(i) >= 0}
        if self.selected_ids:
            self.selected_profile_ids.clear()

    def add_to_selection(self, ids) -> None:
        for i in ids:
            if int(i) >= 0:
                self.selected_ids.add(int(i))
        if self.selected_ids:
            self.selected_profile_ids.clear()

    def clear_profile_selection(self) -> None:
        self.selected_profile_ids.clear()

    def set_profile_selection(self, ids) -> None:
        self.selected_profile_ids = {int(i) for i in ids}
        if self.selected_profile_ids:
            self.selected_ids.clear()

    def toggle_profile_id(self, pid: int) -> None:
        pid = int(pid)
        if pid in self.selected_profile_ids:
            self.selected_profile_ids.discard(pid)
        else:
            self.selected_profile_ids.add(pid)
        self.selected_ids.clear()

    def set_snap_world_tol(self, point_tol: float, *, grid: Optional[float] = None) -> None:
        """Update world-space snap tolerances (from pixel radius × world/px)."""
        self.snap_point_tol = max(1e-9, float(point_tol))
        if grid is not None:
            self.snap_grid = max(1e-9, float(grid))

    def set_tool(self, tool: SketchTool) -> None:
        self.tool = tool
        self.draw = None
        self.drag = None
        self.box_select = None
        self.preview_uv = None
        # Dimension keeps selection context; other draw tools clear it
        if tool not in (SketchTool.SELECT, SketchTool.DIMENSION):
            self.selected_ids.clear()
            self.hover_handle = None

    def is_drawing(self) -> bool:
        return self.draw is not None and len(self.draw.points) > 0

    def is_box_selecting(self) -> bool:
        return self.box_select is not None

    def in_line_chain(self) -> bool:
        return (
            self.draw is not None
            and self.draw.tool is SketchTool.LINE
            and self.draw.chain_start is not None
        )

    def cancel_drawing(self) -> bool:
        if self.draw is None and self.drag is None and self.box_select is None:
            return False
        self.draw = None
        self.drag = None
        self.box_select = None
        self.preview_uv = None
        return True

    def cancel(self) -> None:
        self.cancel_drawing()

    def end_line_chain(self) -> Optional[str]:
        """Finish an open polyline without placing another point (Enter / double-click)."""
        if self.draw is None or self.draw.tool is not SketchTool.LINE:
            return None
        n = self.draw.chain_segments
        self.draw = None
        self.preview_uv = None
        return f"ChainEnd:{n}" if n > 0 else "ChainEnd:0"

    def confirm_current(self) -> Optional[str]:
        """Enter: commit current rubber-band point, or end line chain if idle mid-chain."""
        if self.draw is None or not self.draw.points:
            return None
        # Mid-chain with only the continuing start point → end chain
        if (
            self.draw.tool is SketchTool.LINE
            and len(self.draw.points) == 1
            and self.draw.chain_segments > 0
        ):
            return self.end_line_chain()
        if self.preview_uv is None:
            if len(self.draw.points) < 2:
                return None
        else:
            if len(self.draw.points) == 1:
                self.draw.points.append(self.preview_uv)
        return self._try_finish_draw()

    def commit_line_length(self, length_mm: float) -> Optional[str]:
        if self.tool is not SketchTool.LINE:
            return None
        if self.draw is None or len(self.draw.points) != 1:
            return None
        L = float(length_mm)
        if not np.isfinite(L) or L <= 1e-12:
            return None
        p0 = self.draw.points[0]
        if self.preview_uv is not None:
            du = self.preview_uv[0] - p0[0]
            dv = self.preview_uv[1] - p0[1]
        else:
            du, dv = 1.0, 0.0
        nrm = float(np.hypot(du, dv))
        if nrm < 1e-12:
            du, dv = 1.0, 0.0
            nrm = 1.0
        nu, nv = du / nrm, dv / nrm
        p1 = (p0[0] + nu * L, p0[1] + nv * L)
        self.draw.points.append(p1)
        return self._try_finish_draw()

    # --- snapping ---
    def snap(self, uv: Vec2, *, drawing: bool = False) -> SnapResult:
        """Snap with point-priority over grid; point tol is ``snap_point_tol`` (world)."""
        u, v = float(uv[0]), float(uv[1])
        ptol = self.snap_point_tol

        # Prefer nearest existing endpoint / origin (point snap over grid)
        best: Optional[SnapResult] = None
        best_d = ptol
        targets = list(self.sketch.snap_targets())  # includes origin
        if (
            drawing
            and self.draw
            and self.draw.tool is SketchTool.LINE
            and self.draw.chain_start is not None
            and self.draw.chain_segments >= 2
        ):
            targets.append(self.draw.chain_start)
        for p in targets:
            d = float(np.hypot(u - p[0], v - p[1]))
            if d < best_d:
                best_d = d
                if abs(p[0]) < 1e-15 and abs(p[1]) < 1e-15:
                    kind = "origin"
                elif (
                    self.draw
                    and self.draw.chain_start is not None
                    and abs(p[0] - self.draw.chain_start[0]) < 1e-12
                    and abs(p[1] - self.draw.chain_start[1]) < 1e-12
                ):
                    kind = "chain_start"
                else:
                    kind = "point"
                best = SnapResult((float(p[0]), float(p[1])), kind, p)
        if best is not None:
            self.last_snap = best
            return best

        # Angular H/V when drawing free endpoint
        if drawing and self.draw and self.draw.points:
            p0 = self.draw.points[0]
            du = u - p0[0]
            dv = v - p0[1]
            length = float(np.hypot(du, dv))
            if length > 1e-12:
                angle = float(np.degrees(np.arctan2(dv, du)))
                orthos = (0.0, 90.0, -90.0, 180.0, -180.0)
                nearest = min(orthos, key=lambda o: abs(angle - o))
                if abs(angle - nearest) <= SNAP_ANGLE_DEG:
                    rad = float(np.radians(nearest))
                    u = p0[0] + length * float(np.cos(rad))
                    v = p0[1] + length * float(np.sin(rad))
                    kind = (
                        "h"
                        if abs(nearest) < 1e-9 or abs(abs(nearest) - 180.0) < 1e-9
                        else "v"
                    )
                    res = SnapResult((u, v), kind, None)
                    self.last_snap = res
                    return res
            # Free angle — no grid pull
            res = SnapResult((u, v), "none", None)
            self.last_snap = res
            return res

        # Grid only when not free-drawing a second point and no point snap
        g = self.snap_grid
        gu = round(u / g) * g
        gv = round(v / g) * g
        if abs(u - gu) <= g * 0.35 and abs(v - gv) <= g * 0.35:
            res = SnapResult((gu, gv), "grid")
            self.last_snap = res
            return res

        res = SnapResult((u, v), "none")
        self.last_snap = res
        return res

    def pick_handle(self, uv: Vec2, tol: Optional[float] = None) -> Optional[Handle]:
        t = self.snap_point_tol if tol is None else float(tol)
        best: Optional[Handle] = None
        best_d = t
        for h in self.sketch.all_handles():
            d = float(np.hypot(uv[0] - h.uv[0], uv[1] - h.uv[1]))
            if d < best_d:
                best_d = d
                best = h
        return best

    def pick_entity_body(self, uv: Vec2, tol: Optional[float] = None) -> Optional[int]:
        t = self.snap_point_tol if tol is None else float(tol)
        best_id = -1
        best_d = t
        for e in self.sketch.entities:
            d = self._dist_to_entity(e, uv)
            if d < best_d:
                best_d = d
                best_id = e.id
        return best_id if best_id >= 0 else None

    def _dist_to_entity(self, e: SketchEntity, uv: Vec2) -> float:
        u, v = uv
        if isinstance(e, LineEntity):
            return _dist_point_segment(uv, e.p0, e.p1)
        if isinstance(e, CircleEntity):
            return abs(float(np.hypot(u - e.center[0], v - e.center[1]) - e.radius))
        if isinstance(e, ArcEntity):
            # Distance to arc curve: radial residual if angle on sweep, else to nearer end
            dx, dy = u - e.center[0], v - e.center[1]
            r = float(np.hypot(dx, dy))
            ang = float(np.atan2(dy, dx))
            # Is angle on arc?
            def on_sweep(a: float) -> bool:
                d0 = a - e.a0
                de = e.sweep()
                if e.ccw:
                    while d0 < 0:
                        d0 += 2 * np.pi
                    while d0 >= 2 * np.pi:
                        d0 -= 2 * np.pi
                    return 0 <= d0 <= de + 1e-9
                while d0 > 0:
                    d0 -= 2 * np.pi
                while d0 <= -2 * np.pi:
                    d0 += 2 * np.pi
                return de - 1e-9 <= d0 <= 0

            if on_sweep(ang):
                return abs(r - e.radius)
            p0, p1 = e.p0(), e.p1()
            return min(
                float(np.hypot(u - p0[0], v - p0[1])),
                float(np.hypot(u - p1[0], v - p1[1])),
            )
        if isinstance(e, RectEntity):
            cs = e.corners()
            edges = [(cs[0], cs[1]), (cs[1], cs[2]), (cs[2], cs[3]), (cs[3], cs[0])]
            return min(_dist_point_segment(uv, a, b) for a, b in edges)
        return 1e9

    # --- box hit tests (UV space) ---
    def entities_in_box(
        self, u0: float, v0: float, u1: float, v1: float, *, window: bool
    ) -> Set[int]:
        """Return entity ids matching WINDOW (fully inside) or CROSSING hit rules."""
        out: Set[int] = set()
        for e in self.sketch.entities:
            if window:
                if _entity_fully_inside(e, u0, v0, u1, v1):
                    out.add(e.id)
            else:
                if _entity_crosses_or_inside(e, u0, v0, u1, v1):
                    out.add(e.id)
        return out

    # --- mouse ---
    def on_move(
        self,
        raw_uv: Vec2,
        *,
        display_xy: Optional[Tuple[float, float]] = None,
    ) -> None:
        if self.tool == SketchTool.SELECT:
            if self.box_select is not None:
                self.box_select.current_uv = (float(raw_uv[0]), float(raw_uv[1]))
                if display_xy is not None:
                    self.box_select.current_xy = (
                        float(display_xy[0]),
                        float(display_xy[1]),
                    )
                self.preview_uv = raw_uv
                return
            if self.drag is not None:
                sn = self.snap(raw_uv, drawing=False)
                self._apply_drag(sn.uv)
                self.preview_uv = sn.uv
            else:
                self.hover_handle = self.pick_handle(raw_uv)
                self.preview_uv = raw_uv
        else:
            sn = self.snap(raw_uv, drawing=True)
            self.preview_uv = sn.uv

    def on_press(
        self,
        raw_uv: Vec2,
        *,
        display_xy: Optional[Tuple[float, float]] = None,
        shift: bool = False,
        ctrl: bool = False,
    ) -> Optional[str]:
        if self.tool == SketchTool.DIMENSION:
            # Smart Dimension: one entity → length/diameter; two lines → angle
            from cadcore.sketch import infer_dimension_role

            h = self.pick_handle(raw_uv)
            eid = h.entity_id if h is not None else self.pick_entity_body(raw_uv)
            if eid is None:
                self._dim_first_id = None
                return "DimPickMiss"
            ent = self.sketch.find_entity(eid)
            if ent is None:
                return "DimPickMiss"
            # Second click on another line → angle between them
            first = getattr(self, "_dim_first_id", None)
            if first is not None and int(first) != int(eid):
                e0 = self.sketch.find_entity(int(first))
                if isinstance(e0, LineEntity) and isinstance(ent, LineEntity):
                    self._dim_first_id = None
                    self.selected_ids = {int(first), int(eid)}
                    return f"DimPick:{int(first)}:angle:{int(eid)}"
            if isinstance(ent, LineEntity):
                # Wait for optional second line; first click alone after re-click same = length
                if first is not None and int(first) == int(eid):
                    self._dim_first_id = None
                    self.selected_entity_id = eid
                    return f"DimPick:{eid}:length"
                self._dim_first_id = int(eid)
                self.selected_entity_id = eid
                return "DimPickWait"
            self._dim_first_id = None
            self.selected_entity_id = eid
            role = infer_dimension_role(ent, uv_hint=raw_uv)
            return f"DimPick:{eid}:{role}"

        if self.tool == SketchTool.TRIM:
            from cadcore.sketch_ops import trim_entity_at

            eid = self.pick_entity_body(raw_uv)
            ent = self.sketch.find_entity(eid) if eid is not None else None
            if ent is None:
                return "TrimMiss"
            if not isinstance(ent, (LineEntity, ArcEntity, CircleEntity)):
                return "TrimMiss"
            # Snapshot for undo is handled by viewport when msg starts with Trim:
            n_before = len(self.sketch.entities)
            ok = trim_entity_at(self.sketch, ent, raw_uv)
            if not ok:
                return "TrimMiss"
            return f"Trim:{eid}:n{n_before}->{len(self.sketch.entities)}"

        if self.tool == SketchTool.EXTEND:
            from cadcore.sketch_ops import extend_entity_at

            eid = self.pick_entity_body(raw_uv)
            ent = self.sketch.find_entity(eid) if eid is not None else None
            if ent is None:
                return "ExtendMiss"
            if not isinstance(ent, (LineEntity, ArcEntity)):
                # Circles are closed — nothing to extend
                return "ExtendMiss"
            ok = extend_entity_at(self.sketch, ent, raw_uv)
            if not ok:
                return "ExtendMiss"
            return f"Extend:{eid}"

        if self.tool == SketchTool.OFFSET:
            eid = self.pick_entity_body(raw_uv)
            ent = self.sketch.find_entity(eid) if eid is not None else None
            if ent is None:
                return "OffsetMiss"
            # Fixed default offset 5 mm — refined via PropertyManager later
            from cadcore.sketch_ops import offset_circle, offset_line

            try:
                if isinstance(ent, LineEntity):
                    off = offset_line(ent, 5.0)
                    self.sketch.add_line(off.p0, off.p1)
                elif isinstance(ent, CircleEntity):
                    off = offset_circle(ent, 5.0)
                    self.sketch.add_circle(off.center, off.radius)
                else:
                    return "OffsetMiss"
            except ValueError:
                return "OffsetMiss"
            return f"Offset:{ent.id}"

        if self.tool == SketchTool.SELECT:
            h = self.pick_handle(raw_uv)
            if h is not None:
                self.box_select = None
                self.selected_entity_id = h.entity_id
                self.drag = DragState(
                    entity_id=h.entity_id,
                    handle_name=h.name,
                    handle_kind=h.kind,
                    start_uv=h.uv,
                )
                return f"Drag {h.name}"
            # Edge/body first: clicking the line means the shape, not the region fill.
            eid = self.pick_entity_body(raw_uv)
            if eid is not None:
                self.box_select = None
                self.selected_profile_ids.clear()
                if shift:
                    # Shift-click toggles / adds single entity
                    if eid in self.selected_ids:
                        self.selected_ids.discard(eid)
                    else:
                        self.selected_ids.add(eid)
                else:
                    self.selected_entity_id = eid
                return f"Selected entity {eid}"
            # Interior of a region OR empty canvas: do not decide yet.
            # Press+drag → box select; press+release (no drag) → region pick if inside.
            multi = bool(shift or ctrl)
            baseline = set(self.selected_ids) if multi else set()
            if not multi:
                self.selected_ids.clear()
                self.selected_profile_ids.clear()
            xy = (
                (float(display_xy[0]), float(display_xy[1]))
                if display_xy is not None
                else (0.0, 0.0)
            )
            uv = (float(raw_uv[0]), float(raw_uv[1]))
            self.box_select = BoxSelectState(
                press_uv=uv,
                press_xy=xy,
                current_uv=uv,
                current_xy=xy,
                baseline_ids=baseline,
                add_mode=multi,
            )
            return "BoxSelect"

        sn = self.snap(raw_uv, drawing=True)
        uv = sn.uv
        if self.draw is None:
            self.draw = DrawState(tool=self.tool, points=[uv], chain_start=None, chain_segments=0)
            return f"Place next point ({self.tool.name.title()})"
        self.draw.points.append(uv)
        return self._try_finish_draw()

    def on_release(
        self,
        raw_uv: Vec2,
        *,
        display_xy: Optional[Tuple[float, float]] = None,
        shift: bool = False,
        ctrl: bool = False,
    ) -> Optional[str]:
        if self.drag is not None:
            sn = self.snap(raw_uv, drawing=False)
            self._apply_drag(sn.uv)
            self.drag = None
            return None

        if self.box_select is not None:
            bs = self.box_select
            bs.current_uv = (float(raw_uv[0]), float(raw_uv[1]))
            if display_xy is not None:
                bs.current_xy = (float(display_xy[0]), float(display_xy[1]))
            # Shift/Ctrl on release also enables add
            add = bs.add_mode or bool(shift or ctrl)
            self.box_select = None
            if bs.drag_px < BOX_SELECT_MIN_PX:
                # Click without drag: pick closed region if press was inside one.
                # (Press+drag already diverged into box select above this threshold.)
                from cadcore.profiles import pick_closed_profile_at

                prof = pick_closed_profile_at(self.sketch, bs.press_uv)
                if prof is not None:
                    pid = int(getattr(prof, "id", -1))
                    if add:
                        # Ctrl/Shift: toggle region (profiles kept if press was multi)
                        self.toggle_profile_id(pid)
                    else:
                        self.set_profile_selection([pid])
                    n = len(self.selected_profile_ids)
                    return f"Selected profile {pid} (n={n})"
                # Click empty: keep cleared selection (or baseline if multi-add click)
                if not add:
                    self.selected_ids.clear()
                    self.selected_profile_ids.clear()
                return "BoxSelectClear"
            u0, v0, u1, v1 = bs.uv_rect()
            window = bs.is_window
            hits = self.entities_in_box(u0, v0, u1, v1, window=window)
            if add:
                self.selected_ids = set(bs.baseline_ids) | hits
            else:
                self.selected_ids = set(hits)
            self.selected_profile_ids.clear()
            mode = "window" if window else "crossing"
            return f"BoxSelect:{mode}:{len(self.selected_ids)}"
        return None

    def _apply_drag(self, uv: Vec2) -> None:
        if self.drag is None:
            return
        ent = self.sketch.find_entity(self.drag.entity_id)
        if ent is None:
            return
        from cadcore.sketch import SplineEntity

        if isinstance(ent, (LineEntity, RectEntity, CircleEntity, ArcEntity, SplineEntity)):
            ent.set_handle(self.drag.handle_name, uv)
            # Keep geometric relationships + driving dimensions after the drag
            if getattr(self.sketch, "constraints", None) or getattr(
                self.sketch, "dimensions", None
            ):
                from cadcore.constraints import solve_sketch

                solve_sketch(
                    self.sketch,
                    drag=(
                        int(self.drag.entity_id),
                        str(self.drag.handle_name),
                        (float(uv[0]), float(uv[1])),
                    ),
                )

    def _try_finish_draw(self) -> Optional[str]:
        if self.draw is None:
            return None
        pts = self.draw.points
        tool = self.draw.tool
        if tool is SketchTool.LINE and len(pts) >= 2:
            p0 = (float(pts[-2][0]), float(pts[-2][1]))
            p1 = (float(pts[-1][0]), float(pts[-1][1]))
            # Auto-close onto chain start (exact shared coordinates)
            closed = False
            if (
                self.draw.chain_start is not None
                and self.draw.chain_segments >= 2
            ):
                cs = self.draw.chain_start
                if float(np.hypot(p1[0] - cs[0], p1[1] - cs[1])) <= self.snap_point_tol:
                    p1 = (float(cs[0]), float(cs[1]))
                    closed = True
            # Degenerate zero-length: ignore
            if float(np.hypot(p1[0] - p0[0], p1[1] - p0[1])) < 1e-12:
                self.draw.points.pop()
                return None
            self.sketch.add_line(p0, p1)
            self.draw.chain_segments += 1
            if self.draw.chain_start is None:
                self.draw.chain_start = p0
            if closed:
                self.draw = None
                self.preview_uv = None
                return "LineClosed"
            # Continue chain from exact endpoint
            self.draw = DrawState(
                tool=SketchTool.LINE,
                points=[p1],
                chain_start=self.draw.chain_start,
                chain_segments=self.draw.chain_segments,
            )
            return "Line"
        if tool is SketchTool.RECTANGLE and len(pts) >= 2:
            self.sketch.add_rectangle(pts[0], pts[1])
            self.draw = None
            return "Rectangle"
        if tool is SketchTool.CIRCLE and len(pts) >= 2:
            r = float(np.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]))
            self.sketch.add_circle(pts[0], max(r, 1e-6))
            self.draw = None
            return "Circle"
        if tool is SketchTool.ARC and len(pts) >= 3:
            p0 = (float(pts[0][0]), float(pts[0][1]))
            mid = (float(pts[1][0]), float(pts[1][1]))
            p1 = (float(pts[2][0]), float(pts[2][1]))
            try:
                arc = self.sketch.add_arc(p0, mid, p1)
            except ValueError:
                self.draw.points.pop()
                return None
            # If start lies on a line endpoint, auto coincident + tangent
            self._maybe_auto_tangent_at_line(arc, end="p0")
            self._maybe_auto_tangent_at_line(arc, end="p1")
            self.draw = None
            return "Arc"
        if tool is SketchTool.SPLINE:
            # Keep collecting; finish when same point double-clicked or ≥4 pts + close to start
            if len(pts) >= 2:
                last = pts[-1]
                prev = pts[-2]
                if float(np.hypot(last[0] - prev[0], last[1] - prev[1])) < 1e-9:
                    # double-click finish: drop duplicate last
                    pts = pts[:-1]
                    if len(pts) >= 2:
                        self.sketch.add_spline(pts)
                        self.draw = None
                        return "Spline"
                    self.draw.points.pop()
                    return None
            # Continue collecting (need at least 2 points; keep draw open until double-click)
            if len(pts) >= 2:
                return "SplinePoint"
            return None
        return None

    def _maybe_auto_tangent_at_line(self, arc: ArcEntity, *, end: str) -> None:
        """If arc end coincides with a line end, add coincident + tangent."""
        from cadcore.constraints import (
            ConstraintKind,
            SketchConstraint,
            add_constraint,
        )

        pt = arc.p0() if end == "p0" else arc.p1()
        tol = max(self.snap_point_tol, 1e-6)
        for e in self.sketch.entities:
            if not isinstance(e, LineEntity) or e.id == arc.id:
                continue
            for lh, lp in (("p0", e.p0), ("p1", e.p1)):
                if abs(lp[0] - pt[0]) > tol or abs(lp[1] - pt[1]) > tol:
                    continue
                # Snap exactly
                if end == "p0":
                    # keep arc end on line point via angle update
                    v = np.array(
                        [lp[0] - arc.center[0], lp[1] - arc.center[1]],
                        dtype=np.float64,
                    )
                    n = float(np.linalg.norm(v))
                    if n > 1e-12:
                        arc.radius = n
                        arc.a0 = float(np.atan2(v[1], v[0]))
                else:
                    v = np.array(
                        [lp[0] - arc.center[0], lp[1] - arc.center[1]],
                        dtype=np.float64,
                    )
                    n = float(np.linalg.norm(v))
                    if n > 1e-12:
                        arc.radius = n
                        arc.a1 = float(np.atan2(v[1], v[0]))
                try:
                    add_constraint(
                        self.sketch,
                        SketchConstraint(
                            id=-1,
                            kind=ConstraintKind.COINCIDENT,
                            e0=e.id,
                            h0=lh,
                            e1=arc.id,
                            h1=end,
                        ),
                    )
                except ValueError:
                    pass
                try:
                    add_constraint(
                        self.sketch,
                        SketchConstraint(
                            id=-1,
                            kind=ConstraintKind.TANGENT,
                            e0=e.id,
                            e1=arc.id,
                            h1=end,
                        ),
                    )
                except ValueError:
                    pass
                return


# ---------------------------------------------------------------------------
# Geometry helpers for box select
# ---------------------------------------------------------------------------

def _pt_in_rect(p: Vec2, u0: float, v0: float, u1: float, v1: float) -> bool:
    return u0 <= p[0] <= u1 and v0 <= p[1] <= v1


def _segments_intersect(a: Vec2, b: Vec2, c: Vec2, d: Vec2) -> bool:
    """Segment AB intersects segment CD (including endpoints / collinear overlap)."""

    def _orient(p, q, r) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def _on_segment(p, q, r) -> bool:
        return (
            min(p[0], r[0]) - 1e-12 <= q[0] <= max(p[0], r[0]) + 1e-12
            and min(p[1], r[1]) - 1e-12 <= q[1] <= max(p[1], r[1]) + 1e-12
        )

    o1 = _orient(a, b, c)
    o2 = _orient(a, b, d)
    o3 = _orient(c, d, a)
    o4 = _orient(c, d, b)

    if ((o1 > 0) != (o2 > 0)) and ((o3 > 0) != (o4 > 0)):
        return True
    if abs(o1) <= 1e-12 and _on_segment(a, c, b):
        return True
    if abs(o2) <= 1e-12 and _on_segment(a, d, b):
        return True
    if abs(o3) <= 1e-12 and _on_segment(c, a, d):
        return True
    if abs(o4) <= 1e-12 and _on_segment(c, b, d):
        return True
    return False


def _rect_edges(u0: float, v0: float, u1: float, v1: float):
    c = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
    return [(c[0], c[1]), (c[1], c[2]), (c[2], c[3]), (c[3], c[0])]


def _entity_sample_points(e: SketchEntity) -> List[Vec2]:
    if isinstance(e, LineEntity):
        return [e.p0, e.p1]
    if isinstance(e, RectEntity):
        return list(e.corners())
    if isinstance(e, CircleEntity):
        cx, cy = e.center
        r = e.radius
        # BBox corners of the circle (for fully-inside) + cardinals
        return [
            (cx - r, cy - r),
            (cx + r, cy - r),
            (cx + r, cy + r),
            (cx - r, cy + r),
            (cx + r, cy),
            (cx - r, cy),
            (cx, cy + r),
            (cx, cy - r),
            (cx, cy),
        ]
    if isinstance(e, ArcEntity):
        return e.sample_uv(12)
    return []


def _entity_segments(e: SketchEntity) -> List[Tuple[Vec2, Vec2]]:
    if isinstance(e, LineEntity):
        return [(e.p0, e.p1)]
    if isinstance(e, RectEntity):
        cs = e.corners()
        return [(cs[0], cs[1]), (cs[1], cs[2]), (cs[2], cs[3]), (cs[3], cs[0])]
    if isinstance(e, ArcEntity):
        pts = e.sample_uv(12)
        return [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    return []


def _entity_fully_inside(
    e: SketchEntity, u0: float, v0: float, u1: float, v1: float
) -> bool:
    """WINDOW: entire entity geometry inside the box."""
    if isinstance(e, LineEntity):
        return _pt_in_rect(e.p0, u0, v0, u1, v1) and _pt_in_rect(e.p1, u0, v0, u1, v1)
    if isinstance(e, RectEntity):
        return all(_pt_in_rect(c, u0, v0, u1, v1) for c in e.corners())
    if isinstance(e, CircleEntity):
        # Axis-aligned bbox of the circle must be fully inside
        cx, cy = e.center
        r = e.radius
        return (
            u0 <= cx - r
            and cx + r <= u1
            and v0 <= cy - r
            and cy + r <= v1
        )
    if isinstance(e, ArcEntity):
        return all(_pt_in_rect(p, u0, v0, u1, v1) for p in e.sample_uv(16))
    return False


def _circle_crosses_rect(
    c: CircleEntity, u0: float, v0: float, u1: float, v1: float
) -> bool:
    """Exact-ish circle vs AABB: center in rect, any corner in circle, or edge distance."""
    cx, cy = float(c.center[0]), float(c.center[1])
    r = float(c.radius)
    # Center inside box
    if _pt_in_rect((cx, cy), u0, v0, u1, v1):
        return True
    # Any rect corner inside circle
    for p in ((u0, v0), (u1, v0), (u1, v1), (u0, v1)):
        if float(np.hypot(p[0] - cx, p[1] - cy)) <= r + 1e-12:
            return True
    # Clamp center to rect; if distance to clamp point <= r, circle hits the box
    qx = min(max(cx, u0), u1)
    qy = min(max(cy, v0), v1)
    if float(np.hypot(qx - cx, qy - cy)) <= r + 1e-12:
        return True
    # Any circle point (cardinals) inside rect
    for p in (
        (cx + r, cy),
        (cx - r, cy),
        (cx, cy + r),
        (cx, cy - r),
    ):
        if _pt_in_rect(p, u0, v0, u1, v1):
            return True
    return False


def _entity_crosses_or_inside(
    e: SketchEntity, u0: float, v0: float, u1: float, v1: float
) -> bool:
    """CROSSING: any sample inside OR segment/edge intersection OR circle-AABB hit."""
    if _entity_fully_inside(e, u0, v0, u1, v1):
        return True
    for p in _entity_sample_points(e):
        if _pt_in_rect(p, u0, v0, u1, v1):
            return True
    if isinstance(e, CircleEntity):
        return _circle_crosses_rect(e, u0, v0, u1, v1)
    edges = _rect_edges(u0, v0, u1, v1)
    for a, b in _entity_segments(e):
        if _pt_in_rect(a, u0, v0, u1, v1) or _pt_in_rect(b, u0, v0, u1, v1):
            return True
        for c, d in edges:
            if _segments_intersect(a, b, c, d):
                return True
    return False


def _dist_point_segment(p: Vec2, a: Vec2, b: Vec2) -> float:
    ax, ay = a
    bx, by = b
    px, py = p
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    ab2 = abx * abx + aby * aby
    if ab2 < 1e-18:
        return float(np.hypot(apx, apy))
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab2))
    qx, qy = ax + t * abx, ay + t * aby
    return float(np.hypot(px - qx, py - qy))
