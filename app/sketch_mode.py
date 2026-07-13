"""Sketch mode controller: tools, snapping, handle editing, polyline chaining."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np

from cadcore.sketch import (
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


class SketchTool(Enum):
    SELECT = auto()
    LINE = auto()
    RECTANGLE = auto()
    CIRCLE = auto()


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


class SketchController:
    def __init__(self, sketch: Sketch) -> None:
        self.sketch = sketch
        self.tool = SketchTool.SELECT
        self.draw: Optional[DrawState] = None
        self.drag: Optional[DragState] = None
        self.hover_handle: Optional[Handle] = None
        self.selected_entity_id: int = -1
        self.preview_uv: Optional[Vec2] = None
        self.last_snap: SnapResult = SnapResult((0, 0), "none")
        # World-unit point snap radius; viewport overwrites from pixel scale
        self.snap_point_tol: float = SNAP_POINT
        self.snap_grid: float = SNAP_GRID

    def set_snap_world_tol(self, point_tol: float, *, grid: Optional[float] = None) -> None:
        """Update world-space snap tolerances (from pixel radius × world/px)."""
        self.snap_point_tol = max(1e-9, float(point_tol))
        if grid is not None:
            self.snap_grid = max(1e-9, float(grid))

    def set_tool(self, tool: SketchTool) -> None:
        self.tool = tool
        self.draw = None
        self.drag = None
        self.preview_uv = None
        if tool != SketchTool.SELECT:
            self.selected_entity_id = -1
            self.hover_handle = None

    def is_drawing(self) -> bool:
        return self.draw is not None and len(self.draw.points) > 0

    def in_line_chain(self) -> bool:
        return (
            self.draw is not None
            and self.draw.tool is SketchTool.LINE
            and self.draw.chain_start is not None
        )

    def cancel_drawing(self) -> bool:
        if self.draw is None and self.drag is None:
            return False
        self.draw = None
        self.drag = None
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
        if isinstance(e, RectEntity):
            cs = e.corners()
            edges = [(cs[0], cs[1]), (cs[1], cs[2]), (cs[2], cs[3]), (cs[3], cs[0])]
            return min(_dist_point_segment(uv, a, b) for a, b in edges)
        return 1e9

    # --- mouse ---
    def on_move(self, raw_uv: Vec2) -> None:
        if self.tool == SketchTool.SELECT:
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

    def on_press(self, raw_uv: Vec2) -> Optional[str]:
        if self.tool == SketchTool.SELECT:
            h = self.pick_handle(raw_uv)
            if h is not None:
                self.selected_entity_id = h.entity_id
                self.drag = DragState(
                    entity_id=h.entity_id,
                    handle_name=h.name,
                    handle_kind=h.kind,
                    start_uv=h.uv,
                )
                return f"Drag {h.name}"
            eid = self.pick_entity_body(raw_uv)
            if eid is not None:
                self.selected_entity_id = eid
                return f"Selected entity {eid}"
            self.selected_entity_id = -1
            return None

        sn = self.snap(raw_uv, drawing=True)
        uv = sn.uv
        if self.draw is None:
            self.draw = DrawState(tool=self.tool, points=[uv], chain_start=None, chain_segments=0)
            return f"Place next point ({self.tool.name.title()})"
        self.draw.points.append(uv)
        return self._try_finish_draw()

    def on_release(self, raw_uv: Vec2) -> None:
        if self.drag is not None:
            sn = self.snap(raw_uv, drawing=False)
            self._apply_drag(sn.uv)
            self.drag = None

    def _apply_drag(self, uv: Vec2) -> None:
        if self.drag is None:
            return
        ent = self.sketch.find_entity(self.drag.entity_id)
        if ent is None:
            return
        if isinstance(ent, (LineEntity, RectEntity, CircleEntity)):
            ent.set_handle(self.drag.handle_name, uv)

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
        return None


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
