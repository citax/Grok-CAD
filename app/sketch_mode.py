"""Sketch mode controller: tools, snapping, handle editing (logic only)."""

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

# Pixel-ish tolerances converted using a scale factor (world units per pixel estimate)
SNAP_GRID = 0.25
SNAP_POINT = 0.15
SNAP_HV = 0.12  # world units for H/V inference


class SketchTool(Enum):
    SELECT = auto()
    LINE = auto()
    RECTANGLE = auto()
    CIRCLE = auto()


@dataclass
class SnapResult:
    uv: Vec2
    kind: str  # "grid" | "point" | "origin" | "h" | "v" | "none"
    ref: Optional[Vec2] = None


@dataclass
class DragState:
    entity_id: int
    handle_name: str
    handle_kind: HandleKind
    start_uv: Vec2
    # For whole-entity moves (midpoint / center)
    start_entity_snapshot: object = None


@dataclass
class DrawState:
    tool: SketchTool
    points: List[Vec2] = field(default_factory=list)  # committed clicks so far


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

    def set_tool(self, tool: SketchTool) -> None:
        self.tool = tool
        self.draw = None
        self.drag = None
        self.preview_uv = None
        if tool != SketchTool.SELECT:
            self.selected_entity_id = -1
            self.hover_handle = None

    def cancel(self) -> None:
        """Clear drawing/drag state without tool change."""
        self.draw = None
        self.drag = None
        self.preview_uv = None

    # --- snapping ---
    def snap(self, uv: Vec2, *, drawing: bool = False) -> SnapResult:
        u, v = float(uv[0]), float(uv[1])
        # Origin
        if abs(u) <= SNAP_POINT and abs(v) <= SNAP_POINT:
            return SnapResult((0.0, 0.0), "origin")

        # Existing points
        best: Optional[SnapResult] = None
        best_d = SNAP_POINT
        for p in self.sketch.snap_targets():
            d = float(np.hypot(u - p[0], v - p[1]))
            if d < best_d:
                best_d = d
                best = SnapResult((p[0], p[1]), "point", p)
        if best is not None:
            u, v = best.uv

        # Grid
        gu = round(u / SNAP_GRID) * SNAP_GRID
        gv = round(v / SNAP_GRID) * SNAP_GRID
        if abs(u - gu) <= SNAP_GRID * 0.35 and abs(v - gv) <= SNAP_GRID * 0.35:
            # Prefer point snap over grid if both
            if best is None:
                u, v = gu, gv
                best = SnapResult((u, v), "grid")
            else:
                # keep point but still allow H/V
                pass

        # H/V inference when drawing a line (from first point)
        kind = best.kind if best else "none"
        if drawing and self.draw and self.draw.points:
            p0 = self.draw.points[0]
            if abs(v - p0[1]) <= SNAP_HV:
                v = p0[1]
                kind = "h"
            elif abs(u - p0[0]) <= SNAP_HV:
                u = p0[0]
                kind = "v"

        res = SnapResult((u, v), kind, best.ref if best else None)
        self.last_snap = res
        return res

    # --- handles ---
    def pick_handle(self, uv: Vec2, tol: float = SNAP_POINT) -> Optional[Handle]:
        best: Optional[Handle] = None
        best_d = tol
        for h in self.sketch.all_handles():
            d = float(np.hypot(uv[0] - h.uv[0], uv[1] - h.uv[1]))
            if d < best_d:
                best_d = d
                best = h
        return best

    def pick_entity_body(self, uv: Vec2, tol: float = SNAP_POINT) -> Optional[int]:
        """Pick line/circle by proximity to geometry (not only handles)."""
        best_id = -1
        best_d = tol
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
        """Returns a short status message if any."""
        if self.tool == SketchTool.SELECT:
            h = self.pick_handle(raw_uv)
            if h is not None:
                self.selected_entity_id = h.entity_id
                ent = self.sketch.find_entity(h.entity_id)
                snap = ent  # snapshot not deep — store start uv
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
            self.draw = DrawState(tool=self.tool, points=[uv])
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
            self.sketch.add_line(pts[0], pts[1])
            self.draw = None
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
