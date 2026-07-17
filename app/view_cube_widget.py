"""Interactive chamfered view cube in a corner overlay renderer.

Behaviours (Fusion / SolidWorks style):
  * Click a face  → look straight down that axis (+/−)
  * Click a corner → isometric from that octant
  * Click an edge → half-iso between the two faces
  * Drag on the cube → orbit the main camera (same free look as scene drag)

The cube mesh is real geometry from ``build_chamfered_cube`` — not a ball gizmo.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import numpy as np
import pyvista as pv
from PySide6.QtCore import QEvent, QObject, Qt

from app.view_cube import build_chamfered_cube, color_for_region, region_to_view

if TYPE_CHECKING:
    from app.viewport import Viewport


def _hex_rgb01(h: str) -> Tuple[float, float, float]:
    h = h.lstrip("#")
    return int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0


class ViewCubeController(QObject):
    """Owns a corner renderer with a chamfered cube + mouse handling."""

    # Fraction of the render window (VTK y-up coords)
    VIEWPORT = (0.80, 0.80, 0.995, 0.995)

    def __init__(
        self,
        viewport: "Viewport",
        *,
        on_view: Callable[[str], None],
        on_orbit: Callable[[float, float], None],
    ) -> None:
        super().__init__(viewport)
        self.vp = viewport
        self.on_view = on_view
        self.on_orbit = on_orbit
        self._renderer = None
        self._actor = None
        self._labels: List[str] = []
        self._press: Optional[Tuple[float, float]] = None
        self._dragged = False
        self._active = False  # press started inside cube viewport
        self._filter_installed = False

    def install(self) -> None:
        plotter = self.vp.plotter
        if plotter is None:
            return
        poly, labels = build_chamfered_cube(half=1.0, chamfer=0.30)
        self._labels = labels
        # Per-cell RGB (uint8) for direct colouring on soft-GL
        colors = np.zeros((len(labels), 3), dtype=np.uint8)
        for i, lab in enumerate(labels):
            r, g, b = color_for_region(lab)
            colors[i] = (int(r * 255), int(g * 255), int(b * 255))
        poly.cell_data["RGB"] = colors

        from vtkmodules.vtkRenderingCore import vtkRenderer, vtkActor, vtkPolyDataMapper

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(poly)
        mapper.SetScalarModeToUseCellData()
        mapper.SelectColorArray("RGB")
        mapper.SetColorModeToDirectScalars()
        mapper.ScalarVisibilityOn()
        actor = vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetEdgeVisibility(1)
        prop.SetEdgeColor(0.15, 0.16, 0.18)
        prop.SetLineWidth(1.2)
        prop.SetAmbient(0.55)
        prop.SetDiffuse(0.55)
        prop.SetSpecular(0.20)
        prop.SetSpecularPower(20.0)
        prop.LightingOn()
        self._actor = actor

        ren = vtkRenderer()
        ren.SetViewport(*self.VIEWPORT)
        ren.InteractiveOff()
        # Slight panel behind cube so it reads as a floating control
        ren.SetBackground(0.88, 0.90, 0.93)
        try:
            ren.SetBackgroundAlpha(0.35)
        except Exception:
            pass
        try:
            ren.PreserveColorBufferOn()
        except Exception:
            pass
        ren.AddActor(actor)
        try:
            ren.AddActor2D  # keep API quiet
        except Exception:
            pass
        ren.ResetCamera()
        cam = ren.GetActiveCamera()
        cam.SetParallelProjection(1)
        cam.SetParallelScale(1.55)
        cam.SetPosition(2.4, 2.0, 2.4)
        cam.SetFocalPoint(0, 0, 0)
        cam.SetViewUp(0, 1, 0)

        rw = plotter.render_window
        try:
            n = int(rw.GetNumberOfLayers())
            rw.SetNumberOfLayers(max(n, 2))
            ren.SetLayer(1)
        except Exception:
            pass
        rw.AddRenderer(ren)
        self._renderer = ren

        # Event filter on interactor
        try:
            plotter.interactor.installEventFilter(self)
            self._filter_installed = True
        except Exception:
            pass
        self.sync_orientation()

    def remove(self) -> None:
        if self._renderer is not None and self.vp.plotter is not None:
            try:
                self.vp.plotter.render_window.RemoveRenderer(self._renderer)
            except Exception:
                pass
        self._renderer = None
        self._actor = None

    def sync_orientation(self) -> None:
        """Match cube orientation to the main camera (cube faces world axes)."""
        if self._renderer is None or self.vp.plotter is None:
            return
        try:
            main = self.vp.plotter.camera
            pos = np.asarray(main.GetPosition(), float)
            foc = np.asarray(main.GetFocalPoint(), float)
            up = np.asarray(main.GetViewUp(), float)
            direction = pos - foc
            n = np.linalg.norm(direction)
            if n < 1e-12:
                return
            direction = direction / n
            # Place cube camera along same direction from origin
            cam = self._renderer.GetActiveCamera()
            dist = 3.2
            cam.SetPosition(*(direction * dist))
            cam.SetFocalPoint(0.0, 0.0, 0.0)
            cam.SetViewUp(*up)
            cam.SetParallelProjection(1)
            cam.SetParallelScale(1.55)
            self._renderer.ResetCameraClippingRange()
        except Exception:
            pass

    def _in_cube_viewport(self, display_x: float, display_y: float) -> bool:
        if self.vp.plotter is None:
            return False
        try:
            w = float(self.vp.plotter.window_size[0])
            h = float(self.vp.plotter.window_size[1])
        except Exception:
            return False
        if w < 2 or h < 2:
            return False
        vx0, vy0, vx1, vy1 = self.VIEWPORT
        nx = float(display_x) / w
        ny = 1.0 - float(display_y) / h  # Qt y-down → VTK y-up
        return vx0 <= nx <= vx1 and vy0 <= ny <= vy1

    def _pick_region(self, display_x: float, display_y: float) -> Optional[str]:
        """Cell-pick the cube in its overlay renderer."""
        if self._renderer is None or self.vp.plotter is None or self._actor is None:
            return None
        try:
            from vtkmodules.vtkRenderingCore import vtkCellPicker

            w = float(self.vp.plotter.window_size[0])
            h = float(self.vp.plotter.window_size[1])
            # VTK display coords: origin bottom-left
            vx = float(display_x)
            vy = h - float(display_y)
            picker = vtkCellPicker()
            picker.SetTolerance(0.01)
            rw = self.vp.plotter.render_window
            ok = picker.Pick(vx, vy, 0, self._renderer)
            if not ok:
                return None
            cid = int(picker.GetCellId())
            if cid < 0 or cid >= len(self._labels):
                return None
            return self._labels[cid]
        except Exception:
            return None

    def eventFilter(self, obj, event):  # noqa: N802
        if self.vp.in_sketch_mode:
            return False
        et = event.type()
        if et == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            x, y = float(event.position().x()), float(event.position().y())
            if not self._in_cube_viewport(x, y):
                self._active = False
                return False
            self._active = True
            self._press = (x, y)
            self._dragged = False
            return True  # consume — do not orbit main scene under the cube
        if et == QEvent.Type.MouseMove and self._active and self._press is not None:
            x, y = float(event.position().x()), float(event.position().y())
            dx = x - self._press[0]
            dy = y - self._press[1]
            if abs(dx) + abs(dy) > 3.0:
                self._dragged = True
                # Orbit main camera (azimuth / elevation)
                # dx > 0 → rotate right; dy > 0 (down) → pitch up slightly
                self.on_orbit(float(dx) * 0.35, float(-dy) * 0.35)
                self._press = (x, y)
                self.sync_orientation()
                if self.vp.plotter:
                    self.vp._request_render()
            return True
        if et == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
            if not self._active:
                return False
            x, y = float(event.position().x()), float(event.position().y())
            was_drag = self._dragged
            self._active = False
            self._press = None
            self._dragged = False
            if was_drag:
                return True
            # Click without drag → pick region
            lab = self._pick_region(x, y)
            if lab is None:
                return True
            view = region_to_view(lab)
            if view is None:
                return True
            self.on_view(view)
            self.sync_orientation()
            return True
        return False


def apply_cube_view(viewport: "Viewport", token: str) -> str:
    """Apply a view token from region_to_view; returns standard view name used."""
    if token.startswith("iso:"):
        # iso from octant: e.g. +x+y+z
        signs = token.split(":", 1)[1]
        sx = 1.0 if "+x" in signs else (-1.0 if "-x" in signs else 1.0)
        sy = 1.0 if "+y" in signs else (-1.0 if "-y" in signs else 1.0)
        sz = 1.0 if "+z" in signs else (-1.0 if "-z" in signs else 1.0)
        viewport.set_view_from_direction((sx, sy, sz))
        return "iso"
    if token.startswith("edge:"):
        signs = token.split(":", 1)[1]
        sx = 1.0 if "+x" in signs else (-1.0 if "-x" in signs else 0.0)
        sy = 1.0 if "+y" in signs else (-1.0 if "-y" in signs else 0.0)
        sz = 1.0 if "+z" in signs else (-1.0 if "-z" in signs else 0.0)
        # Normalize diagonal in the plane of the two axes
        v = np.array([sx, sy, sz], float)
        n = np.linalg.norm(v)
        if n > 1e-9:
            v = v / n
        viewport.set_view_from_direction(tuple(v))
        return "edge"
    # standard face view
    viewport.set_view(token)
    return token
