"""Interactive chamfered view cube in a corner overlay renderer.

Behaviours (Fusion / SolidWorks style):
  * Click a face  → look straight down that axis (+/−)
  * Click a corner → isometric from that octant
  * Click an edge → half-iso between the two faces
  * Drag on the cube → orbit the main camera (same free look as scene drag)

Pale labelled faces (not a black blob). Hit-tests use Qt logical → VTK device
conversion so multi-monitor / HiDPI screens work.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QEvent, QObject, Qt

from app.display_coords import (
    in_normalized_viewport,
    qt_to_normalized,
    qt_to_vtk_display,
)
from app.view_cube import (
    build_chamfered_cube,
    color_for_region,
    face_label_position,
    face_label_text,
    region_to_view,
)

if TYPE_CHECKING:
    from app.viewport import Viewport


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
        self._label_actors: list = []
        self._labels: List[str] = []
        self._press: Optional[Tuple[float, float]] = None
        self._dragged = False
        self._active = False
        self._filter_installed = False

    def install(self) -> None:
        plotter = self.vp.plotter
        if plotter is None:
            return
        poly, labels = build_chamfered_cube(half=1.0, chamfer=0.30)
        self._labels = labels
        colors = np.zeros((len(labels), 3), dtype=np.uint8)
        for i, lab in enumerate(labels):
            r, g, b = color_for_region(lab)
            colors[i] = (int(r * 255), int(g * 255), int(b * 255))
        poly.cell_data["RGB"] = colors

        from vtkmodules.vtkRenderingCore import (
            vtkActor,
            vtkBillboardTextActor3D,
            vtkPolyDataMapper,
            vtkRenderer,
        )

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
        prop.SetEdgeColor(0.35, 0.38, 0.42)
        prop.SetLineWidth(1.0)
        # CRITICAL: lighting off — with lighting on soft-GL, direct RGB faces go black
        prop.LightingOff()
        prop.SetAmbient(1.0)
        prop.SetDiffuse(0.0)
        prop.SetSpecular(0.0)
        self._actor = actor

        ren = vtkRenderer()
        ren.SetViewport(*self.VIEWPORT)
        ren.InteractiveOff()
        ren.SetBackground(0.94, 0.95, 0.97)
        try:
            ren.SetBackgroundAlpha(0.55)
        except Exception:
            pass
        try:
            ren.PreserveColorBufferOn()
        except Exception:
            pass
        ren.AddActor(actor)

        # Face labels (SolidWorks/Fusion style readable text)
        self._label_actors = []
        for lab in labels:
            text = face_label_text(lab)
            pos = face_label_position(lab, half=1.0)
            if text is None or pos is None:
                continue
            ta = vtkBillboardTextActor3D()
            ta.SetInput(text)
            ta.SetPosition(*pos)
            tp = ta.GetTextProperty()
            tp.SetFontSize(16)
            tp.SetBold(1)
            tp.SetColor(0.15, 0.17, 0.20)
            tp.SetJustificationToCentered()
            tp.SetVerticalJustificationToCentered()
            try:
                tp.ShadowOff()
            except Exception:
                pass
            ren.AddActor(ta)
            self._label_actors.append(ta)

        ren.ResetCamera()
        cam = ren.GetActiveCamera()
        cam.SetParallelProjection(1)
        cam.SetParallelScale(1.65)
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
        self._label_actors = []

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
            cam = self._renderer.GetActiveCamera()
            dist = 3.2
            cam.SetPosition(*(direction * dist))
            cam.SetFocalPoint(0.0, 0.0, 0.0)
            cam.SetViewUp(*up)
            cam.SetParallelProjection(1)
            cam.SetParallelScale(1.65)
            self._renderer.ResetCameraClippingRange()
        except Exception:
            pass

    def _interactor(self):
        return self.vp.plotter.interactor if self.vp.plotter else None

    def _in_cube_viewport(self, qx: float, qy: float) -> bool:
        iw = self._interactor()
        if iw is None:
            return False
        return in_normalized_viewport(qx, qy, iw, self.VIEWPORT)

    def _pick_region(self, qx: float, qy: float) -> Optional[str]:
        """Cell-pick the cube using DPI-correct VTK display coords."""
        if self._renderer is None or self.vp.plotter is None or self._actor is None:
            return None
        iw = self._interactor()
        if iw is None:
            return None
        try:
            from vtkmodules.vtkRenderingCore import vtkCellPicker

            vx, vy, _dw, _dh = qt_to_vtk_display(qx, qy, iw, self.vp.plotter)
            picker = vtkCellPicker()
            picker.SetTolerance(0.02)
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
            return True
        if et == QEvent.Type.MouseMove and self._active and self._press is not None:
            x, y = float(event.position().x()), float(event.position().y())
            dx = x - self._press[0]
            dy = y - self._press[1]
            if abs(dx) + abs(dy) > 3.0:
                self._dragged = True
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
        v = np.array([sx, sy, sz], float)
        n = np.linalg.norm(v)
        if n > 1e-9:
            v = v / n
        viewport.set_view_from_direction(tuple(v))
        return "edge"
    viewport.set_view(token)
    return token
