"""Interactive chamfered view cube — pale labelled faces, HiDPI-safe hits."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import numpy as np
import pyvista as pv
from PySide6.QtCore import QEvent, QObject, Qt

from app.display_coords import in_normalized_viewport, qt_to_vtk_display
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
    # Top-right inset (normalized VTK viewport). Larger than the old ~19.5%
    # (0.80…0.995) so the orientation cube is easy to read and click.
    VIEWPORT = (0.74, 0.74, 0.995, 0.995)
    # Orthographic half-height in cube units (cube half≈1): smaller → fills inset more
    PARALLEL_SCALE = 1.62
    # Degrees per display pixel — low enough for fine control, high enough to feel direct
    ORBIT_DEG_PER_PX = 0.42
    # Click vs drag: once exceeded, every subsequent pixel moves the view
    DRAG_START_PX = 2.0

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
        self._actors: List = []
        self._label_actors: list = []
        self._labels: List[str] = []
        self._pick_actor = None
        self._pick_poly = None  # mesh used for picking (has region_id)
        self._press: Optional[Tuple[float, float]] = None
        self._last: Optional[Tuple[float, float]] = None
        self._dragged = False
        self._active = False

    def install(self) -> None:
        plotter = self.vp.plotter
        if plotter is None:
            return
        poly, labels = build_chamfered_cube(half=1.0, chamfer=0.28)
        self._labels = labels

        from vtkmodules.vtkRenderingCore import (
            vtkActor,
            vtkPolyDataMapper,
            vtkRenderer,
        )

        ren = vtkRenderer()
        ren.SetViewport(*self.VIEWPORT)
        ren.InteractiveOff()
        ren.SetBackground(0.93, 0.94, 0.96)
        try:
            ren.SetBackgroundAlpha(0.70)
        except Exception:
            pass
        try:
            ren.PreserveColorBufferOn()
        except Exception:
            pass

        # Solid pale fills: one actor per region (reliable on soft-GL; cell RGB
        # scalars often render black under llvmpipe). Geometry is welded so
        # shared face/edge/corner verts meet — no torn gaps or protruding tips.
        self._actors = []
        full = pv.wrap(poly)
        for i, lab in enumerate(labels):
            try:
                cell = full.extract_cells([i])
                # UnstructuredGrid → PolyData for the mapper (coords already correct)
                if not isinstance(cell, pv.PolyData):
                    try:
                        cell = cell.extract_surface(algorithm="dataset_surface")
                    except TypeError:
                        cell = cell.extract_surface()
            except Exception:
                continue
            mapper = vtkPolyDataMapper()
            mapper.SetInputData(cell)
            mapper.ScalarVisibilityOff()
            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.PickableOff()
            r, g, b = color_for_region(lab)
            prop = actor.GetProperty()
            prop.SetColor(float(r), float(g), float(b))
            prop.LightingOff()
            prop.SetAmbient(1.0)
            prop.SetDiffuse(0.0)
            prop.SetSpecular(0.0)
            # Subtle crease lines only — not thick double edges that look torn
            prop.SetEdgeVisibility(1)
            prop.SetEdgeColor(0.55, 0.58, 0.62)
            prop.SetLineWidth(0.8)
            prop.SetOpacity(1.0)
            try:
                prop.SetInterpolationToFlat()
            except Exception:
                pass
            ren.AddActor(actor)
            self._actors.append(actor)

        # Invisible pick mesh: full welded topology with region_id for hits
        pmap = vtkPolyDataMapper()
        pmap.SetInputData(poly)
        pmap.ScalarVisibilityOff()
        pactor = vtkActor()
        pactor.SetMapper(pmap)
        pactor.GetProperty().SetOpacity(0.01)
        pactor.GetProperty().LightingOff()
        pactor.GetProperty().SetColor(1.0, 1.0, 1.0)
        pactor.PickableOn()
        ren.AddActor(pactor)
        self._pick_actor = pactor
        self._pick_poly = poly

        # Face labels as real 3D vector text lying on each face (SW/Fusion style).
        self._label_actors = []  # list of (face_label, actor)
        for lab in labels:
            text = face_label_text(lab)
            if text is None:
                continue
            la = self._make_face_label_actor(lab, text)
            if la is None:
                continue
            ren.AddActor(la)
            self._label_actors.append((lab, la))

        ren.ResetCamera()
        cam = ren.GetActiveCamera()
        cam.SetParallelProjection(1)
        cam.SetParallelScale(self.PARALLEL_SCALE)
        cam.SetPosition(2.5, 2.1, 2.5)
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
        self._actors = []
        self._label_actors = []
        self._pick_actor = None
        self._pick_poly = None

    @staticmethod
    def _face_normal(lab: str) -> Optional[np.ndarray]:
        return {
            "face:+x": np.array([1.0, 0.0, 0.0]),
            "face:-x": np.array([-1.0, 0.0, 0.0]),
            "face:+y": np.array([0.0, 1.0, 0.0]),
            "face:-y": np.array([0.0, -1.0, 0.0]),
            "face:+z": np.array([0.0, 0.0, 1.0]),
            "face:-z": np.array([0.0, 0.0, -1.0]),
        }.get(lab)

    @staticmethod
    def _make_face_label_actor(lab: str, text: str):
        """Build a dark 3D vector-text actor centred on a cube face."""
        from vtkmodules.vtkCommonTransforms import vtkTransform
        from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter
        from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper
        from vtkmodules.vtkRenderingFreeType import vtkVectorText

        # Outward normal and in-face axes so text reads upright on each face
        # when the cube is viewed in the usual iso orientation.
        basis = {
            # right-hand: u across, v up-on-face, n outward
            "face:+x": (np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])),
            "face:-x": (np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]), np.array([-1.0, 0.0, 0.0])),
            "face:+y": (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0])),
            "face:-y": (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([0.0, -1.0, 0.0])),
            "face:+z": (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
            "face:-z": (np.array([-1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, -1.0])),
        }.get(lab)
        if basis is None:
            return None
        u, v, n = basis
        pos = face_label_position(lab, half=1.0)
        if pos is None:
            return None

        vt = vtkVectorText()
        vt.SetText(text)
        vt.Update()
        bounds = vt.GetOutput().GetBounds()
        # Centre the glyph at origin before placing on the face
        cx = 0.5 * (bounds[0] + bounds[1])
        cy = 0.5 * (bounds[2] + bounds[3])
        tw = max(1e-6, bounds[1] - bounds[0])
        th = max(1e-6, bounds[3] - bounds[2])
        # Fit inside face square (~1.4 half-extent after chamfer)
        target = 1.15
        scale = min(target / tw, target / th * 0.55)

        # p' = R * S * (p - center) + origin  where R columns = (u, v, n)
        origin = np.asarray(pos, float) + n * 0.04
        tf = vtkTransform()
        tf.PostMultiply()
        tf.Translate(-cx, -cy, 0.0)
        tf.Scale(scale, scale, scale)
        from vtkmodules.vtkCommonMath import vtkMatrix4x4

        m4 = vtkMatrix4x4()
        m4.Identity()
        for i in range(3):
            m4.SetElement(i, 0, float(u[i]))
            m4.SetElement(i, 1, float(v[i]))
            m4.SetElement(i, 2, float(n[i]))
            m4.SetElement(i, 3, float(origin[i]))
        tf.Concatenate(m4)

        tpd = vtkTransformPolyDataFilter()
        tpd.SetInputConnection(vt.GetOutputPort())
        tpd.SetTransform(tf)
        tpd.Update()

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(tpd.GetOutputPort())
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.PickableOff()
        prop = actor.GetProperty()
        prop.SetColor(0.12, 0.14, 0.18)
        prop.LightingOff()
        prop.SetAmbient(1.0)
        prop.SetDiffuse(0.0)
        prop.SetSpecular(0.0)
        return actor

    def sync_orientation(self) -> None:
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
            cam.SetPosition(*(direction * 3.3))
            cam.SetFocalPoint(0.0, 0.0, 0.0)
            cam.SetViewUp(*up)
            cam.SetParallelProjection(1)
            cam.SetParallelScale(self.PARALLEL_SCALE)
            self._renderer.ResetCameraClippingRange()
            # Only show labels on faces pointing toward the camera (no back-face soup)
            for lab, ta in self._label_actors:
                nrm = self._face_normal(lab)
                if nrm is None:
                    continue
                if float(np.dot(nrm, direction)) > 0.12:
                    ta.VisibilityOn()
                else:
                    ta.VisibilityOff()
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
        if self._renderer is None or self.vp.plotter is None:
            return None
        iw = self._interactor()
        if iw is None:
            return None
        try:
            from vtkmodules.vtkRenderingCore import vtkCellPicker

            vx, vy, _dw, _dh = qt_to_vtk_display(qx, qy, iw, self.vp.plotter)
            picker = vtkCellPicker()
            picker.SetTolerance(0.025)
            ok = picker.Pick(vx, vy, 0, self._renderer)
            if not ok:
                return None
            cid = int(picker.GetCellId())
            if cid < 0:
                return None
            # After triangulate, cell index ≠ region index — use region_id scalar
            poly = self._pick_poly
            if poly is not None and "region_id" in getattr(poly, "cell_data", {}):
                rids = np.asarray(poly.cell_data["region_id"]).reshape(-1)
                if cid < len(rids):
                    rid = int(rids[cid])
                    if 0 <= rid < len(self._labels):
                        return self._labels[rid]
            if cid < len(self._labels):
                return self._labels[cid]
            return None
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
            self._last = (x, y)
            self._dragged = False
            return True
        if et == QEvent.Type.MouseMove and self._active and self._press is not None:
            x, y = float(event.position().x()), float(event.position().y())
            # Start drag after a tiny threshold (click vs drag), then follow
            # *every* pixel — previous code required 3px per step, so small
            # careful moves did nothing and the orbit felt jumpy.
            if not self._dragged:
                dist = float(np.hypot(x - self._press[0], y - self._press[1]))
                if dist >= self.DRAG_START_PX:
                    self._dragged = True
                    self._last = self._press
                else:
                    return True
            last = self._last if self._last is not None else self._press
            dx = x - last[0]
            dy = y - last[1]
            self._last = (x, y)
            if abs(dx) < 1e-9 and abs(dy) < 1e-9:
                return True
            # orbit_camera already syncs the cube and requests an interactive paint
            self.on_orbit(
                float(dx) * self.ORBIT_DEG_PER_PX,
                float(-dy) * self.ORBIT_DEG_PER_PX,
            )
            return True
        if et == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
            if not self._active:
                return False
            x, y = float(event.position().x()), float(event.position().y())
            was_drag = self._dragged
            self._active = False
            self._press = None
            self._last = None
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
