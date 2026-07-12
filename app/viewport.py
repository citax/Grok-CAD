"""PyVista viewport: planes, sketches, async solids, software-GL friendly."""

from __future__ import annotations

import sys
import time
from typing import Dict, Optional, Set, Tuple

import numpy as np
import pyvista as pv
from PySide6.QtCore import QEvent, QObject, Qt, QThreadPool, QTimer, Signal, Slot
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget
from pyvistaqt import QtInteractor

from app.sketch_mode import SketchController, SketchTool
from app.theme import (
    ACCENT,
    TEXT_PRIMARY,
    AXIS_X,
    AXIS_Y,
    AXIS_Z,
    GRID_COLOR,
    HANDLE_COLOR,
    HANDLE_HOVER,
    PLANE_FRONT,
    PLANE_RIGHT,
    PLANE_TOP,
    SKETCH_COLOR,
    SKETCH_GRID,
    SKETCH_H,
    SKETCH_PREVIEW,
    SKETCH_V,
    SOLID_COLOR,
    SOLID_SELECTED,
    VP_BG_BOTTOM,
    VP_BG_TOP,
)
from app.workers import GeometryRebuildJob, snapshot_features
from cadcore.document import Document, FeatureType, is_reference_plane
from cadcore.sketch import (
    CircleEntity,
    LineEntity,
    RectEntity,
    Sketch,
    SketchEntity,
    Vec2,
    line_length,
    snapshot_entity,
)
from cadcore.units import format_length

PLANE_COLORS = {
    FeatureType.PLANE_FRONT: PLANE_FRONT,
    FeatureType.PLANE_TOP: PLANE_TOP,
    FeatureType.PLANE_RIGHT: PLANE_RIGHT,
}
PLANE_HALF = 2.5


def _plane_surface(ftype: FeatureType, half: float = PLANE_HALF) -> pv.PolyData:
    h = half
    if ftype is FeatureType.PLANE_FRONT:
        pts = np.array([[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0]], float)
    elif ftype is FeatureType.PLANE_TOP:
        pts = np.array([[-h, 0, -h], [h, 0, -h], [h, 0, h], [-h, 0, h]], float)
    else:
        pts = np.array([[0, -h, -h], [0, -h, h], [0, h, h], [0, h, -h]], float)
    return pv.PolyData(pts, np.array([4, 0, 1, 2, 3]))


def _plane_border(ftype: FeatureType, half: float = PLANE_HALF) -> pv.PolyData:
    h = half
    if ftype is FeatureType.PLANE_FRONT:
        ring = np.array([[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0]], float)
    elif ftype is FeatureType.PLANE_TOP:
        ring = np.array([[-h, 0, -h], [h, 0, -h], [h, 0, h], [-h, 0, h]], float)
    else:
        ring = np.array([[0, -h, -h], [0, -h, h], [0, h, h], [0, h, -h]], float)
    return pv.lines_from_points(ring, close=True)


def _mesh_to_polydata(vertices: np.ndarray, faces: np.ndarray) -> pv.PolyData:
    face_arr = np.hstack(
        [np.full((len(faces), 1), 3, dtype=np.int64), faces.astype(np.int64)]
    )
    return pv.PolyData(vertices, face_arr)


def _entity_polydata(ent: SketchEntity, sketch: Sketch) -> pv.PolyData:
    """Build sketch entity geometry EXACTLY on the sketch plane (no depth bias).

    In-sketch z-order over grid/axes is handled solely by the layer-1 overlay
    renderer — do not bake a normal offset into the points (that displaces the
    closed sketch after exit when drawn on the main renderer).
    """
    fr = sketch.frame
    if isinstance(ent, LineEntity):
        p0 = fr.to_world(ent.p0)
        p1 = fr.to_world(ent.p1)
        return pv.Line(p0, p1)
    if isinstance(ent, RectEntity):
        cs = ent.corners()
        pts = np.array([fr.to_world(c) for c in cs + [cs[0]]], float)
        return pv.lines_from_points(pts, close=False)
    if isinstance(ent, CircleEntity):
        # polyline circle in plane
        n = 48
        pts = []
        for i in range(n + 1):
            a = 2 * np.pi * i / n
            uv = (
                ent.center[0] + ent.radius * np.cos(a),
                ent.center[1] + ent.radius * np.sin(a),
            )
            pts.append(fr.to_world(uv))
        return pv.lines_from_points(np.array(pts, float), close=False)
    return pv.PolyData()


def _flat_point_cloud(points: np.ndarray) -> pv.PolyData:
    """Vertex-only PolyData for flat 2D point sprites (not spheres)."""
    return pv.PolyData(np.asarray(points, dtype=np.float64).reshape(-1, 3))


def _entity_fingerprint(ent: SketchEntity, *, selected: bool = False) -> str:
    """Geometry + style key for incremental actor upserts."""
    if isinstance(ent, LineEntity):
        g = f"L:{ent.p0[0]:.6g},{ent.p0[1]:.6g},{ent.p1[0]:.6g},{ent.p1[1]:.6g}"
    elif isinstance(ent, RectEntity):
        g = f"R:{ent.c0[0]:.6g},{ent.c0[1]:.6g},{ent.c1[0]:.6g},{ent.c1[1]:.6g}"
    elif isinstance(ent, CircleEntity):
        g = f"C:{ent.center[0]:.6g},{ent.center[1]:.6g},{ent.radius:.6g}"
    else:
        g = f"?{ent.id}"
    return f"{g}|sel={int(selected)}"


class _InteractorFilter(QObject):
    """Qt event filter for sketch mouse input on the VTK interactor."""

    def __init__(self, viewport: "Viewport") -> None:
        super().__init__(viewport)
        self.vp = viewport

    def eventFilter(self, obj, event):  # noqa: N802
        if not self.vp.in_sketch_mode:
            return False
        et = event.type()
        if et == QEvent.Type.MouseMove:
            self.vp._sketch_mouse_move(event.position().x(), event.position().y())
            return True  # consume — lock camera
        if et == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            self.vp._sketch_mouse_press(event.position().x(), event.position().y())
            return True
        if et == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
            self.vp._sketch_mouse_release(event.position().x(), event.position().y())
            return True
        if et == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Escape:
                self.vp.sketch_escape()
                return True
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self.vp.sketch_confirm()
                return True
        # Block middle/right orbit while sketching
        if et in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonDblClick):
            if event.button() != Qt.MouseButton.LeftButton:
                return True
        if et == QEvent.Type.Wheel:
            return True
        return False


class Viewport(QWidget):
    feature_picked = Signal(int)
    status_message = Signal(str)
    busy_changed = Signal(bool, str)
    sketch_exited = Signal()
    sketch_status = Signal(str)
    renderer_info = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._doc: Optional[Document] = None
        self._selected_id = -1
        self._ok = False
        self.gl_renderer = ""
        self._planes_built = False
        self._solid_fps: Dict[int, str] = {}
        self._job_gen = 0
        self._pool = QThreadPool.globalInstance()
        self._pool.setMaxThreadCount(max(1, min(2, self._pool.maxThreadCount())))

        self._sketch_feature_id = -1
        self._sketch_ctrl: Optional[SketchController] = None
        self._sketch_entity_actors: Set[int] = set()
        # Incremental actor caches: name → fingerprint (skip unchanged)
        self._sketch_entity_fps: Dict[int, str] = {}
        self._closed_sketch_fps: Dict[str, str] = {}
        self._sk_overlay = None  # optional 2nd-layer VTK renderer for sketch strokes
        self._overlay_actors: Dict[str, object] = {}  # name → vtkActor on overlay layer
        self._filter: Optional[_InteractorFilter] = None

        self._rebuild_timer = QTimer(self)
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.setInterval(40)
        self._rebuild_timer.timeout.connect(self._start_rebuild_job)

        # Render-on-demand coalesce (throttle full VTK renders)
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(16)
        self._render_timer.timeout.connect(self._do_render)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        try:
            self.plotter = QtInteractor(self, multi_samples=0)
            layout.addWidget(self.plotter.interactor)
            self._ok = True
        except Exception as exc:  # noqa: BLE001
            print(f"[viewport] FAILED: {exc}", file=sys.stderr)
            err = QLabel(f"3D viewport failed:\n{exc}")
            err.setStyleSheet(f"color: {SKETCH_PREVIEW}; background: {VP_BG_BOTTOM}; padding: 24px;")
            err.setWordWrap(True)
            layout.addWidget(err)
            self.plotter = None  # type: ignore
            return

        # Graphite vertical gradient (top lighter → bottom darker)
        try:
            self.plotter.set_background(VP_BG_BOTTOM, top=VP_BG_TOP)
        except TypeError:
            self.plotter.set_background(VP_BG_BOTTOM)
        self._configure_softgl_render()
        self._log_renderer()
        self._setup_helpers()
        self._setup_picking()
        self._setup_interaction_lod()
        self.set_view("iso")

    @property
    def in_sketch_mode(self) -> bool:
        return self._sketch_ctrl is not None

    # ----- setup -----
    def _configure_softgl_render(self) -> None:
        assert self.plotter is not None
        rw = self.plotter.render_window
        try:
            rw.SetMultiSamples(0)
            rw.SetDesiredUpdateRate(30.0)
            rw.SetStillUpdateRate(0.0001)
        except Exception:
            pass
        ren = self.plotter.renderer
        try:
            ren.UseDepthPeelingOff()
            ren.SetMaximumNumberOfPeels(0)
        except Exception:
            pass

    def _log_renderer(self) -> None:
        try:
            rw = self.plotter.render_window
            print(f"[viewport] render_window = {type(rw).__name__}", file=sys.stderr)
            from vtkmodules.vtkRenderingOpenGL2 import vtkOpenGLRenderWindow

            if isinstance(rw, vtkOpenGLRenderWindow):
                caps = rw.ReportCapabilities() or ""
                for line in caps.splitlines():
                    low = line.lower()
                    if any(k in low for k in ("renderer", "version", "vendor")):
                        print(f"[viewport] {line.strip()}", file=sys.stderr)
                    if "opengl renderer string" in low or (
                        "renderer string" in low and "opengl" in low
                    ):
                        # "OpenGL renderer string:  llvmpipe (...)"
                        parts = line.split(":", 1)
                        if len(parts) == 2:
                            self.gl_renderer = parts[1].strip()
                            self.renderer_info.emit(self.gl_renderer)
        except Exception as exc:  # noqa: BLE001
            print(f"[viewport] log: {exc}", file=sys.stderr)

    def _setup_helpers(self) -> None:
        assert self.plotter is not None
        self.plotter.show_bounds(
            grid="back", location="outer", color=GRID_COLOR, font_size=9,
            xtitle="X", ytitle="Y", ztitle="Z",
        )
        # Flat 2D origin dot (not a 3D sphere)
        self.plotter.add_mesh(
            _flat_point_cloud(np.array([[0.0, 0.0, 0.0]])),
            color=TEXT_PRIMARY,
            point_size=12,
            render_points_as_spheres=False,
            name="__origin",
            pickable=False,
        )
        for end, color, name in (
            ((2.4, 0, 0), AXIS_X, "__ax"),
            ((0, 2.4, 0), AXIS_Y, "__ay"),
            ((0, 0, 2.4), AXIS_Z, "__az"),
        ):
            self.plotter.add_mesh(
                pv.Line((0, 0, 0), end), color=color, line_width=3, name=name, pickable=False
            )
        try:
            self.plotter.add_axes(
                line_width=2, xlabel="X", ylabel="Y", zlabel="Z",
                x_color=AXIS_X, y_color=AXIS_Y, z_color=AXIS_Z,
                viewport=(0.0, 0.0, 0.18, 0.18),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[viewport] add_axes: {exc}", file=sys.stderr)

    def _setup_picking(self) -> None:
        assert self.plotter is not None
        try:
            self.plotter.enable_mesh_picking(
                callback=self._on_mesh_pick, left_clicking=True, show=False, show_message=False
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[viewport] picking: {exc}", file=sys.stderr)

    def _setup_interaction_lod(self) -> None:
        assert self.plotter is not None
        try:
            iren = self.plotter.iren.interactor
        except Exception:
            return

        def on_start(_o=None, _e=None) -> None:
            if self.in_sketch_mode:
                return
            try:
                self.plotter.render_window.SetDesiredUpdateRate(45.0)
            except Exception:
                pass

        def on_end(_o=None, _e=None) -> None:
            try:
                self.plotter.render_window.SetDesiredUpdateRate(0.0001)
            except Exception:
                pass
            self._request_render()

        for ev in ("StartInteractionEvent", "LeftButtonPressEvent"):
            try:
                iren.AddObserver(ev, on_start)
            except Exception:
                pass
        for ev in ("EndInteractionEvent", "LeftButtonReleaseEvent"):
            try:
                iren.AddObserver(ev, on_end)
            except Exception:
                pass

    # ----- document / rebuild -----
    def set_document(self, doc: Document) -> None:
        self._doc = doc
        self._planes_built = False
        self._solid_fps.clear()
        self.schedule_rebuild(immediate_planes=True)
        self.refresh_sketches()

    def set_selected_id(self, fid: int) -> None:
        if self._selected_id == fid:
            return
        self._selected_id = fid
        self._restyle_selection_only()
        self._request_render()

    def schedule_rebuild(self, *, immediate_planes: bool = False) -> None:
        if not self._ok or self.plotter is None or self._doc is None:
            return
        if immediate_planes or not self._planes_built:
            self._ensure_planes()
        self._rebuild_timer.start()

    def rebuild(self) -> None:
        self.schedule_rebuild()

    def _start_rebuild_job(self) -> None:
        if not self._ok or self._doc is None:
            return
        self._job_gen += 1
        gen = self._job_gen
        job = GeometryRebuildJob(gen, snapshot_features(self._doc))
        job.signals.started.connect(self._on_job_started)
        job.signals.finished.connect(self._on_job_finished)
        job.signals.failed.connect(self._on_job_failed)
        self.busy_changed.emit(True, "Computing geometry…")
        self._pool.start(job)

    @Slot(int)
    def _on_job_started(self, gen: int) -> None:
        if gen == self._job_gen:
            self.busy_changed.emit(True, "Computing geometry…")

    @Slot(int, object)
    def _on_job_finished(self, gen: int, results: object) -> None:
        if gen != self._job_gen:
            return
        assert isinstance(results, dict)
        t0 = time.perf_counter()
        self._apply_solid_results(results)
        dt = (time.perf_counter() - t0) * 1000.0
        self.busy_changed.emit(False, "")
        self.status_message.emit(f"Geometry ready ({dt:.0f} ms upload)")

    @Slot(int, str)
    def _on_job_failed(self, gen: int, message: str) -> None:
        if gen != self._job_gen:
            return
        self.busy_changed.emit(False, "")
        self.status_message.emit(f"Geometry failed: {message}")

    def _apply_solid_results(self, results: Dict[int, Tuple[np.ndarray, np.ndarray, str]]) -> None:
        if not self.plotter:
            return
        wanted = set(results.keys())
        for fid in set(self._solid_fps) - wanted:
            self._remove_actor(f"solid_{fid}")
            self._solid_fps.pop(fid, None)
        for fid, (verts, faces, fp) in results.items():
            if self._solid_fps.get(fid) == fp:
                continue
            self._remove_actor(f"solid_{fid}")
            pdata = _mesh_to_polydata(verts, faces)
            try:
                if len(faces) > 8000:
                    pdata = pdata.decimate_pro(1.0 - 4000.0 / len(faces), preserve_topology=True)
            except Exception:
                pass
            self.plotter.add_mesh(
                pdata, color=SOLID_COLOR, name=f"solid_{fid}", pickable=True,
                smooth_shading=False, show_edges=False, render=False,
            )
            self._solid_fps[fid] = fp
        self._restyle_selection_only()
        self._request_render()

    def _ensure_planes(self) -> None:
        if not self.plotter or self._doc is None:
            return
        if self._planes_built:
            for f in self._doc.features:
                if not is_reference_plane(f.type):
                    continue
                for prefix in ("plane_", "edge_"):
                    name = f"{prefix}{f.id}"
                    if name in self.plotter.actors:
                        try:
                            self.plotter.actors[name].SetVisibility(1 if f.visible else 0)
                        except Exception:
                            pass
            return
        for f in self._doc.features:
            if not is_reference_plane(f.type):
                continue
            color = PLANE_COLORS[f.type]
            self.plotter.add_mesh(
                _plane_surface(f.type), color=color, opacity=0.40, name=f"plane_{f.id}",
                pickable=True, smooth_shading=False, show_edges=False, render=False,
            )
            self.plotter.add_mesh(
                _plane_border(f.type), color=color, line_width=3, name=f"edge_{f.id}",
                pickable=False, render=False,
            )
        self._planes_built = True
        self._restyle_selection_only()
        self._request_render()

    def _request_render(self) -> None:
        if self._ok and not self._render_timer.isActive():
            self._render_timer.start()

    def _do_render(self) -> None:
        if self.plotter:
            self.plotter.render()

    def _remove_actor(self, name: str) -> None:
        if not self.plotter:
            return
        # Detach from sketch overlay first if present
        if name in self._overlay_actors:
            try:
                if self._sk_overlay is not None:
                    self._sk_overlay.RemoveActor(self._overlay_actors[name])
            except Exception:
                pass
            self._overlay_actors.pop(name, None)
        try:
            self.plotter.remove_actor(name, render=False)
        except Exception:
            pass

    def _restyle_selection_only(self) -> None:
        if not self.plotter:
            return
        for name, actor in list(self.plotter.actors.items()):
            if not (name.startswith("plane_") or name.startswith("solid_")):
                continue
            try:
                fid = int(name.split("_", 1)[1])
            except ValueError:
                continue
            prop = actor.GetProperty()
            selected = fid == self._selected_id and not self.in_sketch_mode
            if name.startswith("plane_"):
                # Dim planes in sketch mode
                base = 0.12 if self.in_sketch_mode else 0.40
                prop.SetOpacity(0.52 if selected else base)
                prop.SetEdgeVisibility(1 if selected else 0)
                if selected:
                    prop.SetEdgeColor(1.0, 0.92, 0.2)
            else:
                if self.in_sketch_mode:
                    prop.SetOpacity(0.25)
                else:
                    prop.SetOpacity(1.0)
                prop.SetColor(*( (0.98, 0.75, 0.14) if selected else (0.60, 0.64, 0.68) ))

    # ----- sketch mode -----
    def _set_parallel_projection(self, enabled: bool) -> None:
        """Toggle orthographic (parallel) camera — true 2D sketch look."""
        if not self.plotter:
            return
        try:
            cam = self.plotter.camera
            cam.SetParallelProjection(1 if enabled else 0)
            if enabled:
                # Scale so the sketch plane fills a comfortable orthographic window
                cam.SetParallelScale(3.2)
        except Exception:
            try:
                self.plotter.enable_parallel_projection() if enabled else self.plotter.disable_parallel_projection()
            except Exception:
                pass

    def _ensure_sketch_overlay_layer(self) -> None:
        """Layer-1 VTK renderer so sketch strokes always composite above scene."""
        if self._sk_overlay is not None or not self.plotter:
            return
        try:
            from vtkmodules.vtkRenderingCore import vtkRenderer

            main = self.plotter.renderer
            rw = self.plotter.render_window
            ov = vtkRenderer()
            ov.SetLayer(1)
            ov.InteractiveOff()
            ov.SetActiveCamera(main.GetActiveCamera())
            # Transparent — only draw props we add (entities / preview / handles)
            ov.SetBackground(0, 0, 0)
            try:
                ov.SetBackgroundAlpha(0.0)
            except Exception:
                pass
            # Draw on top of layer 0 without wiping its color
            try:
                ov.PreserveColorBufferOn()
            except Exception:
                pass
            try:
                # Fresh depth so layer-1 lines aren't occluded by layer-0 planes
                ov.PreserveDepthBufferOff()
            except Exception:
                pass
            nlayers = 1
            try:
                nlayers = int(rw.GetNumberOfLayers())
            except Exception:
                pass
            rw.SetNumberOfLayers(max(2, nlayers))
            rw.AddRenderer(ov)
            self._sk_overlay = ov
        except Exception as exc:  # noqa: BLE001
            print(f"[viewport] sketch overlay layer: {exc}", file=sys.stderr)
            self._sk_overlay = None

    def _teardown_sketch_overlay_layer(self) -> None:
        if self._sk_overlay is None or not self.plotter:
            self._overlay_actors.clear()
            return
        try:
            rw = self.plotter.render_window
            for name, act in list(self._overlay_actors.items()):
                try:
                    self._sk_overlay.RemoveActor(act)
                except Exception:
                    pass
            self._overlay_actors.clear()
            rw.RemoveRenderer(self._sk_overlay)
        except Exception:
            pass
        self._sk_overlay = None

    def _get_named_actor(self, name: str):
        """Lookup actor on main plotter or sketch overlay layer."""
        if self.plotter and name in getattr(self.plotter, "actors", {}):
            return self.plotter.actors[name]
        return self._overlay_actors.get(name)

    def _add_overlay_mesh(self, pdata, **kwargs):
        """add_mesh into the main plotter, then reparent actor onto sketch overlay."""
        assert self.plotter is not None
        name = kwargs.get("name", "")
        self._ensure_sketch_overlay_layer()
        # Remove prior overlay actor with same name
        if name and name in self._overlay_actors:
            try:
                self._sk_overlay.RemoveActor(self._overlay_actors.pop(name))
            except Exception:
                self._overlay_actors.pop(name, None)
        # Add via plotter (creates vtkActor), then move to layer-1 overlay
        self.plotter.add_mesh(pdata, **kwargs)
        if self._sk_overlay is None or not name:
            return
        try:
            actor = self.plotter.actors.get(name)
            if actor is None:
                return
            main = self.plotter.renderer
            try:
                main.RemoveActor(actor)
            except Exception:
                try:
                    main.RemoveViewProp(actor)
                except Exception:
                    pass
            self._sk_overlay.AddActor(actor)
            # PyVista's actors dict only tracks the main renderer — keep our own map
            self._overlay_actors[name] = actor
        except Exception as exc:  # noqa: BLE001
            print(f"[viewport] overlay reparent {name}: {exc}", file=sys.stderr)

    def enter_sketch(self, sketch_feature_id: int) -> None:
        if not self._ok or self._doc is None or self.plotter is None:
            return
        f = self._doc.find(sketch_feature_id)
        if f is None or f.sketch is None:
            return
        self._sketch_feature_id = sketch_feature_id
        self._sketch_ctrl = SketchController(f.sketch)
        self._sketch_ctrl.set_tool(SketchTool.LINE)
        self.sketch_status.emit("Sketch: Line")

        # Orient camera normal to plane + orthographic projection (flat 2D sketch)
        fr = f.sketch.frame
        dist = 10.0
        pos = fr.origin + fr.normal * dist
        self.plotter.camera_position = [
            tuple(pos),
            tuple(fr.origin),
            tuple(fr.v_axis),
        ]
        self._set_parallel_projection(True)
        self._ensure_sketch_overlay_layer()

        self._install_sketch_filter()
        try:
            self.plotter.interactor.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self.plotter.interactor.setFocus()
        except Exception:
            pass
        self._dim_label_names: Set[str] = set()
        self._drag_before: Optional[dict] = None
        self._draw_sketch_overlay()
        self._rebuild_all_sketch_entities()
        self._update_junction_dots()
        self._update_dim_labels()
        self._restyle_selection_only()
        self._request_render()
        plane = self._doc.find(f.plane_id)
        pname = plane.name if plane else "Plane"
        self.sketch_status.emit(f"Editing {f.name} on {pname}")

    def exit_sketch(self) -> None:
        if not self.in_sketch_mode:
            return
        self._remove_sketch_filter()
        self._clear_sketch_overlays()
        self._teardown_sketch_overlay_layer()
        self._sketch_ctrl = None
        self._sketch_feature_id = -1
        # Restore perspective for 3D solid viewing
        self._set_parallel_projection(False)
        self._restyle_selection_only()
        self.refresh_sketches()
        self._request_render()
        self.sketch_exited.emit()

    def set_sketch_tool(self, tool: SketchTool) -> None:
        if self._sketch_ctrl is None:
            return
        self._sketch_ctrl.set_tool(tool)
        self._clear_preview()
        self.sketch_status.emit(f"Sketch: {tool.name.title()}")
        self._update_cursor()

    def sketch_escape(self) -> None:
        """Esc: cancel in-progress draw → Select; if idle, exit sketch mode."""
        if self._sketch_ctrl is None:
            return
        if self._sketch_ctrl.is_drawing() or self._sketch_ctrl.drag is not None:
            self._sketch_ctrl.cancel_drawing()
            self._sketch_ctrl.set_tool(SketchTool.SELECT)
            self._clear_preview()
            self._update_handles_visual()
            self._update_cursor()
            self.sketch_status.emit("Sketch: Select")
            self._request_render()
            return
        # Idle select — exit sketch entirely
        self.exit_sketch()
        self.sketch_status.emit("Exited sketch")

    def sketch_confirm(self) -> None:
        """Enter: finish current entity using rubber-band end point."""
        if self._sketch_ctrl is None:
            return
        msg = self._sketch_ctrl.confirm_current()
        if msg:
            ent = self._sketch_ctrl.sketch.entities[-1]
            if self._doc is not None and self._sketch_feature_id >= 0:
                self._doc.record_entity_add(self._sketch_feature_id, ent)
            self._upsert_entity_actor(ent)
            self._clear_preview()
            self._update_handles_visual()
            self._update_junction_dots()
            self._update_dim_labels()
            self.sketch_status.emit(f"Sketch: {msg}")
            self._request_render()

    def sketch_cancel(self) -> None:
        """Back-compat alias for Esc first stage only (cancel draw)."""
        if self._sketch_ctrl is None:
            return
        if self._sketch_ctrl.is_drawing() or self._sketch_ctrl.drag is not None:
            self.sketch_escape()
        else:
            self.sketch_escape()

    def _install_sketch_filter(self) -> None:
        if not self.plotter:
            return
        self._filter = _InteractorFilter(self)
        self.plotter.interactor.installEventFilter(self._filter)
        # Disable VTK style while sketching
        try:
            self.plotter.iren.interactor.GetInteractorStyle().SetEnabled(0)
        except Exception:
            pass

    def _remove_sketch_filter(self) -> None:
        if self.plotter and self._filter:
            self.plotter.interactor.removeEventFilter(self._filter)
            self._filter = None
        try:
            self.plotter.iren.interactor.GetInteractorStyle().SetEnabled(1)
        except Exception:
            pass
        self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

    def _draw_sketch_overlay(self) -> None:
        """Sketch grid + local H/V axes on the plane."""
        if not self.plotter or self._sketch_ctrl is None:
            return
        fr = self._sketch_ctrl.sketch.frame
        # Grid lines in UV
        half = 3.0
        step = 0.5
        pts_u = []
        pts_v = []
        g = np.arange(-half, half + 1e-9, step)
        for t in g:
            # lines of constant u
            a = fr.to_world((t, -half))
            b = fr.to_world((t, half))
            pts_u.extend([a, b])
            # constant v
            c = fr.to_world((-half, t))
            d = fr.to_world((half, t))
            pts_v.extend([c, d])
        # Build multiline via segments — exactly on the sketch plane (no depth bias).
        # In-sketch z-order over grid/axes is handled solely by the layer-1 overlay.
        def segs(pairs):
            lines = []
            for i in range(0, len(pairs), 2):
                lines.append(pv.Line(pairs[i], pairs[i + 1]))
            if not lines:
                return pv.PolyData()
            return lines[0].merge(lines[1:]) if len(lines) > 1 else lines[0]

        grid = segs(pts_u).merge(segs(pts_v)) if pts_u else pv.PolyData()
        self._remove_actor("__sk_grid")
        self.plotter.add_mesh(
            grid, color=SKETCH_GRID, line_width=1, name="__sk_grid", pickable=False, render=False
        )
        # H axis (u) and V axis (v) through origin — coplanar with user strokes
        self._remove_actor("__sk_h")
        self._remove_actor("__sk_v")
        self.plotter.add_mesh(
            pv.Line(fr.to_world((-half, 0)), fr.to_world((half, 0))),
            color=SKETCH_H, line_width=2, name="__sk_h", pickable=False, render=False,
        )
        self.plotter.add_mesh(
            pv.Line(fr.to_world((0, -half)), fr.to_world((0, half))),
            color=SKETCH_V, line_width=2, name="__sk_v", pickable=False, render=False,
        )

    def _clear_sketch_overlays(self) -> None:
        for name in list(self._sketch_entity_actors):
            self._remove_actor(f"sk_e_{name}")
        self._sketch_entity_actors.clear()
        self._sketch_entity_fps.clear()
        for n in (
            "__sk_grid",
            "__sk_h",
            "__sk_v",
            "__sk_preview",
            "__sk_handles",
            "__sk_infer",
            "__sk_junctions",
            "__sk_dims",
        ):
            self._remove_actor(n)
        # Drop per-line dim labels if any
        for name in list(getattr(self, "_dim_label_names", set()) or set()):
            self._remove_actor(name)
        self._dim_label_names = set()
        # Drop any leftover overlay props
        for n in list(self._overlay_actors.keys()):
            self._remove_actor(n)

    def _apply_sketch_actor_priority(self, name: str) -> None:
        """Ensure sketch entity lines draw above grid/axes in the real viewport."""
        if not self.plotter:
            return
        actor = self._get_named_actor(name)
        if actor is None:
            return
        try:
            mapper = actor.GetMapper()
            if mapper is not None:
                mapper.SetResolveCoincidentTopologyToPolygonOffset()
                try:
                    mapper.SetRelativeCoincidentTopologyLineOffsetParameters(-200000, -200)
                except Exception:
                    try:
                        mapper.SetRelativeCoincidentTopologyLineOffsetParameters(-4, -4)
                    except Exception:
                        pass
            prop = actor.GetProperty()
            prop.SetLineWidth(max(prop.GetLineWidth(), 2.8))
            prop.SetLighting(False)
            try:
                actor.ForceOpaqueOn()
            except Exception:
                pass
        except Exception:
            pass

    def refresh_sketches(self) -> None:
        """Incremental redraw of closed sketches in 3D (when not editing)."""
        if not self.plotter or self._doc is None:
            return
        if self.in_sketch_mode:
            return
        wanted: Dict[str, Tuple[object, str]] = {}  # name -> (ent, fp)  # type: ignore
        for f in self._doc.features:
            if f.type is not FeatureType.SKETCH or f.sketch is None or not f.visible:
                continue
            for ent in f.sketch.entities:
                name = f"sk_closed_{f.id}_{ent.id}"
                fp = _entity_fingerprint(ent, selected=False)
                wanted[name] = (ent, fp)
        # Remove stale
        for name in list(self._closed_sketch_fps.keys()):
            if name not in wanted:
                self._remove_actor(name)
                self._closed_sketch_fps.pop(name, None)
        # Upsert changed only
        dirty = False
        for name, (ent, fp) in wanted.items():
            if self._closed_sketch_fps.get(name) == fp and name in self.plotter.actors:
                continue
            self._remove_actor(name)
            # find parent sketch
            sk = None
            for f in self._doc.features:
                if f.type is FeatureType.SKETCH and f.sketch is not None:
                    if any(e.id == ent.id for e in f.sketch.entities):
                        sk = f.sketch
                        break
            if sk is None:
                continue
            pdata = _entity_polydata(ent, sk)
            self.plotter.add_mesh(
                pdata,
                color=SKETCH_COLOR,
                line_width=2.5,
                name=name,
                pickable=False,
                render=False,
            )
            self._apply_sketch_actor_priority(name)
            self._closed_sketch_fps[name] = fp
            dirty = True
        if dirty:
            self._request_render()

    def _rebuild_all_sketch_entities(self) -> None:
        """Incremental: upsert changed entities, drop removed ones."""
        if self._sketch_ctrl is None:
            return
        present = {e.id: e for e in self._sketch_ctrl.sketch.entities}
        for eid in list(self._sketch_entity_actors):
            if eid not in present:
                self._remove_actor(f"sk_e_{eid}")
                self._sketch_entity_actors.discard(eid)
                self._sketch_entity_fps.pop(eid, None)
        for ent in self._sketch_ctrl.sketch.entities:
            self._upsert_entity_actor(ent)
        self._update_handles_visual()
        self._update_junction_dots()
        self._update_dim_labels()

    def sync_sketch_visuals(self) -> None:
        """Full resync after undo/redo/paste/unit change (clears ghost actors).

        Drops every sketch-entity actor from plotter.actors AND overlay_actors,
        clears the fingerprint cache, then rebuilds from the document sketch.
        """
        if not self.plotter or self._sketch_ctrl is None:
            return
        # Purge entity actors by name from both maps
        for eid in list(self._sketch_entity_actors):
            self._remove_actor(f"sk_e_{eid}")
        self._sketch_entity_actors.clear()
        self._sketch_entity_fps.clear()
        # Also sweep any leftover sk_e_* in overlay/plotter
        for name in list(self._overlay_actors.keys()):
            if name.startswith("sk_e_"):
                self._remove_actor(name)
        if self.plotter:
            for name in list(self.plotter.actors.keys()):
                if name.startswith("sk_e_"):
                    self._remove_actor(name)
        for ent in self._sketch_ctrl.sketch.entities:
            self._upsert_entity_actor(ent)
        self._update_handles_visual()
        self._update_junction_dots()
        self._update_dim_labels()
        self._request_render()

    def refresh_dim_labels(self) -> None:
        """Recompute length labels (e.g. after unit change)."""
        self._update_dim_labels()
        self._request_render()

    def _update_junction_dots(self) -> None:
        """Flat on-plane dots at unique entity endpoints / connection points."""
        if not self.plotter or self._sketch_ctrl is None:
            return
        self._remove_actor("__sk_junctions")
        pts_uv = self._sketch_ctrl.sketch.unique_endpoints()
        if not pts_uv:
            return
        fr = self._sketch_ctrl.sketch.frame
        pts = np.array([fr.to_world(p) for p in pts_uv], float)
        self._add_overlay_mesh(
            _flat_point_cloud(pts),
            color=SKETCH_GRID,
            point_size=7,
            render_points_as_spheres=False,
            name="__sk_junctions",
            pickable=False,
            render=False,
            opacity=0.55,
        )

    def _update_dim_labels(self) -> None:
        """Line-length labels at midpoints in the current display unit."""
        if not self.plotter or self._sketch_ctrl is None or self._doc is None:
            return
        # Clear previous labels
        for name in list(getattr(self, "_dim_label_names", set()) or set()):
            self._remove_actor(name)
        self._dim_label_names = set()
        self._remove_actor("__sk_dims")
        unit = self._doc.display_unit
        fr = self._sketch_ctrl.sketch.frame
        points = []
        labels = []
        for ent in self._sketch_ctrl.sketch.entities:
            if not isinstance(ent, LineEntity):
                continue
            mid = ent.midpoint()
            w = fr.to_world(mid)
            points.append(w)
            labels.append(format_length(line_length(ent), unit))
        if not points:
            return
        try:
            # Single multi-label actor on the main plotter (visible with sketch camera)
            actor = self.plotter.add_point_labels(
                np.array(points, float),
                labels,
                name="__sk_dims",
                font_size=14,
                text_color=TEXT_PRIMARY,
                point_size=0,
                shape=None,
                always_visible=True,
                pickable=False,
                render=False,
                show_points=False,
            )
            self._dim_label_names.add("__sk_dims")
            _ = actor
        except Exception as exc:  # noqa: BLE001
            # Fallback: one label actor per line
            print(f"[viewport] dim labels: {exc}", file=sys.stderr)
            for i, (pt, lab) in enumerate(zip(points, labels)):
                name = f"__sk_dim_{i}"
                try:
                    self.plotter.add_point_labels(
                        np.array([pt], float),
                        [lab],
                        name=name,
                        font_size=14,
                        text_color=TEXT_PRIMARY,
                        point_size=0,
                        shape=None,
                        always_visible=True,
                        pickable=False,
                        render=False,
                        show_points=False,
                    )
                    self._dim_label_names.add(name)
                except Exception:
                    pass

    def dim_label_texts(self) -> list:
        """Return current line-length label strings (for tests / verification)."""
        if self._sketch_ctrl is None or self._doc is None:
            return []
        unit = self._doc.display_unit
        out = []
        for ent in self._sketch_ctrl.sketch.entities:
            if isinstance(ent, LineEntity):
                out.append(format_length(line_length(ent), unit))
        return out

    def _upsert_entity_actor(self, ent: SketchEntity) -> None:
        if not self.plotter or self._sketch_ctrl is None:
            return
        name = f"sk_e_{ent.id}"
        selected = ent.id == self._sketch_ctrl.selected_entity_id
        fp = _entity_fingerprint(ent, selected=selected)
        if self._sketch_entity_fps.get(ent.id) == fp and (
            name in self.plotter.actors or name in self._overlay_actors
        ):
            return  # unchanged — skip VTK teardown/rebuild
        self._remove_actor(name)
        pdata = _entity_polydata(ent, self._sketch_ctrl.sketch)
        col = SKETCH_PREVIEW if selected else SKETCH_COLOR
        self._add_overlay_mesh(
            pdata, color=col, line_width=3.0, name=name, pickable=False, render=False
        )
        self._apply_sketch_actor_priority(name)
        self._sketch_entity_actors.add(ent.id)
        self._sketch_entity_fps[ent.id] = fp

    def _clear_preview(self) -> None:
        self._remove_actor("__sk_preview")
        self._remove_actor("__sk_infer")

    def _clear_handles(self) -> None:
        self._remove_actor("__sk_handles")

    def _update_handles_visual(self) -> None:
        if not self.plotter or self._sketch_ctrl is None:
            return
        self._clear_handles()
        ctrl = self._sketch_ctrl
        fr = ctrl.sketch.frame
        pts = []
        for h in ctrl.sketch.all_handles():
            if ctrl.selected_entity_id >= 0 and h.entity_id != ctrl.selected_entity_id:
                if ctrl.hover_handle is None or ctrl.hover_handle.entity_id != h.entity_id:
                    continue
            pts.append(fr.to_world(h.uv))
        if not pts:
            return
        # Flat 2D dots on-plane — never spheres (overlay layer handles z-order)
        col = HANDLE_HOVER if ctrl.hover_handle else HANDLE_COLOR
        self._add_overlay_mesh(
            _flat_point_cloud(np.array(pts, float)),
            color=col,
            point_size=14 if ctrl.hover_handle else 10,
            render_points_as_spheres=False,
            name="__sk_handles",
            pickable=False,
            render=False,
        )

    def _update_preview_visual(self) -> None:
        if not self.plotter or self._sketch_ctrl is None:
            return
        self._clear_preview()
        ctrl = self._sketch_ctrl
        if ctrl.tool == SketchTool.SELECT or ctrl.preview_uv is None:
            self._update_handles_visual()
            return
        if ctrl.draw is None or not ctrl.draw.points:
            return
        fr = ctrl.sketch.frame
        p0 = ctrl.draw.points[0]
        p1 = ctrl.preview_uv
        if ctrl.tool is SketchTool.LINE:
            # Exactly on-plane; overlay layer composites above grid/axes
            pdata = pv.Line(fr.to_world(p0), fr.to_world(p1))
        elif ctrl.tool is SketchTool.RECTANGLE:
            from cadcore.sketch import EntityKind
            r = RectEntity(id=-1, kind=EntityKind.RECTANGLE, c0=p0, c1=p1)
            pdata = _entity_polydata(r, ctrl.sketch)
        else:
            from cadcore.sketch import EntityKind
            rad = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
            c = CircleEntity(id=-1, kind=EntityKind.CIRCLE, center=p0, radius=max(rad, 1e-6))
            pdata = _entity_polydata(c, ctrl.sketch)
        self._add_overlay_mesh(
            pdata,
            color=SKETCH_PREVIEW,
            line_width=3.0,
            name="__sk_preview",
            pickable=False,
            render=False,
        )
        self._apply_sketch_actor_priority("__sk_preview")
        # Inference cue — flat 2D dot on-plane, not a sphere
        if ctrl.last_snap.kind in ("h", "v", "origin", "point"):
            w = fr.to_world(ctrl.preview_uv)
            self._add_overlay_mesh(
                _flat_point_cloud(np.array([w], float)),
                color=ACCENT,
                point_size=12,
                render_points_as_spheres=False,
                name="__sk_infer",
                pickable=False,
                render=False,
            )

    def _update_cursor(self) -> None:
        if self._sketch_ctrl is None:
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
            return
        if self._sketch_ctrl.tool != SketchTool.SELECT:
            self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        elif self._sketch_ctrl.hover_handle is not None:
            self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
        else:
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

    # ----- mouse → sketch UV -----
    def _display_to_ray(self, x: float, y: float):
        """Return (origin, direction) world ray for widget coords."""
        assert self.plotter is not None
        # Qt y down; VTK display y up
        h = self.plotter.interactor.height()
        dx, dy = float(x), float(h - y)
        ren = self.plotter.renderer
        ren.SetDisplayPoint(dx, dy, 0.0)
        ren.DisplayToWorld()
        near = np.array(ren.GetWorldPoint()[:3], float)
        ren.SetDisplayPoint(dx, dy, 1.0)
        ren.DisplayToWorld()
        far = np.array(ren.GetWorldPoint()[:3], float)
        # Homogeneous divide if needed
        w0 = ren.GetWorldPoint()[3]
        # GetWorldPoint already after DisplayToWorld — use stored
        # re-query properly
        ren.SetDisplayPoint(dx, dy, 0.0)
        ren.DisplayToWorld()
        wp = ren.GetWorldPoint()
        near = np.array([wp[0] / wp[3], wp[1] / wp[3], wp[2] / wp[3]], float)
        ren.SetDisplayPoint(dx, dy, 1.0)
        ren.DisplayToWorld()
        wp = ren.GetWorldPoint()
        far = np.array([wp[0] / wp[3], wp[1] / wp[3], wp[2] / wp[3]], float)
        d = far - near
        n = np.linalg.norm(d)
        if n < 1e-12:
            return near, np.array([0, 0, -1.0])
        return near, d / n

    def _mouse_to_uv(self, x: float, y: float) -> Optional[Vec2]:
        if self._sketch_ctrl is None:
            return None
        o, d = self._display_to_ray(x, y)
        hit = self._sketch_ctrl.sketch.frame.ray_intersect(o, d)
        if hit is None:
            return None
        return self._sketch_ctrl.sketch.frame.to_local(hit)

    def _sketch_mouse_move(self, x: float, y: float) -> None:
        if self._sketch_ctrl is None:
            return
        uv = self._mouse_to_uv(x, y)
        if uv is None:
            return
        prev_hover = self._sketch_ctrl.hover_handle
        self._sketch_ctrl.on_move(uv)
        # Incremental: only update dragged entity or preview
        if self._sketch_ctrl.drag is not None:
            ent = self._sketch_ctrl.sketch.find_entity(self._sketch_ctrl.drag.entity_id)
            if ent:
                self._upsert_entity_actor(ent)
            self._update_handles_visual()
            self._update_dim_labels()
        else:
            self._update_preview_visual()
            if self._sketch_ctrl.hover_handle != prev_hover:
                self._update_handles_visual()
                self._update_cursor()
        self._request_render()

    def _sketch_mouse_press(self, x: float, y: float) -> None:
        if self._sketch_ctrl is None:
            return
        uv = self._mouse_to_uv(x, y)
        if uv is None:
            return
        # Snapshot before drag for undoable move
        self._drag_before = None
        if self._sketch_ctrl.tool == SketchTool.SELECT:
            # peek handle/entity under cursor
            h = self._sketch_ctrl.pick_handle(uv)
            if h is not None:
                ent0 = self._sketch_ctrl.sketch.find_entity(h.entity_id)
                if ent0 is not None:
                    self._drag_before = snapshot_entity(ent0)
        msg = self._sketch_ctrl.on_press(uv)
        sk = self._sketch_ctrl.sketch
        if msg in ("Line", "Rectangle", "Circle"):
            # new entity added — last one
            ent = sk.entities[-1]
            if self._doc is not None and self._sketch_feature_id >= 0:
                self._doc.record_entity_add(self._sketch_feature_id, ent)
            self._upsert_entity_actor(ent)
            self._clear_preview()
            self._update_handles_visual()
            self._update_junction_dots()
            self._update_dim_labels()
            self.sketch_status.emit(f"Sketch: {msg}")
        elif msg and msg.startswith("Drag"):
            self.sketch_status.emit(f"Sketch: {msg}")
        elif msg and msg.startswith("Selected"):
            self._rebuild_all_sketch_entities()
            self.sketch_status.emit(f"Sketch: Select")
        elif msg:
            self.sketch_status.emit(f"Sketch: {msg}")
        self._request_render()

    def _sketch_mouse_release(self, x: float, y: float) -> None:
        if self._sketch_ctrl is None:
            return
        uv = self._mouse_to_uv(x, y)
        if uv is None:
            self._sketch_ctrl.drag = None
            self._drag_before = None
            return
        was = self._sketch_ctrl.drag
        self._sketch_ctrl.on_release(uv)
        if was is not None:
            ent = self._sketch_ctrl.sketch.find_entity(was.entity_id)
            if ent:
                self._upsert_entity_actor(ent)
                if (
                    self._doc is not None
                    and self._sketch_feature_id >= 0
                    and self._drag_before is not None
                ):
                    after = snapshot_entity(ent)
                    self._doc.record_entity_move(
                        self._sketch_feature_id, self._drag_before, after
                    )
            self._update_handles_visual()
            self._update_junction_dots()
            self._update_dim_labels()
            self.sketch_status.emit(f"{self._sketch_ctrl.sketch.name} — edited")
        self._drag_before = None
        self._request_render()

    # ----- pick / views -----
    def _on_mesh_pick(self, mesh) -> None:  # noqa: ANN001
        if self.in_sketch_mode or mesh is None or self._doc is None or not self.plotter:
            return
        fid = -1
        for name, actor in self.plotter.actors.items():
            if not (name.startswith("plane_") or name.startswith("solid_")):
                continue
            try:
                if np.allclose(actor.GetBounds(), mesh.bounds, atol=1e-4):
                    fid = int(name.split("_", 1)[1])
                    break
            except Exception:
                continue
        if fid < 0:
            c = np.asarray(mesh.center)
            h = PLANE_HALF + 0.25
            for f in self._doc.features:
                if not is_reference_plane(f.type) or not f.visible:
                    continue
                if f.type is FeatureType.PLANE_FRONT and abs(c[2]) < 0.25 and abs(c[0]) <= h and abs(c[1]) <= h:
                    fid = f.id
                    break
                if f.type is FeatureType.PLANE_TOP and abs(c[1]) < 0.25 and abs(c[0]) <= h and abs(c[2]) <= h:
                    fid = f.id
                    break
                if f.type is FeatureType.PLANE_RIGHT and abs(c[0]) < 0.25 and abs(c[1]) <= h and abs(c[2]) <= h:
                    fid = f.id
                    break
        if fid >= 0:
            self._selected_id = fid
            self._restyle_selection_only()
            self._request_render()
            self.feature_picked.emit(fid)

    def set_view(self, name: str) -> None:
        if not self.plotter or self.in_sketch_mode:
            if self.in_sketch_mode:
                self.status_message.emit("Exit sketch to change standard views")
            return
        dist = 11.0
        focus = (0.0, 0.0, 0.0)
        up = (0.0, 1.0, 0.0)
        if name == "front":
            pos = (0.0, 0.0, dist)
        elif name == "back":
            pos = (0.0, 0.0, -dist)
        elif name == "top":
            pos = (0.0, dist, 0.0)
            up = (0.0, 0.0, -1.0)
        elif name == "bottom":
            pos = (0.0, -dist, 0.0)
            up = (0.0, 0.0, 1.0)
        elif name == "right":
            pos = (dist, 0.0, 0.0)
        elif name == "left":
            pos = (-dist, 0.0, 0.0)
        else:
            pos = (dist * 0.75, dist * 0.55, dist * 0.75)
            name = "iso"
        self.plotter.camera_position = [pos, focus, up]
        self._request_render()
        self.status_message.emit(f"View: {name.capitalize()}")

    def zoom_to_fit(self) -> None:
        if not self.plotter:
            return
        self.plotter.reset_camera()
        self._request_render()
        self.status_message.emit("Zoom to fit")

    def _request_render(self) -> None:
        if self._ok and not self._render_timer.isActive():
            self._render_timer.start()

    def _do_render(self) -> None:
        if self.plotter:
            self.plotter.render()
