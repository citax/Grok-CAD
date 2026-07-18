"""PyVista viewport: planes, sketches, async solids, software-GL friendly."""

from __future__ import annotations

import math
import sys
import time
from typing import Dict, Optional, Sequence, Set, Tuple

import numpy as np
import pyvista as pv
from PySide6.QtCore import QEvent, QObject, Qt, QThreadPool, QTimer, Signal, Slot
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget
from pyvistaqt import QtInteractor

from app.sketch_mode import SNAP_POINT_PX, SketchController, SketchTool
from app.theme import (
    ACCENT,
    TEXT_PRIMARY,
    AXIS_LABEL,
    AXIS_X,
    AXIS_Y,
    AXIS_Z,
    GRID_COLOR,
    HANDLE_COLOR,
    HANDLE_HOVER,
    PLANE_FRONT,
    PLANE_RIGHT,
    PLANE_TOP,
    PROFILE_FILL,
    SEL_BOX_CROSSING,
    SEL_BOX_WINDOW,
    SKETCH_COLOR,
    SKETCH_GRID,
    SKETCH_H,
    SKETCH_PREVIEW,
    SKETCH_SELECTED,
    SKETCH_V,
    SOLID_COLOR,
    SOLID_SELECTED,
    VP_BG_BOTTOM,
    VP_BG_TOP,
)
from app.workers import GeometryRebuildJob, snapshot_features
from cadcore.document import Document, FeatureType, is_reference_plane, is_solid_feature
from cadcore.faces import plane_frame_from_face
from cadcore.scale import (
    EMPTY_PLANE_HALF_MM,
    axis_length_mm,
    characteristic_length_from_bounds,
    characteristic_length_from_points,
    empty_workspace_camera,
    origin_glyph_sizes,
    origin_triad_policy,
    plane_half_mm,
    reference_planes_should_show,
    sketch_entity_uv_extent,
    sketch_grid_params,
    sketch_parallel_scale,
)
from cadcore.sketch import (
    CircleEntity,
    LineEntity,
    PlaneFrame,
    RectEntity,
    Sketch,
    SketchEntity,
    Vec2,
    dimension_anchor_uv,
    line_length,
    measure_dimension_value,
    snapshot_entity,
)
from cadcore.units import format_length

PLANE_COLORS = {
    FeatureType.PLANE_FRONT: PLANE_FRONT,
    FeatureType.PLANE_TOP: PLANE_TOP,
    FeatureType.PLANE_RIGHT: PLANE_RIGHT,
}
# Default reference-plane half-extent (mm); scaled up for large parts.
PLANE_HALF = 25.0


def _hex_to_rgb01(hex_color: str) -> tuple:
    """'#RRGGBB' → (r, g, b) in 0..1."""
    h = hex_color.lstrip("#")
    return (
        int(h[0:2], 16) / 255.0,
        int(h[2:4], 16) / 255.0,
        int(h[4:6], 16) / 255.0,
    )


# Performance ablation / final tuning (software-GL).
# GROK_PERF_METHOD: none | a | b | c | d | e | final
#   a = (removed) SetSize half-res — incompatible with Qt-embedded window
#   b = LOD: hide labels/junctions/overlay heavy props while interacting
#   c = batch static sketch lines into fewer actors (closed sketches)
#   d = disable MSAA/transparency while interacting
#   e = aggressive render coalesce
#   final = combination of methods that helped in ablation
import os as _os

_PERF_METHOD = (_os.environ.get("GROK_PERF_METHOD") or "final").strip().lower()


def _perf_enabled(*methods: str) -> bool:
    if _PERF_METHOD == "final":
        # Ablation winners: c (batch) ≫ b (LOD) ≥ e (coalesce).
        # a (SetSize half-res) removed — collapses Qt-embedded viewport into a corner.
        return any(m in ("b", "c", "e") for m in methods)
    if _PERF_METHOD == "a":
        return False  # no SetSize path remains
    return _PERF_METHOD in methods


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


def _dashed_polyline_polydata(
    points_world,
    *,
    dash: float = 0.08,
    gap: float = 0.05,
    max_dashes: int = 48,
) -> pv.PolyData:
    """Build a dashed polyline in ONE shot (no per-dash merge loop).

    vtkProperty line stipple is unreliable on OpenGL2 — bake dashes into the mesh
    as a single PolyData with an explicit ``lines`` connectivity array.

    Dash count is hard-capped at ``max_dashes`` by enlarging dash+gap when the
    perimeter would otherwise produce too many segments (keeps per-move cost flat
    as the box grows or zoom changes).
    """
    pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 2:
        return pv.PolyData()

    edges: list[tuple[np.ndarray, np.ndarray, float]] = []
    peri = 0.0
    for i in range(pts.shape[0] - 1):
        a = pts[i]
        b = pts[i + 1]
        length = float(np.linalg.norm(b - a))
        if length < 1e-12:
            continue
        edges.append((a, b, length))
        peri += length
    if not edges:
        return pv.PolyData()

    dash = max(1e-9, float(dash))
    gap = max(1e-9, float(gap))
    period = dash + gap
    # Cap: if perimeter would yield too many dash onsets, scale period up
    max_dashes = max(4, int(max_dashes))
    expected = peri / period
    if expected > max_dashes:
        scale = expected / float(max_dashes)
        dash *= scale
        gap *= scale
        period = dash + gap

    p0_list: list[np.ndarray] = []
    p1_list: list[np.ndarray] = []
    for a, b, length in edges:
        direction = (b - a) / length
        # Dash start parameters along this edge (vectorized)
        t0s = np.arange(0.0, length, period, dtype=np.float64)
        if t0s.size == 0:
            continue
        t1s = np.minimum(t0s + dash, length)
        keep = t1s > t0s + 1e-12
        t0s = t0s[keep]
        t1s = t1s[keep]
        if t0s.size == 0:
            continue
        p0_list.append(a + np.outer(t0s, direction))
        p1_list.append(a + np.outer(t1s, direction))

    if not p0_list:
        return pv.lines_from_points(pts, close=False)

    starts = np.vstack(p0_list)
    ends = np.vstack(p1_list)
    n = int(starts.shape[0])
    # Interleave start/end → [s0, e0, s1, e1, ...]
    points = np.empty((n * 2, 3), dtype=np.float64)
    points[0::2] = starts
    points[1::2] = ends
    # Connectivity: [2, i0, i1, 2, i2, i3, ...]
    lines = np.empty(n * 3, dtype=np.int64)
    lines[0::3] = 2
    lines[1::3] = np.arange(0, 2 * n, 2, dtype=np.int64)
    lines[2::3] = np.arange(1, 2 * n, 2, dtype=np.int64)
    return pv.PolyData(points, lines=lines)


# Origin glyph size defaults (world units / mm) — scaled with scene
ORIGIN_CROSS_HALF = 1.4
ORIGIN_RING_R = 0.9
ORIGIN_RING_N = 36


def _origin_glyph_polydata(
    *,
    half: float = ORIGIN_CROSS_HALF,
    ring_r: float = ORIGIN_RING_R,
    ring_n: int = ORIGIN_RING_N,
) -> pv.PolyData:
    """Flat crosshair + thin ring centered at world origin (XY), not a sphere/point.

    Geometry lies in z=0 (Front plane origin). Modest size for soft-GL readability.
    """
    # Crosshair arms
    h_line = pv.Line((-half, 0.0, 0.0), (half, 0.0, 0.0))
    v_line = pv.Line((0.0, -half, 0.0), (0.0, half, 0.0))
    # Thin ring
    ang = np.linspace(0.0, 2.0 * np.pi, ring_n + 1)
    ring_pts = np.column_stack(
        [ring_r * np.cos(ang), ring_r * np.sin(ang), np.zeros_like(ang)]
    )
    ring = pv.lines_from_points(ring_pts, close=False)
    # Tiny center mark (very short cross, still a line — not a GL point sprite)
    c = 0.025
    c_h = pv.Line((-c, 0.0, 0.0), (c, 0.0, 0.0))
    c_v = pv.Line((0.0, -c, 0.0), (0.0, c, 0.0))
    return h_line.merge([v_line, ring, c_h, c_v])


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
            self.vp._sketch_mouse_move(
                event.position().x(),
                event.position().y(),
                shift=bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier),
                ctrl=bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier),
            )
            return True  # consume — lock camera
        if et == QEvent.Type.MouseButtonDblClick and event.button() == Qt.MouseButton.LeftButton:
            # Double-click ends a polyline chain
            if self.vp._sketch_ctrl and self.vp._sketch_ctrl.in_line_chain():
                self.vp._end_line_chain()
                return True
        if et == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            self.vp._sketch_mouse_press(
                event.position().x(),
                event.position().y(),
                shift=bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier),
                ctrl=bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier),
            )
            return True
        if et == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
            self.vp._sketch_mouse_release(
                event.position().x(),
                event.position().y(),
                shift=bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier),
                ctrl=bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier),
            )
            return True
        if et == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Escape:
                self.vp._clear_length_buffer()
                self.vp.sketch_escape()
                return True
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                # Prefer committing typed length if buffer has digits
                if self.vp._try_commit_length_buffer():
                    return True
                self.vp.sketch_confirm()
                return True
            if event.key() == Qt.Key.Key_Backspace:
                if self.vp._length_buffer:
                    self.vp._length_buffer = self.vp._length_buffer[:-1]
                    self.vp._emit_length_buffer_status()
                    return True
            # Inline numeric length while drawing a line
            text = event.text() or ""
            if text and (text.isdigit() or text in ".-"):
                if self.vp._accept_length_char(text):
                    return True
        # Block middle/right orbit while sketching
        if et in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonDblClick):
            if event.button() != Qt.MouseButton.LeftButton:
                return True
        if et == QEvent.Type.Wheel:
            return True
        return False


class _ViewControlFilter(QObject):
    """Intercept left-clicks on the bottom-left triad to snap standard views."""

    def __init__(self, viewport: "Viewport") -> None:
        super().__init__(viewport)
        self.vp = viewport

    def eventFilter(self, obj, event):  # noqa: N802
        if self.vp.in_sketch_mode:
            return False
        if event.type() != QEvent.Type.MouseButtonPress:
            return False
        if event.button() != Qt.MouseButton.LeftButton:
            return False
        try:
            x = float(event.position().x())
            y = float(event.position().y())
        except Exception:
            return False
        name = self.vp.try_corner_axes_click(x, y)
        if name is None:
            return False
        # Consumed — do not also mesh-pick through the triad
        return True


class Viewport(QWidget):
    feature_picked = Signal(int)
    status_message = Signal(str)
    busy_changed = Signal(bool, str)
    sketch_exited = Signal()
    sketch_status = Signal(str)
    renderer_info = Signal(str)
    # Smart Dimension: entity_id, role ("length"|"width"|"height"|"diameter")
    dimension_requested = Signal(int, str)
    # Camera view changed via triad / cube / space menu (name of view)
    view_changed = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._doc: Optional[Document] = None
        self._selected_id = -1
        self._ok = False
        self.gl_renderer = ""
        self._planes_built = False
        self._solid_fps: Dict[int, str] = {}
        # Cache triangle meshes for face picking (fid → (verts, faces))
        self._solid_mesh_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
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
        # Scene scale (mm) — drives world axes / origin / reference planes
        self._char_mm: float = 50.0
        self._plane_half: float = PLANE_HALF
        self._axis_len: float = axis_length_mm(self._char_mm)
        self._grid_half: float = 50.0
        self._grid_step: float = 5.0
        # Last planar face pick for sketch-on-face (solid_id + PlaneFrame)
        self._face_pick_solid_id: int = -1
        self._face_pick_frame: Optional[PlaneFrame] = None
        self._face_pick_point: Optional[np.ndarray] = None

        self._rebuild_timer = QTimer(self)
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.setInterval(40)
        self._rebuild_timer.timeout.connect(self._start_rebuild_job)

        # Render-on-demand coalesce (throttle full VTK renders)
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(16)
        self._render_timer.timeout.connect(self._do_render)
        # Longer coalesce during live resize/maximize (soft-GL is expensive)
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(50)
        self._resize_timer.timeout.connect(self._end_resize_coalesce)
        self._resizing = False
        self._length_buffer: str = ""
        self._last_sketch_uv: Optional[Vec2] = None
        self._interacting = False
        self._lod_hidden: bool = False
        self._saved_size: Optional[Tuple[int, int]] = None
        self._perf_method = _PERF_METHOD
        self._draw_lod_active: bool = False
        self._draw_saved_size: Optional[Tuple[int, int]] = None
        # Instrumentation: full VTK renders during strokes
        self._render_count: int = 0
        self._render_ms_sum: float = 0.0
        self._stroke_move_ms: list = []  # per-move wall times while drawing

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

        # pyvistaqt maps VTK trackball cursors (hand / size-all / resize) onto the
        # Qt interactor via CursorChangedEvent → setCursor(). Leave the platform
        # default: never set a shape on the GL widget.
        self._disable_vtk_cursor_overrides()

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

    def _disable_vtk_cursor_overrides(self) -> None:
        """Stop pyvistaqt/VTK from calling QWidget.setCursor on the interactor.

        QVTKRenderWindowInteractor (pyvistaqt/rwi.py) observes CursorChangedEvent
        and maps VTK_CURSOR_* to Qt shapes. That is what changes the mouse over
        the 3D view after our own setCursor code was removed.
        """
        if self.plotter is None:
            return
        iw = self.plotter.interactor
        if iw is None:
            return

        def _platform_cursor(*_args, **_kwargs) -> None:
            iw.unsetCursor()

        # Replace ShowCursor / HideCursor used by CursorChangedEvent + VTK hide
        iw.ShowCursor = _platform_cursor  # type: ignore[method-assign]
        iw.HideCursor = _platform_cursor  # type: ignore[method-assign]
        iw.unsetCursor()

    # ----- setup -----
    def _configure_softgl_render(self) -> None:
        assert self.plotter is not None
        rw = self.plotter.render_window
        try:
            # Software GL (llvmpipe) is fill-rate bound at large windows —
            # never enable MSAA; keep interactive rate modest.
            rw.SetMultiSamples(0)
            rw.SetDesiredUpdateRate(20.0)
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
        # SolidWorks-like: no environment bounds box; no scene-spanning grey axes.
        # Orientation: corner triad (bottom-left) + camera orientation cube (top-right).
        self._origin_axes_actor = None
        self._corner_axes_actor = None
        self._corner_axes_widget = None
        self._camera_orient_widget = None  # legacy name; unused (was ball-gizmo)
        self._view_cube = None
        self._axes_viewport = (0.0, 0.0, 0.13, 0.13)  # bottom-left fraction
        self._refresh_world_helpers(force=True)
        self._fit_empty_workspace()
        # Corner reference triad (SW bottom-left compass) — fixed screen size.
        try:
            actor = self.plotter.add_axes(
                interactive=False,
                line_width=3,
                xlabel="X",
                ylabel="Y",
                zlabel="Z",
                x_color=AXIS_X,
                y_color=AXIS_Y,
                z_color=AXIS_Z,
                viewport=self._axes_viewport,
                cone_radius=0.45,
                shaft_length=0.78,
                tip_length=0.22,
                ambient=0.55,
            )
            self._corner_axes_actor = actor
            self._style_axes_captions(actor)
            try:
                self._corner_axes_widget = self.plotter.renderer.axes_widget
            except Exception:
                self._corner_axes_widget = getattr(self.plotter, "axes_widget", None)
        except Exception as exc:  # noqa: BLE001
            print(f"[viewport] add_axes: {exc}", file=sys.stderr)
            raise
        # Real chamfered view cube (top-right) — face/corner click + drag orbit
        try:
            from app.view_cube_widget import ViewCubeController, apply_cube_view

            def _on_cube_view(token: str) -> None:
                apply_cube_view(self, token)
                if self._view_cube is not None:
                    self._view_cube.sync_orientation()

            def _on_cube_orbit(daz: float, delv: float) -> None:
                self.orbit_camera(daz, delv)

            self._view_cube = ViewCubeController(
                self, on_view=_on_cube_view, on_orbit=_on_cube_orbit
            )
            self._view_cube.install()
        except Exception as exc:  # noqa: BLE001
            print(f"[viewport] view cube: {exc}", file=sys.stderr)
            self._view_cube = None

    def _compute_char_mm(self) -> float:
        """Characteristic scene length (mm) from solids + sketches + default."""
        pts: list = []
        for verts, _faces in self._solid_mesh_cache.values():
            if verts is not None and len(verts):
                pts.extend(np.asarray(verts, float).reshape(-1, 3))
        if self._doc is not None:
            for f in self._doc.features:
                if f.type is FeatureType.SKETCH and f.sketch is not None:
                    fr = f.sketch.frame
                    for e in f.sketch.entities:
                        if isinstance(e, LineEntity):
                            pts.append(fr.to_world(e.p0))
                            pts.append(fr.to_world(e.p1))
                        elif isinstance(e, RectEntity):
                            for c in e.corners():
                                pts.append(fr.to_world(c))
                        elif isinstance(e, CircleEntity):
                            pts.append(fr.to_world(e.center))
                            pts.append(
                                fr.to_world(
                                    (e.center[0] + e.radius, e.center[1])
                                )
                            )
        if pts:
            return characteristic_length_from_points(pts, default=50.0)
        return 50.0

    def _part_extent_mm(self) -> float:
        """Largest solid AABB diagonal currently cached (mm)."""
        best = 0.0
        for verts, _faces in self._solid_mesh_cache.values():
            if verts is None or len(verts) == 0:
                continue
            best = max(
                best,
                characteristic_length_from_points(verts, default=0.0),
            )
        return float(best)

    def _has_display_solids(self) -> bool:
        return bool(self._solid_fps) or bool(self._solid_mesh_cache)

    def _selected_is_plane(self) -> bool:
        if self._doc is None:
            return False
        f = self._doc.find(self._selected_id)
        return f is not None and is_reference_plane(f.type)

    def _refresh_world_helpers(self, *, force: bool = False) -> None:
        """Update origin marker + plane scale/visibility for current scene.

        Orientation at any size: **corner triad only** (screen-fixed).
        World-origin RGB arrows are suppressed whenever solids are shown so
        they never skewer a tiny part or hide inside a large one.
        """
        if not self.plotter:
            return
        char = self._compute_char_mm()
        has_solids = self._has_display_solids()
        new_half = plane_half_mm(char, has_display_solids=has_solids)
        show_origin, new_axis = origin_triad_policy(
            has_display_solids=has_solids,
            char_mm=char,
            part_extent_mm=self._part_extent_mm(),
        )
        if (
            not force
            and abs(char - self._char_mm) / max(self._char_mm, 1.0) < 0.08
            and abs(new_axis - self._axis_len) / max(self._axis_len, 1.0) < 0.08
            and show_origin == getattr(self, "_origin_shown", None)
            and has_solids == getattr(self, "_had_solids", None)
        ):
            # Still refresh plane visibility (selection may have changed)
            self._apply_plane_visibility()
            return
        self._char_mm = char
        self._axis_len = new_axis
        self._origin_shown = show_origin
        self._had_solids = has_solids
        prev_plane = self._plane_half
        self._plane_half = new_half

        # Remove legacy monochrome helpers if present
        for name in ("__origin", "__ax", "__ay", "__az"):
            self._remove_actor(name)

        if show_origin and new_axis > 1e-9:
            self._install_origin_rgb_triad(new_axis)
        else:
            self._clear_origin_rgb_triad()

        # Rebuild reference planes when half-extent changes materially
        if force or abs(prev_plane - self._plane_half) / max(prev_plane, 1.0) > 0.08:
            self._rebuild_reference_planes()
        else:
            self._apply_plane_visibility()
        # Keep sketch-mode visibility rules
        if self.in_sketch_mode:
            self._set_sketch_2d_chrome(True)
        elif not has_solids:
            # Opening / empty scene: keep the camera framed so planes aren't a smear
            self._fit_empty_workspace()

    def _clear_origin_rgb_triad(self) -> None:
        prev = getattr(self, "_origin_axes_actor", None)
        if prev is not None and self.plotter is not None:
            try:
                self.plotter.renderer.RemoveActor(prev)
            except Exception:
                try:
                    self.plotter.remove_actor(prev, render=False)
                except Exception:
                    pass
        self._origin_axes_actor = None
        for name in ("__ax", "__ay", "__az"):
            self._remove_actor(name)

    def _install_origin_rgb_triad(self, length_mm: float) -> None:
        """Small RGB origin triad for **empty** scenes only (see origin_triad_policy)."""
        if not self.plotter:
            return
        L = max(1.0, float(length_mm))
        self._clear_origin_rgb_triad()
        try:
            actor = self.plotter.add_axes_at_origin(
                x_color=AXIS_X,
                y_color=AXIS_Y,
                z_color=AXIS_Z,
                xlabel="",
                ylabel="",
                zlabel="",
                line_width=2,
                labels_off=True,
            )
            try:
                actor.SetTotalLength(L, L, L)
            except Exception:
                pass
            try:
                actor.SetShaftTypeToCylinder()
                actor.SetTipTypeToCone()
                actor.SetCylinderRadius(0.03)
                actor.SetConeRadius(0.09)
            except Exception:
                pass
            self._origin_axes_actor = actor
        except Exception as exc:  # noqa: BLE001
            print(f"[viewport] origin triad: {exc}", file=sys.stderr)
            for end, name, col in (
                ((L, 0, 0), "__ax", AXIS_X),
                ((0, L, 0), "__ay", AXIS_Y),
                ((0, 0, L), "__az", AXIS_Z),
            ):
                self._remove_actor(name)
                self.plotter.add_mesh(
                    pv.Line((0, 0, 0), end),
                    color=col,
                    line_width=3.0,
                    name=name,
                    pickable=False,
                    render=False,
                )

    def _apply_plane_visibility(self) -> None:
        """Show/hide reference planes per SolidWorks/Fusion-style policy."""
        if not self.plotter or self._doc is None:
            return
        show = reference_planes_should_show(
            has_display_solids=self._has_display_solids(),
            in_sketch_mode=self.in_sketch_mode,
            selected_is_plane=self._selected_is_plane(),
        )
        self._planes_env_visible = show
        for f in self._doc.features:
            if not is_reference_plane(f.type):
                continue
            # Feature.visible is user pin; env policy is additional gate
            vis = 1 if (show and f.visible) else 0
            for prefix in ("plane_", "edge_"):
                name = f"{prefix}{f.id}"
                act = self._get_named_actor(name)
                if act is None and name in getattr(self.plotter, "actors", {}):
                    act = self.plotter.actors.get(name)
                if act is not None:
                    try:
                        act.SetVisibility(vis)
                    except Exception:
                        pass

    def _rebuild_reference_planes(self) -> None:
        if not self.plotter or self._doc is None:
            return
        half = float(self._plane_half)
        for f in self._doc.features:
            if not is_reference_plane(f.type):
                continue
            color = PLANE_COLORS[f.type]
            for prefix in ("plane_", "edge_"):
                self._remove_actor(f"{prefix}{f.id}")
            # Quieter than before — still readable when empty/sketching
            # Very light fill + clear border (empty scene must stay readable)
            self.plotter.add_mesh(
                _plane_surface(f.type, half=half),
                color=color,
                opacity=0.07,
                name=f"plane_{f.id}",
                pickable=True,
                smooth_shading=False,
                show_edges=False,
                render=False,
            )
            self.plotter.add_mesh(
                _plane_border(f.type, half=half),
                color=color,
                line_width=2.0,
                name=f"edge_{f.id}",
                pickable=False,
                render=False,
            )
        self._planes_built = True
        self._apply_plane_visibility()
        self._restyle_selection_only()

    @staticmethod
    def _style_axes_captions(actor) -> None:
        """Set X/Y/Z caption colour to AXIS_LABEL (0–1 RGB) on vtkAxesActor.

        PyVista's add_axes does not accept label_color in 0.48.x; captions default
        to black and vanish on dark themes unless set here.
        """
        if actor is None:
            raise RuntimeError("add_axes returned None — cannot style captions")
        r, g, b = _hex_to_rgb01(AXIS_LABEL)
        for getter in (
            "GetXAxisCaptionActor2D",
            "GetYAxisCaptionActor2D",
            "GetZAxisCaptionActor2D",
        ):
            cap = getattr(actor, getter)()
            if cap is None:
                continue
            prop = cap.GetCaptionTextProperty()
            prop.SetColor(float(r), float(g), float(b))
            prop.SetOpacity(1.0)
            prop.BoldOn()
            try:
                prop.ShadowOff()
            except Exception:
                pass
            try:
                # Slightly larger for readability at 2560×1440
                prop.SetFontSize(max(14, int(prop.GetFontSize() or 12)))
            except Exception:
                pass

    def _setup_picking(self) -> None:
        assert self.plotter is not None
        try:
            self.plotter.enable_mesh_picking(
                callback=self._on_mesh_pick, left_clicking=True, show=False, show_message=False
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[viewport] picking: {exc}", file=sys.stderr)
        # Corner triad click → standard view (when not in sketch mode)
        try:
            self._view_click_filter = _ViewControlFilter(self)
            self.plotter.interactor.installEventFilter(self._view_click_filter)
        except Exception as exc:  # noqa: BLE001
            print(f"[viewport] view click filter: {exc}", file=sys.stderr)
            self._view_click_filter = None

    def _setup_interaction_lod(self) -> None:
        """Throttle VTK during camera drag; keep view cube locked to main camera."""
        assert self.plotter is not None
        self._interacting = False
        try:
            iren = self.plotter.iren.interactor
        except Exception:
            return

        def on_start(_o=None, _e=None) -> None:
            if self.in_sketch_mode:
                return
            self._begin_interaction_lod()

        def on_end(_o=None, _e=None) -> None:
            self._end_interaction_lod()

        def on_interact(_o=None, _e=None) -> None:
            # Main trackball updates the camera then VTK re-renders without going
            # through our _do_render — so the corner cube used to freeze until
            # mouse-up. Sync orientation on every interaction tick.
            if self.in_sketch_mode:
                return
            cube = getattr(self, "_view_cube", None)
            if cube is not None:
                cube.sync_orientation()

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
        try:
            iren.AddObserver("InteractionEvent", on_interact)
        except Exception:
            pass

        # Belt-and-suspenders: any VTK Render() (wheel zoom, middle-drag, etc.)
        # updates the cube before layers are drawn — not only our _do_render path.
        def on_rw_start(_o=None, _e=None) -> None:
            cube = getattr(self, "_view_cube", None)
            if cube is not None:
                cube.sync_orientation()

        try:
            self.plotter.render_window.AddObserver("StartEvent", on_rw_start)
        except Exception:
            pass

    def _begin_interaction_lod(self) -> None:
        """Camera-drag LOD — never SetSize on the Qt-embedded render window.

        render_window.SetSize on a QtInteractor only paints into a corner of the
        widget (size is owned by Qt). Safe LOD: hide heavy overlays + lower
        DesiredUpdateRate. Fill-rate still limited by software GL / real GPU.
        """
        if self._interacting:
            return
        self._interacting = True
        if not self.plotter:
            return
        try:
            self.plotter.render_window.SetDesiredUpdateRate(12.0)
        except Exception:
            pass
        # Method "a" (SetSize half-res) is intentionally NOT applied here —
        # incompatible with Qt-embedded windows (collapses scene into a corner).
        # (b) hide heavy overlays
        if _perf_enabled("b"):
            self._set_interaction_lod_visible(False)
        # (d) force opaque / no MSAA while interacting
        if _perf_enabled("d"):
            try:
                self.plotter.render_window.SetMultiSamples(0)
                ren = self.plotter.renderer
                ren.SetUseDepthPeeling(0)
            except Exception:
                pass

    def _end_interaction_lod(self) -> None:
        if not self._interacting and not self._lod_hidden:
            return
        self._interacting = False
        if not self.plotter:
            return
        try:
            self.plotter.render_window.SetDesiredUpdateRate(0.0001)
        except Exception:
            pass
        # Clear any legacy saved size from older builds; never SetSize restore
        self._saved_size = None
        if _perf_enabled("b"):
            self._set_interaction_lod_visible(True)
        # Still render only — no feature rebuild on pure camera move
        self._request_render()

    def _set_interaction_lod_visible(self, visible: bool) -> None:
        """Hide/show labels, junctions, plane edges while interacting (method b)."""
        self._lod_hidden = not visible
        if not self.plotter:
            return
        names = []
        names.extend(getattr(self, "_dim_label_names", set()) or set())
        names.extend(
            [
                "__sk_junctions",
                "__sk_dims",
                "__sk_grid",
                "__sk_handles",
                "__sk_h",
                "__sk_v",
            ]
        )
        for n in list(self.plotter.actors.keys()):
            if n.startswith("edge_"):
                names.append(n)
        for n in list(self._overlay_actors.keys()):
            if n in ("__sk_junctions", "__sk_handles", "__sk_dims") or n.startswith(
                "__sk_dim"
            ):
                names.append(n)
        for n in names:
            act = self._get_named_actor(n)
            if act is None:
                continue
            try:
                act.SetVisibility(1 if visible else 0)
            except Exception:
                pass

    def _set_draw_solids_visible(self, visible: bool) -> None:
        """Hide solid meshes during stroke (cheap no-op when none present)."""
        if not self.plotter:
            return
        for n in list(self.plotter.actors.keys()):
            if not n.startswith("solid_"):
                continue
            act = self.plotter.actors.get(n)
            if act is None:
                continue
            try:
                act.SetVisibility(1 if visible else 0)
            except Exception:
                pass

    def _set_draw_planes_visible(self, visible: bool) -> None:
        """Show/hide reference-plane fills AND border edges during a stroke.

        Profiled @2560×1440 (llvmpipe, non-additive overdraw):
          - hide plane_* fills alone: ~35 → ~19 ms (ratio ~0.53)
          - with fills already gone, hide edge_* borders: ~30 → ~18 ms (~12 ms)
          - old chrome/grid/label hide: ~1.00× (worthless — not used here)
        Sketch grid + entity strokes + preview stay visible.
        """
        if not self.plotter:
            return
        for n in list(self.plotter.actors.keys()):
            if not (n.startswith("plane_") or n.startswith("edge_")):
                continue
            act = self.plotter.actors.get(n)
            if act is None:
                continue
            try:
                act.SetVisibility(1 if visible else 0)
            except Exception:
                pass

    def _begin_draw_lod(self) -> None:
        """Stroke LOD: hide plane fills + plane edges (+ solids if any).

        Measured @2560×1440: labels/grid/junction hide was ~1.00× and is NOT
        applied. Hiding translucent plane quads + their border edges is the
        real app-side win above the ~17–18 ms empty-viewport llvmpipe floor.

        UX: during a stroke the colored plane fill and its border ring vanish;
        sketch grid, H/V axes, and committed entities remain. Do NOT SetSize.
        """
        if self._draw_lod_active:
            return
        self._draw_lod_active = True
        self._stroke_move_ms = []
        # Retargeted: plane fills + borders + solids — NOT chrome (was 1.00x)
        self._set_draw_planes_visible(False)
        self._set_draw_solids_visible(False)
        self._draw_saved_size = None
        try:
            if self.plotter is not None:
                self.plotter.render_window.SetDesiredUpdateRate(20.0)
        except Exception:
            pass
        if self._render_timer.isActive():
            self._render_timer.stop()
        self._do_render()

    def _end_draw_lod(self) -> None:
        if not self._draw_lod_active:
            return
        self._draw_lod_active = False
        self._set_draw_planes_visible(True)
        self._set_draw_solids_visible(True)
        self._draw_saved_size = None
        self._clear_preview()
        self._clear_selbox()
        try:
            if self.plotter is not None:
                self.plotter.render_window.SetDesiredUpdateRate(0.0001)
        except Exception:
            pass
        if self._render_timer:
            self._render_timer.setInterval(16)
        # Restore full overlays from live state + one full render
        self._update_junction_dots()
        self._update_dim_labels()
        self._update_handles_visual()
        if self._render_timer.isActive():
            self._render_timer.stop()
        self._do_render()

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
        # Selecting a plane may re-show planes when solids are present
        self._apply_plane_visibility()
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
            self._solid_mesh_cache.pop(fid, None)
        for fid, (verts, faces, fp) in results.items():
            # Always keep full-res mesh for face picking / scale (even if actor reused)
            self._solid_mesh_cache[fid] = (
                np.asarray(verts, dtype=np.float64),
                np.asarray(faces, dtype=np.int32),
            )
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
        self._refresh_world_helpers()
        self._restyle_selection_only()
        self._request_render()

    def _ensure_planes(self) -> None:
        if not self.plotter or self._doc is None:
            return
        if self._planes_built:
            self._apply_plane_visibility()
            return
        self._rebuild_reference_planes()
        self._request_render()

    def _request_render(self, *, interactive: bool = False) -> None:
        """Schedule a VTK paint.

        ``interactive=True`` (camera drag / view-cube orbit): paint immediately
        when the frame budget allows (~60 Hz). Continuous mouse moves must not
        keep pushing the paint into the future.

        IMPORTANT: never call ``QTimer.setInterval`` on an *active* timer.
        Qt restarts the countdown, so a stream of mouse moves deferred the
        single-shot forever — the view only caught up after the mouse stopped.
        """
        if not self._ok:
            return
        now = time.perf_counter()
        # Interactive camera: draw now if ≥16 ms since last paint
        if interactive or getattr(self, "_interacting", False):
            last = float(getattr(self, "_last_render_t", 0.0) or 0.0)
            if (now - last) >= (1.0 / 60.0):
                if self._render_timer.isActive():
                    self._render_timer.stop()
                self._do_render()
                return
            # Budget not free yet — one trailing frame, do not restart countdown
            if not self._render_timer.isActive():
                self._render_timer.setInterval(16)
                self._render_timer.start()
            return

        # (e) aggressive coalesce during resize / idle paints
        if _perf_enabled("e"):
            interval = 120 if self._resizing else 24
        else:
            interval = 80 if self._resizing else 16
        # Only setInterval when starting — never while active (restarts timer).
        if not self._render_timer.isActive():
            self._render_timer.setInterval(interval)
            self._render_timer.start()

    def _do_render(self) -> None:
        if self.plotter:
            if self._view_cube is not None:
                self._view_cube.sync_orientation()
            t0 = time.perf_counter()
            self.plotter.render()
            self._last_render_t = time.perf_counter()
            self._render_count += 1
            self._render_ms_sum += (self._last_render_t - t0) * 1000.0

    def reset_render_stats(self) -> None:
        self._render_count = 0
        self._render_ms_sum = 0.0

    def render_stats(self) -> Tuple[int, float]:
        """Return (full VTK render count, total ms) since last reset."""
        return self._render_count, self._render_ms_sum

    def resizeEvent(self, event) -> None:  # noqa: N802
        # Coalesce paints while the window is being dragged/maximized.
        # Pure camera/window resize must NOT rebuild sketch/solid actors.
        # Never SetSize the embedded render window (Qt owns geometry).
        self._resizing = True
        settle = 120 if _perf_enabled("e") else 80
        self._resize_timer.setInterval(settle)
        self._resize_timer.start()  # restarts → settle
        self._request_render()
        super().resizeEvent(event)

    def _end_resize_coalesce(self) -> None:
        self._resizing = False
        self._request_render()

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
        """Strong, obvious selection feedback for planes and solids."""
        if not self.plotter:
            return
        # Gold highlight for selected plane (fill + border)
        sel_rgb = (1.0, 0.84, 0.15)  # bright amber
        sel_edge = (1.0, 0.95, 0.35)
        for name, actor in list(self.plotter.actors.items()):
            if not (
                name.startswith("plane_")
                or name.startswith("edge_")
                or name.startswith("solid_")
            ):
                continue
            try:
                fid = int(name.split("_", 1)[1])
            except ValueError:
                continue
            prop = actor.GetProperty()
            selected = fid == self._selected_id and not self.in_sketch_mode
            f = self._doc.find(fid) if self._doc is not None else None

            if name.startswith("plane_"):
                # Dim all planes in sketch mode; unselected stay translucent
                if self.in_sketch_mode:
                    prop.SetOpacity(0.10)
                    if f is not None and is_reference_plane(f.type):
                        prop.SetColor(*_hex_to_rgb01(PLANE_COLORS[f.type]))
                elif selected:
                    # Selected: clearer but not a solid brown slab
                    prop.SetOpacity(0.28)
                    prop.SetColor(*sel_rgb)
                    prop.SetEdgeVisibility(1)
                    prop.SetEdgeColor(*sel_edge)
                    prop.SetLineWidth(3.0)
                else:
                    prop.SetOpacity(0.07)
                    if f is not None and is_reference_plane(f.type):
                        prop.SetColor(*_hex_to_rgb01(PLANE_COLORS[f.type]))
                    prop.SetEdgeVisibility(0)

            elif name.startswith("edge_"):
                # Border ring — thick gold when selected, soft type color otherwise
                if self.in_sketch_mode:
                    prop.SetOpacity(0.20)
                    prop.SetLineWidth(1.5)
                    if f is not None and is_reference_plane(f.type):
                        prop.SetColor(*_hex_to_rgb01(PLANE_COLORS[f.type]))
                elif selected:
                    prop.SetOpacity(1.0)
                    prop.SetColor(*sel_edge)
                    prop.SetLineWidth(5.0)
                else:
                    prop.SetOpacity(0.75)
                    prop.SetLineWidth(2.0)
                    if f is not None and is_reference_plane(f.type):
                        prop.SetColor(*_hex_to_rgb01(PLANE_COLORS[f.type]))

            else:  # solid_
                if self.in_sketch_mode:
                    prop.SetOpacity(0.25)
                else:
                    prop.SetOpacity(1.0)
                prop.SetColor(*(sel_rgb if selected else (0.60, 0.64, 0.68)))

    # ----- sketch mode -----
    def _set_parallel_projection(self, enabled: bool, *, parallel_scale: Optional[float] = None) -> None:
        """Toggle orthographic (parallel) camera — true 2D sketch look."""
        if not self.plotter:
            return
        try:
            cam = self.plotter.camera
            cam.SetParallelProjection(1 if enabled else 0)
            if enabled:
                scale = float(parallel_scale) if parallel_scale is not None else 40.0
                cam.SetParallelScale(max(1.0, scale))
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

        # Orient camera normal to plane + orthographic projection (true 2D sketch)
        fr = f.sketch.frame
        ent_ext = sketch_entity_uv_extent(f.sketch.entities)
        p_scale = sketch_parallel_scale(ent_ext, default=max(40.0, self._char_mm * 0.6))
        dist = max(p_scale * 3.0, self._axis_len * 2.0, 50.0)
        pos = fr.origin + fr.normal * dist
        self.plotter.camera_position = [
            tuple(pos),
            tuple(fr.origin),
            tuple(fr.v_axis),
        ]
        self._set_parallel_projection(True, parallel_scale=p_scale)
        self._set_sketch_2d_chrome(True)
        self._ensure_sketch_overlay_layer()
        # Planes visible again in sketch (SW shows the sketch plane context)
        self._apply_plane_visibility()

        self._install_sketch_filter()
        try:
            self.plotter.interactor.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self.plotter.interactor.setFocus()
        except Exception:
            pass
        self._dim_label_names: Set[str] = set()
        self._drag_before: Optional[dict] = None
        self._clear_length_buffer()
        self._last_sketch_uv = (0.0, 0.0)
        self._draw_sketch_overlay()
        self._rebuild_all_sketch_entities()
        self._update_junction_dots()
        self._update_dim_labels()
        self._restyle_selection_only()
        self._request_render()
        plane = self._doc.find(f.plane_id)
        pname = plane.name if plane else "face"
        self.sketch_status.emit(f"Editing {f.name} on {pname} (2D)")

    def exit_sketch(self) -> None:
        if not self.in_sketch_mode:
            return
        self._end_draw_lod()
        self._remove_sketch_filter()
        self._clear_sketch_overlays()
        self._teardown_sketch_overlay_layer()
        self._clear_length_buffer()
        self._sketch_ctrl = None
        self._sketch_feature_id = -1
        # Leave 2D sketch: restore 3D chrome + perspective for solids
        self._set_sketch_2d_chrome(False)
        self._set_parallel_projection(False)
        # Solids present → hide planes again (SW View/Hide Planes behaviour)
        self._apply_plane_visibility()
        self._restyle_selection_only()
        self.refresh_sketches()
        self._request_render()
        self.sketch_exited.emit()

    def _set_sketch_2d_chrome(self, enabled: bool) -> None:
        """Toggle 2D sketch chrome: hide 3D origin triad; keep sketch H/V axes."""
        if not self.plotter:
            return
        vis = 0 if enabled else 1
        # Short RGB origin triad (world) — hide in pure 2D sketch
        for name in ("__ax", "__ay", "__az", "__origin"):
            act = self._get_named_actor(name)
            if act is not None:
                try:
                    act.SetVisibility(vis)
                except Exception:
                    pass
        oa = getattr(self, "_origin_axes_actor", None)
        if oa is not None:
            try:
                oa.SetVisibility(vis)
            except Exception:
                pass
        # Corner orientation triad is still useful in sketch (SW keeps it) — leave on.
        # (Previously hidden; SW sketcher keeps the corner triad for orientation.)

    def sketch_cursor_uv(self) -> Optional[Vec2]:
        """Last known sketch-plane UV under the cursor (for paste placement)."""
        if self._sketch_ctrl is None:
            return None
        if self._sketch_ctrl.preview_uv is not None:
            return self._sketch_ctrl.preview_uv
        return self._last_sketch_uv

    def set_sketch_tool(self, tool: SketchTool) -> None:
        if self._sketch_ctrl is None:
            return
        self._sketch_ctrl.set_tool(tool)
        self._clear_preview()
        self.sketch_status.emit(f"Sketch: {tool.name.title()}")

    def sketch_escape(self) -> None:
        """Esc: cancel in-progress draw/box → Select; if idle, exit sketch mode."""
        if self._sketch_ctrl is None:
            return
        if (
            self._sketch_ctrl.is_drawing()
            or self._sketch_ctrl.drag is not None
            or self._sketch_ctrl.is_box_selecting()
        ):
            self._sketch_ctrl.cancel_drawing()
            self._sketch_ctrl.set_tool(SketchTool.SELECT)
            self._end_draw_lod()
            self._clear_preview()
            self._clear_selbox()
            self._update_handles_visual()
            self.sketch_status.emit("Sketch: Select")
            self._request_render()
            return
        # Idle select — exit sketch entirely
        self.exit_sketch()
        self.sketch_status.emit("Exited sketch")

    def sketch_confirm(self) -> None:
        """Enter: finish rubber-band segment, or end polyline chain if mid-chain."""
        if self._sketch_ctrl is None:
            return
        n_before = len(self._sketch_ctrl.sketch.entities)
        msg = self._sketch_ctrl.confirm_current()
        if not msg:
            return
        if msg.startswith("ChainEnd"):
            self._end_draw_lod()
            self._clear_preview()
            self.sketch_status.emit("Sketch: polyline finished")
            self._request_render()
            return
        # New entity committed
        if len(self._sketch_ctrl.sketch.entities) > n_before:
            ent = self._sketch_ctrl.sketch.entities[-1]
            if self._doc is not None and self._sketch_feature_id >= 0:
                self._doc.record_entity_add(self._sketch_feature_id, ent)
            self._upsert_entity_actor(ent)
        if msg == "Line" and self._sketch_ctrl.is_drawing():
            # Chaining continues — keep preview live
            self._begin_draw_lod()
            self.sketch_status.emit("Sketch: Line (chain…)")
        else:
            self._end_draw_lod()
            self._clear_preview()
            self.sketch_status.emit(f"Sketch: {msg}")
        self._request_render()

    def _end_line_chain(self) -> None:
        if self._sketch_ctrl is None:
            return
        msg = self._sketch_ctrl.end_line_chain()
        self._end_draw_lod()
        self._clear_preview()
        self.sketch_status.emit("Sketch: polyline finished" if msg else "Sketch")
        self._request_render()

    def _world_units_per_pixel(self) -> float:
        """Convert 1 screen pixel to sketch-plane world units (for snap radius)."""
        if not self.plotter:
            return 1.0
        try:
            cam = self.plotter.camera
            h = max(1, int(self.plotter.window_size[1]))
            if bool(cam.GetParallelProjection()):
                # ParallelScale = half-height of the view frustum in world units
                return float(2.0 * cam.GetParallelScale() / h)
            # Perspective: approx at focal distance
            dist = float(np.linalg.norm(
                np.asarray(cam.GetPosition()) - np.asarray(cam.GetFocalPoint())
            ))
            view_angle = float(cam.GetViewAngle())  # degrees
            half_h = dist * float(np.tan(np.radians(view_angle) * 0.5))
            return float(2.0 * half_h / h)
        except Exception:
            return 1.0

    def _update_snap_tolerance(self) -> None:
        """Pixel-based snap: convert SNAP_POINT_PX to world using current camera."""
        if self._sketch_ctrl is None:
            return
        wpp = self._world_units_per_pixel()
        point_tol = max(1e-9, SNAP_POINT_PX * wpp)
        # Grid step matches the drawn sketch grid (world mm)
        self._sketch_ctrl.set_snap_world_tol(point_tol, grid=self._grid_step)

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

    def _draw_sketch_overlay(self) -> None:
        """Sketch grid + local H/V axes on the plane (scale with view + geometry)."""
        if not self.plotter or self._sketch_ctrl is None:
            return
        fr = self._sketch_ctrl.sketch.frame
        # View half-height (orthographic parallel scale) drives grid density
        try:
            view_half = float(self.plotter.camera.GetParallelScale())
        except Exception:
            view_half = max(40.0, self._char_mm * 0.6)
        ent_ext = sketch_entity_uv_extent(self._sketch_ctrl.sketch.entities)
        half, step = sketch_grid_params(view_half, entity_extent_mm=ent_ext)
        self._grid_half = half
        self._grid_step = step
        self._update_snap_tolerance()
        pts_u = []
        pts_v = []
        g = np.arange(-half, half + 1e-9, step)
        for t in g:
            a = fr.to_world((t, -half))
            b = fr.to_world((t, half))
            pts_u.extend([a, b])
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
        # Sketch H (red) / V (green) axes — SolidWorks sketcher convention.
        # Slightly thicker than grid; full grid span so orientation reads at any zoom.
        self._remove_actor("__sk_h")
        self._remove_actor("__sk_v")
        self.plotter.add_mesh(
            pv.Line(fr.to_world((-half, 0)), fr.to_world((half, 0))),
            color=SKETCH_H, line_width=2.5, name="__sk_h", pickable=False, render=False,
        )
        self.plotter.add_mesh(
            pv.Line(fr.to_world((0, -half)), fr.to_world((0, half))),
            color=SKETCH_V, line_width=2.5, name="__sk_v", pickable=False, render=False,
        )

    def selected_profile_ids(self) -> Set[int]:
        """Closed-region ids currently selected in sketch mode (for Extrude)."""
        if self._sketch_ctrl is None:
            return set()
        return set(self._sketch_ctrl.selected_profile_ids)

    def _profile_fill_polydata(self, profile) -> Optional[pv.PolyData]:
        """Filled polygon in world space for a closed profile highlight."""
        from cadcore.profiles import profile_polygon_uv

        if self._sketch_ctrl is None:
            return None
        fr = self._sketch_ctrl.sketch.frame
        uvs = profile_polygon_uv(profile)
        if len(uvs) < 3:
            return None
        pts = np.array([fr.to_world(uv) for uv in uvs], dtype=float)
        n = len(pts)
        # VTK polygon face: [n, i0, i1, ..., i{n-1}]
        faces = np.hstack([[n], np.arange(n, dtype=np.int64)])
        return pv.PolyData(pts, faces=faces)

    def _update_profile_fill_visual(self) -> None:
        """Show translucent fills for selected closed regions (profile pick)."""
        self._remove_actor("__sk_profile_fill")
        # Also clear multi-part fills
        for name in list(self._overlay_actors.keys()):
            if name.startswith("__sk_profile_fill"):
                self._remove_actor(name)
        if self._sketch_ctrl is None or not self._sketch_ctrl.selected_profile_ids:
            return
        from cadcore.profiles import profile_by_id

        meshes: list[pv.PolyData] = []
        for pid in self._sketch_ctrl.selected_profile_ids:
            prof = profile_by_id(self._sketch_ctrl.sketch, pid)
            if prof is None:
                continue
            pd = self._profile_fill_polydata(prof)
            if pd is not None and pd.n_points >= 3:
                meshes.append(pd)
        if not meshes:
            return
        merged = meshes[0]
        for m in meshes[1:]:
            try:
                merged = merged.merge(m)
            except Exception:
                pass
        fill = PROFILE_FILL or SKETCH_SELECTED or ACCENT
        # High opacity + no lighting so the fill reads clearly on both themes
        # (soft-GL still blends a little with the gradient behind).
        self._add_overlay_mesh(
            merged,
            color=fill,
            opacity=0.72,
            name="__sk_profile_fill",
            pickable=False,
            render=False,
            show_edges=False,
            lighting=False,
            ambient=1.0,
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
            "__sk_selbox",
            "__sk_handles",
            "__sk_infer",
            "__sk_junctions",
            "__sk_dims",
            "__sk_profile_fill",
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
        """Ensure sketch entity lines draw as flat 2D strokes above grid/axes."""
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
            # True 2D line look — never tubes / never shaded 3D edges
            try:
                prop.SetRenderLinesAsTubes(False)
            except Exception:
                pass
            try:
                prop.SetRepresentationToSurface()
            except Exception:
                pass
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
        # (c) batch all closed sketch geometry into one actor per sketch feature
        if _perf_enabled("c"):
            self._refresh_sketches_batched()
            return
        wanted: Dict[str, Tuple[object, str]] = {}  # name -> (ent, fp)  # type: ignore
        for f in self._doc.features:
            if f.type is not FeatureType.SKETCH or f.sketch is None or not f.visible:
                continue
            for ent in f.sketch.entities:
                name = f"sk_closed_{f.id}_{ent.id}"
                fp = _entity_fingerprint(ent, selected=False)
                wanted[name] = (ent, fp)
        for name in list(self._closed_sketch_fps.keys()):
            if name not in wanted:
                self._remove_actor(name)
                self._closed_sketch_fps.pop(name, None)
        dirty = False
        for name, (ent, fp) in wanted.items():
            if self._closed_sketch_fps.get(name) == fp and name in self.plotter.actors:
                continue
            self._remove_actor(name)
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

    def _refresh_sketches_batched(self) -> None:
        """Merge each closed sketch's entities into a single multi-block actor."""
        assert self.plotter is not None and self._doc is not None
        wanted_fps: Dict[str, str] = {}
        dirty = False
        # Drop per-entity closed actors from non-batch mode
        for name in list(self._closed_sketch_fps.keys()):
            if name.startswith("sk_closed_") and "_batch_" not in name:
                self._remove_actor(name)
                self._closed_sketch_fps.pop(name, None)
        for f in self._doc.features:
            if f.type is not FeatureType.SKETCH or f.sketch is None or not f.visible:
                continue
            if not f.sketch.entities:
                name = f"sk_closed_batch_{f.id}"
                if name in self._closed_sketch_fps:
                    self._remove_actor(name)
                    self._closed_sketch_fps.pop(name, None)
                continue
            fp = "|".join(_entity_fingerprint(e) for e in f.sketch.entities)
            name = f"sk_closed_batch_{f.id}"
            wanted_fps[name] = fp
            if self._closed_sketch_fps.get(name) == fp and name in self.plotter.actors:
                continue
            self._remove_actor(name)
            meshes = []
            for ent in f.sketch.entities:
                try:
                    meshes.append(_entity_polydata(ent, f.sketch))
                except Exception:
                    pass
            if not meshes:
                continue
            try:
                merged = meshes[0]
                for m in meshes[1:]:
                    merged = merged.merge(m)
            except Exception:
                merged = meshes[0]
            self.plotter.add_mesh(
                merged,
                color=SKETCH_COLOR,
                line_width=2.5,
                name=name,
                pickable=False,
                render=False,
            )
            self._apply_sketch_actor_priority(name)
            self._closed_sketch_fps[name] = fp
            dirty = True
        for name in list(self._closed_sketch_fps.keys()):
            if name.startswith("sk_closed_batch_") and name not in wanted_fps:
                self._remove_actor(name)
                self._closed_sketch_fps.pop(name, None)
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
        # Grow orthographic window + grid when geometry outruns the current view
        try:
            ent_ext = sketch_entity_uv_extent(self._sketch_ctrl.sketch.entities)
            need = sketch_parallel_scale(ent_ext, default=max(40.0, self._char_mm * 0.6))
            cam = self.plotter.camera
            if bool(cam.GetParallelProjection()) and float(cam.GetParallelScale()) < need * 0.95:
                cam.SetParallelScale(need)
        except Exception:
            pass
        self._draw_sketch_overlay()
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
        """Flat faint dots only at shared connection points (≥2 entities meet)."""
        if not self.plotter or self._sketch_ctrl is None:
            return
        self._remove_actor("__sk_junctions")
        pts_uv = self._sketch_ctrl.sketch.shared_endpoints()
        if not pts_uv:
            return
        fr = self._sketch_ctrl.sketch.frame
        pts = np.array([fr.to_world(p) for p in pts_uv], float)
        self._add_overlay_mesh(
            _flat_point_cloud(pts),
            color=SKETCH_GRID,
            point_size=5,
            render_points_as_spheres=False,
            name="__sk_junctions",
            pickable=False,
            render=False,
            opacity=0.35,
        )

    def _update_dim_labels(self) -> None:
        """Show driving dimensions always; also ephemeral labels on selected lines."""
        if not self.plotter or self._sketch_ctrl is None or self._doc is None:
            return
        for name in list(getattr(self, "_dim_label_names", set()) or set()):
            self._remove_actor(name)
        self._dim_label_names = set()
        self._remove_actor("__sk_dims")
        unit = self._doc.display_unit
        fr = self._sketch_ctrl.sketch.frame
        ctrl = self._sketch_ctrl
        sk = ctrl.sketch
        points = []
        labels = []
        seen: Set[Tuple[int, str]] = set()
        # 1) Driving dimensions (persist, drive geometry)
        for dim in getattr(sk, "dimensions", None) or []:
            ent = sk.find_entity(dim.entity_id)
            if ent is None:
                continue
            key = (int(dim.entity_id), str(dim.role))
            if key in seen:
                continue
            seen.add(key)
            try:
                val = measure_dimension_value(ent, dim.role)
            except ValueError:
                val = float(dim.value_mm)
            # Keep stored value in sync if user dragged handles
            dim.value_mm = float(val)
            anchor = dimension_anchor_uv(ent, dim.role)
            points.append(fr.to_world(anchor))
            prefix = "⌀" if dim.role == "diameter" else ""
            labels.append(prefix + format_length(val, unit))
        # 2) Ephemeral selected/hovered line lengths (when no driving dim yet)
        show_ids = set(ctrl.selected_ids)
        if ctrl.hover_handle is not None:
            show_ids.add(ctrl.hover_handle.entity_id)
        for ent in sk.entities:
            if not isinstance(ent, LineEntity) or ent.id not in show_ids:
                continue
            if (ent.id, "length") in seen:
                continue
            mid = ent.midpoint()
            points.append(fr.to_world(mid))
            labels.append(format_length(line_length(ent), unit))
        if not points:
            return
        try:
            self.plotter.add_point_labels(
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
        except Exception as exc:  # noqa: BLE001
            print(f"[viewport] dim labels: {exc}", file=sys.stderr)

    def dim_label_texts(self) -> list:
        """Label strings currently shown (driving dims + selected lines)."""
        if self._sketch_ctrl is None or self._doc is None:
            return []
        unit = self._doc.display_unit
        ctrl = self._sketch_ctrl
        sk = ctrl.sketch
        out = []
        seen: Set[Tuple[int, str]] = set()
        for dim in getattr(sk, "dimensions", None) or []:
            ent = sk.find_entity(dim.entity_id)
            if ent is None:
                continue
            key = (int(dim.entity_id), str(dim.role))
            if key in seen:
                continue
            seen.add(key)
            try:
                val = measure_dimension_value(ent, dim.role)
            except ValueError:
                val = float(dim.value_mm)
            prefix = "⌀" if dim.role == "diameter" else ""
            out.append(prefix + format_length(val, unit))
        show_ids = set(ctrl.selected_ids)
        if ctrl.hover_handle is not None:
            show_ids.add(ctrl.hover_handle.entity_id)
        for ent in sk.entities:
            if isinstance(ent, LineEntity) and ent.id in show_ids and (ent.id, "length") not in seen:
                out.append(format_length(line_length(ent), unit))
        return out

    def labeled_entity_ids(self) -> list:
        """Entity ids that currently have a dimension label."""
        if self._sketch_ctrl is None:
            return []
        ctrl = self._sketch_ctrl
        ids = []
        for dim in getattr(ctrl.sketch, "dimensions", None) or []:
            if dim.entity_id not in ids and ctrl.sketch.find_entity(dim.entity_id):
                ids.append(dim.entity_id)
        for eid in ctrl.selected_ids:
            ent = ctrl.sketch.find_entity(eid)
            if isinstance(ent, LineEntity) and eid not in ids:
                ids.append(eid)
        if ctrl.hover_handle is not None:
            hid = ctrl.hover_handle.entity_id
            if hid not in ids:
                ent = ctrl.sketch.find_entity(hid)
                if isinstance(ent, LineEntity):
                    ids.append(hid)
        return ids

    def face_pick_frame(self) -> Optional[PlaneFrame]:
        """Last solid-face PlaneFrame from a viewport pick, if any."""
        return self._face_pick_frame

    def face_pick_solid_id(self) -> int:
        return int(self._face_pick_solid_id)

    def clear_face_pick(self) -> None:
        self._face_pick_solid_id = -1
        self._face_pick_frame = None
        self._face_pick_point = None

    def solid_display_vertices(self, fid: int) -> Optional[np.ndarray]:
        """Vertices of the solid **as drawn** (VTK actor mapper input).

        This is the mesh that ``_apply_solid_results`` uploaded into
        ``plotter.actors['solid_{fid}']`` — the same PolyData VTK composites
        into the framebuffer. Prefer this over ``Document.evaluate_feature``
        when verifying what the user sees.
        """
        if not self.plotter:
            return None
        actor = self.plotter.actors.get(f"solid_{int(fid)}")
        if actor is None:
            return None
        try:
            from vtkmodules.util.numpy_support import vtk_to_numpy

            mapper = actor.GetMapper()
            if mapper is None:
                return None
            data = mapper.GetInput()
            if data is None or data.GetPoints() is None:
                return None
            pts = vtk_to_numpy(data.GetPoints().GetData())
            return np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        except Exception as exc:  # noqa: BLE001
            print(f"[viewport] solid_display_vertices: {exc}", file=sys.stderr)
            return None

    def sketch_entity_display_points(self, eid: int) -> Optional[np.ndarray]:
        """World points of a sketch entity actor on the overlay / main plotter."""
        name = f"sk_e_{int(eid)}"
        actor = self._overlay_actors.get(name)
        if actor is None and self.plotter is not None:
            actor = self.plotter.actors.get(name)
        if actor is None:
            return None
        try:
            from vtkmodules.util.numpy_support import vtk_to_numpy

            mapper = actor.GetMapper()
            if mapper is None:
                return None
            data = mapper.GetInput()
            if data is None or data.GetPoints() is None:
                return None
            pts = vtk_to_numpy(data.GetPoints().GetData())
            return np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        except Exception as exc:  # noqa: BLE001
            print(f"[viewport] sketch_entity_display_points: {exc}", file=sys.stderr)
            return None

    def _upsert_entity_actor(self, ent: SketchEntity) -> None:
        if not self.plotter or self._sketch_ctrl is None:
            return
        name = f"sk_e_{ent.id}"
        selected = ent.id in self._sketch_ctrl.selected_ids
        fp = _entity_fingerprint(ent, selected=selected)
        if self._sketch_entity_fps.get(ent.id) == fp and (
            name in self.plotter.actors or name in self._overlay_actors
        ):
            return  # unchanged — skip VTK teardown/rebuild
        self._remove_actor(name)
        pdata = _entity_polydata(ent, self._sketch_ctrl.sketch)
        col = SKETCH_SELECTED if selected else SKETCH_COLOR
        self._add_overlay_mesh(
            pdata, color=col, line_width=3.0, name=name, pickable=False, render=False
        )
        self._apply_sketch_actor_priority(name)
        self._sketch_entity_actors.add(ent.id)
        self._sketch_entity_fps[ent.id] = fp

    def _clear_preview(self) -> None:
        self._remove_actor("__sk_preview")
        self._remove_actor("__sk_infer")

    def _clear_selbox(self) -> None:
        self._remove_actor("__sk_selbox")

    def _clear_handles(self) -> None:
        self._remove_actor("__sk_handles")

    def _update_handles_visual(self) -> None:
        if not self.plotter or self._sketch_ctrl is None:
            return
        self._clear_handles()
        ctrl = self._sketch_ctrl
        fr = ctrl.sketch.frame
        pts = []
        sel = ctrl.selected_ids
        for h in ctrl.sketch.all_handles():
            if sel and h.entity_id not in sel:
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

    def _update_selbox_visual(self) -> None:
        """Live selection rectangle as VTK overlay actor (__sk_selbox).

        Window (L→R): solid border in SEL_BOX_WINDOW.
        Crossing (R→L): dashed border (explicit short segments) in SEL_BOX_CROSSING.
        """
        if not self.plotter or self._sketch_ctrl is None:
            return
        bs = self._sketch_ctrl.box_select
        if bs is None or bs.drag_px < 1.0:
            self._clear_selbox()
            return
        fr = self._sketch_ctrl.sketch.frame
        u0, v0, u1, v1 = bs.uv_rect()
        corners_uv = [(u0, v0), (u1, v0), (u1, v1), (u0, v1), (u0, v0)]
        corners_w = [fr.to_world(c) for c in corners_uv]
        if bs.is_window:
            pdata = pv.lines_from_points(np.array(corners_w, float), close=False)
            col = SEL_BOX_WINDOW
            lw = 2.0
        else:
            # Screen-space dash density (~8 px dash / ~5 px gap) + hard cap in
            # _dashed_polyline_polydata so dash count stays O(1) as the box grows.
            wpp = self._world_units_per_pixel()
            dash = max(0.04, float(wpp) * 8.0)
            gap = max(0.025, float(wpp) * 5.0)
            pdata = _dashed_polyline_polydata(
                corners_w, dash=dash, gap=gap, max_dashes=48
            )
            col = SEL_BOX_CROSSING
            lw = 2.2
        self._add_overlay_mesh(
            pdata,
            color=col,
            line_width=lw,
            name="__sk_selbox",
            pickable=False,
            render=False,
        )
        self._apply_sketch_actor_priority("__sk_selbox")

    def _update_preview_visual(self) -> None:
        """Live in-progress preview as a real VTK actor on the layer-1 overlay.

        Same path as committed sketch entities (_add_overlay_mesh). Visible in
        the VTK framebuffer on WSLg/xcb. One full (LOD-light) re-render is
        expected after this on each mouse-move.
        """
        if not self.plotter or self._sketch_ctrl is None:
            return
        ctrl = self._sketch_ctrl
        if ctrl.tool == SketchTool.SELECT or ctrl.preview_uv is None:
            self._clear_preview()
            if not self._draw_lod_active:
                self._update_handles_visual()
            return
        if ctrl.draw is None or not ctrl.draw.points:
            self._clear_preview()
            return
        fr = ctrl.sketch.frame
        p0 = ctrl.draw.points[0]
        p1 = ctrl.preview_uv
        # Degenerate (cursor still at start) — no segment yet
        if (
            abs(float(p1[0]) - float(p0[0])) < 1e-12
            and abs(float(p1[1]) - float(p0[1])) < 1e-12
        ):
            self._clear_preview()
            return
        if ctrl.tool is SketchTool.LINE:
            pdata = pv.Line(fr.to_world(p0), fr.to_world(p1))
        elif ctrl.tool is SketchTool.RECTANGLE:
            from cadcore.sketch import EntityKind

            r = RectEntity(id=-1, kind=EntityKind.RECTANGLE, c0=p0, c1=p1)
            pdata = _entity_polydata(r, ctrl.sketch)
        else:
            from cadcore.sketch import EntityKind

            rad = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
            c = CircleEntity(
                id=-1, kind=EntityKind.CIRCLE, center=p0, radius=max(rad, 1e-6)
            )
            pdata = _entity_polydata(c, ctrl.sketch)
        self._add_overlay_mesh(
            pdata,
            color=SKETCH_PREVIEW,
            line_width=3.5,
            name="__sk_preview",
            pickable=False,
            render=False,
        )
        self._apply_sketch_actor_priority("__sk_preview")
        if ctrl.last_snap.kind in ("h", "v", "origin", "point", "chain_start"):
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
        else:
            self._remove_actor("__sk_infer")

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

    def _sketch_mouse_move(
        self, x: float, y: float, *, shift: bool = False, ctrl: bool = False
    ) -> None:
        if self._sketch_ctrl is None:
            return
        t0 = time.perf_counter()
        self._update_snap_tolerance()
        uv = self._mouse_to_uv(x, y)
        if uv is None:
            return
        self._last_sketch_uv = uv
        prev_hover = self._sketch_ctrl.hover_handle
        drawing = (
            self._sketch_ctrl.is_drawing()
            or self._sketch_ctrl.drag is not None
            or self._sketch_ctrl.is_box_selecting()
        )
        # While drawing/dragging/box-selecting: light LOD (hide chrome + solids)
        if drawing:
            self._begin_draw_lod()
        self._sketch_ctrl.on_move(uv, display_xy=(float(x), float(y)))
        if self._sketch_ctrl.box_select is not None:
            self._update_selbox_visual()
            if self._render_timer.isActive():
                self._render_timer.stop()
            self._do_render()
            self._stroke_move_ms.append((time.perf_counter() - t0) * 1000.0)
            return
        if self._sketch_ctrl.drag is not None:
            ent = self._sketch_ctrl.sketch.find_entity(self._sketch_ctrl.drag.entity_id)
            if ent:
                self._upsert_entity_actor(ent)
            if not self._draw_lod_active:
                self._update_handles_visual()
                self._update_dim_labels()
            if self._render_timer.isActive():
                self._render_timer.stop()
            self._do_render()
            return
        # Live VTK draw preview on layer-1 overlay + one light re-render per move
        self._update_preview_visual()
        if drawing:
            if self._render_timer.isActive():
                self._render_timer.stop()
            self._do_render()
            self._stroke_move_ms.append((time.perf_counter() - t0) * 1000.0)
            return
        if not self._draw_lod_active and self._sketch_ctrl.hover_handle != prev_hover:
            self._update_handles_visual()
        self._request_render()

    def _sketch_mouse_press(
        self, x: float, y: float, *, shift: bool = False, ctrl: bool = False
    ) -> None:
        if self._sketch_ctrl is None:
            return
        self._update_snap_tolerance()
        uv = self._mouse_to_uv(x, y)
        if uv is None:
            return
        # Snapshot before drag for undoable move
        self._drag_before = None
        if self._sketch_ctrl.tool == SketchTool.SELECT:
            h = self._sketch_ctrl.pick_handle(uv)
            if h is not None:
                ent0 = self._sketch_ctrl.sketch.find_entity(h.entity_id)
                if ent0 is not None:
                    self._drag_before = snapshot_entity(ent0)
        n_before = len(self._sketch_ctrl.sketch.entities)
        msg = self._sketch_ctrl.on_press(
            uv, display_xy=(float(x), float(y)), shift=shift, ctrl=ctrl
        )
        sk = self._sketch_ctrl.sketch
        if msg and msg.startswith("DimPick:"):
            # DimPick:<entity_id>:<role>
            parts = msg.split(":")
            if len(parts) >= 3:
                try:
                    eid = int(parts[1])
                    role = parts[2]
                    self.dimension_requested.emit(eid, role)
                    self.sketch_status.emit(f"Dimension on entity {eid} ({role})")
                except ValueError:
                    pass
            self._request_render()
            return
        if msg == "DimPickMiss":
            self.sketch_status.emit("Smart Dimension: click a line, rectangle, or circle")
            return
        if msg in ("Line", "LineClosed", "Rectangle", "Circle"):
            if len(sk.entities) > n_before:
                ent = sk.entities[-1]
                if self._doc is not None and self._sketch_feature_id >= 0:
                    self._doc.record_entity_add(self._sketch_feature_id, ent)
                self._upsert_entity_actor(ent)
            self._clear_preview()
            if msg == "Line" and self._sketch_ctrl.is_drawing():
                # Polyline continues: commit segment as VTK actor, stay in draw LOD
                if not self._draw_lod_active:
                    self._begin_draw_lod()
                if self._sketch_ctrl.draw and self._sketch_ctrl.draw.points:
                    self._sketch_ctrl.preview_uv = self._sketch_ctrl.draw.points[0]
                if self._render_timer.isActive():
                    self._render_timer.stop()
                self._do_render()  # show committed segment(s)
                self.sketch_status.emit("Sketch: Line (chain…)")
            else:
                self._end_draw_lod()
                self._clear_preview()
                self._update_handles_visual()
                self._update_junction_dots()
                self._update_dim_labels()
                self.sketch_status.emit(
                    "Sketch: closed polyline" if msg == "LineClosed" else f"Sketch: {msg}"
                )
                if self._render_timer.isActive():
                    self._render_timer.stop()
                self._do_render()
            return
        elif msg and msg.startswith("Drag"):
            self._begin_draw_lod()
            self.sketch_status.emit(f"Sketch: {msg}")
        elif msg and msg.startswith("Selected profile"):
            self._end_draw_lod()
            self._clear_selbox()
            self._rebuild_all_sketch_entities()
            self._update_profile_fill_visual()
            n = len(self._sketch_ctrl.selected_profile_ids)
            self.sketch_status.emit(
                f"Sketch: {n} region(s) selected" if n != 1 else "Sketch: 1 region selected"
            )
            if self._render_timer.isActive():
                self._render_timer.stop()
            self._do_render()
            return
        elif msg and msg.startswith("Selected"):
            self._end_draw_lod()
            self._clear_selbox()
            self._rebuild_all_sketch_entities()
            self._update_profile_fill_visual()
            n = len(self._sketch_ctrl.selected_ids)
            self.sketch_status.emit(f"Sketch: Select ({n})" if n != 1 else "Sketch: Select")
            if self._render_timer.isActive():
                self._render_timer.stop()
            self._do_render()
            return
        elif msg == "BoxSelect":
            self._begin_draw_lod()
            self._clear_selbox()
            self._rebuild_all_sketch_entities()  # clear highlight if selection wiped
            self._update_profile_fill_visual()
            self.sketch_status.emit("Sketch: box select…")
        elif msg:
            # First click of a draw: enter light LOD; preview appears on next move
            if self._sketch_ctrl.is_drawing():
                self._begin_draw_lod()
                if self._sketch_ctrl.draw and self._sketch_ctrl.draw.points:
                    self._sketch_ctrl.preview_uv = self._sketch_ctrl.draw.points[0]
            self.sketch_status.emit(f"Sketch: {msg}")
        self._request_render()

    def _sketch_mouse_release(
        self, x: float, y: float, *, shift: bool = False, ctrl: bool = False
    ) -> None:
        if self._sketch_ctrl is None:
            return
        uv = self._mouse_to_uv(x, y)
        if uv is None:
            self._sketch_ctrl.drag = None
            self._sketch_ctrl.box_select = None
            self._clear_selbox()
            self._drag_before = None
            return
        was = self._sketch_ctrl.drag
        was_box = self._sketch_ctrl.box_select is not None
        msg = self._sketch_ctrl.on_release(
            uv, display_xy=(float(x), float(y)), shift=shift, ctrl=ctrl
        )
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
            self._end_draw_lod()
            self.sketch_status.emit(f"{self._sketch_ctrl.sketch.name} — edited")
        elif was_box:
            self._clear_selbox()
            self._end_draw_lod()
            self._rebuild_all_sketch_entities()
            self._update_handles_visual()
            self._update_dim_labels()
            self._update_profile_fill_visual()
            if msg and msg.startswith("Selected profile"):
                n = len(self._sketch_ctrl.selected_profile_ids)
                self.sketch_status.emit(
                    f"Sketch: {n} region(s) selected"
                    if n != 1
                    else "Sketch: 1 region selected"
                )
            else:
                n = len(self._sketch_ctrl.selected_ids)
                if msg and msg.startswith("BoxSelect:"):
                    parts = msg.split(":")
                    mode = parts[1] if len(parts) > 1 else "?"
                    self.sketch_status.emit(f"Sketch: {mode} select → {n}")
                elif msg == "BoxSelectClear":
                    self.sketch_status.emit("Sketch: selection cleared")
                else:
                    self.sketch_status.emit(f"Sketch: selected {n}")
            if self._render_timer.isActive():
                self._render_timer.stop()
            self._do_render()
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
            h = float(self._plane_half) + 0.25
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
        if fid < 0:
            return
        self._selected_id = fid
        # Capture planar face under the pick for sketch-on-face
        self._capture_face_pick(fid, mesh)
        self._apply_plane_visibility()
        self._restyle_selection_only()
        self._request_render()
        self.feature_picked.emit(fid)

    def _capture_face_pick(self, fid: int, mesh) -> None:  # noqa: ANN001
        """If ``fid`` is a solid, resolve the planar face under the pick point."""
        self._face_pick_solid_id = -1
        self._face_pick_frame = None
        self._face_pick_point = None
        if self._doc is None:
            return
        f = self._doc.find(fid)
        if f is None or not is_solid_feature(f.type):
            return
        cache = self._solid_mesh_cache.get(fid)
        if cache is None:
            return
        verts, faces = cache
        # Prefer PyVista's last picked world point; fall back to mesh center
        pick_pt = None
        try:
            pp = getattr(self.plotter, "picked_point", None)
            if pp is not None:
                pick_pt = np.asarray(pp, dtype=np.float64).reshape(3)
        except Exception:
            pick_pt = None
        if pick_pt is None:
            try:
                pick_pt = np.asarray(mesh.center, dtype=np.float64).reshape(3)
            except Exception:
                pick_pt = verts.mean(axis=0)
        cell_id = None
        try:
            cells = getattr(self.plotter, "picked_cells", None)
            if cells is not None and getattr(cells, "n_cells", 0):
                # First cell id if available
                cell_id = 0
        except Exception:
            cell_id = None
        try:
            frame = plane_frame_from_face(verts, faces, pick_pt, cell_id=cell_id)
        except Exception as exc:  # noqa: BLE001
            print(f"[viewport] face pick: {exc}", file=sys.stderr)
            return
        self._face_pick_solid_id = int(fid)
        self._face_pick_frame = frame
        self._face_pick_point = pick_pt.copy()
        self.status_message.emit(
            f"Face on {f.name} · n=({frame.normal[0]:.2f},{frame.normal[1]:.2f},{frame.normal[2]:.2f})"
        )

    def set_face_pick_from_mesh(
        self,
        solid_id: int,
        pick_point: Sequence[float],
        *,
        cell_id: Optional[int] = None,
    ) -> Optional[PlaneFrame]:
        """Programmatic face pick (tests / verification) using the real face path."""
        cache = self._solid_mesh_cache.get(int(solid_id))
        if cache is None and self._doc is not None:
            # Evaluate solid if not cached yet
            mesh = self._doc.evaluate_feature(int(solid_id))
            if mesh is not None and not mesh.empty:
                self._solid_mesh_cache[int(solid_id)] = (
                    mesh.vertices.copy(),
                    mesh.faces.copy(),
                )
                cache = self._solid_mesh_cache[int(solid_id)]
        if cache is None:
            return None
        verts, faces = cache
        frame = plane_frame_from_face(verts, faces, pick_point, cell_id=cell_id)
        self._face_pick_solid_id = int(solid_id)
        self._face_pick_frame = frame
        self._face_pick_point = np.asarray(pick_point, dtype=np.float64).reshape(3)
        self._selected_id = int(solid_id)
        return frame

    def _view_distance(self) -> float:
        """Camera distance for standard views (scales with scene)."""
        if self._has_display_solids():
            return max(15.0, self._part_extent_mm() * 1.6, self._char_mm * 1.3)
        h = float(self._plane_half) if self._plane_half > 0 else EMPTY_PLANE_HALF_MM
        return max(h * 3.2, 40.0)

    def _scene_focus(self) -> Tuple[float, float, float]:
        """Focal point: solid centroid if present, else origin."""
        if self._solid_mesh_cache:
            pts = []
            for verts, _ in self._solid_mesh_cache.values():
                if verts is not None and len(verts):
                    pts.append(np.asarray(verts, float).mean(axis=0))
            if pts:
                c = np.mean(np.stack(pts, axis=0), axis=0)
                return (float(c[0]), float(c[1]), float(c[2]))
        return (0.0, 0.0, 0.0)

    def camera_snapshot(self) -> dict:
        """Current camera pose — used by tests to prove the view actually moved."""
        if not self.plotter:
            return {}
        cam = self.plotter.camera
        return {
            "position": tuple(float(x) for x in cam.GetPosition()),
            "focal": tuple(float(x) for x in cam.GetFocalPoint()),
            "up": tuple(float(x) for x in cam.GetViewUp()),
        }

    def _stop_camera_animation(self) -> None:
        timer = getattr(self, "_cam_anim_timer", None)
        if timer is not None:
            try:
                timer.stop()
            except Exception:
                pass
            self._cam_anim_timer = None
        self._cam_anim = None

    def _apply_camera_pose(
        self,
        pos: Sequence[float],
        focus: Sequence[float],
        up: Sequence[float],
        *,
        render: bool = True,
    ) -> None:
        if not self.plotter:
            return
        self.plotter.camera_position = [
            tuple(float(x) for x in pos),
            tuple(float(x) for x in focus),
            tuple(float(x) for x in up),
        ]
        try:
            self.plotter.reset_camera_clipping_range()
        except Exception:
            pass
        if self._view_cube is not None:
            self._view_cube.sync_orientation()
        if render:
            self._request_render()

    @staticmethod
    def _ease_out_cubic(t: float) -> float:
        t = max(0.0, min(1.0, float(t)))
        return 1.0 - (1.0 - t) ** 3

    @staticmethod
    def _camera_basis(
        pos: np.ndarray, foc: np.ndarray, up: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Orthonormal camera frame (right, up, back).

        ``back`` points from focus toward the camera (along the view ray from
        the target). ``up`` is the true view-up after orthogonalization.
        """
        pos = np.asarray(pos, dtype=np.float64).reshape(3)
        foc = np.asarray(foc, dtype=np.float64).reshape(3)
        up = np.asarray(up, dtype=np.float64).reshape(3)
        back = pos - foc
        bn = float(np.linalg.norm(back))
        if bn < 1e-12:
            back = np.array([0.0, 0.0, 1.0])
        else:
            back = back / bn
        # right = normalize(up × back)  so (right, up, back) is right-handed
        # and matches VTK (looking from pos toward foc = −back)
        right = np.cross(up, back)
        rn = float(np.linalg.norm(right))
        if rn < 1e-12:
            # up parallel to view ray — pick a stable alternate
            alt = np.array([0.0, 0.0, 1.0]) if abs(back[1]) > 0.9 else np.array([0.0, 1.0, 0.0])
            right = np.cross(alt, back)
            rn = float(np.linalg.norm(right))
            if rn < 1e-12:
                right = np.array([1.0, 0.0, 0.0])
                rn = 1.0
        right = right / rn
        true_up = np.cross(back, right)
        un = float(np.linalg.norm(true_up))
        if un > 1e-12:
            true_up = true_up / un
        return right, true_up, back

    @staticmethod
    def _basis_to_quat(
        right: np.ndarray, up: np.ndarray, back: np.ndarray
    ) -> np.ndarray:
        """Rotation quaternion (w, x, y, z) for matrix with columns (right, up, back)."""
        m00, m01, m02 = float(right[0]), float(up[0]), float(back[0])
        m10, m11, m12 = float(right[1]), float(up[1]), float(back[1])
        m20, m21, m22 = float(right[2]), float(up[2]), float(back[2])
        tr = m00 + m11 + m22
        if tr > 0.0:
            s = math.sqrt(tr + 1.0) * 2.0
            w = 0.25 * s
            x = (m21 - m12) / s
            y = (m02 - m20) / s
            z = (m10 - m01) / s
        elif m00 > m11 and m00 > m22:
            s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
            w = (m21 - m12) / s
            x = 0.25 * s
            y = (m01 + m10) / s
            z = (m02 + m20) / s
        elif m11 > m22:
            s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
            w = (m02 - m20) / s
            x = (m01 + m10) / s
            y = 0.25 * s
            z = (m12 + m21) / s
        else:
            s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
            w = (m10 - m01) / s
            x = (m02 + m20) / s
            y = (m12 + m21) / s
            z = 0.25 * s
        q = np.array([w, x, y, z], dtype=np.float64)
        n = float(np.linalg.norm(q))
        return q / n if n > 1e-15 else np.array([1.0, 0.0, 0.0, 0.0])

    @staticmethod
    def _quat_to_basis(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Columns of the rotation matrix from quaternion (w,x,y,z)."""
        w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        # right (col0), up (col1), back (col2)
        right = np.array(
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y + z * w),
                2.0 * (x * z - y * w),
            ]
        )
        up = np.array(
            [
                2.0 * (x * y - z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z + x * w),
            ]
        )
        back = np.array(
            [
                2.0 * (x * z + y * w),
                2.0 * (y * z - x * w),
                1.0 - 2.0 * (x * x + y * y),
            ]
        )
        # Renormalize for safety
        for v in (right, up, back):
            n = float(np.linalg.norm(v))
            if n > 1e-15:
                v /= n
        return right, up, back

    @staticmethod
    def _quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
        """Shortest-path spherical lerp of unit quaternions (w,x,y,z)."""
        q0 = np.asarray(q0, dtype=np.float64).reshape(4)
        q1 = np.asarray(q1, dtype=np.float64).reshape(4)
        t = float(t)
        dot = float(np.dot(q0, q1))
        # Shortest arc: flip one quaternion if needed
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        if dot > 0.9995:
            q = q0 + t * (q1 - q0)
            n = float(np.linalg.norm(q))
            return q / n if n > 1e-15 else q0
        dot = min(1.0, max(-1.0, dot))
        theta = math.acos(dot)
        s0 = math.sin((1.0 - t) * theta) / math.sin(theta)
        s1 = math.sin(t * theta) / math.sin(theta)
        q = s0 * q0 + s1 * q1
        n = float(np.linalg.norm(q))
        return q / n if n > 1e-15 else q0

    @classmethod
    def interpolate_camera_pose(
        cls,
        start_pos: Sequence[float],
        start_foc: Sequence[float],
        start_up: Sequence[float],
        end_pos: Sequence[float],
        end_foc: Sequence[float],
        end_up: Sequence[float],
        t: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Shortest-path camera pose at eased parameter ``t`` ∈ [0, 1].

        Interpolates orientation as a single quaternion (not independent
        direction + up slerps). Independent up slerp was spinning the scene
        ~270° of roll on edge views even when start and end up agreed.
        """
        t = max(0.0, min(1.0, float(t)))
        sp = np.asarray(start_pos, dtype=np.float64).reshape(3)
        sf = np.asarray(start_foc, dtype=np.float64).reshape(3)
        su = np.asarray(start_up, dtype=np.float64).reshape(3)
        ep = np.asarray(end_pos, dtype=np.float64).reshape(3)
        ef = np.asarray(end_foc, dtype=np.float64).reshape(3)
        eu = np.asarray(end_up, dtype=np.float64).reshape(3)

        sr, su2, sb = cls._camera_basis(sp, sf, su)
        er, eu2, eb = cls._camera_basis(ep, ef, eu)
        q0 = cls._basis_to_quat(sr, su2, sb)
        q1 = cls._basis_to_quat(er, eu2, eb)
        q = cls._quat_slerp(q0, q1, t)
        _right, up_i, back_i = cls._quat_to_basis(q)

        foc = (1.0 - t) * sf + t * ef
        d0 = float(np.linalg.norm(sp - sf))
        d1 = float(np.linalg.norm(ep - ef))
        if d0 < 1e-9:
            d0 = d1 if d1 > 1e-9 else 1.0
        if d1 < 1e-9:
            d1 = d0
        dist = (1.0 - t) * d0 + t * d1
        pos = foc + back_i * dist
        return pos, foc, up_i

    @classmethod
    def measure_camera_up_path(
        cls,
        start_pos: Sequence[float],
        start_foc: Sequence[float],
        start_up: Sequence[float],
        end_pos: Sequence[float],
        end_foc: Sequence[float],
        end_up: Sequence[float],
        *,
        samples: int = 48,
    ) -> Tuple[float, float]:
        """Return (path_roll_rad, endpoint_up_angle_rad) along the flight.

        ``path_roll_rad`` is the integrated angle between consecutive view-up
        vectors along the interpolated path. ``endpoint_up_angle_rad`` is the
        angle between the true start and end view-up (after orthogonalization).
        """
        samples = max(2, int(samples))
        _, su0, _ = cls._camera_basis(start_pos, start_foc, start_up)
        _, eu0, _ = cls._camera_basis(end_pos, end_foc, end_up)
        end_angle = math.acos(float(np.clip(np.dot(su0, eu0), -1.0, 1.0)))
        path = 0.0
        prev_up = su0
        for i in range(1, samples + 1):
            t = i / float(samples)
            _p, _f, up = cls.interpolate_camera_pose(
                start_pos, start_foc, start_up, end_pos, end_foc, end_up, t
            )
            up = up / (float(np.linalg.norm(up)) + 1e-15)
            d = float(np.clip(np.dot(prev_up, up), -1.0, 1.0))
            path += math.acos(d)
            prev_up = up
        return path, end_angle

    def animate_camera_to(
        self,
        pos: Sequence[float],
        focus: Sequence[float],
        up: Sequence[float],
        *,
        duration_ms: int = 280,
        view_key: str = "",
    ) -> None:
        """Smooth camera travel (Fusion/SW-style) instead of an instant snap.

        Orientation follows the *shortest* rotation between start and end
        frames (quaternion slerp), so edge views no longer roll sideways and
        right themselves mid-flight.
        """
        if not self.plotter:
            return
        end_pos = np.asarray(pos, dtype=np.float64).reshape(3)
        end_foc = np.asarray(focus, dtype=np.float64).reshape(3)
        end_up = np.asarray(up, dtype=np.float64).reshape(3)
        nu = float(np.linalg.norm(end_up))
        if nu > 1e-12:
            end_up = end_up / nu
        # Instant if duration is zero or headless/offscreen without event loop
        if duration_ms <= 0:
            self._stop_camera_animation()
            self._apply_camera_pose(end_pos, end_foc, end_up)
            if view_key:
                self.view_changed.emit(view_key)
            return

        snap = self.camera_snapshot()
        if not snap:
            self._apply_camera_pose(end_pos, end_foc, end_up)
            if view_key:
                self.view_changed.emit(view_key)
            return
        start_pos = np.asarray(snap["position"], dtype=np.float64)
        start_foc = np.asarray(snap["focal"], dtype=np.float64)
        start_up = np.asarray(snap["up"], dtype=np.float64)
        su = float(np.linalg.norm(start_up))
        if su > 1e-12:
            start_up = start_up / su

        # Already there — skip animation
        if (
            float(np.linalg.norm(start_pos - end_pos)) < 1e-6
            and float(np.linalg.norm(start_foc - end_foc)) < 1e-6
            and float(np.dot(start_up, end_up)) > 0.9999
        ):
            self._apply_camera_pose(end_pos, end_foc, end_up)
            if view_key:
                self.view_changed.emit(view_key)
            return

        self._stop_camera_animation()
        interval = 16  # ~60 fps
        steps = max(1, int(round(float(duration_ms) / interval)))
        state = {
            "i": 0,
            "steps": steps,
            "start_pos": start_pos,
            "start_foc": start_foc,
            "start_up": start_up,
            "end_pos": end_pos,
            "end_foc": end_foc,
            "end_up": end_up,
            "view_key": view_key,
        }
        self._cam_anim = state
        timer = QTimer(self)
        timer.setInterval(interval)

        def _tick() -> None:
            st = self._cam_anim
            if st is None:
                timer.stop()
                return
            st["i"] += 1
            t = self._ease_out_cubic(st["i"] / float(st["steps"]))
            pos_i, foc_i, up_i = self.interpolate_camera_pose(
                st["start_pos"],
                st["start_foc"],
                st["start_up"],
                st["end_pos"],
                st["end_foc"],
                st["end_up"],
                t,
            )
            self._apply_camera_pose(pos_i, foc_i, up_i, render=True)
            if st["i"] >= st["steps"]:
                timer.stop()
                self._cam_anim_timer = None
                self._cam_anim = None
                self._apply_camera_pose(st["end_pos"], st["end_foc"], st["end_up"])
                if st["view_key"]:
                    self.view_changed.emit(st["view_key"])

        timer.timeout.connect(_tick)
        self._cam_anim_timer = timer
        timer.start()

    def set_view(self, name: str, *, animate: bool = True) -> None:
        if not self.plotter or self.in_sketch_mode:
            if self.in_sketch_mode:
                self.status_message.emit("Exit sketch to change standard views")
            return
        dist = self._view_distance()
        focus = self._scene_focus()
        fx, fy, fz = focus
        up = (0.0, 1.0, 0.0)
        key = (name or "iso").strip().lower()
        if key in ("front", "+z", "z"):
            pos = (fx, fy, fz + dist)
            key = "front"
        elif key in ("back", "-z"):
            pos = (fx, fy, fz - dist)
            key = "back"
        elif key in ("top", "+y", "y"):
            pos = (fx, fy + dist, fz)
            up = (0.0, 0.0, -1.0)
            key = "top"
        elif key in ("bottom", "-y"):
            pos = (fx, fy - dist, fz)
            up = (0.0, 0.0, 1.0)
            key = "bottom"
        elif key in ("right", "+x", "x"):
            pos = (fx + dist, fy, fz)
            key = "right"
        elif key in ("left", "-x"):
            pos = (fx - dist, fy, fz)
            key = "left"
        else:
            pos = (fx + dist * 0.75, fy + dist * 0.55, fz + dist * 0.75)
            key = "iso"
        if animate:
            self.animate_camera_to(pos, focus, up, duration_ms=280, view_key=key)
        else:
            self._stop_camera_animation()
            self._apply_camera_pose(pos, focus, up)
            self.view_changed.emit(key)
        self.status_message.emit(f"View: {key.capitalize()}")

    def view_along_axis(self, axis: str, *, reverse: bool = False) -> str:
        """Look straight down a world axis (perpendicular view).

        ``axis`` is 'x', 'y', or 'z'. Returns the standard view name applied.
        """
        a = (axis or "z").strip().lower()[:1]
        mapping = {
            "x": ("right", "left"),
            "y": ("top", "bottom"),
            "z": ("front", "back"),
        }
        if a not in mapping:
            a = "z"
        primary, opposite = mapping[a]
        name = opposite if reverse else primary
        self.set_view(name)
        return name

    def set_view_from_direction(self, direction, *, animate: bool = True) -> None:
        """Place camera along ``direction`` looking at scene focus (unit-ish)."""
        if not self.plotter:
            return
        d = np.asarray(direction, dtype=np.float64).reshape(3)
        n = float(np.linalg.norm(d))
        if n < 1e-12:
            self.set_view("iso", animate=animate)
            return
        d = d / n
        dist = self._view_distance()
        fx, fy, fz = self._scene_focus()
        pos = (fx + d[0] * dist, fy + d[1] * dist, fz + d[2] * dist)
        # Prefer world +Y as up unless looking along Y
        up = np.array([0.0, 1.0, 0.0])
        if abs(float(np.dot(d, up))) > 0.9:
            up = np.array([0.0, 0.0, -1.0 if d[1] > 0 else 1.0])
        if animate:
            self.animate_camera_to(
                pos, (fx, fy, fz), tuple(up), duration_ms=280, view_key="direction"
            )
        else:
            self._stop_camera_animation()
            self._apply_camera_pose(pos, (fx, fy, fz), tuple(up))
            self.view_changed.emit("direction")
        self.status_message.emit("View: custom")

    def orbit_camera(self, azimuth_deg: float, elevation_deg: float) -> None:
        """Orbit main camera by degrees (used by view-cube drag)."""
        if not self.plotter:
            return
        # Drag should interrupt a flying view transition
        self._stop_camera_animation()
        try:
            cam = self.plotter.camera
            cam.Azimuth(float(azimuth_deg))
            cam.Elevation(float(elevation_deg))
            cam.OrthogonalizeViewUp()
            self.plotter.reset_camera_clipping_range()
        except Exception as exc:  # noqa: BLE001
            print(f"[viewport] orbit: {exc}", file=sys.stderr)
            return
        if self._view_cube is not None:
            self._view_cube.sync_orientation()
        # interactive=True: paint during the drag, not only after mouse-up
        self._request_render(interactive=True)

    def fit_empty_workspace(self) -> None:
        """Public alias for opening-screen camera fit."""
        self._fit_empty_workspace()

    def _fit_empty_workspace(self) -> None:
        """Frame the empty plane workspace so it is readable (not a brown smear)."""
        if not self.plotter or self._has_display_solids():
            return
        h = float(self._plane_half) if self._plane_half > 0 else EMPTY_PLANE_HALF_MM
        pos, focus, up = empty_workspace_camera(h)
        self.plotter.camera_position = [pos, focus, up]
        try:
            self.plotter.reset_camera_clipping_range()
        except Exception:
            pass
        self._request_render()

    def try_corner_axes_click(self, display_x: float, display_y: float) -> Optional[str]:
        """If click is over the bottom-left triad, snap camera to that axis.

        Returns view name applied, or None if the click was outside the triad.
        Qt display coords: origin top-left, y down (logical pixels).

        Uses Qt widget size for hit-testing (not VTK window_size) so multi-monitor
        HiDPI screens where devicePixelRatio ≠ 1 still hit correctly.
        """
        if not self.plotter or self.in_sketch_mode:
            return None
        try:
            from app.display_coords import in_normalized_viewport, qt_to_normalized

            iw = self.plotter.interactor
            if not in_normalized_viewport(
                display_x, display_y, iw, self._axes_viewport
            ):
                return None
            nx, ny = qt_to_normalized(display_x, display_y, iw)
        except Exception:
            return None
        vx0, vy0, vx1, vy1 = self._axes_viewport  # VTK: bottom-left origin, y up
        # Local coords in triad pad (0..1)
        lx = (nx - vx0) / max(1e-9, vx1 - vx0)
        ly = (ny - vy0) / max(1e-9, vy1 - vy0)
        # Project unit axes into the current camera view plane and pick nearest
        try:
            cam = self.plotter.camera
            # Camera basis
            pos = np.asarray(cam.GetPosition(), float)
            foc = np.asarray(cam.GetFocalPoint(), float)
            up = np.asarray(cam.GetViewUp(), float)
            forward = foc - pos
            forward = forward / (np.linalg.norm(forward) + 1e-12)
            right = np.cross(forward, up)
            rn = np.linalg.norm(right)
            if rn < 1e-12:
                return None
            right = right / rn
            cup = np.cross(right, forward)
            cup = cup / (np.linalg.norm(cup) + 1e-12)
            # Click direction in pad space relative to center
            cx, cy = lx - 0.5, ly - 0.5
            if abs(cx) < 0.06 and abs(cy) < 0.06:
                # Center click → iso
                self.set_view("iso")
                return "iso"
            best_axis = "z"
            best_dot = -1e9
            for axis, vec in (
                ("x", np.array([1.0, 0.0, 0.0])),
                ("y", np.array([0.0, 1.0, 0.0])),
                ("z", np.array([0.0, 0.0, 1.0])),
            ):
                # Project axis into camera plane
                sx = float(np.dot(vec, right))
                sy = float(np.dot(vec, cup))
                sn = math.hypot(sx, sy)
                if sn < 1e-6:
                    continue
                sx, sy = sx / sn, sy / sn
                # Similarity to click vector
                cn = math.hypot(cx, cy)
                if cn < 1e-6:
                    continue
                dot = (sx * cx + sy * cy) / cn
                if dot > best_dot:
                    best_dot = dot
                    best_axis = axis
            if best_dot < 0.15:
                return None
            return self.view_along_axis(best_axis)
        except Exception as exc:  # noqa: BLE001
            print(f"[viewport] corner axes click: {exc}", file=sys.stderr)
            return None

    def zoom_to_fit(self) -> None:
        if not self.plotter:
            return
        if not self._has_display_solids():
            self._fit_empty_workspace()
            self.status_message.emit("Zoom to fit")
            return
        self.plotter.reset_camera()
        self._request_render()
        self.status_message.emit("Zoom to fit")

    # ----- inline length while drawing -----
    def _clear_length_buffer(self) -> None:
        self._length_buffer = ""

    def _emit_length_buffer_status(self) -> None:
        if not self._length_buffer:
            return
        unit = self._doc.display_unit if self._doc else None
        suf = unit.label if unit is not None else "mm"
        self.sketch_status.emit(f"Length: {self._length_buffer} {suf}  (Enter to commit)")

    def _accept_length_char(self, ch: str) -> bool:
        """Accumulate a typed char into the length buffer if drawing a line."""
        if self._sketch_ctrl is None:
            return False
        if self._sketch_ctrl.tool is not SketchTool.LINE:
            return False
        if not self._sketch_ctrl.is_drawing():
            return False
        # only after first endpoint
        if self._sketch_ctrl.draw is None or len(self._sketch_ctrl.draw.points) != 1:
            return False
        if ch == "." and "." in self._length_buffer:
            return True  # consume, ignore second dot
        if ch == "-" and self._length_buffer:
            return True  # no mid-number minus
        if ch == "-" and not self._length_buffer:
            return True  # ignore leading minus for length
        self._length_buffer += ch
        self._emit_length_buffer_status()
        return True

    def _try_commit_length_buffer(self) -> bool:
        """If buffer has a valid length, commit the line at that distance."""
        if not self._length_buffer or self._sketch_ctrl is None or self._doc is None:
            return False
        if self._sketch_ctrl.tool is not SketchTool.LINE:
            return False
        if not self._sketch_ctrl.is_drawing():
            return False
        try:
            from cadcore.units import to_mm

            display_val = float(self._length_buffer)
            length_mm = to_mm(display_val, self._doc.display_unit)
        except ValueError:
            self._clear_length_buffer()
            return False
        msg = self._sketch_ctrl.commit_line_length(length_mm)
        self._clear_length_buffer()
        if not msg:
            return False
        ent = self._sketch_ctrl.sketch.entities[-1]
        if self._sketch_feature_id >= 0:
            self._doc.record_entity_add(self._sketch_feature_id, ent)
        self._upsert_entity_actor(ent)
        self._clear_preview()
        self._update_handles_visual()
        self._update_junction_dots()
        self._update_dim_labels()
        self.sketch_status.emit(f"Sketch: {msg} ({display_val:g} {self._doc.display_unit.label})")
        self._request_render()
        return True
