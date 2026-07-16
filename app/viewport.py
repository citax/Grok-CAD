"""PyVista viewport: planes, sketches, async solids, software-GL friendly."""

from __future__ import annotations

import sys
import time
from typing import Dict, Optional, Set, Tuple

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


# Origin glyph size (world units / mm)
ORIGIN_CROSS_HALF = 0.14
ORIGIN_RING_R = 0.09
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
            )
            return True
        if et == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
            self.vp._sketch_mouse_release(
                event.position().x(),
                event.position().y(),
                shift=bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier),
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
        # No environment bounds box — keeps the scene uncluttered.
        # Flat crosshair + ring origin glyph (not a single GL point, not a sphere)
        self.plotter.add_mesh(
            _origin_glyph_polydata(),
            color=TEXT_PRIMARY,
            line_width=2.0,
            name="__origin",
            pickable=False,
        )
        # Neutral, faint world axes (not loud RGB) — low-opacity grey lines
        axis_color = GRID_COLOR
        for end, name in (
            ((2.4, 0, 0), "__ax"),
            ((0, 2.4, 0), "__ay"),
            ((0, 0, 2.4), "__az"),
        ):
            self.plotter.add_mesh(
                pv.Line((0, 0, 0), end),
                color=axis_color,
                line_width=1.5,
                name=name,
                pickable=False,
                opacity=0.35,
            )
        # Corner orientation triad — RGB axes + high-contrast caption labels.
        # PyVista 0.48 add_axes() has no label_color kwarg — set caption colours
        # on the vtkAxesActor after creation (do not swallow API errors).
        try:
            actor = self.plotter.add_axes(
                line_width=2,
                xlabel="X",
                ylabel="Y",
                zlabel="Z",
                x_color=AXIS_X,
                y_color=AXIS_Y,
                z_color=AXIS_Z,
                viewport=(0.0, 0.0, 0.18, 0.18),
            )
            self._style_axes_captions(actor)
        except Exception as exc:  # noqa: BLE001
            print(f"[viewport] add_axes: {exc}", file=sys.stderr)
            raise

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

    def _setup_interaction_lod(self) -> None:
        """Throttle VTK during camera drag; optional LOD / scaled render (perf methods)."""
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
        self._restyle_selection_only()
        self._request_render()
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
                _plane_surface(f.type), color=color, opacity=0.32, name=f"plane_{f.id}",
                pickable=True, smooth_shading=False, show_edges=False, render=False,
            )
            self.plotter.add_mesh(
                _plane_border(f.type), color=color, line_width=2, name=f"edge_{f.id}",
                pickable=False, render=False,
            )
        self._planes_built = True
        self._restyle_selection_only()
        self._request_render()

    def _request_render(self) -> None:
        if not self._ok:
            return
        # (e) aggressive coalesce during resize / interaction
        if _perf_enabled("e"):
            if self._resizing:
                interval = 120
            elif getattr(self, "_interacting", False):
                interval = 64
            else:
                interval = 24
        else:
            if self._resizing:
                interval = 80
            elif getattr(self, "_interacting", False):
                interval = 48
            else:
                interval = 16
        self._render_timer.setInterval(interval)
        if not self._render_timer.isActive():
            self._render_timer.start()

    def _do_render(self) -> None:
        if self.plotter:
            t0 = time.perf_counter()
            self.plotter.render()
            self._render_count += 1
            self._render_ms_sum += (time.perf_counter() - t0) * 1000.0

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
                    # High opacity + warm amber so the pick is unmistakable
                    prop.SetOpacity(0.78)
                    prop.SetColor(*sel_rgb)
                    prop.SetEdgeVisibility(1)
                    prop.SetEdgeColor(*sel_edge)
                    prop.SetLineWidth(3.0)
                else:
                    prop.SetOpacity(0.32)
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

        # Orient camera normal to plane + orthographic projection (true 2D sketch)
        fr = f.sketch.frame
        dist = 10.0
        pos = fr.origin + fr.normal * dist
        self.plotter.camera_position = [
            tuple(pos),
            tuple(fr.origin),
            tuple(fr.v_axis),
        ]
        self._set_parallel_projection(True)
        self._set_sketch_2d_chrome(True)
        self._ensure_sketch_overlay_layer()

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
        pname = plane.name if plane else "Plane"
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
        self._restyle_selection_only()
        self.refresh_sketches()
        self._request_render()
        self.sketch_exited.emit()

    def _set_sketch_2d_chrome(self, enabled: bool) -> None:
        """Toggle 2D sketch chrome: hide world 3D axes/solids clutter; flat lines only."""
        if not self.plotter:
            return
        # World XYZ helpers are 3D cues — hide while sketching in pure 2D
        for name in ("__ax", "__ay", "__az"):
            act = self._get_named_actor(name)
            if act is not None:
                try:
                    act.SetVisibility(0 if enabled else 1)
                except Exception:
                    pass
        # Corner orientation triad is 3D — hide in sketch
        try:
            for name, actor in list(self.plotter.actors.items()):
                if "Axes" in name or name.startswith("CubeAxes"):
                    try:
                        actor.SetVisibility(0 if enabled else 1)
                    except Exception:
                        pass
        except Exception:
            pass

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
        # Grid stays fixed in world (sketch units); only point snap is pixel-based
        self._sketch_ctrl.set_snap_world_tol(point_tol)

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
            "__sk_selbox",
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
        """Length label only for the selected (or hovered) line — reduces clutter."""
        if not self.plotter or self._sketch_ctrl is None or self._doc is None:
            return
        for name in list(getattr(self, "_dim_label_names", set()) or set()):
            self._remove_actor(name)
        self._dim_label_names = set()
        self._remove_actor("__sk_dims")
        unit = self._doc.display_unit
        fr = self._sketch_ctrl.sketch.frame
        ctrl = self._sketch_ctrl
        show_ids = set(ctrl.selected_ids)
        if ctrl.hover_handle is not None:
            show_ids.add(ctrl.hover_handle.entity_id)
        points = []
        labels = []
        for ent in ctrl.sketch.entities:
            if not isinstance(ent, LineEntity):
                continue
            if ent.id not in show_ids:
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
        """Label strings currently shown (selected/hovered lines only)."""
        if self._sketch_ctrl is None or self._doc is None:
            return []
        unit = self._doc.display_unit
        ctrl = self._sketch_ctrl
        show_ids = set(ctrl.selected_ids)
        if ctrl.hover_handle is not None:
            show_ids.add(ctrl.hover_handle.entity_id)
        out = []
        for ent in ctrl.sketch.entities:
            if isinstance(ent, LineEntity) and ent.id in show_ids:
                out.append(format_length(line_length(ent), unit))
        return out

    def labeled_entity_ids(self) -> list:
        """Entity ids that currently have a dimension label."""
        if self._sketch_ctrl is None:
            return []
        ctrl = self._sketch_ctrl
        ids = []
        for eid in ctrl.selected_ids:
            ent = ctrl.sketch.find_entity(eid)
            if isinstance(ent, LineEntity):
                ids.append(eid)
        if ctrl.hover_handle is not None:
            hid = ctrl.hover_handle.entity_id
            if hid not in ids:
                ent = ctrl.sketch.find_entity(hid)
                if isinstance(ent, LineEntity):
                    ids.append(hid)
        return ids

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

    def _sketch_mouse_move(self, x: float, y: float, *, shift: bool = False) -> None:
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

    def _sketch_mouse_press(self, x: float, y: float, *, shift: bool = False) -> None:
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
            uv, display_xy=(float(x), float(y)), shift=shift
        )
        sk = self._sketch_ctrl.sketch
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
        elif msg and msg.startswith("Selected"):
            self._end_draw_lod()
            self._clear_selbox()
            self._rebuild_all_sketch_entities()
            n = len(self._sketch_ctrl.selected_ids)
            self.sketch_status.emit(f"Sketch: Select ({n})" if n != 1 else "Sketch: Select")
        elif msg == "BoxSelect":
            self._begin_draw_lod()
            self._clear_selbox()
            self._rebuild_all_sketch_entities()  # clear highlight if selection wiped
            self.sketch_status.emit("Sketch: box select…")
        elif msg:
            # First click of a draw: enter light LOD; preview appears on next move
            if self._sketch_ctrl.is_drawing():
                self._begin_draw_lod()
                if self._sketch_ctrl.draw and self._sketch_ctrl.draw.points:
                    self._sketch_ctrl.preview_uv = self._sketch_ctrl.draw.points[0]
            self.sketch_status.emit(f"Sketch: {msg}")
        self._request_render()

    def _sketch_mouse_release(self, x: float, y: float, *, shift: bool = False) -> None:
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
            uv, display_xy=(float(x), float(y)), shift=shift
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
