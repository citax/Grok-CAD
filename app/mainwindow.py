"""Main application window — sketch-first SolidWorks-style workflow."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction, QActionGroup, QBrush, QColor, QCloseEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QSizePolicy,
    QToolBar,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.property_panel import PropertyPanel
from app.sketch_mode import SketchTool
from app.theme import (
    ACCENT,
    CURRENT_THEME,
    PLANE_FRONT,
    PLANE_RIGHT,
    PLANE_TOP,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    fa_icon,
    save_theme_preference,
)
from app.viewport import Viewport
from cadcore.document import (
    Document,
    FeatureType,
    feature_type_name,
    first_closed_profile,
    is_closed_profile,
    is_reference_plane,
    is_sketch_consuming_feature,
    is_solid_feature,
    resolve_profiles,
)
from cadcore.mesh import write_stl_binary
from cadcore.profiles import ClosedLineLoop, list_closed_profiles
from cadcore.project_io import (
    DEFAULT_EXTENSION,
    ProjectIOError,
    load_document,
    replace_document_contents,
    save_document,
)
from cadcore.constraints import (
    ConstraintKind,
    SketchConstraint,
    add_constraint,
    remove_constraints_for_entity,
)
from cadcore.sketch import (
    CircleEntity,
    LineEntity,
    RectEntity,
    apply_dimension_value,
    line_length,
    measure_dimension_value,
    set_line_length,
    snapshot_entity,
    snapshot_sketch_contents,
)
from cadcore.units import Unit, format_length, from_mm, parse_length, to_mm

PROJECT_FILTER = "Grok CAD project (*.gcad);;JSON (*.json);;All files (*)"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Grok CAD")
        self.resize(1280, 800)
        self.doc = Document()
        self.doc.seed_reference_planes()
        self.doc.mark_clean()
        # Path of the open project file, or None if never saved
        self._project_path: Optional[Path] = None

        self.viewport = Viewport(self)
        self.setCentralWidget(self.viewport)
        self.viewport.set_document(self.doc)

        self._sketch_tool_actions: Dict[SketchTool, QAction] = {}
        self._status_env = self._format_env_status()

        self._build_tree_dock()
        self._build_props_dock()
        self._build_menus()
        self._build_command_manager()  # SolidWorks-like ribbon (replaces flat main toolbar)
        self._build_heads_up_view_bar()  # floating view tools over viewport
        self._build_status_bar()

        self.viewport.feature_picked.connect(self._on_pick)
        self.viewport.status_message.connect(self._on_status)
        self.viewport.busy_changed.connect(self._on_busy)
        self.viewport.sketch_exited.connect(self._on_sketch_exited)
        self.viewport.sketch_status.connect(self._on_status)
        self.viewport.renderer_info.connect(self._on_renderer_info)
        self.viewport.dimension_requested.connect(self._on_dimension_requested)
        if getattr(self.viewport, "gl_renderer", ""):
            self._on_renderer_info(self.viewport.gl_renderer)

        # Opening screen: do NOT pre-select a plane — selection paints a heavy
        # amber fill and turns the empty workspace into an unreadable smear.
        self.doc.selected_id = -1
        self._refresh_tree()
        self._sync_selection(-1)
        self._set_ready_status()
        self._update_window_title()
        # Space must work even when focus is inside the VTK interactor
        self._space_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        self._space_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self._space_shortcut.activated.connect(self._on_space_bar)

    # ----- chrome -----
    def _format_env_status(self) -> str:
        platform = os.environ.get("QT_QPA_PLATFORM", "default")
        gl = getattr(self, "_gl_renderer", None) or "…"
        return f"{platform} · {gl}"

    def _build_status_bar(self) -> None:
        sb = self.statusBar()
        sb.setSizeGripEnabled(False)
        self._status_perm = QLabel("")
        self._status_perm.setObjectName("secondaryLabel")
        sb.addPermanentWidget(self._status_perm)
        self._update_perm_status()

    def _update_perm_status(self) -> None:
        self._status_env = self._format_env_status()
        self._status_perm.setText(self._status_env)

    def _set_ready_status(self) -> None:
        self.statusBar().showMessage(f"Ready · {self._status_env}")

    def _on_renderer_info(self, renderer: str) -> None:
        # Shorten e.g. "llvmpipe (LLVM …)" → "llvmpipe"
        short = renderer.split("(")[0].strip() or renderer
        if len(short) > 24:
            short = short[:24]
        self._gl_renderer = short
        self._update_perm_status()
        # Only refresh idle "Ready" message if nothing else is showing
        cur = self.statusBar().currentMessage()
        if not cur or cur.startswith("Ready"):
            self._set_ready_status()

    def _build_tree_dock(self) -> None:
        dock = QDockWidget("Feature Tree", self)
        dock.setObjectName("FeatureTreeDock")
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Features"])
        self.tree.setIndentation(16)
        self.tree.setRootIsDecorated(True)
        self.tree.setUniformRowHeights(True)
        self.tree.setIconSize(QSize(18, 18))
        self.tree.setAnimated(True)
        self.tree.itemSelectionChanged.connect(self._on_tree_sel)
        self.tree.itemDoubleClicked.connect(self._on_tree_double)
        dock.setWidget(self.tree)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def _build_props_dock(self) -> None:
        dock = QDockWidget("PropertyManager", self)
        dock.setObjectName("PropertiesDock")
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        # Compact SolidWorks-like width — do not let the panel dominate the window
        dock.setMinimumWidth(200)
        dock.setMaximumWidth(PropertyPanel.MAX_WIDTH + 16)
        self.props = PropertyPanel(self)
        self.props.set_document(self.doc)
        self.props.params_applied.connect(self._on_props_applied)
        self.props.status_message.connect(self._on_status)
        self.props.command_ok.connect(self._on_command_ok)
        self.props.command_cancel.connect(self._on_command_cancel)
        self.props.edit_sketch_requested.connect(self._on_edit_sketch_requested)
        dock.setWidget(self.props)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self._props_dock = dock
        # Prefer ~240px; user can still drag wider up to MAX_WIDTH
        self.resizeDocks([dock], [PropertyPanel.PREFERRED_WIDTH], Qt.Orientation.Horizontal)
        # Active feature command: {"kind": "extrude"|..., "sketch_id": int, "target_id": int}
        self._feature_cmd: Optional[dict] = None
        self._await_face_sketch: bool = False
        # Feature ids suppressed only for sketch edit roll-back (restored on exit)
        self._edit_rollback_ids: list = []

    def _on_props_applied(self, fid: int) -> None:
        """Rebuild after PropertyManager Apply (feature params or sketch length)."""
        if self.viewport.in_sketch_mode:
            self.viewport.sync_sketch_visuals()
        else:
            self.viewport.schedule_rebuild()
            self.viewport.refresh_sketches()
        self._refresh_tree()
        if fid >= 0 and self.doc.find(fid) is not None:
            self._sync_selection(fid)
        self._update_window_title()

    def _build_menus(self) -> None:
        file_m = self.menuBar().addMenu("&File")
        act_new = QAction(fa_icon("fa5s.file"), "&New", self)
        act_new.setShortcut(QKeySequence.StandardKey.New)
        act_new.setToolTip("Start a new empty project (Ctrl+N)")
        act_new.triggered.connect(self._file_new)
        file_m.addAction(act_new)
        act_open = QAction(fa_icon("fa5s.folder-open"), "&Open…", self)
        act_open.setShortcut(QKeySequence.StandardKey.Open)
        act_open.setToolTip("Open a Grok CAD project (Ctrl+O)")
        act_open.triggered.connect(self._file_open)
        file_m.addAction(act_open)
        file_m.addSeparator()
        act_save = QAction(fa_icon("fa5s.save"), "&Save", self)
        act_save.setShortcut(QKeySequence.StandardKey.Save)
        act_save.setToolTip("Save the current project (Ctrl+S)")
        act_save.triggered.connect(self._file_save)
        file_m.addAction(act_save)
        act_save_as = QAction(fa_icon("fa5s.save"), "Save &As…", self)
        act_save_as.setShortcut(QKeySequence.StandardKey.SaveAs)
        act_save_as.setToolTip("Save the project under a new name")
        act_save_as.triggered.connect(self._file_save_as)
        file_m.addAction(act_save_as)
        file_m.addSeparator()
        self.act_export_stl = QAction(
            fa_icon("fa5s.file-export", color=ACCENT), "Export STL…", self
        )
        self.act_export_stl.setShortcut(QKeySequence("Ctrl+E"))
        self.act_export_stl.setToolTip(
            "Export the selected solid feature (extrude/revolve) as binary STL"
        )
        self.act_export_stl.triggered.connect(self._export_stl)
        file_m.addAction(self.act_export_stl)
        file_m.addSeparator()
        act_exit = QAction(fa_icon("fa5s.sign-out-alt"), "E&xit", self)
        act_exit.setShortcut(QKeySequence.StandardKey.Quit)
        act_exit.triggered.connect(self.close)
        file_m.addAction(act_exit)

        edit_m = self.menuBar().addMenu("&Edit")
        self.act_undo = QAction(fa_icon("fa5s.undo"), "&Undo", self)
        self.act_undo.setShortcut(QKeySequence.StandardKey.Undo)  # Ctrl+Z
        self.act_undo.triggered.connect(self._undo)
        edit_m.addAction(self.act_undo)
        self.act_redo = QAction(fa_icon("fa5s.redo"), "&Redo", self)
        self.act_redo.setShortcuts(
            [QKeySequence.StandardKey.Redo, QKeySequence("Ctrl+Y"), QKeySequence("Ctrl+Shift+Z")]
        )
        self.act_redo.triggered.connect(self._redo)
        edit_m.addAction(self.act_redo)
        edit_m.addSeparator()
        self.act_cut = QAction(fa_icon("fa5s.cut"), "Cu&t", self)
        self.act_cut.setShortcut(QKeySequence.StandardKey.Cut)  # Ctrl+X
        self.act_cut.triggered.connect(self._cut)
        edit_m.addAction(self.act_cut)
        self.act_copy = QAction(fa_icon("fa5s.copy"), "&Copy", self)
        self.act_copy.setShortcut(QKeySequence.StandardKey.Copy)  # Ctrl+C
        self.act_copy.triggered.connect(self._copy)
        edit_m.addAction(self.act_copy)
        self.act_paste = QAction(fa_icon("fa5s.paste"), "&Paste", self)
        self.act_paste.setShortcut(QKeySequence.StandardKey.Paste)  # Ctrl+V
        self.act_paste.triggered.connect(self._paste)
        edit_m.addAction(self.act_paste)
        edit_m.addSeparator()
        self.act_set_length = QAction(fa_icon("fa5s.ruler"), "Set Line &Length…", self)
        self.act_set_length.setShortcut(QKeySequence("Ctrl+L"))
        self.act_set_length.triggered.connect(self._set_line_length)
        edit_m.addAction(self.act_set_length)
        edit_m.addSeparator()
        act_del = QAction(fa_icon("fa5s.trash-alt"), "Delete Feature", self)
        act_del.setShortcut(QKeySequence.StandardKey.Delete)
        act_del.triggered.connect(self._delete_selected)
        edit_m.addAction(act_del)

        settings_m = self.menuBar().addMenu("&Settings")
        unit_m = settings_m.addMenu("&Units")
        self._unit_group = QActionGroup(self)
        self._unit_group.setExclusive(True)
        self._unit_actions = {}
        for u, label in (
            (Unit.MM, "Millimetres (mm)"),
            (Unit.CM, "Centimetres (cm)"),
            (Unit.INCH, "Inches (in)"),
        ):
            act = QAction(label, self)
            act.setCheckable(True)
            act.setData(u.value)
            if u is Unit.MM:
                act.setChecked(True)
            act.triggered.connect(lambda checked=False, unit=u: self._set_unit(unit))
            self._unit_group.addAction(act)
            unit_m.addAction(act)
            self._unit_actions[u] = act
        settings_m.addSeparator()
        theme_m = settings_m.addMenu("&Theme")
        self._theme_group = QActionGroup(self)
        self._theme_group.setExclusive(True)
        for key, label in (("light", "Light (default)"), ("dark", "Dark")):
            act = QAction(label, self)
            act.setCheckable(True)
            act.setData(key)
            if key == CURRENT_THEME:
                act.setChecked(True)
            act.triggered.connect(
                lambda checked=False, k=key: self._request_theme(k)
            )
            self._theme_group.addAction(act)
            theme_m.addAction(act)
        tip = QAction(
            "Theme applies on restart (startup palette — avoids stale colour bindings)",
            self,
        )
        tip.setEnabled(False)
        theme_m.addAction(tip)

        view_m = self.menuBar().addMenu("&View")
        for label, key, icon_name in (
            ("Front", "front", "fa5s.square"),
            ("Back", "back", "fa5s.square"),
            ("Top", "top", "fa5s.border-all"),
            ("Bottom", "bottom", "fa5s.border-all"),
            ("Right", "right", "fa5s.cube"),
            ("Left", "left", "fa5s.cube"),
            ("Isometric", "iso", "fa5s.cubes"),
        ):
            act = QAction(fa_icon(icon_name), label, self)
            act.triggered.connect(lambda checked=False, k=key: self.viewport.set_view(k))
            view_m.addAction(act)
        view_m.addSeparator()
        act_fit = QAction(fa_icon("fa5s.expand"), "Zoom to Fit", self)
        act_fit.setShortcut(QKeySequence("Ctrl+F"))
        act_fit.triggered.connect(self.viewport.zoom_to_fit)
        view_m.addAction(act_fit)

        insert_m = self.menuBar().addMenu("&Insert")
        act = QAction(fa_icon("fa5s.pencil-ruler", color=ACCENT), "Sketch", self)
        act.setShortcut(QKeySequence("S"))
        act.triggered.connect(self._enter_sketch)
        insert_m.addAction(act)
        act_ex = QAction(fa_icon("fa5s.cube", color=ACCENT), "Extrude (Pad)…", self)
        act_ex.setShortcut(QKeySequence("E"))
        act_ex.triggered.connect(self._extrude)
        insert_m.addAction(act_ex)
        act_rev = QAction(fa_icon("fa5s.sync-alt", color=ACCENT), "Revolve…", self)
        act_rev.setShortcut(QKeySequence("R"))
        act_rev.triggered.connect(self._revolve)
        insert_m.addAction(act_rev)
        act_fil = QAction(fa_icon("fa5s.circle-notch", color=ACCENT), "Fillet…", self)
        act_fil.setShortcut(QKeySequence("F"))
        act_fil.triggered.connect(self._fillet)
        insert_m.addAction(act_fil)
        act_pok = QAction(fa_icon("fa5s.dot-circle", color=ACCENT), "Pocket…", self)
        act_pok.setShortcut(QKeySequence("P"))
        act_pok.triggered.connect(self._pocket)
        insert_m.addAction(act_pok)
        act_cut = QAction(fa_icon("fa5s.cut", color=ACCENT), "Cut-Extrude…", self)
        act_cut.setShortcut(QKeySequence("C"))
        act_cut.triggered.connect(self._cut_extrude)
        insert_m.addAction(act_cut)

    def _ribbon_button(self, act: QAction) -> QToolButton:
        """Compact command icon with label under — section strip style."""
        btn = QToolButton()
        btn.setDefaultAction(act)
        btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        btn.setIconSize(QSize(22, 22))
        btn.setAutoRaise(True)
        btn.setObjectName("CmdStripButton")
        btn.setMinimumSize(QSize(52, 48))
        return btn

    @staticmethod
    def _cmd_separator() -> QWidget:
        from PySide6.QtWidgets import QFrame

        sep = QFrame()
        sep.setObjectName("CmdGroupSep")
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFrameShadow(QFrame.Shadow.Plain)
        sep.setFixedWidth(1)
        return sep

    def _make_cmd_section(self, title: str, actions: list) -> QWidget:
        """One labelled section of the always-visible command strip."""
        from PySide6.QtWidgets import QFrame

        box = QWidget()
        box.setObjectName("CmdSection")
        col = QVBoxLayout(box)
        col.setContentsMargins(8, 4, 8, 4)
        col.setSpacing(2)
        head = QLabel(title)
        head.setObjectName("CmdSectionTitle")
        head.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        col.addWidget(head)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)
        for act in actions:
            row.addWidget(self._ribbon_button(act))
        row.addStretch(0)
        col.addLayout(row)
        return box

    def _build_command_manager(self) -> None:
        """Single always-visible strip: Features | Sketch | Evaluate sections."""
        bar = QToolBar("CommandManager")
        bar.setObjectName("CommandManagerBar")
        bar.setMovable(False)
        bar.setFloatable(False)
        bar.setIconSize(QSize(22, 22))
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, bar)

        # No tabs — one strip. Keep cmd_tabs alias None for old call sites.
        self.cmd_tabs = None  # type: ignore[assignment]
        self._sketch_tab_index = -1

        # --- Actions (shared) ---
        self.act_sketch = QAction(
            fa_icon("fa5s.pencil-ruler", color=ACCENT), "Sketch", self
        )
        self.act_sketch.setToolTip(
            "Sketch on the selected reference plane or solid face (S)"
        )
        self.act_sketch.setShortcut(QKeySequence("S"))
        self.act_sketch.triggered.connect(self._enter_sketch)

        self.act_extrude = QAction(fa_icon("fa5s.cube", color=ACCENT), "Extrude", self)
        self.act_extrude.setToolTip(
            "Extrude (pad) a closed sketch profile into a solid (E)"
        )
        self.act_extrude.setShortcut(QKeySequence("E"))
        self.act_extrude.triggered.connect(self._extrude)

        self.act_revolve = QAction(fa_icon("fa5s.sync-alt", color=ACCENT), "Revolve", self)
        self.act_revolve.setToolTip(
            "Revolve a closed sketch profile about the V-axis into a solid (R)"
        )
        self.act_revolve.setShortcut(QKeySequence("R"))
        self.act_revolve.triggered.connect(self._revolve)

        self.act_fillet = QAction(
            fa_icon("fa5s.circle-notch", color=ACCENT), "Fillet", self
        )
        self.act_fillet.setToolTip(
            "Round edges on a solid — select edges, set radius, OK (F)"
        )
        self.act_fillet.setShortcut(QKeySequence("F"))
        self.act_fillet.triggered.connect(self._fillet)

        self.act_chamfer = QAction(
            fa_icon("fa5s.cut", color=ACCENT), "Chamfer", self
        )
        self.act_chamfer.setToolTip(
            "Chamfer edges on a solid — select edges, set distance, OK"
        )
        self.act_chamfer.triggered.connect(self._chamfer)

        self.act_lpattern = QAction(
            fa_icon("fa5s.th", color=ACCENT), "L-Pattern", self
        )
        self.act_lpattern.setToolTip(
            "Linear pattern of a solid along X/Y/Z spacing"
        )
        self.act_lpattern.triggered.connect(self._linear_pattern)

        self.act_cpattern = QAction(
            fa_icon("fa5s.dharmachakra", color=ACCENT), "C-Pattern", self
        )
        self.act_cpattern.setToolTip(
            "Circular pattern of a solid about Z (world up)"
        )
        self.act_cpattern.triggered.connect(self._circular_pattern)

        self.act_mirror = QAction(
            fa_icon("fa5s.adjust", color=ACCENT), "Mirror", self
        )
        self.act_mirror.setToolTip(
            "Mirror a solid about a reference plane"
        )
        self.act_mirror.triggered.connect(self._mirror_feature)

        self.act_offset_plane = QAction(
            fa_icon("fa5s.clone", color=ACCENT), "Plane", self
        )
        self.act_offset_plane.setToolTip(
            "Create an offset reference plane from the selected plane"
        )
        self.act_offset_plane.triggered.connect(self._offset_plane)

        self.act_pocket = QAction(
            fa_icon("fa5s.dot-circle", color=ACCENT), "Pocket", self
        )
        self.act_pocket.setToolTip(
            "Cut a circular through-hole pocket and extrude into a solid (P)"
        )
        self.act_pocket.setShortcut(QKeySequence("P"))
        self.act_pocket.triggered.connect(self._pocket)

        self.act_cut_extrude = QAction(
            fa_icon("fa5s.cut", color=ACCENT), "Cut", self
        )
        self.act_cut_extrude.setToolTip(
            "Extruded Cut — remove material under a sketch on a solid (C)"
        )
        self.act_cut_extrude.setShortcut(QKeySequence("C"))
        self.act_cut_extrude.triggered.connect(self._cut_extrude)

        group = QActionGroup(self)
        group.setExclusive(True)
        tool_defs = (
            (SketchTool.SELECT, "Select", "fa5s.mouse-pointer", "Select and edit entities"),
            (SketchTool.LINE, "Line", "fa5s.minus", "Draw a line"),
            (SketchTool.RECTANGLE, "Rectangle", "fa5s.vector-square", "Draw a rectangle"),
            (SketchTool.CIRCLE, "Circle", "fa5s.circle", "Draw a circle"),
            (
                SketchTool.ARC,
                "Arc",
                "fa5s.circle-notch",
                "Draw an arc (start → on-arc → end)",
            ),
            (
                SketchTool.SPLINE,
                "Spline",
                "fa5s.bezier-curve",
                "Draw a cubic spline (points, double-click last to finish)",
            ),
            (
                SketchTool.DIMENSION,
                "Smart Dim",
                "fa5s.ruler-combined",
                "Driving dimension — click entity, type size (D)",
            ),
            (SketchTool.TRIM, "Trim", "fa5s.cut", "Trim a line at the click"),
            (SketchTool.EXTEND, "Extend", "fa5s.expand", "Extend a line toward the click"),
            (
                SketchTool.OFFSET,
                "Offset",
                "fa5s.copy",
                "Offset a line/circle by 5 mm",
            ),
        )
        sketch_tool_actions: list = []
        for tool, label, icon_name, tip in tool_defs:
            act = QAction(fa_icon(icon_name), label, self)
            act.setToolTip(tip)
            act.setCheckable(True)
            if tool is SketchTool.DIMENSION:
                act.setShortcut(QKeySequence("D"))
            act.triggered.connect(lambda checked=False, t=tool: self._on_sketch_tool(t))
            group.addAction(act)
            self._sketch_tool_actions[tool] = act
            sketch_tool_actions.append(act)

        self.act_horiz = QAction(fa_icon("fa5s.arrows-alt-h"), "Horizontal", self)
        self.act_horiz.setToolTip("Horizontal — selected line(s) stay horizontal")
        self.act_horiz.setShortcut(QKeySequence("H"))
        self.act_horiz.triggered.connect(self._make_horizontal)
        self.act_vert = QAction(fa_icon("fa5s.arrows-alt-v"), "Vertical", self)
        self.act_vert.setToolTip("Vertical — selected line(s) stay vertical")
        self.act_vert.setShortcut(QKeySequence("V"))
        self.act_vert.triggered.connect(self._make_vertical)
        self.act_equal = QAction(fa_icon("fa5s.equals"), "Equal", self)
        self.act_equal.setToolTip("Equal — selected lines keep the same length")
        self.act_equal.setShortcut(QKeySequence("="))
        self.act_equal.triggered.connect(self._make_equal)
        self.act_parallel = QAction(fa_icon("fa5s.grip-lines"), "Parallel", self)
        self.act_parallel.setToolTip("Parallel — two selected lines stay parallel")
        self.act_parallel.triggered.connect(self._make_parallel)
        self.act_perp = QAction(fa_icon("fa5s.plus"), "Perpendicular", self)
        self.act_perp.setToolTip("Perpendicular — two selected lines stay at 90°")
        self.act_perp.triggered.connect(self._make_perpendicular)
        self.act_coincident = QAction(fa_icon("fa5s.dot-circle"), "Coincident", self)
        self.act_coincident.setToolTip(
            "Coincident — stick two selected line endpoints together"
        )
        self.act_coincident.triggered.connect(self._make_coincident)
        self.act_fix = QAction(fa_icon("fa5s.anchor"), "Fix", self)
        self.act_fix.setToolTip("Fix — lock selected line endpoint(s) in place")
        self.act_fix.triggered.connect(self._make_fix)
        self.act_del_cstr = QAction(fa_icon("fa5s.unlink"), "Remove constraint", self)
        self.act_del_cstr.setToolTip(
            "Remove constraints on the selected sketch entities"
        )
        self.act_del_cstr.triggered.connect(self._remove_constraints)
        self.act_angle = QAction(fa_icon("fa5s.drafting-compass"), "Angle", self)
        self.act_angle.setToolTip(
            "Angle dimension — select two lines, type degrees (persists on drag)"
        )
        self.act_angle.triggered.connect(self._make_angle_dimension)
        self.act_radius = QAction(fa_icon("fa5s.ruler"), "Radius", self)
        self.act_radius.setToolTip(
            "Radius dimension — select an arc, type radius (ends stay put)"
        )
        self.act_radius.triggered.connect(self._make_radius_dimension)
        self.act_construction = QAction(fa_icon("fa5s.slash"), "Construction", self)
        self.act_construction.setToolTip(
            "Toggle construction (centerline) on selected entities"
        )
        self.act_construction.triggered.connect(self._toggle_construction)
        self.act_midpoint = QAction(fa_icon("fa5s.arrows-alt-h"), "Midpoint", self)
        self.act_midpoint.setToolTip(
            "Midpoint — selected point handle stays at mid of first selected line"
        )
        self.act_midpoint.triggered.connect(self._make_midpoint)
        self.act_concentric = QAction(fa_icon("fa5s.bullseye"), "Concentric", self)
        self.act_concentric.setToolTip("Concentric — two circles/arcs share a center")
        self.act_concentric.triggered.connect(self._make_concentric)
        self.act_collinear = QAction(fa_icon("fa5s.grip-lines-vertical"), "Collinear", self)
        self.act_collinear.setToolTip("Collinear — two selected lines stay collinear")
        self.act_collinear.triggered.connect(self._make_collinear)
        self.act_symmetric = QAction(fa_icon("fa5s.balance-scale"), "Symmetric", self)
        self.act_symmetric.setToolTip(
            "Symmetric — subject line endpoints about first selected mirror line"
        )
        self.act_symmetric.triggered.connect(self._make_symmetric)
        self.act_equal_r = QAction(fa_icon("fa5s.circle"), "Equal R", self)
        self.act_equal_r.setToolTip("Equal radius — two circles/arcs keep same radius")
        self.act_equal_r.triggered.connect(self._make_equal_radius)
        self.act_convert = QAction(fa_icon("fa5s.project-diagram"), "Convert", self)
        self.act_convert.setToolTip(
            "Convert face edges into sketch lines (construction)"
        )
        self.act_convert.triggered.connect(self._convert_entities)
        self.act_exit_sketch = QAction(
            fa_icon("fa5s.times", color=PLANE_RIGHT), "Exit", self
        )
        self.act_exit_sketch.setToolTip("Exit sketch mode (Esc when idle)")
        self.act_exit_sketch.triggered.connect(self._exit_sketch)

        # Full-width host strip
        host = QWidget()
        host.setObjectName("CommandManagerHost")
        strip = QHBoxLayout(host)
        strip.setContentsMargins(4, 2, 4, 2)
        strip.setSpacing(0)

        strip.addWidget(
            self._make_cmd_section(
                "Features",
                [
                    self.act_sketch,
                    self.act_extrude,
                    self.act_cut_extrude,
                    self.act_revolve,
                    self.act_fillet,
                    self.act_chamfer,
                    self.act_pocket,
                    self.act_lpattern,
                    self.act_cpattern,
                    self.act_mirror,
                    self.act_offset_plane,
                ],
            )
        )
        strip.addWidget(self._cmd_separator())
        strip.addWidget(
            self._make_cmd_section(
                "Sketch",
                sketch_tool_actions
                + [
                    self.act_horiz,
                    self.act_vert,
                    self.act_equal,
                    self.act_parallel,
                    self.act_perp,
                    self.act_coincident,
                    self.act_fix,
                    self.act_midpoint,
                    self.act_concentric,
                    self.act_collinear,
                    self.act_symmetric,
                    self.act_equal_r,
                    self.act_del_cstr,
                    self.act_angle,
                    self.act_radius,
                    self.act_construction,
                    self.act_convert,
                    self.act_exit_sketch,
                ],
            )
        )
        strip.addWidget(self._cmd_separator())
        strip.addWidget(self._make_cmd_section("Evaluate", [self.act_export_stl]))
        strip.addStretch(1)

        host.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        bar.addWidget(host)
        self.sketch_tb = bar  # type: ignore[assignment]
        self._cmd_strip = host
        self._set_sketch_ribbon_enabled(False)

    def _set_sketch_ribbon_enabled(self, on: bool) -> None:
        for act in self._sketch_tool_actions.values():
            act.setEnabled(on)
        for name in (
            "act_exit_sketch",
            "act_horiz",
            "act_vert",
            "act_equal",
            "act_parallel",
            "act_perp",
            "act_coincident",
            "act_fix",
            "act_del_cstr",
            "act_angle",
            "act_radius",
            "act_construction",
            "act_midpoint",
            "act_concentric",
            "act_collinear",
            "act_symmetric",
            "act_equal_r",
            "act_convert",
        ):
            if hasattr(self, name):
                getattr(self, name).setEnabled(on)

    def _build_heads_up_view_bar(self) -> None:
        """Floating SolidWorks-style heads-up view tools over the viewport."""
        self._hud = QWidget(self.viewport)
        self._hud.setObjectName("HeadsUpViewBar")
        row = QHBoxLayout(self._hud)
        row.setContentsMargins(6, 4, 6, 4)
        row.setSpacing(2)
        view_defs = (
            ("Front", "front", "fa5s.square", "Front view"),
            ("Top", "top", "fa5s.border-all", "Top view"),
            ("Right", "right", "fa5s.cube", "Right view"),
            ("Iso", "iso", "fa5s.cubes", "Isometric view"),
            ("Fit", "fit", "fa5s.expand", "Zoom to fit (Ctrl+F)"),
        )
        for label, key, icon_name, tip in view_defs:
            act = QAction(fa_icon(icon_name), label, self)
            act.setToolTip(tip)
            if key == "fit":
                act.triggered.connect(self.viewport.zoom_to_fit)
            else:
                act.triggered.connect(
                    lambda checked=False, k=key: self.viewport.set_view(k)
                )
            btn = QToolButton(self._hud)
            btn.setDefaultAction(act)
            btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
            btn.setIconSize(QSize(18, 18))
            btn.setAutoRaise(True)
            row.addWidget(btn)
        self._hud.adjustSize()
        self._position_heads_up()
        self._hud.raise_()
        self._hud.show()
        self.viewport.installEventFilter(self)

    def _position_heads_up(self) -> None:
        if not hasattr(self, "_hud") or self._hud is None:
            return
        vp = self.viewport
        self._hud.adjustSize()
        x = max(8, (vp.width() - self._hud.width()) // 2)
        self._hud.move(x, 8)

    def eventFilter(self, obj, event):  # noqa: N802
        from PySide6.QtCore import QEvent

        if obj is self.viewport and event.type() == QEvent.Type.Resize:
            self._position_heads_up()
        return super().eventFilter(obj, event)

    def _request_theme(self, name: str) -> None:
        save_theme_preference(name)
        QMessageBox.information(
            self,
            "Theme",
            f"Theme “{name}” will apply the next time you start Grok CAD.\n\n"
            "Startup-only theming avoids half-updated colours from import-time "
            "bindings (restart required).",
        )

    def _on_sketch_tool(self, tool: SketchTool) -> None:
        self.viewport.set_sketch_tool(tool)
        self._sync_sketch_tool_ui(tool)

    def _sync_sketch_tool_ui(self, tool: SketchTool) -> None:
        act = self._sketch_tool_actions.get(tool)
        if act is not None and not act.isChecked():
            act.setChecked(True)
        # Refresh icons: checked tools get white glyphs for contrast on accent
        icon_names = {
            SketchTool.SELECT: "fa5s.mouse-pointer",
            SketchTool.LINE: "fa5s.minus",
            SketchTool.RECTANGLE: "fa5s.vector-square",
            SketchTool.CIRCLE: "fa5s.circle",
            SketchTool.ARC: "fa5s.circle-notch",
            SketchTool.SPLINE: "fa5s.bezier-curve",
            SketchTool.DIMENSION: "fa5s.ruler-combined",
            SketchTool.TRIM: "fa5s.cut",
            SketchTool.EXTEND: "fa5s.expand",
            SketchTool.OFFSET: "fa5s.copy",
        }
        for t, a in self._sketch_tool_actions.items():
            name = icon_names.get(t, "fa5s.pencil-alt")
            col = "#ffffff" if a.isChecked() else TEXT_PRIMARY
            a.setIcon(fa_icon(name, color=col))

    def _icon_for_feature(self, ftype: FeatureType):
        if ftype is FeatureType.PLANE_FRONT:
            return fa_icon("fa5s.clone", color=PLANE_FRONT)
        if ftype is FeatureType.PLANE_TOP:
            return fa_icon("fa5s.clone", color=PLANE_TOP)
        if ftype is FeatureType.PLANE_RIGHT:
            return fa_icon("fa5s.clone", color=PLANE_RIGHT)
        if ftype is FeatureType.SKETCH:
            return fa_icon("fa5s.pencil-alt", color=ACCENT)
        if ftype is FeatureType.EXTRUDE:
            return fa_icon("fa5s.cube", color=ACCENT)
        if ftype is FeatureType.REVOLVE:
            return fa_icon("fa5s.sync-alt", color=ACCENT)
        if ftype is FeatureType.FILLET:
            return fa_icon("fa5s.circle-notch", color=ACCENT)
        if ftype is FeatureType.POCKET:
            return fa_icon("fa5s.dot-circle", color=ACCENT)
        if ftype is FeatureType.CUT_EXTRUDE:
            return fa_icon("fa5s.cut", color=ACCENT)
        return fa_icon("fa5s.cube", color=TEXT_SECONDARY)

    def _refresh_tree(self) -> None:
        """SolidWorks-style tree: sketches absorbed under the feature that uses them."""
        self.tree.blockSignals(True)
        self.tree.clear()

        section_fg = QBrush(QColor(TEXT_SECONDARY))
        planes_root = QTreeWidgetItem(["Reference Planes"])
        planes_root.setFlags(Qt.ItemFlag.ItemIsEnabled)
        planes_root.setIcon(0, fa_icon("fa5s.layer-group", color=TEXT_SECONDARY))
        planes_root.setForeground(0, section_fg)
        self.tree.addTopLevelItem(planes_root)

        features_root = QTreeWidgetItem(["Features"])
        features_root.setFlags(Qt.ItemFlag.ItemIsEnabled)
        features_root.setIcon(0, fa_icon("fa5s.project-diagram", color=TEXT_SECONDARY))
        features_root.setForeground(0, section_fg)
        self.tree.addTopLevelItem(features_root)

        absorbed = self.doc.absorbed_sketch_map()  # sketch_id -> consumer_id
        # Reverse: consumer -> list of absorbed sketches (preserve feature order)
        children: Dict[int, list] = {}
        for sk_id, cons_id in absorbed.items():
            children.setdefault(cons_id, []).append(sk_id)

        def _make_item(f) -> QTreeWidgetItem:
            item = QTreeWidgetItem([f.name])
            item.setData(0, Qt.ItemDataRole.UserRole, f.id)
            item.setIcon(0, self._icon_for_feature(f.type))
            item.setFlags(
                Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            )
            return item

        def _add_sketch_dim_children(parent: QTreeWidgetItem, skf) -> None:
            """Nest driving dimensions under a sketch in the feature tree."""
            sk = getattr(skf, "sketch", None)
            if sk is None:
                return
            unit = self.doc.display_unit
            for dim in getattr(sk, "dimensions", None) or []:
                role = str(dim.role)
                val = float(dim.value_mm)
                if role == "angle":
                    label = f"∠ {val:g}°"
                elif role == "diameter":
                    label = f"⌀ {format_length(val, unit)}"
                else:
                    label = f"{role.title()} {format_length(val, unit)}"
                ditem = QTreeWidgetItem([label])
                ditem.setData(0, Qt.ItemDataRole.UserRole, -100000 - int(dim.id))
                ditem.setIcon(0, fa_icon("fa5s.ruler", color=ACCENT))
                ditem.setFlags(Qt.ItemFlag.ItemIsEnabled)
                parent.addChild(ditem)

        for f in self.doc.features:
            if is_reference_plane(f.type):
                item = _make_item(f)
                planes_root.addChild(item)
                if f.id == self.doc.selected_id:
                    item.setSelected(True)
                    self.tree.setCurrentItem(item)
                continue
            # Absorbed sketches are not top-level — nested under their feature
            if f.type is FeatureType.SKETCH and f.id in absorbed:
                continue
            item = _make_item(f)
            features_root.addChild(item)
            if f.type is FeatureType.SKETCH:
                _add_sketch_dim_children(item, f)
            # Nest absorbed sketch(es) under consuming feature
            for sk_id in children.get(f.id, ()):
                skf = self.doc.find(sk_id)
                if skf is None:
                    continue
                child = _make_item(skf)
                item.addChild(child)
                _add_sketch_dim_children(child, skf)
                if sk_id == self.doc.selected_id:
                    child.setSelected(True)
                    self.tree.setCurrentItem(child)
                    item.setExpanded(True)
            if f.id == self.doc.selected_id:
                item.setSelected(True)
                self.tree.setCurrentItem(item)

        planes_root.setExpanded(True)
        features_root.setExpanded(True)
        self.tree.blockSignals(False)

    def _sync_selection(self, fid: int) -> None:
        if self.viewport.in_sketch_mode:
            return  # don't fight sketch selection
        # During an active feature command, selection feeds the command
        if self._feature_cmd is not None:
            self.doc.selected_id = fid
            self.viewport.set_selected_id(fid)
            self._update_command_from_selection(fid)
            self._sync_tree_highlight(fid)
            return
        self.doc.selected_id = fid
        self.viewport.set_selected_id(fid)
        f = self.doc.find(fid)
        self.props.set_document(self.doc)
        if f is None or fid < 0:
            self.props.show_empty()
            self.statusBar().showMessage(f"Selected: (none) · {self._status_env}")
            self._sync_tree_highlight(-1)
            return
        self.props.show_feature(f, unit=self.doc.display_unit)
        self.statusBar().showMessage(f"Selected: {f.name} · {self._status_env}")
        self._sync_tree_highlight(fid)

    def _sync_tree_highlight(self, fid: int) -> None:
        self.tree.blockSignals(True)
        self.tree.clearSelection()
        if fid < 0:
            self.tree.blockSignals(False)
            return
        for i in range(self.tree.topLevelItemCount()):
            root = self.tree.topLevelItem(i)
            for j in range(root.childCount()):
                item = root.child(j)
                if item.data(0, Qt.ItemDataRole.UserRole) == fid:
                    item.setSelected(True)
                    self.tree.setCurrentItem(item)
                for k in range(item.childCount()):
                    ch = item.child(k)
                    if ch.data(0, Qt.ItemDataRole.UserRole) == fid:
                        item.setExpanded(True)
                        ch.setSelected(True)
                        self.tree.setCurrentItem(ch)
        self.tree.blockSignals(False)

    def _on_tree_sel(self) -> None:
        if self.viewport.in_sketch_mode:
            return
        items = self.tree.selectedItems()
        if not items:
            if self._feature_cmd is None:
                self._sync_selection(-1)
            return
        data = items[0].data(0, Qt.ItemDataRole.UserRole)
        if data is None:
            return
        fid = int(data)
        self._sync_selection(fid)

    def _on_tree_double(self, item: QTreeWidgetItem, _col: int) -> None:
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data is None:
            return
        fid = int(data)
        f = self.doc.find(fid)
        if f is None:
            return
        # Sketch node → re-enter edit
        if f.type is FeatureType.SKETCH and f.sketch is not None:
            self._open_sketch_edit(f.id)
            return
        # Solid that consumes a sketch → Edit Sketch (SolidWorks double-click)
        from cadcore.document import is_sketch_consuming_feature

        if is_sketch_consuming_feature(f.type) and int(f.operand_a) >= 0:
            sk = self.doc.find(int(f.operand_a))
            if sk is not None and sk.type is FeatureType.SKETCH and sk.sketch is not None:
                self._open_sketch_edit(sk.id, rollback_from=f.id)
                return
        # Other solids → focus PropertyManager for param edit
        self._sync_selection(fid)
        self.statusBar().showMessage(
            f"Selected {f.name} — edit parameters in PropertyManager, then Apply",
            3500,
        )

    def _open_sketch_edit(
        self, sketch_id: int, *, rollback_from: Optional[int] = None
    ) -> None:
        """Enter sketch mode; optionally roll back (suppress) later features."""
        skf = self.doc.find(int(sketch_id))
        if skf is None or skf.type is not FeatureType.SKETCH or skf.sketch is None:
            return
        # Clear any previous temporary suppress from a prior edit session
        self._clear_edit_rollback()
        consumer_id = rollback_from
        if consumer_id is None:
            consumer_id = self.doc.absorbed_sketch_map().get(int(sketch_id))
        if consumer_id is not None:
            self._apply_edit_rollback(int(consumer_id))
        self._sync_selection(int(sketch_id))
        self.viewport.enter_sketch(int(sketch_id))
        self._set_sketch_ribbon_enabled(True)
        self._sync_sketch_tool_ui(SketchTool.LINE)
        msg = f"Editing {skf.name}"
        if consumer_id is not None:
            cf = self.doc.find(int(consumer_id))
            msg += f" (rolled back at {cf.name if cf else consumer_id})"
        self.statusBar().showMessage(msg)

    def _apply_edit_rollback(self, from_feature_id: int) -> None:
        """Suppress ``from_feature_id`` and every feature after it (edit-time roll back)."""
        ids = [f.id for f in self.doc.features]
        try:
            idx = ids.index(int(from_feature_id))
        except ValueError:
            return
        suppressed: list[int] = []
        for f in self.doc.features[idx:]:
            from cadcore.document import is_reference_plane

            if is_reference_plane(f.type) or f.type is FeatureType.SKETCH:
                continue
            if not f.suppressed:
                f.suppressed = True
                suppressed.append(int(f.id))
        self._edit_rollback_ids = suppressed
        if suppressed:
            self.viewport.schedule_rebuild()
            self.viewport.refresh_sketches()
            self._refresh_tree()

    def _clear_edit_rollback(self) -> None:
        """Restore features suppressed only for sketch edit (not user suppress)."""
        ids = getattr(self, "_edit_rollback_ids", None) or []
        if not ids:
            self._edit_rollback_ids = []
            return
        for fid in ids:
            f = self.doc.find(int(fid))
            if f is not None:
                f.suppressed = False
        self._edit_rollback_ids = []

    def _on_pick(self, fid: int) -> None:
        if self.viewport.in_sketch_mode:
            return
        # Sketch-awaiting-face: solid face click starts a sketch
        if getattr(self, "_await_face_sketch", False):
            frame = self.viewport.face_pick_frame()
            sid = self.viewport.face_pick_solid_id()
            if frame is not None and sid == fid and is_solid_feature(
                self.doc.find(fid).type if self.doc.find(fid) else FeatureType.SKETCH
            ):
                self._await_face_sketch = False
                skf = self.doc.create_sketch_on_face(fid, frame)
                if skf is not None:
                    self._refresh_tree()
                    self._sync_selection(skf.id)
                    self._update_window_title()
                    self.viewport.enter_sketch(skf.id)
                    self._set_sketch_ribbon_enabled(True)
                    self._sync_sketch_tool_ui(SketchTool.LINE)
                    self.statusBar().showMessage(f"Editing {skf.name} on face")
                    return
            self.statusBar().showMessage(
                "Click a face of the solid to start a sketch (Esc cancels)", 4000
            )
            self._sync_selection(fid)
            return
        self._sync_selection(fid)

    def _on_status(self, msg: str) -> None:
        self.statusBar().showMessage(msg, 4000)

    def _on_busy(self, busy: bool, message: str) -> None:
        if busy:
            self.statusBar().showMessage(message or "Working…")
        elif message:
            self.statusBar().showMessage(message, 2500)

    def _enter_sketch(self) -> None:
        if self.viewport.in_sketch_mode:
            return
        if self._feature_cmd is not None:
            self._cancel_feature_cmd()
        f = self.doc.find(self.doc.selected_id)
        # Re-open existing sketch
        if f is not None and f.type is FeatureType.SKETCH and f.sketch is not None:
            self.viewport.enter_sketch(f.id)
            self._set_sketch_ribbon_enabled(True)
            self._sync_sketch_tool_ui(SketchTool.LINE)
            self.statusBar().showMessage(f"Editing {f.name}")
            return
        # SolidWorks: select face of solid → Sketch sits on that face
        if f is not None and is_solid_feature(f.type):
            frame = self.viewport.face_pick_frame()
            sid = self.viewport.face_pick_solid_id()
            # Accept face pick on this solid, or any displayed body that maps to it
            if frame is not None and (sid == f.id or sid >= 0):
                solid_id = sid if sid >= 0 else f.id
                if self.doc.find(solid_id) is not None and is_solid_feature(
                    self.doc.find(solid_id).type
                ):
                    skf = self.doc.create_sketch_on_face(solid_id, frame)
                    if skf is not None:
                        self._await_face_sketch = False
                        self._refresh_tree()
                        self._sync_selection(skf.id)
                        self._update_window_title()
                        self.viewport.enter_sketch(skf.id)
                        self._set_sketch_ribbon_enabled(True)
                        self._sync_sketch_tool_ui(SketchTool.LINE)
                        self.statusBar().showMessage(
                            f"Editing {skf.name} on face of {self.doc.find(solid_id).name}"
                        )
                        return
            # No face yet — wait for the next face click (SolidWorks-like)
            self._await_face_sketch = True
            self.props.show_empty()
            self.props._hint.setText(
                "Sketch on Face\n\n"
                "Click a flat face of the selected solid in the graphics area.\n"
                "Esc cancels."
            )
            self.statusBar().showMessage(
                "Click a face of the solid to place the sketch (Esc cancels)", 6000
            )
            return
        if f is not None and is_reference_plane(f.type):
            skf = self.doc.create_sketch_on_plane(f.id)
            if skf is None or skf.sketch is None:
                return
            self._refresh_tree()
            self._sync_selection(skf.id)
            self._update_window_title()
            self.viewport.enter_sketch(skf.id)
            self._set_sketch_ribbon_enabled(True)
            self._sync_sketch_tool_ui(SketchTool.LINE)
            self.statusBar().showMessage(f"Editing {skf.name} on {f.name}")
            return
        self.statusBar().showMessage(
            "Select a reference plane or a solid face, then Sketch", 4000
        )

    def _exit_sketch(self) -> None:
        sid = (
            int(self.viewport._sketch_feature_id)
            if self.viewport.in_sketch_mode
            else -1
        )
        self.viewport.exit_sketch()
        self._set_sketch_ribbon_enabled(False)
        # End edit-time roll back so dependent solids reappear and re-evaluate
        self._clear_edit_rollback()
        # Rebuild solids that depend on the sketch (edit-sketch → update extrude/cut/…)
        self.viewport.schedule_rebuild()
        self.viewport.refresh_sketches()
        self._refresh_tree()
        # Prefer selecting the consuming feature after edit (SolidWorks-like)
        absorbed = self.doc.absorbed_sketch_map()
        consumer = absorbed.get(sid) if sid >= 0 else None
        self._sync_selection(int(consumer) if consumer is not None else self.doc.selected_id)
        self._update_window_title()
        self.statusBar().showMessage("Exited sketch — model updated", 2500)

    def _on_sketch_exited(self) -> None:
        self._set_sketch_ribbon_enabled(False)
        self._clear_edit_rollback()
        self.viewport.schedule_rebuild()
        self.viewport.refresh_sketches()
        self._refresh_tree()

    def _resolve_closed_sketch_id(self) -> int:
        """Sketch feature id for extrude/revolve: active sketch mode, else selection."""
        if self.viewport.in_sketch_mode and self.viewport._sketch_feature_id >= 0:
            return int(self.viewport._sketch_feature_id)
        f = self.doc.find(self.doc.selected_id)
        if f is not None and f.type is FeatureType.SKETCH:
            return f.id
        # Prefer the most recently created sketch with a closed profile
        for f in reversed(self.doc.features):
            if f.type is FeatureType.SKETCH and f.sketch is not None:
                if first_closed_profile(f.sketch) is not None:
                    return f.id
        return -1

    def _get_length_mm(
        self,
        title: str,
        label: str,
        default_mm: float,
        *,
        minimum_mm: float = 1e-6,
        maximum_mm: float = 1e6,
        decimals: int = 4,
    ) -> Optional[float]:
        """Length dialog in the active display unit; returns internal mm or None if cancelled."""
        unit = self.doc.display_unit
        lo = from_mm(minimum_mm, unit)
        hi = from_mm(maximum_mm, unit)
        default = from_mm(default_mm, unit)
        # Keep spin range sane for the unit
        lo = max(lo, 1e-9)
        val, ok = QInputDialog.getDouble(
            self,
            title,
            f"{label} ({unit.label}):",
            float(default),
            float(lo),
            float(hi),
            decimals,
        )
        if not ok:
            return None
        return to_mm(float(val), unit)

    @staticmethod
    def _profile_label(prof: object) -> str:
        """Human-readable picker label for a closed profile (entity or line-loop)."""
        if isinstance(prof, ClosedLineLoop):
            n = len(prof.line_ids)
            return f"Line loop ({n} segments) id={prof.id}"
        if isinstance(prof, RectEntity):
            return f"Rectangle id={prof.id}"
        if isinstance(prof, CircleEntity):
            return f"Circle id={prof.id}"
        kind = type(prof).__name__.replace("Entity", "")
        return f"{kind} id={getattr(prof, 'id', '?')}"

    def _resolve_profile_ids_for_command(self, sketch, *, title: str = "Select Profile"):
        """Which closed regions to use for Extrude/Revolve/etc.

        Returns a list of preferred_outer_id values for create_* calls:
          * [-1] — single / nested auto-resolve (no pick needed)
          * [id, ...] — one or more regions selected in the sketch
          * None — user must pick (caller shows a message); empty list cancelled

        Disjoint multi-profile sketches no longer use a text list; click the
        filled region in sketch mode (Ctrl-click to multi-select).
        """
        try:
            profiles = list_closed_profiles(sketch)
        except ValueError:
            raise
        if not profiles:
            raise ValueError("sketch has no closed profile")
        # Single or nested: resolve_profiles succeeds without a preferred id
        try:
            resolve_profiles(sketch)
            return [-1]
        except ValueError as exc:
            if "ambiguous" not in str(exc).lower():
                raise
        # Disjoint: require sketch-region selection (no list popup)
        sel = (
            self.viewport.selected_profile_ids()
            if self.viewport.in_sketch_mode
            else set()
        )
        # Keep only ids that still exist as closed profiles
        valid = {int(getattr(p, "id", -1)) for p in profiles}
        chosen = [pid for pid in sel if pid in valid]
        # Line-loop synthetic ids are in valid; also accept if any profile matches
        if not chosen and sel:
            # sel might use a line id that belongs to a loop
            for p in profiles:
                pid = int(getattr(p, "id", -1))
                if pid in sel:
                    chosen.append(pid)
                elif isinstance(p, ClosedLineLoop) and any(
                    lid in sel for lid in p.line_ids
                ):
                    chosen.append(pid)
        if not chosen:
            return None  # need pick
        # de-dupe preserving order
        out: list[int] = []
        for pid in chosen:
            if pid not in out:
                out.append(pid)
        return out

    def _start_feature_cmd(self, kind: str) -> None:
        """SolidWorks-style: command → select → PropertyManager → OK/Cancel."""
        if self.viewport.in_sketch_mode:
            self.viewport.exit_sketch()
            self._set_sketch_ribbon_enabled(False)
        self._await_face_sketch = False
        self._feature_cmd = {
            "kind": kind,
            "sketch_id": -1,
            "target_id": -1,
            "profile_id": -1,
            "edge_keys": [],  # solid edge fillet multi-select
            "solid_id": -1,
            "plane_id": -1,
        }
        titles = {
            "extrude": "Extrude (Boss/Base)",
            "cut": "Cut-Extrude",
            "fillet": "Fillet",
            "chamfer": "Chamfer",
            "revolve": "Revolve",
            "pocket": "Pocket",
            "lpattern": "Linear Pattern",
            "cpattern": "Circular Pattern",
            "mirror": "Mirror",
        }
        if kind == "fillet":
            sel_text = "Click edges on a solid to fillet…"
            status = "Fillet: click solid edges, set radius, OK — Esc cancels"
        elif kind == "chamfer":
            sel_text = "Click edges on a solid to chamfer…"
            status = "Chamfer: click solid edges, set distance, OK — Esc cancels"
        elif kind in ("lpattern", "cpattern"):
            sel_text = "Select a solid to pattern…"
            status = f"{titles[kind]}: select solid, set params, OK — Esc cancels"
        elif kind == "mirror":
            sel_text = "Select a solid, then a reference plane…"
            status = "Mirror: select solid + plane, OK — Esc cancels"
        else:
            sel_text = "Select a sketch with a closed profile…"
            status = (
                f"{titles.get(kind, kind)}: select a sketch, set parameters, "
                "OK — Esc cancels"
            )
        self.props.show_command(
            kind,
            title=titles.get(kind, kind.title()),
            selection_text=sel_text,
            unit=self.doc.display_unit,
            defaults={
                "depth": 10.0,
                "radius": 2.0,
                "angle": 360.0,
                "segments": 32,
                "through_all": False,
                "reversed": False,
                "count": 3 if kind == "lpattern" else 4,
                "dx": 20.0,
                "dy": 0.0,
                "dz": 0.0,
            },
            ready=False,
        )
        # Prefer current selection
        self._update_command_from_selection(self.doc.selected_id)
        self.statusBar().showMessage(status, 6000)

    def _cancel_feature_cmd(self) -> None:
        self._feature_cmd = None
        self._await_face_sketch = False
        # Restore PM for current selection
        self._sync_selection(self.doc.selected_id)
        self.statusBar().showMessage("Command cancelled", 2000)

    def _on_command_cancel(self) -> None:
        self._cancel_feature_cmd()

    def _update_command_from_selection(self, fid: int) -> None:
        cmd = self._feature_cmd
        if cmd is None:
            return
        kind = cmd["kind"]
        f = self.doc.find(fid)

        # ----- Solid edge fillet / chamfer: pick edges on a solid -----
        if kind in ("fillet", "chamfer"):
            self._try_add_fillet_edge(fid)
            return

        # ----- Pattern: pick solid -----
        if kind in ("lpattern", "cpattern"):
            if f is not None and is_solid_feature(f.type):
                cmd["solid_id"] = f.id
                self.props.update_command_selection(
                    f"Solid: {f.name}", ready=True
                )
            else:
                self.props.update_command_selection(
                    "Select a solid to pattern…", ready=False
                )
            return

        # ----- Mirror: solid then plane -----
        if kind == "mirror":
            if f is not None and is_solid_feature(f.type):
                cmd["solid_id"] = f.id
                pl = self.doc.find(int(cmd.get("plane_id", -1)))
                if pl is not None and is_reference_plane(pl.type):
                    self.props.update_command_selection(
                        f"Solid: {f.name}\nPlane: {pl.name}", ready=True
                    )
                else:
                    self.props.update_command_selection(
                        f"Solid: {f.name}\nSelect a reference plane…",
                        ready=False,
                    )
                return
            if f is not None and is_reference_plane(f.type):
                cmd["plane_id"] = f.id
                solid = self.doc.find(int(cmd.get("solid_id", -1)))
                if solid is not None and is_solid_feature(solid.type):
                    self.props.update_command_selection(
                        f"Solid: {solid.name}\nPlane: {f.name}", ready=True
                    )
                else:
                    self.props.update_command_selection(
                        f"Plane: {f.name}\nSelect a solid…", ready=False
                    )
                return
            self.props.update_command_selection(
                "Select a solid, then a reference plane…", ready=False
            )
            return

        # Sketch selection
        if f is not None and f.type is FeatureType.SKETCH and f.sketch is not None:
            cmd["sketch_id"] = f.id
            # Target solid for cut: sketch-on-face parent
            if kind == "cut" and f.plane_id >= 0:
                parent = self.doc.find(f.plane_id)
                if parent is not None and is_solid_feature(parent.type):
                    cmd["target_id"] = parent.id
            try:
                profiles = list_closed_profiles(f.sketch)
            except ValueError:
                profiles = []
            if not profiles:
                self.props.update_command_selection(
                    f"{f.name}: no closed profile yet", ready=False
                )
                return
            # Default first profile / resolve if unambiguous
            try:
                resolved = resolve_profiles(f.sketch)
                cmd["profile_id"] = int(getattr(resolved.outer, "id", -1))
            except ValueError:
                cmd["profile_id"] = int(getattr(profiles[0], "id", -1))
            if kind == "cut":
                if cmd["target_id"] < 0:
                    # last solid before this sketch
                    for g in reversed(self.doc.features):
                        if is_solid_feature(g.type) and g.id != f.id:
                            cmd["target_id"] = g.id
                            break
                tgt = self.doc.find(cmd["target_id"])
                if tgt is None:
                    self.props.update_command_selection(
                        f"{f.name} selected — also select a solid to cut",
                        ready=False,
                    )
                    return
                self.props.update_command_selection(
                    f"Sketch: {f.name}\nSolid: {tgt.name}", ready=True
                )
            else:
                self.props.update_command_selection(
                    f"Sketch: {f.name} (closed profile ready)", ready=True
                )
            return
        # Solid selection for cut target
        if f is not None and is_solid_feature(f.type) and kind == "cut":
            cmd["target_id"] = f.id
            sk = self.doc.find(cmd["sketch_id"])
            if sk is not None and sk.sketch is not None:
                self.props.update_command_selection(
                    f"Sketch: {sk.name}\nSolid: {f.name}", ready=True
                )
            else:
                self.props.update_command_selection(
                    f"Solid: {f.name} — select a closed sketch to cut with",
                    ready=False,
                )
            return
        self.props.update_command_selection(
            "Select a sketch with a closed profile…", ready=False
        )

    def _try_add_fillet_edge(self, fid: int) -> None:
        """During Fillet/Chamfer command: resolve nearest convex edge at last pick."""
        cmd = self._feature_cmd
        if cmd is None or cmd.get("kind") not in ("fillet", "chamfer"):
            return
        from cadcore.edge_fillet import extract_convex_edges, pick_edge_near_point

        label = "fillet" if cmd.get("kind") == "fillet" else "chamfer"
        f = self.doc.find(fid)
        if f is None or not is_solid_feature(f.type):
            n = len(cmd.get("edge_keys") or [])
            if n:
                solid = self.doc.find(int(cmd.get("solid_id", -1)))
                sname = solid.name if solid else "solid"
                self.props.update_command_selection(
                    f"Solid: {sname}\n{n} edge{'s' if n != 1 else ''} selected",
                    ready=True,
                )
            else:
                self.props.update_command_selection(
                    f"Click edges on a solid to {label}…", ready=False
                )
            return

        # Fillet the solid as selected (including a prior Edge Fillet result)
        solid_id = int(fid)

        pick_pt = None
        try:
            pick_pt = self.viewport.last_pick_point()
        except Exception:
            pick_pt = None
        if pick_pt is None:
            # No 3D pick (e.g. tree click) — just bind the solid
            cmd["solid_id"] = solid_id
            n = len(cmd.get("edge_keys") or [])
            solid = self.doc.find(solid_id)
            sname = solid.name if solid else f"id {solid_id}"
            if n:
                self.props.update_command_selection(
                    f"Solid: {sname}\n{n} edge{'s' if n != 1 else ''} selected",
                    ready=True,
                )
            else:
                self.props.update_command_selection(
                    f"Solid: {sname}\nClick near an edge to add it",
                    ready=False,
                )
            return

        # Mesh for edge extraction: evaluate the solid being filleted
        body = self.doc.evaluate_feature(solid_id)
        if body is None or body.empty:
            cache = getattr(self.viewport, "_solid_mesh_cache", {}).get(fid)
            if cache is None:
                self.props.update_command_selection(
                    "Selected solid has no mesh yet", ready=False
                )
                return
            import numpy as np
            from cadcore.mesh import Mesh

            body = Mesh(np.asarray(cache[0]), np.asarray(cache[1]))

        edges = extract_convex_edges(body.vertices, body.faces)
        if not edges:
            self.props.update_command_selection(
                "No sharp convex edges found on this solid", ready=False
            )
            return
        # Pick tolerance: fraction of model size
        import numpy as np

        bb = body.vertices.max(axis=0) - body.vertices.min(axis=0)
        diag = float(np.linalg.norm(bb)) if np.any(bb > 0) else 10.0
        max_dist = max(0.5, 0.04 * diag)
        edge = pick_edge_near_point(edges, pick_pt, max_dist=max_dist)
        if edge is None:
            solid = self.doc.find(solid_id)
            sname = solid.name if solid else f"id {solid_id}"
            n = len(cmd.get("edge_keys") or [])
            self.props.update_command_selection(
                f"Solid: {sname}\n"
                f"No edge near click ({n} selected). Click closer to an edge.",
                ready=n > 0,
            )
            cmd["solid_id"] = solid_id
            return

        # Switching solids clears prior edge picks
        if int(cmd.get("solid_id", -1)) not in (-1, solid_id):
            cmd["edge_keys"] = []
        cmd["solid_id"] = solid_id
        key = edge.key()
        keys: list = list(cmd.get("edge_keys") or [])
        if key in keys:
            keys.remove(key)
        else:
            keys.append(key)
        cmd["edge_keys"] = keys
        solid = self.doc.find(solid_id)
        sname = solid.name if solid else f"id {solid_id}"
        n = len(keys)
        L = edge.length()
        if n:
            self.props.update_command_selection(
                f"Solid: {sname}\n"
                f"{n} edge{'s' if n != 1 else ''} selected "
                f"(last L={L:.3g} mm)",
                ready=True,
            )
            verb = "radius" if cmd.get("kind") == "fillet" else "distance"
            self.statusBar().showMessage(
                f"{label.title()}: {n} edge(s) on {sname} — set {verb} and OK",
                4000,
            )
        else:
            self.props.update_command_selection(
                f"Solid: {sname}\nClick near an edge to add it",
                ready=False,
            )

    def _on_command_ok(self) -> None:
        cmd = self._feature_cmd
        if cmd is None:
            return
        try:
            params = self.props.read_command_params()
        except ValueError as exc:
            QMessageBox.warning(self, "Command", f"Invalid parameter:\n{exc}")
            return
        kind = cmd["kind"]

        # ----- Solid edge fillet / chamfer (no sketch) -----
        if kind in ("fillet", "chamfer"):
            solid_id = int(cmd.get("solid_id", -1))
            edge_keys = list(cmd.get("edge_keys") or [])
            title = "Fillet" if kind == "fillet" else "Chamfer"
            if solid_id < 0 or self.doc.find(solid_id) is None:
                QMessageBox.warning(
                    self,
                    title,
                    "Select a solid and click at least one edge before pressing OK.",
                )
                return
            if not edge_keys:
                QMessageBox.warning(
                    self,
                    title,
                    "Click one or more edges on the solid, then press OK.",
                )
                return
            created = None
            try:
                if kind == "fillet":
                    created = self.doc.create_edge_fillet(
                        solid_id,
                        edge_keys,
                        float(params["radius"]),
                        segments=int(params.get("segments", 32)),
                    )
                else:
                    created = self.doc.create_edge_chamfer(
                        solid_id,
                        edge_keys,
                        float(params["radius"]),
                    )
            except ValueError as exc:
                QMessageBox.warning(
                    self,
                    f"{title} failed",
                    f"Could not apply the {title.lower()}:\n\n{exc}\n\n"
                    "The part was not changed. Try a smaller size or different edges.",
                )
                self.statusBar().showMessage(f"{title} failed: {exc}", 5000)
                return
            self._feature_cmd = None
            self.viewport.schedule_rebuild()
            self.viewport.refresh_sketches()
            self._refresh_tree()
            if created is not None:
                self._sync_selection(created.id)
            self._update_window_title()
            self.statusBar().showMessage(
                f"Created {created.name}" if created else "Done", 3000
            )
            return

        # ----- Linear / circular pattern -----
        if kind in ("lpattern", "cpattern"):
            solid_id = int(cmd.get("solid_id", -1))
            if solid_id < 0 or self.doc.find(solid_id) is None:
                QMessageBox.warning(
                    self, "Pattern", "Select a solid before pressing OK."
                )
                return
            created = None
            try:
                if kind == "lpattern":
                    created = self.doc.create_linear_pattern(
                        solid_id,
                        int(params.get("count", 3)),
                        float(params.get("dx", 20.0)),
                        float(params.get("dy", 0.0)),
                        float(params.get("dz", 0.0)),
                    )
                else:
                    created = self.doc.create_circular_pattern(
                        solid_id,
                        int(params.get("count", 4)),
                        total_angle_deg=float(params.get("angle", 360.0)),
                    )
            except ValueError as exc:
                QMessageBox.warning(self, "Pattern failed", str(exc))
                return
            self._feature_cmd = None
            self.viewport.schedule_rebuild()
            self.viewport.refresh_sketches()
            self._refresh_tree()
            if created is not None:
                self._sync_selection(created.id)
            self._update_window_title()
            self.statusBar().showMessage(
                f"Created {created.name}" if created else "Done", 3000
            )
            return

        # ----- Mirror -----
        if kind == "mirror":
            solid_id = int(cmd.get("solid_id", -1))
            plane_id = int(cmd.get("plane_id", -1))
            if solid_id < 0 or plane_id < 0:
                QMessageBox.warning(
                    self,
                    "Mirror",
                    "Select a solid and a reference plane before OK.",
                )
                return
            created = None
            try:
                created = self.doc.create_mirror(solid_id, plane_id)
            except ValueError as exc:
                QMessageBox.warning(self, "Mirror failed", str(exc))
                return
            self._feature_cmd = None
            self.viewport.schedule_rebuild()
            self.viewport.refresh_sketches()
            self._refresh_tree()
            if created is not None:
                self._sync_selection(created.id)
            self._update_window_title()
            self.statusBar().showMessage(
                f"Created {created.name}" if created else "Done", 3000
            )
            return

        sid = int(cmd.get("sketch_id", -1))
        skf = self.doc.find(sid)
        if skf is None or skf.sketch is None:
            QMessageBox.warning(
                self,
                "Command",
                "Select a sketch with a closed profile before pressing OK.",
            )
            return
        # Validate closed profile without mutating the document
        try:
            resolve_profiles(skf.sketch, preferred_outer_id=int(cmd.get("profile_id", -1)))
        except ValueError as exc:
            QMessageBox.warning(self, "Command", f"Cannot apply to this selection:\n{exc}")
            return

        created = None
        try:
            if kind == "extrude":
                created = self.doc.create_extrude(
                    sid,
                    float(params["depth"]),
                    profile_entity_id=int(cmd.get("profile_id", -1)),
                    reversed=bool(params.get("reversed", False)),
                )
            elif kind == "cut":
                tid = int(cmd.get("target_id", -1))
                if tid < 0 or self.doc.find(tid) is None:
                    QMessageBox.warning(
                        self,
                        "Cut-Extrude",
                        "Select a solid to cut into (click the solid, then OK).",
                    )
                    return
                created = self.doc.create_cut_extrude(
                    sid,
                    tid,
                    float(params["depth"]),
                    profile_entity_id=int(cmd.get("profile_id", -1)),
                    reversed=bool(params.get("reversed", False)),
                    through_all=bool(params.get("through_all", False)),
                )
            elif kind == "revolve":
                created = self.doc.create_revolve(
                    sid,
                    angle_degrees=float(params["angle"]),
                    profile_entity_id=int(cmd.get("profile_id", -1)),
                )
            elif kind == "pocket":
                created = self.doc.create_pocket(
                    sid,
                    float(params["depth"]),
                    float(params["radius"]),
                    (float(params["hole_u"]), float(params["hole_v"])),
                    profile_entity_id=int(cmd.get("profile_id", -1)),
                )
        except ValueError as exc:
            # Part must be untouched — create_* validates before committing
            QMessageBox.warning(
                self,
                "Command failed",
                f"Could not create the feature:\n\n{exc}\n\n"
                "The part was not changed. Fix the selection or parameters and try again.",
            )
            self.statusBar().showMessage(f"Failed: {exc}", 5000)
            return

        self._feature_cmd = None
        self.viewport.schedule_rebuild()
        self.viewport.refresh_sketches()
        self._refresh_tree()
        if created is not None:
            self._sync_selection(created.id)
        self._update_window_title()
        self.statusBar().showMessage(
            f"Created {created.name}" if created else "Done", 3000
        )

    def _extrude(self) -> None:
        self._start_feature_cmd("extrude")

    def _revolve(self) -> None:
        self._start_feature_cmd("revolve")


    def _fillet(self) -> None:
        self._start_feature_cmd("fillet")

    def _chamfer(self) -> None:
        self._start_feature_cmd("chamfer")

    def _linear_pattern(self) -> None:
        self._start_feature_cmd("lpattern")

    def _circular_pattern(self) -> None:
        self._start_feature_cmd("cpattern")

    def _mirror_feature(self) -> None:
        self._start_feature_cmd("mirror")

    def _offset_plane(self) -> None:
        """Offset plane from selected reference plane (quick dialog)."""
        f = self.doc.find(self.doc.selected_id)
        if f is None or not is_reference_plane(f.type):
            self.statusBar().showMessage(
                "Select a reference plane, then Plane", 4000
            )
            return
        mm = self._get_length_mm("Offset Plane", "Offset", 10.0)
        if mm is None:
            return
        try:
            pf = self.doc.create_offset_plane(f.id, float(mm))
        except ValueError as exc:
            QMessageBox.warning(self, "Offset Plane", str(exc))
            return
        self.viewport.schedule_rebuild()
        self.viewport.refresh_sketches()
        self._refresh_tree()
        self._sync_selection(pf.id)
        self._update_window_title()
        self.statusBar().showMessage(f"Created {pf.name}", 3000)

    def _on_edit_sketch_requested(self, sketch_id: int) -> None:
        """PropertyManager Edit Sketch button."""
        if self._feature_cmd is not None:
            self._cancel_feature_cmd()
        sk = self.doc.find(int(sketch_id))
        if sk is None or sk.type is not FeatureType.SKETCH:
            self.statusBar().showMessage("Sketch not found", 3000)
            return
        consumer = self.doc.absorbed_sketch_map().get(int(sketch_id))
        self._open_sketch_edit(int(sketch_id), rollback_from=consumer)


    def _pocket(self) -> None:
        self._start_feature_cmd("pocket")


    def _cut_extrude(self) -> None:
        self._start_feature_cmd("cut")


    # ----- project file (Save / Open / New) -----
    def _update_window_title(self) -> None:
        if self._project_path is not None:
            name = self._project_path.name
        else:
            name = f"{self.doc.name}{DEFAULT_EXTENSION}" if self.doc.name else "Untitled.gcad"
            if self.doc.name == "Untitled":
                name = "Untitled.gcad"
        dirty = " *" if self.doc.dirty else ""
        self.setWindowTitle(f"Grok CAD — {name}{dirty}")

    @staticmethod
    def _is_unattended() -> bool:
        """True when no human can answer modal dialogs (CI / automated runs).

        Humans still get the unsaved-changes prompt. Automated runs must always
        be able to exit — set GROK_CAD_UNATTENDED=1, or use the offscreen Qt
        platform (typical for headless verification).
        """
        flag = os.environ.get("GROK_CAD_UNATTENDED", "").strip().lower()
        if flag in ("1", "true", "yes", "on"):
            return True
        if os.environ.get("QT_QPA_PLATFORM", "").strip().lower() == "offscreen":
            return True
        return False

    def _confirm_discard_if_dirty(self) -> bool:
        """Return True if it is safe to discard the current document."""
        if not self.doc.dirty:
            return True
        # Automated / headless: discard without blocking (never leave a dialog up).
        if self._is_unattended():
            print(
                "[mainwindow] unattended: discarding unsaved changes (no dialog)",
                file=sys.stderr,
            )
            return True
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Unsaved changes")
        box.setText("The current project has unsaved changes.")
        box.setInformativeText("Do you want to save before continuing?")
        save_btn = box.addButton("Save", QMessageBox.ButtonRole.AcceptRole)
        discard_btn = box.addButton("Don't Save", QMessageBox.ButtonRole.DestructiveRole)
        cancel_btn = box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(save_btn)
        box.exec()
        clicked = box.clickedButton()
        if clicked is cancel_btn:
            return False
        if clicked is save_btn:
            return self._file_save()
        return True  # Don't Save

    def _reload_ui_from_document(self) -> None:
        """After New / Open: rebind views and rebuild the scene."""
        if self.viewport.in_sketch_mode:
            self.viewport.exit_sketch()
            self._set_sketch_ribbon_enabled(False)
        self.viewport.set_document(self.doc)
        self.props.set_document(self.doc)
        self.props.clear()
        # Sync unit menu checks
        if hasattr(self, "_unit_actions"):
            for u, act in self._unit_actions.items():
                act.setChecked(u is self.doc.display_unit or u == self.doc.display_unit)
        self._refresh_tree()
        sid = self.doc.selected_id
        if sid >= 0 and self.doc.find(sid) is not None:
            self._sync_selection(sid)
        else:
            self.doc.selected_id = -1
            self._sync_selection(-1)
        self.viewport.schedule_rebuild()
        self.viewport.refresh_sketches()
        self.viewport.zoom_to_fit()
        self._update_window_title()

    def _file_new(self) -> None:
        if not self._confirm_discard_if_dirty():
            return
        self.doc.clear()
        self.doc.seed_reference_planes()
        self.doc.selected_id = -1
        self.doc.mark_clean()
        self._project_path = None
        self._reload_ui_from_document()
        self.statusBar().showMessage("New project", 2500)

    def _file_open(self) -> None:
        if not self._confirm_discard_if_dirty():
            return
        path, _filt = QFileDialog.getOpenFileName(
            self,
            "Open Project",
            str(self._project_path.parent) if self._project_path else "",
            PROJECT_FILTER,
        )
        if not path:
            return
        try:
            loaded = load_document(path)
        except ProjectIOError as exc:
            QMessageBox.warning(self, "Open Project", str(exc))
            self.statusBar().showMessage(f"Open failed: {exc}", 5000)
            return
        except OSError as exc:
            QMessageBox.warning(self, "Open Project", f"Could not read file:\n{exc}")
            return
        replace_document_contents(self.doc, loaded)
        self._project_path = Path(path)
        self._reload_ui_from_document()
        self.statusBar().showMessage(f"Opened {self._project_path.name}", 4000)

    def _file_save(self) -> bool:
        """Save to current path, or Save As if none. Returns True on success."""
        if self._project_path is None:
            return self._file_save_as()
        return self._write_project(self._project_path)

    def _file_save_as(self) -> bool:
        default = (
            str(self._project_path)
            if self._project_path is not None
            else f"{self.doc.name or 'Untitled'}{DEFAULT_EXTENSION}"
        )
        path, _filt = QFileDialog.getSaveFileName(
            self,
            "Save Project As",
            default,
            PROJECT_FILTER,
        )
        if not path:
            return False
        p = Path(path)
        if p.suffix.lower() not in (".gcad", ".json"):
            p = p.with_suffix(DEFAULT_EXTENSION)
        return self._write_project(p)

    def _write_project(self, path: Path) -> bool:
        try:
            out = save_document(self.doc, path)
        except OSError as exc:
            QMessageBox.warning(self, "Save Project", f"Could not write file:\n{exc}")
            self.statusBar().showMessage(f"Save failed: {exc}", 5000)
            return False
        except (TypeError, ValueError) as exc:
            QMessageBox.warning(self, "Save Project", f"Could not serialize project:\n{exc}")
            return False
        self._project_path = out
        self.doc.mark_clean()
        self._update_window_title()
        self.statusBar().showMessage(f"Saved {out.name}", 4000)
        return True

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if not self._confirm_discard_if_dirty():
            event.ignore()
            return
        event.accept()

    def _export_stl(self) -> None:
        """Export the currently selected solid (extrude/revolve/…) as binary STL."""
        if self.viewport.in_sketch_mode:
            self.statusBar().showMessage("Exit sketch before exporting", 3000)
            return
        f = self.doc.find(self.doc.selected_id)
        if f is None or is_reference_plane(f.type) or f.type is FeatureType.SKETCH:
            QMessageBox.information(
                self,
                "Export STL",
                "Select a solid feature (Extrude or Revolve) in the tree, "
                "then File → Export STL…",
            )
            self.statusBar().showMessage("Select a solid feature to export", 4000)
            return

        # Evaluate on this thread via document path (geometry); no VTK required
        try:
            mesh = self.doc.evaluate_feature(f.id)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Export STL", f"Failed to evaluate solid:\n{exc}")
            return
        if mesh is None or mesh.empty:
            QMessageBox.information(
                self,
                "Export STL",
                f"{f.name} has no solid mesh to export.",
            )
            return
        if not mesh.is_watertight():
            QMessageBox.warning(
                self,
                "Export STL",
                f"{f.name} is not watertight and cannot be exported to STL.",
            )
            return

        default_name = f"{f.name.replace(' ', '_')}.stl"
        path, _filt = QFileDialog.getSaveFileName(
            self,
            "Export STL",
            default_name,
            "STL files (*.stl);;All files (*)",
        )
        if not path:
            return
        if not path.lower().endswith(".stl"):
            path = path + ".stl"

        try:
            write_stl_binary(mesh, path, require_watertight=True)
        except ValueError as exc:
            QMessageBox.warning(self, "Export STL", str(exc))
            self.statusBar().showMessage(f"Export failed: {exc}", 4000)
            return
        except OSError as exc:
            QMessageBox.warning(self, "Export STL", f"Could not write file:\n{exc}")
            return

        ntri = len(mesh.faces)
        self.statusBar().showMessage(
            f"Exported {f.name} → {path}  ({ntri} triangles)", 5000
        )

    def _delete_selected(self) -> None:
        if self.viewport.in_sketch_mode:
            # In sketch mode Delete removes ALL selected entities (one undo step)
            ctrl = self.viewport._sketch_ctrl
            if ctrl is None or not ctrl.selected_ids:
                self.statusBar().showMessage("Select a sketch entity to delete", 2500)
                return
            sid = self.viewport._sketch_feature_id
            eids = list(ctrl.selected_ids)
            n = self.doc.delete_entities(sid, eids)
            if n > 0:
                ctrl.clear_selection()
                self.viewport.sync_sketch_visuals()
                self._update_window_title()
                self.statusBar().showMessage(
                    f"Deleted {n} entit{'y' if n == 1 else 'ies'}", 2000
                )
            return
        fid = self.doc.selected_id
        f = self.doc.find(fid)
        if f and is_reference_plane(f.type):
            QMessageBox.information(self, "Delete", "Reference planes cannot be deleted.")
            return
        if self.doc.delete_feature_undoable(fid):
            self.viewport.schedule_rebuild()
            self.viewport.refresh_sketches()
            self._refresh_tree()
            self._sync_selection(self.doc.selected_id)
            self._update_window_title()
            self.statusBar().showMessage("Feature deleted", 2000)

    def _active_sketch_id(self) -> int:
        if self.viewport.in_sketch_mode and self.viewport._sketch_feature_id >= 0:
            return int(self.viewport._sketch_feature_id)
        f = self.doc.find(self.doc.selected_id)
        if f is not None and f.type is FeatureType.SKETCH:
            return f.id
        return -1

    def _undo(self) -> None:
        if not self.doc.undo():
            self.statusBar().showMessage("Nothing to undo", 1500)
            return
        if self.viewport.in_sketch_mode:
            self.viewport.sync_sketch_visuals()
        else:
            self.viewport.schedule_rebuild()
            self.viewport.refresh_sketches()
        self._refresh_tree()
        self._sync_selection(self.doc.selected_id)
        self._update_window_title()
        self.statusBar().showMessage("Undo", 1500)

    def _redo(self) -> None:
        if not self.doc.redo():
            self.statusBar().showMessage("Nothing to redo", 1500)
            return
        if self.viewport.in_sketch_mode:
            self.viewport.sync_sketch_visuals()
        else:
            self.viewport.schedule_rebuild()
            self.viewport.refresh_sketches()
        self._refresh_tree()
        self._sync_selection(self.doc.selected_id)
        self._update_window_title()
        self.statusBar().showMessage("Redo", 1500)

    def _copy(self) -> None:
        sid = self._active_sketch_id()
        ctrl = self.viewport._sketch_ctrl if self.viewport.in_sketch_mode else None
        if sid < 0 or ctrl is None or not ctrl.selected_ids:
            self.statusBar().showMessage("Select a sketch entity to copy", 2500)
            return
        if len(ctrl.selected_ids) > 1:
            self.statusBar().showMessage(
                "Copy supports a single selection — select one entity", 3000
            )
            return
        eid = ctrl.selected_entity_id
        if eid < 0:
            self.statusBar().showMessage("Select a sketch entity to copy", 2500)
            return
        if self.doc.copy_entity(sid, eid):
            self.statusBar().showMessage("Copied", 1500)

    def _cut(self) -> None:
        sid = self._active_sketch_id()
        ctrl = self.viewport._sketch_ctrl if self.viewport.in_sketch_mode else None
        if sid < 0 or ctrl is None or not ctrl.selected_ids:
            self.statusBar().showMessage("Select a sketch entity to cut", 2500)
            return
        if len(ctrl.selected_ids) > 1:
            self.statusBar().showMessage(
                "Cut supports a single selection — select one entity "
                "(or Delete to remove all selected)",
                3500,
            )
            return
        eid = ctrl.selected_entity_id
        if eid < 0:
            self.statusBar().showMessage("Select a sketch entity to cut", 2500)
            return
        if self.doc.cut_entity(sid, eid):
            if ctrl is not None:
                ctrl.clear_selection()
            if self.viewport.in_sketch_mode:
                self.viewport.sync_sketch_visuals()
            self.statusBar().showMessage("Cut", 1500)

    def _paste(self) -> None:
        """Paste only into a live editable sketch — no silent document/undo pollution."""
        if not self.viewport.in_sketch_mode or self.viewport._sketch_ctrl is None:
            self.statusBar().showMessage("Enter sketch mode to paste", 2500)
            return
        sid = int(self.viewport._sketch_feature_id)
        if sid < 0:
            self.statusBar().showMessage("Enter sketch mode to paste", 2500)
            return
        if self.doc._clipboard is None:
            self.statusBar().showMessage("Clipboard empty", 1500)
            return
        place = self.viewport.sketch_cursor_uv()
        ent = self.doc.paste_entity(sid, place_uv=place)
        if ent is None:
            self.statusBar().showMessage("Clipboard empty", 1500)
            return
        # Always refresh the viewport after a successful paste
        self.viewport._sketch_ctrl.set_selection({ent.id})
        self.viewport.sync_sketch_visuals()
        self.statusBar().showMessage(f"Pasted entity {ent.id}", 2000)

    def _set_unit(self, unit: Unit) -> None:
        self.doc.set_display_unit(unit)
        act = self._unit_actions.get(unit)
        if act is not None:
            act.setChecked(True)
        if self.viewport.in_sketch_mode:
            self.viewport.refresh_dim_labels()
        self.statusBar().showMessage(f"Units: {unit.label}", 2000)

    def _set_line_length(self) -> None:
        """Dialog: set selected line length as a driving dimension (undoable)."""
        if not self.viewport.in_sketch_mode or self.viewport._sketch_ctrl is None:
            QMessageBox.information(
                self, "Set Length", "Enter sketch mode and select a line first."
            )
            return
        ctrl = self.viewport._sketch_ctrl
        if len(ctrl.selected_ids) != 1:
            QMessageBox.information(
                self, "Set Length", "Select exactly one line entity first."
            )
            return
        ent = ctrl.sketch.find_entity(ctrl.selected_entity_id)
        if not isinstance(ent, LineEntity):
            QMessageBox.information(self, "Set Length", "Select a line entity first.")
            return
        self._on_dimension_requested(ent.id, "length")

    def _on_dimension_requested(
        self, entity_id: int, role: str, entity_b_id: int = -1
    ) -> None:
        """Smart Dimension: typed value drives geometry + persists through drag."""
        if not self.viewport.in_sketch_mode or self.viewport._sketch_ctrl is None:
            return
        sid = self.viewport._sketch_feature_id
        sk = self.viewport._sketch_ctrl.sketch
        ent = sk.find_entity(int(entity_id))
        if ent is None:
            return
        unit = self.doc.display_unit
        role = str(role)
        ebid = int(entity_b_id)
        ent_b = sk.find_entity(ebid) if role == "angle" else None
        try:
            if role == "angle":
                from cadcore.sketch import line_angle_degrees_oriented

                if not isinstance(ent, LineEntity) or not isinstance(ent_b, LineEntity):
                    self.statusBar().showMessage(
                        "Angle dimension needs two lines", 3000
                    )
                    return
                cur_disp = line_angle_degrees_oriented(ent, ent_b)
            else:
                cur_mm = measure_dimension_value(ent, role)
                cur_disp = from_mm(cur_mm, unit)
        except ValueError:
            self.statusBar().showMessage(f"Cannot dimension {role} on that entity", 3000)
            return
        role_label = {
            "length": "Length",
            "width": "Width",
            "height": "Height",
            "diameter": "Diameter",
            "radius": "Radius",
            "angle": "Angle",
        }.get(role, role.title())
        if role == "angle":
            val, ok = QInputDialog.getDouble(
                self,
                "Smart Dimension",
                f"{role_label} (°):",
                float(cur_disp),
                0.0,
                180.0,
                4,
            )
            if not ok:
                return
            store_val = float(val)
        else:
            val, ok = QInputDialog.getDouble(
                self,
                "Smart Dimension",
                f"{role_label} ({unit.label}):",
                float(cur_disp),
                1e-6,
                1e9,
                4,
            )
            if not ok:
                return
            store_val = to_mm(val, unit)
        try:
            dim = self.doc.apply_sketch_dimension(
                sid, int(entity_id), role, store_val, entity_b_id=ebid
            )
        except ValueError as exc:
            QMessageBox.warning(
                self,
                "Dimension conflict",
                f"{exc}\n\nThe sketch was not changed.",
            )
            self.statusBar().showMessage(f"Dimension refused: {exc}", 5000)
            return
        self.viewport.sync_sketch_visuals()
        self._refresh_tree()
        self._update_window_title()
        if role == "angle":
            shown = f"{store_val:g}°"
        else:
            shown = format_length(store_val, unit)
            if role == "diameter":
                shown = "⌀" + shown
            elif role == "radius":
                shown = "R" + shown
        self.statusBar().showMessage(
            f"{role_label} → {shown}" + (f"  (dim #{dim.id})" if dim else ""),
            3000,
        )

    def _make_angle_dimension(self) -> None:
        """Angle between two selected lines — type degrees, persists through drag."""
        lines = self._selected_lines()
        if len(lines) < 2:
            self.statusBar().showMessage(
                "Select two lines, then Angle dimension", 3000
            )
            return
        self._on_dimension_requested(lines[0].id, "angle", entity_b_id=lines[1].id)

    def _make_radius_dimension(self) -> None:
        """Radius driving dimension on the selected arc."""
        from cadcore.sketch import ArcEntity

        ctx = self._sketch_context()
        if ctx is None:
            return
        ctrl, _ = ctx
        arcs = [
            ctrl.sketch.find_entity(eid)
            for eid in ctrl.selected_ids
            if isinstance(ctrl.sketch.find_entity(eid), ArcEntity)
        ]
        if not arcs:
            self.statusBar().showMessage("Select an arc, then Radius", 3000)
            return
        self._on_dimension_requested(arcs[0].id, "radius")

    def _toggle_construction(self) -> None:
        ctx = self._sketch_context()
        if ctx is None:
            return
        ctrl, sid = ctx
        if not ctrl.selected_ids:
            self.statusBar().showMessage("Select entities to toggle construction", 2500)
            return
        before = snapshot_sketch_contents(ctrl.sketch)
        from cadcore.sketch_ops import toggle_construction

        ents = [
            ctrl.sketch.find_entity(eid)
            for eid in ctrl.selected_ids
            if ctrl.sketch.find_entity(eid) is not None
        ]
        n = toggle_construction(ents)
        after = snapshot_sketch_contents(ctrl.sketch)
        self.doc.record_sketch_contents(sid, before, after)
        self.viewport.sync_sketch_visuals()
        self.statusBar().showMessage(f"Construction toggled on {n} entit(y/ies)", 2500)

    def _make_midpoint(self) -> None:
        lines = self._selected_lines()
        ctx = self._sketch_context()
        if ctx is None or not lines:
            self.statusBar().showMessage(
                "Select a line and another line (endpoint mid of first)", 3000
            )
            return
        ctrl, _ = ctx
        # Second selected line's p0 becomes midpoint of first
        ids = list(ctrl.selected_ids)
        if len(ids) < 2:
            self.statusBar().showMessage("Select mirror subject: line + point line", 3000)
            return
        ln = lines[0]
        other = ctrl.sketch.find_entity(ids[1] if ids[0] == ln.id else ids[0])
        if other is None:
            return
        self._apply_persistent_constraint(
            SketchConstraint(
                id=-1,
                kind=ConstraintKind.MIDPOINT,
                e0=int(ln.id),
                e1=int(other.id),
                h1="p0",
            )
        ) and self.statusBar().showMessage("Midpoint applied", 2500)

    def _make_concentric(self) -> None:
        from cadcore.sketch import ArcEntity, CircleEntity

        ctx = self._sketch_context()
        if ctx is None:
            return
        ctrl, _ = ctx
        ents = [
            ctrl.sketch.find_entity(eid)
            for eid in ctrl.selected_ids
            if isinstance(ctrl.sketch.find_entity(eid), (CircleEntity, ArcEntity))
        ]
        if len(ents) < 2:
            self.statusBar().showMessage("Select two circles/arcs", 3000)
            return
        self._apply_persistent_constraint(
            SketchConstraint(
                id=-1,
                kind=ConstraintKind.CONCENTRIC,
                e0=int(ents[0].id),
                e1=int(ents[1].id),
            )
        ) and self.statusBar().showMessage("Concentric applied", 2500)

    def _make_collinear(self) -> None:
        lines = self._selected_lines()
        if len(lines) < 2:
            self.statusBar().showMessage("Select two lines for collinear", 3000)
            return
        self._apply_persistent_constraint(
            SketchConstraint(
                id=-1,
                kind=ConstraintKind.COLLINEAR,
                e0=int(lines[0].id),
                e1=int(lines[1].id),
            )
        ) and self.statusBar().showMessage("Collinear applied", 2500)

    def _make_symmetric(self) -> None:
        lines = self._selected_lines()
        if len(lines) < 2:
            self.statusBar().showMessage(
                "Select mirror line + subject line (endpoints)", 3000
            )
            return
        self._apply_persistent_constraint(
            SketchConstraint(
                id=-1,
                kind=ConstraintKind.SYMMETRIC,
                e0=int(lines[0].id),
                e1=int(lines[1].id),
                h0="p0",
                h1="p1",
            )
        ) and self.statusBar().showMessage("Symmetric applied", 2500)

    def _make_equal_radius(self) -> None:
        from cadcore.sketch import ArcEntity, CircleEntity

        ctx = self._sketch_context()
        if ctx is None:
            return
        ctrl, _ = ctx
        ents = [
            ctrl.sketch.find_entity(eid)
            for eid in ctrl.selected_ids
            if isinstance(ctrl.sketch.find_entity(eid), (CircleEntity, ArcEntity))
        ]
        if len(ents) < 2:
            self.statusBar().showMessage("Select two circles/arcs for Equal R", 3000)
            return
        self._apply_persistent_constraint(
            SketchConstraint(
                id=-1,
                kind=ConstraintKind.EQUAL_RADIUS,
                e0=int(ents[0].id),
                e1=int(ents[1].id),
            )
        ) and self.statusBar().showMessage("Equal radius applied", 2500)

    def _convert_entities(self) -> None:
        """Project convex edges of the last face-pick solid into the active sketch."""
        ctx = self._sketch_context()
        if ctx is None:
            self.statusBar().showMessage("Enter a sketch first", 3000)
            return
        ctrl, sid = ctx
        skf = self.doc.find(sid)
        if skf is None or skf.sketch is None:
            return
        # Prefer parent solid of face sketch
        solid_id = int(skf.plane_id) if skf.plane_id >= 0 else -1
        if solid_id < 0 or not is_solid_feature(
            self.doc.find(solid_id).type if self.doc.find(solid_id) else FeatureType.SKETCH
        ):
            # last solid in doc
            for f in reversed(self.doc.features):
                if is_solid_feature(f.type):
                    solid_id = f.id
                    break
        if solid_id < 0:
            self.statusBar().showMessage("No solid to convert edges from", 3000)
            return
        body = self.doc.evaluate_feature(solid_id)
        if body is None or body.empty:
            self.statusBar().showMessage("Solid has no mesh", 3000)
            return
        from cadcore.edge_fillet import extract_convex_edges
        from cadcore.sketch_ops import convert_face_edges_to_sketch

        frame = skf.sketch.frame
        edges = extract_convex_edges(body.vertices, body.faces)
        pairs = []
        for e in edges[:24]:  # cap
            p0 = frame.to_local(e.p0)
            p1 = frame.to_local(e.p1)
            # only edges nearly on the sketch plane
            w0 = frame.to_world(p0)
            w1 = frame.to_world(p1)
            if abs(float(np.dot(w0 - frame.origin, frame.normal))) > 0.5:
                continue
            if abs(float(np.dot(w1 - frame.origin, frame.normal))) > 0.5:
                continue
            pairs.append((p0, p1))
        if not pairs:
            self.statusBar().showMessage("No face edges on this sketch plane", 3000)
            return
        before = snapshot_sketch_contents(ctrl.sketch)
        convert_face_edges_to_sketch(ctrl.sketch, pairs, construction=True)
        after = snapshot_sketch_contents(ctrl.sketch)
        self.doc.record_sketch_contents(sid, before, after)
        self.viewport.sync_sketch_visuals()
        self.statusBar().showMessage(
            f"Converted {len(pairs)} edge(s) as construction", 3000
        )

    def _sketch_context(self):
        if not self.viewport.in_sketch_mode or self.viewport._sketch_ctrl is None:
            return None
        return self.viewport._sketch_ctrl, self.viewport._sketch_feature_id

    def _apply_persistent_constraint(self, c: SketchConstraint) -> bool:
        """Add constraint with undo; False on conflict (message shown)."""
        ctx = self._sketch_context()
        if ctx is None:
            return False
        ctrl, sid = ctx
        before = snapshot_sketch_contents(ctrl.sketch)
        try:
            add_constraint(ctrl.sketch, c)
        except ValueError as exc:
            QMessageBox.warning(
                self,
                "Constraint conflict",
                f"{exc}\n\nThe sketch was not changed.",
            )
            self.statusBar().showMessage(f"Constraint refused: {exc}", 5000)
            return False
        after = snapshot_sketch_contents(ctrl.sketch)
        self.doc.record_sketch_contents(sid, before, after)
        self.viewport.sync_sketch_visuals()
        return True

    def _make_horizontal(self) -> None:
        ctx = self._sketch_context()
        if ctx is None:
            return
        ctrl, _sid = ctx
        n = 0
        for eid in list(ctrl.selected_ids):
            ent = ctrl.sketch.find_entity(eid)
            if not isinstance(ent, LineEntity):
                continue
            if self._apply_persistent_constraint(
                SketchConstraint(id=-1, kind=ConstraintKind.HORIZONTAL, e0=int(eid))
            ):
                n += 1
            else:
                break
        if n == 0 and not any(
            isinstance(ctrl.sketch.find_entity(e), LineEntity) for e in ctrl.selected_ids
        ):
            self.statusBar().showMessage("Select one or more lines first", 2500)
        elif n:
            self.statusBar().showMessage(f"Horizontal → {n} line(s)", 2500)

    def _make_vertical(self) -> None:
        ctx = self._sketch_context()
        if ctx is None:
            return
        ctrl, _sid = ctx
        n = 0
        for eid in list(ctrl.selected_ids):
            ent = ctrl.sketch.find_entity(eid)
            if not isinstance(ent, LineEntity):
                continue
            if self._apply_persistent_constraint(
                SketchConstraint(id=-1, kind=ConstraintKind.VERTICAL, e0=int(eid))
            ):
                n += 1
            else:
                break
        if n == 0 and not any(
            isinstance(ctrl.sketch.find_entity(e), LineEntity) for e in ctrl.selected_ids
        ):
            self.statusBar().showMessage("Select one or more lines first", 2500)
        elif n:
            self.statusBar().showMessage(f"Vertical → {n} line(s)", 2500)

    def _selected_lines(self):
        ctx = self._sketch_context()
        if ctx is None:
            return []
        ctrl, _ = ctx
        ids = sorted(
            eid
            for eid in ctrl.selected_ids
            if isinstance(ctrl.sketch.find_entity(eid), LineEntity)
        )
        return [ctrl.sketch.find_entity(i) for i in ids]

    def _make_equal(self) -> None:
        lines = self._selected_lines()
        if len(lines) < 2:
            self.statusBar().showMessage("Select at least two lines for Equal", 3000)
            return
        n = 0
        src = lines[0]
        for ent in lines[1:]:
            if self._apply_persistent_constraint(
                SketchConstraint(
                    id=-1,
                    kind=ConstraintKind.EQUAL,
                    e0=int(src.id),
                    e1=int(ent.id),
                )
            ):
                n += 1
            else:
                break
        if n:
            self.statusBar().showMessage(
                f"Equal → {n} line(s) = "
                f"{format_length(line_length(src), self.doc.display_unit)}",
                3000,
            )

    def _make_parallel(self) -> None:
        lines = self._selected_lines()
        if len(lines) < 2:
            self.statusBar().showMessage("Select two lines for Parallel", 3000)
            return
        if self._apply_persistent_constraint(
            SketchConstraint(
                id=-1,
                kind=ConstraintKind.PARALLEL,
                e0=int(lines[0].id),
                e1=int(lines[1].id),
            )
        ):
            self.statusBar().showMessage("Parallel applied", 2500)

    def _make_perpendicular(self) -> None:
        lines = self._selected_lines()
        if len(lines) < 2:
            self.statusBar().showMessage("Select two lines for Perpendicular", 3000)
            return
        if self._apply_persistent_constraint(
            SketchConstraint(
                id=-1,
                kind=ConstraintKind.PERPENDICULAR,
                e0=int(lines[0].id),
                e1=int(lines[1].id),
            )
        ):
            self.statusBar().showMessage("Perpendicular applied", 2500)

    def _nearest_endpoints(self, a: LineEntity, b: LineEntity):
        pairs = [
            ("p0", "p0", a.p0, b.p0),
            ("p0", "p1", a.p0, b.p1),
            ("p1", "p0", a.p1, b.p0),
            ("p1", "p1", a.p1, b.p1),
        ]
        best = min(
            pairs,
            key=lambda t: (t[2][0] - t[3][0]) ** 2 + (t[2][1] - t[3][1]) ** 2,
        )
        return best[0], best[1]

    def _make_coincident(self) -> None:
        lines = self._selected_lines()
        if len(lines) < 2:
            self.statusBar().showMessage(
                "Select two lines — nearest endpoints will stick together", 3000
            )
            return
        h0, h1 = self._nearest_endpoints(lines[0], lines[1])
        if self._apply_persistent_constraint(
            SketchConstraint(
                id=-1,
                kind=ConstraintKind.COINCIDENT,
                e0=int(lines[0].id),
                h0=h0,
                e1=int(lines[1].id),
                h1=h1,
            )
        ):
            self.statusBar().showMessage(
                f"Coincident → line {lines[0].id}.{h0} ↔ line {lines[1].id}.{h1}",
                3000,
            )

    def _make_fix(self) -> None:
        ctx = self._sketch_context()
        if ctx is None:
            return
        ctrl, _ = ctx
        n = 0
        for eid in list(ctrl.selected_ids):
            ent = ctrl.sketch.find_entity(eid)
            if not isinstance(ent, LineEntity):
                continue
            # Fix both endpoints of selected lines
            for h in ("p0", "p1"):
                if self._apply_persistent_constraint(
                    SketchConstraint(
                        id=-1, kind=ConstraintKind.FIX, e0=int(eid), h0=h
                    )
                ):
                    n += 1
                else:
                    return
        if n == 0:
            self.statusBar().showMessage("Select a line to Fix its endpoints", 2500)
        else:
            self.statusBar().showMessage(f"Fix → {n} point(s)", 2500)

    def _remove_constraints(self) -> None:
        ctx = self._sketch_context()
        if ctx is None:
            return
        ctrl, sid = ctx
        if not ctrl.selected_ids:
            self.statusBar().showMessage("Select entities to clear their constraints", 2500)
            return
        before = snapshot_sketch_contents(ctrl.sketch)
        removed = 0
        for eid in list(ctrl.selected_ids):
            removed += remove_constraints_for_entity(ctrl.sketch, int(eid))
        if removed == 0:
            self.statusBar().showMessage("No constraints on selection", 2500)
            return
        after = snapshot_sketch_contents(ctrl.sketch)
        self.doc.record_sketch_contents(sid, before, after)
        self.viewport.sync_sketch_visuals()
        self.statusBar().showMessage(f"Removed {removed} constraint(s)", 2500)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            if self._feature_cmd is not None:
                self._cancel_feature_cmd()
                return
            if getattr(self, "_await_face_sketch", False):
                self._await_face_sketch = False
                self._sync_selection(self.doc.selected_id)
                self.statusBar().showMessage("Sketch cancelled", 2000)
                return
        if self.viewport.in_sketch_mode:
            if event.key() == Qt.Key.Key_Escape:
                # Two-stage: cancel draw → Select, or exit sketch if idle
                was_drawing = (
                    self.viewport._sketch_ctrl is not None
                    and self.viewport._sketch_ctrl.is_drawing()
                )
                self.viewport.sketch_escape()
                if not self.viewport.in_sketch_mode:
                    self._set_sketch_ribbon_enabled(False)
                    self._refresh_tree()
                    self._sync_selection(self.doc.selected_id)
                    self.statusBar().showMessage("Exited sketch", 2000)
                elif was_drawing:
                    self._sync_sketch_tool_ui(SketchTool.SELECT)
                    self.statusBar().showMessage("Sketch: Select", 2000)
                else:
                    # Idle exit path also handled above; if still in sketch, sync tool
                    ctrl = self.viewport._sketch_ctrl
                    if ctrl is not None:
                        self._sync_sketch_tool_ui(ctrl.tool)
                return
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if self.viewport._try_commit_length_buffer():
                    return
                self.viewport.sketch_confirm()
                return
            if event.key() == Qt.Key.Key_Backspace:
                if self.viewport._length_buffer:
                    self.viewport._length_buffer = self.viewport._length_buffer[:-1]
                    self.viewport._emit_length_buffer_status()
                    return
            text = event.text() or ""
            if text and (text.isdigit() or text in ".-"):
                if self.viewport._accept_length_char(text):
                    return
        super().keyPressEvent(event)

    def _on_space_bar(self) -> None:
        """Application-wide Space: view picker (VTK steals focus from MainWindow)."""
        if self.viewport.in_sketch_mode:
            return
        self._show_view_orientation_menu()

    def _show_view_orientation_menu(self) -> None:
        """Space-bar view menu: standard views + axis look-along."""
        menu = QMenu(self)
        menu.setObjectName("ViewOrientationMenu")
        menu.setTitle("View Orientation")
        views = (
            ("Front", "front"),
            ("Back", "back"),
            ("Top", "top"),
            ("Bottom", "bottom"),
            ("Right", "right"),
            ("Left", "left"),
            ("Isometric", "iso"),
        )
        for label, key in views:
            act = menu.addAction(label)
            act.triggered.connect(
                lambda checked=False, k=key: self.viewport.set_view(k)
            )
        menu.addSeparator()
        for label, axis in (("Look +X", "x"), ("Look +Y", "y"), ("Look +Z", "z")):
            act = menu.addAction(label)
            act.triggered.connect(
                lambda checked=False, a=axis: self.viewport.view_along_axis(a)
            )
        menu.addSeparator()
        act_fit = menu.addAction("Zoom to Fit")
        act_fit.triggered.connect(self.viewport.zoom_to_fit)
        # Pop near the viewport centre so the user sees it
        gp = self.viewport.mapToGlobal(self.viewport.rect().center())
        menu.popup(gp)
        # Keep a ref so GC does not kill the menu before the user clicks
        self._view_menu = menu
