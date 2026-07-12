"""Main application window — sketch-first SolidWorks-style workflow."""

from __future__ import annotations

import os
from typing import Dict

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction, QActionGroup, QBrush, QColor, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QWidget,
)

from app.sketch_mode import SketchTool
from app.theme import (
    ACCENT,
    PLANE_FRONT,
    PLANE_RIGHT,
    PLANE_TOP,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    fa_icon,
)
from app.viewport import Viewport
from cadcore.document import (
    Document,
    FeatureType,
    feature_type_name,
    first_closed_profile,
    is_reference_plane,
)
from cadcore.mesh import write_stl_binary


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Grok CAD")
        self.resize(1280, 800)
        self.doc = Document()
        self.doc.seed_reference_planes()

        self.viewport = Viewport(self)
        self.setCentralWidget(self.viewport)
        self.viewport.set_document(self.doc)

        self._sketch_tool_actions: Dict[SketchTool, QAction] = {}
        self._status_env = self._format_env_status()

        self._build_tree_dock()
        self._build_props_dock()
        self._build_menus()
        self._build_toolbar()
        self._build_sketch_toolbar()
        self._build_status_bar()

        self.viewport.feature_picked.connect(self._on_pick)
        self.viewport.status_message.connect(self._on_status)
        self.viewport.busy_changed.connect(self._on_busy)
        self.viewport.sketch_exited.connect(self._on_sketch_exited)
        self.viewport.sketch_status.connect(self._on_status)
        self.viewport.renderer_info.connect(self._on_renderer_info)
        if getattr(self.viewport, "gl_renderer", ""):
            self._on_renderer_info(self.viewport.gl_renderer)

        for f in self.doc.features:
            if f.type is FeatureType.PLANE_FRONT:
                self.doc.selected_id = f.id
                break
        self._refresh_tree()
        self._sync_selection(self.doc.selected_id)
        self._set_ready_status()

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
        dock = QDockWidget("Properties", self)
        dock.setObjectName("PropertiesDock")
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        w = QWidget()
        w.setObjectName("PropertiesPanel")
        form = QFormLayout(w)
        form.setContentsMargins(12, 12, 12, 12)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(10)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self.prop_name = QLabel("—")
        self.prop_name.setObjectName("fieldValue")
        self.prop_name.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.prop_type = QLabel("—")
        self.prop_type.setObjectName("fieldValue")
        self.prop_type.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.prop_detail = QLabel("—")
        self.prop_detail.setObjectName("fieldValue")
        self.prop_detail.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        lbl_name = QLabel("Name")
        lbl_name.setObjectName("fieldLabel")
        lbl_type = QLabel("Type")
        lbl_type.setObjectName("fieldLabel")
        lbl_detail = QLabel("Detail")
        lbl_detail.setObjectName("fieldLabel")
        form.addRow(lbl_name, self.prop_name)
        form.addRow(lbl_type, self.prop_type)
        form.addRow(lbl_detail, self.prop_detail)

        hint = QLabel("Select a plane → Sketch → Extrude closed profile.")
        hint.setObjectName("secondaryLabel")
        hint.setWordWrap(True)
        form.addRow(hint)

        dock.setWidget(w)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _build_menus(self) -> None:
        file_m = self.menuBar().addMenu("&File")
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
        act_del = QAction(fa_icon("fa5s.trash-alt"), "Delete Feature", self)
        act_del.setShortcut(QKeySequence.StandardKey.Delete)
        act_del.triggered.connect(self._delete_selected)
        edit_m.addAction(act_del)

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

    def _build_toolbar(self) -> None:
        views = QToolBar("Views")
        views.setObjectName("ViewsToolBar")
        views.setMovable(False)
        views.setIconSize(QSize(20, 20))
        views.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.addToolBar(views)

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
                act.triggered.connect(lambda checked=False, k=key: self.viewport.set_view(k))
            views.addAction(act)

        views.addSeparator()

        main = QToolBar("Main")
        main.setObjectName("MainToolBar")
        main.setMovable(False)
        main.setIconSize(QSize(20, 20))
        main.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.addToolBar(main)
        self.act_sketch = QAction(
            fa_icon("fa5s.pencil-ruler", color=ACCENT), "Sketch", self
        )
        self.act_sketch.setToolTip("Create or edit a sketch on the selected plane (S)")
        self.act_sketch.triggered.connect(self._enter_sketch)
        main.addAction(self.act_sketch)

        self.act_extrude = QAction(fa_icon("fa5s.cube", color=ACCENT), "Extrude", self)
        self.act_extrude.setToolTip(
            "Extrude (pad) a closed sketch profile into a solid (E)"
        )
        self.act_extrude.setShortcut(QKeySequence("E"))
        self.act_extrude.triggered.connect(self._extrude)
        main.addAction(self.act_extrude)

        self.act_revolve = QAction(fa_icon("fa5s.sync-alt", color=ACCENT), "Revolve", self)
        self.act_revolve.setToolTip(
            "Revolve a closed sketch profile about the V-axis into a solid (R)"
        )
        self.act_revolve.setShortcut(QKeySequence("R"))
        self.act_revolve.triggered.connect(self._revolve)
        main.addAction(self.act_revolve)

        self.act_fillet = QAction(
            fa_icon("fa5s.circle-notch", color=ACCENT), "Fillet", self
        )
        self.act_fillet.setToolTip(
            "Fillet closed profile corners, then extrude into a solid (F)"
        )
        self.act_fillet.setShortcut(QKeySequence("F"))
        self.act_fillet.triggered.connect(self._fillet)
        main.addAction(self.act_fillet)

        self.act_pocket = QAction(
            fa_icon("fa5s.dot-circle", color=ACCENT), "Pocket", self
        )
        self.act_pocket.setToolTip(
            "Cut a circular through-hole pocket and extrude into a solid (P)"
        )
        self.act_pocket.setShortcut(QKeySequence("P"))
        self.act_pocket.triggered.connect(self._pocket)
        main.addAction(self.act_pocket)

    def _build_sketch_toolbar(self) -> None:
        self.sketch_tb = QToolBar("Sketch")
        self.sketch_tb.setObjectName("SketchToolBar")
        self.sketch_tb.setMovable(False)
        self.sketch_tb.setIconSize(QSize(20, 20))
        self.sketch_tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.addToolBar(self.sketch_tb)

        group = QActionGroup(self)
        group.setExclusive(True)

        tool_defs = (
            (SketchTool.SELECT, "Select", "fa5s.mouse-pointer", "Select and edit entities"),
            (SketchTool.LINE, "Line", "fa5s.minus", "Draw a line"),
            (SketchTool.RECTANGLE, "Rectangle", "fa5s.vector-square", "Draw a rectangle"),
            (SketchTool.CIRCLE, "Circle", "fa5s.circle", "Draw a circle"),
        )
        for tool, label, icon_name, tip in tool_defs:
            act = QAction(fa_icon(icon_name), label, self)
            act.setToolTip(tip)
            act.setCheckable(True)
            act.triggered.connect(lambda checked=False, t=tool: self._on_sketch_tool(t))
            group.addAction(act)
            self.sketch_tb.addAction(act)
            self._sketch_tool_actions[tool] = act

        self.sketch_tb.addSeparator()
        act_exit = QAction(fa_icon("fa5s.times", color=PLANE_RIGHT), "Exit Sketch", self)
        act_exit.setToolTip("Exit sketch mode (Esc when idle)")
        act_exit.triggered.connect(self._exit_sketch)
        self.sketch_tb.addAction(act_exit)
        self.sketch_tb.setVisible(False)

    def _on_sketch_tool(self, tool: SketchTool) -> None:
        self.viewport.set_sketch_tool(tool)
        self._sync_sketch_tool_ui(tool)

    def _sync_sketch_tool_ui(self, tool: SketchTool) -> None:
        act = self._sketch_tool_actions.get(tool)
        if act is not None and not act.isChecked():
            act.setChecked(True)
        # Refresh icons: checked tools get white glyphs for contrast on accent
        for t, a in self._sketch_tool_actions.items():
            name = {
                SketchTool.SELECT: "fa5s.mouse-pointer",
                SketchTool.LINE: "fa5s.minus",
                SketchTool.RECTANGLE: "fa5s.vector-square",
                SketchTool.CIRCLE: "fa5s.circle",
            }[t]
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
        return fa_icon("fa5s.cube", color=TEXT_SECONDARY)

    def _refresh_tree(self) -> None:
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

        for f in self.doc.features:
            item = QTreeWidgetItem([f.name])
            item.setData(0, Qt.ItemDataRole.UserRole, f.id)
            item.setIcon(0, self._icon_for_feature(f.type))
            item.setFlags(
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
            )
            if is_reference_plane(f.type):
                planes_root.addChild(item)
            else:
                features_root.addChild(item)
            if f.id == self.doc.selected_id:
                item.setSelected(True)
                self.tree.setCurrentItem(item)

        planes_root.setExpanded(True)
        features_root.setExpanded(True)
        self.tree.blockSignals(False)

    def _sync_selection(self, fid: int) -> None:
        if self.viewport.in_sketch_mode:
            return  # don't fight sketch selection
        self.doc.selected_id = fid
        self.viewport.set_selected_id(fid)
        f = self.doc.find(fid)
        if f is None:
            self.prop_name.setText("—")
            self.prop_type.setText("—")
            self.prop_detail.setText("—")
            self.statusBar().showMessage(f"Selected: (none) · {self._status_env}")
            return
        self.prop_name.setText(f.name)
        self.prop_type.setText(feature_type_name(f.type))
        if f.type is FeatureType.EXTRUDE:
            self.prop_detail.setText(f"Distance = {f.depth:g}")
        elif f.type is FeatureType.REVOLVE:
            self.prop_detail.setText(f"Angle = {f.revolve_angle:g}°")
        elif f.type is FeatureType.FILLET:
            self.prop_detail.setText(
                f"r={f.radius:g}, segs={f.segments}, dist={f.depth:g}"
            )
        elif f.type is FeatureType.POCKET:
            self.prop_detail.setText(
                f"hole r={f.radius:g}, center=({f.hole_center_u:g},{f.hole_center_v:g}), "
                f"dist={f.depth:g}"
            )
        elif f.type is FeatureType.SKETCH and f.sketch is not None:
            n = len(f.sketch.entities)
            self.prop_detail.setText(f"{n} entit{'y' if n == 1 else 'ies'}")
        elif is_reference_plane(f.type):
            self.prop_detail.setText("Reference")
        else:
            self.prop_detail.setText("—")
        self.statusBar().showMessage(f"Selected: {f.name} · {self._status_env}")
        self.tree.blockSignals(True)
        self.tree.clearSelection()
        for i in range(self.tree.topLevelItemCount()):
            root = self.tree.topLevelItem(i)
            for j in range(root.childCount()):
                item = root.child(j)
                if item.data(0, Qt.ItemDataRole.UserRole) == fid:
                    item.setSelected(True)
                    self.tree.setCurrentItem(item)
        self.tree.blockSignals(False)

    def _on_tree_sel(self) -> None:
        if self.viewport.in_sketch_mode:
            return
        items = self.tree.selectedItems()
        if not items:
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
        if f and f.type is FeatureType.SKETCH and f.sketch is not None:
            self._sync_selection(fid)
            self.viewport.enter_sketch(f.id)
            self.sketch_tb.setVisible(True)
            self._sync_sketch_tool_ui(SketchTool.LINE)
            self.statusBar().showMessage(f"Editing {f.name}")

    def _on_pick(self, fid: int) -> None:
        if self.viewport.in_sketch_mode:
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
        f = self.doc.find(self.doc.selected_id)
        # If a sketch is selected, re-open it
        if f is not None and f.type is FeatureType.SKETCH and f.sketch is not None:
            self.viewport.enter_sketch(f.id)
            self.sketch_tb.setVisible(True)
            self._sync_sketch_tool_ui(SketchTool.LINE)
            self.statusBar().showMessage(f"Editing {f.name}")
            return
        if f is None or not is_reference_plane(f.type):
            QMessageBox.information(
                self,
                "Sketch",
                "Select a reference plane first, then click Sketch.",
            )
            self.statusBar().showMessage("Select a reference plane to start a sketch", 4000)
            return
        skf = self.doc.create_sketch_on_plane(f.id)
        if skf is None or skf.sketch is None:
            return
        self._refresh_tree()
        self._sync_selection(skf.id)
        self.viewport.enter_sketch(skf.id)
        self.sketch_tb.setVisible(True)
        self._sync_sketch_tool_ui(SketchTool.LINE)
        self.statusBar().showMessage(f"Editing {skf.name} on {f.name}")

    def _exit_sketch(self) -> None:
        self.viewport.exit_sketch()
        self.sketch_tb.setVisible(False)
        self._refresh_tree()
        self._sync_selection(self.doc.selected_id)
        self.statusBar().showMessage("Exited sketch", 2000)

    def _on_sketch_exited(self) -> None:
        self.sketch_tb.setVisible(False)
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

    def _resolve_extrude_sketch_id(self) -> int:
        """Back-compat alias."""
        return self._resolve_closed_sketch_id()

    def _extrude(self) -> None:
        """Extrude (pad) a closed sketch profile via distance dialog + rebuild worker."""
        sid = self._resolve_closed_sketch_id()
        skf = self.doc.find(sid) if sid >= 0 else None
        if skf is None or skf.sketch is None:
            QMessageBox.information(
                self,
                "Extrude",
                "Create or select a sketch with a closed rectangle or circle first.",
            )
            self.statusBar().showMessage(
                "Extrude needs a closed sketch profile", 4000
            )
            return
        if first_closed_profile(skf.sketch) is None:
            QMessageBox.information(
                self,
                "Extrude",
                "The sketch has no closed profile.\n"
                "Draw a rectangle or circle, then Extrude.",
            )
            self.statusBar().showMessage("No closed profile to extrude", 4000)
            return

        dist, ok = QInputDialog.getDouble(
            self,
            "Extrude (Pad)",
            "Distance:",
            1.0,
            1e-6,
            1e6,
            4,
        )
        if not ok:
            return

        # Leave sketch mode so the solid rebuild is visible
        if self.viewport.in_sketch_mode:
            self.viewport.exit_sketch()
            self.sketch_tb.setVisible(False)

        try:
            feat = self.doc.create_extrude(sid, float(dist))
        except ValueError as exc:
            QMessageBox.warning(self, "Extrude", str(exc))
            self.statusBar().showMessage(f"Extrude failed: {exc}", 4000)
            return

        self.viewport.schedule_rebuild()
        self.viewport.refresh_sketches()
        self._refresh_tree()
        self._sync_selection(feat.id)
        self.statusBar().showMessage(
            f"Created {feat.name} (distance={dist:g})", 3000
        )

    def _revolve(self) -> None:
        """Revolve a closed sketch profile about the V-axis via angle dialog + rebuild."""
        sid = self._resolve_closed_sketch_id()
        skf = self.doc.find(sid) if sid >= 0 else None
        if skf is None or skf.sketch is None:
            QMessageBox.information(
                self,
                "Revolve",
                "Create or select a sketch with a closed rectangle or circle first.\n"
                "The profile must lie entirely on one side of the V-axis (u=0).",
            )
            self.statusBar().showMessage(
                "Revolve needs a closed sketch profile", 4000
            )
            return
        if first_closed_profile(skf.sketch) is None:
            QMessageBox.information(
                self,
                "Revolve",
                "The sketch has no closed profile.\n"
                "Draw a rectangle or circle offset from the V-axis, then Revolve.",
            )
            self.statusBar().showMessage("No closed profile to revolve", 4000)
            return

        ang, ok = QInputDialog.getDouble(
            self,
            "Revolve",
            "Angle (degrees):",
            360.0,
            1e-3,
            360.0,
            2,
        )
        if not ok:
            return

        if self.viewport.in_sketch_mode:
            self.viewport.exit_sketch()
            self.sketch_tb.setVisible(False)

        try:
            feat = self.doc.create_revolve(sid, angle_degrees=float(ang))
        except ValueError as exc:
            QMessageBox.warning(self, "Revolve", str(exc))
            self.statusBar().showMessage(f"Revolve failed: {exc}", 4000)
            return

        self.viewport.schedule_rebuild()
        self.viewport.refresh_sketches()
        self._refresh_tree()
        self._sync_selection(feat.id)
        self.statusBar().showMessage(
            f"Created {feat.name} (angle={ang:g}°)", 3000
        )

    def _fillet(self) -> None:
        """Fillet closed profile corners, then extrude via dialogs + rebuild worker."""
        sid = self._resolve_closed_sketch_id()
        skf = self.doc.find(sid) if sid >= 0 else None
        if skf is None or skf.sketch is None:
            QMessageBox.information(
                self,
                "Fillet",
                "Create or select a sketch with a closed rectangle or circle first.",
            )
            self.statusBar().showMessage("Fillet needs a closed sketch profile", 4000)
            return
        if first_closed_profile(skf.sketch) is None:
            QMessageBox.information(
                self,
                "Fillet",
                "The sketch has no closed profile.\n"
                "Draw a rectangle or circle, then Fillet.",
            )
            self.statusBar().showMessage("No closed profile to fillet", 4000)
            return

        radius, ok = QInputDialog.getDouble(
            self,
            "Fillet",
            "Corner radius:",
            0.25,
            1e-6,
            1e6,
            4,
        )
        if not ok:
            return
        dist, ok = QInputDialog.getDouble(
            self,
            "Fillet",
            "Extrude distance:",
            1.0,
            1e-6,
            1e6,
            4,
        )
        if not ok:
            return
        segs, ok = QInputDialog.getInt(
            self,
            "Fillet",
            "Arc segments:",
            32,
            3,
            512,
        )
        if not ok:
            return

        if self.viewport.in_sketch_mode:
            self.viewport.exit_sketch()
            self.sketch_tb.setVisible(False)

        try:
            feat = self.doc.create_fillet(
                sid, float(dist), float(radius), segments=int(segs)
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Fillet", str(exc))
            self.statusBar().showMessage(f"Fillet failed: {exc}", 4000)
            return

        self.viewport.schedule_rebuild()
        self.viewport.refresh_sketches()
        self._refresh_tree()
        self._sync_selection(feat.id)
        self.statusBar().showMessage(
            f"Created {feat.name} (r={radius:g}, segs={segs}, dist={dist:g})",
            3000,
        )

    def _pocket(self) -> None:
        """Circular through-hole pocket + extrude via dialogs and worker rebuild."""
        sid = self._resolve_closed_sketch_id()
        skf = self.doc.find(sid) if sid >= 0 else None
        if skf is None or skf.sketch is None:
            QMessageBox.information(
                self,
                "Pocket",
                "Create or select a sketch with a closed rectangle or circle first.",
            )
            self.statusBar().showMessage("Pocket needs a closed sketch profile", 4000)
            return
        if first_closed_profile(skf.sketch) is None:
            QMessageBox.information(
                self,
                "Pocket",
                "The sketch has no closed profile.\n"
                "Draw a rectangle or circle, then Pocket.",
            )
            self.statusBar().showMessage("No closed profile for pocket", 4000)
            return

        hr, ok = QInputDialog.getDouble(
            self, "Pocket", "Hole radius:", 0.5, 1e-6, 1e6, 4
        )
        if not ok:
            return
        dist, ok = QInputDialog.getDouble(
            self, "Pocket", "Extrude distance:", 1.0, 1e-6, 1e6, 4
        )
        if not ok:
            return
        # Default hole center at profile centroid for rectangles
        cx, cy = 0.0, 0.0
        ent = first_closed_profile(skf.sketch)
        if ent is not None:
            from cadcore.sketch import RectEntity, CircleEntity

            if isinstance(ent, RectEntity):
                cx = 0.5 * (ent.c0[0] + ent.c1[0])
                cy = 0.5 * (ent.c0[1] + ent.c1[1])
            elif isinstance(ent, CircleEntity):
                cx, cy = ent.center[0], ent.center[1]
        cx, ok = QInputDialog.getDouble(
            self, "Pocket", "Hole center U:", cx, -1e6, 1e6, 4
        )
        if not ok:
            return
        cy, ok = QInputDialog.getDouble(
            self, "Pocket", "Hole center V:", cy, -1e6, 1e6, 4
        )
        if not ok:
            return
        segs, ok = QInputDialog.getInt(self, "Pocket", "Hole segments:", 32, 3, 512)
        if not ok:
            return

        if self.viewport.in_sketch_mode:
            self.viewport.exit_sketch()
            self.sketch_tb.setVisible(False)

        try:
            feat = self.doc.create_pocket(
                sid,
                float(dist),
                float(hr),
                (float(cx), float(cy)),
                segments=int(segs),
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Pocket", str(exc))
            self.statusBar().showMessage(f"Pocket failed: {exc}", 4000)
            return

        self.viewport.schedule_rebuild()
        self.viewport.refresh_sketches()
        self._refresh_tree()
        self._sync_selection(feat.id)
        self.statusBar().showMessage(
            f"Created {feat.name} (hole r={hr:g}, dist={dist:g})", 3000
        )

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
            self.statusBar().showMessage("Exit sketch before deleting features", 3000)
            return
        fid = self.doc.selected_id
        f = self.doc.find(fid)
        if f and is_reference_plane(f.type):
            QMessageBox.information(self, "Delete", "Reference planes cannot be deleted.")
            return
        if self.doc.remove_feature(fid):
            self.viewport.schedule_rebuild()
            self.viewport.refresh_sketches()
            self._refresh_tree()
            self._sync_selection(self.doc.selected_id)
            self.statusBar().showMessage("Feature deleted", 2000)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if self.viewport.in_sketch_mode:
            if event.key() == Qt.Key.Key_Escape:
                # Two-stage: cancel draw → Select, or exit sketch if idle
                was_drawing = (
                    self.viewport._sketch_ctrl is not None
                    and self.viewport._sketch_ctrl.is_drawing()
                )
                self.viewport.sketch_escape()
                if not self.viewport.in_sketch_mode:
                    self.sketch_tb.setVisible(False)
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
                self.viewport.sketch_confirm()
                return
        super().keyPressEvent(event)
