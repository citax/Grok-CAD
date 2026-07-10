"""Main application window — sketch-first SolidWorks-style workflow."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QFormLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QWidget,
)

from app.sketch_mode import SketchTool
from app.viewport import Viewport
from cadcore.document import (
    Document,
    FeatureType,
    feature_type_name,
    is_reference_plane,
)


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

        self._build_tree_dock()
        self._build_props_dock()
        self._build_menus()
        self._build_toolbar()
        self._build_sketch_toolbar()

        self.viewport.feature_picked.connect(self._on_pick)
        self.viewport.status_message.connect(self._on_status)
        self.viewport.busy_changed.connect(self._on_busy)
        self.viewport.sketch_exited.connect(self._on_sketch_exited)
        self.viewport.sketch_status.connect(self._on_status)

        for f in self.doc.features:
            if f.type is FeatureType.PLANE_FRONT:
                self.doc.selected_id = f.id
                break
        self._refresh_tree()
        self._sync_selection(self.doc.selected_id)
        self.statusBar().showMessage("Ready — select a plane, then Sketch")

    def _build_tree_dock(self) -> None:
        dock = QDockWidget("Feature Tree", self)
        dock.setObjectName("FeatureTreeDock")
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Features"])
        self.tree.itemSelectionChanged.connect(self._on_tree_sel)
        self.tree.itemDoubleClicked.connect(self._on_tree_double)
        dock.setWidget(self.tree)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def _build_props_dock(self) -> None:
        dock = QDockWidget("Properties", self)
        dock.setObjectName("PropertiesDock")
        w = QWidget()
        form = QFormLayout(w)
        self.prop_name = QLabel("—")
        self.prop_type = QLabel("—")
        form.addRow("Name", self.prop_name)
        form.addRow("Type", self.prop_type)
        form.addRow(QLabel("Select a plane → Sketch to draw."))
        dock.setWidget(w)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _build_menus(self) -> None:
        file_m = self.menuBar().addMenu("&File")
        file_m.addAction("E&xit", QKeySequence.StandardKey.Quit, self.close)

        edit_m = self.menuBar().addMenu("&Edit")
        edit_m.addAction("Delete Feature", QKeySequence.StandardKey.Delete, self._delete_selected)

        view_m = self.menuBar().addMenu("&View")
        for label, key in (
            ("Front", "front"),
            ("Back", "back"),
            ("Top", "top"),
            ("Bottom", "bottom"),
            ("Right", "right"),
            ("Left", "left"),
            ("Isometric", "iso"),
        ):
            view_m.addAction(label, lambda k=key: self.viewport.set_view(k))
        view_m.addSeparator()
        view_m.addAction("Zoom to Fit", QKeySequence("Ctrl+F"), self.viewport.zoom_to_fit)

        insert_m = self.menuBar().addMenu("&Insert")
        act = QAction("Sketch", self)
        act.setShortcut(QKeySequence("S"))
        act.triggered.connect(self._enter_sketch)
        insert_m.addAction(act)

    def _build_toolbar(self) -> None:
        views = QToolBar("Views")
        views.setObjectName("ViewsToolBar")
        self.addToolBar(views)
        for label, key in (
            ("Front", "front"),
            ("Top", "top"),
            ("Right", "right"),
            ("Iso", "iso"),
            ("Fit", "fit"),
        ):
            if key == "fit":
                views.addAction(label, self.viewport.zoom_to_fit)
            else:
                views.addAction(label, lambda k=key: self.viewport.set_view(k))

        main = QToolBar("Main")
        main.setObjectName("MainToolBar")
        self.addToolBar(main)
        self.act_sketch = main.addAction("Sketch", self._enter_sketch)
        self.act_sketch.setToolTip("Create or edit a sketch on the selected plane (S)")

    def _build_sketch_toolbar(self) -> None:
        self.sketch_tb = QToolBar("Sketch")
        self.sketch_tb.setObjectName("SketchToolBar")
        self.addToolBar(self.sketch_tb)
        self.sketch_tb.addAction("Select", lambda: self.viewport.set_sketch_tool(SketchTool.SELECT))
        self.sketch_tb.addAction("Line", lambda: self.viewport.set_sketch_tool(SketchTool.LINE))
        self.sketch_tb.addAction("Rectangle", lambda: self.viewport.set_sketch_tool(SketchTool.RECTANGLE))
        self.sketch_tb.addAction("Circle", lambda: self.viewport.set_sketch_tool(SketchTool.CIRCLE))
        self.sketch_tb.addSeparator()
        self.sketch_tb.addAction("Exit Sketch", self._exit_sketch)
        self.sketch_tb.setVisible(False)

    def _refresh_tree(self) -> None:
        self.tree.blockSignals(True)
        self.tree.clear()
        for f in self.doc.features:
            item = QTreeWidgetItem([f.name])
            item.setData(0, Qt.ItemDataRole.UserRole, f.id)
            self.tree.addTopLevelItem(item)
            if f.id == self.doc.selected_id:
                item.setSelected(True)
                self.tree.setCurrentItem(item)
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
            self.statusBar().showMessage("Selected: (none)")
            return
        self.prop_name.setText(f.name)
        self.prop_type.setText(feature_type_name(f.type))
        self.statusBar().showMessage(f"Selected: {f.name}")
        self.tree.blockSignals(True)
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            item.setSelected(item.data(0, Qt.ItemDataRole.UserRole) == fid)
            if item.isSelected():
                self.tree.setCurrentItem(item)
        self.tree.blockSignals(False)

    def _on_tree_sel(self) -> None:
        if self.viewport.in_sketch_mode:
            return
        items = self.tree.selectedItems()
        if not items:
            return
        fid = int(items[0].data(0, Qt.ItemDataRole.UserRole))
        self._sync_selection(fid)

    def _on_tree_double(self, item: QTreeWidgetItem, _col: int) -> None:
        fid = int(item.data(0, Qt.ItemDataRole.UserRole))
        f = self.doc.find(fid)
        if f and f.type is FeatureType.SKETCH and f.sketch is not None:
            self._sync_selection(fid)
            self.viewport.enter_sketch(f.id)
            self.sketch_tb.setVisible(True)
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

