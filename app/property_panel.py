"""SolidWorks-style PropertyManager.

Modes:
  * empty — nothing selected, no form fields
  * plane — reference plane (read-only info)
  * feature — edit an existing feature (Apply)
  * sketch line — edit line length (Apply)
  * command — create a new feature (OK / Cancel)
"""

from __future__ import annotations

from typing import Dict, Optional, Union

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from cadcore.document import Document, Feature, FeatureType, is_reference_plane
from cadcore.sketch import LineEntity, line_length, set_line_length, snapshot_entity
from cadcore.units import Unit, format_length, from_mm, parse_length, to_mm


class PropertyPanel(QWidget):
    params_applied = Signal(int)
    status_message = Signal(str)
    command_ok = Signal()
    command_cancel = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("PropertiesPanel")
        self._doc: Optional[Document] = None
        self._feature_id: int = -1
        self._sketch_line: Optional[tuple] = None
        self._building = False
        self._unit: Unit = Unit.MM
        self._mode: str = "empty"  # empty | plane | feature | line | command
        self._command: Optional[str] = None
        self._editors: Dict[str, Union[QLineEdit, QCheckBox]] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        body = QWidget()
        self._form = QFormLayout(body)
        self._form.setContentsMargins(12, 12, 12, 12)
        self._form.setHorizontalSpacing(12)
        self._form.setVerticalSpacing(10)
        self._form.setLabelAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )
        scroll.setWidget(body)
        root.addWidget(scroll, 1)

        self._name_label = self._lbl("Name")
        self.prop_name = QLineEdit()
        self.prop_name.setPlaceholderText("Name")
        self._type_label = self._lbl("Type")
        self.prop_type = QLabel("—")
        self.prop_type.setObjectName("fieldValue")
        self._form.addRow(self._name_label, self.prop_name)
        self._form.addRow(self._type_label, self.prop_type)

        self._selection_label = QLabel("")
        self._selection_label.setObjectName("fieldValue")
        self._selection_label.setWordWrap(True)
        self._selection_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self._selection_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )
        self._selection_label.setMinimumHeight(0)
        self._selection_label.hide()
        self._form.addRow(self._lbl("Selection"), self._selection_label)

        self._dyn_host = QWidget()
        self._dyn_layout = QFormLayout(self._dyn_host)
        self._dyn_layout.setContentsMargins(0, 0, 0, 0)
        self._dyn_layout.setHorizontalSpacing(12)
        self._dyn_layout.setVerticalSpacing(10)
        self._form.addRow(self._dyn_host)

        self._hint = QLabel("Nothing selected.")
        self._hint.setObjectName("secondaryLabel")
        self._hint.setWordWrap(True)
        self._hint.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self._hint.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )
        self._hint.setMinimumHeight(0)
        self._form.addRow(self._hint)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(12, 4, 12, 12)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self._on_cancel)
        self.btn_cancel.hide()
        self.btn_ok = QPushButton("OK")
        self.btn_ok.setObjectName("primaryButton")
        self.btn_ok.clicked.connect(self._on_ok)
        self.btn_ok.hide()
        self.btn_apply = QPushButton("Apply")
        self.btn_apply.setObjectName("primaryButton")
        self.btn_apply.setEnabled(False)
        self.btn_apply.clicked.connect(self._on_apply)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_cancel)
        btn_row.addWidget(self.btn_ok)
        btn_row.addWidget(self.btn_apply)
        root.addLayout(btn_row)

        self.show_empty()

    @staticmethod
    def _lbl(text: str) -> QLabel:
        l = QLabel(text)
        l.setObjectName("fieldLabel")
        return l

    def _fit_wrapped_label(self, label: QLabel, *, min_lines: int = 1) -> None:
        """Ensure wrapped labels are tall enough that text is never vertically clipped.

        QFormLayout often sizes multi-line QLabels before the final width is known,
        which clips the bottom of the glyphs. heightForWidth fixes that.
        """
        label.setWordWrap(True)
        # Prefer the label's current width; fall back to a typical PropertyManager width
        w = int(label.width())
        if w < 40:
            # Dock panel body is typically ~220–280px; leave room for the form label
            host_w = int(self.width()) if self.width() > 80 else 260
            w = max(160, host_w - 100)
        # Qt heightForWidth includes the text layout; add padding for the styled box
        h = int(label.heightForWidth(w))
        fm = label.fontMetrics()
        line_h = max(fm.height() + fm.leading(), 16)
        pad = 16  # matches fieldValue vertical padding (≈8 top + 8 bottom)
        floor = min_lines * line_h + pad
        label.setMinimumHeight(max(h + 4, floor))
        label.updateGeometry()

    def set_document(self, doc: Document) -> None:
        self._doc = doc

    def _set_header_visible(self, on: bool) -> None:
        self._name_label.setVisible(on)
        self.prop_name.setVisible(on)
        self._type_label.setVisible(on)
        self.prop_type.setVisible(on)

    def _set_mode_buttons(self, *, apply: bool = False, ok: bool = False) -> None:
        self.btn_apply.setVisible(apply)
        self.btn_apply.setEnabled(apply)
        self.btn_ok.setVisible(ok)
        self.btn_ok.setEnabled(ok)
        self.btn_cancel.setVisible(ok)

    def _clear_dyn(self) -> None:
        while self._dyn_layout.rowCount():
            self._dyn_layout.removeRow(0)
        self._editors = {}

    def show_empty(self) -> None:
        """Nothing selected — no editable form."""
        self._building = True
        self._mode = "empty"
        self._command = None
        self._feature_id = -1
        self._sketch_line = None
        self._set_header_visible(False)
        self._selection_label.hide()
        self._clear_dyn()
        self.prop_name.setText("")
        self.prop_type.setText("")
        self._hint.setText(
            "Nothing selected.\n\n"
            "Select a feature in the tree or viewport, or start a command "
            "(Sketch, Extrude, Cut, Fillet…)."
        )
        self._fit_wrapped_label(self._hint, min_lines=3)
        self._set_mode_buttons(apply=False, ok=False)
        self._building = False

    def show_plane_info(self, f: Feature) -> None:
        """Reference plane — fixed fixtures, not user-editable."""
        self._building = True
        self._mode = "plane"
        self._command = None
        self._feature_id = f.id
        self._sketch_line = None
        self._set_header_visible(True)
        self._selection_label.hide()
        self.prop_name.setText(f.name)
        self.prop_name.setReadOnly(True)
        self.prop_type.setText(f.type.name.replace("_", " ").title())
        self._clear_dyn()
        self._hint.setText(
            "Reference plane — fixed document geometry.\n"
            "Not renameable or editable. Select a plane and click Sketch to draw on it."
        )
        self._fit_wrapped_label(self._hint, min_lines=2)
        self._set_mode_buttons(apply=False, ok=False)
        self._building = False

    def _add_text(
        self, key: str, label: str, value: str, *, placeholder: str = ""
    ) -> QLineEdit:
        edit = QLineEdit()
        edit.setText(str(value))
        if placeholder:
            edit.setPlaceholderText(placeholder)
        self._dyn_layout.addRow(self._lbl(label), edit)
        self._editors[key] = edit
        return edit

    def _add_length(self, key: str, label: str, value_mm: float, unit: Unit) -> QLineEdit:
        v = from_mm(float(value_mm), unit)
        return self._add_text(
            key,
            f"{label} ({unit.label})",
            f"{v:.6g}",
            placeholder="e.g. 12.5 or 0.5in",
        )

    def _add_number(self, key: str, label: str, value: float) -> QLineEdit:
        return self._add_text(key, label, f"{float(value):.6g}")

    def _add_checkbox(self, key: str, label: str, checked: bool) -> QCheckBox:
        cb = QCheckBox(label)
        cb.setChecked(bool(checked))
        self._dyn_layout.addRow(self._lbl(""), cb)
        self._editors[key] = cb
        return cb

    def _editor(self, key: str) -> Union[QLineEdit, QCheckBox]:
        w = self._editors.get(key)
        if w is None:
            raise RuntimeError(f"editor {key!r} not present")
        return w

    def _read_length_mm(self, key: str, unit: Unit) -> float:
        edit = self._editor(key)
        assert isinstance(edit, QLineEdit)
        return parse_length(edit.text(), unit)

    def _read_float(self, key: str) -> float:
        edit = self._editor(key)
        assert isinstance(edit, QLineEdit)
        return float((edit.text() or "").strip().replace(",", "."))

    def _read_int(self, key: str) -> int:
        return int(round(self._read_float(key)))

    def _read_bool(self, key: str) -> bool:
        cb = self._editor(key)
        assert isinstance(cb, QCheckBox)
        return bool(cb.isChecked())

    def show_feature(self, f: Feature, *, unit: Unit = Unit.MM) -> None:
        if is_reference_plane(f.type):
            self.show_plane_info(f)
            return
        self._building = True
        self._mode = "feature"
        self._command = None
        self._unit = unit
        self._feature_id = f.id
        self._sketch_line = None
        self._set_header_visible(True)
        self._selection_label.hide()
        self.prop_name.setReadOnly(False)
        self.prop_name.setText(f.name)
        self.prop_type.setText(f.type.name.replace("_", " ").title())
        self._clear_dyn()
        self._set_mode_buttons(apply=True, ok=False)

        if f.type is FeatureType.EXTRUDE:
            self._add_length("depth", "Depth", f.depth, unit)
            self._add_checkbox("reversed", "Reverse direction", bool(f.reversed))
            self._hint.setText("Edit extrude depth or reverse direction, then Apply.")
        elif f.type is FeatureType.CUT_EXTRUDE:
            self._add_length("depth", "Depth", f.depth, unit)
            self._add_checkbox("through_all", "Through all", bool(f.through_all))
            self._add_checkbox("reversed", "Reverse direction", bool(f.reversed))
            self._hint.setText("Edit cut depth / through-all, then Apply.")
        elif f.type is FeatureType.EDGE_FILLET:
            n = len(getattr(f, "edge_keys", None) or [])
            self._add_length("radius", "Radius", f.radius, unit)
            self._add_number("segments", "Arc segments", int(f.segments))
            self._hint.setText(
                f"Solid edge fillet — {n} edge{'s' if n != 1 else ''}. "
                "Edit radius, then Apply."
            )
        elif f.type is FeatureType.FILLET:
            self._add_length("radius", "Radius", f.radius, unit)
            self._add_length("depth", "Depth", f.depth, unit)
            self._add_number("segments", "Arc segments", int(f.segments))
            self._hint.setText("Edit fillet radius and depth, then Apply.")
        elif f.type is FeatureType.REVOLVE:
            self._add_number("angle", "Angle (°)", float(f.revolve_angle))
            self._hint.setText("Edit revolve angle, then Apply.")
        elif f.type is FeatureType.POCKET:
            self._add_length("radius", "Hole r", f.radius, unit)
            self._add_length("depth", "Depth", f.depth, unit)
            self._add_length("hole_u", "Center U", f.hole_center_u, unit)
            self._add_length("hole_v", "Center V", f.hole_center_v, unit)
            self._hint.setText("Edit pocket parameters, then Apply.")
        elif f.type is FeatureType.SKETCH:
            n = len(f.sketch.entities) if f.sketch else 0
            self._hint.setText(
                f"Sketch — {n} entit{'y' if n == 1 else 'ies'}. "
                "Double-click to edit. Rename and Apply, or select a line for length."
            )
            self.btn_apply.setEnabled(True)
        else:
            self._hint.setText("—")
            self.btn_apply.setEnabled(False)
        self._fit_wrapped_label(self._hint, min_lines=2)
        self._building = False

    def show_sketch_line(
        self, sketch_fid: int, ent: LineEntity, *, unit: Unit = Unit.MM
    ) -> None:
        self._building = True
        self._mode = "line"
        self._command = None
        self._unit = unit
        self._feature_id = -1
        self._sketch_line = (int(sketch_fid), int(ent.id))
        self._set_header_visible(True)
        self._selection_label.hide()
        self.prop_name.setReadOnly(True)
        self.prop_name.setText(f"Line {ent.id}")
        self.prop_type.setText("Sketch Line")
        self._clear_dyn()
        self._add_length("line_len", "Length", line_length(ent), unit)
        self._hint.setText("Type exact length, then Apply.")
        self._fit_wrapped_label(self._hint, min_lines=1)
        self._set_mode_buttons(apply=True, ok=False)
        self._building = False

    def show_command(
        self,
        command: str,
        *,
        title: str,
        selection_text: str,
        unit: Unit = Unit.MM,
        defaults: Optional[dict] = None,
        ready: bool = False,
    ) -> None:
        """Active feature command — settings + OK/Cancel (SolidWorks PM style)."""
        self._building = True
        self._mode = "command"
        self._command = command
        self._unit = unit
        self._feature_id = -1
        self._sketch_line = None
        defaults = defaults or {}
        self._set_header_visible(True)
        self.prop_name.setReadOnly(True)
        self.prop_name.setText(title)
        self.prop_type.setText("Command")
        self._selection_label.setText(selection_text or "Nothing selected yet.")
        self._selection_label.show()
        self._fit_wrapped_label(self._selection_label, min_lines=2)
        self._clear_dyn()

        if command in ("extrude", "cut"):
            self._add_length("depth", "Depth", float(defaults.get("depth", 10.0)), unit)
            if command == "cut":
                self._add_checkbox(
                    "through_all", "Through all", bool(defaults.get("through_all", False))
                )
            self._add_checkbox(
                "reversed", "Reverse direction", bool(defaults.get("reversed", False))
            )
        elif command == "fillet":
            # Solid edge fillet — radius only (no sketch depth)
            self._add_length("radius", "Radius", float(defaults.get("radius", 2.0)), unit)
            self._add_number("segments", "Arc segments", int(defaults.get("segments", 32)))
        elif command == "revolve":
            self._add_number("angle", "Angle (°)", float(defaults.get("angle", 360.0)))
        elif command == "pocket":
            self._add_length("radius", "Hole r", float(defaults.get("radius", 5.0)), unit)
            self._add_length("depth", "Depth", float(defaults.get("depth", 10.0)), unit)
            self._add_length(
                "hole_u", "Center U", float(defaults.get("hole_u", 0.0)), unit
            )
            self._add_length(
                "hole_v", "Center V", float(defaults.get("hole_v", 0.0)), unit
            )

        if ready:
            self._hint.setText("Adjust settings, then OK to apply — or Cancel.")
        else:
            if command == "fillet":
                self._hint.setText(
                    "Click edges on a solid to fillet, set the radius, "
                    "then press OK. Cancel exits the command."
                )
            else:
                self._hint.setText(
                    "Select what this feature acts on (sketch / solid), "
                    "then adjust settings and press OK. Cancel exits the command."
                )
        self._fit_wrapped_label(self._hint, min_lines=2)
        self._set_mode_buttons(apply=False, ok=ready)
        # Allow OK only when selection is ready; user can still cancel
        self.btn_ok.setEnabled(ready)
        self.btn_ok.setVisible(True)
        self.btn_cancel.setVisible(True)
        self._building = False

    def update_command_selection(self, selection_text: str, *, ready: bool) -> None:
        if self._mode != "command":
            return
        self._selection_label.setText(selection_text)
        self._fit_wrapped_label(self._selection_label, min_lines=2)
        self.btn_ok.setEnabled(ready)
        if ready:
            self._hint.setText("Selection ready. Adjust settings, then OK — or Cancel.")
        else:
            if self._command == "fillet":
                self._hint.setText(
                    "Click edges on a solid to fillet, then OK. Cancel exits."
                )
            else:
                self._hint.setText(
                    "Select what this feature acts on, then OK. Cancel exits."
                )
        self._fit_wrapped_label(self._hint, min_lines=2)

    def read_command_params(self) -> dict:
        """Read typed values for the active command. Raises ValueError if invalid."""
        if self._mode != "command" or not self._command:
            raise ValueError("no active command")
        unit = self._unit
        cmd = self._command
        out: dict = {}
        if cmd in ("extrude", "cut"):
            out["depth"] = self._read_length_mm("depth", unit)
            out["reversed"] = self._read_bool("reversed")
            if cmd == "cut":
                out["through_all"] = self._read_bool("through_all")
        elif cmd == "fillet":
            out["radius"] = self._read_length_mm("radius", unit)
            out["segments"] = self._read_int("segments")
        elif cmd == "revolve":
            out["angle"] = self._read_float("angle")
        elif cmd == "pocket":
            out["radius"] = self._read_length_mm("radius", unit)
            out["depth"] = self._read_length_mm("depth", unit)
            out["hole_u"] = self._read_length_mm("hole_u", unit)
            out["hole_v"] = self._read_length_mm("hole_v", unit)
        return out

    def clear(self) -> None:
        self.show_empty()

    def _on_cancel(self) -> None:
        if self._mode == "command":
            self.command_cancel.emit()

    def _on_ok(self) -> None:
        if self._mode == "command":
            self.command_ok.emit()

    def _on_apply(self) -> None:
        if self._doc is None or self._building:
            return
        unit = self._doc.display_unit
        if self._sketch_line is not None:
            sid, eid = self._sketch_line
            skf = self._doc.find(sid)
            if skf is None or skf.sketch is None:
                return
            ent = skf.sketch.find_entity(eid)
            if not isinstance(ent, LineEntity):
                return
            try:
                mm = self._read_length_mm("line_len", unit)
            except ValueError as exc:
                self.status_message.emit(f"Invalid length: {exc}")
                return
            before = snapshot_entity(ent)
            set_line_length(ent, mm, free_end="p1")
            after = snapshot_entity(ent)
            self._doc.record_entity_move(sid, before, after)
            self.status_message.emit(f"Line length → {format_length(mm, unit)}")
            self.params_applied.emit(sid)
            return

        f = self._doc.find(self._feature_id)
        if f is None or is_reference_plane(f.type):
            return
        name = self.prop_name.text().strip()
        params: dict = {}
        if name and name != f.name and not self.prop_name.isReadOnly():
            params["name"] = name
        try:
            if f.type is FeatureType.EXTRUDE:
                params["depth"] = self._read_length_mm("depth", unit)
                params["reversed"] = self._read_bool("reversed")
            elif f.type is FeatureType.CUT_EXTRUDE:
                params["depth"] = self._read_length_mm("depth", unit)
                params["through_all"] = self._read_bool("through_all")
                params["reversed"] = self._read_bool("reversed")
            elif f.type is FeatureType.EDGE_FILLET:
                params["radius"] = self._read_length_mm("radius", unit)
                params["segments"] = self._read_int("segments")
            elif f.type is FeatureType.FILLET:
                params["radius"] = self._read_length_mm("radius", unit)
                params["depth"] = self._read_length_mm("depth", unit)
                params["segments"] = self._read_int("segments")
            elif f.type is FeatureType.REVOLVE:
                params["revolve_angle"] = self._read_float("angle")
            elif f.type is FeatureType.POCKET:
                params["radius"] = self._read_length_mm("radius", unit)
                params["depth"] = self._read_length_mm("depth", unit)
                params["hole_center_u"] = self._read_length_mm("hole_u", unit)
                params["hole_center_v"] = self._read_length_mm("hole_v", unit)
            elif f.type is FeatureType.SKETCH:
                if name and name != f.name:
                    f.name = name
                    self._doc.mark_dirty()
                    self.status_message.emit(f"Renamed → {name}")
                    self.params_applied.emit(f.id)
                return
        except ValueError as exc:
            self.status_message.emit(f"Invalid parameter: {exc}")
            return
        except Exception as exc:  # noqa: BLE001
            self.status_message.emit(f"Invalid parameter: {exc}")
            return

        if not params:
            self.status_message.emit("No changes")
            return
        try:
            ok = self._doc.update_feature_params(f.id, **params)
        except ValueError as exc:
            self.status_message.emit(f"Apply failed: {exc}")
            return
        if ok:
            self.status_message.emit(f"Updated {f.name}")
            self.params_applied.emit(f.id)
        else:
            self.status_message.emit("No changes")
