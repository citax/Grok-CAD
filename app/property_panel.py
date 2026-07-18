"""SolidWorks-style PropertyManager panel for feature / sketch parameters.

Size fields are plain text entry (no spin/slider) so exact values can be typed.
Editors are built fresh on every show — QFormLayout.removeRow deletes widgets.
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
    QVBoxLayout,
    QWidget,
)

from cadcore.document import Document, Feature, FeatureType, is_reference_plane
from cadcore.sketch import LineEntity, line_length, set_line_length, snapshot_entity
from cadcore.units import Unit, format_length, from_mm, parse_length, to_mm


class PropertyPanel(QWidget):
    """Editable feature settings (right dock) — Apply commits + rebuilds."""

    params_applied = Signal(int)  # feature id or sketch feature id
    status_message = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("PropertiesPanel")
        self._doc: Optional[Document] = None
        self._feature_id: int = -1
        self._sketch_line: Optional[tuple] = None  # (sketch_fid, entity_id)
        self._building = False
        self._unit: Unit = Unit.MM
        # Fresh editors for the *current* form only (never survive removeRow)
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

        self.prop_name = QLineEdit()
        self.prop_name.setPlaceholderText("Name")
        self.prop_type = QLabel("—")
        self.prop_type.setObjectName("fieldValue")
        self._form.addRow(self._lbl("Name"), self.prop_name)
        self._form.addRow(self._lbl("Type"), self.prop_type)

        self._dyn_host = QWidget()
        self._dyn_layout = QFormLayout(self._dyn_host)
        self._dyn_layout.setContentsMargins(0, 0, 0, 0)
        self._dyn_layout.setHorizontalSpacing(12)
        self._dyn_layout.setVerticalSpacing(10)
        self._form.addRow(self._dyn_host)

        self._hint = QLabel(
            "Select a feature to edit parameters (Extrude depth, Fillet radius, …)."
        )
        self._hint.setObjectName("secondaryLabel")
        self._hint.setWordWrap(True)
        self._form.addRow(self._hint)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(12, 4, 12, 12)
        self.btn_apply = QPushButton("Apply")
        self.btn_apply.setObjectName("primaryButton")
        self.btn_apply.setEnabled(False)
        self.btn_apply.clicked.connect(self._on_apply)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_apply)
        root.addLayout(btn_row)

    @staticmethod
    def _lbl(text: str) -> QLabel:
        l = QLabel(text)
        l.setObjectName("fieldLabel")
        return l

    def set_document(self, doc: Document) -> None:
        self._doc = doc

    def clear(self) -> None:
        self._building = True
        self._feature_id = -1
        self._sketch_line = None
        self.prop_name.setText("")
        self.prop_type.setText("—")
        self._clear_dyn()
        self.btn_apply.setEnabled(False)
        self._hint.setText("Select a feature or sketch entity to edit parameters.")
        self._building = False

    def _clear_dyn(self) -> None:
        while self._dyn_layout.rowCount():
            self._dyn_layout.removeRow(0)
        self._editors = {}

    def _add_text(
        self, key: str, label: str, value: str, *, placeholder: str = ""
    ) -> QLineEdit:
        """Plain typed field — no spin/slider; user types the exact number."""
        edit = QLineEdit()
        edit.setText(str(value))
        if placeholder:
            edit.setPlaceholderText(placeholder)
        edit.setClearButtonEnabled(False)
        self._dyn_layout.addRow(self._lbl(label), edit)
        self._editors[key] = edit
        return edit

    def _add_length(self, key: str, label: str, value_mm: float, unit: Unit) -> QLineEdit:
        """Length in display units, typed as plain text."""
        v = from_mm(float(value_mm), unit)
        # Compact fixed decimals so the field is type-friendly
        text = f"{v:.6g}"
        return self._add_text(
            key,
            f"{label} ({unit.label})",
            text,
            placeholder=f"e.g. 12.5 or 0.5in",
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
            raise RuntimeError(f"editor {key!r} not present for current form")
        return w

    def _read_length_mm(self, key: str, unit: Unit) -> float:
        edit = self._editor(key)
        assert isinstance(edit, QLineEdit)
        return parse_length(edit.text(), unit)

    def _read_float(self, key: str) -> float:
        edit = self._editor(key)
        assert isinstance(edit, QLineEdit)
        s = (edit.text() or "").strip().replace(",", ".")
        return float(s)

    def _read_int(self, key: str) -> int:
        return int(round(self._read_float(key)))

    def show_feature(self, f: Feature, *, unit: Unit = Unit.MM) -> None:
        self._building = True
        self._unit = unit
        self._feature_id = f.id
        self._sketch_line = None
        self.prop_name.setText(f.name)
        self.prop_type.setText(f.type.name.replace("_", " ").title())
        self._clear_dyn()
        self.btn_apply.setEnabled(not is_reference_plane(f.type))

        if f.type is FeatureType.EXTRUDE:
            self._add_length("depth", "Depth", f.depth, unit)
            self._add_checkbox(
                "reversed",
                "Reverse direction",
                bool(getattr(f, "reversed", False)),
            )
            self._hint.setText(
                "Extrude (Boss/Base) — type an exact depth; "
                "tick Reverse direction to pad along −plane normal."
            )
        elif f.type is FeatureType.CUT_EXTRUDE:
            self._add_length("depth", "Depth", f.depth, unit)
            self._add_checkbox(
                "through_all",
                "Through all",
                bool(getattr(f, "through_all", False)),
            )
            self._add_checkbox(
                "reversed",
                "Reverse direction",
                bool(getattr(f, "reversed", False)),
            )
            self._hint.setText(
                "Cut-Extrude — removes material under the sketch. "
                "Type depth, or enable Through all."
            )
        elif f.type is FeatureType.FILLET:
            self._add_length("radius", "Radius", f.radius, unit)
            self._add_length("depth", "Depth", f.depth, unit)
            self._add_number("segments", "Arc segments", int(f.segments))
            self._hint.setText(
                "Fillet — type corner radius and extrude depth."
            )
        elif f.type is FeatureType.REVOLVE:
            self._add_number("angle", "Angle (°)", float(f.revolve_angle))
            self._hint.setText("Revolve — type angle about the sketch V-axis.")
        elif f.type is FeatureType.POCKET:
            self._add_length("radius", "Hole r", f.radius, unit)
            self._add_length("depth", "Depth", f.depth, unit)
            self._add_length("hole_u", "Center U", f.hole_center_u, unit)
            self._add_length("hole_v", "Center V", f.hole_center_v, unit)
            self._hint.setText("Pocket — type hole radius, center, and depth.")
        elif f.type is FeatureType.SKETCH:
            n = len(f.sketch.entities) if f.sketch else 0
            self._hint.setText(
                f"Sketch — {n} entit{'y' if n == 1 else 'ies'}. "
                "Select a line in sketch mode to set length."
            )
            self.btn_apply.setEnabled(bool(self.prop_name.text()))
        elif is_reference_plane(f.type):
            self._hint.setText("Reference plane — not editable.")
            self.btn_apply.setEnabled(False)
        else:
            self._hint.setText("—")
            self.btn_apply.setEnabled(False)
        self._building = False

    def show_sketch_line(
        self, sketch_fid: int, ent: LineEntity, *, unit: Unit = Unit.MM
    ) -> None:
        self._building = True
        self._unit = unit
        self._feature_id = -1
        self._sketch_line = (int(sketch_fid), int(ent.id))
        self.prop_name.setText(f"Line {ent.id}")
        self.prop_type.setText("Sketch Line")
        self._clear_dyn()
        self._add_length("line_len", "Length", line_length(ent), unit)
        self._hint.setText("Sketch line — type exact length (moves free endpoint p1).")
        self.btn_apply.setEnabled(True)
        self._building = False

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
            self.status_message.emit(
                f"Line length → {format_length(mm, unit)}"
            )
            self.params_applied.emit(sid)
            return

        f = self._doc.find(self._feature_id)
        if f is None:
            return
        name = self.prop_name.text().strip()
        params: dict = {}
        if name and name != f.name:
            params["name"] = name
        try:
            if f.type is FeatureType.EXTRUDE:
                params["depth"] = self._read_length_mm("depth", unit)
                rev = self._editor("reversed")
                assert isinstance(rev, QCheckBox)
                params["reversed"] = bool(rev.isChecked())
            elif f.type is FeatureType.CUT_EXTRUDE:
                params["depth"] = self._read_length_mm("depth", unit)
                ta = self._editor("through_all")
                assert isinstance(ta, QCheckBox)
                params["through_all"] = bool(ta.isChecked())
                rev = self._editor("reversed")
                assert isinstance(rev, QCheckBox)
                params["reversed"] = bool(rev.isChecked())
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
                if name:
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
