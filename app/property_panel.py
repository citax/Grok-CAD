"""SolidWorks-style PropertyManager panel for feature / sketch parameters.

Editors are built fresh on every show_feature/show_sketch_line. Dynamic rows use
QFormLayout.removeRow(), which deletes widgets — so spin boxes must NOT be
long-lived attributes reused across shows (that caused dangling Shiboken
pointers on the second show).
"""

from __future__ import annotations

from typing import Dict, Optional, Union

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from cadcore.document import Document, Feature, FeatureType, is_reference_plane
from cadcore.sketch import LineEntity, line_length, set_line_length, snapshot_entity
from cadcore.units import Unit, from_mm, to_mm


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
        # Fresh editors for the *current* form only (never survive removeRow)
        self._editors: Dict[str, Union[QDoubleSpinBox, QSpinBox, QCheckBox]] = {}

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

        # Header only — long-lived, never placed in the dynamic layout
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

    @staticmethod
    def _dspin(lo: float, hi: float, dec: int) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(dec)
        s.setSingleStep(0.1 if dec >= 2 else 1.0)
        s.setKeyboardTracking(False)
        return s

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
        """Destroy dynamic rows (labels + fields). Editors are not reused."""
        while self._dyn_layout.rowCount():
            self._dyn_layout.removeRow(0)
        self._editors = {}

    def _add_dspin(
        self, key: str, label: str, value: float, lo: float, hi: float, dec: int
    ) -> QDoubleSpinBox:
        spin = self._dspin(lo, hi, dec)
        spin.setValue(float(value))
        self._dyn_layout.addRow(self._lbl(label), spin)
        self._editors[key] = spin
        return spin

    def _add_ispin(self, key: str, label: str, value: int, lo: int, hi: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(lo, hi)
        spin.setValue(int(value))
        self._dyn_layout.addRow(self._lbl(label), spin)
        self._editors[key] = spin
        return spin

    def _editor(self, key: str) -> Union[QDoubleSpinBox, QSpinBox, QCheckBox]:
        w = self._editors.get(key)
        if w is None:
            raise RuntimeError(f"editor {key!r} not present for current form")
        return w

    def _add_checkbox(self, key: str, label: str, checked: bool) -> QCheckBox:
        cb = QCheckBox(label)
        cb.setChecked(bool(checked))
        self._dyn_layout.addRow(self._lbl(""), cb)
        self._editors[key] = cb
        return cb

    def show_feature(self, f: Feature, *, unit: Unit = Unit.MM) -> None:
        self._building = True
        self._feature_id = f.id
        self._sketch_line = None
        self.prop_name.setText(f.name)
        self.prop_type.setText(f.type.name.replace("_", " ").title())
        self._clear_dyn()
        self.btn_apply.setEnabled(not is_reference_plane(f.type))

        if f.type is FeatureType.EXTRUDE:
            self._add_dspin(
                "depth", f"Depth ({unit.label})", from_mm(f.depth, unit), 0.001, 1e6, 4
            )
            self._add_checkbox(
                "reversed",
                "Reverse direction",
                bool(getattr(f, "reversed", False)),
            )
            self._hint.setText(
                "Extrude (Boss/Base) — depth is always positive; "
                "tick Reverse direction to pad along −plane normal."
            )
        elif f.type is FeatureType.FILLET:
            self._add_dspin(
                "radius",
                f"Radius ({unit.label})",
                from_mm(f.radius, unit),
                0.001,
                1e6,
                4,
            )
            self._add_dspin(
                "depth", f"Depth ({unit.label})", from_mm(f.depth, unit), 0.001, 1e6, 4
            )
            self._add_ispin("segments", "Arc segments", int(f.segments), 3, 512)
            self._hint.setText(
                "Fillet — corner radius (sharp corners removed from sketch) + extrude depth."
            )
        elif f.type is FeatureType.REVOLVE:
            self._add_dspin(
                "angle", "Angle (°)", float(f.revolve_angle), 0.001, 360.0, 2
            )
            self._hint.setText("Revolve — angle about the sketch V-axis.")
        elif f.type is FeatureType.POCKET:
            self._add_dspin(
                "radius",
                f"Hole r ({unit.label})",
                from_mm(f.radius, unit),
                0.001,
                1e6,
                4,
            )
            self._add_dspin(
                "depth", f"Depth ({unit.label})", from_mm(f.depth, unit), 0.001, 1e6, 4
            )
            self._add_dspin(
                "hole_u",
                f"Center U ({unit.label})",
                from_mm(f.hole_center_u, unit),
                -1e6,
                1e6,
                4,
            )
            self._add_dspin(
                "hole_v",
                f"Center V ({unit.label})",
                from_mm(f.hole_center_v, unit),
                -1e6,
                1e6,
                4,
            )
            self._hint.setText("Pocket — through-hole radius, center, and extrude depth.")
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
        self._feature_id = -1
        self._sketch_line = (int(sketch_fid), int(ent.id))
        self.prop_name.setText(f"Line {ent.id}")
        self.prop_type.setText("Sketch Line")
        self._clear_dyn()
        self._add_dspin(
            "line_len",
            f"Length ({unit.label})",
            from_mm(line_length(ent), unit),
            0.001,
            1e9,
            4,
        )
        self._hint.setText("Sketch line — set length (moves free endpoint p1).")
        self.btn_apply.setEnabled(True)
        self._building = False

    def _on_apply(self) -> None:
        if self._doc is None or self._building:
            return
        unit = self._doc.display_unit
        # Sketch line length
        if self._sketch_line is not None:
            sid, eid = self._sketch_line
            skf = self._doc.find(sid)
            if skf is None or skf.sketch is None:
                return
            ent = skf.sketch.find_entity(eid)
            if not isinstance(ent, LineEntity):
                return
            spin = self._editor("line_len")
            assert isinstance(spin, QDoubleSpinBox)
            before = snapshot_entity(ent)
            set_line_length(ent, to_mm(spin.value(), unit), free_end="p1")
            after = snapshot_entity(ent)
            self._doc.record_entity_move(sid, before, after)
            self.status_message.emit(
                f"Line length → {spin.value():g} {unit.label}"
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
                params["depth"] = to_mm(float(self._editor("depth").value()), unit)
                rev = self._editor("reversed")
                assert isinstance(rev, QCheckBox)
                params["reversed"] = bool(rev.isChecked())
            elif f.type is FeatureType.FILLET:
                params["radius"] = to_mm(float(self._editor("radius").value()), unit)
                params["depth"] = to_mm(float(self._editor("depth").value()), unit)
                params["segments"] = int(self._editor("segments").value())
            elif f.type is FeatureType.REVOLVE:
                params["revolve_angle"] = float(self._editor("angle").value())
            elif f.type is FeatureType.POCKET:
                params["radius"] = to_mm(float(self._editor("radius").value()), unit)
                params["depth"] = to_mm(float(self._editor("depth").value()), unit)
                params["hole_center_u"] = to_mm(
                    float(self._editor("hole_u").value()), unit
                )
                params["hole_center_v"] = to_mm(
                    float(self._editor("hole_v").value()), unit
                )
            elif f.type is FeatureType.SKETCH:
                if name:
                    f.name = name
                    self.status_message.emit(f"Renamed → {name}")
                    self.params_applied.emit(f.id)
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
