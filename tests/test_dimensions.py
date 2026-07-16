"""Driving dimensions + H/V + Equal constraints."""

from __future__ import annotations

import pytest

from cadcore.document import Document, FeatureType
from cadcore.sketch import (
    CircleEntity,
    LineEntity,
    PlaneFrame,
    RectEntity,
    Sketch,
    apply_dimension_value,
    infer_dimension_role,
    line_length,
    make_line_horizontal,
    make_line_vertical,
    make_lines_equal_length,
    measure_dimension_value,
    rect_height,
    rect_width,
    set_circle_diameter,
)


def test_line_dimension_drives_length():
    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    ln = sk.add_line((0, 0), (3, 4))  # length 5
    assert measure_dimension_value(ln, "length") == pytest.approx(5.0)
    apply_dimension_value(ln, "length", 40.0)
    assert line_length(ln) == pytest.approx(40.0)
    # Direction preserved (3-4-5 → scaled)
    assert abs(ln.p1[0] - ln.p0[0]) == pytest.approx(24.0)
    assert abs(ln.p1[1] - ln.p0[1]) == pytest.approx(32.0)


def test_rect_width_height_drive():
    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    r = sk.add_rectangle((0, 0), (10, 20))
    apply_dimension_value(r, "width", 40.0)
    assert rect_width(r) == pytest.approx(40.0)
    apply_dimension_value(r, "height", 15.0)
    assert rect_height(r) == pytest.approx(15.0)


def test_circle_diameter_drives():
    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    c = sk.add_circle((0, 0), 5.0)
    apply_dimension_value(c, "diameter", 40.0)
    assert c.radius == pytest.approx(20.0)
    set_circle_diameter(c, 10.0)
    assert c.radius == pytest.approx(5.0)


def test_infer_role_for_rect_edges():
    """Clicked edge's length is the dimension — not the perpendicular span."""
    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    # Rect: width=10 (horizontal edges), height=20 (vertical edges)
    r = sk.add_rectangle((0, 0), (10, 20))
    # Bottom / top horizontal edges → width
    assert infer_dimension_role(r, uv_hint=(5.0, 0.0)) == "width"
    assert infer_dimension_role(r, uv_hint=(5.0, 20.0)) == "width"
    assert infer_dimension_role(r, uv_hint=(5.0, -0.1)) == "width"
    # Left / right vertical edges → height
    assert infer_dimension_role(r, uv_hint=(0.0, 10.0)) == "height"
    assert infer_dimension_role(r, uv_hint=(10.0, 10.0)) == "height"
    assert infer_dimension_role(r, uv_hint=(-0.1, 10.0)) == "height"
    assert infer_dimension_role(sk.add_line((0, 0), (1, 0))) == "length"
    assert infer_dimension_role(sk.add_circle((0, 0), 1)) == "diameter"


def test_old_backwards_edge_rule_rejected():
    """The previous inverted rule (vertical→width, horizontal→height) must fail.

    This test exists so that reintroducing the bug cannot hide behind a green suite.
    """
    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    r = sk.add_rectangle((0, 0), (10, 20))
    # Old wrong rule said: near left vertical → "width". Correct is "height".
    left = infer_dimension_role(r, uv_hint=(-0.1, 10.0))
    bottom = infer_dimension_role(r, uv_hint=(5.0, -0.1))
    assert left == "height"
    assert bottom == "width"
    # Explicitly reject the inverted mapping
    assert not (left == "width" and bottom == "height"), (
        "backwards edge rule returned (vertical→width, horizontal→height)"
    )


def test_dimension_tool_infers_role_from_click_not_caller():
    """Smart Dimension tool: role is decided inside on_press from click UV."""
    from app.sketch_mode import SketchController, SketchTool

    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    r = sk.add_rectangle((0.0, 0.0), (30.0, 20.0))
    ctrl = SketchController(sk)
    ctrl.set_tool(SketchTool.DIMENSION)
    # Bottom edge midpoint — must be width (long horizontal edge of a 30×20)
    msg_b = ctrl.on_press((15.0, 0.0))
    assert msg_b == f"DimPick:{r.id}:width", msg_b
    # Left edge midpoint — must be height
    msg_l = ctrl.on_press((0.0, 10.0))
    assert msg_l == f"DimPick:{r.id}:height", msg_l
    # Top and right
    assert ctrl.on_press((15.0, 20.0)) == f"DimPick:{r.id}:width"
    assert ctrl.on_press((30.0, 10.0)) == f"DimPick:{r.id}:height"


def test_document_apply_sketch_dimension_undo():
    doc = Document()
    doc.seed_reference_planes()
    front = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(front.id)
    assert skf and skf.sketch
    ln = skf.sketch.add_line((0, 0), (10, 0))
    doc.record_entity_add(skf.id, ln)
    dim = doc.apply_sketch_dimension(skf.id, ln.id, "length", 40.0)
    assert dim is not None
    assert line_length(ln) == pytest.approx(40.0)
    assert any(d.entity_id == ln.id for d in skf.sketch.dimensions)
    # Labels stay after change
    dim2 = doc.apply_sketch_dimension(skf.id, ln.id, "length", 25.0)
    assert line_length(ln) == pytest.approx(25.0)
    assert dim2 is not None and dim2.id == dim.id  # updated in place
    # Undo restores previous geometry
    assert doc.undo()
    assert line_length(ln) == pytest.approx(40.0)
    assert doc.undo()
    assert line_length(ln) == pytest.approx(10.0)


def test_horizontal_vertical_equal():
    from cadcore.sketch import EntityKind

    a = LineEntity(id=1, kind=EntityKind.LINE, p0=(0, 0), p1=(3, 4))
    make_line_horizontal(a)
    assert abs(a.p0[1] - a.p1[1]) < 1e-12
    assert line_length(a) == pytest.approx(5.0)

    b = LineEntity(id=2, kind=EntityKind.LINE, p0=(0, 0), p1=(3, 4))
    make_line_vertical(b)
    assert abs(b.p0[0] - b.p1[0]) < 1e-12
    assert line_length(b) == pytest.approx(5.0)

    c = LineEntity(id=3, kind=EntityKind.LINE, p0=(0, 0), p1=(1, 0))
    make_lines_equal_length(a, c)
    assert line_length(c) == pytest.approx(line_length(a))


def test_dimension_persists_on_sketch():
    sk = Sketch()
    r = sk.add_rectangle((0, 0), (5, 5))
    d = sk.add_or_update_dimension(r.id, "width", 5.0)
    apply_dimension_value(r, "width", 40.0)
    d.value_mm = 40.0
    assert sk.dimensions_for_entity(r.id)[0].value_mm == 40.0
    sk.remove_dimensions_for_entity(r.id)
    assert sk.dimensions_for_entity(r.id) == []
