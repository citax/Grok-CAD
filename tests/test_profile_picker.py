"""Disjoint profile picker includes virtual ClosedLineLoops."""

from __future__ import annotations

from cadcore.document import Document, FeatureType, resolve_profiles
from cadcore.profiles import ClosedLineLoop, find_closed_line_loops, list_closed_profiles
from cadcore.sketch import PlaneFrame, Sketch


def _add_square(sk: Sketch, x0: float, y0: float, s: float = 1.0) -> None:
    sk.add_line((x0, y0), (x0 + s, y0))
    sk.add_line((x0 + s, y0), (x0 + s, y0 + s))
    sk.add_line((x0 + s, y0 + s), (x0, y0 + s))
    sk.add_line((x0, y0 + s), (x0, y0))


def test_two_line_loops_listed():
    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    _add_square(sk, 0.0, 0.0, 2.0)
    _add_square(sk, 5.0, 0.0, 1.0)
    profs = list_closed_profiles(sk)
    loops = [p for p in profs if isinstance(p, ClosedLineLoop)]
    assert len(loops) == 2
    areas = sorted(p.area() for p in loops)
    assert abs(areas[0] - 1.0) < 1e-9
    assert abs(areas[1] - 4.0) < 1e-9


def test_picker_labels_cover_line_loops():
    from app.mainwindow import MainWindow

    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    _add_square(sk, 0, 0, 2)
    _add_square(sk, 4, 0, 2)
    labels = [MainWindow._profile_label(p) for p in list_closed_profiles(sk)]
    assert len(labels) == 2
    assert all("Line loop" in lab for lab in labels)
    assert all("4 segments" in lab for lab in labels)


def test_select_either_loop_extrudes_correct_volume():
    doc = Document()
    doc.seed_reference_planes()
    front = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(front.id)
    sk = skf.sketch
    _add_square(sk, 0, 0, 2.0)  # area 4
    _add_square(sk, 5, 0, 1.0)  # area 1
    loops = find_closed_line_loops(sk)
    assert len(loops) == 2
    by_area = sorted(loops, key=lambda L: L.area())
    small, large = by_area[0], by_area[1]
    h = 3.0
    f_small = doc.create_extrude(skf.id, h, profile_entity_id=small.id)
    m_small = doc.evaluate_feature(f_small.id)
    assert m_small is not None and m_small.is_watertight()
    assert abs(m_small.volume() - 1.0 * h) / (1.0 * h) < 0.01

    # New sketch for large (avoid feature dependency on same sketch rebuild ambiguity)
    skf2 = doc.create_sketch_on_plane(front.id)
    _add_square(skf2.sketch, 0, 0, 2.0)
    _add_square(skf2.sketch, 5, 0, 1.0)
    loops2 = find_closed_line_loops(skf2.sketch)
    large2 = max(loops2, key=lambda L: L.area())
    f_large = doc.create_extrude(skf2.id, h, profile_entity_id=large2.id)
    m_large = doc.evaluate_feature(f_large.id)
    assert m_large is not None and m_large.is_watertight()
    assert abs(m_large.volume() - 4.0 * h) / (4.0 * h) < 0.01


def test_rect_and_line_loop_both_listed():
    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    sk.add_rectangle((0, 0), (2, 2))
    _add_square(sk, 5, 0, 1.5)
    profs = list_closed_profiles(sk)
    kinds = {type(p).__name__ for p in profs}
    assert "RectEntity" in kinds
    assert "ClosedLineLoop" in kinds
    labels = []
    from app.mainwindow import MainWindow

    for p in profs:
        labels.append(MainWindow._profile_label(p))
    assert any("Rectangle" in lab for lab in labels)
    assert any("Line loop" in lab for lab in labels)


def test_ambiguous_resolve_needs_preferred_id():
    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    _add_square(sk, 0, 0, 1)
    _add_square(sk, 3, 0, 1)
    import pytest

    with pytest.raises(ValueError, match="ambiguous"):
        resolve_profiles(sk)
    loops = find_closed_line_loops(sk)
    r = resolve_profiles(sk, preferred_outer_id=loops[0].id)
    assert isinstance(r.outer, ClosedLineLoop)
    assert r.outer.id == loops[0].id


def test_click_interior_selects_region_not_on_press():
    """Region pick is deferred to release (press alone starts box-select)."""
    from app.sketch_mode import SketchController, SketchTool

    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    r = sk.add_rectangle((0, 0), (2, 2))
    ctrl = SketchController(sk)
    ctrl.set_tool(SketchTool.SELECT)
    msg = ctrl.on_press((1.0, 1.0), display_xy=(100.0, 100.0), shift=False)
    assert msg == "BoxSelect"
    assert ctrl.is_box_selecting()
    assert not ctrl.selected_profile_ids  # not decided yet
    msg = ctrl.on_release((1.0, 1.0), display_xy=(100.0, 100.0), shift=False)
    assert msg and msg.startswith("Selected profile")
    assert r.id in ctrl.selected_profile_ids
    assert not ctrl.selected_ids


def test_drag_from_interior_is_box_select():
    from app.sketch_mode import SketchController, SketchTool

    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    sk.add_rectangle((0, 0), (2, 2))
    # Line fully inside the rect, well away from the press point (1,1)
    line = sk.add_line((0.5, 1.2), (1.5, 1.2))
    ctrl = SketchController(sk)
    ctrl.set_tool(SketchTool.SELECT)
    # Press in open interior (not near handles/edges), then drag a real UV box
    msg = ctrl.on_press((0.3, 0.5), display_xy=(100.0, 200.0), shift=False)
    assert msg == "BoxSelect"
    ctrl.on_move((1.7, 1.5), display_xy=(300.0, 50.0))
    msg = ctrl.on_release((1.7, 1.5), display_xy=(300.0, 50.0), shift=False)
    assert msg and msg.startswith("BoxSelect:window:")
    assert not ctrl.selected_profile_ids
    assert line.id in ctrl.selected_ids


def test_edge_click_selects_entity_not_region():
    from app.sketch_mode import SketchController, SketchTool

    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    r = sk.add_rectangle((0, 0), (2, 2))
    ctrl = SketchController(sk)
    ctrl.set_tool(SketchTool.SELECT)
    # Midpoint of bottom edge
    msg = ctrl.on_press((1.0, 0.0), display_xy=(50.0, 50.0), shift=False)
    assert msg and msg.startswith("Selected entity")
    assert ctrl.selected_ids == {r.id}
    assert not ctrl.selected_profile_ids
    assert ctrl.box_select is None
