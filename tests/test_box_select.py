"""SolidWorks-style box multi-select: window vs crossing, multi-id state, multi-delete."""

from __future__ import annotations

import time

import numpy as np

from app.sketch_mode import SketchController, SketchTool
from app.viewport import _dashed_polyline_polydata
from cadcore.document import Document, FeatureType
from cadcore.sketch import PlaneFrame, Sketch


def _ctrl_with_scene():
    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    # Fully inside window box [0,0]–[2,2]
    inside = sk.add_line((0.5, 0.5), (1.5, 0.5))
    # Crosses right edge of [0,0]–[2,2]
    crossing = sk.add_line((1.5, 1.0), (3.0, 1.0))
    # Outside
    outside = sk.add_circle((4.0, 4.0), 0.3)
    # Rect fully inside
    rect = sk.add_rectangle((0.2, 1.2), (1.0, 1.8))
    ctrl = SketchController(sk)
    ctrl.set_tool(SketchTool.SELECT)
    return ctrl, inside, crossing, outside, rect


def test_selected_entity_id_property_over_set():
    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    a = sk.add_line((0, 0), (1, 0))
    b = sk.add_line((0, 1), (1, 1))
    ctrl = SketchController(sk)
    assert ctrl.selected_entity_id == -1
    ctrl.selected_entity_id = a.id
    assert ctrl.selected_ids == {a.id}
    assert ctrl.selected_entity_id == a.id
    ctrl.selected_ids = {a.id, b.id}
    assert ctrl.selected_entity_id == -1  # multi → -1
    ctrl.selected_entity_id = -1
    assert ctrl.selected_ids == set()


def test_set_tool_clears_selection():
    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    a = sk.add_line((0, 0), (1, 0))
    ctrl = SketchController(sk)
    ctrl.selected_ids = {a.id}
    ctrl.set_tool(SketchTool.LINE)
    assert ctrl.selected_ids == set()


def test_window_vs_crossing_different_sets():
    ctrl, inside, crossing, outside, rect = _ctrl_with_scene()
    # Box UV [0,0]–[2,2]
    win = ctrl.entities_in_box(0.0, 0.0, 2.0, 2.0, window=True)
    cross = ctrl.entities_in_box(0.0, 0.0, 2.0, 2.0, window=False)
    assert inside.id in win
    assert rect.id in win
    assert crossing.id not in win  # only partially inside
    assert outside.id not in win
    assert inside.id in cross
    assert rect.id in cross
    assert crossing.id in cross  # touches / partial
    assert outside.id not in cross
    assert win != cross
    assert win < cross  # window is subset of crossing here


def test_circle_crossing_without_center_in_box():
    """Circle whose rim crosses the box but center is outside must be detected."""
    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    # Center at (3, 1), r=1.2 → left rim at x=1.8, so crosses box [0,0]–[2,2]
    circ = sk.add_circle((3.0, 1.0), 1.2)
    ctrl = SketchController(sk)
    win = ctrl.entities_in_box(0.0, 0.0, 2.0, 2.0, window=True)
    cross = ctrl.entities_in_box(0.0, 0.0, 2.0, 2.0, window=False)
    assert circ.id not in win
    assert circ.id in cross


def test_box_select_release_window_and_crossing():
    ctrl, inside, crossing, outside, rect = _ctrl_with_scene()
    # L→R window via display x
    msg = ctrl.on_press((0.0, 0.0), display_xy=(100.0, 200.0), shift=False)
    assert msg == "BoxSelect"
    assert ctrl.is_box_selecting()
    ctrl.on_move((2.0, 2.0), display_xy=(300.0, 50.0))
    msg = ctrl.on_release((2.0, 2.0), display_xy=(300.0, 50.0), shift=False)
    assert msg and msg.startswith("BoxSelect:window:")
    assert ctrl.selected_ids == {inside.id, rect.id}

    # R→L crossing
    ctrl.clear_selection()
    ctrl.on_press((2.0, 0.0), display_xy=(300.0, 200.0), shift=False)
    ctrl.on_move((0.0, 2.0), display_xy=(100.0, 50.0))
    msg = ctrl.on_release((0.0, 2.0), display_xy=(100.0, 50.0), shift=False)
    assert msg and msg.startswith("BoxSelect:crossing:")
    assert inside.id in ctrl.selected_ids
    assert crossing.id in ctrl.selected_ids
    assert rect.id in ctrl.selected_ids
    assert outside.id not in ctrl.selected_ids


def test_box_select_tiny_drag_clears():
    ctrl, inside, *_ = _ctrl_with_scene()
    ctrl.selected_ids = {inside.id}
    ctrl.on_press((5.0, 5.0), display_xy=(400.0, 400.0), shift=False)
    assert ctrl.selected_ids == set()  # cleared on empty press
    # < 3 px drag
    ctrl.on_move((5.01, 5.01), display_xy=(401.0, 401.0))
    msg = ctrl.on_release((5.01, 5.01), display_xy=(401.0, 401.0), shift=False)
    assert msg == "BoxSelectClear"
    assert ctrl.selected_ids == set()


def test_shift_add_box_select():
    ctrl, inside, crossing, outside, rect = _ctrl_with_scene()
    ctrl.selected_ids = {outside.id}
    # Shift + L→R window over inside+rect
    ctrl.on_press((0.0, 0.0), display_xy=(100.0, 200.0), shift=True)
    assert outside.id in ctrl.selected_ids  # baseline kept during drag
    ctrl.on_move((2.0, 2.0), display_xy=(300.0, 50.0))
    ctrl.on_release((2.0, 2.0), display_xy=(300.0, 50.0), shift=True)
    assert outside.id in ctrl.selected_ids
    assert inside.id in ctrl.selected_ids
    assert rect.id in ctrl.selected_ids


def test_click_entity_still_single_selects():
    ctrl, inside, crossing, *_ = _ctrl_with_scene()
    # On line body between mid and end (not on a handle)
    msg = ctrl.on_press((1.2, 0.5), display_xy=(10.0, 10.0), shift=False)
    # Body press selects and starts whole-entity translate drag (daily-driver edit)
    assert msg and (msg.startswith("Selected") or msg.startswith("Drag body"))
    assert ctrl.selected_ids == {inside.id}
    assert ctrl.box_select is None
    assert ctrl.drag is not None
    assert ctrl.drag.handle_name == "__body__"


def test_multi_delete_one_undo():
    doc = Document()
    doc.seed_reference_planes()
    front = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(front.id)
    sk = skf.sketch
    a = sk.add_line((0, 0), (1, 0))
    b = sk.add_line((0, 1), (1, 1))
    c = sk.add_circle((2, 2), 0.5)
    doc.record_entity_add(skf.id, a)
    doc.record_entity_add(skf.id, b)
    doc.record_entity_add(skf.id, c)
    n = doc.delete_entities(skf.id, [a.id, b.id])
    assert n == 2
    assert sk.find_entity(a.id) is None
    assert sk.find_entity(b.id) is None
    assert sk.find_entity(c.id) is not None
    # ONE undo restores BOTH
    assert doc.undo()
    assert sk.find_entity(a.id) is not None
    assert sk.find_entity(b.id) is not None
    assert doc.redo()
    assert sk.find_entity(a.id) is None
    assert sk.find_entity(b.id) is None


def test_dashed_polyline_one_shot_flat_cost():
    """Dash build must be O(1)-ish: large boxes stay under a few ms (no merge loop)."""
    # Closed rectangle corners (world)
    def rect(s: float):
        return [
            (0.0, 0.0, 0.0),
            (s, 0.0, 0.0),
            (s, s, 0.0),
            (0.0, s, 0.0),
            (0.0, 0.0, 0.0),
        ]

    small = _dashed_polyline_polydata(rect(0.5), dash=0.08, gap=0.05, max_dashes=48)
    large = _dashed_polyline_polydata(rect(8.0), dash=0.08, gap=0.05, max_dashes=48)
    assert small.n_cells > 0
    assert large.n_cells > 0
    # Cap keeps large from exploding past ~max_dashes
    assert large.n_cells <= 48 + 4  # small tolerance for edge split

    t0 = time.perf_counter()
    for _ in range(20):
        _dashed_polyline_polydata(rect(8.0), dash=0.08, gap=0.05, max_dashes=48)
    ms = (time.perf_counter() - t0) * 1000.0 / 20.0
    # Pre-fix was 500+ ms for 8x8; post-fix target is sub-millisecond class
    assert ms < 5.0, f"dashed polydata still too slow: {ms:.2f} ms/call"
