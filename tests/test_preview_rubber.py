"""Preview is a VTK overlay actor — no Qt rubber-band widget remains."""

from __future__ import annotations

import app.viewport as vp
from app.viewport import Viewport


def test_no_qt_rubber_band_class():
    assert not hasattr(vp, "_SketchRubberBand")


def test_no_qt_rubber_fields_on_viewport():
    assert not hasattr(Viewport, "_sync_rubber_geometry")
    assert not hasattr(Viewport, "_update_preview_qt")


def test_vtk_preview_path_exists():
    assert hasattr(Viewport, "_update_preview_visual")
    assert hasattr(Viewport, "_add_overlay_mesh")
    assert hasattr(Viewport, "_begin_draw_lod")
    assert hasattr(Viewport, "_end_draw_lod")
    assert hasattr(Viewport, "_set_draw_solids_visible")


def test_entity_line_is_polyline_not_volume():
    """Sketch lines are 2D poly lines (2 points), not solid meshes."""
    from app.viewport import _entity_polydata
    from cadcore.sketch import PlaneFrame, Sketch

    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    line = sk.add_line((0.0, 0.0), (1.0, 0.5))
    pdata = _entity_polydata(line, sk)
    assert pdata.n_points == 2
    assert pdata.n_cells >= 1
