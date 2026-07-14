"""Line rubber-band preview invariants (sibling of OpenGL, post-first-click)."""

from __future__ import annotations

from app.viewport import Viewport, _SketchRubberBand


def test_rubber_band_is_not_child_of_interactor_class():
    """Rubber-band must be a Viewport sibling overlay, not under OpenGL."""
    import inspect

    src = inspect.getsource(Viewport.__init__)
    # Construction must parent to Viewport (self), never plotter.interactor
    assert "_SketchRubberBand(self)" in src or "_SketchRubberBand(self," in src
    assert "_SketchRubberBand(self.plotter.interactor)" not in src


def test_rubber_band_api():
    """API surface only — do not construct QWidget without QApplication."""
    assert hasattr(_SketchRubberBand, "sync_over")
    assert hasattr(_SketchRubberBand, "set_segments")
    assert hasattr(_SketchRubberBand, "clear")
    assert hasattr(_SketchRubberBand, "set_line")


def test_viewport_has_sync_and_2d_chrome():
    assert hasattr(Viewport, "_sync_rubber_geometry")
    assert hasattr(Viewport, "_update_preview_qt")
    assert hasattr(Viewport, "_set_sketch_2d_chrome")
    assert hasattr(Viewport, "_set_parallel_projection")


def test_entity_line_is_polyline_not_volume():
    """Sketch lines are 2D poly lines (2 points), not solid meshes."""
    import numpy as np

    from app.viewport import _entity_polydata
    from cadcore.sketch import PlaneFrame, Sketch

    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    line = sk.add_line((0.0, 0.0), (1.0, 0.5))
    pdata = _entity_polydata(line, sk)
    assert pdata.n_points == 2
    # No solid cells — pure line
    assert pdata.n_cells >= 1
    assert pdata.n_points == 2
