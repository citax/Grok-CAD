"""Sketch-mode rendering invariants (no hanging full Viewport under offscreen).

Full end-to-end proof is the xcb script that prints SKETCH_ONPLANE_OK through
the real MainWindow.enter_sketch path (on-plane after exit + in-sketch z-order).
These tests lock pure geometry invariants: entity polydata lies exactly on the
sketch plane, flat (non-sphere) point clouds, and overlay-layer helpers exist.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.viewport import (
    _entity_polydata,
    _flat_point_cloud,
)
from cadcore.sketch import PlaneFrame, Sketch

PLANES = ("PLANE_FRONT", "PLANE_TOP", "PLANE_RIGHT")


def _max_plane_deviation(pts: np.ndarray, fr: PlaneFrame) -> float:
    """max |n · (p − origin)| over polydata points."""
    n = np.asarray(fr.normal, float)
    o = np.asarray(fr.origin, float)
    pts = np.asarray(pts, float).reshape(-1, 3)
    if pts.size == 0:
        return 0.0
    return float(np.max(np.abs((pts - o) @ n)))


@pytest.mark.parametrize("plane", PLANES)
def test_line_entity_polydata_on_plane(plane: str):
    fr = PlaneFrame.from_plane_type(plane)
    sk = Sketch(frame=fr)
    line = sk.add_line((-2.0, 0.0), (2.0, 0.0))
    pts = np.asarray(_entity_polydata(line, sk).points, float)
    dev = _max_plane_deviation(pts, fr)
    assert dev < 1e-6, f"{plane} line off-plane max |n·(p-o)|={dev}"


@pytest.mark.parametrize("plane", PLANES)
def test_rect_entity_polydata_on_plane(plane: str):
    fr = PlaneFrame.from_plane_type(plane)
    sk = Sketch(frame=fr)
    rect = sk.add_rectangle((-1.0, -0.5), (1.5, 1.0))
    pts = np.asarray(_entity_polydata(rect, sk).points, float)
    dev = _max_plane_deviation(pts, fr)
    assert dev < 1e-6, f"{plane} rect off-plane max |n·(p-o)|={dev}"


@pytest.mark.parametrize("plane", PLANES)
def test_circle_entity_polydata_on_plane(plane: str):
    fr = PlaneFrame.from_plane_type(plane)
    sk = Sketch(frame=fr)
    circ = sk.add_circle((0.25, -0.3), 0.8)
    pts = np.asarray(_entity_polydata(circ, sk).points, float)
    dev = _max_plane_deviation(pts, fr)
    assert dev < 1e-6, f"{plane} circle off-plane max |n·(p-o)|={dev}"


def test_flat_point_cloud_is_vertices_only():
    cloud = _flat_point_cloud(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    assert cloud.n_points == 2
    assert cloud.n_cells == 0 or cloud.n_verts == cloud.n_points


def test_no_depth_bias_symbols_exported():
    """Geometric depth bias must not exist — overlay layer owns z-order."""
    import app.viewport as vp

    assert not hasattr(vp, "_SKETCH_DEPTH_BIAS")
    assert not hasattr(vp, "_plane_bias")


def test_set_parallel_projection_helper_exists():
    """Viewport exposes parallel projection toggle used by enter/exit sketch."""
    from app.viewport import Viewport

    assert hasattr(Viewport, "_set_parallel_projection")
    assert hasattr(Viewport, "_ensure_sketch_overlay_layer")
    assert hasattr(Viewport, "enter_sketch")
    assert hasattr(Viewport, "exit_sketch")
    assert hasattr(Viewport, "_set_sketch_2d_chrome")
    assert hasattr(Viewport, "_sync_rubber_geometry")
