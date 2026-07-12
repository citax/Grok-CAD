"""Sketch-mode rendering invariants (no hanging full Viewport under offscreen).

Full end-to-end pixel proof is the xcb script that prints SKETCH_2D_OK through
the real MainWindow.enter_sketch path. These tests lock the pure invariants:
parallel projection helpers, depth bias, and flat (non-sphere) point clouds.
"""

from __future__ import annotations

import numpy as np

from app.viewport import (
    _SKETCH_DEPTH_BIAS,
    _entity_polydata,
    _flat_point_cloud,
    _plane_bias,
)
from cadcore.sketch import PlaneFrame, Sketch


def test_entity_bias_toward_camera_vs_axis_back():
    fr = PlaneFrame.from_plane_type("PLANE_FRONT")
    sk = Sketch(frame=fr)
    line = sk.add_line((-2.0, 0.0), (2.0, 0.0))
    pts = np.asarray(_entity_polydata(line, sk).points, float)
    assert pts[:, 2].mean() >= _SKETCH_DEPTH_BIAS * 0.9
    back = _plane_bias(fr, toward_camera=False)
    assert back[2] < 0
    assert _SKETCH_DEPTH_BIAS >= 0.05


def test_flat_point_cloud_is_vertices_only():
    cloud = _flat_point_cloud(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    assert cloud.n_points == 2
    assert cloud.n_cells == 0 or cloud.n_verts == cloud.n_points


def test_bias_applied_to_line_on_u_axis():
    """Line on the u-axis is lifted along +normal so it is not coplanar with axes."""
    fr = PlaneFrame.from_plane_type("PLANE_FRONT")
    sk = Sketch(frame=fr)
    line = sk.add_line((-1.5, 0.0), (1.5, 0.0))
    pts = np.asarray(_entity_polydata(line, sk).points, float)
    # Coplanar axis would have z≈0; entity z ≈ bias
    assert np.allclose(pts[:, 2], _SKETCH_DEPTH_BIAS, atol=1e-6)


def test_set_parallel_projection_helper_exists():
    """Viewport exposes parallel projection toggle used by enter/exit sketch."""
    from app.viewport import Viewport

    assert hasattr(Viewport, "_set_parallel_projection")
    assert hasattr(Viewport, "_ensure_sketch_overlay_layer")
    assert hasattr(Viewport, "enter_sketch")
    assert hasattr(Viewport, "exit_sketch")
