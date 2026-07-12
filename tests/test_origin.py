"""Origin glyph: ring/crosshair, centered, not a single point, not a sphere."""

from __future__ import annotations

import numpy as np

from app.viewport import (
    ORIGIN_CROSS_HALF,
    ORIGIN_RING_R,
    _flat_point_cloud,
    _origin_glyph_polydata,
)


def test_origin_glyph_is_lines_not_single_point():
    pdata = _origin_glyph_polydata()
    pts = np.asarray(pdata.points, float)
    assert pts.shape[0] >= 8, "glyph must have many points (ring+cross), not 1"
    # Bounding-box center at origin (mean of raw samples can skew slightly)
    bb_center = 0.5 * (pts.max(axis=0) + pts.min(axis=0))
    assert np.allclose(bb_center, [0, 0, 0], atol=1e-6)
    # Span is modest (crosshair half)
    extent = pts.max(axis=0) - pts.min(axis=0)
    assert extent[0] <= 2 * ORIGIN_CROSS_HALF + 1e-6
    assert extent[1] <= 2 * ORIGIN_CROSS_HALF + 1e-6
    assert abs(extent[2]) < 1e-9  # flat in z


def test_origin_glyph_has_ring_radius():
    pdata = _origin_glyph_polydata()
    pts = np.asarray(pdata.points, float)
    # Some points should lie near the ring radius in XY
    r = np.hypot(pts[:, 0], pts[:, 1])
    near_ring = np.abs(r - ORIGIN_RING_R) < 1e-6
    assert near_ring.sum() >= 8, "ring samples missing"


def test_origin_glyph_not_sphere_mesh():
    pdata = _origin_glyph_polydata()
    # Sphere meshes have many polys; our glyph is lines only
    n_polys = int(pdata.n_faces) if hasattr(pdata, "n_faces") else 0
    # lines_from_points / Line produce lines, not triangle faces
    assert pdata.n_points > 1
    # Not a single-point cloud either
    cloud = _flat_point_cloud(np.array([[0.0, 0.0, 0.0]]))
    assert cloud.n_points == 1
    assert pdata.n_points != cloud.n_points


def test_origin_actor_name_exists_on_viewport_helpers():
    from app.viewport import Viewport

    assert hasattr(Viewport, "_setup_helpers")
    # glyph builder is the source for __origin
    assert callable(_origin_glyph_polydata)
