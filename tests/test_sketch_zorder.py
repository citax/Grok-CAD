"""Regression: sketch entity lines must win z-order over sketch axes."""

from __future__ import annotations

import numpy as np

from app.theme import SKETCH_COLOR, SKETCH_H
from app.viewport import _SKETCH_DEPTH_BIAS, _entity_polydata
from cadcore.sketch import PlaneFrame, Sketch


def _hex_to_rgb01(hex_color: str):
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def test_sketch_entity_depth_bias_above_plane():
    """Entity geometry is offset along the plane normal vs axis (no bias)."""
    fr = PlaneFrame.from_plane_type("PLANE_FRONT")
    sk = Sketch(frame=fr)
    # Line exactly on u-axis (v=0) — same as __sk_h
    line = sk.add_line((-2.0, 0.0), (2.0, 0.0))
    pdata = _entity_polydata(line, sk)
    pts = np.asarray(pdata.points, dtype=np.float64)
    # Axis line would have z≈0; entity must be pushed along +normal (z for front)
    assert pts[:, 2].mean() > _SKETCH_DEPTH_BIAS * 0.5
    # Endpoints still project to v=0 in plane (u,v) — only normal component shifted
    for p in pts:
        local = fr.to_local(p - fr.normal * _SKETCH_DEPTH_BIAS)
        assert abs(local[1]) < 1e-9


def test_sketch_entity_bias_constant_for_rect_on_axis():
    fr = PlaneFrame.from_plane_type("PLANE_FRONT")
    sk = Sketch(frame=fr)
    # Rectangle with bottom edge on u-axis
    rect = sk.add_rectangle((-1.0, 0.0), (1.0, 1.0))
    pdata = _entity_polydata(rect, sk)
    pts = np.asarray(pdata.points, dtype=np.float64)
    assert np.all(pts[:, 2] > _SKETCH_DEPTH_BIAS * 0.5)


def test_zorder_pixel_offscreen_axis_vs_entity():
    """Offscreen: at u-axis overlap, sketch entity color must win (not axis color).

    Uses a minimal VTK offscreen render when available; otherwise falls back to
    geometric bias assertion (same invariant).
    """
    try:
        import pyvista as pv
    except ImportError:
        pytest_skip = True
    else:
        pytest_skip = False

    fr = PlaneFrame.from_plane_type("PLANE_FRONT")
    sk = Sketch(frame=fr)
    line = sk.add_line((-1.5, 0.0), (1.5, 0.0))
    entity_pd = _entity_polydata(line, sk)
    # Axis without bias at z=0
    axis_pd = pv.Line((-2.0, 0.0, 0.0), (2.0, 0.0, 0.0))

    if pytest_skip:
        # Geometry-only fallback
        assert np.asarray(entity_pd.points)[:, 2].mean() > 0.005
        return

    pl = pv.Plotter(off_screen=True, window_size=(200, 200))
    try:
        pl.set_background("black")
        # Camera looking down -Z at origin (front view)
        pl.camera_position = [(0, 0, 8), (0, 0, 0), (0, 1, 0)]
        pl.add_mesh(axis_pd, color=SKETCH_H, line_width=6, name="axis", render=False)
        pl.add_mesh(
            entity_pd, color=SKETCH_COLOR, line_width=6, name="entity", render=False
        )
        # Prefer entity coincident topology
        try:
            actor = pl.actors["entity"]
            m = actor.GetMapper()
            m.SetResolveCoincidentTopologyToPolygonOffset()
            m.SetRelativeCoincidentTopologyLineOffsetParameters(-4, -4)
        except Exception:
            pass
        pl.show(auto_close=False)
        img = pl.screenshot(return_img=True)
    finally:
        pl.close()

    # Sample center pixel (should be on the line through origin)
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    # Average a small window
    patch = img[cy - 2 : cy + 3, cx - 2 : cx + 3].astype(np.float64)
    mean_rgb = patch.reshape(-1, 3).mean(axis=0) / 255.0
    ent = np.array(_hex_to_rgb01(SKETCH_COLOR))
    ax = np.array(_hex_to_rgb01(SKETCH_H))
    d_ent = float(np.linalg.norm(mean_rgb - ent))
    d_ax = float(np.linalg.norm(mean_rgb - ax))
    # Entity color closer than axis color
    assert d_ent < d_ax, (
        f"center pixel closer to axis than entity: rgb={mean_rgb} "
        f"d_ent={d_ent:.3f} d_ax={d_ax:.3f}"
    )
