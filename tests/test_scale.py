"""Adaptive scene scale: axes, planes, sketch grid stay readable at real sizes."""

from __future__ import annotations

import math

import pytest

from cadcore.scale import (
    axis_length_mm,
    characteristic_length_from_bounds,
    nice_step,
    origin_glyph_sizes,
    plane_half_mm,
    sketch_entity_uv_extent,
    sketch_grid_params,
    sketch_parallel_scale,
)
from cadcore.sketch import PlaneFrame, Sketch


def test_nice_step_1_2_5():
    assert nice_step(3.7) == 5.0
    assert nice_step(47) == 50.0
    assert nice_step(0.3) == 0.2 or nice_step(0.3) == 0.5
    assert nice_step(100) == 100.0
    assert nice_step(1e-20) == 1.0


def test_axis_scales_with_part_size():
    small = axis_length_mm(10.0)
    large = axis_length_mm(400.0)
    assert large > small
    assert large == pytest.approx(400.0 * 0.22, rel=0.05)
    # Tiny scenes still get a usable axis
    assert axis_length_mm(1.0) >= 8.0


def test_plane_half_grows_with_char():
    assert plane_half_mm(50.0) < plane_half_mm(500.0)
    assert plane_half_mm(500.0) >= 200.0


def test_sketch_grid_readable_at_metre_scale():
    half, step = sketch_grid_params(250.0, entity_extent_mm=200.0)
    # Grid covers the view
    assert half >= 250.0
    # Step is a nice number and not microscopic noise
    assert step >= 10.0
    n = half / step
    assert 4 <= n <= 30


def test_sketch_grid_readable_at_mm_scale():
    half, step = sketch_grid_params(8.0, entity_extent_mm=5.0)
    assert half >= 8.0
    assert step >= 0.5
    assert half / step <= 30


def test_sketch_entity_extent_and_parallel_scale():
    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    sk.add_rectangle((0, 0), (80, 40))
    ext = sketch_entity_uv_extent(sk.entities)
    assert ext == pytest.approx(0.5 * math.hypot(80, 40), rel=1e-6)
    ps = sketch_parallel_scale(ext)
    assert ps >= ext


def test_origin_glyph_sizes_track_axes():
    h1, r1 = origin_glyph_sizes(50.0)
    h2, r2 = origin_glyph_sizes(500.0)
    assert h2 > h1 and r2 > r1


def test_characteristic_from_bounds():
    # 100×50×20 box diagonal
    d = characteristic_length_from_bounds((0, 100, 0, 50, 0, 20))
    assert d == pytest.approx(math.sqrt(100**2 + 50**2 + 20**2), rel=1e-9)
