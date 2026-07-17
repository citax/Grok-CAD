"""Chamfered view cube geometry and region → view mapping."""

from __future__ import annotations

import numpy as np
import pytest

from app.view_cube import build_chamfered_cube, region_to_view


def test_chamfered_cube_topology():
    poly, labels = build_chamfered_cube()
    assert poly.n_cells == 26
    assert sum(1 for l in labels if l.startswith("face:")) == 6
    assert sum(1 for l in labels if l.startswith("edge:")) == 12
    assert sum(1 for l in labels if l.startswith("corner:")) == 8


def test_face_regions_map_to_distinct_views():
    views = {region_to_view(f"face:{s}") for s in ("+x", "-x", "+y", "-y", "+z", "-z")}
    assert views == {"right", "left", "top", "bottom", "front", "back"}


def test_corner_maps_to_iso_token():
    assert region_to_view("corner:+x+y+z") == "iso:+x+y+z"


def test_opposite_faces_are_not_same_view():
    assert region_to_view("face:+x") != region_to_view("face:-x")
    assert region_to_view("face:+z") != region_to_view("face:-z")


def test_cube_has_volume_extent():
    poly, _ = build_chamfered_cube(half=1.0, chamfer=0.3)
    b = poly.bounds
    # spans roughly ±1
    assert b[1] - b[0] > 1.5
    assert b[3] - b[2] > 1.5
    assert b[5] - b[4] > 1.5
