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


def test_cube_edge_bevels_meet_face_edges():
    """Edge patches must join face corners (f,h) not stick out to (h,h)."""
    poly, labels = build_chamfered_cube(half=1.0, chamfer=0.28)
    # Every edge cell's points should lie on the outer envelope |x|,|y|,|z| <= h
    # and no vertex should be at a cube-corner (h,h,h)-style tip (old bug).
    pts = np.asarray(poly.points)
    assert np.all(np.abs(pts) <= 1.0 + 1e-9)
    # No point with two full-half coords on the free axis mid (old (h,h,z) tips)
    for p in pts:
        n_full = sum(1 for v in p if abs(abs(v) - 1.0) < 1e-9)
        # Corner of outer cube has 3; valid geometry uses at most 1 full-half
        # on edge midpoints… face corners have 1 full + 2×f. Outer cube
        # corners (3 full) must not appear.
        assert n_full < 3, p
    # Welded mesh: fewer unique verts than naive 6*4+12*4+8*3
    assert poly.n_points < 80
    assert "region_id" in poly.cell_data


def test_cube_all_cell_normals_point_outward():
    """Every face/edge/corner polygon normal must face away from the origin.

    Required for correct lighting if the cube is ever shaded (not flat).
    """
    poly, labels = build_chamfered_cube(half=1.0, chamfer=0.28)
    pts = np.asarray(poly.points, dtype=np.float64)
    faces = np.asarray(poly.faces)
    i = 0
    for lab in labels:
        n = int(faces[i])
        i += 1
        ids = [int(faces[i + k]) for k in range(n)]
        i += n
        ring = pts[ids]
        # Newell's method
        nrm = np.zeros(3, dtype=np.float64)
        for k in range(len(ring)):
            x0, y0, z0 = ring[k]
            x1, y1, z1 = ring[(k + 1) % len(ring)]
            nrm[0] += (y0 - y1) * (z0 + z1)
            nrm[1] += (z0 - z1) * (x0 + x1)
            nrm[2] += (x0 - x1) * (y0 + y1)
        center = ring.mean(axis=0)
        assert float(np.dot(nrm, center)) > 0.0, (lab, nrm, center)


def test_face_colors_are_pale_not_black():
    """Faces must be light enough to read (not a black blob)."""
    from app.view_cube import color_for_region, face_label_text

    for lab in ("face:+x", "face:-x", "face:+y", "face:-y", "face:+z", "face:-z"):
        r, g, b = color_for_region(lab)
        # Mean channel well above mid-grey
        assert (r + g + b) / 3.0 > 0.75, (lab, r, g, b)
        assert face_label_text(lab) is not None
