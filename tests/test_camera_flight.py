"""Camera flight path: no gratuitous roll on edge/face transitions."""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.viewport import Viewport
from app.view_cube import build_chamfered_cube, region_to_view


def _iso_pose(dist: float = 50.0):
    d = np.array([0.75, 0.55, 0.75], dtype=np.float64)
    d = d / np.linalg.norm(d) * dist
    return d, np.zeros(3), np.array([0.0, 1.0, 0.0])


def _direction_pose(direction, dist: float = 50.0):
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    d = d / np.linalg.norm(d)
    up = np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(d, up))) > 0.9:
        up = np.array([0.0, 0.0, -1.0 if d[1] > 0 else 1.0])
    return d * dist, np.zeros(3), up


def _edge_direction(signs: str):
    sx = 1.0 if "+x" in signs else (-1.0 if "-x" in signs else 0.0)
    sy = 1.0 if "+y" in signs else (-1.0 if "-y" in signs else 0.0)
    sz = 1.0 if "+z" in signs else (-1.0 if "-z" in signs else 0.0)
    v = np.array([sx, sy, sz], dtype=np.float64)
    return v / np.linalg.norm(v)


def test_quat_roundtrip_preserves_basis():
    r = np.array([1.0, 0.0, 0.0])
    u = np.array([0.0, 1.0, 0.0])
    b = np.array([0.0, 0.0, 1.0])
    q = Viewport._basis_to_quat(r, u, b)
    r2, u2, b2 = Viewport._quat_to_basis(q)
    assert np.allclose(r, r2, atol=1e-9)
    assert np.allclose(u, u2, atol=1e-9)
    assert np.allclose(b, b2, atol=1e-9)


def test_shortest_slerp_does_not_flip_hemisphere():
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    q1 = np.array([-0.999, 0.0447, 0.0, 0.0])  # nearly opposite sign of identity
    q1 = q1 / np.linalg.norm(q1)
    # Without shortest-path, slerp would take the long way; with it, stays near q0 early
    q = Viewport._quat_slerp(q0, q1, 0.1)
    assert float(np.dot(q, q0)) > 0.9


def test_all_cube_edges_roll_path_near_endpoint_need():
    """For each edge flight from iso: path roll ≈ endpoint up-angle (no 270° spin)."""
    _poly, labels = build_chamfered_cube()
    edges = [lab for lab in labels if lab.startswith("edge:")]
    assert len(edges) == 12
    start_pos, start_foc, start_up = _iso_pose()
    report = []
    for lab in edges:
        token = region_to_view(lab)
        assert token and token.startswith("edge:")
        signs = token.split(":", 1)[1]
        end_pos, end_foc, end_up = _direction_pose(_edge_direction(signs))
        path, need = Viewport.measure_camera_up_path(
            start_pos, start_foc, start_up, end_pos, end_foc, end_up, samples=48
        )
        report.append((lab, path, need, path - need))
        # Path must not wind far beyond what the endpoints require.
        # A modest overhead is normal (look changes, up is one axis of a
        # rotating frame). The old bug was ~270–320° extras — never that.
        assert path <= need + math.radians(25.0), (
            f"{lab}: path_roll={math.degrees(path):.1f}° "
            f"endpoint_need={math.degrees(need):.1f}°"
        )
        assert path < math.radians(120.0), (
            f"{lab}: path_roll={math.degrees(path):.1f}° looks like the old spin"
        )
        # And if endpoints agree on up, path roll should be near zero
        if need < math.radians(5.0):
            assert path < math.radians(10.0), (
                f"{lab}: start/end up nearly same (need={math.degrees(need):.1f}°) "
                f"but path rolled {math.degrees(path):.1f}°"
            )
    # Smoke: at least one edge had a non-trivial direction change
    assert any(need > 0.01 for _lab, _p, need, _e in report)


def test_edge_flight_from_front_no_spin():
    """Front → top/front edge: final up may differ; path must stay short."""
    start_pos = np.array([0.0, 0.0, 50.0])
    start_foc = np.zeros(3)
    start_up = np.array([0.0, 1.0, 0.0])
    # edge +y+z (top-front)
    end_pos, end_foc, end_up = _direction_pose((0.0, 1.0, 1.0))
    path, need = Viewport.measure_camera_up_path(
        start_pos, start_foc, start_up, end_pos, end_foc, end_up, samples=48
    )
    assert path <= need + math.radians(15.0)
    assert path < math.radians(120.0)  # never the old ~270° roll
