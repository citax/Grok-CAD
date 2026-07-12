"""Junction dots = unique entity endpoints; all on-plane."""

from __future__ import annotations

import numpy as np
import pytest

from cadcore.sketch import PlaneFrame, Sketch


@pytest.mark.parametrize("plane", ["PLANE_FRONT", "PLANE_TOP", "PLANE_RIGHT"])
def test_unique_endpoints_and_on_plane(plane: str):
    fr = PlaneFrame.from_plane_type(plane)
    sk = Sketch(frame=fr)
    # Two lines sharing an endpoint (junction)
    sk.add_line((0.0, 0.0), (2.0, 0.0))
    sk.add_line((2.0, 0.0), (2.0, 1.5))
    sk.add_circle((0.5, 0.5), 0.25)
    pts = sk.unique_endpoints()
    # line ends: (0,0),(2,0),(2,1.5) + circle center (0.5,0.5) = 4 unique
    assert len(pts) == 4
    # (2,0) not duplicated
    assert sum(1 for p in pts if abs(p[0] - 2.0) < 1e-12 and abs(p[1]) < 1e-12) == 1
    for uv in pts:
        w = fr.to_world(uv)
        dev = abs(float(np.dot(fr.normal, np.asarray(w) - fr.origin)))
        assert dev < 1e-9, f"off-plane {plane} dev={dev}"


def test_empty_sketch_no_junctions():
    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    assert sk.unique_endpoints() == []
