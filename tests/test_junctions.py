"""Junction dots = shared endpoints only; all on-plane."""

from __future__ import annotations

import numpy as np
import pytest

from cadcore.sketch import PlaneFrame, Sketch


@pytest.mark.parametrize("plane", ["PLANE_FRONT", "PLANE_TOP", "PLANE_RIGHT"])
def test_shared_endpoints_and_on_plane(plane: str):
    fr = PlaneFrame.from_plane_type(plane)
    sk = Sketch(frame=fr)
    sk.add_line((0.0, 0.0), (2.0, 0.0))
    sk.add_line((2.0, 0.0), (2.0, 1.5))
    pts = sk.shared_endpoints()
    assert len(pts) == 1
    assert abs(pts[0][0] - 2.0) < 1e-9
    for uv in pts:
        w = fr.to_world(uv)
        dev = abs(float(np.dot(fr.normal, np.asarray(w) - fr.origin)))
        assert dev < 1e-9, f"off-plane {plane} dev={dev}"


def test_empty_and_isolated_no_junctions():
    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    assert sk.shared_endpoints() == []
    sk.add_line((0, 0), (1, 0))
    assert sk.shared_endpoints() == []
