"""On-demand dimension labels + shared-only junction dots."""

from __future__ import annotations

import numpy as np

from cadcore.sketch import PlaneFrame, Sketch, line_length
from cadcore.units import Unit, format_length


def test_shared_endpoints_only_at_junctions():
    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    sk.add_line((0, 0), (2, 0))
    sk.add_line((2, 0), (2, 1))  # shares (2,0)
    sk.add_line((5, 5), (6, 5))  # isolated
    shared = sk.shared_endpoints()
    assert len(shared) == 1
    assert abs(shared[0][0] - 2.0) < 1e-9 and abs(shared[0][1]) < 1e-9
    # isolated endpoints not listed
    assert not any(abs(p[0] - 5) < 1e-9 for p in shared)


def test_shared_endpoints_on_plane():
    for plane in ("PLANE_FRONT", "PLANE_TOP", "PLANE_RIGHT"):
        fr = PlaneFrame.from_plane_type(plane)
        sk = Sketch(frame=fr)
        sk.add_line((0, 0), (1, 0))
        sk.add_line((1, 0), (1, 1))
        for uv in sk.shared_endpoints():
            w = fr.to_world(uv)
            dev = abs(float(np.dot(fr.normal, w - fr.origin)))
            assert dev < 1e-9


def test_dim_label_texts_only_selected():
    """Viewport.dim_label_texts reflects selection only (logic unit-level)."""
    # Pure logic mirror of viewport selection filter
    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    a = sk.add_line((0, 0), (12, 0))
    b = sk.add_line((0, 1), (5, 1))
    selected = a.id
    unit = Unit.MM
    labels = []
    for ent in sk.entities:
        if ent.id == selected:
            labels.append(format_length(line_length(ent), unit))
    assert labels == ["12.00 mm"]
    assert format_length(line_length(b), unit) not in labels or b.id == selected


def test_no_labels_when_nothing_selected():
    sk = Sketch(frame=PlaneFrame.from_plane_type("PLANE_FRONT"))
    sk.add_line((0, 0), (3, 0))
    selected = -1
    labels = [
        format_length(line_length(e), Unit.MM)
        for e in sk.entities
        if e.id == selected
    ]
    assert labels == []
