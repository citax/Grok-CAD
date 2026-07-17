"""Environment display policy: origin triad + reference planes vs solids.

These pure tests encode the SolidWorks/Fusion behaviour we claim to follow:
  * world-origin arrows must never dominate a tiny part (no large floor)
  * world-origin arrows are suppressed when solids are displayed
  * reference planes are hidden when solids are on screen (unless plane selected
    or sketch mode) — matching SW View→Hide/Show→Planes / Fusion object visibility
"""

from __future__ import annotations

import pytest

from cadcore.scale import origin_triad_policy, reference_planes_should_show


def test_origin_hidden_when_solids_present():
    show, L = origin_triad_policy(has_display_solids=True, char_mm=50.0, part_extent_mm=40.0)
    assert show is False
    assert L == 0.0


def test_origin_shown_empty_scene_modest():
    show, L = origin_triad_policy(has_display_solids=False, char_mm=50.0)
    assert show is True
    assert 2.0 <= L <= 12.0


def test_origin_never_has_six_mm_floor_on_tiny_char():
    """Regression: max(6.0, …) used to make 6 mm arrows on a 3 mm part."""
    show, L = origin_triad_policy(
        has_display_solids=False, char_mm=3.0, part_extent_mm=3.0
    )
    assert show is True
    # Must be smaller than the part span — never longer than 25% of 3 mm = 0.75
    # (also capped by empty-scene policy; either way must not be 6 mm)
    assert L < 3.0
    assert L <= 3.0 * 0.25 + 1e-9
    assert L != pytest.approx(6.0)


def test_origin_suppressed_for_large_solid_not_swallowed_inside():
    """Large solid → no world triad (orientation = corner triad only)."""
    show, L = origin_triad_policy(
        has_display_solids=True, char_mm=250.0, part_extent_mm=220.0
    )
    assert show is False
    assert L == 0.0


def test_planes_shown_when_empty():
    assert reference_planes_should_show(
        has_display_solids=False, in_sketch_mode=False, selected_is_plane=False
    )


def test_planes_hidden_with_solids_and_solid_selected():
    assert not reference_planes_should_show(
        has_display_solids=True, in_sketch_mode=False, selected_is_plane=False
    )


def test_planes_shown_with_solids_when_plane_selected():
    """User picked a plane to sketch on — need to see it."""
    assert reference_planes_should_show(
        has_display_solids=True, in_sketch_mode=False, selected_is_plane=True
    )


def test_planes_shown_in_sketch_mode_even_with_solids():
    assert reference_planes_should_show(
        has_display_solids=True, in_sketch_mode=True, selected_is_plane=False
    )
