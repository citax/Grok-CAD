"""Display units: mm/cm/inch conversion and line-length setting."""

from __future__ import annotations

import pytest

from cadcore.sketch import LineEntity, line_length, set_line_length
from cadcore.units import Unit, format_length, from_mm, parse_length, to_mm


@pytest.mark.parametrize(
    "unit,factor",
    [(Unit.MM, 1.0), (Unit.CM, 10.0), (Unit.INCH, 25.4)],
)
def test_to_from_mm_roundtrip(unit, factor):
    assert abs(to_mm(1.0, unit) - factor) < 1e-12
    assert abs(from_mm(factor, unit) - 1.0) < 1e-12


def test_format_length_matches_geometry():
    line = LineEntity(id=1, kind=None, p0=(0.0, 0.0), p1=(12.0, 0.0))  # type: ignore[arg-type]
    L = line_length(line)
    assert format_length(L, Unit.MM) == "12.00 mm"
    assert format_length(L, Unit.CM) == "1.20 cm"
    # 12 mm = 12/25.4 in
    assert format_length(L, Unit.INCH).endswith(" in")
    assert abs(from_mm(L, Unit.INCH) - 12.0 / 25.4) < 1e-9


@pytest.mark.parametrize(
    "unit,display,expect_mm",
    [
        (Unit.MM, 25.0, 25.0),
        (Unit.CM, 2.5, 25.0),
        (Unit.INCH, 1.0, 25.4),
    ],
)
def test_set_line_length_in_each_unit(unit, display, expect_mm):
    line = LineEntity(id=1, kind=None, p0=(0.0, 0.0), p1=(1.0, 0.0))  # type: ignore[arg-type]
    set_line_length(line, to_mm(display, unit), free_end="p1")
    assert abs(line_length(line) - expect_mm) < 1e-9
    assert abs(line.p1[0] - expect_mm) < 1e-9
    assert abs(line.p1[1]) < 1e-9
    # label matches
    assert format_length(line_length(line), unit) == format_length(expect_mm, unit)


def test_parse_length_with_suffix():
    assert abs(parse_length("10", Unit.MM) - 10.0) < 1e-12
    assert abs(parse_length("1 cm", Unit.MM) - 10.0) < 1e-12
    assert abs(parse_length("1in", Unit.MM) - 25.4) < 1e-12


def test_set_length_preserves_direction():
    line = LineEntity(id=1, kind=None, p0=(0.0, 0.0), p1=(3.0, 4.0))  # type: ignore[arg-type]
    set_line_length(line, 10.0, free_end="p1")
    assert abs(line_length(line) - 10.0) < 1e-9
    # direction 3-4-5
    assert abs(line.p1[0] - 6.0) < 1e-9
    assert abs(line.p1[1] - 8.0) < 1e-9
