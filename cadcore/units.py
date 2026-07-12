"""Display units for sketch dimensions (internal world unit = mm)."""

from __future__ import annotations

from enum import Enum
from typing import Union


class Unit(str, Enum):
    MM = "mm"
    CM = "cm"
    INCH = "inch"

    @property
    def label(self) -> str:
        return {Unit.MM: "mm", Unit.CM: "cm", Unit.INCH: "in"}[self]


# mm per display unit
_MM_PER: dict[Unit, float] = {
    Unit.MM: 1.0,
    Unit.CM: 10.0,
    Unit.INCH: 25.4,
}


def mm_per_unit(unit: Unit) -> float:
    return float(_MM_PER[Unit(unit)])


def to_mm(value: float, unit: Union[Unit, str]) -> float:
    """Convert a display-unit value to internal mm."""
    u = Unit(unit) if not isinstance(unit, Unit) else unit
    return float(value) * mm_per_unit(u)


def from_mm(value_mm: float, unit: Union[Unit, str]) -> float:
    """Convert internal mm to display units."""
    u = Unit(unit) if not isinstance(unit, Unit) else unit
    return float(value_mm) / mm_per_unit(u)


def format_length(value_mm: float, unit: Union[Unit, str], *, decimals: int = 2) -> str:
    """Human-readable length in the current unit, e.g. '12.00 mm'."""
    u = Unit(unit) if not isinstance(unit, Unit) else unit
    v = from_mm(value_mm, u)
    suffix = {Unit.MM: "mm", Unit.CM: "cm", Unit.INCH: "in"}[u]
    return f"{v:.{decimals}f} {suffix}"


def parse_length(text: str, unit: Union[Unit, str]) -> float:
    """Parse a length string possibly containing a unit suffix → internal mm.

    Bare numbers use ``unit``. Suffixes mm/cm/in/inch override.
    """
    s = (text or "").strip().lower().replace(",", ".")
    if not s:
        raise ValueError("empty length")
    # strip known suffixes
    override: Unit | None = None
    for suf, u in (
        ("inch", Unit.INCH),
        ("in", Unit.INCH),
        ("cm", Unit.CM),
        ("mm", Unit.MM),
    ):
        if s.endswith(suf):
            s = s[: -len(suf)].strip()
            override = u
            break
    val = float(s)
    return to_mm(val, override if override is not None else unit)
