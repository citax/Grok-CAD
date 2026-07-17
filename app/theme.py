"""UI/scene palette — light default, dark available (startup-only binding).

Approach (b): theme is resolved once at import (GROK_THEME env or QSettings
ui/theme, default \"light\"). Module-level constants are set then; consumers
that ``from app.theme import SKETCH_COLOR`` bind that snapshot. Switching theme
in Settings is saved and applied on next restart (no half-updated runtime
rebind). Dual-theme benches set GROK_THEME before importing app modules.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

from PySide6.QtGui import QColor, QIcon, QPalette
from PySide6.QtWidgets import QApplication

_THEME_QSS = Path(__file__).with_name("theme.qss")

# ---------------------------------------------------------------------------
# Palettes (designed, not inverted). Pairwise RGB distance kept well above the
# framebuffer benches' tol=48 so SEL/PREVIEW/SKETCH stay distinguishable.
# ---------------------------------------------------------------------------
_PALETTES: Dict[str, Dict[str, str]] = {
    "light": {
        # SolidWorks-like chrome: cool light grey panels, SW blue accent
        "BG_APP": "#D6D9DE",
        "BG_PANEL": "#E8EAED",
        "BG_ELEVATED": "#F7F8FA",
        "ACCENT": "#0070C0",  # classic SW command blue
        "TEXT_PRIMARY": "#1B1F23",
        "TEXT_SECONDARY": "#5C6570",
        "BORDER": "#B8BEC6",
        # Viewport: soft neutral grey (SW graphics area — not blue-tinted)
        "VP_BG_TOP": "#E4E7EC",
        "VP_BG_BOTTOM": "#C8CDD6",
        "GRID_COLOR": "#8A929C",
        # RGB triad (SW reference triad / origin arrows)
        "AXIS_X": "#E53935",  # red  — Right
        "AXIS_Y": "#43A047",  # green — Top
        "AXIS_Z": "#1E88E5",  # blue  — Front
        "AXIS_LABEL": "#1B1F23",
        "SOLID_COLOR": "#8B9199",
        "SOLID_SELECTED": "#E65100",
        "PLANE_FRONT": "#1E88E5",
        "PLANE_TOP": "#43A047",
        "PLANE_RIGHT": "#E53935",
        # Dark geometry on light viewport (high contrast)
        "SKETCH_COLOR": "#1B1F23",
        "SKETCH_PREVIEW": "#F9A825",
        "SKETCH_SELECTED": "#6A1B9A",
        "PROFILE_FILL": "#7B1FA2",
        "SEL_BOX_WINDOW": "#0D47A1",
        "SEL_BOX_CROSSING": "#C62828",
        "HANDLE_COLOR": "#EF6C00",
        "HANDLE_HOVER": "#E65100",
        # Sketch: faint grid; H=red / V=green like SW sketcher
        "SKETCH_GRID": "#B0B7C0",
        "SKETCH_H": "#E53935",
        "SKETCH_V": "#43A047",
    },
    "dark": {
        "BG_APP": "#1e2228",
        "BG_PANEL": "#252a31",
        "BG_ELEVATED": "#2d333c",
        "ACCENT": "#3B82F6",
        "TEXT_PRIMARY": "#e6e9ee",
        "TEXT_SECONDARY": "#9aa3ad",
        "BORDER": "#333a44",
        "VP_BG_TOP": "#2b313a",
        "VP_BG_BOTTOM": "#1a1d22",
        "GRID_COLOR": "#6b7280",
        "AXIS_X": "#F87171",
        "AXIS_Y": "#4ADE80",
        "AXIS_Z": "#60A5FA",
        "AXIS_LABEL": "#F1F5F9",
        "SOLID_COLOR": "#9aa3ad",
        "SOLID_SELECTED": "#FBBF24",
        "PLANE_FRONT": "#60A5FA",
        "PLANE_TOP": "#34D399",
        "PLANE_RIGHT": "#F87171",
        "SKETCH_COLOR": "#E2E8F0",
        "SKETCH_PREVIEW": "#FBBF24",
        "SKETCH_SELECTED": "#34D399",  # green (was cyan — farther from blue window)
        # Filled closed-region highlight — bright amber on dark bg
        "PROFILE_FILL": "#FBBF24",
        "SEL_BOX_WINDOW": "#2563EB",
        "SEL_BOX_CROSSING": "#E879F9",  # pink-magenta
        "HANDLE_COLOR": "#FDE047",
        "HANDLE_HOVER": "#FB923C",
        "SKETCH_GRID": "#64748B",
        "SKETCH_H": "#F87171",
        "SKETCH_V": "#4ADE80",
    },
}

# Module-level colour names (set by apply_palette)
BG_APP = BG_PANEL = BG_ELEVATED = ACCENT = TEXT_PRIMARY = TEXT_SECONDARY = BORDER = ""
VP_BG_TOP = VP_BG_BOTTOM = GRID_COLOR = ""
AXIS_X = AXIS_Y = AXIS_Z = AXIS_LABEL = ""
SOLID_COLOR = SOLID_SELECTED = ""
PLANE_FRONT = PLANE_TOP = PLANE_RIGHT = ""
SKETCH_COLOR = SKETCH_PREVIEW = SKETCH_SELECTED = ""
PROFILE_FILL = ""
SEL_BOX_WINDOW = SEL_BOX_CROSSING = ""
HANDLE_COLOR = HANDLE_HOVER = SKETCH_GRID = SKETCH_H = SKETCH_V = ""

CURRENT_THEME = "light"


def resolve_theme_name() -> str:
    env = (os.environ.get("GROK_THEME") or "").strip().lower()
    if env in _PALETTES:
        return env
    try:
        from PySide6.QtCore import QSettings

        s = QSettings("CadCore", "Grok CAD")
        t = str(s.value("ui/theme", "light")).strip().lower()
        if t in _PALETTES:
            return t
    except Exception:
        pass
    return "light"


def apply_palette(name: str) -> str:
    """Set module-level colour constants from a named palette. Returns name used."""
    global CURRENT_THEME
    global BG_APP, BG_PANEL, BG_ELEVATED, ACCENT, TEXT_PRIMARY, TEXT_SECONDARY, BORDER
    global VP_BG_TOP, VP_BG_BOTTOM, GRID_COLOR
    global AXIS_X, AXIS_Y, AXIS_Z, AXIS_LABEL
    global SOLID_COLOR, SOLID_SELECTED
    global PLANE_FRONT, PLANE_TOP, PLANE_RIGHT
    global SKETCH_COLOR, SKETCH_PREVIEW, SKETCH_SELECTED, PROFILE_FILL
    global SEL_BOX_WINDOW, SEL_BOX_CROSSING
    global HANDLE_COLOR, HANDLE_HOVER, SKETCH_GRID, SKETCH_H, SKETCH_V

    key = name if name in _PALETTES else "light"
    p = _PALETTES[key]
    CURRENT_THEME = key
    BG_APP = p["BG_APP"]
    BG_PANEL = p["BG_PANEL"]
    BG_ELEVATED = p["BG_ELEVATED"]
    ACCENT = p["ACCENT"]
    TEXT_PRIMARY = p["TEXT_PRIMARY"]
    TEXT_SECONDARY = p["TEXT_SECONDARY"]
    BORDER = p["BORDER"]
    VP_BG_TOP = p["VP_BG_TOP"]
    VP_BG_BOTTOM = p["VP_BG_BOTTOM"]
    GRID_COLOR = p["GRID_COLOR"]
    AXIS_X = p["AXIS_X"]
    AXIS_Y = p["AXIS_Y"]
    AXIS_Z = p["AXIS_Z"]
    AXIS_LABEL = p["AXIS_LABEL"]
    SOLID_COLOR = p["SOLID_COLOR"]
    SOLID_SELECTED = p["SOLID_SELECTED"]
    PLANE_FRONT = p["PLANE_FRONT"]
    PLANE_TOP = p["PLANE_TOP"]
    PLANE_RIGHT = p["PLANE_RIGHT"]
    SKETCH_COLOR = p["SKETCH_COLOR"]
    SKETCH_PREVIEW = p["SKETCH_PREVIEW"]
    SKETCH_SELECTED = p["SKETCH_SELECTED"]
    PROFILE_FILL = p["PROFILE_FILL"]
    SEL_BOX_WINDOW = p["SEL_BOX_WINDOW"]
    SEL_BOX_CROSSING = p["SEL_BOX_CROSSING"]
    HANDLE_COLOR = p["HANDLE_COLOR"]
    HANDLE_HOVER = p["HANDLE_HOVER"]
    SKETCH_GRID = p["SKETCH_GRID"]
    SKETCH_H = p["SKETCH_H"]
    SKETCH_V = p["SKETCH_V"]
    return key


def save_theme_preference(name: str) -> None:
    if name not in _PALETTES:
        return
    try:
        from PySide6.QtCore import QSettings

        QSettings("CadCore", "Grok CAD").setValue("ui/theme", name)
    except Exception:
        pass


def _substitutions() -> dict[str, str]:
    return {
        "BG_APP": BG_APP,
        "BG_PANEL": BG_PANEL,
        "BG_ELEVATED": BG_ELEVATED,
        "ACCENT": ACCENT,
        "TEXT_PRIMARY": TEXT_PRIMARY,
        "TEXT_SECONDARY": TEXT_SECONDARY,
        "BORDER": BORDER,
    }


def load_stylesheet() -> str:
    raw = _THEME_QSS.read_text(encoding="utf-8")
    for key, val in _substitutions().items():
        raw = raw.replace(f"@{key}", val)
    return raw


def apply_theme(app: QApplication) -> None:
    """Apply Fusion palette + QSS for the active (startup) theme."""
    app.setStyle("Fusion")
    pal = QPalette()
    bg = QColor(BG_APP)
    panel = QColor(BG_PANEL)
    text = QColor(TEXT_PRIMARY)
    muted = QColor(TEXT_SECONDARY)
    accent = QColor(ACCENT)
    pal.setColor(QPalette.ColorRole.Window, bg)
    pal.setColor(QPalette.ColorRole.WindowText, text)
    pal.setColor(QPalette.ColorRole.Base, panel)
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(BG_ELEVATED))
    pal.setColor(QPalette.ColorRole.Text, text)
    pal.setColor(QPalette.ColorRole.Button, panel)
    pal.setColor(QPalette.ColorRole.ButtonText, text)
    pal.setColor(QPalette.ColorRole.Highlight, accent)
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(BG_ELEVATED))
    pal.setColor(QPalette.ColorRole.ToolTipText, text)
    pal.setColor(QPalette.ColorRole.PlaceholderText, muted)
    pal.setColor(QPalette.ColorRole.Link, accent)
    app.setPalette(pal)
    app.setStyleSheet(load_stylesheet())


def fa_icon(name: str, color: Optional[str] = None, *, scale: float = 1.0) -> QIcon:
    import qtawesome as qta

    return qta.icon(name, color=color or TEXT_PRIMARY, scale_factor=scale)


def status_env_suffix() -> str:
    return os.environ.get("QT_QPA_PLATFORM", "default")


def plane_color_for(feature_type_name: str) -> str:
    mapping = {
        "PLANE_FRONT": PLANE_FRONT,
        "PLANE_TOP": PLANE_TOP,
        "PLANE_RIGHT": PLANE_RIGHT,
    }
    return mapping.get(feature_type_name, ACCENT)


def rgb_distance(a: str, b: str) -> float:
    """L2 distance in 0–255 RGB space."""
    def to_rgb(h: str):
        h = h.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    ra, ga, ba = to_rgb(a)
    rb, gb, bb = to_rgb(b)
    return float(((ra - rb) ** 2 + (ga - gb) ** 2 + (ba - bb) ** 2) ** 0.5)


def relative_luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    rgb = [int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]

    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (lin(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(fg: str, bg: str) -> float:
    l1 = relative_luminance(fg)
    l2 = relative_luminance(bg)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


# Resolve palette at import (startup-only)
apply_palette(resolve_theme_name())
