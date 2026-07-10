"""Central UI/scene palette and helpers for the dark CAD chrome."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from PySide6.QtGui import QColor, QIcon, QPalette
from PySide6.QtWidgets import QApplication

# ---------------------------------------------------------------------------
# Exact app chrome palette (single source of truth)
# ---------------------------------------------------------------------------
BG_APP = "#1e2228"
BG_PANEL = "#252a31"
BG_ELEVATED = "#2d333c"
ACCENT = "#3B82F6"
TEXT_PRIMARY = "#e6e9ee"
TEXT_SECONDARY = "#9aa3ad"
BORDER = "#333a44"

# Viewport / scene (not UI chrome)
VP_BG_TOP = "#2b313a"
VP_BG_BOTTOM = "#1a1d22"
GRID_COLOR = "#6b7280"
AXIS_X = "#F87171"
AXIS_Y = "#4ADE80"
AXIS_Z = "#60A5FA"
SOLID_COLOR = "#9aa3ad"
SOLID_SELECTED = "#FBBF24"

# Reference planes — Front deliberately shifted off UI accent #3B82F6
PLANE_FRONT = "#60A5FA"  # lighter sky; distinct from accent
PLANE_TOP = "#34D399"
PLANE_RIGHT = "#F87171"

SKETCH_COLOR = "#E2E8F0"
SKETCH_PREVIEW = "#FBBF24"
HANDLE_COLOR = "#FDE047"
HANDLE_HOVER = "#FB923C"
SKETCH_GRID = "#64748B"
SKETCH_H = "#F87171"
SKETCH_V = "#4ADE80"

_THEME_QSS = Path(__file__).with_name("theme.qss")


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
    """Load theme.qss and substitute palette tokens (@TOKEN)."""
    raw = _THEME_QSS.read_text(encoding="utf-8")
    for key, val in _substitutions().items():
        raw = raw.replace(f"@{key}", val)
    return raw


def apply_theme(app: QApplication) -> None:
    """Apply fusion-ish dark base palette + bundled QSS."""
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
    """Font-Awesome icon tinted to the theme (qtawesome)."""
    import qtawesome as qta

    return qta.icon(name, color=color or TEXT_PRIMARY, scale_factor=scale)


def status_env_suffix() -> str:
    """e.g. 'xcb · llvmpipe' for the status bar."""
    platform = os.environ.get("QT_QPA_PLATFORM", "default")
    # Renderer string is filled later by viewport if available
    return platform


def plane_color_for(feature_type_name: str) -> str:
    mapping = {
        "PLANE_FRONT": PLANE_FRONT,
        "PLANE_TOP": PLANE_TOP,
        "PLANE_RIGHT": PLANE_RIGHT,
    }
    return mapping.get(feature_type_name, ACCENT)
