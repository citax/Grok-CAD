"""Qt logical ↔ VTK device display coordinates.

Multi-monitor / HiDPI: Qt mouse events are in *logical* pixels; VTK
``RenderWindow.GetSize()`` and pickers use *device* pixels. Mixing them makes
hit tests miss on any screen whose devicePixelRatio ≠ 1 (common when a window
is dragged between monitors with different scaling).

All cube/triad hit-tests MUST go through these helpers.
"""

from __future__ import annotations

from typing import Optional, Tuple


def widget_size_logical(interactor) -> Tuple[float, float]:
    """Qt widget size in logical pixels (same space as QMouseEvent.position())."""
    return float(interactor.width()), float(interactor.height())


def render_size_device(plotter) -> Tuple[float, float]:
    """VTK framebuffer size in device pixels."""
    try:
        rw = plotter.render_window
        w, h = rw.GetSize()
        return float(w), float(h)
    except Exception:
        pass
    try:
        w, h = plotter.window_size
        return float(w), float(h)
    except Exception:
        return 1.0, 1.0


def device_pixel_ratio(interactor, plotter=None) -> float:
    """Effective scale from Qt logical → VTK device along X."""
    try:
        dpr = float(interactor.devicePixelRatioF())
        if dpr > 0.01:
            return dpr
    except Exception:
        pass
    if plotter is not None:
        lw, lh = widget_size_logical(interactor)
        dw, dh = render_size_device(plotter)
        if lw > 1 and dw > 1:
            return dw / lw
    return 1.0


def qt_to_normalized(qx: float, qy: float, interactor) -> Tuple[float, float]:
    """Qt (logical, y-down) → normalized 0..1 with y-up (VTK viewport space)."""
    w, h = widget_size_logical(interactor)
    if w < 1 or h < 1:
        return 0.0, 0.0
    nx = float(qx) / w
    ny = 1.0 - float(qy) / h
    return nx, ny


def qt_to_vtk_display(
    qx: float, qy: float, interactor, plotter
) -> Tuple[float, float, float, float]:
    """Qt (logical, y-down) → VTK display (device, y-up) + device size.

    Returns (vx, vy, device_w, device_h).
    """
    lw, lh = widget_size_logical(interactor)
    dw, dh = render_size_device(plotter)
    sx = dw / max(lw, 1.0)
    sy = dh / max(lh, 1.0)
    vx = float(qx) * sx
    vy = dh - float(qy) * sy
    return vx, vy, dw, dh


def in_normalized_viewport(
    qx: float, qy: float, interactor, viewport: Tuple[float, float, float, float]
) -> bool:
    """True if Qt click lies inside a VTK normalized viewport rect."""
    nx, ny = qt_to_normalized(qx, qy, interactor)
    vx0, vy0, vx1, vy1 = viewport
    return vx0 <= nx <= vx1 and vy0 <= ny <= vy1
