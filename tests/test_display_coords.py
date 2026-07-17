"""DPI / multi-monitor coordinate conversion for view cube and triad hits."""

from __future__ import annotations

from app.display_coords import (
    in_normalized_viewport,
    qt_to_normalized,
    qt_to_vtk_display,
)


class _FakeInteractor:
    def __init__(self, w: int, h: int, dpr: float = 1.0) -> None:
        self._w, self._h, self._dpr = w, h, dpr

    def width(self) -> int:
        return self._w

    def height(self) -> int:
        return self._h

    def devicePixelRatioF(self) -> float:
        return self._dpr


class _FakeRW:
    def __init__(self, w: int, h: int) -> None:
        self._w, self._h = w, h

    def GetSize(self):
        return (self._w, self._h)


class _FakePlotter:
    def __init__(self, dw: int, dh: int) -> None:
        self.render_window = _FakeRW(dw, dh)
        self.window_size = (dw, dh)


def test_normalized_uses_qt_logical_not_device():
    """Hit-test must use interactor width/height (logical), not device framebuffer."""
    # 2× DPI: logical 800×600, device 1600×1200
    iw = _FakeInteractor(800, 600, dpr=2.0)
    # Click at right edge of cube pad in logical space
    # VIEWPORT x 0.80..0.995 → logical x ≈ 0.90 * 800 = 720
    assert in_normalized_viewport(720, 50, iw, (0.80, 0.80, 0.995, 0.995))
    # Same click if we wrongly divided by device width would miss
    nx_wrong = 720 / 1600  # 0.45 — NOT in cube pad
    assert nx_wrong < 0.80


def test_qt_to_vtk_scales_by_dpr():
    iw = _FakeInteractor(800, 600, dpr=2.0)
    pl = _FakePlotter(1600, 1200)
    vx, vy, dw, dh = qt_to_vtk_display(100, 50, iw, pl)
    assert dw == 1600 and dh == 1200
    assert abs(vx - 200.0) < 1e-6  # 100 * 2
    # y: 1200 - 50*2 = 1100
    assert abs(vy - 1100.0) < 1e-6


def test_qt_to_normalized_y_up():
    iw = _FakeInteractor(100, 100)
    nx, ny = qt_to_normalized(25, 25, iw)
    assert abs(nx - 0.25) < 1e-9
    assert abs(ny - 0.75) < 1e-9  # top of widget → high ny


def test_dpr1_identity():
    iw = _FakeInteractor(640, 480, dpr=1.0)
    pl = _FakePlotter(640, 480)
    vx, vy, _, _ = qt_to_vtk_display(10, 20, iw, pl)
    assert abs(vx - 10) < 1e-6
    assert abs(vy - (480 - 20)) < 1e-6
