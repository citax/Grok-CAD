"""Unattended runs must never block on unsaved-changes dialogs."""

from __future__ import annotations

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication


def _app():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def test_is_unattended_env_flag(monkeypatch):
    from app.mainwindow import MainWindow

    monkeypatch.setenv("GROK_CAD_UNATTENDED", "1")
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    assert MainWindow._is_unattended() is True
    monkeypatch.setenv("GROK_CAD_UNATTENDED", "0")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    assert MainWindow._is_unattended() is True
    monkeypatch.setenv("GROK_CAD_UNATTENDED", "")
    monkeypatch.setenv("QT_QPA_PLATFORM", "xcb")
    assert MainWindow._is_unattended() is False


def test_confirm_discard_unattended_no_dialog(monkeypatch):
    """Dirty document + unattended → True without raising / hanging."""
    _app()
    from app.mainwindow import MainWindow
    from cadcore.document import Document

    monkeypatch.setenv("GROK_CAD_UNATTENDED", "1")

    class StubWin:
        def __init__(self):
            self.doc = Document()
            self.doc.seed_reference_planes()
            # make dirty
            plane = self.doc.features[0]
            self.doc.create_sketch_on_plane(plane.id)
            assert self.doc.dirty

        _is_unattended = staticmethod(MainWindow._is_unattended)
        _confirm_discard_if_dirty = MainWindow._confirm_discard_if_dirty

    win = StubWin()
    # Must return immediately True (discard) — never open QMessageBox
    assert win._confirm_discard_if_dirty() is True


def test_confirm_discard_clean_doc():
    _app()
    from app.mainwindow import MainWindow
    from cadcore.document import Document

    class StubWin:
        def __init__(self):
            self.doc = Document()
            self.doc.seed_reference_planes()
            self.doc.mark_clean()

        _is_unattended = staticmethod(MainWindow._is_unattended)
        _confirm_discard_if_dirty = MainWindow._confirm_discard_if_dirty

    win = StubWin()
    assert win.doc.dirty is False
    assert win._confirm_discard_if_dirty() is True
