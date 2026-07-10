"""Application entry: python -m app.main"""

from __future__ import annotations

import os
import sys
import traceback


def main() -> int:
    # Log software-GL hint early
    print(
        f"[main] LIBGL_ALWAYS_SOFTWARE={os.environ.get('LIBGL_ALWAYS_SOFTWARE', '')}",
        file=sys.stderr,
    )
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
        from app.mainwindow import MainWindow
    except Exception as exc:  # noqa: BLE001
        print(f"[main] Import failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    app = QApplication(sys.argv)
    app.setApplicationName("Grok CAD")
    app.setOrganizationName("CadCore")
    try:
        win = MainWindow()
        win.show()
    except Exception as exc:  # noqa: BLE001
        print(f"[main] Window init failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        QMessageBox.critical(
            None,
            "Grok CAD",
            f"Failed to start:\n{exc}\n\n"
            "Under WSL, ensure LIBGL_ALWAYS_SOFTWARE=1 (run_cad.sh does this).",
        )
        return 1
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
