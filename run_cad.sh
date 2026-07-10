#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Prefer a Linux-native venv (VTK .so loads are unreliable from /mnt/c)
VENV_DIR="${GROK_CAD_VENV:-$HOME/.venvs/grok-cad}"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  # Fallback: project-local .venv (symlink or real)
  if [[ -x .venv/bin/python ]]; then
    VENV_DIR="$(cd .venv && pwd -P)"
  else
    echo "ERROR: Python venv not found at $VENV_DIR or ./.venv" >&2
    echo "Create it with:" >&2
    echo "  uv venv \"\$HOME/.venvs/grok-cad\" --python 3.12" >&2
    echo "  uv pip install --python \"\$HOME/.venvs/grok-cad/bin/python\" -r requirements.txt" >&2
    echo "  ln -sfn \"\$HOME/.venvs/grok-cad\" .venv" >&2
    exit 1
  fi
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

export LIBGL_ALWAYS_SOFTWARE=1   # WSLg hardware GL fails on this host; force software GL
# WSLg: Qt defaults to Wayland while VTK uses X11/XWayland → BadWindow on embed.
# Force Qt onto xcb so both share the same X server.
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-none}"
# Help VTK find its bundled libs (path depends on Python version)
PY_VER="$("$VENV_DIR/bin/python" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")"
VTK_MOD="$VENV_DIR/lib/python${PY_VER}/site-packages/vtkmodules"
if [[ -d "$VTK_MOD" ]]; then
  export LD_LIBRARY_PATH="${VTK_MOD}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
# Qt xcb needs xcb-cursor + xkbcommon-x11 etc. (often missing on minimal WSL)
# Bundled user libs (downloaded .deb extracts — no sudo required)
QT_XCB_LIBS="${HOME}/.local/lib/qt-xcb"
if [[ -d "$QT_XCB_LIBS" ]]; then
  export LD_LIBRARY_PATH="${QT_XCB_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# Quick preflight so failures are clear
python - <<'PY' || exit 1
import sys
print(f"[preflight] python {sys.version.split()[0]}", file=sys.stderr)
try:
    import vtkmodules.vtkImagingCore  # noqa: F401
    import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
    import pyvista  # noqa: F401
    from pyvistaqt import QtInteractor  # noqa: F401
    from PySide6.QtWidgets import QApplication  # noqa: F401
except Exception as e:
    print(f"[preflight] Import failed: {e}", file=sys.stderr)
    print(
        "Hint: reinstall deps into a Linux-home venv (not under /mnt/c):\n"
        "  rm -rf ~/.venvs/grok-cad .venv\n"
        "  uv venv ~/.venvs/grok-cad --python 3.12\n"
        "  uv pip install --python ~/.venvs/grok-cad/bin/python -r requirements.txt\n"
        "  ln -sfn ~/.venvs/grok-cad .venv",
        file=sys.stderr,
    )
    raise SystemExit(1)
print("[preflight] OK", file=sys.stderr)
PY

exec python -m app.main "$@"
