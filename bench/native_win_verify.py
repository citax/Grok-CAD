"""Native Windows diagnosis: HWND ownership, window styles, GL, frametime.

Run with the Windows venv only. Do not set WSL Qt/GL env vars.
"""
from __future__ import annotations

import os
import sys
import time
import traceback

# Refuse accidental WSL software-GL path
for k in ("LIBGL_ALWAYS_SOFTWARE", "QT_QPA_PLATFORM", "QT_XCB_GL_INTEGRATION"):
    if os.environ.get(k):
        print(f"WARN clearing leftover {k}={os.environ.get(k)!r}", flush=True)
        os.environ.pop(k, None)


def _pump(app, n=20):
    for _ in range(n):
        app.processEvents()
        time.sleep(0.01)


def _hwnd_info(hwnd: int) -> dict:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    GetWindowLongPtrW = user32.GetWindowLongPtrW
    GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
    GetWindowLongPtrW.restype = ctypes.c_ssize_t

    GetWindowThreadProcessId = user32.GetWindowThreadProcessId
    GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    GetWindowThreadProcessId.restype = wintypes.DWORD

    GWL_STYLE = -16
    WS_THICKFRAME = 0x00040000
    WS_MAXIMIZEBOX = 0x00010000
    WS_CAPTION = 0x00C00000
    WS_SYSMENU = 0x00080000
    WS_MINIMIZEBOX = 0x00020000

    style = int(GetWindowLongPtrW(wintypes.HWND(hwnd), GWL_STYLE))
    pid = wintypes.DWORD(0)
    GetWindowThreadProcessId(wintypes.HWND(hwnd), ctypes.byref(pid))
    our_pid = int(kernel32.GetCurrentProcessId())

    return {
        "hwnd": int(hwnd),
        "style": style,
        "style_hex": f"0x{style:08X}",
        "WS_THICKFRAME": bool(style & WS_THICKFRAME),
        "WS_MAXIMIZEBOX": bool(style & WS_MAXIMIZEBOX),
        "WS_CAPTION": bool(style & WS_CAPTION),
        "WS_SYSMENU": bool(style & WS_SYSMENU),
        "WS_MINIMIZEBOX": bool(style & WS_MINIMIZEBOX),
        "owner_pid": int(pid.value),
        "our_pid": our_pid,
        "owner_is_us": int(pid.value) == our_pid,
    }


def main() -> int:
    print(f"PLATFORM sys.platform={sys.platform}", flush=True)
    print(f"PYTHON {sys.version.split()[0]} {sys.executable}", flush=True)
    if sys.platform != "win32":
        print("FAIL not win32 — refuse to run", flush=True)
        return 2

    import numpy as np

    # PyVista before PySide6 — required on Windows (shiboken/six/matplotlib).
    import pyvista as pv
    from pyvistaqt import QtInteractor  # noqa: F401 — import path check
    from PySide6.QtWidgets import QApplication

    print(f"DEPS pyvista={pv.__version__} numpy={np.__version__}", flush=True)

    from app.theme import apply_theme, CURRENT_THEME
    from app.mainwindow import MainWindow

    app = QApplication(sys.argv)
    apply_theme(app)
    print(f"THEME {CURRENT_THEME} QT_QPA={os.environ.get('QT_QPA_PLATFORM','(unset)')}", flush=True)
    print(f"ENV LIBGL={os.environ.get('LIBGL_ALWAYS_SOFTWARE','(unset)')}", flush=True)

    win = MainWindow()
    win.resize(2560, 1440)
    win.show()
    _pump(app, 30)

    if not win.viewport._ok or win.viewport.plotter is None:
        print("FAIL viewport not ok", flush=True)
        return 1
    print("START_OK viewport ready", flush=True)

    # Force a real render + screenshot (not black)
    try:
        win.viewport._do_render()
    except Exception:
        pass
    _pump(app, 15)
    img = np.asarray(win.viewport.plotter.screenshot(return_img=True))
    print(f"FB shape={img.shape} dtype={img.dtype}", flush=True)
    # Non-black evidence
    mean = img.reshape(-1, img.shape[-1])[:, :3].mean(axis=0)
    mx = int(img.max())
    nonzero = int(np.count_nonzero(img))
    print(f"FB mean_rgb={mean.astype(int).tolist()} max={mx} nonzero={nonzero}", flush=True)
    if mx < 5 or nonzero < 1000:
        print("FAIL framebuffer looks black/empty", flush=True)
        return 1
    print("FB_OK non-black framebuffer", flush=True)

    # GL renderer string from VTK
    try:
        ren_win = win.viewport.plotter.ren_win
        # VTK often prints on first MakeCurrent; also query OpenGL
        try:
            from vtkmodules.vtkRenderingOpenGL2 import vtkOpenGLRenderWindow

            gl = ren_win
            # force init
            gl.Render()
        except Exception:
            pass
        # PyVista / VTK: get openGL info if available
        rstr = None
        try:
            # Many builds expose this via GPUInfo or scoper
            from vtkmodules.vtkRenderingCore import vtkRenderWindow

            # Read back last printed — also try scooby/pyvista report
            rstr = str(getattr(ren_win, "GetRenderingBackend", lambda: "")())
        except Exception:
            rstr = None
        # Robust: use OpenGL via vtk
        try:
            import vtk

            # vtkOpenGLRenderWindow has ReportCapabilities
            caps = ren_win.ReportCapabilities() if hasattr(ren_win, "ReportCapabilities") else ""
            print("GL_CAPS_BEGIN", flush=True)
            for line in (caps or "").splitlines():
                if any(
                    k in line
                    for k in (
                        "OpenGL vendor",
                        "OpenGL renderer",
                        "OpenGL version",
                        "DirectRendering",
                    )
                ):
                    print(f"GL {line.strip()}", flush=True)
            print("GL_CAPS_END", flush=True)
            # Also full renderer line extract
            for line in (caps or "").splitlines():
                if "OpenGL renderer" in line:
                    print(f"GL_RENDERER_LINE {line.strip()}", flush=True)
        except Exception as exc:
            print(f"GL_QUERY_FAIL {exc!r}", flush=True)
    except Exception as exc:
        print(f"GL_BLOCK_FAIL {exc!r}", flush=True)
        traceback.print_exc()

    # HWND / Win32 ownership
    try:
        wid = int(win.winId())
        info = _hwnd_info(wid)
        print(f"HWND {info}", flush=True)
        print(
            f"HWND_OWNER our_pid={info['our_pid']} owner_pid={info['owner_pid']} "
            f"owner_is_us={info['owner_is_us']}",
            flush=True,
        )
        print(
            f"HWND_STYLE {info['style_hex']} "
            f"WS_THICKFRAME={info['WS_THICKFRAME']} "
            f"WS_MAXIMIZEBOX={info['WS_MAXIMIZEBOX']} "
            f"WS_CAPTION={info['WS_CAPTION']} "
            f"WS_SYSMENU={info['WS_SYSMENU']} "
            f"WS_MINIMIZEBOX={info['WS_MINIMIZEBOX']}",
            flush=True,
        )
        if not info["owner_is_us"]:
            print(
                "DIAGNOSIS_NOTE owner_pid != our_pid — not a normal process-owned top-level window",
                flush=True,
            )
        else:
            print(
                "DIAGNOSIS_NOTE owner_pid == our_pid — real Win32 HWND owned by this process",
                flush=True,
            )
    except Exception as exc:
        print(f"HWND_FAIL {exc!r}", flush=True)
        traceback.print_exc()
        return 1

    # Frame time empty viewport @ 2560x1440 (show maximized-ish client area)
    # Use current size first; also try showFullScreen briefly if possible
    plotter = win.viewport.plotter
    times = []
    for i in range(40):
        t0 = time.perf_counter()
        plotter.render()
        # ensure GPU flush if available
        try:
            plotter.ren_win.WaitForCompletion()
        except Exception:
            pass
        times.append((time.perf_counter() - t0) * 1000.0)
        if i < 5:
            app.processEvents()
    times_s = sorted(times[5:])  # drop warmup
    med = float(np.median(times_s))
    p95 = float(times_s[int(0.95 * (len(times_s) - 1))])
    print(
        f"FRAMETIME_MS n={len(times_s)} median={med:.2f} p95={p95:.2f} "
        f"min={min(times_s):.2f} max={max(times_s):.2f} "
        f"window={win.width()}x{win.height()} "
        f"interactor={plotter.interactor.width()}x{plotter.interactor.height()}",
        flush=True,
    )

    # Fullscreen attempt for fill-rate floor
    try:
        win.showFullScreen()
        _pump(app, 25)
        plotter.render()
        ft = []
        for i in range(40):
            t0 = time.perf_counter()
            plotter.render()
            try:
                plotter.ren_win.WaitForCompletion()
            except Exception:
                pass
            ft.append((time.perf_counter() - t0) * 1000.0)
            if i < 5:
                app.processEvents()
        ft_s = sorted(ft[5:])
        med_fs = float(np.median(ft_s))
        print(
            f"FRAMETIME_FULLSCREEN_MS n={len(ft_s)} median={med_fs:.2f} "
            f"p95={float(ft_s[int(0.95*(len(ft_s)-1))]):.2f} "
            f"window={win.width()}x{win.height()} "
            f"interactor={plotter.interactor.width()}x{plotter.interactor.height()}",
            flush=True,
        )
        win.showNormal()
        _pump(app, 10)
    except Exception as exc:
        print(f"FULLSCREEN_SKIP {exc!r}", flush=True)

    print(
        "SNAP_GESTURE: not measured (interactive shell gesture; human drag-to-top required). "
        "Window styles/ownership above are the measurable prerequisites.",
        flush=True,
    )
    print("NATIVE_WIN_VERIFY_OK", flush=True)
    return 0


if __name__ == "__main__":
    # Ensure project root on path
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    os.environ.setdefault("PYTHONPATH", root)
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"EXC_FAIL {exc!r}", flush=True)
        traceback.print_exc()
        raise SystemExit(1)
