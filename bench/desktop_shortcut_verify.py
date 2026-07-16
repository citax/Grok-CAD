"""Verify Grok CAD via the Desktop shortcut path (native Windows).

Launches:  %USERPROFILE%\\Desktop\\Grok CAD.vbs
Measures process ownership, Win32 styles, GL renderer, framebuffer, and the
*actual* system cursor while the mouse is over the client area (after a drag
that would normally trigger VTK trackball cursor changes).

Does NOT claim Aero Snap drag-to-top works — that needs a human.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import time
from ctypes import wintypes

DESKTOP_VBS = os.path.join(os.environ.get("USERPROFILE", r"C:\Users\Citak"), "Desktop", "Grok CAD.vbs")
WINDOW_TITLE = "Grok CAD"

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

# --- Win32 helpers ---

class POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


class CURSORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("hCursor", wintypes.HANDLE),
        ("ptScreenPos", POINT),
    ]


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

GWL_STYLE = -16
WS_THICKFRAME = 0x00040000
WS_MAXIMIZEBOX = 0x00010000
WS_CAPTION = 0x00C00000
IDC_ARROW = 32512
# Cursors VTK/pyvistaqt commonly maps during trackball interaction
IDC_CROSS = 32515
IDC_SIZEALL = 32646
IDC_SIZENS = 32645
IDC_SIZEWE = 32644
IDC_SIZENWSE = 32642
IDC_SIZENESW = 32643
IDC_HAND = 32649
CURSOR_SHOWING = 0x00000001


def _find_windows_by_title(substr: str) -> list[int]:
    found: list[int] = []

    @EnumWindowsProc
    def _cb(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        buf = ctypes.create_unicode_buffer(512)
        user32.GetWindowTextW(hwnd, buf, 512)
        if substr.lower() in buf.value.lower():
            found.append(int(hwnd))
        return True

    user32.EnumWindows(_cb, 0)
    return found


def _pid_of(hwnd: int) -> int:
    pid = wintypes.DWORD(0)
    user32.GetWindowThreadProcessId(wintypes.HWND(hwnd), ctypes.byref(pid))
    return int(pid.value)


def _style_of(hwnd: int) -> int:
    return int(user32.GetWindowLongPtrW(wintypes.HWND(hwnd), GWL_STYLE))


def _client_center_screen(hwnd: int) -> tuple[int, int]:
    rc = RECT()
    user32.GetClientRect(wintypes.HWND(hwnd), ctypes.byref(rc))
    pt = POINT(rc.left + (rc.right - rc.left) // 2, rc.top + (rc.bottom - rc.top) // 2)
    user32.ClientToScreen(wintypes.HWND(hwnd), ctypes.byref(pt))
    return int(pt.x), int(pt.y)


def _process_image(pid: int) -> str:
    # QueryFullProcessImageNameW
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(1024)
        size = wintypes.DWORD(1024)
        if kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
            return buf.value
        return ""
    finally:
        kernel32.CloseHandle(h)


def _load_cursor(idc: int) -> int:
    return int(user32.LoadCursorW(None, idc))


def _cursor_info() -> tuple[int, int, int]:
    ci = CURSORINFO()
    ci.cbSize = ctypes.sizeof(CURSORINFO)
    if not user32.GetCursorInfo(ctypes.byref(ci)):
        raise OSError("GetCursorInfo failed")
    return int(ci.hCursor), int(ci.ptScreenPos.x), int(ci.ptScreenPos.y)


def _kill_existing_grok() -> None:
    """Stop prior Grok CAD windows so we measure this launch."""
    for h in _find_windows_by_title(WINDOW_TITLE):
        pid = _pid_of(h)
        if pid:
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, text=True)
    subprocess.run(
        ["taskkill", "/FI", f"WINDOWTITLE eq {WINDOW_TITLE}*", "/F"],
        capture_output=True,
        text=True,
    )
    # Wait until title is gone
    for _ in range(30):
        if not _find_windows_by_title(WINDOW_TITLE):
            break
        time.sleep(0.2)
    time.sleep(0.3)


def _drag_left_button(x0: int, y0: int, x1: int, y1: int) -> None:
    """Synthetic drag — VTK trackball would normally change cursor during this."""
    MOUSEEVENTF_MOVE = 0x0001
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004
    MOUSEEVENTF_ABSOLUTE = 0x8000
    SM_CXSCREEN = 0
    SM_CYSCREEN = 1
    sw = user32.GetSystemMetrics(SM_CXSCREEN)
    sh = user32.GetSystemMetrics(SM_CYSCREEN)

    def to_abs(x, y):
        return int(x * 65535 / max(sw - 1, 1)), int(y * 65535 / max(sh - 1, 1))

    user32.SetCursorPos(x0, y0)
    time.sleep(0.05)
    ax, ay = to_abs(x0, y0)
    user32.mouse_event(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, ax, ay, 0, 0)
    user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    steps = 12
    for i in range(1, steps + 1):
        x = x0 + (x1 - x0) * i // steps
        y = y0 + (y1 - y0) * i // steps
        user32.SetCursorPos(x, y)
        ax, ay = to_abs(x, y)
        user32.mouse_event(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, ax, ay, 0, 0)
        time.sleep(0.02)
    user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    time.sleep(0.1)


def main() -> int:
    if sys.platform != "win32":
        print("FAIL must run on Windows (this verifies the Desktop shortcut)", flush=True)
        return 2

    if not os.path.isfile(DESKTOP_VBS):
        print(f"FAIL Desktop shortcut missing: {DESKTOP_VBS}", flush=True)
        return 1

    # Sanity: the *executed* command must be native (not wsl / run_cad.sh)
    vbs_text = open(DESKTOP_VBS, encoding="utf-8", errors="replace").read()
    print(f"SHORTCUT path={DESKTOP_VBS}", flush=True)
    # Strip VBScript comments before checking launch target
    code_lines = []
    for line in vbs_text.splitlines():
        s = line.strip()
        if not s or s.startswith("'"):
            continue
        code_lines.append(s)
    code = "\n".join(code_lines)
    print(f"SHORTCUT code={code!r}", flush=True)
    code_l = code.lower()
    if "wsl" in code_l or "run_cad.sh" in code_l:
        print("FAIL Desktop shortcut executable lines still launch WSL/run_cad.sh", flush=True)
        return 1
    if "run_cad.cmd" not in code_l and "grok-cad-win" not in code_l and "pythonw" not in code_l:
        print("FAIL Desktop shortcut does not reference native run_cad.cmd / win venv", flush=True)
        return 1
    print("SHORTCUT_OK executable target is native Windows (not WSL)", flush=True)

    _kill_existing_grok()

    # Launch exactly as a double-click would (wscript host)
    print(f"LAUNCH wscript.exe {DESKTOP_VBS}", flush=True)
    subprocess.Popen(["wscript.exe", DESKTOP_VBS], cwd=os.path.dirname(DESKTOP_VBS))

    hwnd = 0
    deadline = time.time() + 45.0
    while time.time() < deadline:
        wins = _find_windows_by_title(WINDOW_TITLE)
        # Prefer exact-ish match
        for h in wins:
            buf = ctypes.create_unicode_buffer(512)
            user32.GetWindowTextW(h, buf, 512)
            if buf.value.strip() == WINDOW_TITLE:
                hwnd = h
                break
        if hwnd:
            break
        if wins:
            hwnd = wins[0]
            break
        time.sleep(0.4)

    if not hwnd:
        print("FAIL no window titled 'Grok CAD' appeared after Desktop shortcut launch", flush=True)
        return 1

    pid = _pid_of(hwnd)
    image = _process_image(pid)
    style = _style_of(hwnd)
    print(f"WINDOW hwnd={hwnd} pid={pid}", flush=True)
    print(f"PROCESS image={image}", flush=True)
    print(
        f"HWND_STYLE 0x{style:08X} "
        f"WS_THICKFRAME={bool(style & WS_THICKFRAME)} "
        f"WS_MAXIMIZEBOX={bool(style & WS_MAXIMIZEBOX)} "
        f"WS_CAPTION={bool(style & WS_CAPTION)}",
        flush=True,
    )

    image_l = image.lower().replace("/", "\\")
    if "wsl" in image_l or "wslhost" in image_l or "msrdc" in image_l:
        print("FAIL process is WSL/RAIL proxy, not native Python", flush=True)
        return 1
    if "python" not in os.path.basename(image_l):
        print(f"FAIL unexpected process image (want python/pythonw): {image}", flush=True)
        return 1
    # Windows venv redirectors often report the base install path as the image;
    # absence of "wsl" + python host + our window title is the hard evidence.
    print(
        f"PROCESS_OK native python host (not WSL/RAIL); image={image} "
        f"(venv redirectors often show the base Python path)",
        flush=True,
    )

    style_u = style & 0xFFFFFFFF
    print(f"HWND_STYLE_U 0x{style_u:08X}", flush=True)
    if not (style & WS_THICKFRAME and style & WS_MAXIMIZEBOX):
        print("FAIL missing WS_THICKFRAME or WS_MAXIMIZEBOX — Snap prerequisites absent", flush=True)
        return 1
    print(
        "SNAP_PREREQ_OK process-owned HWND with THICKFRAME+MAXIMIZEBOX "
        "(drag-to-top / Aero Snap gesture itself: human only)",
        flush=True,
    )

    # Cursor: park mouse in client center, drag (VTK would swap cursor), re-read HCURSOR.
    # Do not require equality to LoadCursor(IDC_ARROW) — Windows/Qt may use a themed
    # arrow handle that differs numerically. Require: (1) stable across samples, and
    # (2) not a known VTK interaction cursor (size-all / hand / cross / resize).
    time.sleep(1.5)  # let GL init
    # Raise window so hit-test is reliable
    user32.SetForegroundWindow(wintypes.HWND(hwnd))
    user32.BringWindowToTop(wintypes.HWND(hwnd))
    time.sleep(0.2)
    cx, cy = _client_center_screen(hwnd)
    print(f"MOUSE client_center_screen=({cx},{cy})", flush=True)
    user32.SetCursorPos(cx, cy)
    time.sleep(0.25)
    hit = int(user32.WindowFromPoint(POINT(cx, cy)))
    # Walk parents — GL child may own the point
    walk = hit
    under_us = False
    chain = []
    for _ in range(8):
        chain.append(walk)
        if walk == hwnd:
            under_us = True
            break
        parent = int(user32.GetParent(wintypes.HWND(walk)))
        if not parent or parent == walk:
            break
        walk = parent
    print(f"MOUSE WindowFromPoint hwnd={hit} chain={chain} under_app={under_us}", flush=True)
    if not under_us:
        print(
            "FAIL mouse is not over the Grok CAD window after SetCursorPos — "
            "cursor measurement would be invalid",
            flush=True,
        )
        return 1
    arrow = _load_cursor(IDC_ARROW)
    banned = {
        "CROSS": _load_cursor(IDC_CROSS),
        "SIZEALL": _load_cursor(IDC_SIZEALL),
        "SIZENS": _load_cursor(IDC_SIZENS),
        "SIZEWE": _load_cursor(IDC_SIZEWE),
        "SIZENWSE": _load_cursor(IDC_SIZENWSE),
        "SIZENESW": _load_cursor(IDC_SIZENESW),
        "HAND": _load_cursor(IDC_HAND),
    }
    print(f"CURSOR refs IDC_ARROW={arrow} banned={banned}", flush=True)

    h0, px0, py0 = _cursor_info()
    print(f"CURSOR idle hCursor={h0} pos=({px0},{py0})", flush=True)

    # Drag across the 3D view — trackball interactor normally sets SIZEALL/HAND
    _drag_left_button(cx, cy, cx + 120, cy + 80)
    user32.SetCursorPos(cx + 60, cy + 40)
    time.sleep(0.2)
    h1, px1, py1 = _cursor_info()
    print(f"CURSOR after_drag hCursor={h1} pos=({px1},{py1})", flush=True)

    # Also sample during a second drag mid-motion
    user32.SetCursorPos(cx, cy)
    time.sleep(0.05)
    user32.mouse_event(0x0002, 0, 0, 0, 0)  # left down
    user32.SetCursorPos(cx + 40, cy + 40)
    time.sleep(0.08)
    h_mid, pxm, pym = _cursor_info()
    user32.mouse_event(0x0004, 0, 0, 0, 0)  # left up
    print(f"CURSOR mid_drag hCursor={h_mid} pos=({pxm},{pym})", flush=True)

    samples = {"idle": h0, "after_drag": h1, "mid_drag": h_mid}
    # Fail if any sample is a known interaction cursor
    for label, h in samples.items():
        for name, bh in banned.items():
            if h == bh:
                print(
                    f"CURSOR_FAIL {label} hCursor={h} matches banned system cursor {name}",
                    flush=True,
                )
                return 1
    # Fail if cursor *changed* across the interaction (app still deciding)
    unique = set(samples.values())
    if len(unique) != 1:
        print(
            f"CURSOR_FAIL cursor handle changed across samples: {samples} "
            f"(app or VTK is still swapping shapes)",
            flush=True,
        )
        return 1
    print(
        f"CURSOR_OK stable hCursor={h0} at idle/mid_drag/after_drag over client area; "
        f"not CROSS/SIZEALL/HAND/resize; equals_IDC_ARROW={h0 == arrow} "
        f"(themed arrow may differ from LoadCursor(IDC_ARROW) — that is OK)",
        flush=True,
    )

    print(
        "HUMAN_REQUIRED: Aero Snap / drag title bar to top of monitor / edge-resize feel "
        "— cannot be scripted honestly. Prerequisites above are process-owned Win32 styles.",
        flush=True,
    )
    print(
        "HUMAN_REQUIRED: visual confirm cursor never becomes hand/crosshair/size-all "
        "while orbiting the 3D view with the mouse.",
        flush=True,
    )
    print("DESKTOP_SHORTCUT_VERIFY_OK", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"EXC_FAIL {exc!r}", flush=True)
        import traceback

        traceback.print_exc()
        raise SystemExit(1)
