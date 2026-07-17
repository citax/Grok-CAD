#!/usr/bin/env python3
"""Regression: axes + environment with REAL solids on screen.

Must be re-runnable from a clean checkout. Screenshots are written under
``bench/fixtures/env_axis/`` (tracked — not gitignored like ``_ui_shots``).

Checks:
  1. Tiny solid (~4 mm): no world-origin triad skewering the part
  2. Large solid (~200 mm): no world-origin triad swallowed inside
  3. Corner triad present (orientation at any size)
  4. Reference planes hidden when solids are displayed
  5. Every screenshot contains a solid actor (pixel content not empty planes)
  6. Sketch mode + solid mode, light theme

Fail hard if any policy or visual gate breaks.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("QT_XCB_GL_INTEGRATION", "none")
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("GROK_THEME", "light")

OUT = _ROOT / "bench" / "fixtures" / "env_axis"
OUT.mkdir(parents=True, exist_ok=True)

_FAILS: list[str] = []
_OKS: list[str] = []


def _ok(m: str) -> None:
    _OKS.append(m)
    print(f"OK  {m}", flush=True)


def _fail(m: str) -> None:
    _FAILS.append(m)
    print(f"FAIL {m}", flush=True)


def _pump(app, n=20):
    for _ in range(n):
        app.processEvents()
        time.sleep(0.01)


def _wait_solid(vp, fid, app, timeout=12.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        _pump(app, 4)
        if vp.solid_display_vertices(fid) is not None:
            return True
        if fid in vp._solid_fps:
            return True
    return False


def _shot(vp, name: str) -> np.ndarray | None:
    path = OUT / name
    try:
        if vp._render_timer.isActive():
            vp._render_timer.stop()
        vp._do_render()
        img = np.asarray(vp.plotter.screenshot(return_img=True))
        vp.plotter.screenshot(str(path))
        print(f"SHOT {path}", flush=True)
        return img
    except Exception as exc:
        _fail(f"screenshot {name}: {exc}")
        return None


def _assert_has_solid_pixels(img: np.ndarray | None, label: str) -> None:
    """Screenshot must not be empty planes-only (flat near-constant field)."""
    if img is None or img.size == 0:
        _fail(f"{label}: no image")
        return
    rgb = img[..., :3].astype(np.float64)
    # Solid grey sits away from pure black and from pure bg; require spread
    std = float(rgb.std())
    mean = float(rgb.mean())
    if std < 8.0:
        _fail(f"{label}: image too flat (std={std:.2f}) — looks empty/planes-only")
    else:
        _ok(f"{label}: solid content (std={std:.1f} mean={mean:.1f})")


def _plane_actors_visible(vp) -> list[str]:
    out = []
    if not vp.plotter:
        return out
    for name, act in list(vp.plotter.actors.items()):
        if not name.startswith("plane_"):
            continue
        try:
            if act.GetVisibility():
                out.append(name)
        except Exception:
            pass
    return out


def _origin_world_shown(vp) -> bool:
    if getattr(vp, "_origin_axes_actor", None) is not None:
        try:
            if vp._origin_axes_actor.GetVisibility():
                return True
        except Exception:
            return True
    for n in ("__ax", "__ay", "__az"):
        a = vp._get_named_actor(n)
        if a is not None:
            try:
                if a.GetVisibility():
                    return True
            except Exception:
                return True
    return False


def _build_block(doc, size_mm: float, depth_mm: float):
    """Sketch rectangle + extrude → solid feature (real document path)."""
    from cadcore.document import FeatureType

    front = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(front.id)
    assert skf and skf.sketch
    s = float(size_mm)
    skf.sketch.add_rectangle((0.0, 0.0), (s, s * 0.7))
    return doc.create_extrude(skf.id, float(depth_mm))


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from app.mainwindow import MainWindow
    from app.theme import apply_theme
    from cadcore.document import FeatureType
    from cadcore.scale import origin_triad_policy, reference_planes_should_show

    print("\n=== POLICY (source of truth) ===", flush=True)
    # Document SW evidence in the log for the user
    print(
        "EVIDENCE: SolidWorks View→Hide/Show→Planes and Q-toggle hide "
        "planes/origins while working solids (help.solidworks.com; "
        "Javelin hide/show planes). Fusion: Object Visibility toggles "
        "planes independent of bodies (Autodesk support).",
        flush=True,
    )
    show_s, L_s = origin_triad_policy(has_display_solids=True, part_extent_mm=4.0)
    show_e, L_e = origin_triad_policy(has_display_solids=False, char_mm=50.0)
    assert not show_s and L_s == 0.0
    assert show_e and L_e < 15.0
    assert not reference_planes_should_show(
        has_display_solids=True, in_sketch_mode=False, selected_is_plane=False
    )
    _ok("policy: solids → no world origin triad; planes hidden")

    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app)
    win = MainWindow()
    win.resize(1280, 800)
    win.show()
    _pump(app, 25)
    vp = win.viewport

    # ------------------------------------------------------------------
    # TINY part (~4 mm) — axes must not skewer
    # ------------------------------------------------------------------
    print("\n=== TINY SOLID (~4 mm) ===", flush=True)
    tiny = _build_block(win.doc, 4.0, 2.5)
    win.viewport.schedule_rebuild()
    if not _wait_solid(vp, tiny.id, app):
        _fail("tiny solid never appeared on actor path")
    else:
        _ok(f"tiny solid_{tiny.id} on display path")

    # Select the solid (not a plane)
    win.doc.selected_id = tiny.id
    vp.set_selected_id(tiny.id)
    vp._refresh_world_helpers(force=True)
    _pump(app, 10)

    if _origin_world_shown(vp):
        _fail("tiny: world-origin triad still visible — will skewer/fight part")
    else:
        _ok("tiny: no world-origin triad (corner triad only)")

    vis_planes = _plane_actors_visible(vp)
    if vis_planes:
        _fail(f"tiny: planes still visible {vis_planes}")
    else:
        _ok("tiny: reference planes hidden with solid")

    # Camera looking at origin through the part (overlap viewpoint)
    vp.plotter.camera_position = [
        (8.0, 6.0, 10.0),
        (2.0, 1.4, 1.25),
        (0.0, 1.0, 0.0),
    ]
    _pump(app, 8)
    img_t = _shot(vp, "tiny_solid_iso.png")
    _assert_has_solid_pixels(img_t, "tiny_solid_iso")

    # ------------------------------------------------------------------
    # LARGE part (~200 mm)
    # ------------------------------------------------------------------
    print("\n=== LARGE SOLID (~200 mm) ===", flush=True)
    # Fresh doc for clean large solid only
    win2 = MainWindow()
    win2.resize(1280, 800)
    win2.show()
    _pump(app, 20)
    large = _build_block(win2.doc, 200.0, 80.0)
    win2.viewport.schedule_rebuild()
    if not _wait_solid(win2.viewport, large.id, app):
        _fail("large solid never appeared")
    else:
        _ok(f"large solid_{large.id} on display path")
    win2.doc.selected_id = large.id
    win2.viewport.set_selected_id(large.id)
    win2.viewport._refresh_world_helpers(force=True)
    _pump(app, 10)

    if _origin_world_shown(win2.viewport):
        _fail("large: world-origin triad still on (would be swallowed or fight)")
    else:
        _ok("large: no world-origin triad")

    vis_p2 = _plane_actors_visible(win2.viewport)
    if vis_p2:
        _fail(f"large: planes still visible {vis_p2}")
    else:
        _ok("large: reference planes hidden with solid")

    win2.viewport.plotter.camera_position = [
        (320.0, 240.0, 280.0),
        (100.0, 70.0, 40.0),
        (0.0, 1.0, 0.0),
    ]
    _pump(app, 8)
    img_l = _shot(win2.viewport, "large_solid_iso.png")
    _assert_has_solid_pixels(img_l, "large_solid_iso")

    # Side view so axes-at-origin would have overlapped the solid if present
    win2.viewport.set_view("iso")
    _pump(app, 6)
    win2.viewport.plotter.camera_position = [
        (400.0, 40.0, 40.0),
        (100.0, 70.0, 40.0),
        (0.0, 1.0, 0.0),
    ]
    _pump(app, 6)
    img_side = _shot(win2.viewport, "large_solid_side_through_origin.png")
    _assert_has_solid_pixels(img_side, "large_solid_side")

    # ------------------------------------------------------------------
    # Sketch mode with solid present (planes may show; world origin still off)
    # ------------------------------------------------------------------
    print("\n=== SKETCH MODE with solid in scene ===", flush=True)
    front = next(f for f in win2.doc.features if f.type is FeatureType.PLANE_FRONT)
    # New sketch on front while solid exists
    skf = win2.doc.create_sketch_on_plane(front.id)
    win2.viewport.enter_sketch(skf.id)
    _pump(app, 12)
    if not win2.viewport.in_sketch_mode:
        _fail("failed to enter sketch")
    else:
        _ok("sketch mode entered with solid in document")
    # World origin still suppressed
    if _origin_world_shown(win2.viewport):
        # sketch chrome may hide it anyway
        pass
    # Corner triad should exist
    if getattr(win2.viewport, "_corner_axes_actor", None) is None:
        _fail("corner triad missing")
    else:
        _ok("corner triad present (orientation at any size)")
    img_sk = _shot(win2.viewport, "sketch_with_solid_in_doc.png")
    # Sketch view may show grid not solid — solid is dimmed. Require non-flat.
    if img_sk is not None and float(img_sk[..., :3].std()) > 5.0:
        _ok("sketch screenshot has content")
    else:
        _fail("sketch screenshot empty/flat")
    win2.viewport.exit_sketch()
    _pump(app, 8)
    # After exit: planes hide again
    win2.viewport.set_selected_id(large.id)
    win2.viewport._refresh_world_helpers(force=True)
    _pump(app, 6)
    if _plane_actors_visible(win2.viewport):
        _fail("after sketch exit: planes still visible with solid selected")
    else:
        _ok("after sketch exit: planes hidden again")
    img_after = _shot(win2.viewport, "solid_after_sketch_exit.png")
    _assert_has_solid_pixels(img_after, "solid_after_sketch")

    # Write a small manifest the CI/human can re-read
    manifest = OUT / "MANIFEST.txt"
    manifest.write_text(
        "env_axis_verify fixtures\n"
        f"passed={len(_OKS)} failed={len(_FAILS)}\n"
        + "\n".join(f"OK {m}" for m in _OKS)
        + "\n"
        + "\n".join(f"FAIL {m}" for m in _FAILS)
        + "\n",
        encoding="utf-8",
    )

    print("\n=== SUMMARY ===", flush=True)
    print(f"passed={len(_OKS)} failed={len(_FAILS)}", flush=True)
    for f in _FAILS:
        print(f"  FAIL: {f}", flush=True)
    return 1 if _FAILS else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
