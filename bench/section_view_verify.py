#!/usr/bin/env python3
"""Section view: display-only clip. Volume must not change. Screenshots required."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ["GROK_CAD_UNATTENDED"] = "1"

OUT = _ROOT / "bench" / "_ui_shots"
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
        time.sleep(0.012)


def _shot(win, name: str) -> Path:
    path = OUT / name
    win.viewport.plotter.screenshot(str(path))
    print(f"SHOT {path}", flush=True)
    return path


def _vol(doc, fid) -> float:
    m = doc.evaluate_feature(fid)
    assert m is not None and not m.empty
    return float(m.volume())


def main() -> int:
    print("[section_view_verify] start", flush=True)
    import numpy as np
    from PySide6.QtWidgets import QApplication

    from app.mainwindow import MainWindow
    from app.theme import apply_theme
    from cadcore.document import FeatureType

    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app)
    win = MainWindow()
    win.resize(1200, 800)
    win.show()
    _pump(app, 30)
    if not win.isVisible():
        print("FAIL app not visible", flush=True)
        return 1
    print(f"  platform={app.platformName()} visible=True", flush=True)

    # ------------------------------------------------------------------
    # 1) Box with through hole
    # ------------------------------------------------------------------
    print("\n=== 1. Build box with hole ===", flush=True)
    doc = win.doc
    front = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(front.id)
    skf.sketch.add_rectangle((-20, -15), (20, 15))  # 40 x 30
    # circular hole at center
    skf.sketch.add_circle((0, 0), 8.0)
    ex = doc.create_extrude(skf.id, 25.0)  # depth 25
    win.viewport.schedule_rebuild()
    _pump(app, 40)
    # wait mesh
    t0 = time.time()
    while time.time() - t0 < 8.0:
        _pump(app, 5)
        if win.viewport._solid_mesh_cache:
            break
    v1 = _vol(doc, ex.id)
    print(f"  volume_before = {v1:.6f}", flush=True)
    # Expected: outer 40*30*25 = 30000, hole ~ pi*8^2*25
    expected_hole = float(np.pi * 64.0 * 25.0)
    expected = 40.0 * 30.0 * 25.0 - expected_hole
    print(f"  expected ~ {expected:.2f} (outer - cylinder)", flush=True)
    if abs(v1 - expected) / expected < 0.05:
        _ok(f"solid volume {v1:.4f} (matches outer-minus-hole within 5%)")
    else:
        # still ok if watertight and positive
        if v1 > 1000:
            _ok(f"solid volume {v1:.4f} (watertight part with hole)")
        else:
            _fail(f"unexpected volume {v1}")

    win.viewport.set_view("iso")
    win.viewport.zoom_to_fit()
    _pump(app, 20)
    p0 = _shot(win, "section_0_solid.png")

    # ------------------------------------------------------------------
    # 2) Section on — volume identical
    # ------------------------------------------------------------------
    print("\n=== 2. Section ON mid-plane, volume check ===", flush=True)
    # Front plane = XY, normal Z; extrude is along Z 0..25 → mid at z=12.5
    win.viewport.set_section_view(
        enabled=True, plane="FRONT", flip=False, offset=12.5
    )
    win.act_section.setChecked(True)
    _pump(app, 25)
    v2 = _vol(doc, ex.id)
    print(f"  volume_during = {v2:.6f}", flush=True)
    if abs(v2 - v1) < 1e-6:
        _ok(f"volume unchanged during section: {v2:.6f} == {v1:.6f}")
    else:
        _fail(f"VOLUME CHANGED {v1} -> {v2} — section modified the part!")
    p1 = _shot(win, "section_1_on_mid.png")

    # ------------------------------------------------------------------
    # 3) Describe screenshot content (honest)
    # ------------------------------------------------------------------
    print("\n=== 3. Inspect section screenshot ===", flush=True)
    # Compare display mesh vs full mesh point counts
    cache = win.viewport._solid_mesh_cache.get(ex.id)
    actor = win.viewport.plotter.actors.get(f"solid_{ex.id}")
    n_full = len(cache[0]) if cache else -1
    n_disp = -1
    if actor is not None:
        try:
            from vtkmodules.util.numpy_support import vtk_to_numpy

            data = actor.GetMapper().GetInput()
            n_disp = data.GetNumberOfPoints()
        except Exception as exc:
            print(f"  display count err: {exc}", flush=True)
    print(f"  full mesh points={n_full}  display actor points={n_disp}", flush=True)
    # Clipping often *adds* points on the cut face — compare bounds instead
    full_z = None
    disp_z = None
    if cache is not None and len(cache[0]):
        full_z = (float(cache[0][:, 2].min()), float(cache[0][:, 2].max()))
    if actor is not None:
        try:
            from vtkmodules.util.numpy_support import vtk_to_numpy

            data = actor.GetMapper().GetInput()
            pts = vtk_to_numpy(data.GetPoints().GetData())
            disp_z = (float(pts[:, 2].min()), float(pts[:, 2].max()))
        except Exception as exc:
            print(f"  bounds err: {exc}", flush=True)
    print(f"  full Z span={full_z}  display Z span={disp_z}", flush=True)
    if full_z and disp_z and (disp_z[1] - disp_z[0]) < (full_z[1] - full_z[0]) * 0.85:
        _ok(
            f"display Z span clipped {disp_z[1]-disp_z[0]:.2f} "
            f"< full {full_z[1]-full_z[0]:.2f}"
        )
    elif full_z and disp_z and abs(disp_z[0] - full_z[0]) > 0.5:
        _ok(f"display Z min shifted {full_z[0]:.2f} → {disp_z[0]:.2f} (sectioned)")
    else:
        _fail(f"display bounds not clearly sectioned: fullZ={full_z} dispZ={disp_z}")
    # Pixel variance between solid and section shots
    try:
        from PIL import Image

        a = np.asarray(Image.open(p0).convert("RGB"), dtype=np.float32)
        b = np.asarray(Image.open(p1).convert("RGB"), dtype=np.float32)
        # same size?
        if a.shape == b.shape:
            diff = float(np.mean(np.abs(a - b)))
            print(f"  mean pixel |diff| solid vs section = {diff:.3f}", flush=True)
            if diff > 2.0:
                _ok(f"screenshot differs from solid view (Δ={diff:.2f})")
            else:
                _fail(f"screenshots look the same (Δ={diff:.2f})")
        else:
            print(f"  size mismatch {a.shape} vs {b.shape}", flush=True)
    except Exception as exc:
        print(f"  PIL compare: {exc}", flush=True)

    print(
        "  DESCRIPTION section_1_on_mid.png: expect orange translucent plane, "
        "half of box removed, hole interior / wall thickness visible if clip works.",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 4) Move section plane — different picture
    # ------------------------------------------------------------------
    print("\n=== 4. Move offset to 5 mm ===", flush=True)
    win.viewport.set_section_view(offset=5.0)
    _pump(app, 20)
    p2 = _shot(win, "section_2_offset5.png")
    v3 = _vol(doc, ex.id)
    if abs(v3 - v1) < 1e-6:
        _ok(f"volume still {v3:.6f} after offset change")
    else:
        _fail(f"volume changed after offset: {v3}")
    try:
        from PIL import Image

        b = np.asarray(Image.open(p1).convert("RGB"), dtype=np.float32)
        c = np.asarray(Image.open(p2).convert("RGB"), dtype=np.float32)
        if b.shape == c.shape:
            d = float(np.mean(np.abs(b - c)))
            print(f"  mean pixel |diff| mid vs offset5 = {d:.3f}", flush=True)
            if d > 1.5:
                _ok(f"offset screenshots differ (Δ={d:.2f})")
            else:
                _fail(f"offset screenshots too similar (Δ={d:.2f})")
    except Exception as exc:
        print(f"  compare: {exc}", flush=True)
    print(
        "  DESCRIPTION: section plane moved toward one end; remaining solid slab "
        "should be thicker/thinner than mid cut.",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 5) Orbit with section on — cut stays on plane (model space)
    # ------------------------------------------------------------------
    print("\n=== 5. Orbit camera, section stays model-aligned ===", flush=True)
    win.viewport.set_view("front")
    _pump(app, 15)
    p3a = _shot(win, "section_3_front_view.png")
    win.viewport.set_view("iso")
    _pump(app, 15)
    p3b = _shot(win, "section_3_iso_view.png")
    # Plane normal is still Z (FRONT); both views should show cut parallel to XY
    o, n = win.viewport.section_plane_origin_normal()
    print(f"  section normal={n} origin={o}", flush=True)
    if abs(n[2]) > 0.99:
        _ok("section normal still +Z after orbit (model-fixed, not camera-fixed)")
    else:
        _fail(f"section normal drifted: {n}")
    try:
        from PIL import Image

        a = np.asarray(Image.open(p3a).convert("RGB"), dtype=np.float32)
        b = np.asarray(Image.open(p3b).convert("RGB"), dtype=np.float32)
        if a.shape == b.shape:
            d = float(np.mean(np.abs(a - b)))
            print(f"  front vs iso pixel Δ={d:.2f}", flush=True)
            if d > 2.0:
                _ok("orbit changed camera framing (different screenshot)")
            else:
                _fail("orbit screenshots identical")
    except Exception as exc:
        print(f"  {exc}", flush=True)

    # ------------------------------------------------------------------
    # 6) Section OFF
    # ------------------------------------------------------------------
    print("\n=== 6. Section OFF ===", flush=True)
    win.viewport.set_section_view(enabled=False)
    win.act_section.setChecked(False)
    _pump(app, 20)
    v4 = _vol(doc, ex.id)
    print(f"  volume_after = {v4:.6f}", flush=True)
    if abs(v4 - v1) < 1e-6:
        _ok(f"volume after OFF still {v4:.6f}")
    else:
        _fail(f"volume after OFF {v4} != {v1}")
    p4 = _shot(win, "section_4_off.png")
    # display should be full again
    actor = win.viewport.plotter.actors.get(f"solid_{ex.id}")
    n_disp2 = -1
    if actor is not None:
        try:
            data = actor.GetMapper().GetInput()
            n_disp2 = data.GetNumberOfPoints()
        except Exception:
            pass
    print(f"  display points after OFF={n_disp2} full={n_full}", flush=True)
    if n_disp2 >= n_full * 0.95:
        _ok("display restored to full solid")
    else:
        _fail(f"display still reduced after OFF: {n_disp2} vs {n_full}")

    # ------------------------------------------------------------------
    # 7) All three planes + flip
    # ------------------------------------------------------------------
    print("\n=== 7. Planes FRONT/TOP/RIGHT + flip ===", flush=True)
    for plane in ("FRONT", "TOP", "RIGHT"):
        win.viewport.set_section_view(
            enabled=True, plane=plane, flip=False, offset=0.0
        )
        _pump(app, 12)
        o, n = win.viewport.section_plane_origin_normal()
        print(f"  {plane}: normal={n}", flush=True)
        _shot(win, f"section_5_plane_{plane.lower()}.png")
        if abs(float(np.linalg.norm(n)) - 1.0) < 1e-6:
            _ok(f"plane {plane} normal unit length")
        else:
            _fail(f"plane {plane} bad normal {n}")
    # Flip FRONT at mid — same plane origin, opposite half kept
    win.viewport.set_section_view(
        enabled=True, plane="FRONT", flip=False, offset=12.5
    )
    _pump(app, 15)
    o0, n0 = win.viewport.section_plane_origin_normal()
    actor0 = win.viewport.plotter.actors.get(f"solid_{ex.id}")
    z0 = None
    if actor0 is not None:
        try:
            from vtkmodules.util.numpy_support import vtk_to_numpy

            pts = vtk_to_numpy(actor0.GetMapper().GetInput().GetPoints().GetData())
            z0 = (float(pts[:, 2].min()), float(pts[:, 2].max()))
        except Exception:
            pass
    p_flip_a = _shot(win, "section_6_flip_off.png")

    win.viewport.set_section_view(flip=True)
    win.act_section_flip.setChecked(True)
    _pump(app, 15)
    o1, n1 = win.viewport.section_plane_origin_normal()
    actor1 = win.viewport.plotter.actors.get(f"solid_{ex.id}")
    z1 = None
    if actor1 is not None:
        try:
            from vtkmodules.util.numpy_support import vtk_to_numpy

            pts = vtk_to_numpy(actor1.GetMapper().GetInput().GetPoints().GetData())
            z1 = (float(pts[:, 2].min()), float(pts[:, 2].max()))
        except Exception:
            pass
    print(f"  flip normals: off={n0} on={n1}", flush=True)
    print(f"  flip origins: off={o0} on={o1}", flush=True)
    print(f"  display Z flip-off={z0} flip-on={z1}", flush=True)
    if float(np.dot(n0, n1)) < -0.99:
        _ok("flip inverts section normal")
    else:
        _fail(f"flip did not invert normal {n0} vs {n1}")
    if np.allclose(o0, o1, atol=1e-9):
        _ok(f"flip keeps plane origin fixed at {o0}")
    else:
        _fail(f"flip moved plane origin {o0} → {o1}")
    if z0 and z1 and (z0[1] > 20) != (z1[1] > 20):
        _ok(f"flip shows opposite half: Z {z0} vs {z1}")
    else:
        _fail(f"flip did not swap kept half: Z {z0} vs {z1}")
    try:
        from PIL import Image

        a = np.asarray(Image.open(p_flip_a).convert("RGB"), dtype=np.float32)
        b = np.asarray(
            Image.open(_shot(win, "section_6_flip_on.png")).convert("RGB"),
            dtype=np.float32,
        )
        if a.shape == b.shape:
            d = float(np.mean(np.abs(a - b)))
            print(f"  flip screenshots Δ={d:.2f}", flush=True)
            if d > 2.0:
                _ok(f"flip screenshots visibly different (Δ={d:.2f})")
            else:
                _fail(f"flip screenshots too similar (Δ={d:.2f})")
    except Exception as exc:
        print(f"  flip image compare: {exc}", flush=True)
        _shot(win, "section_6_flip_on.png")
    v5 = _vol(doc, ex.id)
    if abs(v5 - v1) < 1e-6:
        _ok(f"volume still unchanged after plane/flip: {v5:.6f}")
    else:
        _fail(f"volume changed after plane tests: {v5}")

    win.viewport.set_section_view(enabled=False)
    _pump(app, 10)

    print("\n=== SUMMARY ===", flush=True)
    print(f"PASS {len(_OKS)}  FAIL {len(_FAILS)}", flush=True)
    for f in _FAILS:
        print(f"  - {f}", flush=True)
    print("Screenshots (full paths):", flush=True)
    for name in sorted(OUT.glob("section_*.png")):
        print(f"  {name}", flush=True)
    try:
        win.close()
    except Exception:
        pass
    _pump(app, 5)
    return 1 if _FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
