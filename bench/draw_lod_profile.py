#!/usr/bin/env python3
"""Per-actor / per-group render cost at 2560x1440 with draw-LOD active.

hide-group → measure N frames → restore. Also measures combinations because
overdraw/blending is non-additive. Prints PROFILE table + empty-scene floor.
"""
from __future__ import annotations

import os
import statistics
import sys
import time

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("QT_XCB_GL_INTEGRATION", "none")
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")


def _median_render_ms(vp, n: int = 12) -> float:
    times = []
    for _ in range(n):
        if vp._render_timer.isActive():
            vp._render_timer.stop()
        t0 = time.perf_counter()
        vp._do_render()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def _set_vis(vp, names, visible: bool) -> None:
    for n in names:
        act = vp._get_named_actor(n)
        if act is None and vp.plotter:
            act = vp.plotter.actors.get(n)
        if act is None:
            continue
        try:
            act.SetVisibility(1 if visible else 0)
        except Exception:
            pass


def _names_matching(vp, pred) -> list[str]:
    names = set()
    if vp.plotter:
        names.update(vp.plotter.actors.keys())
    names.update(getattr(vp, "_overlay_actors", {}).keys())
    return sorted(n for n in names if pred(n))


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from app.mainwindow import MainWindow
    from app.sketch_mode import SketchTool
    from app.theme import apply_theme
    from cadcore.document import FeatureType

    app = QApplication(sys.argv)
    apply_theme(app)
    win = MainWindow()
    win.resize(2560, 1440)
    win.show()
    for _ in range(30):
        app.processEvents()
        time.sleep(0.01)
    if not win.viewport._ok:
        print("PROFILE_FAIL viewport", flush=True)
        return 1

    doc = win.doc
    front = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(front.id)
    sk = skf.sketch
    for i in range(60):
        r, c = divmod(i, 10)
        sk.add_line((c * 0.3, r * 0.3), (c * 0.3 + 0.15, r * 0.3))
    # try a few extrudes if API exists
    try:
        for i in range(3):
            sk2 = doc.create_sketch_on_plane(front.id)
            sk2.sketch.add_rectangle((-1.5 + i * 0.4, -1.5), (-1.0 + i * 0.4, -1.0))
            if hasattr(doc, "add_extrude"):
                doc.add_extrude(sk2.id, depth=0.4)
    except Exception as exc:
        print(f"[profile] extrude skip: {exc}", flush=True)

    win.viewport.set_document(doc)
    for _ in range(15):
        app.processEvents()
        time.sleep(0.01)
    win.viewport.enter_sketch(skf.id)
    for _ in range(12):
        app.processEvents()
        time.sleep(0.01)

    vp = win.viewport
    ctrl = vp._sketch_ctrl
    ctrl.set_tool(SketchTool.LINE)
    # Start a stroke so draw-LOD is active (same as real draw path)
    iw = vp.plotter.interactor
    cx, cy = iw.width() // 2, iw.height() // 2
    vp._sketch_mouse_press(cx - 100, cy + 50)
    vp._begin_draw_lod()
    assert vp._draw_lod_active

    # Collect groups
    planes = _names_matching(vp, lambda n: n.startswith("plane_"))
    edges = _names_matching(vp, lambda n: n.startswith("edge_"))
    solids = _names_matching(vp, lambda n: n.startswith("solid_"))
    sk_ents = _names_matching(vp, lambda n: n.startswith("sk_e_"))
    chrome = _names_matching(
        vp,
        lambda n: n
        in (
            "__sk_grid",
            "__sk_h",
            "__sk_v",
            "__sk_junctions",
            "__sk_dims",
            "__sk_handles",
            "__origin",
            "__ax",
            "__ay",
            "__az",
        )
        or n.startswith("__sk_dim"),
    )
    # Current LOD hides chrome+edges+solids — not planes
    lod_targets = list(dict.fromkeys(chrome + edges + solids))

    print(
        f"PROFILE size=2560x1440 draw_lod={int(vp._draw_lod_active)} "
        f"planes={planes} solids={len(solids)} sk_e={len(sk_ents)} "
        f"lod_targets={len(lod_targets)}",
        flush=True,
    )

    # Warmup
    _median_render_ms(vp, 4)

    base = _median_render_ms(vp, 15)
    print(f"BASE_WITH_LOD_ACTIVE median={base:.2f}ms", flush=True)

    def measure_hidden(label: str, names: list[str]) -> float:
        # save vis
        saved = {}
        for n in names:
            act = vp._get_named_actor(n) or (
                vp.plotter.actors.get(n) if vp.plotter else None
            )
            if act is None:
                continue
            try:
                saved[n] = act.GetVisibility()
                act.SetVisibility(0)
            except Exception:
                pass
        med = _median_render_ms(vp, 12)
        for n, v in saved.items():
            act = vp._get_named_actor(n) or (
                vp.plotter.actors.get(n) if vp.plotter else None
            )
            if act is not None:
                try:
                    act.SetVisibility(v)
                except Exception:
                    pass
        saved_ms = base - med
        print(
            f"GROUP {label:28s}  n={len(names):3d}  med={med:6.2f}ms  "
            f"delta_vs_base={saved_ms:+6.2f}ms  ratio={med/base:.3f}",
            flush=True,
        )
        return med

    # Singles
    measure_hidden("planes_only", planes)
    measure_hidden("solids_only", solids)
    measure_hidden("sk_entities", sk_ents)
    measure_hidden("chrome_grid_labels", chrome)
    measure_hidden("plane_edges", edges)
    measure_hidden("current_lod_targets", lod_targets)

    # Combinations (non-additive overdraw)
    measure_hidden("planes+solids", planes + solids)
    measure_hidden("planes+sk_e", planes + sk_ents)
    measure_hidden("planes+lod_targets", planes + lod_targets)
    measure_hidden("planes+solids+sk_e", planes + solids + sk_ents)
    measure_hidden(
        "all_above",
        list(dict.fromkeys(planes + solids + sk_ents + chrome + edges)),
    )

    # Empty floor: hide everything
    all_names = _names_matching(vp, lambda n: True)
    empty = measure_hidden("EMPTY_all_actors", all_names)
    print(
        f"FLOOR empty={empty:.2f}ms  base_above_floor={base - empty:.2f}ms  "
        f"app_side_target_is_above_floor",
        flush=True,
    )

    # A/B: current LOD on vs force all lod targets visible (LOD effectively off)
    # With LOD already active (targets hidden), measure "LOD off" by showing them
    for n in lod_targets:
        act = vp._get_named_actor(n) or (
            vp.plotter.actors.get(n) if vp.plotter else None
        )
        if act:
            try:
                act.SetVisibility(1)
            except Exception:
                pass
    lod_off = _median_render_ms(vp, 12)
    for n in lod_targets:
        act = vp._get_named_actor(n) or (
            vp.plotter.actors.get(n) if vp.plotter else None
        )
        if act:
            try:
                act.SetVisibility(0)
            except Exception:
                pass
    lod_on = _median_render_ms(vp, 12)
    print(
        f"AB_CURRENT_LOD off={lod_off:.2f}ms on={lod_on:.2f}ms "
        f"ratio={lod_on/lod_off if lod_off else 0:.3f} "
        f"(1.00=worthless)",
        flush=True,
    )

    # A/B: hide planes during stroke (proposed retarget)
    for n in planes:
        act = vp._get_named_actor(n) or (
            vp.plotter.actors.get(n) if vp.plotter else None
        )
        if act:
            try:
                act.SetVisibility(0)
            except Exception:
                pass
    planes_hid = _median_render_ms(vp, 12)
    for n in planes:
        act = vp._get_named_actor(n) or (
            vp.plotter.actors.get(n) if vp.plotter else None
        )
        if act:
            try:
                act.SetVisibility(1)
            except Exception:
                pass
    planes_vis = _median_render_ms(vp, 12)
    print(
        f"AB_HIDE_PLANES visible={planes_vis:.2f}ms hidden={planes_hid:.2f}ms "
        f"ratio={planes_hid/planes_vis if planes_vis else 0:.3f} "
        f"saved={planes_vis-planes_hid:+.2f}ms",
        flush=True,
    )

    # A/B: make planes opaque instead of translucent
    saved_op = {}
    for n in planes:
        act = vp._get_named_actor(n) or (
            vp.plotter.actors.get(n) if vp.plotter else None
        )
        if act is None:
            continue
        try:
            prop = act.GetProperty()
            saved_op[n] = prop.GetOpacity()
            prop.SetOpacity(1.0)
        except Exception:
            pass
    opaque = _median_render_ms(vp, 12)
    for n, op in saved_op.items():
        act = vp._get_named_actor(n) or (
            vp.plotter.actors.get(n) if vp.plotter else None
        )
        if act:
            try:
                act.GetProperty().SetOpacity(op)
            except Exception:
                pass
    translucent = _median_render_ms(vp, 12)
    print(
        f"AB_PLANE_OPAQUE translucent={translucent:.2f}ms opaque={opaque:.2f}ms "
        f"ratio={opaque/translucent if translucent else 0:.3f} "
        f"saved={translucent-opaque:+.2f}ms",
        flush=True,
    )

    print("PROFILE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
