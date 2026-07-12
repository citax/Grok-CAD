#!/usr/bin/env python3
"""Sweep fillet radius × segments × hole-grid; multi-minute soak.

Each combo:
  1. Fillet a large square (CrossSection dual-offset) and extrude.
  2. Record fillet solid volume error vs analytic (L²−(4−π)r²)·h.
  3. Subtract a G×G grid of box cutters (CSG load for wall time).

Streams progress; writes bench/fillet_results.csv.
"""

from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cadcore.mesh import (  # noqa: E402
    BooleanOp,
    boolean_op,
    extrude_filleted_profile,
    extrude_rectangle,
)
from cadcore.sketch import EntityKind, PlaneFrame, RectEntity  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "fillet_results.csv"
LOG = Path(__file__).resolve().parent / "fillet_sweep.log"

L = 50.0
H = 12.0
RADII = [round(0.75 + i * 0.5, 4) for i in range(0, 28)]  # 0.75 .. 14.25
SEGMENTS = [8, 16, 32, 64]
HOLE_GRIDS = [4, 6, 8, 10, 12, 14, 16]
FRAME = PlaneFrame.from_plane_type("PLANE_FRONT")


def analytic(L: float, r: float, h: float) -> float:
    return (L * L - (4.0 - math.pi) * r * r) * h


def log(msg: str) -> None:
    line = msg if msg.endswith("\n") else msg + "\n"
    sys.stdout.write(line)
    sys.stdout.flush()
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line)


def main() -> int:
    LOG.write_text("# fillet_sweep start\n", encoding="utf-8")
    combos = [(r, s, g) for r in RADII for s in SEGMENTS for g in HOLE_GRIDS]
    log(
        f"=== Fillet sweep L={L} H={H} radii={len(RADII)} "
        f"segs={SEGMENTS} grids={HOLE_GRIDS} combos={len(combos)}"
    )
    log(f"Results → {RESULTS}")
    rows = []
    t_all = time.perf_counter()

    for idx, (r, segs, grid) in enumerate(combos, 1):
        t0 = time.perf_counter()
        rect = RectEntity(id=1, kind=EntityKind.RECTANGLE, c0=(0.0, 0.0), c1=(L, L))
        solid0 = extrude_filleted_profile(rect, H, FRAME, r, segments=segs)
        vol0 = float(solid0.volume())
        exact = analytic(L, r, H)
        err = abs(vol0 - exact) / exact if exact > 0 else 0.0
        tight0 = bool(solid0.is_watertight())

        solid = solid0
        margin = L / (grid + 1)
        half = min(margin * 0.30, 2.0)
        for i in range(grid):
            for j in range(grid):
                cx = margin * (i + 1)
                cy = margin * (j + 1)
                cutter = extrude_rectangle(
                    (cx - half, cy - half),
                    (cx + half, cy + half),
                    H * 2.5,
                    FRAME,
                ).translate((0.0, 0.0, -0.5))
                solid = boolean_op(solid, cutter, BooleanOp.DIFFERENCE)

        wall = time.perf_counter() - t0
        tight = bool(solid.is_watertight())
        row = {
            "radius": f"{r:.4f}",
            "segments": segs,
            "hole_grid": grid,
            "wall_s": f"{wall:.6f}",
            "tris": int(len(solid.faces)),
            "tris_fillet": int(len(solid0.faces)),
            "volume_fillet": f"{vol0:.8g}",
            "analytic": f"{exact:.8g}",
            "vol_err_rel": f"{err:.6e}",
            "watertight_fillet": int(tight0),
            "watertight": int(tight),
        }
        rows.append(row)
        log(
            f"[{idx}/{len(combos)}] r={r:.2f} segs={segs} grid={grid} "
            f"wall={wall:.3f}s tris={len(solid.faces)} "
            f"err={err * 100:.3f}% tight_f={tight0} tight={tight}"
        )

    with RESULTS.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    total = time.perf_counter() - t_all
    max_err = max(float(r["vol_err_rel"]) for r in rows)
    all_tight = all(int(r["watertight_fillet"]) == 1 for r in rows)
    log("")
    log("── Fillet sweep summary ──")
    log(
        f"rows={len(rows)} total_wall={total:.1f}s "
        f"max_fillet_vol_err={max_err * 100:.3f}% "
        f"all_fillet_watertight={all_tight}"
    )
    log(f"Wrote {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
