#!/usr/bin/env python3
"""Headless performance soak / regression benchmark for extrude + CSG holes.

Workflow per step M:
  1. Extrude a large closed rectangular profile (manifold pad).
  2. Subtract an M×M grid of high-resolution cylindrical holes.
  3. Record cumulative rebuild wall-clock time and triangle count.

Sized intentionally so a full sweep takes several minutes of wall time.
Results are written to bench/results.csv next to this script.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

# Allow running as `python bench/extrude_bench.py` from repo root
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cadcore.mesh import (  # noqa: E402
    BooleanOp,
    boolean_op,
    extrude_circle,
    extrude_rectangle,
)
from cadcore.sketch import PlaneFrame  # noqa: E402

# ---------------------------------------------------------------------------
# Parameters — tuned for multi-minute soak on typical WSL/laptop CPUs
# ---------------------------------------------------------------------------
PLATE_U = 100.0
PLATE_V = 100.0
EXTRUDE_DIST = 8.0

# High circular resolution makes each boolean costly
HOLE_SEGMENTS = 80
# M sweep: each step rebuilds plate + M×M holes from scratch
# 3..24 step 1 → many steps; largest has 576 holes
M_VALUES = list(range(3, 25))

REGRESSION_FACTOR = 10.0  # flag if time/hole grows more than this vs previous

RESULTS_PATH = Path(__file__).resolve().parent / "results.csv"
FRAME = PlaneFrame.from_plane_type("PLANE_FRONT")


def build_plate():
    half_u, half_v = PLATE_U * 0.5, PLATE_V * 0.5
    return extrude_rectangle((-half_u, -half_v), (half_u, half_v), EXTRUDE_DIST, FRAME)


def rebuild_with_holes(m: int):
    """Extrude base plate, subtract M×M cylindrical holes; return mesh + hole count."""
    solid = build_plate()
    margin = PLATE_U / (m + 1)
    r = min(margin * 0.30, 2.8)
    half = PLATE_U * 0.5
    # Through-hole: extrude taller than plate, shift so it fully penetrates
    hole_h = EXTRUDE_DIST + 4.0
    z_shift = -2.0
    n_holes = 0
    for i in range(m):
        for j in range(m):
            x = -half + margin * (i + 1)
            y = -half + margin * (j + 1)
            hole = extrude_circle((x, y), r, hole_h, FRAME, segments=HOLE_SEGMENTS)
            hole = hole.translate((0.0, 0.0, z_shift))
            solid = boolean_op(solid, hole, BooleanOp.DIFFERENCE)
            n_holes += 1
    return solid, n_holes


def main() -> int:
    print("=== Extrude + hole-grid performance benchmark ===")
    print(f"Plate: {PLATE_U}×{PLATE_V}×{EXTRUDE_DIST}  hole_segs={HOLE_SEGMENTS}")
    print(f"M values: {M_VALUES[0]}..{M_VALUES[-1]} ({len(M_VALUES)} steps)")
    print(f"Results → {RESULTS_PATH}")
    print()

    t0 = time.perf_counter()
    plate = build_plate()
    print(
        f"Warm-up extrude: {time.perf_counter() - t0:.3f}s  "
        f"tris={len(plate.faces)} watertight={plate.is_watertight()}"
    )

    rows = []
    prev_time_per_hole = None
    flags = []

    for m in M_VALUES:
        t0 = time.perf_counter()
        solid, n_holes = rebuild_with_holes(m)
        wall = time.perf_counter() - t0
        tris = int(len(solid.faces))
        verts = int(len(solid.vertices))
        vol = float(solid.volume())
        tight = bool(solid.is_watertight())
        tph = wall / max(n_holes, 1)
        flag = ""
        if prev_time_per_hole is not None and prev_time_per_hole > 1e-9:
            ratio = tph / prev_time_per_hole
            if ratio > REGRESSION_FACTOR:
                flag = "REGRESSION"
                flags.append((m, ratio, wall, tris))
        prev_time_per_hole = tph
        row = {
            "M": m,
            "holes": n_holes,
            "wall_s": f"{wall:.4f}",
            "tris": tris,
            "verts": verts,
            "volume": f"{vol:.6g}",
            "watertight": int(tight),
            "s_per_hole": f"{tph:.6f}",
            "flag": flag,
        }
        rows.append(row)
        mark = f"  *** {flag} ***" if flag else ""
        print(
            f"M={m:3d}  holes={n_holes:4d}  wall={wall:8.2f}s  "
            f"tris={tris:8d}  vol={vol:10.1f}  tight={tight}{mark}",
            flush=True,
        )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "M",
                "holes",
                "wall_s",
                "tris",
                "verts",
                "volume",
                "watertight",
                "s_per_hole",
                "flag",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("── Summary ──")
    print(f"{'M':>4} {'holes':>6} {'wall_s':>10} {'tris':>10} {'s/hole':>10} flag")
    for r in rows:
        print(
            f"{int(r['M']):4d} {int(r['holes']):6d} {float(r['wall_s']):10.2f} "
            f"{int(r['tris']):10d} {float(r['s_per_hole']):10.4f} {r['flag']}"
        )
    total = sum(float(r["wall_s"]) for r in rows)
    print(f"\nTotal wall time: {total:.1f}s  steps={len(rows)}")
    print(f"Wrote {RESULTS_PATH}")
    if flags:
        print(f"\nRegression flags ({len(flags)}):")
        for m, ratio, wall, tris in flags:
            print(
                f"  M={m}: time/hole ×{ratio:.1f} vs previous "
                f"(wall={wall:.1f}s, tris={tris})"
            )
    else:
        print("\nNo severe per-step regressions flagged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
