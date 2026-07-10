#!/usr/bin/env python3
"""Headless soak benchmark for revolve at increasing angular resolution / cutters.

Each step M:
  1. Revolve a large closed rectangular profile about the V-axis (angular segs grow with M).
  2. Subtract M rings of high-resolution revolved circular cutters (feature count grows).
  3. Record wall time, triangle count, and watertightness.

Sized so a full sweep takes several minutes of wall time.
Results → bench/revolve_results.csv
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cadcore.mesh import (  # noqa: E402
    BooleanOp,
    boolean_op,
    revolve_circle,
    revolve_rectangle,
)
from cadcore.sketch import PlaneFrame  # noqa: E402

FRAME = PlaneFrame.from_plane_type("PLANE_FRONT")

# Large profile offset from V-axis
PROFILE_C0 = (5.0, -45.0)
PROFILE_C1 = (24.0, 45.0)

# M sweep: angular segments and cutter count both grow
M_VALUES = list(range(3, 29))  # 3..28 → 26 steps

RESULTS_PATH = Path(__file__).resolve().parent / "revolve_results.csv"
REGRESSION_FACTOR = 10.0


def segs_for(m: int) -> int:
    return 32 + m * 6


def cutters_for(m: int) -> int:
    return m * 3


def rebuild(m: int):
    segs = segs_for(m)
    n_cut = cutters_for(m)
    solid = revolve_rectangle(
        PROFILE_C0, PROFILE_C1, FRAME, angle_degrees=360.0, segments=segs
    )
    v0, v1 = PROFILE_C0[1] + 3.0, PROFILE_C1[1] - 3.0
    span = v1 - v0
    u_c = 0.5 * (PROFILE_C0[0] + PROFILE_C1[0])
    r = 1.6
    for i in range(n_cut):
        v = v0 + (i + 1) / (n_cut + 1) * span
        cutter = revolve_circle(
            (u_c, v),
            r,
            FRAME,
            angle_degrees=360.0,
            segments=segs,
            profile_segments=max(20, segs // 3),
        )
        solid = boolean_op(solid, cutter, BooleanOp.DIFFERENCE)
    return solid, segs, n_cut


def main() -> int:
    print("=== Revolve performance soak benchmark ===")
    print(f"Profile: {PROFILE_C0}→{PROFILE_C1}")
    print(f"M values: {M_VALUES[0]}..{M_VALUES[-1]} ({len(M_VALUES)} steps)")
    print(f"Results → {RESULTS_PATH}")
    print()

    t0 = time.perf_counter()
    warm, _, _ = rebuild(3)
    print(
        f"Warm-up: {time.perf_counter() - t0:.3f}s  "
        f"tris={len(warm.faces)} watertight={warm.is_watertight()}"
    )

    rows = []
    prev_norm = None
    flags = []

    for m in M_VALUES:
        t0 = time.perf_counter()
        solid, segs, n_cut = rebuild(m)
        wall = time.perf_counter() - t0
        tris = int(len(solid.faces))
        verts = int(len(solid.vertices))
        vol = float(solid.volume())
        tight = bool(solid.is_watertight())
        work = max(1, segs * n_cut)
        tnorm = wall / work
        flag = ""
        if prev_norm is not None and prev_norm > 1e-12:
            ratio = tnorm / prev_norm
            if ratio > REGRESSION_FACTOR:
                flag = "REGRESSION"
                flags.append((m, ratio, wall, tris))
        prev_norm = tnorm
        row = {
            "M": m,
            "segments": segs,
            "cutters": n_cut,
            "wall_s": f"{wall:.4f}",
            "tris": tris,
            "verts": verts,
            "volume": f"{vol:.6g}",
            "watertight": int(tight),
            "s_per_work": f"{tnorm:.8f}",
            "flag": flag,
        }
        rows.append(row)
        mark = f"  *** {flag} ***" if flag else ""
        print(
            f"M={m:3d}  segs={segs:3d}  cutters={n_cut:3d}  wall={wall:8.2f}s  "
            f"tris={tris:8d}  tight={tight}{mark}",
            flush=True,
        )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "M",
                "segments",
                "cutters",
                "wall_s",
                "tris",
                "verts",
                "volume",
                "watertight",
                "s_per_work",
                "flag",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("── Revolve soak summary ──")
    print(f"{'M':>4} {'segs':>5} {'cut':>5} {'wall_s':>10} {'tris':>10} {'tight':>6} flag")
    for r in rows:
        print(
            f"{int(r['M']):4d} {int(r['segments']):5d} {int(r['cutters']):5d} "
            f"{float(r['wall_s']):10.2f} {int(r['tris']):10d} "
            f"{int(r['watertight']):6d} {r['flag']}"
        )
    total = sum(float(r["wall_s"]) for r in rows)
    print(f"\nTotal wall time: {total:.1f}s  steps={len(rows)}")
    print(f"Wrote {RESULTS_PATH}")
    if flags:
        print(f"\nRegression flags ({len(flags)}):")
        for m, ratio, wall, tris in flags:
            print(f"  M={m}: work-normalized ×{ratio:.1f} (wall={wall:.1f}s, tris={tris})")
    else:
        print("\nNo severe per-step regressions flagged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
