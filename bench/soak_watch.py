#!/usr/bin/env python3
"""Live soak / leak watch on the rebuild pipeline.

In a loop, builds and tears down random feature trees (extrude + revolve +
boolean cuts), verifies watertightness, and every ~3 seconds appends a sample
to bench/soak.log: elapsed, iteration, process RSS (MB), last rebuild time.

Runs for at least ~6 minutes unless stopped. Headless — no GUI.
"""

from __future__ import annotations

import gc
import os
import random
import resource
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cadcore.document import Document, Feature, FeatureType  # noqa: E402
from cadcore.mesh import (  # noqa: E402
    BooleanOp,
    boolean_op,
    extrude_rectangle,
    revolve_rectangle,
)
from cadcore.sketch import PlaneFrame  # noqa: E402

LOG_PATH = Path(__file__).resolve().parent / "soak.log"
DURATION_S = float(os.environ.get("SOAK_DURATION_S", "360"))  # 6 minutes
SAMPLE_INTERVAL_S = 3.0
FRAME = PlaneFrame.from_plane_type("PLANE_FRONT")


def rss_mb() -> float:
    """Resident set size in MiB (Linux: ru_maxrss is KiB)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # Linux: kilobytes; macOS: bytes — detect via typical magnitude
    rss = float(usage.ru_maxrss)
    if rss > 10_000_000:  # macOS bytes
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0  # Linux KiB → MiB


def proc_rss_mb() -> float:
    """Current RSS from /proc if available (better than maxrss for trends)."""
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    # kB
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return rss_mb()


def random_rebuild(rng: random.Random) -> tuple[float, int, float]:
    """Build a random solid tree, verify watertight; return (dt, n_tris, volume)."""
    t0 = time.perf_counter()
    # Mix of extrude plate + revolve boss + boolean difference holes
    w = rng.uniform(4.0, 12.0)
    h = rng.uniform(4.0, 12.0)
    d = rng.uniform(0.8, 3.0)
    plate = extrude_rectangle((-w * 0.5, -h * 0.5), (w * 0.5, h * 0.5), d, FRAME)

    # Optional revolve torus-like boss unioned
    if rng.random() < 0.7:
        u0 = rng.uniform(0.8, 2.5)
        du = rng.uniform(0.3, 0.9)
        v0 = rng.uniform(-1.5, 0.0)
        dv = rng.uniform(0.4, 1.5)
        segs = rng.choice([16, 24, 32, 48])
        boss = revolve_rectangle(
            (u0, v0), (u0 + du, v0 + dv), FRAME, angle_degrees=360.0, segments=segs
        )
        # Place boss near plate
        boss = boss.translate(
            (rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(0.0, d * 0.5))
        )
        solid = boolean_op(plate, boss, BooleanOp.UNION)
    else:
        solid = plate

    # Boolean cuts: 1..N cylindrical-ish revolved or extruded holes
    n_cuts = rng.randint(1, 6)
    for _ in range(n_cuts):
        if rng.random() < 0.5:
            # extruded box cutter through plate
            cx = rng.uniform(-w * 0.3, w * 0.3)
            cy = rng.uniform(-h * 0.3, h * 0.3)
            s = rng.uniform(0.3, 1.2)
            cutter = extrude_rectangle(
                (cx - s, cy - s), (cx + s, cy + s), d * 2.5, FRAME
            )
            cutter = cutter.translate((0.0, 0.0, -0.5))
        else:
            u0 = rng.uniform(1.0, 3.0)
            segs = rng.choice([12, 16, 24, 32])
            cutter = revolve_rectangle(
                (u0, -0.4),
                (u0 + 0.5, 0.4),
                FRAME,
                angle_degrees=360.0,
                segments=segs,
            )
            cutter = cutter.translate(
                (
                    rng.uniform(-w * 0.25, w * 0.25),
                    rng.uniform(-h * 0.25, h * 0.25),
                    rng.uniform(-0.5, 0.5),
                )
            )
        try:
            solid = boolean_op(solid, cutter, BooleanOp.DIFFERENCE)
        except Exception:
            # occasional non-manifold edge cases — skip this cut
            continue

    dt = time.perf_counter() - t0
    if solid.empty or not solid.is_watertight():
        raise RuntimeError("rebuild produced non-watertight solid")
    return dt, int(len(solid.faces)), float(solid.volume())


def append_sample(
    elapsed: float,
    iteration: int,
    rss: float,
    last_rebuild: float,
    note: str = "",
) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    # Compact key=value tokens (no spaces around '=') for easy parsing
    line = (
        f"{ts} elapsed_s={elapsed:.1f} iter={iteration} "
        f"rss_mb={rss:.2f} last_rebuild_s={last_rebuild:.4f}"
    )
    if note:
        line += f"  {note}"
    line += "\n"
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
    print(line, end="", flush=True)


def main() -> int:
    # Truncate log for this run
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("w", encoding="utf-8") as fh:
        fh.write(
            f"# soak_watch start pid={os.getpid()} duration_s={DURATION_S}\n"
        )

    print(
        f"=== Soak / leak watch (pid={os.getpid()}) ===\n"
        f"Duration ≥ {DURATION_S:.0f}s  sample every {SAMPLE_INTERVAL_S:.0f}s\n"
        f"Log → {LOG_PATH}\n",
        flush=True,
    )

    rng = random.Random(0xC0FFEE)
    t_start = time.perf_counter()
    last_sample = t_start
    iteration = 0
    last_rebuild = 0.0
    samples: list[tuple[float, float]] = []  # (elapsed, rss)

    # Initial sample
    rss0 = proc_rss_mb()
    append_sample(0.0, 0, rss0, 0.0, note="start")
    samples.append((0.0, rss0))

    try:
        while True:
            elapsed = time.perf_counter() - t_start
            if elapsed >= DURATION_S:
                break
            try:
                last_rebuild, _tris, _vol = random_rebuild(rng)
            except Exception as exc:  # noqa: BLE001
                last_rebuild = 0.0
                # still count iteration; log occasionally
                if iteration % 20 == 0:
                    print(f"[soak] rebuild warning: {exc}", flush=True)
            iteration += 1
            # Drop refs / encourage GC between trees
            gc.collect()

            now = time.perf_counter()
            if now - last_sample >= SAMPLE_INTERVAL_S:
                last_sample = now
                elapsed = now - t_start
                rss = proc_rss_mb()
                append_sample(elapsed, iteration, rss, last_rebuild)
                samples.append((elapsed, rss))

                # Early leak abort: >25% growth sustained (compare last 3 avg vs first 3)
                if len(samples) >= 8:
                    early = sum(s[1] for s in samples[:3]) / 3.0
                    late = sum(s[1] for s in samples[-3:]) / 3.0
                    if early > 1.0 and late > early * 1.25:
                        append_sample(
                            elapsed,
                            iteration,
                            rss,
                            last_rebuild,
                            note=f"LEAK_SUSPECTED early_avg={early:.2f} late_avg={late:.2f}",
                        )
                        print(
                            f"\n*** Suspected leak: RSS {early:.1f} → {late:.1f} MiB "
                            f"(>{25}% growth). Stopping early.\n",
                            flush=True,
                        )
                        return 2
    except KeyboardInterrupt:
        print("\n[soak] interrupted", flush=True)

    elapsed = time.perf_counter() - t_start
    rss = proc_rss_mb()
    append_sample(elapsed, iteration, rss, last_rebuild, note="end")
    samples.append((elapsed, rss))

    first_rss = samples[0][1]
    last_rss = samples[-1][1]
    # Simple linear slope (MiB per minute)
    if elapsed > 1e-6:
        slope = (last_rss - first_rss) / (elapsed / 60.0)
    else:
        slope = 0.0
    growth = (last_rss / first_rss - 1.0) * 100.0 if first_rss > 1e-6 else 0.0
    print(
        f"\n── Soak summary ──\n"
        f"iterations={iteration}  samples={len(samples)}  elapsed={elapsed:.1f}s\n"
        f"RSS first={first_rss:.2f} MiB  last={last_rss:.2f} MiB  "
        f"growth={growth:+.1f}%  slope={slope:+.3f} MiB/min\n"
        f"verdict={'LEAK_SUSPECTED' if growth > 25 else 'FLAT/OK'}\n"
        f"log={LOG_PATH}\n",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
