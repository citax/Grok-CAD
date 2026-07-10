#!/usr/bin/env python3
"""Re-run extrude_bench and diff fresh numbers against committed baseline.

Reads baseline from bench/results.csv (committed), writes fresh run to
bench/results_fresh.csv, prints per-step deltas, and flags material slowdowns.
"""

from __future__ import annotations

import csv
import subprocess
import sys
import time
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
ROOT = BENCH_DIR.parent
BASELINE = BENCH_DIR / "results.csv"
FRESH = BENCH_DIR / "results_fresh.csv"
EXTRUDE_BENCH = BENCH_DIR / "extrude_bench.py"

# Material slowdown: absolute + relative thresholds
SLOW_ABS_S = 2.0       # at least 2s slower
SLOW_REL = 1.25        # and ≥25% slower than baseline


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    if not BASELINE.is_file():
        print(f"ERROR: baseline missing: {BASELINE}", file=sys.stderr)
        return 2
    if not EXTRUDE_BENCH.is_file():
        print(f"ERROR: extrude bench missing: {EXTRUDE_BENCH}", file=sys.stderr)
        return 2

    print("=== Extrude regression check ===")
    print(f"Baseline: {BASELINE}")
    print(f"Running {EXTRUDE_BENCH.name} (several minutes)…")
    print()

    # Patch RESULTS_PATH by env is not supported — run module and copy by
    # temporarily redirecting: invoke with PYTHONPATH and monkey via cwd.
    # Simplest: run extrude_bench as-is (writes results.csv), then restore baseline.
    import shutil

    backup = BENCH_DIR / "results_baseline_bak.csv"
    shutil.copy2(BASELINE, backup)

    t0 = time.perf_counter()
    env = {**dict(**{k: v for k, v in __import__("os").environ.items()}), "PYTHONPATH": str(ROOT)}
    proc = subprocess.run(
        [sys.executable, str(EXTRUDE_BENCH)],
        cwd=str(ROOT),
        env=env,
        capture_output=False,
    )
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        shutil.copy2(backup, BASELINE)
        print(f"extrude_bench failed rc={proc.returncode}", file=sys.stderr)
        return proc.returncode

    # Fresh results were written over results.csv — move to results_fresh.csv
    if BASELINE.is_file():
        shutil.copy2(BASELINE, FRESH)
    shutil.copy2(backup, BASELINE)  # restore committed baseline
    backup.unlink(missing_ok=True)

    base_rows = load_csv(BASELINE)
    fresh_rows = load_csv(FRESH)
    by_m_base = {int(r["M"]): r for r in base_rows}
    by_m_fresh = {int(r["M"]): r for r in fresh_rows}

    print()
    print("── Extrude regression diff (fresh − baseline) ──")
    print(
        f"{'M':>4} {'holes':>6} {'base_s':>10} {'fresh_s':>10} "
        f"{'Δs':>10} {'Δ%':>8} {'base_tris':>10} {'fresh_tris':>10} flag"
    )
    flags = []
    common = sorted(set(by_m_base) & set(by_m_fresh))
    for m in common:
        b, f = by_m_base[m], by_m_fresh[m]
        bs, fs = float(b["wall_s"]), float(f["wall_s"])
        delta = fs - bs
        pct = (delta / bs * 100.0) if bs > 1e-12 else 0.0
        bt, ft = int(b["tris"]), int(f["tris"])
        flag = ""
        if delta >= SLOW_ABS_S and (fs / bs) >= SLOW_REL:
            flag = "SLOWER"
            flags.append((m, bs, fs, delta, pct))
        print(
            f"{m:4d} {int(f['holes']):6d} {bs:10.2f} {fs:10.2f} "
            f"{delta:10.2f} {pct:7.1f}% {bt:10d} {ft:10d} {flag}"
        )

    missing = sorted(set(by_m_base) - set(by_m_fresh))
    extra = sorted(set(by_m_fresh) - set(by_m_base))
    if missing:
        print(f"\nMissing in fresh run: M={missing}")
    if extra:
        print(f"Extra in fresh run: M={extra}")

    print(f"\nFresh run wall time: {elapsed:.1f}s")
    print(f"Fresh CSV: {FRESH}")
    print(f"Baseline restored: {BASELINE}")
    if flags:
        print(f"\nMaterial slowdowns ({len(flags)}):")
        for m, bs, fs, delta, pct in flags:
            print(f"  M={m}: {bs:.2f}s → {fs:.2f}s (Δ={delta:+.2f}s, {pct:+.1f}%)")
    else:
        print("\nNo material per-step slowdowns vs baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
