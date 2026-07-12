#!/usr/bin/env python3
"""Live drift watcher — rebuilds a canonical reference part forever.

At startup, builds the CANONICAL REFERENCE PART and records baseline volume
+ triangle count. Every ~15s rebuilds the same part and logs drift vs baseline.

Does NOT self-terminate; runs until killed (Ctrl+C / kill).
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

LOG = Path(__file__).resolve().parent / "drift_watch.log"
INTERVAL_S = float(os.environ.get("DRIFT_INTERVAL_S", "15"))

# Canonical geometry constants (50×50×12)
REF_L = 50.0
REF_H = 12.0
REF_FILLET_R = 2.0
REF_HOLE_R = 8.0  # through-hole (used when pocket path is available)
REF_SEGS = 32


def build_canonical_reference():
    """Build the canonical reference solid (updated as features land).

    Current: pocketed-and-filleted box when pocket APIs exist; otherwise
    filleted box only (startup bootstrap).
    """
    from cadcore.mesh import extrude_filleted_profile
    from cadcore.sketch import EntityKind, PlaneFrame, RectEntity

    frame = PlaneFrame.from_plane_type("PLANE_FRONT")
    half = REF_L * 0.5
    rect = RectEntity(
        id=1,
        kind=EntityKind.RECTANGLE,
        c0=(-half, -half),
        c1=(half, half),
    )

    # Prefer pocketed-and-filleted if kernel supports it
    try:
        from cadcore.mesh import extrude_pocketed_filleted_profile

        return extrude_pocketed_filleted_profile(
            rect,
            REF_H,
            frame,
            fillet_radius=REF_FILLET_R,
            hole_center=(0.0, 0.0),
            hole_radius=REF_HOLE_R,
            segments=REF_SEGS,
        )
    except ImportError:
        return extrude_filleted_profile(
            rect, REF_H, frame, REF_FILLET_R, segments=REF_SEGS
        )


def log(msg: str) -> None:
    line = msg if msg.endswith("\n") else msg + "\n"
    sys.stdout.write(line)
    sys.stdout.flush()
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line)


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("w", encoding="utf-8") as fh:
        fh.write(f"# drift_watch start pid={os.getpid()}\n")

    log(f"=== Drift watch pid={os.getpid()} interval={INTERVAL_S}s ===")
    log("Building CANONICAL REFERENCE PART baseline…")
    t0 = time.perf_counter()
    base = build_canonical_reference()
    dt0 = time.perf_counter() - t0
    base_vol = float(base.volume())
    base_tris = int(len(base.faces))
    base_tight = bool(base.is_watertight())
    log(
        f"baseline volume={base_vol:.8g} tris={base_tris} "
        f"watertight={base_tight} build_s={dt0:.4f}"
    )
    if not base_tight:
        log("ERROR: baseline not watertight — aborting")
        return 1

    iteration = 0
    try:
        while True:
            time.sleep(INTERVAL_S)
            iteration += 1
            t0 = time.perf_counter()
            mesh = build_canonical_reference()
            dt = time.perf_counter() - t0
            vol = float(mesh.volume())
            tris = int(len(mesh.faces))
            tight = bool(mesh.is_watertight())
            drift_vol = vol - base_vol
            drift_tris = tris - base_tris
            rel_vol = abs(drift_vol) / max(abs(base_vol), 1e-30)
            ts = time.strftime("%Y-%m-%dT%H:%M:%S")
            log(
                f"{ts} iter={iteration} volume={vol:.8g} tris={tris} "
                f"drift_vol={drift_vol:+.6e} drift_tris={drift_tris:+d} "
                f"rel_vol={rel_vol:.3e} watertight={int(tight)} rebuild_s={dt:.4f}"
            )
    except KeyboardInterrupt:
        log("drift_watch interrupted")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
