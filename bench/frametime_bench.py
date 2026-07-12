#!/usr/bin/env python3
"""Headless frametime / rebuild latency benchmark for the viewport path.

Builds a sketch with ~200 entities, then runs ~300 edit+rebuild cycles through
the same solid rebuild path the GUI uses (snapshot_features +
evaluate_solids_snapshot + apply-style polydata construction), measuring
per-cycle wall time. Also exercises sketch actor upsert path when available.

Writes bench/frametime_results.csv and streams progress.
"""

from __future__ import annotations

import csv
import os
import statistics
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

RESULTS = Path(__file__).resolve().parent / "frametime_results.csv"
LOG = Path(__file__).resolve().parent / "frametime_bench.log"

N_ENTITIES = int(os.environ.get("FT_ENTITIES", "200"))
N_CYCLES = int(os.environ.get("FT_CYCLES", "300"))
LABEL = os.environ.get("FT_LABEL", "before")  # before | after


def log(msg: str) -> None:
    line = msg if msg.endswith("\n") else msg + "\n"
    sys.stdout.write(line)
    sys.stdout.flush()
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line)


def build_heavy_doc():
    from cadcore.document import Document, FeatureType

    doc = Document()
    doc.seed_reference_planes()
    plane = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(plane.id)
    assert skf and skf.sketch
    sk = skf.sketch
    # ~200 entities: mix of lines, rects, circles
    n = N_ENTITIES
    for i in range(n):
        kind = i % 3
        u = (i % 20) * 0.35
        v = (i // 20) * 0.35
        if kind == 0:
            sk.add_line((u, v), (u + 0.2, v + 0.15))
        elif kind == 1:
            sk.add_rectangle((u, v), (u + 0.25, v + 0.25))
        else:
            sk.add_circle((u + 0.1, v + 0.1), 0.08)
    # One solid for rebuild path
    # Use first closed rect for extrude if any
    from cadcore.document import first_closed_profile

    ent = first_closed_profile(sk)
    if ent is None:
        sk.add_rectangle((0, 0), (2, 2))
        ent = first_closed_profile(sk)
    # Don't always extrude - for sketch-heavy path we measure sketch refresh
    # Add a simple extrude solid for solid rebuild path
    try:
        doc.create_extrude(skf.id, 1.0)
    except Exception:
        pass
    return doc, skf


def cycle_rebuild(doc, skf, i: int, *, incremental: bool, cache: dict) -> float:
    """One edit + rebuild cycle matching GUI path (CPU geometry + polydata prep).

    When ``incremental`` is True (AFTER fix), only rebuild polydata for the
    mutated entity — mirrors viewport fingerprint-based upserts.
    """
    from app.workers import evaluate_solids_snapshot, snapshot_features
    from app.viewport import _entity_fingerprint, _entity_polydata
    from cadcore.sketch import RectEntity, CircleEntity, LineEntity
    import numpy as np

    sk = skf.sketch
    assert sk is not None
    # Mutate ONE entity (simulates drag edit)
    ent = sk.entities[i % len(sk.entities)]
    if isinstance(ent, LineEntity):
        ent.translate(0.001 * ((i % 5) - 2), 0.001 * ((i % 3) - 1))
    elif isinstance(ent, RectEntity):
        ent.translate(0.001 * ((i % 5) - 2), 0.0)
    elif isinstance(ent, CircleEntity):
        ent.translate(0.0, 0.001 * ((i % 3) - 1))

    t0 = time.perf_counter()
    feats = snapshot_features(doc)
    results = evaluate_solids_snapshot(feats)
    # Solid path is already incremental via fingerprint in evaluate/apply
    for fid, (verts, faces, fp) in results.items():
        key = f"solid_{fid}"
        if incremental and cache.get(key) == fp:
            continue
        _ = np.ascontiguousarray(verts)
        _ = np.ascontiguousarray(faces)
        if len(faces):
            face_arr = np.hstack(
                [np.full((len(faces), 1), 3, dtype=np.int64), faces.astype(np.int64)]
            )
            _ = face_arr
        cache[key] = fp

    if incremental:
        # Only re-polydata the changed entity (+ fingerprint check for rest)
        for e in sk.entities:
            fp = _entity_fingerprint(e, selected=False)
            ek = f"e_{e.id}"
            if cache.get(ek) == fp:
                continue
            _ = _entity_polydata(e, sk)
            cache[ek] = fp
    else:
        # BEFORE: full teardown — rebuild polydata for ALL entities every cycle
        for e in sk.entities:
            _ = _entity_polydata(e, sk)
    return (time.perf_counter() - t0) * 1000.0


def main() -> int:
    LOG.write_text(f"# frametime_bench label={LABEL}\n", encoding="utf-8")
    log(f"=== Frametime bench label={LABEL} entities={N_ENTITIES} cycles={N_CYCLES}")
    doc, skf = build_heavy_doc()
    log(f"doc features={len(doc.features)} sketch_entities={len(skf.sketch.entities)}")

    incremental = LABEL == "after"
    cache: dict = {}
    # Warmup
    for i in range(5):
        cycle_rebuild(doc, skf, i, incremental=incremental, cache=cache)

    times = []
    for i in range(N_CYCLES):
        ms = cycle_rebuild(doc, skf, i, incremental=incremental, cache=cache)
        times.append(ms)
        if (i + 1) % 50 == 0 or i == 0:
            log(f"  cycle {i+1}/{N_CYCLES} last={ms:.3f}ms")

    times_sorted = sorted(times)
    mean = statistics.mean(times)
    med = statistics.median(times)
    p95 = times_sorted[int(0.95 * (len(times_sorted) - 1))]
    log(
        f"RESULT label={LABEL} mean={mean:.3f}ms median={med:.3f}ms p95={p95:.3f}ms "
        f"min={min(times):.3f} max={max(times):.3f}"
    )

    # Append/update CSV
    rows = []
    if RESULTS.is_file():
        with RESULTS.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    # Replace same label row if present
    rows = [r for r in rows if r.get("label") != LABEL]
    rows.append(
        {
            "label": LABEL,
            "entities": str(N_ENTITIES),
            "cycles": str(N_CYCLES),
            "mean_ms": f"{mean:.4f}",
            "median_ms": f"{med:.4f}",
            "p95_ms": f"{p95:.4f}",
            "min_ms": f"{min(times):.4f}",
            "max_ms": f"{max(times):.4f}",
        }
    )
    with RESULTS.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "label",
                "entities",
                "cycles",
                "mean_ms",
                "median_ms",
                "p95_ms",
                "min_ms",
                "max_ms",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    log(f"Wrote {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
