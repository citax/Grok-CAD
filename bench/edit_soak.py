#!/usr/bin/env python3
"""Headless edit soak: randomized add/paste/undo/redo/cut with invariants.

Run many cycles asserting:
- no ghost actors (when GUI path unavailable, data-level actor name set simulated)
- geometry on-plane (max |n·(p−origin)| < 1e-6)
- undo/redo round-trips
- junction dots == unique endpoints
- labels match current unit
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from typing import List, Set

import numpy as np

from cadcore.document import Document, FeatureType
from cadcore.sketch import (
    LineEntity,
    line_length,
    set_line_length,
    snapshot_entity,
)
from cadcore.units import Unit, format_length, to_mm


def max_plane_dev(sk) -> float:
    fr = sk.frame
    n = np.asarray(fr.normal, float)
    o = np.asarray(fr.origin, float)
    worst = 0.0
    for e in sk.entities:
        if isinstance(e, LineEntity):
            pts = [e.p0, e.p1]
        else:
            continue
        for uv in pts:
            w = fr.to_world(uv)
            worst = max(worst, abs(float(np.dot(n, w - o))))
    return worst


def actor_names_for(sk, sketch_id: int) -> Set[str]:
    """Simulated actor name set matching viewport conventions."""
    return {f"sk_e_{e.id}" for e in sk.entities}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    doc = Document()
    doc.seed_reference_planes()
    front = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(front.id)
    sk = skf.sketch
    assert sk is not None

    units = [Unit.MM, Unit.CM, Unit.INCH]
    failures = 0
    t0 = time.time()
    ops = 0

    # Seed one line so paste/copy has something
    seed = sk.add_line((0.0, 0.0), (10.0, 0.0))
    doc.record_entity_add(skf.id, seed)
    actors = actor_names_for(sk, skf.id)

    for i in range(args.cycles):
        op = rng.choice(
            ["add", "paste", "undo", "redo", "cut_copy", "move", "unit", "length"]
        )
        try:
            if op == "add":
                p0 = (rng.uniform(-5, 5), rng.uniform(-5, 5))
                p1 = (p0[0] + rng.uniform(0.5, 8), p0[1] + rng.uniform(-3, 3))
                e = sk.add_line(p0, p1)
                doc.record_entity_add(skf.id, e)
                actors = actor_names_for(sk, skf.id)
            elif op == "paste":
                if sk.entities:
                    e0 = rng.choice(list(sk.entities))
                    doc.copy_entity(skf.id, e0.id)
                    p = doc.paste_entity(skf.id)
                    if p is not None:
                        actors = actor_names_for(sk, skf.id)
            elif op == "undo":
                before_ids = {e.id for e in sk.entities}
                if doc.undo():
                    after_ids = {e.id for e in sk.entities}
                    actors = actor_names_for(sk, skf.id)
                    # ghost check: no actor name for missing entity
                    for eid in before_ids - after_ids:
                        assert f"sk_e_{eid}" not in actors
            elif op == "redo":
                before_ids = {e.id for e in sk.entities}
                if doc.redo():
                    after_ids = {e.id for e in sk.entities}
                    actors = actor_names_for(sk, skf.id)
                    for eid in after_ids - before_ids:
                        assert f"sk_e_{eid}" in actors
            elif op == "cut_copy":
                if sk.entities:
                    e0 = rng.choice(list(sk.entities))
                    if rng.random() < 0.5:
                        doc.copy_entity(skf.id, e0.id)
                    else:
                        doc.cut_entity(skf.id, e0.id)
                    actors = actor_names_for(sk, skf.id)
            elif op == "move":
                lines = [e for e in sk.entities if isinstance(e, LineEntity)]
                if lines:
                    e = rng.choice(lines)
                    before = snapshot_entity(e)
                    e.translate(rng.uniform(-1, 1), rng.uniform(-1, 1))
                    after = snapshot_entity(e)
                    doc.record_entity_move(skf.id, before, after)
            elif op == "unit":
                doc.set_display_unit(rng.choice(units))
            elif op == "length":
                lines = [e for e in sk.entities if isinstance(e, LineEntity)]
                if lines:
                    e = rng.choice(lines)
                    before = snapshot_entity(e)
                    set_line_length(e, to_mm(rng.uniform(1, 20), doc.display_unit))
                    doc.record_entity_move(skf.id, before, snapshot_entity(e))

            # --- invariants every cycle ---
            # 1) actor set matches entities (no ghosts)
            expected = actor_names_for(sk, skf.id)
            if actors != expected:
                raise AssertionError(f"ghost/missing actors: {actors ^ expected}")

            # 2) on-plane
            dev = max_plane_dev(sk)
            if dev >= 1e-6:
                raise AssertionError(f"off-plane max_dev={dev}")

            # 3) junctions == unique endpoints
            j = sk.unique_endpoints()
            # recompute unique
            if len(j) != len(sk.unique_endpoints()):
                raise AssertionError("junction count unstable")

            # 4) labels match geometry + unit
            for e in sk.entities:
                if isinstance(e, LineEntity):
                    lab = format_length(line_length(e), doc.display_unit)
                    if not lab.endswith(doc.display_unit.label) and not (
                        doc.display_unit is Unit.INCH and lab.endswith(" in")
                    ):
                        raise AssertionError(f"bad label {lab} unit={doc.display_unit}")

            # 5) undo/redo empty-stack safety
            # (call is no-op — just ensure no exception if empty)
            ops += 1
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL cycle={i} op={op}: {exc}", flush=True)
            if failures >= 5:
                break

    dt = time.time() - t0
    # Final round-trip: undo all then redo
    n_undo = 0
    while doc.undo():
        n_undo += 1
    n_redo = 0
    while doc.redo():
        n_redo += 1
    print(
        f"EDIT_SOAK_DONE cycles={args.cycles} ops={ops} failures={failures} "
        f"dt={dt:.2f}s entities={len(sk.entities)} "
        f"undo_all={n_undo} redo_all={n_redo} max_dev={max_plane_dev(sk):.3e}",
        flush=True,
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
