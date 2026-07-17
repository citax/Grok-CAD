"""Chamfered view cube — face/edge/corner pick + drag orbit (Fusion/SW style).

Geometry is a real cube with truncated corners and bevelled edges so corner
hits are distinct faces, not decoration. Cell data ``region`` tags each cell:
  face:+x face:-x face:+y face:-y face:+z face:-z
  edge:±x±y … (12)
  corner:±x±y±z (8)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pyvista as pv

# region id encoding
FACE = 0
EDGE = 1
CORNER = 2


def _sign_tuple(sx: int, sy: int, sz: int) -> str:
    def bit(s, ax):
        if s > 0:
            return f"+{ax}"
        if s < 0:
            return f"-{ax}"
        return ""

    return "".join(bit(s, a) for s, a in ((sx, "x"), (sy, "y"), (sz, "z")))


def build_chamfered_cube(
    *,
    half: float = 1.0,
    chamfer: float = 0.28,
) -> Tuple[pv.PolyData, List[str]]:
    """Return (polydata, region_labels) for a chamfered unit-ish cube.

    ``region_labels[i]`` is e.g. ``face:+z``, ``edge:+x+y``, ``corner:+x+y+z``.
    """
    h = float(half)
    c = float(np.clip(chamfer, 0.12, 0.40)) * h
    # Face plane inset from outer cube so edges/corners have room
    f = h - c  # face square half-extent

    points: List[List[float]] = []
    faces: List[List[int]] = []  # each [n, i0, i1, ...]
    labels: List[str] = []

    def add_poly(pts: List[Tuple[float, float, float]], label: str) -> None:
        base = len(points)
        for p in pts:
            points.append([float(p[0]), float(p[1]), float(p[2])])
        n = len(pts)
        faces.append([n] + [base + i for i in range(n)])
        labels.append(label)

    # --- 6 faces (axis-aligned squares) ---
    # +Z / -Z
    add_poly([(f, f, h), (-f, f, h), (-f, -f, h), (f, -f, h)], "face:+z")
    add_poly([(f, f, -h), (f, -f, -h), (-f, -f, -h), (-f, f, -h)], "face:-z")
    # +Y / -Y
    add_poly([(f, h, f), (f, h, -f), (-f, h, -f), (-f, h, f)], "face:+y")
    add_poly([(f, -h, f), (-f, -h, f), (-f, -h, -f), (f, -h, -f)], "face:-y")
    # +X / -X
    add_poly([(h, f, f), (h, -f, f), (h, -f, -f), (h, f, -f)], "face:+x")
    add_poly([(-h, f, f), (-h, f, -f), (-h, -f, -f), (-h, -f, f)], "face:-x")

    # --- 12 edge bevels (rectangles linking adjacent face edges) ---
    # Edge along Z at +X+Y
    def edge_rect(axis: str, s1: int, s2: int) -> None:
        """Edge between two axes (s1,s2 = ±1), extruded along the free axis."""
        # free axis is the one not in {axis pair}
        if axis == "z":
            # edge parallel to Z between +X+Y etc.
            x, y = s1 * h, s2 * h
            xi, yi = s1 * f, s2 * f
            add_poly(
                [(xi, yi, f), (x, y, f), (x, y, -f), (xi, yi, -f)],
                f"edge:{_sign_tuple(s1, s2, 0)}",
            )
        elif axis == "y":
            x, z = s1 * h, s2 * h
            xi, zi = s1 * f, s2 * f
            add_poly(
                [(xi, f, zi), (xi, -f, zi), (x, -f, z), (x, f, z)],
                f"edge:{_sign_tuple(s1, 0, s2)}",
            )
        else:  # axis == "x" free along X
            y, z = s1 * h, s2 * h
            yi, zi = s1 * f, s2 * f
            add_poly(
                [(f, yi, zi), (-f, yi, zi), (-f, y, z), (f, y, z)],
                f"edge:{_sign_tuple(0, s1, s2)}",
            )

    for sx in (-1, 1):
        for sy in (-1, 1):
            edge_rect("z", sx, sy)
    for sx in (-1, 1):
        for sz in (-1, 1):
            edge_rect("y", sx, sz)
    for sy in (-1, 1):
        for sz in (-1, 1):
            edge_rect("x", sy, sz)

    # --- 8 corner triangles ---
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                # three points on the three adjacent face-edge intersections
                p_xy = (sx * f, sy * f, sz * h)
                p_xz = (sx * f, sy * h, sz * f)
                p_yz = (sx * h, sy * f, sz * f)
                # Winding so outward normal points away from origin
                pts = [p_yz, p_xz, p_xy]
                # Ensure outward
                n = np.cross(
                    np.subtract(pts[1], pts[0]),
                    np.subtract(pts[2], pts[0]),
                )
                center = np.mean(pts, axis=0)
                if np.dot(n, center) < 0:
                    pts = [pts[0], pts[2], pts[1]]
                add_poly(pts, f"corner:{_sign_tuple(sx, sy, sz)}")

    # Build PolyData
    pts_arr = np.asarray(points, dtype=np.float64)
    # flatten faces connectivity
    conn: List[int] = []
    for fr in faces:
        conn.extend(fr)
    poly = pv.PolyData(pts_arr, faces=np.asarray(conn, dtype=np.int64))
    # cell scalars for colouring: face=0 edge=1 corner=2
    kinds = np.zeros(len(labels), dtype=np.int32)
    for i, lab in enumerate(labels):
        if lab.startswith("edge"):
            kinds[i] = 1
        elif lab.startswith("corner"):
            kinds[i] = 2
    poly.cell_data["kind"] = kinds
    poly.cell_data["region_id"] = np.arange(len(labels), dtype=np.int32)
    return poly, labels


def region_to_view(label: str) -> Optional[str]:
    """Map a region label to a set_view name or iso variant."""
    if not label:
        return None
    if label.startswith("face:"):
        key = label.split(":", 1)[1]
        return {
            "+x": "right",
            "-x": "left",
            "+y": "top",
            "-y": "bottom",
            "+z": "front",
            "-z": "back",
        }.get(key)
    if label.startswith("corner:"):
        # All corners → isometric from that octant
        return "iso:" + label.split(":", 1)[1]
    if label.startswith("edge:"):
        # Edge → look from midpoint of the two face normals (half-iso)
        return "edge:" + label.split(":", 1)[1]
    return None


def color_for_region(label: str) -> Tuple[float, float, float]:
    """RGB 0..1 — pale readable faces (Fusion/SW style), not saturated dark.

    Lighting is off on the cube actor; these are final display colours.
    """
    # Pale face fills with a hint of axis colour so +X vs +Y is obvious
    if label == "face:+x":
        return (0.98, 0.82, 0.82)  # pale red  (Right)
    if label == "face:-x":
        return (0.95, 0.88, 0.88)  # softer red (Left)
    if label == "face:+y":
        return (0.82, 0.95, 0.84)  # pale green (Top)
    if label == "face:-y":
        return (0.88, 0.95, 0.90)  # softer green (Bottom)
    if label == "face:+z":
        return (0.82, 0.88, 0.98)  # pale blue (Front)
    if label == "face:-z":
        return (0.88, 0.90, 0.96)  # softer blue (Back)
    if label.startswith("corner"):
        return (0.97, 0.97, 0.98)  # near-white chamfer
    # edges — light grey bevels
    return (0.90, 0.91, 0.93)


def face_label_text(label: str) -> Optional[str]:
    """Human-readable text drawn on a face (SolidWorks-style)."""
    return {
        "face:+x": "Right",
        "face:-x": "Left",
        "face:+y": "Top",
        "face:-y": "Bottom",
        "face:+z": "Front",
        "face:-z": "Back",
    }.get(label)


def face_label_position(label: str, half: float = 1.0) -> Optional[Tuple[float, float, float]]:
    """World position slightly outside the face centre for a text label."""
    o = float(half) * 1.02
    return {
        "face:+x": (o, 0.0, 0.0),
        "face:-x": (-o, 0.0, 0.0),
        "face:+y": (0.0, o, 0.0),
        "face:-y": (0.0, -o, 0.0),
        "face:+z": (0.0, 0.0, o),
        "face:-z": (0.0, 0.0, -o),
    }.get(label)
