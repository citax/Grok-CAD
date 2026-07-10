"""Unit tests for binary STL export/import — cadcore only, no GUI imports."""

from pathlib import Path

import numpy as np
import pytest

from cadcore.document import Document, FeatureType
from cadcore.mesh import (
    Mesh,
    extrude_rectangle,
    make_box,
    read_stl_binary,
    revolve_rectangle,
    write_stl_binary,
)
from cadcore.sketch import PlaneFrame


def test_stl_roundtrip_extruded_rectangle(tmp_path: Path):
    frame = PlaneFrame.from_plane_type("PLANE_FRONT")
    mesh = extrude_rectangle((0.0, 0.0), (2.0, 3.0), 4.0, frame)
    assert mesh.is_watertight()
    path = tmp_path / "extrude.stl"
    write_stl_binary(mesh, path)
    assert path.is_file() and path.stat().st_size > 84

    loaded = read_stl_binary(path)
    assert len(loaded.faces) == len(mesh.faces)
    assert loaded.is_watertight()
    assert abs(loaded.volume() - mesh.volume()) / max(abs(mesh.volume()), 1e-9) < 1e-3


def test_stl_roundtrip_revolved_profile(tmp_path: Path):
    frame = PlaneFrame.from_plane_type("PLANE_FRONT")
    mesh = revolve_rectangle(
        (1.0, 0.0),
        (2.0, 1.0),
        frame,
        angle_degrees=360.0,
        segments=48,
    )
    assert mesh.is_watertight()
    path = tmp_path / "revolve.stl"
    write_stl_binary(mesh, path)
    loaded = read_stl_binary(path)
    assert len(loaded.faces) == len(mesh.faces)
    assert loaded.is_watertight()
    assert abs(loaded.volume() - mesh.volume()) / max(abs(mesh.volume()), 1e-9) < 0.02


def test_stl_roundtrip_box(tmp_path: Path):
    mesh = make_box(2, 3, 4)
    path = tmp_path / "box.stl"
    write_stl_binary(mesh, path)
    loaded = read_stl_binary(path)
    assert len(loaded.faces) == len(mesh.faces)
    assert abs(loaded.volume() - 24.0) < 1e-3


def test_stl_rejects_empty_mesh(tmp_path: Path):
    empty = Mesh()
    with pytest.raises(ValueError, match="empty"):
        write_stl_binary(empty, tmp_path / "empty.stl")


def test_stl_rejects_open_non_watertight(tmp_path: Path):
    # Single triangle — open surface, not a solid
    open_mesh = Mesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64),
        faces=np.array([[0, 1, 2]], dtype=np.int32),
    )
    assert not open_mesh.is_watertight()
    with pytest.raises(ValueError, match="watertight|non-solid|open"):
        write_stl_binary(open_mesh, tmp_path / "open.stl")


def test_document_export_extrude_via_mesh(tmp_path: Path):
    doc = Document()
    doc.seed_reference_planes()
    plane = next(f for f in doc.features if f.type is FeatureType.PLANE_FRONT)
    skf = doc.create_sketch_on_plane(plane.id)
    assert skf and skf.sketch
    skf.sketch.add_rectangle((0, 0), (1.5, 2.0))
    ex = doc.create_extrude(skf.id, 2.5)
    mesh = doc.evaluate_feature(ex.id)
    assert mesh is not None and mesh.is_watertight()
    path = tmp_path / "doc_extrude.stl"
    write_stl_binary(mesh, path)
    loaded = read_stl_binary(path)
    assert len(loaded.faces) == len(mesh.faces)
    assert abs(loaded.volume() - mesh.volume()) < 1e-3


def test_no_gui_imports_in_stl_path():
    import cadcore.mesh as m

    src = open(m.__file__, encoding="utf-8").read()
    assert "PySide" not in src
    assert "pyvista" not in src
    assert "PySide6" not in src
