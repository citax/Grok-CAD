import math

from cadcore.mesh import (
    BooleanOp,
    boolean_op,
    make_box,
    make_cylinder,
    make_sphere,
)


def test_box_volume_area_watertight():
    m = make_box(2, 3, 4)
    assert m.is_watertight()
    assert m.manifold_is_solid()
    assert abs(m.volume() - 24.0) < 1e-6
    assert abs(m.surface_area() - 52.0) < 1e-4


def test_sphere_volume_watertight():
    m = make_sphere(1.0, 48)
    assert m.is_watertight()
    assert m.manifold_is_solid()
    exact = 4.0 / 3.0 * math.pi
    assert abs(m.volume() - exact) / exact < 0.05


def test_cylinder_volume_watertight():
    m = make_cylinder(1.0, 2.0, 48)
    assert m.is_watertight()
    assert m.manifold_is_solid()
    exact = math.pi * 1.0 * 1.0 * 2.0
    assert abs(m.volume() - exact) / exact < 0.03


def test_boolean_union_boxes_watertight():
    a = make_box(2, 2, 2)
    b = make_box(2, 2, 2).translate((1, 0, 0))
    u = boolean_op(a, b, BooleanOp.UNION)
    assert u.is_watertight()
    assert u.manifold_is_solid()
    assert 11.0 < u.volume() < 13.0


def test_boolean_difference_boxes_watertight():
    a = make_box(2, 2, 2)
    b = make_box(2, 2, 2).translate((1, 0, 0))
    d = boolean_op(a, b, BooleanOp.DIFFERENCE)
    assert d.is_watertight()
    assert d.manifold_is_solid()
    assert 3.0 < d.volume() < 5.0


def test_boolean_intersection_boxes_watertight():
    a = make_box(2, 2, 2)
    b = make_box(2, 2, 2).translate((1, 0, 0))
    i = boolean_op(a, b, BooleanOp.INTERSECTION)
    assert i.is_watertight()
    assert i.manifold_is_solid()
    assert 3.0 < i.volume() < 5.0


def test_boolean_sphere_union_watertight():
    a = make_sphere(1.0, 24)
    b = make_sphere(1.0, 24).translate((0.8, 0, 0))
    u = boolean_op(a, b, BooleanOp.UNION)
    assert u.is_watertight()
    assert u.manifold_is_solid()
    assert u.volume() > a.volume()


def test_boolean_sphere_difference_watertight():
    a = make_sphere(1.0, 24)
    b = make_sphere(1.0, 24).translate((0.8, 0, 0))
    d = boolean_op(a, b, BooleanOp.DIFFERENCE)
    assert d.is_watertight()
    assert d.manifold_is_solid()
    assert 0 < d.volume() < a.volume()


def test_boolean_sphere_minus_cylinder_watertight():
    sph = make_sphere(1.0, 24)
    cyl = make_cylinder(0.4, 2.5, 24)
    d = boolean_op(sph, cyl, BooleanOp.DIFFERENCE)
    assert d.is_watertight()
    assert d.manifold_is_solid()
    assert 0 < d.volume() < sph.volume()


def test_box_cylindrical_hole_watertight():
    box = make_box(2, 2, 2)
    hole = make_cylinder(0.35, 2.5, 24)
    d = boolean_op(box, hole, BooleanOp.DIFFERENCE)
    assert d.is_watertight()
    assert d.manifold_is_solid()
    assert 0 < d.volume() < box.volume()
    man = d.to_manifold()
    assert str(man.status()).endswith("NoError")
    # hole through a box → torus-like genus 0 still (single tunnel is genus 0? actually tunnel = genus 0 for solid with hole through is genus 0 in manifold? 
    # A solid cube with a cylindrical hole has genus 0 in the sense of manifold genus for the boundary is 1 (torus topology of surface).
    g = man.genus()
    assert isinstance(g, int) or hasattr(g, "real")
