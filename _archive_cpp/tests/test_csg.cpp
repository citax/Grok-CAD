#include "cadcore/mesh/csg.hpp"
#include "cadcore/mesh/primitives.hpp"
#include "cadcore/mesh/transform.hpp"

#include <cmath>
#include <cstdio>

extern int g_failures;
extern int g_tests;
#define CHECK(cond)                                                          \
  do {                                                                       \
    ++g_tests;                                                               \
    if (!(cond)) {                                                           \
      ++g_failures;                                                          \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);   \
    }                                                                        \
  } while (0)
#define CHECK_NEAR(a, b, eps)                                                \
  do {                                                                       \
    ++g_tests;                                                               \
    const double _a = (a);                                                   \
    const double _b = (b);                                                   \
    if (!(std::abs(_a - _b) <= (eps))) {                                     \
      ++g_failures;                                                          \
      std::fprintf(stderr, "FAIL %s:%d: |%g - %g| > %g\n", __FILE__,         \
                   __LINE__, _a, _b, static_cast<double>(eps));              \
    }                                                                        \
  } while (0)

static void check_boolean_result(const char* label, const cad::Mesh& m,
                                 double min_vol, double max_vol) {
  (void)label;
  CHECK(!m.empty());
  CHECK(m.triangle_count() > 0);
  CHECK(m.is_watertight());
  const double v = m.volume();
  CHECK(v > min_vol);
  CHECK(v < max_vol);
}

void test_csg() {
  using namespace cad;

  // --- Case 1: two overlapping axis-aligned boxes ---
  // Box 2x2x2 at origin (vol=8) and translated +1 on X (vol=8), overlap vol=4
  Mesh a = make_box(2, 2, 2);
  Mesh b = transformed(make_box(2, 2, 2), Mat4::translation({1, 0, 0}));
  CHECK(a.is_watertight());
  CHECK(b.is_watertight());

  Mesh u = boolean_union(a, b);
  // union vol = 8+8-4 = 12
  check_boolean_result("box-union", u, 11.0, 13.0);
  CHECK(u.is_watertight());
  CHECK_NEAR(u.volume(), 12.0, 0.25);

  Mesh d = boolean_difference(a, b);
  // difference vol = 8-4 = 4
  check_boolean_result("box-diff", d, 3.0, 5.0);
  CHECK(d.is_watertight());
  CHECK(d.volume() < a.volume() + 0.1);

  Mesh i = boolean_intersection(a, b);
  // intersection vol = 4
  check_boolean_result("box-inter", i, 3.0, 5.0);
  CHECK(i.is_watertight());
  CHECK(i.volume() < a.volume());

  // --- Case 2: two offset spheres (union / difference / intersection) ---
  Mesh s1 = make_sphere(1.0, 16, 8);
  Mesh s2 = transformed(make_sphere(1.0, 16, 8), Mat4::translation({0.8, 0, 0}));
  CHECK(s1.is_watertight());
  CHECK(s2.is_watertight());

  Mesh su = boolean_union(s1, s2);
  CHECK(su.is_watertight());
  CHECK(su.volume() > s1.volume());
  CHECK(su.volume() < s1.volume() + s2.volume() + 0.1);

  Mesh sd = boolean_difference(s1, s2);
  CHECK(sd.is_watertight());
  CHECK(sd.volume() > 0.0);
  CHECK(sd.volume() < s1.volume());

  Mesh si = boolean_intersection(s1, s2);
  CHECK(si.is_watertight());
  CHECK(si.volume() > 0.0);
  CHECK(si.volume() < s1.volume());

  // --- Case 3: sphere minus cylinder ---
  Mesh sph = make_sphere(1.0, 16, 8);
  Mesh cyl = make_cylinder(0.4, 2.5, 16);
  CHECK(sph.is_watertight());
  CHECK(cyl.is_watertight());

  Mesh sph_minus_cyl = boolean_difference(sph, cyl);
  CHECK(sph_minus_cyl.is_watertight());
  CHECK(sph_minus_cyl.volume() > 0.0);
  CHECK(sph_minus_cyl.volume() < sph.volume());

  // --- Case 4: box with cylindrical hole (difference) ---
  Mesh box = make_box(2, 2, 2);
  Mesh hole = make_cylinder(0.35, 2.5, 16);  // pierces the box along Y
  CHECK(box.is_watertight());
  CHECK(hole.is_watertight());

  Mesh box_hole = boolean_difference(box, hole);
  CHECK(box_hole.is_watertight());
  CHECK(box_hole.volume() > 0.0);
  CHECK(box_hole.volume() < box.volume());

  // Hole should also work as intersection emptiness check: hole ∩ exterior-ish
  Mesh box_and_hole = boolean_intersection(box, hole);
  CHECK(box_and_hole.is_watertight());
  CHECK(box_and_hole.volume() > 0.0);
  CHECK(box_and_hole.volume() < box.volume());
}
