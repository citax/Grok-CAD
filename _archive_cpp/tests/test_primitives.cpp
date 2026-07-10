#include "cadcore/math/constants.hpp"
#include "cadcore/mesh/primitives.hpp"

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

void test_primitives() {
  using namespace cad;

  // Box 2x3x4 → volume 24, area 2*(2*3+3*4+4*2)=52
  Mesh box = make_box(2, 3, 4);
  CHECK(box.is_watertight());
  CHECK_NEAR(box.volume(), 24.0, 1e-9);
  CHECK_NEAR(box.surface_area(), 52.0, 1e-9);

  // Unit sphere — volume 4/3 pi r^3, area 4 pi r^2
  // Tessellated sphere will be slightly under
  Mesh sph = make_sphere(1.0, 64, 32);
  CHECK(sph.is_watertight());
  const double vol_exact = 4.0 / 3.0 * kPi;
  const double area_exact = 4.0 * kPi;
  // Allow 3% relative error for tessellation
  CHECK(std::abs(sph.volume() - vol_exact) / vol_exact < 0.03);
  CHECK(std::abs(sph.surface_area() - area_exact) / area_exact < 0.03);

  // Cylinder r=1 h=2 → V=2pi, A=2pi r h + 2 pi r^2 = 4pi + 2pi = 6pi
  Mesh cyl = make_cylinder(1.0, 2.0, 64);
  CHECK(cyl.is_watertight());
  const double cvol = kPi * 1.0 * 1.0 * 2.0;
  const double carea = 2.0 * kPi * 1.0 * 2.0 + 2.0 * kPi * 1.0 * 1.0;
  CHECK(std::abs(cyl.volume() - cvol) / cvol < 0.02);
  CHECK(std::abs(cyl.surface_area() - carea) / carea < 0.02);
}
