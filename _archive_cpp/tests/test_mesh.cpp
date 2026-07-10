#include "cadcore/mesh/mesh.hpp"
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

void test_mesh() {
  using namespace cad;
  Mesh m = make_box(2, 2, 2);
  CHECK(m.triangle_count() == 12);
  m.compute_normals();
  CHECK(m.normals.size() == m.positions.size());
  auto [mn, mx] = m.bounds();
  CHECK_NEAR(mn.x, -1, 1e-12);
  CHECK_NEAR(mx.x, 1, 1e-12);
}
