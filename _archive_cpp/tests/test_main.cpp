// Minimal test harness (no external deps)
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

int g_failures = 0;
int g_tests = 0;

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

void test_math();
void test_mesh();
void test_primitives();
void test_csg();
void test_stl();
void test_document();

int main() {
  test_math();
  test_mesh();
  test_primitives();
  test_csg();
  test_stl();
  test_document();
  std::printf("%d tests, %d failures\n", g_tests, g_failures);
  return g_failures == 0 ? 0 : 1;
}
