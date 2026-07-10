#include "cadcore/mesh/primitives.hpp"
#include "cadcore/mesh/stl.hpp"

#include <cstdio>
#include <cstring>

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

void test_stl() {
  using namespace cad;
  Mesh m = make_box(1, 2, 3);
  auto a = export_stl_binary(m);
  auto b = export_stl_binary(m);
  CHECK(a.size() == b.size());
  CHECK(a.size() >= 84);
  CHECK(std::memcmp(a.data(), b.data(), a.size()) == 0);

  // Second independent run on a fresh mesh with same params
  Mesh m2 = make_box(1, 2, 3);
  auto c = export_stl_binary(m2);
  CHECK(a.size() == c.size());
  CHECK(std::memcmp(a.data(), c.data(), a.size()) == 0);

  // Round-trip triangle count
  Mesh loaded = import_stl_binary(a);
  CHECK(loaded.triangle_count() == m.triangle_count());
}
