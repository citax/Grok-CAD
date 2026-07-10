#include "cadcore/document/document.hpp"
#include "cadcore/document/json_io.hpp"

#include <cmath>
#include <cstdio>
#include <string>

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

void test_document() {
  using namespace cad;
  Document doc;
  doc.seed_reference_planes();
  CHECK(doc.features().size() == 3);
  CHECK(is_reference_plane(doc.features()[0].type));
  CHECK(!doc.remove_feature(doc.features()[0].id));  // planes protected

  Feature box;
  box.type = FeatureType::Box;
  box.width = 2;
  box.height = 2;
  box.depth = 2;
  const int id = doc.add_feature(box);
  CHECK(id > 0);

  auto mesh = doc.evaluate_feature(id);
  CHECK(mesh.has_value());
  CHECK_NEAR(mesh->volume(), 8.0, 1e-9);

  // Parametric edit
  Feature* f = doc.find(id);
  CHECK(f != nullptr);
  f->width = 4;
  mesh = doc.evaluate_feature(id);
  CHECK_NEAR(mesh->volume(), 16.0, 1e-9);

  // JSON round-trip
  const std::string json = document_to_json(doc);
  Document loaded;
  std::string err;
  CHECK(document_from_json(json, loaded, &err));
  // document has 3 planes + 1 box
  CHECK(loaded.features().size() == 4);
  const Feature* boxf = nullptr;
  for (const auto& lf : loaded.features()) {
    if (lf.type == FeatureType::Box) { boxf = &lf; break; }
  }
  CHECK(boxf != nullptr);
  CHECK(boxf->width == 4.0);
  auto m2 = loaded.evaluate_feature(boxf->id);
  CHECK(m2.has_value());
  CHECK_NEAR(m2->volume(), 16.0, 1e-9);
}
