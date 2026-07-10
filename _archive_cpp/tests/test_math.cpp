#include "cadcore/math/constants.hpp"
#include "cadcore/math/mat4.hpp"
#include "cadcore/math/quat.hpp"
#include "cadcore/math/vec3.hpp"

#include <cmath>

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

#include <cstdio>

void test_math() {
  using namespace cad;

  // Vec3 basics
  Vec3 a{1, 2, 3};
  Vec3 b{4, 5, 6};
  CHECK_NEAR(dot(a, b), 32.0, 1e-12);
  const Vec3 c = cross(a, b);
  CHECK_NEAR(c.x, -3.0, 1e-12);
  CHECK_NEAR(c.y, 6.0, 1e-12);
  CHECK_NEAR(c.z, -3.0, 1e-12);
  CHECK_NEAR(a.normalized().length(), 1.0, 1e-12);

  // Mat4 identity and multiply
  Mat4 I = Mat4::identity();
  Vec3 p = I.transform_point({1, 2, 3});
  CHECK_NEAR(p.x, 1, 1e-12);
  CHECK_NEAR(p.y, 2, 1e-12);
  CHECK_NEAR(p.z, 3, 1e-12);

  Mat4 T = Mat4::translation({10, 0, 0});
  p = T.transform_point({1, 2, 3});
  CHECK_NEAR(p.x, 11, 1e-12);

  // Column-major: translation lives in m[12],m[13],m[14]
  CHECK_NEAR(T.m[12], 10.0, 1e-12);
  CHECK_NEAR(T.m[13], 0.0, 1e-12);

  // Inverse of translation
  Mat4 Ti = T.inverted();
  p = Ti.transform_point({11, 2, 3});
  CHECK_NEAR(p.x, 1, 1e-9);
  CHECK_NEAR(p.y, 2, 1e-9);

  // Rotation 90 deg about Z: (1,0,0) -> (0,1,0)
  Mat4 Rz = Mat4::rotation_z(kPi * 0.5);
  Vec3 r = Rz.transform_vector({1, 0, 0});
  CHECK_NEAR(r.x, 0.0, 1e-12);
  CHECK_NEAR(r.y, 1.0, 1e-12);
  CHECK_NEAR(r.z, 0.0, 1e-12);

  // Quaternion 90 deg about Y: (1,0,0) -> (0,0,-1)
  Quat q = Quat::from_axis_angle({0, 1, 0}, kPi * 0.5);
  Vec3 qr = q.rotate({1, 0, 0});
  CHECK_NEAR(qr.x, 0.0, 1e-12);
  CHECK_NEAR(qr.y, 0.0, 1e-12);
  CHECK_NEAR(qr.z, -1.0, 1e-12);

  // Quaternion composition identity
  Quat qi = q * q.inverse();
  CHECK_NEAR(qi.w, 1.0, 1e-12);
  CHECK_NEAR(qi.x, 0.0, 1e-12);

  // Mat4 * Mat4 associativity spot-check
  Mat4 M = T * Rz * Mat4::scale({2, 2, 2});
  Vec3 mp = M.transform_point({1, 0, 0});
  // scale 2 -> (2,0,0), rot Z 90 -> (0,2,0), translate -> (10,2,0)
  CHECK_NEAR(mp.x, 10.0, 1e-9);
  CHECK_NEAR(mp.y, 2.0, 1e-9);
  CHECK_NEAR(mp.z, 0.0, 1e-9);

  CHECK(nearly_equal(1.0, 1.0 + 1e-12));
  CHECK(near_zero(1e-15));
}
