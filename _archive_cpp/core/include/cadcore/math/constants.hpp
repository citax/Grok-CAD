#pragma once

#include <cmath>
#include <algorithm>

namespace cad {

/// Epsilon strategy (documented for geometric robustness):
///
/// 1. kAbsEps (1e-12) — absolute tolerance for near-zero tests on
///    dimensionless quantities (dot products after normalization, det, etc.).
/// 2. kRelEps (1e-9)  — relative tolerance: two values a,b are "equal" when
///    |a-b| <= kRelEps * max(1, |a|, |b|). Used for general scalar compare.
/// 3. kGeomEps (1e-9) — absolute geometric length tolerance in model units
///    (assumed metres). Used for coplanarity, point-on-plane, and mesh
///    watertightness vertex welding.
/// 4. kAreaEps / kVolEps — absolute thresholds derived from kGeomEps for
///    discarding degenerate triangles / zero-volume tetrahedra.

inline constexpr double kAbsEps  = 1e-12;
inline constexpr double kRelEps  = 1e-9;
inline constexpr double kGeomEps = 1e-9;
inline constexpr double kPi      = 3.14159265358979323846;
inline constexpr double kTwoPi   = 6.28318530717958647692;
inline constexpr double kDeg2Rad = kPi / 180.0;
inline constexpr double kRad2Deg = 180.0 / kPi;

[[nodiscard]] inline bool near_zero(double x, double eps = kAbsEps) noexcept {
  return std::abs(x) <= eps;
}

[[nodiscard]] inline bool nearly_equal(double a, double b,
                                       double rel = kRelEps,
                                       double abs_eps = kAbsEps) noexcept {
  const double diff = std::abs(a - b);
  if (diff <= abs_eps) {
    return true;
  }
  return diff <= rel * std::max({1.0, std::abs(a), std::abs(b)});
}

[[nodiscard]] inline double clamp(double v, double lo, double hi) noexcept {
  return std::max(lo, std::min(hi, v));
}

}  // namespace cad
