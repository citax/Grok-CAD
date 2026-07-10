#pragma once

#include "cadcore/math/constants.hpp"

#include <array>
#include <cmath>
#include <ostream>

namespace cad {

/// Double-precision 3-vector. Value semantics, no heap allocation.
struct Vec3 {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;

  constexpr Vec3() = default;
  constexpr Vec3(double x_, double y_, double z_) noexcept : x(x_), y(y_), z(z_) {}

  [[nodiscard]] constexpr double operator[](int i) const noexcept {
    return i == 0 ? x : (i == 1 ? y : z);
  }
  [[nodiscard]] constexpr double& operator[](int i) noexcept {
    return i == 0 ? x : (i == 1 ? y : z);
  }

  constexpr Vec3& operator+=(const Vec3& o) noexcept {
    x += o.x;
    y += o.y;
    z += o.z;
    return *this;
  }
  constexpr Vec3& operator-=(const Vec3& o) noexcept {
    x -= o.x;
    y -= o.y;
    z -= o.z;
    return *this;
  }
  constexpr Vec3& operator*=(double s) noexcept {
    x *= s;
    y *= s;
    z *= s;
    return *this;
  }
  constexpr Vec3& operator/=(double s) noexcept {
    x /= s;
    y /= s;
    z /= s;
    return *this;
  }

  [[nodiscard]] friend constexpr Vec3 operator+(Vec3 a, const Vec3& b) noexcept {
    a += b;
    return a;
  }
  [[nodiscard]] friend constexpr Vec3 operator-(Vec3 a, const Vec3& b) noexcept {
    a -= b;
    return a;
  }
  [[nodiscard]] friend constexpr Vec3 operator*(Vec3 a, double s) noexcept {
    a *= s;
    return a;
  }
  [[nodiscard]] friend constexpr Vec3 operator*(double s, Vec3 a) noexcept {
    a *= s;
    return a;
  }
  [[nodiscard]] friend constexpr Vec3 operator/(Vec3 a, double s) noexcept {
    a /= s;
    return a;
  }
  [[nodiscard]] friend constexpr Vec3 operator-(const Vec3& a) noexcept {
    return {-a.x, -a.y, -a.z};
  }

  [[nodiscard]] friend constexpr bool operator==(const Vec3& a, const Vec3& b) noexcept {
    return a.x == b.x && a.y == b.y && a.z == b.z;
  }
  [[nodiscard]] friend constexpr bool operator!=(const Vec3& a, const Vec3& b) noexcept {
    return !(a == b);
  }

  [[nodiscard]] constexpr double length_sq() const noexcept {
    return x * x + y * y + z * z;
  }
  [[nodiscard]] double length() const noexcept { return std::sqrt(length_sq()); }

  [[nodiscard]] Vec3 normalized() const noexcept {
    const double len = length();
    if (near_zero(len)) {
      return {0.0, 0.0, 0.0};
    }
    return *this / len;
  }

  [[nodiscard]] bool is_near(const Vec3& o, double eps = kGeomEps) const noexcept {
    return (x - o.x) * (x - o.x) + (y - o.y) * (y - o.y) + (z - o.z) * (z - o.z) <=
           eps * eps;
  }

  [[nodiscard]] std::array<double, 3> to_array() const noexcept { return {x, y, z}; }
};

[[nodiscard]] inline constexpr double dot(const Vec3& a, const Vec3& b) noexcept {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

[[nodiscard]] inline constexpr Vec3 cross(const Vec3& a, const Vec3& b) noexcept {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

[[nodiscard]] inline double distance(const Vec3& a, const Vec3& b) noexcept {
  return (a - b).length();
}

[[nodiscard]] inline Vec3 lerp(const Vec3& a, const Vec3& b, double t) noexcept {
  return a * (1.0 - t) + b * t;
}

inline std::ostream& operator<<(std::ostream& os, const Vec3& v) {
  return os << '(' << v.x << ", " << v.y << ", " << v.z << ')';
}

}  // namespace cad
