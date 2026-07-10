#pragma once

#include "cadcore/math/vec3.hpp"

#include <array>
#include <cmath>

namespace cad {

/// 4×4 double matrix stored in **column-major** order (OpenGL convention).
///
/// Layout of `m[16]`:
///   column 0: m[0], m[1], m[2], m[3]
///   column 1: m[4], m[5], m[6], m[7]
///   column 2: m[8], m[9], m[10], m[11]
///   column 3: m[12], m[13], m[14], m[15]
///
/// Element (row r, column c) is at index `c * 4 + r`.
/// Multiplication is right-to-left: `M * v` applies M to column vector v.
struct Mat4 {
  std::array<double, 16> m{
      1, 0, 0, 0,  // col0
      0, 1, 0, 0,  // col1
      0, 0, 1, 0,  // col2
      0, 0, 0, 1   // col3
  };

  constexpr Mat4() = default;

  [[nodiscard]] static Mat4 identity() noexcept { return Mat4{}; }

  [[nodiscard]] static Mat4 translation(const Vec3& t) noexcept {
    Mat4 r;
    r.m[12] = t.x;
    r.m[13] = t.y;
    r.m[14] = t.z;
    return r;
  }

  [[nodiscard]] static Mat4 scale(const Vec3& s) noexcept {
    Mat4 r;
    r.m[0] = s.x;
    r.m[5] = s.y;
    r.m[10] = s.z;
    return r;
  }

  /// Rotation about X by angle in radians.
  [[nodiscard]] static Mat4 rotation_x(double rad) noexcept;
  /// Rotation about Y by angle in radians.
  [[nodiscard]] static Mat4 rotation_y(double rad) noexcept;
  /// Rotation about Z by angle in radians.
  [[nodiscard]] static Mat4 rotation_z(double rad) noexcept;

  /// Euler XYZ intrinsic rotations (radians).
  [[nodiscard]] static Mat4 rotation_xyz(double rx, double ry, double rz) noexcept;

  [[nodiscard]] static Mat4 look_at(const Vec3& eye, const Vec3& target,
                                    const Vec3& up) noexcept;
  [[nodiscard]] static Mat4 perspective(double fovy_rad, double aspect, double znear,
                                        double zfar) noexcept;
  [[nodiscard]] static Mat4 orthographic(double left, double right, double bottom,
                                         double top, double znear, double zfar) noexcept;

  [[nodiscard]] double operator()(int row, int col) const noexcept {
    return m[static_cast<size_t>(col * 4 + row)];
  }
  [[nodiscard]] double& operator()(int row, int col) noexcept {
    return m[static_cast<size_t>(col * 4 + row)];
  }

  [[nodiscard]] Mat4 operator*(const Mat4& o) const noexcept;
  [[nodiscard]] Vec3 transform_point(const Vec3& p) const noexcept;
  [[nodiscard]] Vec3 transform_vector(const Vec3& v) const noexcept;
  [[nodiscard]] Vec3 transform_normal(const Vec3& n) const noexcept;

  [[nodiscard]] Mat4 transposed() const noexcept;
  [[nodiscard]] Mat4 inverted() const noexcept;
};

}  // namespace cad
