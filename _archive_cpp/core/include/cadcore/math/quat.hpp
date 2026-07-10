#pragma once

#include "cadcore/math/mat4.hpp"
#include "cadcore/math/vec3.hpp"

namespace cad {

/// Unit quaternion for rotation (w, x, y, z). Avoids gimbal lock for cameras.
struct Quat {
  double w = 1.0;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;

  constexpr Quat() = default;
  constexpr Quat(double w_, double x_, double y_, double z_) noexcept
      : w(w_), x(x_), y(y_), z(z_) {}

  [[nodiscard]] static Quat identity() noexcept { return {}; }

  /// Axis-angle rotation. Axis need not be unit (will be normalized).
  [[nodiscard]] static Quat from_axis_angle(const Vec3& axis, double rad) noexcept;

  /// Yaw (Y) then pitch (X) then roll (Z) — common FPS/orbit construction.
  [[nodiscard]] static Quat from_yaw_pitch(double yaw_rad, double pitch_rad) noexcept;

  [[nodiscard]] Quat normalized() const noexcept;
  [[nodiscard]] Quat conjugate() const noexcept { return {w, -x, -y, -z}; }
  [[nodiscard]] Quat inverse() const noexcept { return conjugate().normalized(); }

  [[nodiscard]] Quat operator*(const Quat& o) const noexcept;
  [[nodiscard]] Vec3 rotate(const Vec3& v) const noexcept;
  [[nodiscard]] Mat4 to_mat4() const noexcept;

  [[nodiscard]] double length() const noexcept;
};

/// Spherical linear interpolation (unit quaternions).
[[nodiscard]] Quat slerp(const Quat& a, const Quat& b, double t) noexcept;

}  // namespace cad
