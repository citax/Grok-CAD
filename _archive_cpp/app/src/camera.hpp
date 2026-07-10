#pragma once

#include "cadcore/math/mat4.hpp"
#include "cadcore/math/quat.hpp"
#include "cadcore/math/vec3.hpp"

namespace app {

enum class StandardView {
  Front,
  Back,
  Top,
  Bottom,
  Right,
  Left,
  Isometric
};

/// Yaw/pitch orbit camera around a target. Pitch is clamped — no gimbal flip.
class Camera {
 public:
  void orbit(double dx_pixels, double dy_pixels);
  void pan(double dx_pixels, double dy_pixels, int viewport_h);
  void zoom(double wheel_delta);

  [[nodiscard]] cad::Mat4 view_matrix() const;
  [[nodiscard]] cad::Mat4 projection_matrix(double aspect) const;

  [[nodiscard]] cad::Vec3 eye() const;
  [[nodiscard]] cad::Vec3 target() const noexcept { return target_; }
  [[nodiscard]] double distance() const noexcept { return distance_; }
  [[nodiscard]] double yaw() const noexcept { return yaw_; }
  [[nodiscard]] double pitch() const noexcept { return pitch_; }
  [[nodiscard]] double fovy() const noexcept { return fovy_; }

  void set_target(const cad::Vec3& t) noexcept { target_ = t; }
  void set_distance(double d) noexcept;
  void frame_bounds(const cad::Vec3& mn, const cad::Vec3& mx);
  void set_standard_view(StandardView v);
  void zoom_to_fit(const cad::Vec3& mn, const cad::Vec3& mx) { frame_bounds(mn, mx); }

 private:
  cad::Vec3 target_{0, 0, 0};
  double yaw_ = 0.6;
  double pitch_ = 0.45;
  double distance_ = 8.0;
  double fovy_ = 45.0 * cad::kDeg2Rad;
  double znear_ = 0.05;
  double zfar_ = 2000.0;

  static constexpr double kOrbitSpeed = 0.005;
  static constexpr double kZoomFactor = 0.0015;
};

}  // namespace app
