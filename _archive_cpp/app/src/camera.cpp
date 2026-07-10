#include "camera.hpp"

#include <algorithm>
#include <cmath>

namespace app {

void Camera::set_distance(double d) noexcept {
  distance_ = cad::clamp(d, 0.05, 500.0);
}

void Camera::orbit(double dx_pixels, double dy_pixels) {
  yaw_ -= dx_pixels * kOrbitSpeed;
  pitch_ += dy_pixels * kOrbitSpeed;
  constexpr double lim = cad::kPi * 0.5 - 0.05;
  pitch_ = cad::clamp(pitch_, -lim, lim);
}

void Camera::pan(double dx_pixels, double dy_pixels, int viewport_h) {
  if (viewport_h <= 0) {
    return;
  }
  const double world_per_pixel =
      2.0 * distance_ * std::tan(fovy_ * 0.5) / static_cast<double>(viewport_h);
  const cad::Quat q = cad::Quat::from_yaw_pitch(yaw_, pitch_);
  const cad::Vec3 right = q.rotate({1, 0, 0});
  const cad::Vec3 up = q.rotate({0, 1, 0});
  target_ = target_ - right * (dx_pixels * world_per_pixel) + up * (dy_pixels * world_per_pixel);
}

void Camera::zoom(double wheel_delta) {
  const double factor = std::exp(-wheel_delta * kZoomFactor);
  distance_ = cad::clamp(distance_ * factor, 0.05, 500.0);
}

cad::Vec3 Camera::eye() const {
  const cad::Quat q = cad::Quat::from_yaw_pitch(yaw_, pitch_);
  const cad::Vec3 forward = q.rotate({0, 0, -1});
  return target_ - forward * distance_;
}

cad::Mat4 Camera::view_matrix() const {
  return cad::Mat4::look_at(eye(), target_, {0, 1, 0});
}

cad::Mat4 Camera::projection_matrix(double aspect) const {
  const double a = aspect > 1e-6 ? aspect : 1.0;
  return cad::Mat4::perspective(fovy_, a, znear_, zfar_);
}

void Camera::frame_bounds(const cad::Vec3& mn, const cad::Vec3& mx) {
  target_ = (mn + mx) * 0.5;
  const cad::Vec3 ext = mx - mn;
  const double radius = std::max({ext.x, ext.y, ext.z, 0.5}) * 0.5;
  distance_ = radius / std::tan(fovy_ * 0.5) * 1.8;
  distance_ = cad::clamp(distance_, 0.5, 500.0);
}

void Camera::set_standard_view(StandardView v) {
  target_ = {0, 0, 0};
  switch (v) {
    case StandardView::Front:  // look -Z toward origin (XY plane)
      yaw_ = 0.0;
      pitch_ = 0.0;
      break;
    case StandardView::Back:
      yaw_ = cad::kPi;
      pitch_ = 0.0;
      break;
    case StandardView::Top:  // look -Y
      yaw_ = 0.0;
      pitch_ = cad::kPi * 0.5 - 0.05;
      break;
    case StandardView::Bottom:
      yaw_ = 0.0;
      pitch_ = -(cad::kPi * 0.5 - 0.05);
      break;
    case StandardView::Right:  // look -X
      yaw_ = -cad::kPi * 0.5;
      pitch_ = 0.0;
      break;
    case StandardView::Left:
      yaw_ = cad::kPi * 0.5;
      pitch_ = 0.0;
      break;
    case StandardView::Isometric:
      yaw_ = cad::kPi * 0.25;
      pitch_ = cad::kPi / 6.0;
      break;
  }
  if (distance_ < 1.0) {
    distance_ = 8.0;
  }
}

}  // namespace app
