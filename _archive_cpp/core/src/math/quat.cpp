#include "cadcore/math/quat.hpp"

#include <cmath>

namespace cad {

Quat Quat::from_axis_angle(const Vec3& axis, double rad) noexcept {
  const Vec3 n = axis.normalized();
  const double half = rad * 0.5;
  const double s = std::sin(half);
  return Quat{std::cos(half), n.x * s, n.y * s, n.z * s}.normalized();
}

Quat Quat::from_yaw_pitch(double yaw_rad, double pitch_rad) noexcept {
  const Quat qy = from_axis_angle({0, 1, 0}, yaw_rad);
  const Quat qx = from_axis_angle({1, 0, 0}, pitch_rad);
  return (qy * qx).normalized();
}

double Quat::length() const noexcept {
  return std::sqrt(w * w + x * x + y * y + z * z);
}

Quat Quat::normalized() const noexcept {
  const double len = length();
  if (near_zero(len)) {
    return identity();
  }
  return {w / len, x / len, y / len, z / len};
}

Quat Quat::operator*(const Quat& o) const noexcept {
  return {w * o.w - x * o.x - y * o.y - z * o.z, w * o.x + x * o.w + y * o.z - z * o.y,
          w * o.y - x * o.z + y * o.w + z * o.x, w * o.z + x * o.y - y * o.x + z * o.w};
}

Vec3 Quat::rotate(const Vec3& v) const noexcept {
  // q * (0,v) * q^-1
  const Quat p{0.0, v.x, v.y, v.z};
  const Quat r = (*this) * p * inverse();
  return {r.x, r.y, r.z};
}

Mat4 Quat::to_mat4() const noexcept {
  const Quat q = normalized();
  const double xx = q.x * q.x;
  const double yy = q.y * q.y;
  const double zz = q.z * q.z;
  const double xy = q.x * q.y;
  const double xz = q.x * q.z;
  const double yz = q.y * q.z;
  const double wx = q.w * q.x;
  const double wy = q.w * q.y;
  const double wz = q.w * q.z;

  Mat4 m;
  m.m[0] = 1.0 - 2.0 * (yy + zz);
  m.m[1] = 2.0 * (xy + wz);
  m.m[2] = 2.0 * (xz - wy);
  m.m[3] = 0.0;

  m.m[4] = 2.0 * (xy - wz);
  m.m[5] = 1.0 - 2.0 * (xx + zz);
  m.m[6] = 2.0 * (yz + wx);
  m.m[7] = 0.0;

  m.m[8] = 2.0 * (xz + wy);
  m.m[9] = 2.0 * (yz - wx);
  m.m[10] = 1.0 - 2.0 * (xx + yy);
  m.m[11] = 0.0;

  m.m[12] = 0.0;
  m.m[13] = 0.0;
  m.m[14] = 0.0;
  m.m[15] = 1.0;
  return m;
}

Quat slerp(const Quat& a, const Quat& b, double t) noexcept {
  Quat q1 = a.normalized();
  Quat q2 = b.normalized();
  double cos_theta = q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z;
  if (cos_theta < 0.0) {
    q2 = {-q2.w, -q2.x, -q2.y, -q2.z};
    cos_theta = -cos_theta;
  }
  if (cos_theta > 0.9995) {
    return Quat{q1.w + t * (q2.w - q1.w), q1.x + t * (q2.x - q1.x),
                q1.y + t * (q2.y - q1.y), q1.z + t * (q2.z - q1.z)}
        .normalized();
  }
  const double theta = std::acos(clamp(cos_theta, -1.0, 1.0));
  const double sin_theta = std::sin(theta);
  const double w1 = std::sin((1.0 - t) * theta) / sin_theta;
  const double w2 = std::sin(t * theta) / sin_theta;
  return {w1 * q1.w + w2 * q2.w, w1 * q1.x + w2 * q2.x, w1 * q1.y + w2 * q2.y,
          w1 * q1.z + w2 * q2.z};
}

}  // namespace cad
