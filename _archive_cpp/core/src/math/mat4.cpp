#include "cadcore/math/mat4.hpp"

#include <cmath>

namespace cad {

Mat4 Mat4::rotation_x(double rad) noexcept {
  const double c = std::cos(rad);
  const double s = std::sin(rad);
  Mat4 r;
  r.m[5] = c;
  r.m[6] = s;
  r.m[9] = -s;
  r.m[10] = c;
  return r;
}

Mat4 Mat4::rotation_y(double rad) noexcept {
  const double c = std::cos(rad);
  const double s = std::sin(rad);
  Mat4 r;
  r.m[0] = c;
  r.m[2] = -s;
  r.m[8] = s;
  r.m[10] = c;
  return r;
}

Mat4 Mat4::rotation_z(double rad) noexcept {
  const double c = std::cos(rad);
  const double s = std::sin(rad);
  Mat4 r;
  r.m[0] = c;
  r.m[1] = s;
  r.m[4] = -s;
  r.m[5] = c;
  return r;
}

Mat4 Mat4::rotation_xyz(double rx, double ry, double rz) noexcept {
  return rotation_z(rz) * rotation_y(ry) * rotation_x(rx);
}

Mat4 Mat4::look_at(const Vec3& eye, const Vec3& target, const Vec3& up) noexcept {
  const Vec3 f = (target - eye).normalized();
  const Vec3 s = cross(f, up).normalized();
  const Vec3 u = cross(s, f);

  Mat4 r;
  r.m[0] = s.x;
  r.m[4] = s.y;
  r.m[8] = s.z;
  r.m[1] = u.x;
  r.m[5] = u.y;
  r.m[9] = u.z;
  r.m[2] = -f.x;
  r.m[6] = -f.y;
  r.m[10] = -f.z;
  r.m[12] = -dot(s, eye);
  r.m[13] = -dot(u, eye);
  r.m[14] = dot(f, eye);
  return r;
}

Mat4 Mat4::perspective(double fovy_rad, double aspect, double znear,
                       double zfar) noexcept {
  Mat4 r;
  const double f = 1.0 / std::tan(fovy_rad * 0.5);
  r.m = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  r.m[0] = f / aspect;
  r.m[5] = f;
  r.m[10] = (zfar + znear) / (znear - zfar);
  r.m[11] = -1.0;
  r.m[14] = (2.0 * zfar * znear) / (znear - zfar);
  return r;
}

Mat4 Mat4::orthographic(double left, double right, double bottom, double top,
                        double znear, double zfar) noexcept {
  Mat4 r;
  r.m[0] = 2.0 / (right - left);
  r.m[5] = 2.0 / (top - bottom);
  r.m[10] = -2.0 / (zfar - znear);
  r.m[12] = -(right + left) / (right - left);
  r.m[13] = -(top + bottom) / (top - bottom);
  r.m[14] = -(zfar + znear) / (zfar - znear);
  return r;
}

Mat4 Mat4::operator*(const Mat4& o) const noexcept {
  Mat4 r;
  for (int col = 0; col < 4; ++col) {
    for (int row = 0; row < 4; ++row) {
      double sum = 0.0;
      for (int k = 0; k < 4; ++k) {
        sum += (*this)(row, k) * o(k, col);
      }
      r(row, col) = sum;
    }
  }
  return r;
}

Vec3 Mat4::transform_point(const Vec3& p) const noexcept {
  const double x = m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12];
  const double y = m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13];
  const double z = m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14];
  const double w = m[3] * p.x + m[7] * p.y + m[11] * p.z + m[15];
  if (!near_zero(w - 1.0) && !near_zero(w)) {
    return {x / w, y / w, z / w};
  }
  return {x, y, z};
}

Vec3 Mat4::transform_vector(const Vec3& v) const noexcept {
  return {m[0] * v.x + m[4] * v.y + m[8] * v.z, m[1] * v.x + m[5] * v.y + m[9] * v.z,
          m[2] * v.x + m[6] * v.y + m[10] * v.z};
}

Vec3 Mat4::transform_normal(const Vec3& n) const noexcept {
  // Inverse-transpose of upper-left 3x3. For pure rotations this is the same.
  const Mat4 inv = inverted().transposed();
  return inv.transform_vector(n).normalized();
}

Mat4 Mat4::transposed() const noexcept {
  Mat4 r;
  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 4; ++col) {
      r(row, col) = (*this)(col, row);
    }
  }
  return r;
}

Mat4 Mat4::inverted() const noexcept {
  // General 4x4 inverse via adjugate / determinant.
  const auto& a = m;
  std::array<double, 16> inv{};

  inv[0] = a[5] * a[10] * a[15] - a[5] * a[11] * a[14] - a[9] * a[6] * a[15] +
           a[9] * a[7] * a[14] + a[13] * a[6] * a[11] - a[13] * a[7] * a[10];
  inv[4] = -a[4] * a[10] * a[15] + a[4] * a[11] * a[14] + a[8] * a[6] * a[15] -
           a[8] * a[7] * a[14] - a[12] * a[6] * a[11] + a[12] * a[7] * a[10];
  inv[8] = a[4] * a[9] * a[15] - a[4] * a[11] * a[13] - a[8] * a[5] * a[15] +
           a[8] * a[7] * a[13] + a[12] * a[5] * a[11] - a[12] * a[7] * a[9];
  inv[12] = -a[4] * a[9] * a[14] + a[4] * a[10] * a[13] + a[8] * a[5] * a[14] -
            a[8] * a[6] * a[13] - a[12] * a[5] * a[10] + a[12] * a[6] * a[9];
  inv[1] = -a[1] * a[10] * a[15] + a[1] * a[11] * a[14] + a[9] * a[2] * a[15] -
           a[9] * a[3] * a[14] - a[13] * a[2] * a[11] + a[13] * a[3] * a[10];
  inv[5] = a[0] * a[10] * a[15] - a[0] * a[11] * a[14] - a[8] * a[2] * a[15] +
           a[8] * a[3] * a[14] + a[12] * a[2] * a[11] - a[12] * a[3] * a[10];
  inv[9] = -a[0] * a[9] * a[15] + a[0] * a[11] * a[13] + a[8] * a[1] * a[15] -
           a[8] * a[3] * a[13] - a[12] * a[1] * a[11] + a[12] * a[3] * a[9];
  inv[13] = a[0] * a[9] * a[14] - a[0] * a[10] * a[13] - a[8] * a[1] * a[14] +
            a[8] * a[2] * a[13] + a[12] * a[1] * a[10] - a[12] * a[2] * a[9];
  inv[2] = a[1] * a[6] * a[15] - a[1] * a[7] * a[14] - a[5] * a[2] * a[15] +
           a[5] * a[3] * a[14] + a[13] * a[2] * a[7] - a[13] * a[3] * a[6];
  inv[6] = -a[0] * a[6] * a[15] + a[0] * a[7] * a[14] + a[4] * a[2] * a[15] -
           a[4] * a[3] * a[14] - a[12] * a[2] * a[7] + a[12] * a[3] * a[6];
  inv[10] = a[0] * a[5] * a[15] - a[0] * a[7] * a[13] - a[4] * a[1] * a[15] +
            a[4] * a[3] * a[13] + a[12] * a[1] * a[7] - a[12] * a[3] * a[5];
  inv[14] = -a[0] * a[5] * a[14] + a[0] * a[6] * a[13] + a[4] * a[1] * a[14] -
            a[4] * a[2] * a[13] - a[12] * a[1] * a[6] + a[12] * a[2] * a[5];
  inv[3] = -a[1] * a[6] * a[11] + a[1] * a[7] * a[10] + a[5] * a[2] * a[11] -
           a[5] * a[3] * a[10] - a[9] * a[2] * a[7] + a[9] * a[3] * a[6];
  inv[7] = a[0] * a[6] * a[11] - a[0] * a[7] * a[10] - a[4] * a[2] * a[11] +
           a[4] * a[3] * a[10] + a[8] * a[2] * a[7] - a[8] * a[3] * a[6];
  inv[11] = -a[0] * a[5] * a[11] + a[0] * a[7] * a[9] + a[4] * a[1] * a[11] -
            a[4] * a[3] * a[9] - a[8] * a[1] * a[7] + a[8] * a[3] * a[5];
  inv[15] = a[0] * a[5] * a[10] - a[0] * a[6] * a[9] - a[4] * a[1] * a[10] +
            a[4] * a[2] * a[9] + a[8] * a[1] * a[6] - a[8] * a[2] * a[5];

  double det = a[0] * inv[0] + a[1] * inv[4] + a[2] * inv[8] + a[3] * inv[12];
  Mat4 out;
  if (near_zero(det)) {
    return Mat4::identity();
  }
  det = 1.0 / det;
  for (int i = 0; i < 16; ++i) {
    out.m[static_cast<size_t>(i)] = inv[static_cast<size_t>(i)] * det;
  }
  return out;
}

}  // namespace cad
