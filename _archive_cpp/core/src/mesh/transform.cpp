#include "cadcore/mesh/transform.hpp"

#include "cadcore/math/constants.hpp"

namespace cad {

Mesh transformed(const Mesh& mesh, const Mat4& m) {
  Mesh out;
  out.positions.reserve(mesh.positions.size());
  out.normals.reserve(mesh.normals.size());
  out.indices = mesh.indices;

  const Mat4 normal_matrix = m.inverted().transposed();

  for (const auto& p : mesh.positions) {
    out.positions.push_back(m.transform_point(p));
  }
  if (mesh.normals.size() == mesh.positions.size()) {
    for (const auto& n : mesh.normals) {
      out.normals.push_back(normal_matrix.transform_vector(n).normalized());
    }
  } else {
    out.compute_normals();
  }
  return out;
}

Mat4 make_trs(const Vec3& translation, const Vec3& rotation_deg, const Vec3& scale) {
  const Mat4 T = Mat4::translation(translation);
  const Mat4 R = Mat4::rotation_xyz(rotation_deg.x * kDeg2Rad, rotation_deg.y * kDeg2Rad,
                                    rotation_deg.z * kDeg2Rad);
  const Mat4 S = Mat4::scale(scale);
  return T * R * S;
}

}  // namespace cad
