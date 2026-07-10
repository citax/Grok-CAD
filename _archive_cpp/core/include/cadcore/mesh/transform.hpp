#pragma once

#include "cadcore/math/mat4.hpp"
#include "cadcore/mesh/mesh.hpp"

namespace cad {

/// Apply an affine transform to all positions; re-transform normals by
/// inverse-transpose of the linear part (handles non-uniform scale).
[[nodiscard]] Mesh transformed(const Mesh& mesh, const Mat4& m);

/// Build TRS matrix: T * R_xyz * S (scale, then rotate, then translate).
/// Rotations are Euler XYZ in **degrees**.
[[nodiscard]] Mat4 make_trs(const Vec3& translation, const Vec3& rotation_deg,
                            const Vec3& scale);

}  // namespace cad
