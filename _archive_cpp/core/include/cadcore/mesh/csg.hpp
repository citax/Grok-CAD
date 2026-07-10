#pragma once

#include "cadcore/mesh/mesh.hpp"

namespace cad {

enum class BooleanOp { Union, Difference, Intersection };

/// BSP-tree constructive solid geometry on triangle meshes.
/// Input meshes should be closed and manifold (watertight). Result is
/// rebuilt into an indexed Mesh with recomputed normals.
///
/// Complexity is O(n log n) typical / O(n²) worst case for n triangles.
[[nodiscard]] Mesh boolean_op(const Mesh& a, const Mesh& b, BooleanOp op);

[[nodiscard]] inline Mesh boolean_union(const Mesh& a, const Mesh& b) {
  return boolean_op(a, b, BooleanOp::Union);
}
[[nodiscard]] inline Mesh boolean_difference(const Mesh& a, const Mesh& b) {
  return boolean_op(a, b, BooleanOp::Difference);
}
[[nodiscard]] inline Mesh boolean_intersection(const Mesh& a, const Mesh& b) {
  return boolean_op(a, b, BooleanOp::Intersection);
}

}  // namespace cad
