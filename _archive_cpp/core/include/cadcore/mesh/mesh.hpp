#pragma once

#include "cadcore/math/vec3.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace cad {

/// Indexed triangle mesh with per-vertex positions and normals.
/// Triangles are listed as packed index triples (i0,i1,i2) with CCW winding
/// outward for solid bodies.
struct Mesh {
  std::vector<Vec3> positions;
  std::vector<Vec3> normals;
  std::vector<std::uint32_t> indices;  // 3 * triangle_count entries

  [[nodiscard]] std::size_t vertex_count() const noexcept { return positions.size(); }
  [[nodiscard]] std::size_t triangle_count() const noexcept {
    return indices.size() / 3;
  }
  [[nodiscard]] bool empty() const noexcept { return indices.empty(); }

  void clear() noexcept {
    positions.clear();
    normals.clear();
    indices.clear();
  }

  /// Recompute smooth per-vertex normals from triangle face normals
  /// (area-weighted average, then normalize).
  void compute_normals();

  /// Signed volume of a closed mesh (origin-based tetrahedron sum).
  /// Positive when winding is outward CCW.
  [[nodiscard]] double volume() const;

  /// Total triangle surface area.
  [[nodiscard]] double surface_area() const;

  /// Heuristic watertightness: every edge shared by exactly two triangles,
  /// consistent opposite winding, non-degenerate faces. Does not detect
  /// self-intersections.
  [[nodiscard]] bool is_watertight(double area_eps = kGeomEps * kGeomEps) const;

  /// Axis-aligned bounding box as (min, max). Empty mesh returns zeros.
  [[nodiscard]] std::pair<Vec3, Vec3> bounds() const;

  /// Append another mesh's geometry (indices offset correctly).
  void append(const Mesh& other);
};

}  // namespace cad
