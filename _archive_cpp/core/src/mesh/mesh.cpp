#include "cadcore/mesh/mesh.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <utility>
#include <vector>

namespace cad {
namespace {

// Map each vertex index to a representative welded by position.
std::vector<std::uint32_t> weld_map(const std::vector<Vec3>& positions, double eps) {
  std::vector<std::uint32_t> rep(positions.size());
  for (std::size_t i = 0; i < positions.size(); ++i) {
    rep[i] = static_cast<std::uint32_t>(i);
    for (std::size_t j = 0; j < i; ++j) {
      if (positions[i].is_near(positions[j], eps)) {
        rep[i] = rep[j];
        break;
      }
    }
  }
  return rep;
}

}  // namespace

void Mesh::compute_normals() {
  normals.assign(positions.size(), Vec3{0, 0, 0});
  const std::size_t tri_count = triangle_count();
  for (std::size_t t = 0; t < tri_count; ++t) {
    const auto i0 = indices[t * 3 + 0];
    const auto i1 = indices[t * 3 + 1];
    const auto i2 = indices[t * 3 + 2];
    const Vec3& p0 = positions[i0];
    const Vec3& p1 = positions[i1];
    const Vec3& p2 = positions[i2];
    const Vec3 fn = cross(p1 - p0, p2 - p0);  // area-weighted
    normals[i0] += fn;
    normals[i1] += fn;
    normals[i2] += fn;
  }
  for (auto& n : normals) {
    n = n.normalized();
    if (n.length_sq() < kAbsEps) {
      n = {0, 1, 0};
    }
  }
}

double Mesh::volume() const {
  double vol = 0.0;
  const std::size_t tri_count = triangle_count();
  for (std::size_t t = 0; t < tri_count; ++t) {
    const Vec3& a = positions[indices[t * 3 + 0]];
    const Vec3& b = positions[indices[t * 3 + 1]];
    const Vec3& c = positions[indices[t * 3 + 2]];
    vol += dot(a, cross(b, c));
  }
  return vol / 6.0;
}

double Mesh::surface_area() const {
  double area = 0.0;
  const std::size_t tri_count = triangle_count();
  for (std::size_t t = 0; t < tri_count; ++t) {
    const Vec3& a = positions[indices[t * 3 + 0]];
    const Vec3& b = positions[indices[t * 3 + 1]];
    const Vec3& c = positions[indices[t * 3 + 2]];
    area += 0.5 * cross(b - a, c - a).length();
  }
  return area;
}

bool Mesh::is_watertight(double area_eps) const {
  if (indices.size() % 3 != 0 || indices.empty()) {
    return false;
  }

  // Weld by position so hard-edge split vertices still form a closed shell.
  const auto rep = weld_map(positions, kGeomEps * 100.0);

  struct EdgeKey {
    std::uint32_t a, b;
    bool operator<(const EdgeKey& o) const {
      return a < o.a || (a == o.a && b < o.b);
    }
  };
  std::map<EdgeKey, int> signed_count;
  std::map<EdgeKey, int> abs_count;

  const std::size_t tri_count = triangle_count();
  for (std::size_t t = 0; t < tri_count; ++t) {
    const std::uint32_t raw[3] = {indices[t * 3], indices[t * 3 + 1], indices[t * 3 + 2]};
    for (int k = 0; k < 3; ++k) {
      if (raw[k] >= positions.size()) {
        return false;
      }
    }
    const std::uint32_t i0 = rep[raw[0]];
    const std::uint32_t i1 = rep[raw[1]];
    const std::uint32_t i2 = rep[raw[2]];
    if (i0 == i1 || i1 == i2 || i0 == i2) {
      return false;
    }
    const Vec3& a = positions[raw[0]];
    const Vec3& b = positions[raw[1]];
    const Vec3& c = positions[raw[2]];
    if (0.5 * cross(b - a, c - a).length() <= area_eps) {
      return false;
    }
    const std::uint32_t ids[3] = {i0, i1, i2};
    for (int e = 0; e < 3; ++e) {
      const std::uint32_t u = ids[e];
      const std::uint32_t v = ids[(e + 1) % 3];
      EdgeKey key{std::min(u, v), std::max(u, v)};
      abs_count[key] += 1;
      if (u < v) {
        signed_count[key] += 1;
      } else {
        signed_count[key] -= 1;
      }
    }
  }
  for (const auto& [key, count] : abs_count) {
    (void)key;
    if (count != 2) {
      return false;
    }
  }
  for (const auto& [key, count] : signed_count) {
    (void)key;
    if (count != 0) {
      return false;
    }
  }
  return true;
}

std::pair<Vec3, Vec3> Mesh::bounds() const {
  if (positions.empty()) {
    return {{0, 0, 0}, {0, 0, 0}};
  }
  Vec3 mn = positions[0];
  Vec3 mx = positions[0];
  for (const auto& p : positions) {
    mn.x = std::min(mn.x, p.x);
    mn.y = std::min(mn.y, p.y);
    mn.z = std::min(mn.z, p.z);
    mx.x = std::max(mx.x, p.x);
    mx.y = std::max(mx.y, p.y);
    mx.z = std::max(mx.z, p.z);
  }
  return {mn, mx};
}

void Mesh::append(const Mesh& other) {
  const auto base = static_cast<std::uint32_t>(positions.size());
  positions.insert(positions.end(), other.positions.begin(), other.positions.end());
  normals.insert(normals.end(), other.normals.begin(), other.normals.end());
  indices.reserve(indices.size() + other.indices.size());
  for (auto i : other.indices) {
    indices.push_back(base + i);
  }
}

}  // namespace cad
