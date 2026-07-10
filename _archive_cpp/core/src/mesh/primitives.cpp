#include "cadcore/mesh/primitives.hpp"

#include "cadcore/math/constants.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cad {

Mesh make_box(double width, double height, double depth) {
  const double hx = width * 0.5;
  const double hy = height * 0.5;
  const double hz = depth * 0.5;

  // 24 vertices (unique normals per face corner) for hard edges
  const Vec3 positions[24] = {
      // +X
      {hx, -hy, -hz},
      {hx, hy, -hz},
      {hx, hy, hz},
      {hx, -hy, hz},
      // -X
      {-hx, -hy, hz},
      {-hx, hy, hz},
      {-hx, hy, -hz},
      {-hx, -hy, -hz},
      // +Y
      {-hx, hy, -hz},
      {-hx, hy, hz},
      {hx, hy, hz},
      {hx, hy, -hz},
      // -Y
      {-hx, -hy, hz},
      {-hx, -hy, -hz},
      {hx, -hy, -hz},
      {hx, -hy, hz},
      // +Z
      {-hx, -hy, hz},
      {hx, -hy, hz},
      {hx, hy, hz},
      {-hx, hy, hz},
      // -Z
      {hx, -hy, -hz},
      {-hx, -hy, -hz},
      {-hx, hy, -hz},
      {hx, hy, -hz},
  };
  const Vec3 normals[24] = {
      {1, 0, 0},  {1, 0, 0},  {1, 0, 0},  {1, 0, 0},  {-1, 0, 0}, {-1, 0, 0},
      {-1, 0, 0}, {-1, 0, 0}, {0, 1, 0},  {0, 1, 0},  {0, 1, 0},  {0, 1, 0},
      {0, -1, 0}, {0, -1, 0}, {0, -1, 0}, {0, -1, 0}, {0, 0, 1},  {0, 0, 1},
      {0, 0, 1},  {0, 0, 1},  {0, 0, -1}, {0, 0, -1}, {0, 0, -1}, {0, 0, -1},
  };

  Mesh mesh;
  mesh.positions.assign(positions, positions + 24);
  mesh.normals.assign(normals, normals + 24);
  mesh.indices.reserve(36);
  for (std::uint32_t face = 0; face < 6; ++face) {
    const std::uint32_t b = face * 4;
    mesh.indices.push_back(b + 0);
    mesh.indices.push_back(b + 1);
    mesh.indices.push_back(b + 2);
    mesh.indices.push_back(b + 0);
    mesh.indices.push_back(b + 2);
    mesh.indices.push_back(b + 3);
  }
  return mesh;
}

Mesh make_sphere(double radius, int segments, int rings) {
  if (segments < 3) {
    segments = 3;
  }
  if (rings < 2) {
    rings = 2;
  }
  Mesh mesh;
  // poles + ring vertices
  mesh.positions.push_back({0, radius, 0});
  mesh.normals.push_back({0, 1, 0});
  for (int lat = 1; lat < rings; ++lat) {
    const double v = static_cast<double>(lat) / static_cast<double>(rings);
    const double theta = v * kPi;  // 0..pi
    const double y = radius * std::cos(theta);
    const double r = radius * std::sin(theta);
    for (int lon = 0; lon < segments; ++lon) {
      const double u = static_cast<double>(lon) / static_cast<double>(segments);
      const double phi = u * kTwoPi;
      const double x = r * std::cos(phi);
      const double z = r * std::sin(phi);
      const Vec3 p{x, y, z};
      mesh.positions.push_back(p);
      mesh.normals.push_back(p.normalized());
    }
  }
  mesh.positions.push_back({0, -radius, 0});
  mesh.normals.push_back({0, -1, 0});

  const auto north = 0u;
  const auto south = static_cast<std::uint32_t>(mesh.positions.size() - 1);
  const auto ring_start = 1u;

  // Top cap
  for (int lon = 0; lon < segments; ++lon) {
    const auto i0 = ring_start + static_cast<std::uint32_t>(lon);
    const auto i1 = ring_start + static_cast<std::uint32_t>((lon + 1) % segments);
    mesh.indices.push_back(north);
    mesh.indices.push_back(i1);
    mesh.indices.push_back(i0);
  }
  // Quads between rings
  for (int lat = 0; lat < rings - 2; ++lat) {
    const auto row0 = ring_start + static_cast<std::uint32_t>(lat * segments);
    const auto row1 = ring_start + static_cast<std::uint32_t>((lat + 1) * segments);
    for (int lon = 0; lon < segments; ++lon) {
      const auto i0 = row0 + static_cast<std::uint32_t>(lon);
      const auto i1 = row0 + static_cast<std::uint32_t>((lon + 1) % segments);
      const auto i2 = row1 + static_cast<std::uint32_t>((lon + 1) % segments);
      const auto i3 = row1 + static_cast<std::uint32_t>(lon);
      mesh.indices.push_back(i0);
      mesh.indices.push_back(i1);
      mesh.indices.push_back(i2);
      mesh.indices.push_back(i0);
      mesh.indices.push_back(i2);
      mesh.indices.push_back(i3);
    }
  }
  // Bottom cap
  const auto last_ring =
      ring_start + static_cast<std::uint32_t>((rings - 2) * segments);
  for (int lon = 0; lon < segments; ++lon) {
    const auto i0 = last_ring + static_cast<std::uint32_t>(lon);
    const auto i1 = last_ring + static_cast<std::uint32_t>((lon + 1) % segments);
    mesh.indices.push_back(south);
    mesh.indices.push_back(i0);
    mesh.indices.push_back(i1);
  }
  if (mesh.volume() < 0.0) {
    for (std::size_t t = 0; t < mesh.indices.size(); t += 3) {
      std::swap(mesh.indices[t + 1], mesh.indices[t + 2]);
    }
  }
  return mesh;
}

Mesh make_cylinder(double radius, double height, int segments) {
  if (segments < 3) {
    segments = 3;
  }
  const double hy = height * 0.5;
  Mesh mesh;

  // Side vertices (two rings with outward normals)
  for (int i = 0; i < segments; ++i) {
    const double a = kTwoPi * static_cast<double>(i) / static_cast<double>(segments);
    const double cx = std::cos(a);
    const double cz = std::sin(a);
    const double x = radius * cx;
    const double z = radius * cz;
    mesh.positions.push_back({x, hy, z});
    mesh.normals.push_back({cx, 0, cz});
    mesh.positions.push_back({x, -hy, z});
    mesh.normals.push_back({cx, 0, cz});
  }
  // Top cap ring + center
  const auto top_center = static_cast<std::uint32_t>(mesh.positions.size());
  mesh.positions.push_back({0, hy, 0});
  mesh.normals.push_back({0, 1, 0});
  const auto top_ring = static_cast<std::uint32_t>(mesh.positions.size());
  for (int i = 0; i < segments; ++i) {
    const double a = kTwoPi * static_cast<double>(i) / static_cast<double>(segments);
    mesh.positions.push_back({radius * std::cos(a), hy, radius * std::sin(a)});
    mesh.normals.push_back({0, 1, 0});
  }
  // Bottom cap ring + center
  const auto bot_center = static_cast<std::uint32_t>(mesh.positions.size());
  mesh.positions.push_back({0, -hy, 0});
  mesh.normals.push_back({0, -1, 0});
  const auto bot_ring = static_cast<std::uint32_t>(mesh.positions.size());
  for (int i = 0; i < segments; ++i) {
    const double a = kTwoPi * static_cast<double>(i) / static_cast<double>(segments);
    mesh.positions.push_back({radius * std::cos(a), -hy, radius * std::sin(a)});
    mesh.normals.push_back({0, -1, 0});
  }

  // Side quads — outward normals, manifold edge orientation vs caps
  for (int i = 0; i < segments; ++i) {
    const auto top_i = static_cast<std::uint32_t>(i * 2);
    const auto bot_i = static_cast<std::uint32_t>(i * 2 + 1);
    const auto j = (i + 1) % segments;
    const auto top_j = static_cast<std::uint32_t>(j * 2);
    const auto bot_j = static_cast<std::uint32_t>(j * 2 + 1);
    // CCW when viewed from outside
    mesh.indices.push_back(top_i);
    mesh.indices.push_back(bot_i);
    mesh.indices.push_back(bot_j);
    mesh.indices.push_back(top_i);
    mesh.indices.push_back(bot_j);
    mesh.indices.push_back(top_j);
  }
  // Top cap: ring edge direction opposite side's top edge (top_j -> top_i on side)
  // Side top edge is top_j -> top_i (from second tri: bot_j->top_j->top_i closes...
  // Actually second tri: top_i, bot_j, top_j → edges top_i->bot_j, bot_j->top_j, top_j->top_i
  // So top edge is top_j -> top_i. Cap needs top_i -> top_j.
  for (int i = 0; i < segments; ++i) {
    const auto i0 = top_ring + static_cast<std::uint32_t>(i);
    const auto i1 = top_ring + static_cast<std::uint32_t>((i + 1) % segments);
    mesh.indices.push_back(top_center);
    mesh.indices.push_back(i0);
    mesh.indices.push_back(i1);
  }
  // Bottom: side bottom edge is bot_i -> bot_j; need opposite bot_j -> bot_i on cap
  for (int i = 0; i < segments; ++i) {
    const auto i0 = bot_ring + static_cast<std::uint32_t>(i);
    const auto i1 = bot_ring + static_cast<std::uint32_t>((i + 1) % segments);
    mesh.indices.push_back(bot_center);
    mesh.indices.push_back(i1);
    mesh.indices.push_back(i0);
  }

  // Ensure outward orientation for solid volume
  if (mesh.volume() < 0.0) {
    for (std::size_t t = 0; t < mesh.indices.size(); t += 3) {
      std::swap(mesh.indices[t + 1], mesh.indices[t + 2]);
    }
  }
  return mesh;
}

}  // namespace cad
