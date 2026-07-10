#include "cadcore/mesh/stl.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>

namespace cad {
namespace {

void write_le_f32(std::vector<std::uint8_t>& out, float v) {
  static_assert(sizeof(float) == 4, "float must be 32-bit");
  std::uint8_t bytes[4];
  std::memcpy(bytes, &v, 4);
  // Assume little-endian host (x86/ARM typical). For portability swap if needed.
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  out.push_back(bytes[3]);
  out.push_back(bytes[2]);
  out.push_back(bytes[1]);
  out.push_back(bytes[0]);
#else
  out.push_back(bytes[0]);
  out.push_back(bytes[1]);
  out.push_back(bytes[2]);
  out.push_back(bytes[3]);
#endif
}

void write_le_u32(std::vector<std::uint8_t>& out, std::uint32_t v) {
  out.push_back(static_cast<std::uint8_t>(v & 0xFF));
  out.push_back(static_cast<std::uint8_t>((v >> 8) & 0xFF));
  out.push_back(static_cast<std::uint8_t>((v >> 16) & 0xFF));
  out.push_back(static_cast<std::uint8_t>((v >> 24) & 0xFF));
}

float read_le_f32(const std::uint8_t* p) {
  std::uint8_t bytes[4];
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  bytes[0] = p[3];
  bytes[1] = p[2];
  bytes[2] = p[1];
  bytes[3] = p[0];
#else
  bytes[0] = p[0];
  bytes[1] = p[1];
  bytes[2] = p[2];
  bytes[3] = p[3];
#endif
  float v;
  std::memcpy(&v, bytes, 4);
  return v;
}

std::uint32_t read_le_u32(const std::uint8_t* p) {
  return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
         (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}

struct Facet {
  Vec3 n, a, b, c;
  Vec3 centroid() const { return (a + b + c) * (1.0 / 3.0); }
};

bool facet_less(const Facet& u, const Facet& v) {
  const Vec3 cu = u.centroid();
  const Vec3 cv = v.centroid();
  if (cu.x != cv.x) return cu.x < cv.x;
  if (cu.y != cv.y) return cu.y < cv.y;
  if (cu.z != cv.z) return cu.z < cv.z;
  if (u.n.x != v.n.x) return u.n.x < v.n.x;
  if (u.n.y != v.n.y) return u.n.y < v.n.y;
  if (u.n.z != v.n.z) return u.n.z < v.n.z;
  if (u.a.x != v.a.x) return u.a.x < v.a.x;
  if (u.a.y != v.a.y) return u.a.y < v.a.y;
  if (u.a.z != v.a.z) return u.a.z < v.a.z;
  if (u.b.x != v.b.x) return u.b.x < v.b.x;
  if (u.b.y != v.b.y) return u.b.y < v.b.y;
  if (u.b.z != v.b.z) return u.b.z < v.b.z;
  if (u.c.x != v.c.x) return u.c.x < v.c.x;
  if (u.c.y != v.c.y) return u.c.y < v.c.y;
  return u.c.z < v.c.z;
}

}  // namespace

std::vector<std::uint8_t> export_stl_binary(const Mesh& mesh) {
  std::vector<Facet> facets;
  facets.reserve(mesh.triangle_count());
  for (std::size_t t = 0; t < mesh.triangle_count(); ++t) {
    const Vec3& a = mesh.positions[mesh.indices[t * 3 + 0]];
    const Vec3& b = mesh.positions[mesh.indices[t * 3 + 1]];
    const Vec3& c = mesh.positions[mesh.indices[t * 3 + 2]];
    Vec3 n = cross(b - a, c - a).normalized();
    if (n.length_sq() < kAbsEps) {
      n = {0, 0, 0};
    }
    facets.push_back({n, a, b, c});
  }
  std::sort(facets.begin(), facets.end(), facet_less);

  std::vector<std::uint8_t> out;
  out.reserve(84 + facets.size() * 50);
  out.insert(out.end(), 80, 0);  // header zeros
  write_le_u32(out, static_cast<std::uint32_t>(facets.size()));

  for (const auto& f : facets) {
    write_le_f32(out, static_cast<float>(f.n.x));
    write_le_f32(out, static_cast<float>(f.n.y));
    write_le_f32(out, static_cast<float>(f.n.z));
    write_le_f32(out, static_cast<float>(f.a.x));
    write_le_f32(out, static_cast<float>(f.a.y));
    write_le_f32(out, static_cast<float>(f.a.z));
    write_le_f32(out, static_cast<float>(f.b.x));
    write_le_f32(out, static_cast<float>(f.b.y));
    write_le_f32(out, static_cast<float>(f.b.z));
    write_le_f32(out, static_cast<float>(f.c.x));
    write_le_f32(out, static_cast<float>(f.c.y));
    write_le_f32(out, static_cast<float>(f.c.z));
    out.push_back(0);
    out.push_back(0);  // attribute byte count
  }
  return out;
}

bool write_stl_file(const Mesh& mesh, const std::string& path) {
  const auto data = export_stl_binary(mesh);
  std::ofstream ofs(path, std::ios::binary);
  if (!ofs) {
    return false;
  }
  ofs.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size()));
  return static_cast<bool>(ofs);
}

Mesh import_stl_binary(const std::vector<std::uint8_t>& data) {
  Mesh mesh;
  if (data.size() < 84) {
    return mesh;
  }
  const std::uint32_t count = read_le_u32(data.data() + 80);
  if (data.size() < 84 + static_cast<std::size_t>(count) * 50) {
    return mesh;
  }
  mesh.positions.reserve(static_cast<std::size_t>(count) * 3);
  mesh.indices.reserve(static_cast<std::size_t>(count) * 3);
  const std::uint8_t* p = data.data() + 84;
  for (std::uint32_t i = 0; i < count; ++i) {
    p += 12;  // skip normal
    const Vec3 a{read_le_f32(p), read_le_f32(p + 4), read_le_f32(p + 8)};
    const Vec3 b{read_le_f32(p + 12), read_le_f32(p + 16), read_le_f32(p + 20)};
    const Vec3 c{read_le_f32(p + 24), read_le_f32(p + 28), read_le_f32(p + 32)};
    p += 36 + 2;
    const auto base = static_cast<std::uint32_t>(mesh.positions.size());
    mesh.positions.push_back(a);
    mesh.positions.push_back(b);
    mesh.positions.push_back(c);
    mesh.indices.push_back(base);
    mesh.indices.push_back(base + 1);
    mesh.indices.push_back(base + 2);
  }
  mesh.compute_normals();
  return mesh;
}

}  // namespace cad
