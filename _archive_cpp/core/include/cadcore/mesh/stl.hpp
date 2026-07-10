#pragma once

#include "cadcore/mesh/mesh.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace cad {

/// Binary STL writer — **byte-for-byte deterministic** for the same mesh:
/// 1. Triangles are sorted by (centroid.x, centroid.y, centroid.z, n.x, n.y, n.z).
/// 2. Header is 80 zero bytes.
/// 3. Attribute byte count is always 0.
/// 4. Floats are IEEE-754 little-endian; no NaN/Inf generated for finite meshes.
[[nodiscard]] std::vector<std::uint8_t> export_stl_binary(const Mesh& mesh);

/// Write binary STL to a file path. Returns false on I/O failure.
bool write_stl_file(const Mesh& mesh, const std::string& path);

/// Parse a binary STL into a Mesh (no vertex welding). Returns empty on failure.
[[nodiscard]] Mesh import_stl_binary(const std::vector<std::uint8_t>& data);

}  // namespace cad
