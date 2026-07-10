#pragma once

#include "cadcore/mesh/mesh.hpp"

namespace cad {

/// Axis-aligned box centered at origin. Size is full extent (width,height,depth)
/// along X,Y,Z. Watertight, 12 triangles, outward normals.
[[nodiscard]] Mesh make_box(double width, double height, double depth);

/// UV sphere centered at origin. `segments` = longitude divisions,
/// `rings` = latitude divisions (must be >= 3 and >= 2 respectively).
[[nodiscard]] Mesh make_sphere(double radius, int segments = 32, int rings = 16);

/// Cylinder along Y, centered at origin. Height along Y, radius in XZ.
/// Caps included. Watertight.
[[nodiscard]] Mesh make_cylinder(double radius, double height, int segments = 32);

}  // namespace cad
