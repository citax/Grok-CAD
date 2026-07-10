#pragma once

#include "cadcore/math/vec3.hpp"

#include <string>

namespace cad {

enum class FeatureType {
  // Reference geometry (SolidWorks-style tree roots)
  PlaneFront,  // XY
  PlaneTop,    // XZ
  PlaneRight,  // YZ
  // Solid body history (kept for later extrude/revolve / existing CSG)
  Box,
  Sphere,
  Cylinder,
  BooleanUnion,
  BooleanDifference,
  BooleanIntersection
};

[[nodiscard]] inline const char* feature_type_name(FeatureType t) noexcept {
  switch (t) {
    case FeatureType::PlaneFront:
      return "Front Plane";
    case FeatureType::PlaneTop:
      return "Top Plane";
    case FeatureType::PlaneRight:
      return "Right Plane";
    case FeatureType::Box:
      return "Box";
    case FeatureType::Sphere:
      return "Sphere";
    case FeatureType::Cylinder:
      return "Cylinder";
    case FeatureType::BooleanUnion:
      return "Union";
    case FeatureType::BooleanDifference:
      return "Difference";
    case FeatureType::BooleanIntersection:
      return "Intersection";
  }
  return "Unknown";
}

[[nodiscard]] inline bool is_reference_plane(FeatureType t) noexcept {
  return t == FeatureType::PlaneFront || t == FeatureType::PlaneTop ||
         t == FeatureType::PlaneRight;
}

[[nodiscard]] inline bool is_boolean(FeatureType t) noexcept {
  return t == FeatureType::BooleanUnion || t == FeatureType::BooleanDifference ||
         t == FeatureType::BooleanIntersection;
}

/// Unit normal for a reference plane in world space (outward for display).
[[nodiscard]] inline Vec3 plane_normal(FeatureType t) noexcept {
  switch (t) {
    case FeatureType::PlaneFront:
      return {0, 0, 1};  // XY, looking along +Z
    case FeatureType::PlaneTop:
      return {0, 1, 0};  // XZ, looking along +Y
    case FeatureType::PlaneRight:
      return {1, 0, 0};  // YZ, looking along +X
    default:
      return {0, 0, 1};
  }
}

/// A single history / tree item. Reference planes are fixed roots; solids will
/// attach later (sketch → extrude). Transform applies to solid bodies.
struct Feature {
  int id = -1;
  std::string name;
  FeatureType type = FeatureType::Box;

  // Primitive parameters
  double width = 1.0;
  double height = 1.0;
  double depth = 1.0;
  double radius = 0.5;

  int segments = 32;
  int rings = 16;

  int operand_a = -1;
  int operand_b = -1;

  Vec3 translation{0, 0, 0};
  Vec3 rotation_deg{0, 0, 0};
  Vec3 scale{1, 1, 1};

  bool visible = true;
  bool suppressed = false;
};

}  // namespace cad
