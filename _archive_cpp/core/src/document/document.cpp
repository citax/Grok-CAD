#include "cadcore/document/document.hpp"

#include "cadcore/mesh/csg.hpp"
#include "cadcore/mesh/primitives.hpp"
#include "cadcore/mesh/transform.hpp"

#include <algorithm>
#include <unordered_set>

namespace cad {

int Document::add_feature(Feature f) {
  f.id = next_id_++;
  if (f.name.empty()) {
    f.name = std::string(feature_type_name(f.type));
    if (!is_reference_plane(f.type)) {
      f.name += " " + std::to_string(f.id);
    }
  }
  features_.push_back(std::move(f));
  selected_id_ = features_.back().id;
  return features_.back().id;
}

bool Document::remove_feature(int id) {
  const Feature* existing = find(id);
  if (!existing) {
    return false;
  }
  // Reference planes are permanent roots (SolidWorks-style).
  if (is_reference_plane(existing->type)) {
    return false;
  }
  const auto it =
      std::remove_if(features_.begin(), features_.end(),
                     [id](const Feature& f) { return f.id == id; });
  if (it == features_.end()) {
    return false;
  }
  features_.erase(it, features_.end());
  if (selected_id_ == id) {
    selected_id_ = features_.empty() ? -1 : features_.front().id;
  }
  return true;
}

Feature* Document::find(int id) {
  for (auto& f : features_) {
    if (f.id == id) {
      return &f;
    }
  }
  return nullptr;
}

const Feature* Document::find(int id) const {
  for (const auto& f : features_) {
    if (f.id == id) {
      return &f;
    }
  }
  return nullptr;
}

std::optional<Mesh> Document::evaluate_feature(int id) const {
  const Feature* f = find(id);
  if (!f || f->suppressed) {
    return std::nullopt;
  }
  // Reference planes have no solid mesh — rendered by the viewport.
  if (is_reference_plane(f->type)) {
    return std::nullopt;
  }

  Mesh base;
  switch (f->type) {
    case FeatureType::PlaneFront:
    case FeatureType::PlaneTop:
    case FeatureType::PlaneRight:
      return std::nullopt;
    case FeatureType::Box:
      base = make_box(f->width, f->height, f->depth);
      break;
    case FeatureType::Sphere:
      base = make_sphere(f->radius, f->segments, f->rings);
      break;
    case FeatureType::Cylinder:
      base = make_cylinder(f->radius, f->height, f->segments);
      break;
    case FeatureType::BooleanUnion:
    case FeatureType::BooleanDifference:
    case FeatureType::BooleanIntersection: {
      auto ma = evaluate_feature(f->operand_a);
      auto mb = evaluate_feature(f->operand_b);
      if (!ma || !mb) {
        return std::nullopt;
      }
      BooleanOp op = BooleanOp::Union;
      if (f->type == FeatureType::BooleanDifference) {
        op = BooleanOp::Difference;
      } else if (f->type == FeatureType::BooleanIntersection) {
        op = BooleanOp::Intersection;
      }
      base = boolean_op(*ma, *mb, op);
      break;
    }
  }

  const Mat4 trs = make_trs(f->translation, f->rotation_deg, f->scale);
  return transformed(base, trs);
}

std::unordered_map<int, Mesh> Document::evaluate_all() const {
  std::unordered_map<int, Mesh> result;
  for (const auto& f : features_) {
    if (auto m = evaluate_feature(f.id)) {
      result.emplace(f.id, std::move(*m));
    }
  }
  return result;
}

Mesh Document::evaluate_display() const {
  std::unordered_set<int> used_as_operand;
  for (const auto& f : features_) {
    if (is_boolean(f.type)) {
      if (f.operand_a >= 0) {
        used_as_operand.insert(f.operand_a);
      }
      if (f.operand_b >= 0) {
        used_as_operand.insert(f.operand_b);
      }
    }
  }

  Mesh display;
  for (const auto& f : features_) {
    if (!f.visible || f.suppressed || is_reference_plane(f.type)) {
      continue;
    }
    if (used_as_operand.count(f.id) != 0) {
      continue;
    }
    if (auto m = evaluate_feature(f.id)) {
      display.append(*m);
    }
  }
  return display;
}

void Document::clear() {
  features_.clear();
  next_id_ = 1;
  selected_id_ = -1;
  name_ = "Untitled";
}

void Document::seed_reference_planes() {
  auto has = [&](FeatureType t) {
    for (const auto& f : features_) {
      if (f.type == t) {
        return true;
      }
    }
    return false;
  };
  if (!has(FeatureType::PlaneFront)) {
    Feature f;
    f.type = FeatureType::PlaneFront;
    f.name = "Front Plane";
    add_feature(std::move(f));
  }
  if (!has(FeatureType::PlaneTop)) {
    Feature f;
    f.type = FeatureType::PlaneTop;
    f.name = "Top Plane";
    add_feature(std::move(f));
  }
  if (!has(FeatureType::PlaneRight)) {
    Feature f;
    f.type = FeatureType::PlaneRight;
    f.name = "Right Plane";
    add_feature(std::move(f));
  }
  // Prefer Front selected by default when only planes exist
  if (selected_id_ < 0) {
    for (const auto& f : features_) {
      if (f.type == FeatureType::PlaneFront) {
        selected_id_ = f.id;
        break;
      }
    }
  }
}

void Document::ensure_next_id_after(int max_id) {
  if (max_id + 1 > next_id_) {
    next_id_ = max_id + 1;
  }
}

}  // namespace cad
