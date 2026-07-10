#pragma once

#include "cadcore/document/feature.hpp"
#include "cadcore/mesh/mesh.hpp"

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace cad {

/// Document = ordered feature history + evaluation cache.
/// Evaluating walks features in id-order (creation order) and builds meshes.
/// Ready to evolve into a full DAG / parametric rebuild graph later.
class Document {
 public:
  Document() = default;

  [[nodiscard]] const std::vector<Feature>& features() const noexcept {
    return features_;
  }
  [[nodiscard]] std::vector<Feature>& features() noexcept { return features_; }

  [[nodiscard]] int selected_id() const noexcept { return selected_id_; }
  void set_selected_id(int id) noexcept { selected_id_ = id; }

  [[nodiscard]] const std::string& name() const noexcept { return name_; }
  void set_name(std::string n) { name_ = std::move(n); }

  /// Create a new feature, assign id and default name, return id.
  int add_feature(Feature f);

  /// Remove feature by id. Boolean dependents referring to it keep the id
  /// (evaluation will skip broken booleans). Returns true if found.
  bool remove_feature(int id);

  [[nodiscard]] Feature* find(int id);
  [[nodiscard]] const Feature* find(int id) const;

  /// Evaluate a single feature (including its transform). Returns nullopt if
  /// the feature is missing, suppressed, or a boolean with invalid operands.
  [[nodiscard]] std::optional<Mesh> evaluate_feature(int id) const;

  /// Evaluate all non-suppressed, visible tip features that are not used as
  /// pure operands of a later boolean. For simplicity in turn 1: union of
  /// every feature that is not exclusively an intermediate boolean operand.
  /// Practically: evaluate every top-level feature (all of them) and keep the
  /// last solid / or merge all leaf results for display.
  ///
  /// Display strategy: return the evaluated mesh of every feature that is not
  /// referenced as an operand by a later boolean; append them together.
  [[nodiscard]] Mesh evaluate_display() const;

  /// Map of feature id → evaluated mesh (transformed). Empty if eval failed.
  [[nodiscard]] std::unordered_map<int, Mesh> evaluate_all() const;

  void clear();

  /// Insert Front / Top / Right planes as fixed tree roots (ids 1,2,3 when empty).
  /// Idempotent if planes already present. Ready for future "sketch on plane".
  void seed_reference_planes();

  /// After bulk-loading features with explicit ids, ensure next id is unique.
  void ensure_next_id_after(int max_id);

 private:
  std::vector<Feature> features_;
  int next_id_ = 1;
  int selected_id_ = -1;
  std::string name_ = "Untitled";
};

}  // namespace cad
