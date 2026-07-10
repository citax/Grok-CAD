#pragma once

#include "cadcore/document/document.hpp"

#include <QAbstractItemModel>

namespace app {

class FeatureTreeModel : public QAbstractItemModel {
  Q_OBJECT
 public:
  explicit FeatureTreeModel(cad::Document* doc, QObject* parent = nullptr);

  void set_document(cad::Document* doc);
  void refresh();

  [[nodiscard]] QModelIndex index(int row, int column,
                                  const QModelIndex& parent = {}) const override;
  [[nodiscard]] QModelIndex parent(const QModelIndex& child) const override;
  [[nodiscard]] int rowCount(const QModelIndex& parent = {}) const override;
  [[nodiscard]] int columnCount(const QModelIndex& parent = {}) const override;
  [[nodiscard]] QVariant data(const QModelIndex& index, int role) const override;
  [[nodiscard]] QVariant headerData(int section, Qt::Orientation orientation,
                                    int role) const override;
  [[nodiscard]] Qt::ItemFlags flags(const QModelIndex& index) const override;

  [[nodiscard]] int feature_id_at(const QModelIndex& index) const;
  [[nodiscard]] QModelIndex index_for_feature(int id) const;

 private:
  cad::Document* doc_ = nullptr;
};

}  // namespace app
