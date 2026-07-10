#include "feature_tree_model.hpp"

namespace app {

FeatureTreeModel::FeatureTreeModel(cad::Document* doc, QObject* parent)
    : QAbstractItemModel(parent), doc_(doc) {}

void FeatureTreeModel::set_document(cad::Document* doc) {
  beginResetModel();
  doc_ = doc;
  endResetModel();
}

void FeatureTreeModel::refresh() {
  beginResetModel();
  endResetModel();
}

QModelIndex FeatureTreeModel::index(int row, int column, const QModelIndex& parent) const {
  if (!doc_ || parent.isValid() || column != 0 || row < 0 ||
      row >= static_cast<int>(doc_->features().size())) {
    return {};
  }
  return createIndex(row, column, static_cast<quintptr>(doc_->features()[static_cast<size_t>(row)].id));
}

QModelIndex FeatureTreeModel::parent(const QModelIndex&) const { return {}; }

int FeatureTreeModel::rowCount(const QModelIndex& parent) const {
  if (!doc_ || parent.isValid()) {
    return 0;
  }
  return static_cast<int>(doc_->features().size());
}

int FeatureTreeModel::columnCount(const QModelIndex&) const { return 1; }

QVariant FeatureTreeModel::data(const QModelIndex& index, int role) const {
  if (!doc_ || !index.isValid()) {
    return {};
  }
  const auto& feats = doc_->features();
  if (index.row() < 0 || index.row() >= static_cast<int>(feats.size())) {
    return {};
  }
  const auto& f = feats[static_cast<size_t>(index.row())];
  if (role == Qt::DisplayRole || role == Qt::EditRole) {
    return QString::fromStdString(f.name);
  }
  if (role == Qt::ToolTipRole) {
    return QString("%1 (id=%2)").arg(QString::fromStdString(cad::feature_type_name(f.type))).arg(f.id);
  }
  return {};
}

QVariant FeatureTreeModel::headerData(int section, Qt::Orientation orientation, int role) const {
  if (orientation == Qt::Horizontal && role == Qt::DisplayRole && section == 0) {
    return QStringLiteral("Features");
  }
  return {};
}

Qt::ItemFlags FeatureTreeModel::flags(const QModelIndex& index) const {
  if (!index.isValid()) {
    return Qt::NoItemFlags;
  }
  return Qt::ItemIsEnabled | Qt::ItemIsSelectable;
}

int FeatureTreeModel::feature_id_at(const QModelIndex& index) const {
  if (!index.isValid()) {
    return -1;
  }
  return static_cast<int>(index.internalId());
}

QModelIndex FeatureTreeModel::index_for_feature(int id) const {
  if (!doc_) {
    return {};
  }
  const auto& feats = doc_->features();
  for (int i = 0; i < static_cast<int>(feats.size()); ++i) {
    if (feats[static_cast<size_t>(i)].id == id) {
      return index(i, 0);
    }
  }
  return {};
}

}  // namespace app
