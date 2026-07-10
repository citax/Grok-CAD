#pragma once

#include "cadcore/document/document.hpp"

#include <QWidget>

class QDoubleSpinBox;
class QFormLayout;
class QLabel;
class QCheckBox;
class QSpinBox;

namespace app {

class PropertyPanel : public QWidget {
  Q_OBJECT
 public:
  explicit PropertyPanel(QWidget* parent = nullptr);

  void set_document(cad::Document* doc);
  void show_feature(int id);

 signals:
  void feature_changed(int id);

 private:
  void rebuild_form();
  void apply_from_widgets();
  void block_signals(bool block);

  cad::Document* doc_ = nullptr;
  int current_id_ = -1;

  QLabel* title_ = nullptr;
  QFormLayout* form_ = nullptr;
  QWidget* fields_host_ = nullptr;

  QDoubleSpinBox* w_ = nullptr;
  QDoubleSpinBox* h_ = nullptr;
  QDoubleSpinBox* d_ = nullptr;
  QDoubleSpinBox* r_ = nullptr;
  QDoubleSpinBox* tx_ = nullptr;
  QDoubleSpinBox* ty_ = nullptr;
  QDoubleSpinBox* tz_ = nullptr;
  QDoubleSpinBox* rx_ = nullptr;
  QDoubleSpinBox* ry_ = nullptr;
  QDoubleSpinBox* rz_ = nullptr;
  QDoubleSpinBox* sx_ = nullptr;
  QDoubleSpinBox* sy_ = nullptr;
  QDoubleSpinBox* sz_ = nullptr;
  QCheckBox* visible_ = nullptr;
  QCheckBox* suppressed_ = nullptr;
};

}  // namespace app
