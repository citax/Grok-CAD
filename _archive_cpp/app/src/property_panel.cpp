#include "property_panel.hpp"

#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QLabel>
#include <QVBoxLayout>

namespace app {
namespace {

QDoubleSpinBox* make_spin(double minv, double maxv, double step, QWidget* parent) {
  auto* s = new QDoubleSpinBox(parent);
  s->setRange(minv, maxv);
  s->setDecimals(4);
  s->setSingleStep(step);
  s->setKeyboardTracking(false);
  return s;
}

}  // namespace

PropertyPanel::PropertyPanel(QWidget* parent) : QWidget(parent) {
  auto* layout = new QVBoxLayout(this);
  title_ = new QLabel(tr("No selection"), this);
  title_->setStyleSheet(QStringLiteral("font-weight: bold;"));
  layout->addWidget(title_);

  fields_host_ = new QWidget(this);
  form_ = new QFormLayout(fields_host_);
  layout->addWidget(fields_host_);
  layout->addStretch(1);

  auto connect_spin = [this](QDoubleSpinBox* s) {
    connect(s, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            [this](double) { apply_from_widgets(); });
  };

  w_ = make_spin(0.001, 1e6, 0.1, fields_host_);
  h_ = make_spin(0.001, 1e6, 0.1, fields_host_);
  d_ = make_spin(0.001, 1e6, 0.1, fields_host_);
  r_ = make_spin(0.001, 1e6, 0.1, fields_host_);
  tx_ = make_spin(-1e6, 1e6, 0.1, fields_host_);
  ty_ = make_spin(-1e6, 1e6, 0.1, fields_host_);
  tz_ = make_spin(-1e6, 1e6, 0.1, fields_host_);
  rx_ = make_spin(-3600, 3600, 5.0, fields_host_);
  ry_ = make_spin(-3600, 3600, 5.0, fields_host_);
  rz_ = make_spin(-3600, 3600, 5.0, fields_host_);
  sx_ = make_spin(0.001, 1e3, 0.1, fields_host_);
  sy_ = make_spin(0.001, 1e3, 0.1, fields_host_);
  sz_ = make_spin(0.001, 1e3, 0.1, fields_host_);
  visible_ = new QCheckBox(tr("Visible"), fields_host_);
  suppressed_ = new QCheckBox(tr("Suppressed"), fields_host_);

  form_->addRow(tr("Width"), w_);
  form_->addRow(tr("Height"), h_);
  form_->addRow(tr("Depth"), d_);
  form_->addRow(tr("Radius"), r_);
  form_->addRow(tr("Translate X"), tx_);
  form_->addRow(tr("Translate Y"), ty_);
  form_->addRow(tr("Translate Z"), tz_);
  form_->addRow(tr("Rotate X°"), rx_);
  form_->addRow(tr("Rotate Y°"), ry_);
  form_->addRow(tr("Rotate Z°"), rz_);
  form_->addRow(tr("Scale X"), sx_);
  form_->addRow(tr("Scale Y"), sy_);
  form_->addRow(tr("Scale Z"), sz_);
  form_->addRow(QString(), visible_);
  form_->addRow(QString(), suppressed_);

  for (auto* s : {w_, h_, d_, r_, tx_, ty_, tz_, rx_, ry_, rz_, sx_, sy_, sz_}) {
    connect_spin(s);
  }
  connect(visible_, &QCheckBox::toggled, this, [this](bool) { apply_from_widgets(); });
  connect(suppressed_, &QCheckBox::toggled, this, [this](bool) { apply_from_widgets(); });
}

void PropertyPanel::set_document(cad::Document* doc) {
  doc_ = doc;
  show_feature(-1);
}

void PropertyPanel::block_signals(bool block) {
  for (auto* s : {w_, h_, d_, r_, tx_, ty_, tz_, rx_, ry_, rz_, sx_, sy_, sz_}) {
    s->blockSignals(block);
  }
  visible_->blockSignals(block);
  suppressed_->blockSignals(block);
}

void PropertyPanel::show_feature(int id) {
  current_id_ = id;
  if (!doc_ || id < 0) {
    title_->setText(tr("No selection"));
    fields_host_->setEnabled(false);
    return;
  }
  const cad::Feature* f = doc_->find(id);
  if (!f) {
    title_->setText(tr("No selection"));
    fields_host_->setEnabled(false);
    return;
  }
  fields_host_->setEnabled(true);
  title_->setText(QString::fromStdString(f->name));

  block_signals(true);
  w_->setValue(f->width);
  h_->setValue(f->height);
  d_->setValue(f->depth);
  r_->setValue(f->radius);
  tx_->setValue(f->translation.x);
  ty_->setValue(f->translation.y);
  tz_->setValue(f->translation.z);
  rx_->setValue(f->rotation_deg.x);
  ry_->setValue(f->rotation_deg.y);
  rz_->setValue(f->rotation_deg.z);
  sx_->setValue(f->scale.x);
  sy_->setValue(f->scale.y);
  sz_->setValue(f->scale.z);
  visible_->setChecked(f->visible);
  suppressed_->setChecked(f->suppressed);

  const bool is_plane = cad::is_reference_plane(f->type);
  const bool is_box = f->type == cad::FeatureType::Box;
  const bool is_sph = f->type == cad::FeatureType::Sphere;
  const bool is_cyl = f->type == cad::FeatureType::Cylinder;
  w_->setEnabled(is_box);
  h_->setEnabled(is_box || is_cyl);
  d_->setEnabled(is_box);
  r_->setEnabled(is_sph || is_cyl);
  // Planes are fixed roots — only visibility is editable for now
  for (auto* s : {tx_, ty_, tz_, rx_, ry_, rz_, sx_, sy_, sz_}) {
    s->setEnabled(!is_plane);
  }
  suppressed_->setEnabled(!is_plane);
  block_signals(false);
}

void PropertyPanel::apply_from_widgets() {
  if (!doc_ || current_id_ < 0) {
    return;
  }
  cad::Feature* f = doc_->find(current_id_);
  if (!f) {
    return;
  }
  f->width = w_->value();
  f->height = h_->value();
  f->depth = d_->value();
  f->radius = r_->value();
  f->translation = {tx_->value(), ty_->value(), tz_->value()};
  f->rotation_deg = {rx_->value(), ry_->value(), rz_->value()};
  f->scale = {sx_->value(), sy_->value(), sz_->value()};
  f->visible = visible_->isChecked();
  f->suppressed = suppressed_->isChecked();
  emit feature_changed(current_id_);
}

}  // namespace app
