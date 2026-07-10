#pragma once

#include "cadcore/document/document.hpp"
#include "feature_tree_model.hpp"
#include "property_panel.hpp"
#include "viewport.hpp"

#include <QMainWindow>

class QTreeView;

namespace app {

class MainWindow : public QMainWindow {
  Q_OBJECT
 public:
  explicit MainWindow(QWidget* parent = nullptr);

 private slots:
  void new_document();
  void open_document();
  void save_document();
  void export_stl();
  void add_box();
  void add_sphere();
  void add_cylinder();
  void add_boolean_union();
  void add_boolean_difference();
  void add_boolean_intersection();
  void on_tree_selection_changed();
  void on_feature_picked(int id);
  void on_feature_changed(int id);
  void delete_selected();
  void on_status_message(const QString& msg);

 private:
  void setup_ui();
  void setup_menus();
  void setup_toolbar();
  void sync_selection(int id);
  void refresh_all();
  void seed_document();
  [[nodiscard]] std::pair<int, int> pick_two_operands() const;
  void show_selection_status(int id);

  cad::Document document_;
  Viewport* viewport_ = nullptr;
  QTreeView* tree_view_ = nullptr;
  FeatureTreeModel* tree_model_ = nullptr;
  PropertyPanel* props_ = nullptr;
  QString current_path_;
};

}  // namespace app
