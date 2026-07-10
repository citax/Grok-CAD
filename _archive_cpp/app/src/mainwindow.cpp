#include "mainwindow.hpp"

#include "cadcore/document/json_io.hpp"
#include "cadcore/mesh/stl.hpp"

#include <QAction>
#include <QDockWidget>
#include <QFileDialog>
#include <QItemSelectionModel>
#include <QMenuBar>
#include <QMessageBox>
#include <QStatusBar>
#include <QToolBar>
#include <QTreeView>

namespace app {

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
  setWindowTitle(tr("Grok CAD — Reference Planes"));
  resize(1280, 800);
  setup_ui();
  setup_menus();
  setup_toolbar();
  seed_document();
  statusBar()->showMessage(tr("Ready — select a reference plane to begin"));
}

void MainWindow::seed_document() {
  document_.seed_reference_planes();
  // Select Front Plane by default
  for (const auto& f : document_.features()) {
    if (f.type == cad::FeatureType::PlaneFront) {
      document_.set_selected_id(f.id);
      break;
    }
  }
  refresh_all();
  show_selection_status(document_.selected_id());
}

void MainWindow::setup_ui() {
  viewport_ = new Viewport(this);
  setCentralWidget(viewport_);
  viewport_->set_document(&document_);

  tree_model_ = new FeatureTreeModel(&document_, this);
  tree_view_ = new QTreeView(this);
  tree_view_->setModel(tree_model_);
  tree_view_->setHeaderHidden(false);
  tree_view_->setSelectionMode(QAbstractItemView::SingleSelection);
  tree_view_->setUniformRowHeights(true);

  auto* left_dock = new QDockWidget(tr("Feature Tree"), this);
  left_dock->setObjectName(QStringLiteral("FeatureTreeDock"));
  left_dock->setWidget(tree_view_);
  addDockWidget(Qt::LeftDockWidgetArea, left_dock);

  props_ = new PropertyPanel(this);
  props_->set_document(&document_);
  auto* right_dock = new QDockWidget(tr("Properties"), this);
  right_dock->setObjectName(QStringLiteral("PropertiesDock"));
  right_dock->setWidget(props_);
  addDockWidget(Qt::RightDockWidgetArea, right_dock);

  connect(tree_view_->selectionModel(), &QItemSelectionModel::selectionChanged, this,
          &MainWindow::on_tree_selection_changed);
  connect(viewport_, &Viewport::feature_picked, this, &MainWindow::on_feature_picked);
  connect(viewport_, &Viewport::status_message, this, &MainWindow::on_status_message);
  connect(props_, &PropertyPanel::feature_changed, this, &MainWindow::on_feature_changed);
}

void MainWindow::setup_menus() {
  auto* file = menuBar()->addMenu(tr("&File"));
  file->addAction(tr("&New"), QKeySequence::New, this, &MainWindow::new_document);
  file->addAction(tr("&Open…"), QKeySequence::Open, this, &MainWindow::open_document);
  file->addAction(tr("&Save"), QKeySequence::Save, this, &MainWindow::save_document);
  file->addAction(tr("Export &STL…"), this, &MainWindow::export_stl);
  file->addSeparator();
  file->addAction(tr("E&xit"), QKeySequence::Quit, this, &QWidget::close);

  auto* edit = menuBar()->addMenu(tr("&Edit"));
  edit->addAction(tr("&Delete Feature"), QKeySequence::Delete, this, &MainWindow::delete_selected);

  auto* view = menuBar()->addMenu(tr("&View"));
  view->addAction(tr("&Front"), this, [this] {
    viewport_->set_standard_view(StandardView::Front);
    statusBar()->showMessage(tr("View: Front"), 2000);
  });
  view->addAction(tr("&Back"), this, [this] {
    viewport_->set_standard_view(StandardView::Back);
    statusBar()->showMessage(tr("View: Back"), 2000);
  });
  view->addAction(tr("&Top"), this, [this] {
    viewport_->set_standard_view(StandardView::Top);
    statusBar()->showMessage(tr("View: Top"), 2000);
  });
  view->addAction(tr("Botto&m"), this, [this] {
    viewport_->set_standard_view(StandardView::Bottom);
    statusBar()->showMessage(tr("View: Bottom"), 2000);
  });
  view->addAction(tr("&Right"), this, [this] {
    viewport_->set_standard_view(StandardView::Right);
    statusBar()->showMessage(tr("View: Right"), 2000);
  });
  view->addAction(tr("&Left"), this, [this] {
    viewport_->set_standard_view(StandardView::Left);
    statusBar()->showMessage(tr("View: Left"), 2000);
  });
  view->addAction(tr("&Isometric"), QKeySequence(tr("Ctrl+1")), this, [this] {
    viewport_->set_standard_view(StandardView::Isometric);
    statusBar()->showMessage(tr("View: Isometric"), 2000);
  });
  view->addSeparator();
  view->addAction(tr("Zoom to &Fit"), QKeySequence(tr("Ctrl+F")), this, [this] {
    viewport_->zoom_to_fit();
  });
  view->addAction(tr("Rebuild Geometry"), this, [this] { refresh_all(); });

  // Extension point for sketch mode (next turn)
  auto* insert = menuBar()->addMenu(tr("&Insert"));
  insert->addAction(tr("Sketch on Plane…"), this, [this] {
    const cad::Feature* f = document_.find(document_.selected_id());
    if (!f || !cad::is_reference_plane(f->type)) {
      QMessageBox::information(this, tr("Sketch"),
                               tr("Select a reference plane first.\n"
                                  "(Sketch mode will be implemented next.)"));
      statusBar()->showMessage(tr("Select a reference plane to start a sketch"), 4000);
      return;
    }
    statusBar()->showMessage(
        tr("Sketch mode coming next — plane: %1").arg(QString::fromStdString(f->name)), 4000);
  });
}

void MainWindow::setup_toolbar() {
  auto* views = addToolBar(tr("Views"));
  views->setObjectName(QStringLiteral("ViewsToolBar"));
  views->addAction(tr("Front"), this, [this] { viewport_->set_standard_view(StandardView::Front); });
  views->addAction(tr("Top"), this, [this] { viewport_->set_standard_view(StandardView::Top); });
  views->addAction(tr("Right"), this, [this] { viewport_->set_standard_view(StandardView::Right); });
  views->addAction(tr("Iso"), this, [this] { viewport_->set_standard_view(StandardView::Isometric); });
  views->addAction(tr("Fit"), this, [this] { viewport_->zoom_to_fit(); });

  auto* solids = addToolBar(tr("Solids (legacy)"));
  solids->setObjectName(QStringLiteral("SolidsToolBar"));
  solids->addAction(tr("Box"), this, &MainWindow::add_box);
  solids->addAction(tr("Sphere"), this, &MainWindow::add_sphere);
  solids->addAction(tr("Cylinder"), this, &MainWindow::add_cylinder);
  solids->addSeparator();
  solids->addAction(tr("Union"), this, &MainWindow::add_boolean_union);
  solids->addAction(tr("Difference"), this, &MainWindow::add_boolean_difference);
  solids->addAction(tr("Intersect"), this, &MainWindow::add_boolean_intersection);
}

void MainWindow::show_selection_status(int id) {
  const cad::Feature* f = document_.find(id);
  if (!f) {
    statusBar()->showMessage(tr("Selected: (none)"));
    return;
  }
  statusBar()->showMessage(tr("Selected: %1").arg(QString::fromStdString(f->name)));
}

void MainWindow::on_status_message(const QString& msg) {
  statusBar()->showMessage(msg, 2500);
}

void MainWindow::refresh_all() {
  tree_model_->refresh();
  viewport_->rebuild_geometry();
  props_->show_feature(document_.selected_id());
  sync_selection(document_.selected_id());
}

void MainWindow::sync_selection(int id) {
  document_.set_selected_id(id);
  viewport_->set_selected_id(id);
  props_->show_feature(id);
  show_selection_status(id);
  if (id >= 0) {
    const QModelIndex idx = tree_model_->index_for_feature(id);
    if (idx.isValid()) {
      tree_view_->selectionModel()->blockSignals(true);
      tree_view_->selectionModel()->select(idx, QItemSelectionModel::ClearAndSelect |
                                                    QItemSelectionModel::Rows);
      tree_view_->selectionModel()->blockSignals(false);
    }
  }
}

void MainWindow::new_document() {
  document_.clear();
  current_path_.clear();
  setWindowTitle(tr("Grok CAD — Untitled"));
  seed_document();
  statusBar()->showMessage(tr("New document"), 2000);
}

void MainWindow::open_document() {
  const QString path = QFileDialog::getOpenFileName(this, tr("Open Document"), QString(),
                                                    tr("CAD Document (*.cad.json *.json)"));
  if (path.isEmpty()) return;
  cad::Document loaded;
  std::string err;
  if (!cad::load_document(path.toStdString(), loaded, &err)) {
    QMessageBox::warning(this, tr("Open Failed"), QString::fromStdString(err));
    return;
  }
  loaded.seed_reference_planes();
  document_ = std::move(loaded);
  current_path_ = path;
  setWindowTitle(tr("Grok CAD — %1").arg(path));
  tree_model_->set_document(&document_);
  props_->set_document(&document_);
  viewport_->set_document(&document_);
  refresh_all();
  statusBar()->showMessage(tr("Opened %1").arg(path), 3000);
}

void MainWindow::save_document() {
  QString path = current_path_;
  if (path.isEmpty()) {
    path = QFileDialog::getSaveFileName(this, tr("Save Document"),
                                        QStringLiteral("untitled.cad.json"),
                                        tr("CAD Document (*.cad.json *.json)"));
    if (path.isEmpty()) return;
  }
  if (!cad::save_document(document_, path.toStdString())) {
    QMessageBox::warning(this, tr("Save Failed"), tr("Could not write file."));
    return;
  }
  current_path_ = path;
  setWindowTitle(tr("Grok CAD — %1").arg(path));
  statusBar()->showMessage(tr("Saved %1").arg(path), 3000);
}

void MainWindow::export_stl() {
  const QString path = QFileDialog::getSaveFileName(this, tr("Export STL"),
                                                    QStringLiteral("model.stl"),
                                                    tr("STL Binary (*.stl)"));
  if (path.isEmpty()) return;
  const cad::Mesh mesh = document_.evaluate_display();
  if (mesh.empty()) {
    QMessageBox::information(this, tr("Export STL"),
                             tr("No solid geometry to export (reference planes only)."));
    return;
  }
  if (!cad::write_stl_file(mesh, path.toStdString())) {
    QMessageBox::warning(this, tr("Export Failed"), tr("Could not write STL."));
    return;
  }
  statusBar()->showMessage(tr("Exported %1").arg(path), 3000);
}

void MainWindow::add_box() {
  cad::Feature f;
  f.type = cad::FeatureType::Box;
  f.width = 1.0;
  f.height = 1.0;
  f.depth = 1.0;
  document_.add_feature(std::move(f));
  refresh_all();
  statusBar()->showMessage(tr("Created Box"), 2000);
}

void MainWindow::add_sphere() {
  cad::Feature f;
  f.type = cad::FeatureType::Sphere;
  f.radius = 0.5;
  f.translation = {1.5, 0, 0};
  document_.add_feature(std::move(f));
  refresh_all();
  statusBar()->showMessage(tr("Created Sphere"), 2000);
}

void MainWindow::add_cylinder() {
  cad::Feature f;
  f.type = cad::FeatureType::Cylinder;
  f.radius = 0.4;
  f.height = 1.2;
  f.translation = {-1.5, 0, 0};
  document_.add_feature(std::move(f));
  refresh_all();
  statusBar()->showMessage(tr("Created Cylinder"), 2000);
}

std::pair<int, int> MainWindow::pick_two_operands() const {
  const auto& feats = document_.features();
  std::vector<int> solids;
  for (const auto& f : feats) {
    if (!cad::is_reference_plane(f.type) && !cad::is_boolean(f.type)) {
      solids.push_back(f.id);
    } else if (cad::is_boolean(f.type)) {
      solids.push_back(f.id);
    }
  }
  // Prefer non-plane features
  solids.clear();
  for (const auto& f : feats) {
    if (!cad::is_reference_plane(f.type)) {
      solids.push_back(f.id);
    }
  }
  if (solids.size() < 2) {
    return {-1, -1};
  }
  const int sel = document_.selected_id();
  int a = -1;
  int b = -1;
  if (sel >= 0 && document_.find(sel) && !cad::is_reference_plane(document_.find(sel)->type)) {
    a = sel;
    for (auto it = solids.rbegin(); it != solids.rend(); ++it) {
      if (*it != sel) {
        b = *it;
        break;
      }
    }
  } else {
    a = solids[solids.size() - 2];
    b = solids[solids.size() - 1];
  }
  return {a, b};
}

void MainWindow::add_boolean_union() {
  auto [a, b] = pick_two_operands();
  if (a < 0 || b < 0) {
    QMessageBox::information(this, tr("Boolean"), tr("Need at least two solid features."));
    return;
  }
  cad::Feature f;
  f.type = cad::FeatureType::BooleanUnion;
  f.operand_a = a;
  f.operand_b = b;
  document_.add_feature(std::move(f));
  refresh_all();
  statusBar()->showMessage(tr("Created Union"), 2000);
}

void MainWindow::add_boolean_difference() {
  auto [a, b] = pick_two_operands();
  if (a < 0 || b < 0) {
    QMessageBox::information(this, tr("Boolean"), tr("Need at least two solid features."));
    return;
  }
  cad::Feature f;
  f.type = cad::FeatureType::BooleanDifference;
  f.operand_a = a;
  f.operand_b = b;
  document_.add_feature(std::move(f));
  refresh_all();
  statusBar()->showMessage(tr("Created Difference"), 2000);
}

void MainWindow::add_boolean_intersection() {
  auto [a, b] = pick_two_operands();
  if (a < 0 || b < 0) {
    QMessageBox::information(this, tr("Boolean"), tr("Need at least two solid features."));
    return;
  }
  cad::Feature f;
  f.type = cad::FeatureType::BooleanIntersection;
  f.operand_a = a;
  f.operand_b = b;
  document_.add_feature(std::move(f));
  refresh_all();
  statusBar()->showMessage(tr("Created Intersection"), 2000);
}

void MainWindow::on_tree_selection_changed() {
  const auto indexes = tree_view_->selectionModel()->selectedIndexes();
  if (indexes.isEmpty()) return;
  const int id = tree_model_->feature_id_at(indexes.first());
  document_.set_selected_id(id);
  viewport_->set_selected_id(id);
  props_->show_feature(id);
  show_selection_status(id);
}

void MainWindow::on_feature_picked(int id) {
  sync_selection(id);
}

void MainWindow::on_feature_changed(int) {
  viewport_->rebuild_geometry();
  tree_model_->refresh();
  sync_selection(document_.selected_id());
  statusBar()->showMessage(tr("Feature updated"), 1500);
}

void MainWindow::delete_selected() {
  const int id = document_.selected_id();
  if (id < 0) return;
  const cad::Feature* f = document_.find(id);
  if (f && cad::is_reference_plane(f->type)) {
    QMessageBox::information(this, tr("Delete"),
                             tr("Reference planes cannot be deleted."));
    statusBar()->showMessage(tr("Cannot delete reference planes"), 3000);
    return;
  }
  if (!document_.remove_feature(id)) {
    statusBar()->showMessage(tr("Delete failed"), 2000);
    return;
  }
  refresh_all();
  statusBar()->showMessage(tr("Feature deleted"), 2000);
}

}  // namespace app
