#include "mainwindow.hpp"

#include <QApplication>
#include <QSurfaceFormat>

int main(int argc, char* argv[]) {
  QSurfaceFormat fmt;
  fmt.setVersion(3, 3);
  fmt.setProfile(QSurfaceFormat::CoreProfile);
  fmt.setDepthBufferSize(24);
  fmt.setSamples(4);
  QSurfaceFormat::setDefaultFormat(fmt);

  QApplication app(argc, argv);
  QApplication::setApplicationName(QStringLiteral("Grok CAD"));
  QApplication::setOrganizationName(QStringLiteral("CadCore"));

  app::MainWindow window;
  window.show();
  return app.exec();
}
