#pragma once

#include "camera.hpp"
#include "cadcore/document/document.hpp"
#include "cadcore/mesh/mesh.hpp"

#include <QOpenGLFunctions_3_3_Core>
#include <QOpenGLWidget>
#include <QMatrix4x4>
#include <QString>

#include <vector>

namespace app {

class Viewport : public QOpenGLWidget, protected QOpenGLFunctions_3_3_Core {
  Q_OBJECT
 public:
  explicit Viewport(QWidget* parent = nullptr);
  ~Viewport() override;

  void set_document(cad::Document* doc);
  void rebuild_geometry();
  void set_selected_id(int id);

  void set_standard_view(StandardView v);
  void zoom_to_fit();

  [[nodiscard]] Camera& camera() noexcept { return camera_; }
  [[nodiscard]] const Camera& camera() const noexcept { return camera_; }
  [[nodiscard]] bool gl_ok() const noexcept { return gl_ok_; }

 signals:
  void feature_picked(int id);
  void status_message(const QString& msg);

 protected:
  void initializeGL() override;
  void resizeGL(int w, int h) override;
  void paintGL() override;
  void paintEvent(QPaintEvent* event) override;
  void mousePressEvent(QMouseEvent* e) override;
  void mouseMoveEvent(QMouseEvent* e) override;
  void mouseReleaseEvent(QMouseEvent* e) override;
  void wheelEvent(QWheelEvent* e) override;

 private:
  struct GpuMesh {
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint nbo = 0;
    GLuint ebo = 0;
    GLsizei index_count = 0;
    int feature_id = -1;
    cad::Vec3 color{0.7, 0.75, 0.8};
    bool translucent = false;
  };

  void ensure_programs();
  GLuint compile_shader(GLenum type, const char* src);
  GLuint link_program(GLuint vs, GLuint fs);
  void upload_mesh(GpuMesh& gpu, const cad::Mesh& mesh);
  void upload_colored_lines(GpuMesh& gpu, const std::vector<float>& interleaved_pos_color,
                            GLenum mode);
  void destroy_mesh(GpuMesh& gpu);
  void build_scene_helpers();
  void build_reference_planes();
  void draw_mesh(const GpuMesh& gpu, const QMatrix4x4& mvp, const QMatrix4x4& model,
                 bool selected, float alpha);
  void draw_lines(GLuint vao, GLsizei count, const QMatrix4x4& mvp, GLenum mode = GL_LINES);
  void draw_corner_gizmo(int fb_w, int fb_h);
  int pick_feature(const QPoint& pos);
  [[nodiscard]] double aspect() const;
  [[nodiscard]] int framebuffer_w() const;
  [[nodiscard]] int framebuffer_h() const;

  static QMatrix4x4 to_qmat(const cad::Mat4& m);

  cad::Document* doc_ = nullptr;
  Camera camera_;
  int selected_id_ = -1;
  double aspect_ = 1.0;
  int widget_w_ = 1;
  int widget_h_ = 1;

  GLuint mesh_prog_ = 0;
  GLuint line_prog_ = 0;

  std::vector<GpuMesh> solid_meshes_;
  std::vector<GpuMesh> plane_fills_;   // translucent quads
  std::vector<GpuMesh> plane_edges_;   // borders
  GpuMesh grid_;
  GpuMesh axes_;
  GpuMesh origin_marker_;
  GpuMesh gizmo_axes_;

  QPoint last_pos_;
  QPoint press_pos_;
  bool rotating_ = false;
  bool panning_ = false;
  bool gl_ready_ = false;
  bool gl_ok_ = false;
  QString gl_error_;
};

}  // namespace app
