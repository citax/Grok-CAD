#include "viewport.hpp"

#include <QCoreApplication>
#include <QFile>
#include <QMouseEvent>
#include <QPainter>
#include <QPaintEvent>
#include <QWheelEvent>
#include <QtMath>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <unordered_map>
#include <vector>

namespace app {
namespace {

std::string load_shader_file(const QString& path) {
  QFile f(path);
  if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
    return {};
  }
  return f.readAll().toStdString();
}

cad::Vec3 color_for_id(int id) {
  const float h = std::fmod(static_cast<float>(id) * 0.61803398875f, 1.0f);
  const float s = 0.45f;
  const float v = 0.85f;
  const float c = v * s;
  const float x = c * (1.0f - std::fabs(std::fmod(h * 6.0f, 2.0f) - 1.0f));
  const float m = v - c;
  float r = 0, g = 0, b = 0;
  switch (static_cast<int>(h * 6.0f) % 6) {
    case 0: r = c; g = x; break;
    case 1: r = x; g = c; break;
    case 2: g = c; b = x; break;
    case 3: g = x; b = c; break;
    case 4: r = x; b = c; break;
    default: r = c; b = x; break;
  }
  return {r + m, g + m, b + m};
}

cad::Mesh make_plane_quad(cad::FeatureType type, double half) {
  // Two triangles, CCW when viewed along +normal
  cad::Mesh m;
  const float h = static_cast<float>(half);
  if (type == cad::FeatureType::PlaneFront) {
    // XY plane, z=0
    m.positions = {{-h, -h, 0}, {h, -h, 0}, {h, h, 0}, {-h, h, 0}};
    m.normals = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}};
  } else if (type == cad::FeatureType::PlaneTop) {
    // XZ plane, y=0
    m.positions = {{-h, 0, -h}, {h, 0, -h}, {h, 0, h}, {-h, 0, h}};
    m.normals = {{0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}};
  } else {
    // YZ plane, x=0
    m.positions = {{0, -h, -h}, {0, -h, h}, {0, h, h}, {0, h, -h}};
    m.normals = {{1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}};
  }
  m.indices = {0, 1, 2, 0, 2, 3};
  return m;
}

cad::Vec3 plane_tint(cad::FeatureType t) {
  switch (t) {
    case cad::FeatureType::PlaneFront:
      return {0.35, 0.55, 0.95};  // blue-ish
    case cad::FeatureType::PlaneTop:
      return {0.45, 0.85, 0.50};  // green-ish
    case cad::FeatureType::PlaneRight:
      return {0.95, 0.45, 0.40};  // red-ish
    default:
      return {0.7, 0.7, 0.7};
  }
}

}  // namespace

Viewport::Viewport(QWidget* parent) : QOpenGLWidget(parent) {
  setMinimumSize(320, 240);
  setFocusPolicy(Qt::StrongFocus);
  setMouseTracking(true);
  // Request 3.3 core (also set in main.cpp)
  QSurfaceFormat fmt;
  fmt.setVersion(3, 3);
  fmt.setProfile(QSurfaceFormat::CoreProfile);
  fmt.setDepthBufferSize(24);
  fmt.setSamples(4);
  setFormat(fmt);
}

Viewport::~Viewport() {
  if (!gl_ready_) {
    return;
  }
  makeCurrent();
  for (auto& m : solid_meshes_) destroy_mesh(m);
  for (auto& m : plane_fills_) destroy_mesh(m);
  for (auto& m : plane_edges_) destroy_mesh(m);
  destroy_mesh(grid_);
  destroy_mesh(axes_);
  destroy_mesh(origin_marker_);
  destroy_mesh(gizmo_axes_);
  if (mesh_prog_) glDeleteProgram(mesh_prog_);
  if (line_prog_) glDeleteProgram(line_prog_);
  doneCurrent();
}

void Viewport::set_document(cad::Document* doc) {
  doc_ = doc;
  if (gl_ready_ && gl_ok_) {
    rebuild_geometry();
  }
}

void Viewport::set_selected_id(int id) {
  selected_id_ = id;
  update();
}

void Viewport::set_standard_view(StandardView v) {
  camera_.set_standard_view(v);
  update();
  emit status_message(tr("View changed"));
}

void Viewport::zoom_to_fit() {
  cad::Vec3 mn{-2, -2, -2};
  cad::Vec3 mx{2, 2, 2};
  if (doc_) {
    bool any = false;
    cad::Vec3 bmin{1e9, 1e9, 1e9};
    cad::Vec3 bmax{-1e9, -1e9, -1e9};
    for (const auto& f : doc_->features()) {
      if (cad::is_reference_plane(f.type) || !f.visible) continue;
      auto mesh = doc_->evaluate_feature(f.id);
      if (!mesh || mesh->empty()) continue;
      auto [a, b] = mesh->bounds();
      bmin.x = std::min(bmin.x, a.x);
      bmin.y = std::min(bmin.y, a.y);
      bmin.z = std::min(bmin.z, a.z);
      bmax.x = std::max(bmax.x, b.x);
      bmax.y = std::max(bmax.y, b.y);
      bmax.z = std::max(bmax.z, b.z);
      any = true;
    }
    if (any) {
      mn = bmin;
      mx = bmax;
    }
  }
  camera_.zoom_to_fit(mn, mx);
  update();
  emit status_message(tr("Zoom to fit"));
}

QMatrix4x4 Viewport::to_qmat(const cad::Mat4& m) {
  float data[16];
  for (int i = 0; i < 16; ++i) {
    data[i] = static_cast<float>(m.m[static_cast<size_t>(i)]);
  }
  return QMatrix4x4(data);
}

int Viewport::framebuffer_w() const {
  return std::max(1, static_cast<int>(std::lround(widget_w_ * devicePixelRatioF())));
}
int Viewport::framebuffer_h() const {
  return std::max(1, static_cast<int>(std::lround(widget_h_ * devicePixelRatioF())));
}
double Viewport::aspect() const {
  return aspect_ > 1e-6 ? aspect_ : 1.0;
}

void Viewport::initializeGL() {
  initializeOpenGLFunctions();
  gl_ready_ = true;

  const char* vendor = reinterpret_cast<const char*>(glGetString(GL_VENDOR));
  const char* renderer = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
  const char* version = reinterpret_cast<const char*>(glGetString(GL_VERSION));
  std::fprintf(stderr, "[viewport] GL_VENDOR   = %s\n", vendor ? vendor : "(null)");
  std::fprintf(stderr, "[viewport] GL_RENDERER = %s\n", renderer ? renderer : "(null)");
  std::fprintf(stderr, "[viewport] GL_VERSION  = %s\n", version ? version : "(null)");

  const QSurfaceFormat fmt = format();
  std::fprintf(stderr, "[viewport] Context    = %d.%d profile=%d\n", fmt.majorVersion(),
               fmt.minorVersion(), static_cast<int>(fmt.profile()));

  if (fmt.majorVersion() < 3 || (fmt.majorVersion() == 3 && fmt.minorVersion() < 3) ||
      fmt.profile() != QSurfaceFormat::CoreProfile) {
    gl_ok_ = false;
    gl_error_ = tr("OpenGL 3.3 Core profile was not obtained.\n"
                   "Got %1.%2. Try: LIBGL_ALWAYS_SOFTWARE=1 ./run_cad.sh")
                    .arg(fmt.majorVersion())
                    .arg(fmt.minorVersion());
    std::fprintf(stderr, "[viewport] ERROR: %s\n", gl_error_.toUtf8().constData());
    return;
  }

  gl_ok_ = true;
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glClearColor(0.16f, 0.17f, 0.20f, 1.0f);

  ensure_programs();
  build_scene_helpers();
  rebuild_geometry();
  camera_.set_standard_view(StandardView::Isometric);
}

void Viewport::resizeGL(int w, int h) {
  widget_w_ = std::max(1, w);
  widget_h_ = std::max(1, h);
  aspect_ = static_cast<double>(widget_w_) / static_cast<double>(widget_h_);
  if (gl_ok_) {
    const int fbw = framebuffer_w();
    const int fbh = framebuffer_h();
    glViewport(0, 0, fbw, fbh);
  }
}

void Viewport::paintEvent(QPaintEvent* event) {
  if (gl_ready_ && !gl_ok_) {
    QPainter painter(this);
    painter.fillRect(rect(), QColor(30, 32, 36));
    painter.setPen(QColor(240, 200, 80));
    QFont f = painter.font();
    f.setPointSize(11);
    painter.setFont(f);
    painter.drawText(rect().adjusted(24, 24, -24, -24), Qt::AlignCenter | Qt::TextWordWrap,
                     gl_error_.isEmpty()
                         ? tr("OpenGL failed to initialize.\nSee stderr for details.")
                         : gl_error_);
    return;
  }
  QOpenGLWidget::paintEvent(event);
}

void Viewport::ensure_programs() {
  const QString base = QCoreApplication::applicationDirPath() + QStringLiteral("/shaders/");
  std::string vsrc = load_shader_file(base + "mesh.vert");
  std::string fsrc = load_shader_file(base + "mesh.frag");
  if (vsrc.empty()) {
    vsrc = R"(#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
uniform mat4 uMVP;
uniform mat4 uModel;
uniform mat3 uNormalMat;
out vec3 vWorldPos;
out vec3 vNormal;
void main(){
  vec4 world=uModel*vec4(aPos,1.0);
  vWorldPos=world.xyz;
  vNormal=normalize(uNormalMat*aNormal);
  gl_Position=uMVP*vec4(aPos,1.0);
})";
  }
  if (fsrc.empty()) {
    fsrc = R"(#version 330 core
in vec3 vWorldPos; in vec3 vNormal;
uniform vec3 uColor; uniform vec3 uLightDir; uniform vec3 uEye;
uniform float uSelected; uniform float uAlpha;
out vec4 FragColor;
void main(){
  vec3 n=normalize(vNormal); vec3 l=normalize(-uLightDir);
  float ndotl=max(dot(n,l),0.0);
  float ndotl2=max(dot(-n,l),0.0);
  float lit=max(ndotl, ndotl2*0.35);
  vec3 view=normalize(uEye-vWorldPos); vec3 h=normalize(l+view);
  float spec=pow(max(dot(n,h),0.0),32.0);
  vec3 base=uColor;
  if(uSelected>0.5) base=mix(base,vec3(1.0,0.9,0.25),0.55);
  FragColor=vec4(0.28*base+0.62*lit*base+0.12*spec*vec3(1.0), uAlpha);
})";
  }
  GLuint vs = compile_shader(GL_VERTEX_SHADER, vsrc.c_str());
  GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fsrc.c_str());
  mesh_prog_ = link_program(vs, fs);
  glDeleteShader(vs);
  glDeleteShader(fs);

  std::string lv = load_shader_file(base + "line.vert");
  std::string lf = load_shader_file(base + "line.frag");
  if (lv.empty()) {
    lv = R"(#version 330 core
layout(location=0) in vec3 aPos; layout(location=1) in vec3 aColor;
uniform mat4 uMVP; out vec3 vColor;
void main(){ vColor=aColor; gl_Position=uMVP*vec4(aPos,1.0); })";
  }
  if (lf.empty()) {
    lf = R"(#version 330 core
in vec3 vColor; out vec4 FragColor; void main(){ FragColor=vec4(vColor,1.0); })";
  }
  vs = compile_shader(GL_VERTEX_SHADER, lv.c_str());
  fs = compile_shader(GL_FRAGMENT_SHADER, lf.c_str());
  line_prog_ = link_program(vs, fs);
  glDeleteShader(vs);
  glDeleteShader(fs);
}

GLuint Viewport::compile_shader(GLenum type, const char* src) {
  GLuint s = glCreateShader(type);
  glShaderSource(s, 1, &src, nullptr);
  glCompileShader(s);
  GLint ok = 0;
  glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    char log[512];
    glGetShaderInfoLog(s, 512, nullptr, log);
    std::fprintf(stderr, "[viewport] Shader compile error: %s\n", log);
  }
  return s;
}

GLuint Viewport::link_program(GLuint vs, GLuint fs) {
  GLuint p = glCreateProgram();
  glAttachShader(p, vs);
  glAttachShader(p, fs);
  glLinkProgram(p);
  GLint ok = 0;
  glGetProgramiv(p, GL_LINK_STATUS, &ok);
  if (!ok) {
    char log[512];
    glGetProgramInfoLog(p, 512, nullptr, log);
    std::fprintf(stderr, "[viewport] Program link error: %s\n", log);
  }
  return p;
}

void Viewport::destroy_mesh(GpuMesh& gpu) {
  if (gpu.ebo) glDeleteBuffers(1, &gpu.ebo);
  if (gpu.nbo) glDeleteBuffers(1, &gpu.nbo);
  if (gpu.vbo) glDeleteBuffers(1, &gpu.vbo);
  if (gpu.vao) glDeleteVertexArrays(1, &gpu.vao);
  gpu = {};
}

void Viewport::upload_mesh(GpuMesh& gpu, const cad::Mesh& mesh) {
  destroy_mesh(gpu);
  if (mesh.empty()) return;
  std::vector<float> pos;
  std::vector<float> nrm;
  pos.reserve(mesh.positions.size() * 3);
  nrm.reserve(mesh.positions.size() * 3);
  for (std::size_t i = 0; i < mesh.positions.size(); ++i) {
    pos.push_back(static_cast<float>(mesh.positions[i].x));
    pos.push_back(static_cast<float>(mesh.positions[i].y));
    pos.push_back(static_cast<float>(mesh.positions[i].z));
    cad::Vec3 n = (i < mesh.normals.size()) ? mesh.normals[i] : cad::Vec3{0, 1, 0};
    nrm.push_back(static_cast<float>(n.x));
    nrm.push_back(static_cast<float>(n.y));
    nrm.push_back(static_cast<float>(n.z));
  }
  glGenVertexArrays(1, &gpu.vao);
  glBindVertexArray(gpu.vao);
  glGenBuffers(1, &gpu.vbo);
  glBindBuffer(GL_ARRAY_BUFFER, gpu.vbo);
  glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(pos.size() * sizeof(float)), pos.data(),
               GL_STATIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
  glGenBuffers(1, &gpu.nbo);
  glBindBuffer(GL_ARRAY_BUFFER, gpu.nbo);
  glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(nrm.size() * sizeof(float)), nrm.data(),
               GL_STATIC_DRAW);
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
  glGenBuffers(1, &gpu.ebo);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gpu.ebo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER,
               static_cast<GLsizeiptr>(mesh.indices.size() * sizeof(std::uint32_t)),
               mesh.indices.data(), GL_STATIC_DRAW);
  gpu.index_count = static_cast<GLsizei>(mesh.indices.size());
  glBindVertexArray(0);
}

void Viewport::upload_colored_lines(GpuMesh& gpu, const std::vector<float>& data, GLenum) {
  destroy_mesh(gpu);
  if (data.empty()) return;
  glGenVertexArrays(1, &gpu.vao);
  glBindVertexArray(gpu.vao);
  glGenBuffers(1, &gpu.vbo);
  glBindBuffer(GL_ARRAY_BUFFER, gpu.vbo);
  glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(data.size() * sizeof(float)), data.data(),
               GL_STATIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), nullptr);
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                        reinterpret_cast<void*>(3 * sizeof(float)));
  gpu.index_count = static_cast<GLsizei>(data.size() / 6);
  glBindVertexArray(0);
}

void Viewport::build_scene_helpers() {
  // Ground grid on XZ
  std::vector<float> grid;
  const int half = 10;
  for (int i = -half; i <= half; ++i) {
    const float t = static_cast<float>(i);
    const float c = (i == 0) ? 0.42f : 0.26f;
    grid.insert(grid.end(), {-static_cast<float>(half), 0, t, c, c, c});
    grid.insert(grid.end(), {static_cast<float>(half), 0, t, c, c, c});
    grid.insert(grid.end(), {t, 0, -static_cast<float>(half), c, c, c});
    grid.insert(grid.end(), {t, 0, static_cast<float>(half), c, c, c});
  }
  upload_colored_lines(grid_, grid, GL_LINES);

  // World axes at origin
  std::vector<float> ax = {
      0, 0, 0, 1, 0.2f, 0.2f, 2.5f, 0, 0, 1, 0.2f, 0.2f,
      0, 0, 0, 0.2f, 1, 0.2f, 0, 2.5f, 0, 0.2f, 1, 0.2f,
      0, 0, 0, 0.25f, 0.45f, 1, 0, 0, 2.5f, 0.25f, 0.45f, 1,
  };
  upload_colored_lines(axes_, ax, GL_LINES);

  // Small origin marker (cross)
  const float s = 0.12f;
  std::vector<float> om = {
      -s, 0, 0, 1, 1, 1, s, 0, 0, 1, 1, 1, 0, -s, 0, 1, 1, 1, 0, s, 0, 1, 1, 1,
      0, 0, -s, 1, 1, 1, 0, 0, s, 1, 1, 1,
  };
  upload_colored_lines(origin_marker_, om, GL_LINES);

  // Corner gizmo axes (unit)
  std::vector<float> gz = {
      0, 0, 0, 1, 0.15f, 0.15f, 1, 0, 0, 1, 0.15f, 0.15f,
      0, 0, 0, 0.15f, 1, 0.15f, 0, 1, 0, 0.15f, 1, 0.15f,
      0, 0, 0, 0.2f, 0.4f, 1, 0, 0, 1, 0.2f, 0.4f, 1,
  };
  upload_colored_lines(gizmo_axes_, gz, GL_LINES);
}

void Viewport::build_reference_planes() {
  for (auto& m : plane_fills_) destroy_mesh(m);
  for (auto& m : plane_edges_) destroy_mesh(m);
  plane_fills_.clear();
  plane_edges_.clear();
  if (!doc_) return;

  constexpr double half = 2.5;
  for (const auto& f : doc_->features()) {
    if (!cad::is_reference_plane(f.type) || !f.visible) continue;
    GpuMesh fill;
    fill.feature_id = f.id;
    fill.color = plane_tint(f.type);
    fill.translucent = true;
    upload_mesh(fill, make_plane_quad(f.type, half));
    plane_fills_.push_back(fill);

    // Soft border as line loop
    std::vector<float> border;
    const float h = static_cast<float>(half);
    auto push = [&](float x, float y, float z, float r, float g, float b) {
      border.insert(border.end(), {x, y, z, r, g, b});
    };
    const float r = static_cast<float>(fill.color.x);
    const float g = static_cast<float>(fill.color.y);
    const float b = static_cast<float>(fill.color.z);
    if (f.type == cad::FeatureType::PlaneFront) {
      push(-h, -h, 0, r, g, b); push(h, -h, 0, r, g, b);
      push(h, -h, 0, r, g, b); push(h, h, 0, r, g, b);
      push(h, h, 0, r, g, b); push(-h, h, 0, r, g, b);
      push(-h, h, 0, r, g, b); push(-h, -h, 0, r, g, b);
    } else if (f.type == cad::FeatureType::PlaneTop) {
      push(-h, 0, -h, r, g, b); push(h, 0, -h, r, g, b);
      push(h, 0, -h, r, g, b); push(h, 0, h, r, g, b);
      push(h, 0, h, r, g, b); push(-h, 0, h, r, g, b);
      push(-h, 0, h, r, g, b); push(-h, 0, -h, r, g, b);
    } else {
      push(0, -h, -h, r, g, b); push(0, -h, h, r, g, b);
      push(0, -h, h, r, g, b); push(0, h, h, r, g, b);
      push(0, h, h, r, g, b); push(0, h, -h, r, g, b);
      push(0, h, -h, r, g, b); push(0, -h, -h, r, g, b);
    }
    GpuMesh edge;
    edge.feature_id = f.id;
    edge.color = fill.color;
    upload_colored_lines(edge, border, GL_LINES);
    plane_edges_.push_back(edge);
  }
}

void Viewport::rebuild_geometry() {
  if (!gl_ready_ || !gl_ok_) return;
  makeCurrent();
  for (auto& m : solid_meshes_) destroy_mesh(m);
  solid_meshes_.clear();
  build_reference_planes();

  if (doc_) {
    std::unordered_map<int, int> use_count;
    for (const auto& f : doc_->features()) {
      if (cad::is_boolean(f.type)) {
        if (f.operand_a >= 0) use_count[f.operand_a]++;
        if (f.operand_b >= 0) use_count[f.operand_b]++;
      }
    }
    for (const auto& f : doc_->features()) {
      if (!f.visible || f.suppressed || cad::is_reference_plane(f.type)) continue;
      if (use_count.count(f.id)) continue;
      auto mesh = doc_->evaluate_feature(f.id);
      if (!mesh || mesh->empty()) continue;
      GpuMesh gpu;
      gpu.feature_id = f.id;
      gpu.color = color_for_id(f.id);
      upload_mesh(gpu, *mesh);
      solid_meshes_.push_back(gpu);
    }
  }
  doneCurrent();
  update();
}

void Viewport::draw_mesh(const GpuMesh& gpu, const QMatrix4x4& mvp, const QMatrix4x4& model,
                         bool selected, float alpha) {
  if (!gpu.vao || gpu.index_count == 0) return;
  glUseProgram(mesh_prog_);
  glUniformMatrix4fv(glGetUniformLocation(mesh_prog_, "uMVP"), 1, GL_FALSE, mvp.constData());
  glUniformMatrix4fv(glGetUniformLocation(mesh_prog_, "uModel"), 1, GL_FALSE, model.constData());
  QMatrix3x3 nrm = model.normalMatrix();
  glUniformMatrix3fv(glGetUniformLocation(mesh_prog_, "uNormalMat"), 1, GL_FALSE, nrm.constData());
  glUniform3f(glGetUniformLocation(mesh_prog_, "uColor"), static_cast<float>(gpu.color.x),
              static_cast<float>(gpu.color.y), static_cast<float>(gpu.color.z));
  glUniform3f(glGetUniformLocation(mesh_prog_, "uLightDir"), 0.35f, -1.0f, 0.4f);
  const auto eye = camera_.eye();
  glUniform3f(glGetUniformLocation(mesh_prog_, "uEye"), static_cast<float>(eye.x),
              static_cast<float>(eye.y), static_cast<float>(eye.z));
  glUniform1f(glGetUniformLocation(mesh_prog_, "uSelected"), selected ? 1.0f : 0.0f);
  glUniform1f(glGetUniformLocation(mesh_prog_, "uAlpha"), alpha);
  glBindVertexArray(gpu.vao);
  glDrawElements(GL_TRIANGLES, gpu.index_count, GL_UNSIGNED_INT, nullptr);
  glBindVertexArray(0);
}

void Viewport::draw_lines(GLuint vao, GLsizei count, const QMatrix4x4& mvp, GLenum mode) {
  if (!vao || count == 0) return;
  glUseProgram(line_prog_);
  glUniformMatrix4fv(glGetUniformLocation(line_prog_, "uMVP"), 1, GL_FALSE, mvp.constData());
  glBindVertexArray(vao);
  glDrawArrays(mode, 0, count);
  glBindVertexArray(0);
}

void Viewport::draw_corner_gizmo(int fb_w, int fb_h) {
  // Small viewport bottom-left for orientation triad
  const int size = std::max(80, fb_w / 10);
  glViewport(12, 12, size, size);
  glDisable(GL_DEPTH_TEST);
  // Orientation-only view (no translation)
  cad::Mat4 rot = cad::Quat::from_yaw_pitch(camera_.yaw(), camera_.pitch()).to_mat4();
  // Invert rotation for viewing directions matching main camera
  cad::Mat4 view = rot.inverted();
  // Pull back slightly
  view.m[14] = -3.0;
  cad::Mat4 proj = cad::Mat4::perspective(35.0 * cad::kDeg2Rad, 1.0, 0.1, 20.0);
  QMatrix4x4 mvp = to_qmat(proj * view);
  glLineWidth(2.0f);
  draw_lines(gizmo_axes_.vao, gizmo_axes_.index_count, mvp);
  glLineWidth(1.0f);
  glEnable(GL_DEPTH_TEST);
  glViewport(0, 0, fb_w, fb_h);
}

void Viewport::paintGL() {
  if (!gl_ok_) return;
  const int fbw = framebuffer_w();
  const int fbh = framebuffer_h();
  glViewport(0, 0, fbw, fbh);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  const QMatrix4x4 view = to_qmat(camera_.view_matrix());
  const QMatrix4x4 proj = to_qmat(camera_.projection_matrix(aspect()));
  const QMatrix4x4 vp = proj * view;
  const QMatrix4x4 model;  // identity

  // Grid + axes (opaque lines)
  glDisable(GL_CULL_FACE);
  draw_lines(grid_.vao, grid_.index_count, vp);
  glDisable(GL_DEPTH_TEST);
  draw_lines(axes_.vao, axes_.index_count, vp);
  draw_lines(origin_marker_.vao, origin_marker_.index_count, vp);
  glEnable(GL_DEPTH_TEST);

  // Solids
  glEnable(GL_CULL_FACE);
  for (const auto& m : solid_meshes_) {
    draw_mesh(m, vp * model, model, m.feature_id == selected_id_, 1.0f);
  }
  glDisable(GL_CULL_FACE);

  // Reference planes: depth write off for soft transparency, then borders
  glDepthMask(GL_FALSE);
  for (const auto& m : plane_fills_) {
    const bool sel = m.feature_id == selected_id_;
    draw_mesh(m, vp * model, model, sel, sel ? 0.45f : 0.22f);
  }
  glDepthMask(GL_TRUE);
  glLineWidth(2.0f);
  for (const auto& m : plane_edges_) {
    draw_lines(m.vao, m.index_count, vp);
  }
  glLineWidth(1.0f);

  draw_corner_gizmo(fbw, fbh);
}

void Viewport::mousePressEvent(QMouseEvent* e) {
  last_pos_ = e->pos();
  press_pos_ = e->pos();
  if (e->button() == Qt::LeftButton || e->button() == Qt::MiddleButton) {
    if (e->modifiers() & Qt::ShiftModifier) {
      panning_ = true;
    } else {
      rotating_ = true;
    }
  }
}

void Viewport::mouseMoveEvent(QMouseEvent* e) {
  const QPoint d = e->pos() - last_pos_;
  last_pos_ = e->pos();
  if (rotating_) {
    camera_.orbit(d.x(), d.y());
    update();
  } else if (panning_) {
    camera_.pan(d.x(), d.y(), height());
    update();
  }
}

void Viewport::mouseReleaseEvent(QMouseEvent* e) {
  const bool was_rotating = rotating_;
  rotating_ = false;
  panning_ = false;
  if (e->button() == Qt::LeftButton && was_rotating) {
    const QPoint delta = e->pos() - press_pos_;
    if (delta.manhattanLength() < 5) {
      const int id = pick_feature(e->pos());
      if (id >= 0) {
        selected_id_ = id;
        emit feature_picked(id);
        update();
      }
    }
  }
}

void Viewport::wheelEvent(QWheelEvent* e) {
  camera_.zoom(e->angleDelta().y());
  update();
}

int Viewport::pick_feature(const QPoint& pos) {
  if (!doc_ || !gl_ok_) return -1;
  // Ray in world space
  const double x = (2.0 * pos.x()) / width() - 1.0;
  const double y = 1.0 - (2.0 * pos.y()) / height();
  const cad::Mat4 inv_vp =
      (camera_.projection_matrix(aspect()) * camera_.view_matrix()).inverted();
  const cad::Vec3 near_p = inv_vp.transform_point({x, y, -1});
  const cad::Vec3 far_p = inv_vp.transform_point({x, y, 1});
  const cad::Vec3 origin = near_p;
  const cad::Vec3 dir = (far_p - near_p).normalized();

  double best_t = 1e30;
  int best_id = -1;

  // Prefer solid hits, then planes
  auto try_mesh = [&](const cad::Mesh& mesh, int fid) {
    for (std::size_t t = 0; t < mesh.triangle_count(); ++t) {
      const cad::Vec3& a = mesh.positions[mesh.indices[t * 3 + 0]];
      const cad::Vec3& b = mesh.positions[mesh.indices[t * 3 + 1]];
      const cad::Vec3& c = mesh.positions[mesh.indices[t * 3 + 2]];
      const cad::Vec3 e1 = b - a;
      const cad::Vec3 e2 = c - a;
      const cad::Vec3 pvec = cad::cross(dir, e2);
      const double det = cad::dot(e1, pvec);
      if (std::abs(det) < 1e-12) continue;
      const double inv_det = 1.0 / det;
      const cad::Vec3 tvec = origin - a;
      const double u = cad::dot(tvec, pvec) * inv_det;
      if (u < 0.0 || u > 1.0) continue;
      const cad::Vec3 qvec = cad::cross(tvec, e1);
      const double v = cad::dot(dir, qvec) * inv_det;
      if (v < 0.0 || u + v > 1.0) continue;
      const double tt = cad::dot(e2, qvec) * inv_det;
      if (tt > 1e-6 && tt < best_t) {
        best_t = tt;
        best_id = fid;
      }
    }
  };

  for (const auto& f : doc_->features()) {
    if (!f.visible || f.suppressed || cad::is_reference_plane(f.type)) continue;
    auto mesh = doc_->evaluate_feature(f.id);
    if (mesh) try_mesh(*mesh, f.id);
  }
  if (best_id >= 0) return best_id;

  // Planes (finite quads)
  best_t = 1e30;
  for (const auto& f : doc_->features()) {
    if (!f.visible || !cad::is_reference_plane(f.type)) continue;
    cad::Mesh quad = make_plane_quad(f.type, 2.5);
    for (std::size_t t = 0; t < quad.triangle_count(); ++t) {
      const cad::Vec3& a = quad.positions[quad.indices[t * 3 + 0]];
      const cad::Vec3& b = quad.positions[quad.indices[t * 3 + 1]];
      const cad::Vec3& c = quad.positions[quad.indices[t * 3 + 2]];
      const cad::Vec3 e1 = b - a;
      const cad::Vec3 e2 = c - a;
      const cad::Vec3 pvec = cad::cross(dir, e2);
      const double det = cad::dot(e1, pvec);
      if (std::abs(det) < 1e-12) continue;
      const double inv_det = 1.0 / det;
      const cad::Vec3 tvec = origin - a;
      const double u = cad::dot(tvec, pvec) * inv_det;
      if (u < 0.0 || u > 1.0) continue;
      const cad::Vec3 qvec = cad::cross(tvec, e1);
      const double v = cad::dot(dir, qvec) * inv_det;
      if (v < 0.0 || u + v > 1.0) continue;
      const double tt = cad::dot(e2, qvec) * inv_det;
      if (tt > 1e-6 && tt < best_t) {
        best_t = tt;
        best_id = f.id;
      }
    }
  }
  return best_id;
}

}  // namespace app
