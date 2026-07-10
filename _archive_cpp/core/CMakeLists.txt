add_library(cadcore
  src/math/mat4.cpp
  src/math/quat.cpp
  src/mesh/mesh.cpp
  src/mesh/primitives.cpp
  src/mesh/csg.cpp
  src/mesh/transform.cpp
  src/mesh/stl.cpp
  src/document/document.cpp
  src/document/json_io.cpp
)

add_library(cad::core ALIAS cadcore)

target_include_directories(cadcore
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(cadcore PUBLIC cxx_std_20)
