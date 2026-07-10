find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets OpenGL OpenGLWidgets)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

add_executable(cadapp
  src/main.cpp
  src/mainwindow.cpp
  src/mainwindow.hpp
  src/viewport.cpp
  src/viewport.hpp
  src/feature_tree_model.cpp
  src/feature_tree_model.hpp
  src/property_panel.cpp
  src/property_panel.hpp
  src/camera.cpp
  src/camera.hpp
)

target_link_libraries(cadapp PRIVATE
  cad::core
  Qt6::Core
  Qt6::Gui
  Qt6::Widgets
  Qt6::OpenGL
  Qt6::OpenGLWidgets
)

target_compile_features(cadapp PRIVATE cxx_std_20)

# Embed shader sources next to the binary for runtime loading
add_custom_command(TARGET cadapp POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:cadapp>/shaders
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders
    $<TARGET_FILE_DIR:cadapp>/shaders
)
