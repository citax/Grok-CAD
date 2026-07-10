cmake_minimum_required(VERSION 3.21)
project(CadApp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Strict warnings, treat as errors for clean builds
if(MSVC)
  add_compile_options(/W4 /permissive-)
else()
  add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

option(CAD_BUILD_APP "Build the Qt6 GUI application" ON)
option(CAD_BUILD_TESTS "Build unit tests" ON)

add_subdirectory(core)

if(CAD_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(CAD_BUILD_APP)
  add_subdirectory(app)
endif()
