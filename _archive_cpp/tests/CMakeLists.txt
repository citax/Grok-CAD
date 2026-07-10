add_executable(cadcore_tests
  test_main.cpp
  test_math.cpp
  test_mesh.cpp
  test_primitives.cpp
  test_csg.cpp
  test_stl.cpp
  test_document.cpp
)

target_link_libraries(cadcore_tests PRIVATE cad::core)
target_compile_features(cadcore_tests PRIVATE cxx_std_20)

add_test(NAME cadcore_tests COMMAND cadcore_tests)
