cmake_minimum_required(VERSION 3.18)

project(cpp_oti_lib LANGUAGES CXX)

option(OTI_ENABLE_KOKKOS "Build Kokkos-enabled otinum tests" OFF)
option(OTI_BUILD_PYTHON "Build the Python extension module" ON)

add_library(otinum_headers INTERFACE)
target_compile_features(otinum_headers INTERFACE cxx_std_17)
target_include_directories(otinum_headers INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(OTI_ENABLE_KOKKOS)
    find_package(Kokkos REQUIRED)

    target_compile_definitions(otinum_headers INTERFACE OTI_ENABLE_KOKKOS)
    target_link_libraries(otinum_headers INTERFACE Kokkos::kokkos)

    enable_testing()
    add_executable(test_kokkos_smoke tests/test_kokkos_smoke.cpp)
    target_link_libraries(test_kokkos_smoke PRIVATE otinum_headers)
    add_test(NAME test_kokkos_smoke COMMAND test_kokkos_smoke)
endif()

if(OTI_BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(otinum bindings/python/otinum_py.cpp)

    target_link_libraries(otinum PRIVATE otinum_headers)

    install(TARGETS otinum LIBRARY DESTINATION .)
endif()
