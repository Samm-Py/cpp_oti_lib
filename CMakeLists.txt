cmake_minimum_required(VERSION 3.18)

project(cpp_oti_lib LANGUAGES CXX)

option(OTI_ENABLE_KOKKOS "Build Kokkos-enabled otinum tests" OFF)
option(OTI_BUILD_PYTHON "Build the Python extension module" OFF)
option(OTI_BUILD_TESTS "Build focused otinum unit tests" ON)

add_library(otinum_headers INTERFACE)
target_compile_features(otinum_headers INTERFACE cxx_std_17)
target_include_directories(otinum_headers INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(OTI_BUILD_TESTS OR OTI_ENABLE_KOKKOS)
    enable_testing()
endif()

if(OTI_BUILD_TESTS)
    set(OTI_TEST_SOURCES
        tests/test_abs_large_shapes.cpp
        tests/test_construction_access.cpp
        tests/test_edge_cases.cpp
        tests/test_exp_log_pow.cpp
        tests/test_float_coefficients.cpp
        tests/test_layout_tables.cpp
        tests/test_linear_arithmetic.cpp
        tests/test_multiplication_division.cpp
        tests/test_profile.cpp
        tests/test_trig_hyperbolic.cpp
        tests/test_truncated_operations.cpp
    )

    foreach(test_source IN LISTS OTI_TEST_SOURCES)
        get_filename_component(test_name "${test_source}" NAME_WE)
        add_executable("${test_name}" "${test_source}")
        target_include_directories("${test_name}" PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
        target_link_libraries("${test_name}" PRIVATE otinum_headers)
        add_test(NAME "${test_name}" COMMAND "${test_name}")
    endforeach()
endif()

if(OTI_ENABLE_KOKKOS)
    find_package(Kokkos REQUIRED)

    target_compile_definitions(otinum_headers INTERFACE OTI_ENABLE_KOKKOS)
    target_link_libraries(otinum_headers INTERFACE Kokkos::kokkos)

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
