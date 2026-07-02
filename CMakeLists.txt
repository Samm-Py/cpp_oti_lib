cmake_minimum_required(VERSION 3.18)

project(cpp_oti_lib LANGUAGES CXX)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

option(OTI_ENABLE_KOKKOS "Enable Kokkos support and its smoke test" OFF)
option(OTI_BUILD_PYTHON "Build the Python extension module" OFF)
option(OTI_BUILD_TESTS "Build focused otinum unit tests" ON)
option(OTI_BUILD_BENCHMARKS "Build optional otinum microbenchmarks" OFF)
option(OTI_INSTALL_CMAKE_PACKAGE "Install headers and CMake package files" ON)

add_library(otinum_headers INTERFACE)
add_library(oti::otinum ALIAS otinum_headers)
set_target_properties(otinum_headers PROPERTIES EXPORT_NAME otinum)
target_compile_features(otinum_headers INTERFACE cxx_std_17)
target_include_directories(otinum_headers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
# Clang caps fold-expression expansion at 256 arguments by default, but the
# unrolled product folds have tables<M, N>::nproducts = C(2M + N, N) arguments
# (286 already at <5, 3>). 65536 covers every shape whose compile-time memory
# cost is feasible on real hardware (<6, 6> needs 18564).
#
# -Wno-psabi: shapes sized a multiple of 32 bytes are alignas(32) (the 256-bit
# load rung in otinum_alignment), and passing such a type by value triggers a
# GCC/Clang note that the parameter-passing ABI for >16-byte alignment changed
# in GCC 4.6. It is purely informational for a single consistent toolchain, so
# silence it rather than spam every consumer build that instantiates one.
target_compile_options(otinum_headers INTERFACE
    $<$<CXX_COMPILER_ID:Clang,AppleClang>:-fbracket-depth=65536>
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wno-psabi>
)

if(OTI_BUILD_TESTS OR OTI_ENABLE_KOKKOS)
    enable_testing()
endif()

if(OTI_BUILD_TESTS)
    set(OTI_TEST_SOURCES
        tests/test_abs_large_shapes.cpp
        tests/test_comparisons.cpp
        tests/test_construction_access.cpp
        tests/test_edge_cases.cpp
        tests/test_exp_log_pow.cpp
        tests/test_float_coefficients.cpp
        tests/test_fused_ops.cpp
        tests/test_interop.cpp
        tests/test_layout_tables.cpp
        tests/test_linear_arithmetic.cpp
        tests/test_multiplication_division.cpp
        tests/test_profile.cpp
        tests/test_soa_span.cpp
        tests/test_trig_hyperbolic.cpp
        tests/test_truncated_operations.cpp
        tests/test_validity.cpp
    )

    foreach(test_source IN LISTS OTI_TEST_SOURCES)
        get_filename_component(test_name "${test_source}" NAME_WE)
        add_executable("${test_name}" "${test_source}")
        target_include_directories("${test_name}" PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
        target_link_libraries("${test_name}" PRIVATE otinum_headers)
        add_test(NAME "${test_name}" COMMAND "${test_name}")
    endforeach()

    foreach(arithmetic_path 0 1)
        foreach(test_source
                tests/test_multiplication_division.cpp
                tests/test_exp_log_pow.cpp)
            get_filename_component(base_name "${test_source}" NAME_WE)
            set(test_name "${base_name}_benchmark_path_${arithmetic_path}")
            add_executable("${test_name}" "${test_source}")
            target_include_directories("${test_name}" PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
            target_compile_definitions("${test_name}" PRIVATE
                OTI_BENCHMARK_ARITHMETIC_PATH=${arithmetic_path})
            target_link_libraries("${test_name}" PRIVATE otinum_headers)
            add_test(NAME "${test_name}" COMMAND "${test_name}")
        endforeach()
    endforeach()
endif()

if(OTI_BUILD_BENCHMARKS)
    add_executable(benchmark_operations tests/benchmark_operations.cpp)
    target_link_libraries(benchmark_operations PRIVATE otinum_headers)
endif()

if(OTI_ENABLE_KOKKOS)
    find_package(Kokkos REQUIRED)

    target_compile_definitions(otinum_headers INTERFACE OTI_ENABLE_KOKKOS)
    target_link_libraries(otinum_headers INTERFACE Kokkos::kokkos)

    add_executable(test_kokkos_smoke tests/test_kokkos_smoke.cpp)
    target_link_libraries(test_kokkos_smoke PRIVATE otinum_headers)
    add_test(NAME test_kokkos_smoke COMMAND test_kokkos_smoke)

    add_executable(test_validity_kokkos tests/test_validity_kokkos.cpp)
    target_link_libraries(test_validity_kokkos PRIVATE otinum_headers)
    add_test(NAME test_validity_kokkos COMMAND test_validity_kokkos)

    if(OTI_BUILD_BENCHMARKS)
        add_executable(benchmark_soa_layout tests/benchmark_soa_layout.cpp)
        target_link_libraries(benchmark_soa_layout PRIVATE otinum_headers)

        add_subdirectory(benchmarks)
    endif()
endif()

if(OTI_BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(otinum bindings/python/otinum_py.cpp)

    target_link_libraries(otinum PRIVATE otinum_headers)

    install(TARGETS otinum LIBRARY DESTINATION .)
endif()

if(OTI_INSTALL_CMAKE_PACKAGE)
    set(OTINUM_CMAKE_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/otinum")

    configure_package_config_file(
        cmake/otinumConfig.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/otinumConfig.cmake"
        INSTALL_DESTINATION "${OTINUM_CMAKE_INSTALL_DIR}"
    )

    install(TARGETS otinum_headers EXPORT otinumTargets)
    install(
        DIRECTORY include/
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        FILES_MATCHING PATTERN "*.hpp"
    )
    install(
        EXPORT otinumTargets
        FILE otinumTargets.cmake
        NAMESPACE oti::
        DESTINATION "${OTINUM_CMAKE_INSTALL_DIR}"
    )
    install(
        FILES "${CMAKE_CURRENT_BINARY_DIR}/otinumConfig.cmake"
        DESTINATION "${OTINUM_CMAKE_INSTALL_DIR}"
    )
endif()
