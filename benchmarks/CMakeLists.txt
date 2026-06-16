# otinum optimization isolation benchmarks. Added from the top-level
# CMakeLists when OTI_BUILD_BENCHMARKS and OTI_ENABLE_KOKKOS are both on. Each
# benchmark isolates one optimization (see bench_common.hpp) and emits the same
# tidy CSV schema so one plotter consumes all of them.

# Single-binary benchmarks: the variant axis is selected by templates/runtime.
foreach(bench layout alignment fused)
    add_executable(bench_${bench} bench_${bench}.cpp)
    target_link_libraries(bench_${bench} PRIVATE otinum_headers)
    target_include_directories(bench_${bench} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endforeach()

# Arithmetic path (no table / runtime table / compile-time fold) is a
# compile-time switch over the whole otinum, so build one binary per path.
set(arith_0 naive)
set(arith_1 lookup)
set(arith_2 unrolled)
foreach(path 0 1 2)
    set(target bench_arithmetic_${arith_${path}})
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/bench_arithmetic.cpp)
        add_executable(${target} bench_arithmetic.cpp)
        target_link_libraries(${target} PRIVATE otinum_headers)
        target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        target_compile_definitions(${target} PRIVATE OTI_BENCHMARK_ARITHMETIC_PATH=${path})
    endif()
endforeach()
