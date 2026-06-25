# Standalone build for the MPI + OTI global-reduction example. Independent of the
# parent project's CMake -- it only needs MPI and the header-only library.
#
#   cmake -S . -B build && cmake --build build
#   mpirun -np 4 ./build/mpi_oti_reduce
cmake_minimum_required(VERSION 3.16)
project(mpi_oti_reduce CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

# The plain-double starting point (before): same QoI, MPI_SUM, no derivatives.
add_executable(mpi_reduce_before main_before.cpp)
target_link_libraries(mpi_reduce_before PRIVATE MPI::MPI_CXX)

# The OTI version (after): custom MPI_Op over jets -> gradient + Hessian.
add_executable(mpi_oti_reduce main.cpp)
target_include_directories(mpi_oti_reduce PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(mpi_oti_reduce PRIVATE MPI::MPI_CXX)

# Confidence test: sum / prod / max / min reduction ops vs serial recompute.
add_executable(test_reduce_ops test_reduce_ops.cpp)
target_include_directories(test_reduce_ops PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(test_reduce_ops PRIVATE MPI::MPI_CXX)

# Compliance test over the over-aligned (alignas(32)) jet shapes promoted by the
# 32-byte alignment rung: sum/prod/maxloc/minloc bit-exact on real MPI buffers.
add_executable(test_reduce_promoted test_reduce_promoted.cpp)
target_include_directories(test_reduce_promoted PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(test_reduce_promoted PRIVATE MPI::MPI_CXX)

# Alignment-safety test for reduce_fn: runs the user callback over deliberately
# under-aligned buffers under UBSan, so a regression to a direct otinum& bind
# (misaligned UB) aborts. Single process; no MPI runtime needed.
add_executable(test_reduce_alignment test_reduce_alignment.cpp)
target_include_directories(test_reduce_alignment PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(test_reduce_alignment PRIVATE MPI::MPI_CXX)
target_compile_options(test_reduce_alignment PRIVATE
    -fsanitize=undefined -fno-sanitize-recover=undefined)
target_link_options(test_reduce_alignment PRIVATE -fsanitize=undefined)

# Strong-scaling harness: local accumulation + one-element Allreduce latency.
add_executable(mpi_oti_reduce_scaling mpi_oti_reduce_scaling.cpp)
target_include_directories(mpi_oti_reduce_scaling PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(mpi_oti_reduce_scaling PRIVATE MPI::MPI_CXX)

# ctest suite: the under-aligned reduce_fn check (single process) plus the MPI
# reduction tests at np = 1 (serial), 3 (non-power), and 4.
enable_testing()
add_test(NAME reduce_alignment COMMAND test_reduce_alignment)
foreach(np 1 3 4)
    add_test(NAME reduce_ops_np${np}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_reduce_ops> ${MPIEXEC_POSTFLAGS})
    add_test(NAME reduce_promoted_np${np}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_reduce_promoted> ${MPIEXEC_POSTFLAGS})
endforeach()
