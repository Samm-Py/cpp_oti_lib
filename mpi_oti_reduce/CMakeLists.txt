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

# Strong-scaling harness: local accumulation + one-element Allreduce latency.
add_executable(mpi_oti_reduce_scaling mpi_oti_reduce_scaling.cpp)
target_include_directories(mpi_oti_reduce_scaling PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(mpi_oti_reduce_scaling PRIVATE MPI::MPI_CXX)
