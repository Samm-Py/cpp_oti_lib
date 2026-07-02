# Standalone build for the MPI + OTI halo-exchange (Jacobi) example. Independent
# of the parent project's CMake -- it only needs MPI and the header-only library.
#
#   cmake -S . -B build && cmake --build build
#   mpirun -np 4 ./build/mpi_oti_halo
cmake_minimum_required(VERSION 3.16)
project(mpi_oti_halo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

# The plain-double starting point (before): same solver, no derivatives.
add_executable(mpi_halo_before main_before.cpp)
target_link_libraries(mpi_halo_before PRIVATE MPI::MPI_CXX)

# The OTI version (after): jets carry the hot-wall sensitivities.
add_executable(mpi_oti_halo main.cpp)
target_include_directories(mpi_oti_halo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../include)
target_link_libraries(mpi_oti_halo PRIVATE MPI::MPI_CXX)
