# Standalone build for the MPI + OTI unstructured ghost-exchange example.
# Independent of the parent project's CMake -- it only needs MPI and the
# header-only library.
#
#   cmake -S . -B build && cmake --build build
#   mpirun -np 4 ./build/mpi_oti_unstructured
cmake_minimum_required(VERSION 3.16)
project(mpi_oti_unstructured CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

# The plain-double starting point (before): same solver, no derivatives.
add_executable(mpi_unstructured_before main_before.cpp)
target_link_libraries(mpi_unstructured_before PRIVATE MPI::MPI_CXX)

# The OTI version (after): jets carry the source sensitivities.
add_executable(mpi_oti_unstructured main.cpp)
target_include_directories(mpi_oti_unstructured PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../include)
target_link_libraries(mpi_oti_unstructured PRIVATE MPI::MPI_CXX)
