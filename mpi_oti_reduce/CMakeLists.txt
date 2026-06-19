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

add_executable(mpi_oti_reduce main.cpp)
target_include_directories(mpi_oti_reduce PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(mpi_oti_reduce PRIVATE MPI::MPI_CXX)
