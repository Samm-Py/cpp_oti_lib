# Standalone build for the MPI + OTI toy. Independent of the parent project's
# CMake -- it only needs MPI and the header-only library on the include path.
#
#   cmake -S . -B build && cmake --build build
#   mpirun -np 4 ./build/mpi_oti_toy
cmake_minimum_required(VERSION 3.16)
project(mpi_oti_toy CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

add_executable(mpi_oti_toy main.cpp)
target_include_directories(mpi_oti_toy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(mpi_oti_toy PRIVATE MPI::MPI_CXX)
