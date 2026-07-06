# Standalone build for the device-buffer (CUDA-aware MPI) example. Needs Kokkos
# and MPI; for a CUDA backend configure with the Kokkos nvcc_wrapper as the C++
# compiler and point at the Kokkos install:
#
#   cmake -S . -B build \
#     -DCMAKE_CXX_COMPILER=/path/to/kokkos/bin/nvcc_wrapper \
#     -DKokkos_ROOT=/path/to/kokkos
#   cmake --build build
#   mpirun -np 4 ./build/mpi_oti_device
#
# A host-parallel Kokkos (OpenMP) works too: the "device" views are then host
# memory and the direct-pointer path is safe under any MPI (OTI_MPI_DEVICE=1).
cmake_minimum_required(VERSION 3.16)
project(mpi_oti_device CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Kokkos REQUIRED)
find_package(MPI REQUIRED)

add_executable(mpi_oti_device main.cpp)
target_compile_definitions(mpi_oti_device PRIVATE OTI_ENABLE_KOKKOS)
target_include_directories(mpi_oti_device PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../include)
target_link_libraries(mpi_oti_device PRIVATE Kokkos::kokkos MPI::MPI_CXX)
