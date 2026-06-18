# Standalone build for the GPU (Kokkos+CUDA) MPI + OTI toy. Needs a CUDA-enabled
# Kokkos install and MPI. Configure with the Kokkos nvcc_wrapper as the C++
# compiler (required for CUDA-backend Kokkos) and point at the Kokkos install:
#
#   cmake -S . -B build \
#     -DCMAKE_CXX_COMPILER=/root/Research/kokkos-cuda-install/bin/nvcc_wrapper \
#     -DKokkos_ROOT=/root/Research/kokkos-cuda-install
#   cmake --build build
#   mpirun -np 2 ./build/mpi_oti_gpu_toy
cmake_minimum_required(VERSION 3.16)
project(mpi_oti_gpu_toy CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Kokkos REQUIRED)
find_package(MPI REQUIRED)

add_executable(mpi_oti_gpu_toy main_gpu.cpp)
target_compile_definitions(mpi_oti_gpu_toy PRIVATE OTI_ENABLE_KOKKOS)
target_include_directories(mpi_oti_gpu_toy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(mpi_oti_gpu_toy PRIVATE Kokkos::kokkos MPI::MPI_CXX)
