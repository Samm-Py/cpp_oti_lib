# Before/after pair showing how to OTI-enable a Kokkos + MPI program.
# Needs a Kokkos install (any backend) and MPI. Configure with the Kokkos
# compiler wrapper (nvcc_wrapper for a CUDA Kokkos):
#
#   cmake -S . -B build \
#     -DCMAKE_CXX_COMPILER=/root/Research/kokkos-cuda-install/bin/nvcc_wrapper \
#     -DKokkos_ROOT=/root/Research/kokkos-cuda-install
#   cmake --build build
#   mpirun -np 4 ./build/convert_before
#   mpirun -np 4 ./build/convert_after
cmake_minimum_required(VERSION 3.16)
project(mpi_oti_convert CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Kokkos REQUIRED)
find_package(MPI REQUIRED)

# BEFORE: plain double. Does not touch otinum at all.
add_executable(convert_before convert_before.cpp)
target_link_libraries(convert_before PRIVATE Kokkos::kokkos MPI::MPI_CXX)

# AFTER: the OTI-enabled version. The only build-side changes vs BEFORE are the
# OTI_ENABLE_KOKKOS define (device-callable jets) and the header include path.
add_executable(convert_after convert_after.cpp)
target_compile_definitions(convert_after PRIVATE OTI_ENABLE_KOKKOS)
target_include_directories(convert_after PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(convert_after PRIVATE Kokkos::kokkos MPI::MPI_CXX)
