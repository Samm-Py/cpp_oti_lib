# Before/after pair showing how to OTI-enable an MPI program. Plain MPI -- no
# Kokkos, no GPU. Needs only MPI and the header-only library.
#
#   cmake -S . -B build && cmake --build build
#   mpirun -np 4 ./build/convert_before
#   mpirun -np 4 ./build/convert_after
cmake_minimum_required(VERSION 3.16)
project(mpi_oti_convert CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

# BEFORE: plain double. Does not touch otinum at all.
add_executable(convert_before convert_before.cpp)
target_link_libraries(convert_before PRIVATE MPI::MPI_CXX)

# AFTER: the OTI-enabled version. The only build-side change vs BEFORE is the
# header include path -- otinum is header-only, so there is nothing to link.
add_executable(convert_after convert_after.cpp)
target_include_directories(convert_after PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(convert_after PRIVATE MPI::MPI_CXX)
