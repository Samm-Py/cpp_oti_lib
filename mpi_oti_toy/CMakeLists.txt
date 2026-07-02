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

# Confidence test: datatype layout (size/extent) + point-to-point round-trip,
# across float/double and odd-ncoeffs shapes.
add_executable(test_mpi_datatype test_mpi_datatype.cpp)
target_include_directories(test_mpi_datatype PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(test_mpi_datatype PRIVATE MPI::MPI_CXX)

# Rank-scaling benchmark: times the parallel compute region over the grid.
# Sweep np and feed the CSV to plot_scaling.py.
add_executable(mpi_oti_scaling mpi_oti_scaling.cpp)
target_include_directories(mpi_oti_scaling PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(mpi_oti_scaling PRIVATE MPI::MPI_CXX)

# Derivative accuracy vs analytical (serial -- accuracy is independent of ranks).
# Emits per-algebra error samples + exact RMSE; feed to plot_accuracy.py.
add_executable(verify_derivatives verify_derivatives.cpp)
target_include_directories(verify_derivatives PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)

# Confidence test: the oti::validity error-analysis methods run under MPI and are
# decomposition-invariant (trusted count + global-min reach identical across np).
add_executable(test_mpi_validity test_mpi_validity.cpp)
target_include_directories(test_mpi_validity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(test_mpi_validity PRIVATE MPI::MPI_CXX)
