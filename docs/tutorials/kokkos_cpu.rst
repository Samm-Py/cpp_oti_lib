Kokkos CPU Tutorial
===================

Kokkos support is optional. The header-only scalar type uses ``std::array`` in
ordinary C++ builds and ``Kokkos::Array`` when ``OTI_ENABLE_KOKKOS`` is enabled.

Install Or Build Kokkos
-----------------------

Build a CPU-backed Kokkos installation with OpenMP:

.. code-block:: console

   cmake -S /path/to/kokkos -B /tmp/kokkos-build-openmp \
     -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_INSTALL_PREFIX=/tmp/kokkos-install-openmp \
     -DCMAKE_CXX_COMPILER=g++ \
     -DKokkos_ENABLE_OPENMP=ON \
     -DKokkos_ARCH_NATIVE=ON

   cmake --build /tmp/kokkos-build-openmp --target install --parallel

Configure cpp_oti_lib
---------------------

Point ``CMAKE_PREFIX_PATH`` at the Kokkos install:

.. code-block:: console

   cmake -S . -B build-kokkos-cpu \
     -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_PREFIX_PATH=/tmp/kokkos-install-openmp \
     -DOTI_ENABLE_KOKKOS=ON \
     -DOTI_BUILD_PYTHON=OFF

   cmake --build build-kokkos-cpu --parallel
   ctest --test-dir build-kokkos-cpu --output-on-failure

What To Expect
--------------

The build creates and runs ``test_kokkos_smoke``. That test launches a Kokkos
kernel using OTI values and checks selected coefficients after copying results
back to the host.
