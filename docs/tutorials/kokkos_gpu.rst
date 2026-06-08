Kokkos GPU Tutorial
===================

GPU execution uses the same OTI headers, but the Kokkos installation must be
built with a GPU backend such as CUDA.

Build CUDA Kokkos
-----------------

The exact architecture flag depends on the GPU. Replace ``Kokkos_ARCH_AMPERE80``
with the architecture that matches the target machine.

.. code-block:: console

   cmake -S /path/to/kokkos -B /tmp/kokkos-build-cuda \
     -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_INSTALL_PREFIX=/tmp/kokkos-install-cuda \
     -DCMAKE_CXX_COMPILER=/path/to/kokkos/bin/nvcc_wrapper \
     -DKokkos_ENABLE_CUDA=ON \
     -DKokkos_ENABLE_CUDA_LAMBDA=ON \
     -DKokkos_ARCH_AMPERE80=ON

   cmake --build /tmp/kokkos-build-cuda --target install --parallel

Configure cpp_oti_lib
---------------------

Use the same Kokkos compiler wrapper and installation prefix:

.. code-block:: console

   cmake -S . -B build-kokkos-gpu \
     -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_CXX_COMPILER=/path/to/kokkos/bin/nvcc_wrapper \
     -DCMAKE_PREFIX_PATH=/tmp/kokkos-install-cuda \
     -DOTI_ENABLE_KOKKOS=ON \
     -DOTI_BUILD_PYTHON=OFF

   cmake --build build-kokkos-gpu --parallel
   ctest --test-dir build-kokkos-gpu --output-on-failure

Notes
-----

The OTI arithmetic functions are annotated through the compatibility layer in
``include/otinum/detail/kokkos_compat.hpp``. When ``OTI_ENABLE_KOKKOS`` is set,
those annotations make the relevant operations callable inside Kokkos device
kernels.

If compilation fails, check the Kokkos compiler wrapper, CUDA toolkit version,
host compiler compatibility, and architecture flag first.
