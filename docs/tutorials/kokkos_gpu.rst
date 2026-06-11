Kokkos GPU Tutorial
===================

GPU execution uses the same OTI headers, but the Kokkos installation must be
built with a GPU backend such as CUDA. The ``cpp_oti_lib`` side still uses the
same CMake option, ``-DOTI_ENABLE_KOKKOS=ON``; the main difference is that the
Kokkos package and compiler wrapper must be configured for device compilation.

This tutorial focuses on CUDA because that is the GPU backend exercised by the
project's conditional CI job.

What Gets Tested
----------------

The GPU build runs the same ``test_kokkos_smoke`` executable used by the CPU
Kokkos tutorial. With a CUDA-backed Kokkos install, that executable launches
the smoke-test kernel on the selected Kokkos execution space and checks the
computed OTI coefficients after copying them back to the host.

The expected successful test output includes:

.. code-block:: text

   Kokkos otinum kernel tests passed

Prerequisites
-------------

You need:

* a CUDA-capable GPU
* a CUDA toolkit with ``nvcc``
* a host compiler supported by the selected CUDA toolkit and Kokkos version
* CMake 3.18 or newer
* a Kokkos source checkout or existing CUDA-backed Kokkos install

Check that the GPU and CUDA compiler are visible before building:

.. code-block:: console

   nvidia-smi -L
   nvcc --version

The exact host compiler and architecture flag depend on the target machine.
Those are CUDA/Kokkos configuration choices rather than ``cpp_oti_lib`` API
choices.

Build CUDA Kokkos
-----------------

Skip this section if you already have a CUDA-backed Kokkos installation and
know its install prefix. Otherwise, build Kokkos with CUDA enabled. Replace
``Kokkos_ARCH_AMPERE80`` with the architecture flag that matches the target
GPU.

.. code-block:: console

   git clone --branch 5.1.1 --depth 1 \
     https://github.com/kokkos/kokkos.git /tmp/kokkos

   cmake -S /tmp/kokkos -B /tmp/kokkos-build-cuda \
     -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_INSTALL_PREFIX=/tmp/kokkos-install-cuda \
     -DCMAKE_CXX_COMPILER=/tmp/kokkos/bin/nvcc_wrapper \
     -DKokkos_ENABLE_CUDA=ON \
     -DKokkos_ENABLE_CUDA_LAMBDA=ON \
     -DKokkos_ARCH_AMPERE80=ON

   cmake --build /tmp/kokkos-build-cuda --target install --parallel

``nvcc_wrapper`` is the C++ compiler for this configuration. Use it both when
building Kokkos and when configuring ``cpp_oti_lib`` against that Kokkos
install.

Configure cpp_oti_lib
---------------------

Use the same Kokkos compiler wrapper and installation prefix. This ensures
that ``find_package(Kokkos)`` imports the CUDA-enabled ``Kokkos::kokkos`` target
and that this project's test executable is compiled for the same backend.

.. code-block:: console

   cmake -S . -B build-kokkos-gpu \
     -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_CXX_COMPILER=/tmp/kokkos/bin/nvcc_wrapper \
     -DCMAKE_PREFIX_PATH=/tmp/kokkos-install-cuda \
     -DOTI_ENABLE_KOKKOS=ON \
     -DOTI_BUILD_PYTHON=OFF \
     -DOTI_BUILD_TESTS=OFF

   cmake --build build-kokkos-gpu --parallel
   ctest --test-dir build-kokkos-gpu --output-on-failure

``-DOTI_BUILD_TESTS=OFF`` is optional, but it keeps this build focused on the
Kokkos smoke test. The scalar tests do not require CUDA.

Choosing The Architecture Flag
------------------------------

The ``Kokkos_ARCH_*`` option should match the GPU where the test will run. A
specific architecture flag is usually preferable for reproducible local builds.
For generic CI environments, ``Kokkos_ARCH_NATIVE=ON`` can be useful when the
runner has a visible GPU and the installed Kokkos version supports detecting
it.

If the architecture flag is wrong, failures often appear as CUDA compilation
errors, unsupported instruction errors, or runtime launch failures.

How The OTI Headers Reach The Device
------------------------------------

The OTI arithmetic functions are annotated through the compatibility layer in
``include/otinum/detail/kokkos_compat.hpp``. When ``OTI_ENABLE_KOKKOS`` is set,
those annotations make the relevant operations callable inside Kokkos device
kernels.

The CMake target also switches the coefficient container from ``std::array`` to
``Kokkos::Array``. That is why the Kokkos test needs to be built through CMake
with ``Kokkos::kokkos`` linked, rather than through the direct scalar test
script.

Troubleshooting
---------------

If CMake cannot find Kokkos, check that ``CMAKE_PREFIX_PATH`` points at the
CUDA-backed Kokkos install prefix.

If the configure step says CUDA is unavailable, confirm that ``nvcc`` is on the
``PATH`` and that the selected host compiler is compatible with the CUDA
toolkit.

If compilation fails in device code, check the Kokkos compiler wrapper, CUDA
toolkit version, host compiler compatibility, and architecture flag first.

If the test builds but does not run, confirm that the job is running on a
machine with a visible CUDA device. ``nvidia-smi -L`` should list at least one
GPU.

CI Behavior
-----------

The GitHub Actions GPU job is conditional. It checks for both ``nvidia-smi``
and ``nvcc`` and skips the CUDA Kokkos build when no matching device/toolkit
pair is visible. Set the repository variable ``KOKKOS_GPU_RUNNER`` to a
CUDA-capable runner label when you want CI to exercise this path regularly.

CUDA device execution is not merged into the GCC coverage report. The coverage
report uses the OpenMP Kokkos path; the GPU job is a runtime smoke test for the
CUDA backend.
