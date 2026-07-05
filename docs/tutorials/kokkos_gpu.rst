Kokkos GPU Tutorial
===================

GPU execution uses the same OTI headers, but the Kokkos installation must be
built with a GPU backend such as CUDA. The ``cpp_oti_lib`` side still uses the
same CMake option, ``-DOTI_ENABLE_KOKKOS=ON``; the main difference is that the
Kokkos package and compiler wrapper must be configured for device compilation.

This tutorial focuses on CUDA because that is the GPU backend exercised by the
project's conditional CI job. It assumes you have already worked through
:doc:`kokkos_cpu`; only the Kokkos install and the compiler wrapper differ.

Prerequisites
-------------

You need:

* a CUDA-capable GPU
* a CUDA toolkit with ``nvcc``
* a host compiler that satisfies Kokkos's C++ standard *and whose flag spelling
  NVCC accepts*. Kokkos 5.1.1 requires C++20, and its CUDA build accepts only
  the literal ``-std=c++20`` flag. CMake emits that spelling for GCC 11 or
  newer (and recent Clang); GCC 10 emits ``-std=c++2a``, which NVCC rejects.
  So unlike the OpenMP build — where GCC 10 is fine — the CUDA build needs
  **GCC 11+ or a recent Clang** as the host compiler.
* CMake 3.18 or newer
* a Kokkos source checkout or existing CUDA-backed Kokkos install

Check that the GPU and CUDA compiler are visible before building:

.. code-block:: console

   nvidia-smi -L
   nvcc --version

Installing the CUDA toolkit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If ``nvcc --version`` fails, install the CUDA toolkit. The authoritative,
distribution-specific instructions are on NVIDIA's download page
(https://developer.nvidia.com/cuda-downloads); follow them to get a toolkit
that matches your driver. After installing, make sure ``nvcc`` is on the
``PATH`` (a system install usually lives in ``/usr/local/cuda/bin``):

.. code-block:: console

   export PATH=/usr/local/cuda/bin:$PATH
   nvcc --version

A distribution package also exists — ``sudo apt-get install
nvidia-cuda-toolkit`` on Debian/Ubuntu — but it is often several CUDA releases
behind (CUDA 10.1 on Ubuntu 20.04, for example), which may be too old for a
recent GPU or Kokkos version. Prefer NVIDIA's installer unless that older
toolkit is known to work for your hardware.

Installing a new-enough host compiler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your default compiler is older than GCC 11 (check with ``g++ --version``),
install a newer one. On Ubuntu 20.04, ``g++-11`` comes from the
``ppa:ubuntu-toolchain-r/test`` archive; newer distributions already ship a
recent enough default:

.. code-block:: console

   sudo add-apt-repository ppa:ubuntu-toolchain-r/test
   sudo apt-get update
   sudo apt-get install g++-11

``nvcc_wrapper`` does not take the host compiler as a CMake flag; it reads the
``NVCC_WRAPPER_DEFAULT_COMPILER`` environment variable. Export it before
configuring Kokkos *and* ``cpp_oti_lib``, so both use the same host compiler:

.. code-block:: console

   export NVCC_WRAPPER_DEFAULT_COMPILER=g++-11

Build CUDA Kokkos
-----------------

Skip this section if you already have a CUDA-backed Kokkos installation and
know its install prefix. Otherwise, build Kokkos with CUDA enabled. Replace
``Kokkos_ARCH_AMPERE80`` with the architecture flag that matches the target
GPU (see `Choosing The Architecture Flag`_ below). These commands use absolute
paths, so they can run from any directory.

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

If you already attempted this configure with an older compiler, delete the
Kokkos build directory before retrying. CMake caches the detected compiler
identity, so a configure re-run on an existing build directory keeps using the
old compiler — and the ``-std=c++2a`` error — even after you set
``NVCC_WRAPPER_DEFAULT_COMPILER``. A missing ``-- The CXX compiler
identification is ...`` line in the output is the sign that the cache, not your
new compiler, is in effect.

The architecture flag also depends on the target machine; that is a
CUDA/Kokkos configuration choice rather than a ``cpp_oti_lib`` API choice.

Configure cpp_oti_lib
---------------------

Use the same Kokkos compiler wrapper and installation prefix. This ensures
that ``find_package(Kokkos)`` imports the CUDA-enabled ``Kokkos::kokkos`` target
and that this project's test executable is compiled for the same backend. Run
this from the repository root, where ``-S .`` and the relative
``build-kokkos-gpu`` path resolve:

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

``-DOTI_BUILD_TESTS=OFF`` keeps this build to the Kokkos smoke test. The scalar
tests are host code and gain nothing from being compiled through
``nvcc_wrapper``, so the GPU CI job leaves them off too.

Verify The Backend
------------------

Still from the repository root, CTest should report two passing tests -- the
smoke test and the device validity test:

.. code-block:: text

   100% tests passed, 0 tests failed out of 2

That test, ``test_kokkos_smoke``, is the same executable the CPU tutorial
builds, but compiled for CUDA: it launches the smoke-test kernel on the device,
copies the resulting OTI coefficients back to the host, and checks them. Run it
directly to see its output:

.. code-block:: console

   ./build-kokkos-gpu/test_kokkos_smoke

.. code-block:: text

   Kokkos otinum kernel tests passed

.. note::

   As in the CPU tutorial, a result of 16 scalar tests with no
   ``test_kokkos_smoke`` means this build directory was configured without the
   Kokkos flags, so nothing ran on the device. Delete ``build-kokkos-gpu`` and
   re-run the configure on a clean directory.

Running A Kernel On The GPU
---------------------------

The smoke test proves the backend works; the kernel from
:ref:`tutorials/kokkos_cpu:Writing Your Own Kernel` runs on the GPU with **no
source changes at all**. Take that project's ``main.cpp`` and ``CMakeLists.txt``
unchanged, and configure it with the CUDA wrapper and prefix instead of the
OpenMP ones:

.. code-block:: console

   cmake -S . -B build \
     -DCMAKE_CXX_COMPILER=/tmp/kokkos/bin/nvcc_wrapper \
     -DCMAKE_PREFIX_PATH=/tmp/kokkos-install-cuda
   cmake --build build
   ./build/jet_grid

The ``parallel_for`` body now executes on the GPU, and the output is identical
to the CPU run to the last bit:

.. code-block:: text

   x=0  f=1  f'=1  f''=1
   x=0.1  f=1.10499  f'=1.09947  f''=0.983659
   x=0.2  f=1.21978  f'=1.19546  f''=0.929302
   ...

That portability is the whole point: ``otinum`` is a trivially copyable value
type backed by ``Kokkos::Array`` under ``OTI_ENABLE_KOKKOS``, so the same jet
arithmetic compiles for host or device depending only on the Kokkos install you
point CMake at.

Choosing The Architecture Flag
------------------------------

The ``Kokkos_ARCH_*`` option should match the GPU where the test will run. The
flag name is derived from the GPU's *compute capability*, which you can read
directly:

.. code-block:: console

   nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

For a GTX 1650 this prints ``NVIDIA GeForce GTX 1650, 7.5``. Map the compute
capability to the Kokkos flag:

.. list-table::
   :header-rows: 1

   * - Compute capability
     - Kokkos flag
     - Example GPUs
   * - 6.1
     - ``Kokkos_ARCH_PASCAL61``
     - GTX 10-series
   * - 7.0
     - ``Kokkos_ARCH_VOLTA70``
     - V100
   * - 7.5
     - ``Kokkos_ARCH_TURING75``
     - GTX 1650, RTX 20-series
   * - 8.0
     - ``Kokkos_ARCH_AMPERE80``
     - A100
   * - 8.6
     - ``Kokkos_ARCH_AMPERE86``
     - RTX 30-series
   * - 8.9
     - ``Kokkos_ARCH_ADA89``
     - RTX 40-series
   * - 9.0
     - ``Kokkos_ARCH_HOPPER90``
     - H100
   * - 12.0
     - ``Kokkos_ARCH_BLACKWELL120``
     - RTX 50-series

So the GTX 1650 (7.5) uses ``-DKokkos_ARCH_TURING75=ON``, and an RTX 50-series
card (12.0) uses ``-DKokkos_ARCH_BLACKWELL120=ON``. A specific architecture
flag is preferable for reproducible local builds; for generic CI environments,
``Kokkos_ARCH_NATIVE=ON`` can detect a visible GPU when the installed Kokkos
version supports it.

The newest architectures also need a recent enough CUDA toolkit *and* Kokkos
version — the compute capability must be known to both. The Blackwell flags,
for instance, exist only in recent Kokkos (5.x) and require a CUDA toolkit that
targets ``sm_120`` (CUDA 12.8 or newer); pairing a new GPU with an older
toolkit produces ``nvcc`` errors about an unknown architecture.

If the architecture flag is wrong, failures often appear as CUDA compilation
errors, unsupported instruction errors, or runtime launch failures (commonly
``no kernel image is available for execution on the device``).

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

If the Kokkos CUDA configure stops with ``CMake wants to use -std=c++2a which
is not supported by NVCC``, the host compiler is too old: it spells C++20 as
``-std=c++2a``, which Kokkos rejects for NVCC. Install a newer host compiler
and point ``nvcc_wrapper`` at it as shown in
`Installing a new-enough host compiler`_, then re-run the configure on a clean
build directory.

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
