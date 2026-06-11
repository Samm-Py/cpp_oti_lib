Kokkos CPU Tutorial
===================

Kokkos support is optional. The header-only scalar type uses ``std::array`` in
ordinary C++ builds and ``Kokkos::Array`` when ``OTI_ENABLE_KOKKOS`` is enabled.
The same OTI arithmetic code is used, but Kokkos supplies the array type,
execution-space annotations, backend compile flags, and link flags.

This tutorial builds the Kokkos smoke test with an OpenMP-backed Kokkos
installation. It is the most useful local Kokkos path when you want to check
the ``OTI_ENABLE_KOKKOS`` headers without needing a GPU.

What Gets Tested
----------------

The ``test_kokkos_smoke`` executable launches a Kokkos kernel and evaluates OTI
expressions inside that kernel. It then copies the coefficients back to the
host and compares them with host-computed values. The test exercises several
template shapes, arithmetic operations, elementary functions, truncation, and
the ``Kokkos::Array`` compatibility path.

The expected successful test output includes:

.. code-block:: text

   Kokkos otinum kernel tests passed

Prerequisites
-------------

You need:

* CMake 3.18 or newer
* a C++17 compiler supported by the Kokkos version you are building
* a Kokkos source checkout or existing Kokkos install
* OpenMP support in the selected compiler

Use the same compiler family for Kokkos and ``cpp_oti_lib``. For example, if
Kokkos is installed with ``g++-10``, configure ``cpp_oti_lib`` with ``g++-10``
as well.

Install Or Build Kokkos
-----------------------

Skip this section if you already have a Kokkos installation and know its
install prefix. Otherwise, build a CPU-backed Kokkos installation with OpenMP:

.. code-block:: console

   git clone --branch 5.1.1 --depth 1 \
     https://github.com/kokkos/kokkos.git /tmp/kokkos

   cmake -S /tmp/kokkos -B /tmp/kokkos-build-openmp \
     -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_INSTALL_PREFIX=/tmp/kokkos-install-openmp \
     -DCMAKE_CXX_COMPILER=g++ \
     -DKokkos_ENABLE_OPENMP=ON \
     -DKokkos_ARCH_NATIVE=ON

   cmake --build /tmp/kokkos-build-openmp --target install --parallel

The install step writes the CMake package files that ``find_package(Kokkos)``
uses later. In this example, those files are under
``/tmp/kokkos-install-openmp``.

Configure cpp_oti_lib
---------------------

Point ``CMAKE_PREFIX_PATH`` at the Kokkos install. This lets CMake find the
``Kokkos::kokkos`` target, which carries the required include directories,
compile definitions, backend flags, and link flags.

.. code-block:: console

   cmake -S . -B build-kokkos-cpu \
     -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_CXX_COMPILER=g++ \
     -DCMAKE_PREFIX_PATH=/tmp/kokkos-install-openmp \
     -DOTI_ENABLE_KOKKOS=ON \
     -DOTI_BUILD_PYTHON=OFF \
     -DOTI_BUILD_TESTS=OFF

   cmake --build build-kokkos-cpu --parallel
   ctest --test-dir build-kokkos-cpu --output-on-failure

``-DOTI_BUILD_TESTS=OFF`` is optional, but it keeps this build focused on the
Kokkos smoke test. The regular scalar tests are already covered by the normal
C++ build.

Verify The Backend
------------------

CTest should report one passing test:

.. code-block:: text

   100% tests passed, 0 tests failed out of 1

If you want to run the executable directly, use the path inside the build
directory:

.. code-block:: console

   ./build-kokkos-cpu/test_kokkos_smoke

Troubleshooting
---------------

If CMake cannot find Kokkos, check that ``CMAKE_PREFIX_PATH`` points at the
install prefix, not the Kokkos source tree or build tree.

If the build fails with missing Kokkos headers or unresolved Kokkos symbols,
make sure you are using the CMake workflow above. The direct
``tests/run_unit_tests.sh`` script intentionally does not discover or link a
Kokkos package.

If Kokkos itself fails to configure, check the compiler version and OpenMP
support first. Kokkos versions can have stricter compiler requirements than
this header-only library.

CI Coverage
-----------

The project CI runs this OpenMP-backed Kokkos smoke test separately from the
ordinary scalar tests. The documentation coverage job also builds an OpenMP
Kokkos configuration with coverage flags so the ``OTI_ENABLE_KOKKOS`` header
path appears in the generated coverage report.
