Kokkos CPU Tutorial
===================

Kokkos support is optional. The header-only scalar type uses ``std::array`` in
ordinary C++ builds and ``Kokkos::Array`` when ``OTI_ENABLE_KOKKOS`` is enabled.
The same OTI arithmetic code is used, but Kokkos supplies the array type,
execution-space annotations, backend compile flags, and link flags.

This tutorial builds the Kokkos smoke test with an OpenMP-backed Kokkos
installation. It is the most useful local Kokkos path when you want to check
the ``OTI_ENABLE_KOKKOS`` headers without needing a GPU.

Prerequisites
-------------

You need:

* CMake 3.18 or newer
* a compiler supported by the Kokkos version you are building — Kokkos is much
  stricter than this library: Kokkos 5.1.1 requires GCC 10.4 or Clang 14 at
  minimum, so a distribution default like GCC 9 that compiles the scalar
  library fine will be rejected by Kokkos
* a Kokkos source checkout or existing Kokkos install
* OpenMP support in the selected compiler

On Ubuntu, a suitable compiler is one package away if the default is too old:

.. code-block:: console

   sudo apt-get install g++-10

Use the same compiler for Kokkos and ``cpp_oti_lib``: whatever
``-DCMAKE_CXX_COMPILER`` value you give the Kokkos configure below, give the
``cpp_oti_lib`` configure the same one.

Install Or Build Kokkos
-----------------------

Skip this section if you already have a Kokkos installation and know its
install prefix. Otherwise, build a CPU-backed Kokkos installation with OpenMP:

.. code-block:: console

   git clone --branch 5.1.1 --depth 1 \
     https://github.com/kokkos/kokkos.git /tmp/kokkos

   # If your default g++ is older than 10.4 (see Prerequisites), replace
   # g++ with an explicit newer compiler such as g++-10 here and in every
   # later configure command.
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
     -DOTI_BUILD_TESTS=OFF

   cmake --build build-kokkos-cpu --parallel
   ctest --test-dir build-kokkos-cpu --output-on-failure

``-DOTI_BUILD_TESTS=OFF`` keeps this build to the Kokkos smoke test, the
focused check for a Kokkos configure, and the steps below assume it. Setting
it to ``ON`` instead also builds the full scalar test suite against the
``Kokkos::Array`` backing; the project's OpenMP CI job does exactly that, so
that storage path is exercised by every test and not only the smoke test.

Verify The Backend
------------------

CTest should report one passing test:

.. code-block:: text

   100% tests passed, 0 tests failed out of 1

If CTest instead lists 15 scalar tests (``test_abs_large_shapes`` and the
rest) with no ``test_kokkos_smoke``, this build directory was configured
earlier *without* the Kokkos flags — for example by a plain
``cmake -S . -B build-kokkos-cpu``, which defaults to ``OTI_BUILD_TESTS=ON``
and ``OTI_ENABLE_KOKKOS=OFF``. CTest reflects the most recent configure of a
directory, so delete ``build-kokkos-cpu`` and re-run the configure above on a
clean directory.

The test that just passed — ``test_kokkos_smoke`` — launches a Kokkos kernel,
evaluates OTI expressions inside it, copies the coefficients back to the
host, and compares them with host-computed values. It exercises several
template shapes, arithmetic operations, elementary functions, truncation, and
the ``Kokkos::Array`` compatibility path.

If you want to run the executable directly, use the path inside the build
directory:

.. code-block:: console

   ./build-kokkos-cpu/test_kokkos_smoke

The expected output is:

.. code-block:: text

   Kokkos otinum kernel tests passed

A direct run may print a ``Kokkos::OpenMP::initialize WARNING`` about
``OMP_PROC_BIND`` not being set. That is a performance hint, not an error —
the test result is unaffected. Set ``OMP_PROC_BIND=spread OMP_PLACES=threads``
in the environment to silence it (and to pin threads sensibly for real
workloads).

Writing Your Own Kernel
-----------------------

The smoke test proves the backend works; this section shows what using it
looks like. The program below fills a device array with the jets of
``f(x) = exp(sin(x))`` over a small grid — one grid point per thread — then
copies them back and reads the derivatives on the host.

Create a project directory with the two files side by side:

.. code-block:: text

   oti_kokkos_demo/
     CMakeLists.txt
     main.cpp

``main.cpp``:

.. code-block:: cpp

   #include <iostream>

   #include <Kokkos_Core.hpp>

   #include "otinum/otinum.hpp"

   int main(int argc, char* argv[])
   {
       Kokkos::ScopeGuard guard(argc, argv);

       using T = oti::otinum<1, 2>;
       int const n = 8;

       // A device array of jets, one per grid point.
       Kokkos::View<T*> ys("ys", n);

       // Evaluate f and its first two derivatives in parallel.
       Kokkos::parallel_for("evaluate", n, KOKKOS_LAMBDA(int i) {
           T x = T::variable(0, 0.1 * i);
           ys(i) = oti::exp(oti::sin(x));
       });

       // Copy the jets back and read the derivatives on the host.
       auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, ys);
       for (int i = 0; i < n; ++i) {
           std::cout << "x=" << 0.1 * i
                     << "  f=" << host(i).real()
                     << "  f'=" << host(i).partial({1})
                     << "  f''=" << host(i).partial({2}) << '\n';
       }
       return 0;
   }

``CMakeLists.txt`` — using the checkout's include directory directly, in
which case the ``OTI_ENABLE_KOKKOS`` definition must be set by hand (a
consumer of an installed Kokkos-enabled package via ``find_package(otinum)``
gets both the definition and the Kokkos dependency automatically):

.. code-block:: cmake

   cmake_minimum_required(VERSION 3.18)
   project(oti_kokkos_demo CXX)

   find_package(Kokkos REQUIRED)

   add_executable(jet_grid main.cpp)
   target_include_directories(jet_grid PRIVATE /path/to/cpp_oti_lib/include)
   target_compile_definitions(jet_grid PRIVATE OTI_ENABLE_KOKKOS)
   target_link_libraries(jet_grid PRIVATE Kokkos::kokkos)

From inside the ``oti_kokkos_demo`` directory, configure with the same
compiler and Kokkos prefix as before, build, and run. ``jet_grid`` is the
executable name from ``add_executable`` above:

.. code-block:: console

   # Use the same compiler that built Kokkos (g++-10 in the Prerequisites
   # example). Configuring with an older g++ can succeed here but then fail
   # to compile the Kokkos headers.
   cmake -S . -B build \
     -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_CXX_COMPILER=g++ \
     -DCMAKE_PREFIX_PATH=/tmp/kokkos-install-openmp
   cmake --build build
   ./build/jet_grid

Expected output (the first row is easy to check by hand: at ``x = 0``,
``f``, ``f'``, and ``f''`` of ``exp(sin(x))`` are all exactly ``1``):

.. code-block:: text

   x=0  f=1  f'=1  f''=1
   x=0.1  f=1.10499  f'=1.09947  f''=0.983659
   x=0.2  f=1.21978  f'=1.19546  f''=0.929302
   ...

Two Kokkos-specific points worth noticing:

* The kernel body is ordinary OTI code — the same ``T::variable`` /
  ``oti::exp`` calls as the scalar tutorials. ``KOKKOS_LAMBDA`` and the
  ``View`` are the only Kokkos-isms.
* ``Kokkos::View<T*>`` of OTI values works directly because ``otinum`` is a
  trivially copyable value type backed by ``Kokkos::Array`` in this build.
  The same program recompiles unchanged against a CUDA-backed Kokkos — that
  is the subject of :doc:`kokkos_gpu`.

Consuming It As An Installed Package
------------------------------------

The example above puts the checkout's include directory on one target by hand.
A larger project is usually cleaner consuming an *installed* otinum package
with :doc:`cmake_package`-style ``find_package``. The only twist for the
Kokkos case is that the package must be installed with Kokkos enabled, and the
consumer then needs two install prefixes on its search path.

The ``build-kokkos-cpu`` directory from `Configure cpp_oti_lib`_ is already a
Kokkos-enabled build, so no second configure is needed — install it to a
prefix directly:

.. code-block:: console

   cmake --install build-kokkos-cpu --prefix /tmp/otinum-install-kokkos

Because that build had ``OTI_ENABLE_KOKKOS=ON``, the install records otinum's
Kokkos dependency.

The consumer ``CMakeLists.txt`` then mentions only otinum — no
``find_package(Kokkos)``, no ``OTI_ENABLE_KOKKOS`` definition, no include
path. The installed ``oti::otinum`` target carries the include directory, the
definition, and a transitive ``Kokkos::kokkos`` link, and the generated
``otinumConfig.cmake`` pulls Kokkos in with ``find_dependency`` on its own:

.. code-block:: cmake

   cmake_minimum_required(VERSION 3.18)
   project(oti_kokkos_demo CXX)

   find_package(otinum CONFIG REQUIRED)

   add_executable(jet_grid main.cpp)
   target_link_libraries(jet_grid PRIVATE oti::otinum)

Because the consumer must locate both the otinum package and the Kokkos
package that otinum depends on, give ``CMAKE_PREFIX_PATH`` *both* install
prefixes, separated by a semicolon:

.. code-block:: console

   cmake -S . -B build \
     -DCMAKE_CXX_COMPILER=g++ \
     -DCMAKE_PREFIX_PATH="/tmp/otinum-install-kokkos;/tmp/kokkos-install-openmp"
   cmake --build build
   ./build/jet_grid

The output is identical to the direct-include build above. As everywhere on
this page, use the same compiler that built Kokkos in place of ``g++`` if your
default is too old.

Troubleshooting
---------------

If CMake cannot find Kokkos, check that ``CMAKE_PREFIX_PATH`` points at the
install prefix, not the Kokkos source tree or build tree.

If a configure that previously worked suddenly reports ``Could not find a
package configuration file provided by "Kokkos"`` right after printing ``You
have changed variables that require your cache to be deleted``, the cause is a
reused build directory, not a missing install. Changing
``-DCMAKE_CXX_COMPILER`` on an existing build directory makes CMake wipe and
re-run the configure, and on that forced re-run the command-line
``CMAKE_PREFIX_PATH`` is dropped, so Kokkos is no longer found. Always start
from a clean build directory when changing the compiler:

.. code-block:: console

   rm -rf build
   cmake -S . -B build -DCMAKE_CXX_COMPILER=g++-10 \
     -DCMAKE_PREFIX_PATH=/tmp/kokkos-install-openmp ...

If the build fails with missing Kokkos headers or unresolved Kokkos symbols,
make sure you are using the CMake workflow above. The direct
``tests/run_unit_tests.sh`` script intentionally does not discover or link a
Kokkos package.

If ``find_package(otinum)`` reports ``Could not find a package configuration
file provided by "otinum"``, the library has not been installed, or its
install prefix is not on ``CMAKE_PREFIX_PATH``. Install it first (see
`Consuming It As An Installed Package`_) and remember that the Kokkos build
needs *both* the otinum and Kokkos prefixes on ``CMAKE_PREFIX_PATH``.

If the Kokkos configure stops with ``Compiler not supported by Kokkos`` and a
table of minimum versions, your default compiler is older than Kokkos
accepts. Install a supported compiler (see Prerequisites), delete the Kokkos
build directory so the cached compiler choice is forgotten, and re-run the
configure with ``-DCMAKE_CXX_COMPILER`` pointing at the new compiler — for
example ``g++-10``. Remember to pass the same compiler to the ``cpp_oti_lib``
configure afterwards.
