Consuming The CMake Package
===========================

The :ref:`installation:Header-Only C++` workflow passes ``-I`` flags to the
compiler by hand. That is fine for a single source file, but it does not scale
to a real project: every target needs the same setup, and optional integration
requirements are easy to miss. CMake's ``find_package`` mechanism solves this —
your project asks for ``otinum`` once, and the imported target carries the
installed include directory and the usage requirements recorded by that package,
including any optional dependencies it was built with.

This tutorial builds a complete, separate consumer project against an
installed copy of the library.

Install The Package
-------------------

From the ``cpp_oti_lib`` repository root, install the headers and the CMake
package files to a prefix. Any writable directory works; this example uses
``/tmp/otinum-install``:

.. code-block:: console

   cmake -S . -B build-install -DOTI_BUILD_TESTS=OFF
   cmake --install build-install --prefix /tmp/otinum-install

``-DOTI_BUILD_TESTS=OFF`` turns off the unit tests, which are otherwise
configured by default — the library itself is header-only, so installing
requires no compilation at all, and skipping the tests keeps the configure
step minimal.

The installed tree is small — the ``include/otinum`` headers plus the package
files under ``lib/cmake/otinum`` that make ``find_package`` work.

The Consumer Project
--------------------

Create a project directory anywhere outside the repository with two files:

.. code-block:: text

   oti_consumer/
     CMakeLists.txt
     main.cpp

``CMakeLists.txt`` declares the dependency and links the imported target:

.. code-block:: cmake

   cmake_minimum_required(VERSION 3.18)
   project(falling_sphere CXX)

   find_package(otinum CONFIG REQUIRED)

   add_executable(falling_sphere main.cpp)
   target_link_libraries(falling_sphere PRIVATE oti::otinum)

Note what is *absent*: no manually specified include directory or duplicated
package configuration. The ``oti::otinum`` target supplies those usage
requirements.

``main.cpp`` computes the terminal velocity of a falling sphere under
quadratic drag and its sensitivities to mass and radius:

.. code-block:: cpp

   #include <iostream>

   #include "otinum/otinum.hpp"

   // Terminal velocity of a falling sphere under quadratic drag,
   //   v(m, r) = sqrt(2 m g / (rho cd pi r^2)),
   // differentiated with respect to mass and radius.
   int main()
   {
       using T = oti::otinum<2, 1>;

       double const g = 9.81;
       double const rho = 1.225;
       double const cd = 0.47;
       double const pi = 3.14159265358979323846;

       T m = T::variable(0, 0.5);   // mass [kg]
       T r = T::variable(1, 0.11);  // radius [m]

       T v = oti::sqrt(2.0 * m * g / (rho * cd * pi * r * r));

       std::cout << "v       = " << v.real() << " m/s\n";
       std::cout << "dv/dm   = " << v.partial({1, 0}) << " (m/s)/kg\n";
       std::cout << "dv/dr   = " << v.partial({0, 1}) << " (m/s)/m\n";
       return 0;
   }

Configure, Build, Run
---------------------

CMake works in two phases, which is why two ``cmake`` commands follow. The
first (*configure*) compiles nothing: it reads ``CMakeLists.txt``, locates the
compiler, resolves ``find_package(otinum)``, and generates a native build
system (Makefiles on Linux) into the build directory. The second
(``--build``) drives that generated build system to actually compile and
link. You configure once and re-run only the build step as you edit sources;
re-configuring is needed only when the project structure or options change.

Point CMake at the installation prefix when configuring — resolving packages
is a configure-phase job, so this is the only place the install location
appears:

.. code-block:: console

   cmake -S oti_consumer -B oti_consumer/build \
     -DCMAKE_PREFIX_PATH=/tmp/otinum-install
   cmake --build oti_consumer/build
   ./oti_consumer/build/falling_sphere

Expected output:

.. code-block:: text

   v       = 21.1714 m/s
   dv/dm   = 21.1714 (m/s)/kg
   dv/dr   = -192.467 (m/s)/m

Both sensitivities are easy to confirm analytically: for this ``v``,
``dv/dm = v / (2 m)`` and ``dv/dr = -v / r``.

Practical Notes
---------------

* ``CMAKE_PREFIX_PATH`` is only needed for a non-standard prefix like
  ``/tmp/otinum-install``. Installing to a standard location (for example
  ``--prefix /usr/local``, or omitting ``--prefix`` entirely) lets
  ``find_package(otinum)`` succeed with no extra configure arguments.
* The package is relocatable: you can copy or move the installed prefix and
  only the ``CMAKE_PREFIX_PATH`` value changes.
* If the library was configured with ``-DOTI_ENABLE_KOKKOS=ON`` before
  installing, the installed package also records the ``Kokkos::kokkos``
  dependency, so consumers get Kokkos include paths and libraries through the
  same single ``target_link_libraries`` line.
