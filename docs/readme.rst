Minimal C++ Example
===================

This page is the shortest complete C++ program in the documentation. Use it
when you want to check that the header include path is correct and that a small
OTI calculation compiles.

For a fuller setup guide, including CMake builds, CTest, Python bindings, and
Kokkos builds, see :doc:`installation`.

The core ideas are:

* ``oti::otinum<M, N, Coeff = double>`` stores all Taylor coefficients for
  ``M`` variables through total order ``N``.
* Coefficients are stored in graded multi-index order, with the real value at
  flat index zero.
* ``partial(alpha)`` returns ordinary derivative values, while
  ``coeff(alpha)`` returns normalized Taylor coefficients.
* ``T::variable(i, value)`` seeds variable ``i`` at the supplied real value.
* The default coefficient type is ``double``. Use ``float`` explicitly for
  single-precision storage and math.

For details on normalized coefficients, multi-index access, and low-level
coefficient layout, see :doc:`api/core`.

Program
-------

The same source is available in the repository as ``examples/minimal.cpp``.

.. literalinclude:: ../examples/minimal.cpp
   :language: cpp

Compile And Run
---------------

From the repository root, compile the example with the local ``include``
directory on the compiler include path:

.. code-block:: console

   cd /root/Research/cpp_oti_lib
   c++ -std=c++17 -I include examples/minimal.cpp -o /tmp/oti_minimal
   /tmp/oti_minimal

Expected output is approximately:

.. code-block:: console

   f        = 4.91665
   df/dx    = 4.75182
   df/dy    = 1.35067
   d2f/dxdy = 0.704713

The exact formatting can vary slightly by standard library and compiler.

The important compiler option is ``-I include``. It points the compiler at this
repository's headers so ``#include "otinum/otinum.hpp"`` can be resolved. There
is no separate library binary to link for the header-only scalar C++ path.

Where To Go Next
----------------

* :doc:`installation` covers local builds, CTest, Python bindings, and Kokkos.
* :doc:`tutorials/basic_usage` shows a fuller version of this example with
  analytic derivative checks.
* :doc:`api/core` documents coefficient access, setters, type properties, and
  the flat coefficient layout.
* :doc:`tutorials/float_coefficients` shows how to use ``float`` coefficient
  storage.
* :doc:`tutorials/directional_derivatives` shows how to seed a direction vector
  and compute directional derivatives.
* :doc:`tutorials/kokkos_cpu` and :doc:`tutorials/kokkos_gpu` cover Kokkos
  builds.
