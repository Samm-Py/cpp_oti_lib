Project Overview
================

The root ``README.md`` is the short project entry point. It covers the same
core ideas as this documentation:

* ``oti::otinum<M, N, Coeff = double>`` stores all Taylor coefficients for
  ``M`` variables through total order ``N``.
* Coefficients are stored in graded multi-index order, with the real value at
  flat index zero.
* ``partial(alpha)`` returns ordinary derivative values, while
  ``coeff(alpha)`` returns normalized Taylor coefficients.
* The default coefficient type is ``double``. Use ``float`` explicitly when
  single-precision storage and math are desired.

The README remains useful as a compact command reference. These Sphinx pages
expand it into tutorial workflows and build/reporting notes.

For details on normalized coefficients, multi-index access, and low-level
coefficient layout, see :doc:`api/core`.

Minimal Example
---------------

.. code-block:: cpp

   #include <iostream>

   #include "otinum/otinum.hpp"

   int main()
   {
       using T = oti::otinum<2, 2>;

       T x = T::variable(0, 1.5);
       T y = T::variable(1, 0.3);
       T f = oti::sin(x * y) + oti::exp(x);

       std::cout << "f        = " << f.real() << '\n';
       std::cout << "df/dx    = " << f.partial({1, 0}) << '\n';
       std::cout << "df/dy    = " << f.partial({0, 1}) << '\n';
       std::cout << "d2f/dxdy = " << f.partial({1, 1}) << '\n';
   }

Compile with the repository ``include`` directory on the include path:

.. code-block:: console

   c++ -std=c++17 -I include my_program.cpp -o /tmp/my_program
   /tmp/my_program

Where To Go Next
----------------

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
