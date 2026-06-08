Public Math Functions
=====================

This page is generated from ``include/otinum/functions.hpp``. These overloads
lift familiar scalar functions into the truncated Taylor algebra.

Analytic Functions
------------------

Use the ``oti`` overloads the same way you would use scalar ``std`` functions.
The function is evaluated on the real coefficient and its Taylor expansion is
applied to the nilpotent part of the OTI value:

.. code-block:: cpp

   using T = oti::otinum<1, 3>;

   T x = T::variable(0, 0.5);
   T f = oti::exp(x) * oti::sin(x);

   double value = f.real();
   double first = f.partial({1});
   double second = f.partial({2});
   double third = f.partial({3});

Scalar Powers
-------------

Use scalar ``pow`` for expressions such as square roots, inverse powers, and
non-integer exponents:

.. code-block:: cpp

   using T = oti::otinum<1, 3>;

   T x = T::variable(0, 4.0);
   T sqrt_x = oti::sqrt(x);
   T x_to_three_halves = oti::pow(x, 1.5);
   T reciprocal = oti::pow(x, -1.0);

Two-Argument OTI Powers
-----------------------

When both base and exponent are OTI values, ``pow(lhs, rhs)`` is evaluated as
``exp(rhs * log(lhs))``. This inherits the real-valued logarithm requirement on
the base:

.. code-block:: cpp

   using T = oti::otinum<2, 2>;

   T x = T::variable(0, 2.0);
   T y = T::variable(1, 0.25);
   T f = oti::pow(x, y);

Domain Behavior
---------------

Domain behavior follows the underlying scalar math at ``value.real()``. For
example, ``log(value)`` requires a positive real part for real-valued results,
and ``tan(value)`` is singular where ``cos(value.real())`` is zero.

Generated Reference
-------------------

.. doxygenfile:: functions.hpp
   :sections: func
