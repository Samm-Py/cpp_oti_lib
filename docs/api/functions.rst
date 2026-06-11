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

Inverse Trigonometric Functions
-------------------------------

``atan``, ``asin``, and ``acos`` use the same Taylor-composition machinery as
the other analytic functions:

.. code-block:: cpp

   using T = oti::otinum<1, 2>;

   T x = T::variable(0, 0.4);
   T angle = oti::atan(x);
   T recovered = oti::asin(oti::sin(x));

``atan`` is analytic for every finite real input. ``asin`` and ``acos`` are
real-valued for ``-1 <= real() <= 1``; their derivatives are singular at the
endpoints and invalid outside that interval.

Two-Argument Arctangent
-----------------------

``atan2(y, x)`` supports OTI/OTI and mixed OTI/scalar arguments. Its real
coefficient uses the quadrant-aware scalar ``atan2`` value:

.. code-block:: cpp

   using T = oti::otinum<2, 1>;

   T y = T::variable(0, 1.0);
   T x = T::variable(1, 2.0);
   T angle = oti::atan2(y, x);

   double dangle_dy = angle.partial({1, 0});
   double dangle_dx = angle.partial({0, 1});

For ``x.real() != 0``, derivatives agree with the local derivative of
``atan(y / x)`` while the real coefficient retains the correct quadrant. On the
``y`` axis, the real value is still correct but derivative coefficients are
``nan`` under the library's singular-point convention.

Domain Behavior
---------------

The value of every function follows the underlying scalar ``std`` (or
``Kokkos``) math evaluated at ``value.real()``. The derivatives are the Taylor
expansion of that function about ``value.real()``, so they can be infinite where
the function has a vertical tangent and ``nan`` where it has no derivative at
all. No exceptions are thrown: out-of-domain inputs propagate ``inf``/``nan``
exactly as the scalar ``<cmath>`` calls would.

Functions that are finite for every real input -- ``exp``, ``sin``, ``cos``,
``sinh``, ``cosh``, ``tanh``, and ``pow`` with an integer exponent -- are not
listed. The table gives only the inputs where the value or its derivatives stop
being finite, with the value and the derivatives shown in separate columns. Read
each row as: for this function, at this input, the value is X and the
derivatives are Y.

.. list-table:: Where the value or the derivatives are not finite
   :header-rows: 1
   :widths: 30 26 14 18

   * - Function(s)
     - At this input
     - Value
     - Derivatives
   * - ``log``, ``log10``, ``log_base``
     - ``real() == 0`` (pole)
     - ``-inf``
     - ``nan``
   * - ``log``, ``log10``, ``log_base``
     - ``real() < 0``
     - ``nan``
     - ``nan``
   * - ``sqrt``
     - ``real() == 0`` (vertical tangent)
     - ``0``
     - ``nan``
   * - ``sqrt``
     - ``real() < 0``
     - ``nan``
     - ``nan``
   * - ``cbrt``
     - ``real() == 0`` (vertical tangent)
     - ``0``
     - ``nan``
   * - ``asin``, ``acos``
     - ``abs(real()) == 1`` (vertical tangent)
     - finite
     - ``nan``
   * - ``asin``, ``acos``
     - ``abs(real()) > 1``
     - ``nan``
     - ``nan``
   * - ``atan2(y, x)``
     - ``x.real() == 0``
     - quadrant-aware finite value
     - ``nan``
   * - ``pow(x, p)``, non-integer ``p``
     - ``real() < 0``
     - ``nan``
     - ``nan``
   * - ``pow(x, y)``, both OTI
     - ``x.real() <= 0``
     - ``nan``
     - ``nan``
   * - ``tan``
     - ``cos(real()) == 0`` (pole)
     - ``inf``
     - ``nan``
   * - ``inv``, ``operator/``
     - divisor ``real() == 0`` (pole)
     - ``inf``
     - ``nan``
   * - ``abs``
     - ``real() == 0``, higher-order part nonzero (kink)
     - ``0``
     - ``nan``

The derivatives follow a single rule: they are finite where the function is
analytic and ``nan`` at every singularity -- a pole, a vertical tangent, or a
kink. The value is independent and is whatever the scalar function gives:
finite (``sqrt(0)`` is ``0``), ``-inf`` (``log(0)``), ``inf`` (``inv(0)``), or
``nan`` (``log(-1)``). A ``nan`` value always forces the whole result to ``nan``.

.. note::

   ``pow(x, p)`` with ``real() < 0`` and a non-integer ``p`` returns ``nan`` by
   design, matching ``std::pow``. A ``double`` exponent cannot encode whether the
   intended power is a real odd root (e.g. ``x^(1/3)``) or a complex even root
   (e.g. ``x^(1/2)``), so the real branch is not guessed. Use ``cbrt`` for the
   real cube root of negative numbers. At ``real() == 0`` a non-integer ``pow``
   behaves like ``sqrt``: a finite value with ``nan`` derivatives (vertical
   tangent).

Generated Reference
-------------------

.. doxygenfile:: functions.hpp
   :sections: func
