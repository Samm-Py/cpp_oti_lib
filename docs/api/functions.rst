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

The value of every function follows the underlying scalar ``std`` (or
``Kokkos``) math evaluated at ``value.real()``. The derivatives are the Taylor
expansion of that function about ``value.real()``, so they can be infinite where
the function has a vertical tangent and ``nan`` where it has no derivative at
all. No exceptions are thrown: out-of-domain inputs propagate ``inf``/``nan``
exactly as the scalar ``<cmath>`` calls would.

The table summarizes where the value and/or the derivatives are not finite.
"Smooth everywhere" means both the value and all derivatives are defined for
every real input.

.. list-table:: Function and derivative behavior at domain edges
   :header-rows: 1
   :widths: 26 20 54

   * - Function(s)
     - Real-valued domain
     - Behavior at the edge of the domain
   * - ``exp``, ``sin``, ``cos``, ``sinh``, ``cosh``, ``tanh``
     - all reals
     - Smooth everywhere; no special cases.
   * - ``log``, ``log10``, ``log_base``
     - ``real() > 0``
     - ``real() == 0``: value ``-inf``, derivatives ``inf``/``nan``.
       ``real() < 0``: ``nan`` (the real logarithm is undefined).
   * - ``sqrt``
     - ``real() >= 0``
     - ``real() == 0``: value ``0`` with ``inf`` derivatives (vertical tangent).
       ``real() < 0``: ``nan`` (no real square root).
   * - ``cbrt``
     - all reals
     - Defined for negative inputs (real cube root). ``real() == 0``: value
       ``0`` with ``inf`` derivatives (vertical tangent).
   * - ``pow(x, p)`` (scalar ``p``)
     - depends on ``p``
     - Integer ``p``: defined for all reals. Non-integer ``p`` with
       ``real() < 0``: ``nan`` (see note below). ``real() == 0`` with a negative
       effective power (e.g. ``p`` between 0 and 1): ``inf`` derivatives.
   * - ``pow(x, y)`` (both OTI)
     - ``x.real() > 0``
     - Evaluated as ``exp(y * log(x))``, so it inherits the logarithm's domain:
       ``x.real() <= 0`` gives ``nan``.
   * - ``tan``
     - ``cos(real()) != 0``
     - Singular at odd multiples of ``pi/2``. As ``cos(real())`` approaches zero
       the value and derivatives grow without bound (``inf`` when it is exactly
       zero).
   * - ``inv``, ``operator/``
     - divisor ``real() != 0``
     - Divisor ``real() == 0``: value ``inf``, derivatives ``inf``/``nan``.
   * - ``abs``
     - all reals (value)
     - Smooth for ``real() != 0``. At an exact inner zero (``real() == 0`` with a
       nonzero higher-order part) the value is ``0`` but the derivatives are
       ``nan``: ``abs`` is not differentiable there, so the library signals this
       rather than guessing a subgradient. A ``nan`` real part propagates.

.. note::

   ``pow(x, p)`` with ``real() < 0`` and a non-integer ``p`` returns ``nan`` by
   design, matching ``std::pow``. A ``double`` exponent cannot encode whether the
   intended power is a real odd root (e.g. ``x^(1/3)``) or a complex even root
   (e.g. ``x^(1/2)``), so the real branch is not guessed. Use ``cbrt`` for the
   real cube root of negative numbers.

Generated Reference
-------------------

.. doxygenfile:: functions.hpp
   :sections: func
