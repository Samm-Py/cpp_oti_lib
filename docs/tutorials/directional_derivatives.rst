Directional Derivatives
=======================

OTI variables do not have to correspond one-to-one with physical variables.
They can also represent perturbation parameters. This lets you compute
directional derivatives by embedding a line through the physical input space.

For a scalar function ``f(x, y)`` at point ``(x0, y0)`` and direction
``v = (vx, vy)``, define:

.. code-block:: text

   x(t) = x0 + vx t
   y(t) = y0 + vy t

Evaluating ``f(x(t), y(t))`` with ``otinum<1, N>`` stores derivatives with
respect to the single parameter ``t``. The first derivative is
``grad f(x0, y0) dot v``. The second derivative is ``v^T H f(x0, y0) v``.

Complete Example
----------------

This example uses:

.. code-block:: text

   f(x, y) = sin(x y) + exp(x)

and compares the OTI result to analytic directional derivatives.

.. code-block:: cpp

   #include <cmath>
   #include <iomanip>
   #include <iostream>

   #include "otinum/otinum.hpp"

   int main()
   {
       using T = oti::otinum<1, 2>;

       double x0 = 1.5;
       double y0 = 0.3;
       double vx = 2.0;
       double vy = -1.0;

       T x(x0);
       T y(y0);

       x.set_coeff({1}, vx);
       y.set_coeff({1}, vy);

       T f = oti::sin(x * y) + oti::exp(x);

       double xy = x0 * y0;
       double fx = y0 * std::cos(xy) + std::exp(x0);
       double fy = x0 * std::cos(xy);
       double fxx = -y0 * y0 * std::sin(xy) + std::exp(x0);
       double fxy = std::cos(xy) - xy * std::sin(xy);
       double fyy = -x0 * x0 * std::sin(xy);

       double analytic_first = fx * vx + fy * vy;
       double analytic_second =
           fxx * vx * vx + 2.0 * fxy * vx * vy + fyy * vy * vy;

       auto print_check = [](const char* name, double analytic, double ad) {
           std::cout << std::setw(18) << name
                     << " analytic=" << std::setw(16) << analytic
                     << " ad=" << std::setw(16) << ad
                     << " abs_diff=" << std::abs(analytic - ad) << '\n';
       };

       print_check("directional d1", analytic_first, f.partial({1}));
       print_check("directional d2", analytic_second, f.partial({2}));
   }

Compile And Run
---------------

From a scratch directory:

.. code-block:: console

   c++ -std=c++17 \
     -I /root/Research/cpp_oti_lib/include \
     directional_derivatives.cpp \
     -o directional_derivatives

   ./directional_derivatives

Expected output should have differences near floating-point roundoff:

.. code-block:: text

      directional d1 analytic=         8.15298 ad=         8.15298 abs_diff=0
      directional d2 analytic=         13.9726 ad=         13.9726 abs_diff=0

Why ``set_coeff`` Works Here
----------------------------

The single OTI variable represents the line parameter ``t``. Setting:

.. code-block:: cpp

   x.set_coeff({1}, vx);
   y.set_coeff({1}, vy);

means:

.. code-block:: text

   x = x0 + vx t
   y = y0 + vy t

Because this is a first-order coefficient, ``set_coeff({1}, value)`` and
``set_partial({1}, value)`` are equivalent. For higher-order manual seeding,
``set_partial`` is usually safer because it accepts ordinary derivative values
and handles the ``alpha!`` normalization internally.

When To Use This Pattern
------------------------

Use this pattern when you care about derivatives along a few selected
directions instead of the full gradient or Hessian. It can reduce the OTI
dimension and coefficient count:

* Full two-variable Hessian information: ``otinum<2, 2>`` stores six
  coefficients.
* One selected direction through a two-variable space: ``otinum<1, 2>`` stores
  three coefficients.

The tradeoff is that ``otinum<1, 2>`` gives derivatives only along the seeded
direction. It does not recover every individual partial derivative.
