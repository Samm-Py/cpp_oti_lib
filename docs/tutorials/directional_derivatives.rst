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

and compares the OTI result to analytic directional derivatives. The same
source is available in the repository as
``examples/directional_derivatives.cpp``.

.. literalinclude:: ../../examples/directional_derivatives.cpp
   :language: cpp

Compile And Run
---------------

From the repository root:

.. code-block:: console

   c++ -std=c++17 -I include examples/directional_derivatives.cpp -o /tmp/directional_derivatives
   /tmp/directional_derivatives

Output
------

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
