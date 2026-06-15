Directional Derivatives
=======================

OTI variables do not have to correspond one-to-one with physical variables.
They can also represent perturbation parameters. For a scalar function
``f(x, y)`` at point ``(x0, y0)`` and direction ``v = (vx, vy)``, a single
first-order perturbation ``eps`` shared by both inputs is sufficient:
evaluating

.. code-block:: text

   f(x0 + vx eps,  y0 + vy eps)

with ``otinum<1, N>`` — where ``eps`` denotes the type's nilpotent
perturbation unit — produces the Taylor expansion of ``f`` along ``v``,
truncated at order ``N``:

.. code-block:: text

   f(x0, y0)  +  (grad f dot v) eps  +  (v^T H v / 2!) eps^2  +  ...

The directional derivatives are read off the ``eps`` coefficients:
``partial({1})`` returns ``grad f(x0, y0) dot v``, ``partial({2})`` returns
``v^T H f(x0, y0) v``, and so on up to order ``N`` — without ever computing
the full gradient or Hessian.

Seeding The Direction
---------------------

Building ``x0 + vx eps`` takes two steps. ``T x(x0)`` constructs a
*constant* — real part ``x0``, every derivative coefficient zero — and
``set_coeff`` then writes the direction component into the first-order slot:

.. code-block:: cpp

   using T = oti::otinum<1, 2>;

   T x(x0);
   T y(y0);
   x.set_coeff({1}, vx);   // x = x0 + vx eps
   y.set_coeff({1}, vy);   // y = y0 + vy eps

``set_coeff(alpha, value)`` is the writing counterpart of ``coeff(alpha)``,
and ``set_partial`` is the counterpart of ``partial``. For a first-order slot
the two are interchangeable (``1! = 1``); for higher-order manual seeding,
prefer ``set_partial`` — it accepts ordinary derivative values and handles
the ``alpha!`` normalization internally.

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

Multiple Directions At Once
---------------------------

Several directions can be evaluated simultaneously by seeding one
perturbation unit per direction. With two directions ``v`` and ``w``, use
``otinum<2, 2>`` and give each input a coefficient in both units:

.. code-block:: cpp

   using T = oti::otinum<2, 2>;   // 2 directions, order 2

   T x(x0), y(y0);
   x.set_coeff({1, 0}, vx);  x.set_coeff({0, 1}, wx);
   y.set_coeff({1, 0}, vy);  y.set_coeff({0, 1}, wy);

   T f = oti::sin(x * y) + oti::exp(x);   // one evaluation

One evaluation then holds every first- and second-order combination of the
two directions:

.. code-block:: cpp

   f.partial({1, 0});   // grad f . v
   f.partial({0, 1});   // grad f . w
   f.partial({2, 0});   // v^T H v
   f.partial({0, 2});   // w^T H w
   f.partial({1, 1});   // v^T H w  -- the cross-direction curvature

The mixed index is the part you cannot get from two separate
single-direction evaluations. Here the first template argument counts
*directions*, not physical inputs, so the coefficient count grows only with
the number of directions you seed. With many directions, the sparse
multi-index form from :doc:`basic_usage` works for seeding too —
``x.set_coeff(oti::sparse({{0, 1}}), vx)`` is equivalent to
``x.set_coeff({1, 0}, vx)``.

When To Use This Pattern
------------------------

Use this pattern when you care about derivatives along a few selected
directions instead of the full gradient or Hessian. The savings are modest in
this two-variable example — ``otinum<2, 2>`` stores six coefficients,
``otinum<1, 2>`` stores three — but they grow combinatorially with the number
of physical inputs, because the perturbation parameter count stays fixed
while the full space does not:

* Second order across 100 physical inputs: ``otinum<100, 2>`` stores 5,151
  coefficients per value.
* Second order along one direction through that same space: ``otinum<1, 2>``
  still stores three.

The tradeoff is that ``otinum<1, 2>`` gives derivatives only along the seeded
direction. It does not recover every individual partial derivative.

One convention to keep in mind: the direction vector is used as given, not
normalized. Scaling ``v`` scales the first derivative linearly and the second
derivative quadratically, so compare against a textbook unit-direction
directional derivative only after dividing by ``|v|`` and ``|v|^2``
respectively (or by seeding a unit vector to begin with).
