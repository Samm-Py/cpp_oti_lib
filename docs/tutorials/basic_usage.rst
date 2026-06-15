Basic C++ Usage
===============

This tutorial evaluates ``f(x, y) = sin(x*y) + exp(x)`` at ``(1.5, 0.3)`` with
OTI arithmetic and checks the value and all first- and second-order
derivatives against their analytic formulas. It is the place where the
library's core ideas — variable seeding, multi-index derivative access, and
the difference between derivatives and stored coefficients — are introduced.

Program
-------

The same source is available in the repository as
``examples/basic_usage.cpp``.

.. literalinclude:: ../../examples/basic_usage.cpp
   :language: cpp

Compile And Run
---------------

From the repository root:

.. code-block:: console

   c++ -std=c++17 -I include examples/basic_usage.cpp -o /tmp/basic_usage
   /tmp/basic_usage

Output
------

The analytic values and OTI values should agree to roundoff:

.. code-block:: text

          f analytic=         4.91665 ad=         4.91665 abs_diff=0
      df/dx analytic=         4.75182 ad=         4.75182 abs_diff=0
      df/dy analytic=         1.35067 ad=         1.35067 abs_diff=0
    d2f/dx2 analytic=         4.44254 ad=         4.44254 abs_diff=0
   d2f/dxdy analytic=        0.704713 ad=        0.704713 abs_diff=1.11022e-16
    d2f/dy2 analytic=       -0.978672 ad=       -0.978672 abs_diff=2.22045e-16

How It Works
------------

**The type.** ``oti::otinum<2, 2>`` tracks ``2`` independent variables through
total derivative order ``2``. A value of this type stores six coefficients:
the function value, the two first derivatives, and the three second
derivatives (``d2f/dx2``, ``d2f/dxdy``, ``d2f/dy2``). The two template
arguments are the only thing you change to track more variables or higher
orders.

**Seeding.** ``T::variable(0, 1.5)`` creates the value of variable 0: its real
part is ``1.5`` and its first derivative with respect to variable 0 is one
(every other coefficient is zero). Seed each independent variable exactly
once, then write the calculation with ordinary operators and the ``oti::``
elementary functions — every operation propagates all six coefficients
automatically.

**Reading derivatives.** ``partial(alpha)`` takes a multi-index with one entry
per variable; entry ``i`` is the derivative order with respect to variable
``i``. In the program above:

* ``f.partial({1, 0})`` is ``df/dx`` — first derivative in variable 0.
* ``f.partial({0, 1})`` is ``df/dy``.
* ``f.partial({1, 1})`` is the mixed second derivative ``d2f/dxdy``.
* ``f.partial({2, 0})`` is ``d2f/dx2``.

The entries may sum to at most the order (here ``2``); higher derivatives are
truncated away and are not stored.

**Derivatives versus coefficients.** Internally the value is a truncated
Taylor polynomial, and a Taylor coefficient differs from the corresponding
derivative by the factorials of the multi-index:
``coeff(alpha) = partial(alpha) / alpha!``. For this program,
``f.coeff({2, 0})`` returns ``2.22127`` — exactly half of
``f.partial({2, 0}) = 4.44254`` — while for first-order and mixed indices like
``{1, 1}`` the two are identical. Use ``partial`` when you want derivatives
(the common case) and ``coeff`` when you want the polynomial's stored
coefficient. Both have writing counterparts — ``set_partial`` and
``set_coeff`` — for seeding values by hand instead of through
``T::variable``; :doc:`directional_derivatives` shows the main use for them.

More Variables, Higher Orders
-----------------------------

The multi-index convention is easiest to see at a shape where the indices have
more room. With four variables through total order three —
``oti::otinum<4, 3>``, 35 coefficients — a multi-index has four entries, and
entry ``i`` is still the derivative order with respect to variable ``i``:

.. code-block:: cpp

   using T = oti::otinum<4, 3>;

   T w = T::variable(0, 2.0);
   T x = T::variable(1, 3.0);
   T y = T::variable(2, 5.0);
   T z = T::variable(3, 7.0);

   T f = w * y * y + w * x * z;

   f.partial({1, 0, 0, 0});   // df/dw           -> 46  (= y*y + x*z)
   f.partial({1, 0, 1, 0});   // d2f/dw dy       -> 10  (= 2*y)
   f.partial({1, 0, 2, 0});   // d3f/dw dy2      -> 2
   f.partial({1, 1, 0, 1});   // d3f/dw dx dz    -> 1

Dense multi-indices spell out every variable, including the ones a derivative
does not touch. The same accessors also take a *sparse* multi-index built from
``{variable, order}`` pairs via ``oti::sparse``, naming only the variables
involved:

.. code-block:: cpp

   f.partial(oti::sparse({{0, 1}, {2, 2}}));          // d3f/dw dy2     -> 2
   f.partial(oti::sparse({{0, 1}, {1, 1}, {3, 1}}));  // d3f/dw dx dz   -> 1
   f.partial(oti::sparse({{0, 1}, {2, 1}, {2, 1}}));  // pairs with the same
                                                      // variable add: same as
                                                      // {{0, 1}, {2, 2}} -> 2

Both forms read the same stored coefficient; sparse indices are purely a
notation convenience that pays off as the variable count grows.
:doc:`../api/core` holds the full multi-index rules — truncation behavior, why
the explicit ``oti::sparse`` wrapper exists, which accessors take sparse
indices, and their host-only restriction.

Where To Go Next
----------------

* :doc:`float_coefficients` switches the stored coefficient type to ``float``.
* :doc:`directional_derivatives` handles many physical inputs with a small
  number of OTI variables.
* :doc:`../api/core` documents coefficient layout and access in full.
