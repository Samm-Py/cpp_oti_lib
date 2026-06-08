Core Type
=========

This page is generated from ``include/otinum/core.hpp`` and focuses on the
primary OTI scalar type and arithmetic operators.

Typical Use
-----------

Most user code starts by choosing a fixed OTI type:

.. code-block:: cpp

   using T = oti::otinum<2, 3>;        // 2 variables, order 3, double coeffs
   using Tf = oti::otinum<2, 3, float>; // same algebra with float coeffs

Construct independent variables with ``T::variable(index, value)``. The index
selects the derivative direction and the value becomes the real coefficient:

.. code-block:: cpp

   T x = T::variable(0, 1.5);
   T y = T::variable(1, 0.3);

   T f = x * x + 3.0 * x * y + y;

   double value = f.real();
   double dfdx = f.partial({1, 0});
   double dfdy = f.partial({0, 1});
   double d2fdxdy = f.partial({1, 1});

Constants and zero values use ordinary construction:

.. code-block:: cpp

   T zero;          // all coefficients are zero
   T constant(3.0); // real coefficient is 3.0, derivative coefficients are zero

Arithmetic works with other OTI values and with arithmetic scalar values:

.. code-block:: cpp

   T sum = x + y;
   T product = x * y;
   T quotient = x / (y + 2.0);
   T affine = 3.0 * x - 1.0;

For ``oti::otinum<M, N, Coeff>``, valid variable indices are ``0`` through
``M - 1``. For example, ``oti::otinum<2, 3>`` has two derivative directions, so
``T::variable(0, value)`` and ``T::variable(1, value)`` are valid, while
``T::variable(2, value)`` is out of range. The implementation asserts
``index >= 0 && index < M``; callers should treat that range as a hard API
requirement even when assertions are disabled in optimized builds.

Compile-Time Type Properties
----------------------------

Each ``otinum`` instantiation exposes compile-time properties through static
members and type aliases. These belong to the type ``T`` itself, not to a
particular runtime value:

.. code-block:: cpp

   using T = oti::otinum<2, 3, float>;

   static_assert(T::nvars == 2);
   static_assert(T::order == 3);
   static_assert(T::ncoeffs == 10);
   static_assert(std::is_same<T::coeff_type, float>::value);

The main properties are:

``T::nvars``
   Number of independent derivative directions. This is the template parameter
   ``M``. Valid variable indices for ``T::variable(index, value)`` are
   ``0`` through ``T::nvars - 1``.

``T::order``
   Maximum stored total derivative order. This is the template parameter ``N``.
   Requests above this total order return zero from ``coeff`` and ``partial``.
   For ``N == 0``, only the real coefficient is stored. ``T::variable`` still
   constructs the requested real value, but there is no derivative coefficient
   available to seed.

``T::ncoeffs``
   Number of coefficients stored in each value. This is
   ``binomial(T::nvars + T::order, T::order)``. It is useful for loops over
   ``data()`` or ``operator[]`` and for sizing arrays passed to
   ``T::from_coeffs``.

``T::coeff_type``
   Floating-point storage type for each coefficient. This is the template
   parameter ``Coeff`` and defaults to ``double``.

``T::alpha_type``
   Multi-index type used by ``coeff`` and ``partial``. For ``otinum<M, N>`` this
   behaves like a fixed-size array of ``M`` integers.

``T::table_type``
   Compile-time layout and multiplication metadata type. Most user code does
   not need this directly, but it is useful for tests, diagnostics, and
   understanding the coefficient layout.

For example, ``oti::otinum<2, 2>`` has ``T::ncoeffs == 6`` because it stores:

.. code-block:: text

   {0,0}, {1,0}, {0,1}, {2,0}, {1,1}, {0,2}

Access Patterns
---------------

For ``oti::otinum<2, N>``, derivative requests use a two-entry multi-index
``{x_order, y_order}``. Each entry says how many derivatives to take in that
variable direction:

.. code-block:: cpp

   double value = f.real();          // f(x, y), no derivative
   double dfdx = f.partial({1, 0});  // first derivative with respect to x
   double dfdy = f.partial({0, 1});  // first derivative with respect to y
   double d2fdx2 = f.partial({2, 0}); // second derivative with respect to x
   double d2fdxdy = f.partial({1, 1}); // mixed derivative d^2 f / dx dy
   double d2fdy2 = f.partial({0, 2}); // second derivative with respect to y

The same pattern extends to more variables. For ``oti::otinum<3, N>``, the
multi-index has three entries ``{x_order, y_order, z_order}``:

.. code-block:: cpp

   using T3 = oti::otinum<3, 2>;

   T3 f = /* ... */;
   double dfdz = f.partial({0, 0, 1});
   double d2fdxdz = f.partial({1, 0, 1});

The sum of the multi-index entries is the total derivative order. If that total
order is greater than ``N``, the requested derivative is outside the stored
truncation order and ``partial`` returns zero.

For high-dimensional types, writing every zero entry in the dense multi-index
can be tedious. Use ``oti::sparse`` to name only the nonzero derivative
directions:

.. code-block:: cpp

   using T100 = oti::otinum<100, 2>;

   T100 f = /* ... */;

   double df_dv21 = f.partial(oti::sparse({{21, 1}}));
   double d2f_dv21_dv32 = f.partial(oti::sparse({{21, 1}, {32, 1}}));

The sparse entries are ``{variable_index, derivative_order}`` pairs. Variable
indices are zero-based, matching ``T::variable(index, value)``. Repeated sparse
entries for the same variable are added, so ``oti::sparse({{21, 1}, {21, 1}})``
requests the same derivative as a dense multi-index with order ``2`` in
variable ``21``.

The explicit ``oti::sparse(...)`` wrapper is intentional. In C++, a call such
as ``f.partial({{21, 1}})`` can be interpreted as a dense ``std::array``
initializer with the remaining entries filled by zero, which is not the same
request for large ``M``.

Use ``partial(alpha)`` for ordinary derivatives and ``coeff(alpha)`` for the
stored normalized Taylor coefficient:

.. code-block:: cpp

   T x = T::variable(0, 2.0);
   T f = x * x;

   double normalized = f.coeff({2, 0}); // f_xx / 2! = 1
   double derivative = f.partial({2, 0}); // f_xx = 2

``operator[]`` exposes the flat coefficient storage directly. The flat index is
the position in the internal coefficient array, not a derivative direction. For
``oti::otinum<2, 2>``, the flat layout is:

.. code-block:: text

   flat index 0 -> alpha {0, 0} -> real value
   flat index 1 -> alpha {1, 0} -> normalized df/dx coefficient
   flat index 2 -> alpha {0, 1} -> normalized df/dy coefficient
   flat index 3 -> alpha {2, 0} -> normalized d2f/dx2 coefficient
   flat index 4 -> alpha {1, 1} -> normalized d2f/dxdy coefficient
   flat index 5 -> alpha {0, 2} -> normalized d2f/dy2 coefficient

This ordering is graded by total derivative order. All order-0 coefficients
come first, then all order-1 coefficients, then all order-2 coefficients, and so
on.

Use ``operator[]`` when you need low-level access, such as printing every stored
coefficient, serializing the raw coefficient array, or writing layout tests:

.. code-block:: cpp

   for (int i = 0; i < T::ncoeffs; ++i) {
       std::cout << "flat coefficient " << i << " = " << f[i] << '\n';
   }

``operator[]`` does not perform bounds checking. The caller is responsible for
using only flat indices in the range ``0 <= i < T::ncoeffs``.

Prefer ``coeff`` or ``partial`` in ordinary mathematical code. They accept
multi-indices, make the derivative request clear at the call site, and return
zero when the requested multi-index is outside the configured truncation order:

.. code-block:: cpp

   double stored = f.coeff({1, 1});      // normalized mixed coefficient
   double derivative = f.partial({1, 1}); // ordinary mixed derivative
   double zero = f.partial({3, 0});       // zero for T = otinum<2, 2>
   double sparse = f.partial(oti::sparse({{0, 1}, {1, 1}}));

The matching setters follow the same distinction:

.. code-block:: cpp

   T f;

   f.set_coeff({1, 0}, 3.5);  // stores normalized df/dx coefficient
   f.set_partial({2, 0}, 8.0); // stores 8.0 / 2! for d2f/dx2
   f.set_partial(oti::sparse({{0, 1}, {1, 1}}), 1.25);

   double dfdx = f.partial({1, 0}); // 3.5
   double fxx = f.partial({2, 0});  // 8.0
   double fxy = f.partial({1, 1});  // 1.25

Use ``set_coeff`` when you already have normalized Taylor coefficients. Use
``set_partial`` when you have ordinary derivative values and want the library to
divide by ``alpha!`` before storing. Like ``coeff`` and ``partial``,
out-of-order multi-indices are ignored by the setters. Sparse setters follow
the same normalization rules as dense setters.

The ``data()`` accessor returns the full coefficient array by const reference:

.. code-block:: cpp

   auto const& raw = f.data();

   for (double coefficient : raw) {
       std::cout << coefficient << '\n';
   }

Use ``data()`` when integrating with serialization, diagnostics, or code that
needs to inspect every stored normalized coefficient without copying the array.

Direct Coefficient Construction
-------------------------------

``from_coeffs`` is useful for tests, serialization, or reconstructing a complete
OTI value from stored coefficients. It expects coefficients in the same flat
layout used by ``data()`` and ``operator[]``. The values you provide are
normalized Taylor coefficients, not ordinary derivative values.

.. code-block:: cpp

   using T = oti::otinum<2, 2>;

   std::array<double, T::ncoeffs> coeffs{};
   coeffs[0] = 4.0; // alpha {0, 0}: real part
   coeffs[1] = 1.5; // alpha {1, 0}: df/dx
   coeffs[2] = 2.0; // alpha {0, 1}: df/dy
   coeffs[3] = 3.0; // alpha {2, 0}: normalized d2f/dx2, i.e. f_xx / 2!
   coeffs[4] = 0.5; // alpha {1, 1}: normalized d2f/dxdy
   coeffs[5] = 7.0; // alpha {0, 2}: normalized d2f/dy2, i.e. f_yy / 2!

   T value = T::from_coeffs(coeffs);

   double real = value.real();          // 4.0
   double dfdx = value.partial({1, 0}); // 1.5
   double dfdy = value.partial({0, 1}); // 2.0
   double fxx = value.partial({2, 0});  // 6.0, because 2! * 3.0
   double fxy = value.partial({1, 1});  // 0.5
   double fyy = value.partial({0, 2});  // 14.0, because 2! * 7.0

This constructor does not reinterpret ordinary derivative values for you. If
you have ordinary second derivatives and want to build an OTI value manually,
divide by ``alpha!`` before storing them in the coefficient array:

.. code-block:: cpp

   double ordinary_fxx = 6.0;
   coeffs[3] = ordinary_fxx / 2.0; // alpha {2, 0}

For layout-independent code, prefer ``T::table_type`` or ``oti::detail::rank``
to compute flat indices instead of hard-coding numbers:

.. code-block:: cpp

   int fxy_index = oti::detail::rank<2, 2>({1, 1});
   coeffs[static_cast<std::size_t>(fxy_index)] = 0.5;

If you are setting only a few coefficients manually, ``set_coeff`` and
``set_partial`` are usually clearer than constructing the full flat array:

.. code-block:: cpp

   T value;
   value.set_partial({1, 0}, 1.5); // df/dx
   value.set_partial({2, 0}, 6.0); // d2f/dx2, stored internally as 3.0

Truncated Operations
--------------------

``trunc_add`` and ``trunc_mul`` keep only terms through a requested total order:

.. code-block:: cpp

   T full = x * y;
   T first_order_only = oti::trunc_mul(x, y, 1);

This is mainly useful when experimenting with order-limited approximations or
when implementing algorithms that intentionally discard higher-order terms.

Division And Inverses
---------------------

Division by another OTI value is implemented by expanding an inverse around the
denominator's real coefficient:

.. code-block:: cpp

   T x = T::variable(0, 1.5);
   T denominator = x + 2.0;
   T reciprocal = 1.0 / denominator;
   T quotient = x / denominator;

For a valid real-valued inverse, ``denominator.real()`` must be nonzero. If the
real coefficient is zero, the inverse has a singular scalar part and the result
will follow ordinary floating-point division behavior.

Generated Reference
-------------------

.. doxygenclass:: oti::otinum
   :members:
   :undoc-members:

Free Operators And Helpers
--------------------------

.. doxygenfile:: core.hpp
   :sections: func typedef
