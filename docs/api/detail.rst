Detail Namespace
================

The ``oti::detail`` namespace is included in the documentation because it
defines the core compile-time design choices: coefficient storage aliases,
multi-index ranking, product tables, factorial tables, and scalar Taylor
composition helpers.

When To Look Here
-----------------

Most user code should not call ``oti::detail`` directly. It is documented
because the internals explain how the library stores coefficients and avoids
runtime table construction. This is useful when debugging coefficient order,
writing tests, validating serialization, or extending the algebra.

Inspecting Coefficient Layout
-----------------------------

The flat coefficient array is ordered by total derivative order. Use
``detail::rank`` to map a multi-index to the flat storage index:

.. code-block:: cpp

   using T = oti::otinum<2, 2>;

   int real_index = oti::detail::rank<2, 2>({0, 0}); // 0
   int dx_index = oti::detail::rank<2, 2>({1, 0});   // 1
   int dy_index = oti::detail::rank<2, 2>({0, 1});   // 2
   int dxy_index = oti::detail::rank<2, 2>({1, 1});  // 4

   T x = T::variable(0, 1.5);
   T y = T::variable(1, 0.3);
   T f = x * y;

   double dxy = f[dxy_index];

Inspecting Compile-Time Tables
------------------------------

``detail::tables<M, N>`` exposes compile-time metadata used by arithmetic:

.. code-block:: cpp

   using Tables = oti::detail::tables<2, 2>;

   static_assert(Tables::ncoeffs == 6);

   for (int i = 0; i < Tables::ncoeffs; ++i) {
       auto alpha = Tables::alpha_at(i);
       int order = Tables::order_of_value(i);
       double alpha_factorial = Tables::factorial_alpha_value(i);
   }

The multiplication implementation uses precomputed product terms. Each term
describes one contribution ``lhs[term.lhs] * rhs[term.rhs]`` to
``out[term.out]``:

.. code-block:: cpp

   using Tables = oti::detail::tables<2, 2>;

   for (int p = 0; p < Tables::nproducts; ++p) {
       auto term = Tables::product_term_value(p);
       // out[term.out] += lhs[term.lhs] * rhs[term.rhs]
   }

Multi-Index Layout
------------------

.. doxygenfile:: multi_index.hpp
   :sections: innerclass typedef func var

Taylor Composition Helpers
--------------------------

.. doxygenfile:: taylor.hpp
   :sections: func

Combinatorics
-------------

.. doxygenfile:: binom.hpp
   :sections: func

Kokkos Compatibility
--------------------

.. doxygenfile:: kokkos_compat.hpp
   :sections: typedef func define
