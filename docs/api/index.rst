API Reference
=============

The public C++ API is intentionally small. The primary include is:

.. code-block:: cpp

   #include "otinum/otinum.hpp"

Primary Type
------------

``oti::otinum<M, N, Coeff = double>``
   Static OTI number with ``M`` infinitesimal variables, truncation order ``N``,
   and coefficient storage type ``Coeff``.

Important members:

* ``T::variable(i, value)`` creates ``value + e_i``.
* ``real()`` returns the scalar coefficient.
* ``operator[](flat_index)`` accesses the raw normalized coefficient.
* ``coeff(alpha)`` returns the normalized Taylor coefficient.
* ``partial(alpha)`` returns the ordinary derivative value.
* ``data()`` returns the backing fixed-size coefficient array.

Normalized Coefficients
-----------------------

In this documentation, *normalized Taylor coefficient* means the ordinary
partial derivative divided by the multi-index factorial:

.. code-block:: text

   coeff(alpha) = partial^alpha f / alpha!

For ``alpha = (a_0, a_1, ..., a_{M-1})``, the multi-index factorial is:

.. code-block:: text

   alpha! = a_0! * a_1! * ... * a_{M-1}!

The library stores normalized coefficients because they are the coefficients
that appear directly in a Taylor polynomial. ``coeff(alpha)`` returns this
stored value. ``partial(alpha)`` multiplies the stored value by ``alpha!`` and
returns the ordinary derivative.

For example, if ``f(x) = x^2`` at a point, then the second derivative is
``2``. The stored second-order coefficient is ``1`` because ``2 / 2! = 1``:

.. code-block:: cpp

   f.coeff({2});    // 1
   f.partial({2});  // 2

Math Functions
--------------

The library provides ``oti`` overloads for common scalar functions:

``exp``, ``log``, ``log10``, ``log_base``, ``pow``, ``sqrt``, ``cbrt``, ``sin``,
``cos``, ``tan``, ``sinh``, ``cosh``, ``tanh``, and ``abs``.

Coefficient Layout and Lookup Tables
------------------------------------

The lookup tables exist to avoid repeating work that never changes. To multiply
two OTI numbers, the library must, for each pair of input coefficients, find
which output coefficient their product belongs to. That answer depends only on
``M`` and ``N`` -- not on the coefficient *values* -- so it is identical on
every multiply of every ``otinum<M, N>``. Recomputing it at run time would be
pure waste. Instead the library computes it once, at compile time, and stores
the answers in flat integer tables. The rest of this section explains what
"which output coefficient" means and what each table holds.

**Arithmetic is truncated polynomial convolution.** An ``otinum<M, N>`` is a
multivariate polynomial in ``M`` nilpotent variables, truncated at total order
``N``. Multiplication is therefore a convolution over multi-indices:

.. code-block:: text

   c[gamma] = sum over alpha + beta = gamma, |gamma| <= N  of  a[alpha] * b[beta]

Done naively, every product would have to add multi-index vectors and discard
the terms that overflow order ``N`` at run time. The library does that work
once, at compile time, and stores the result as flat integer tables, so each
run-time product reduces to "load two coefficients at known indices, multiply,
and accumulate" with no multi-index arithmetic or ranking in the loop.

**Numbering the coefficients.** Each coefficient is identified by a multi-index
``alpha = (a_1, ..., a_M)`` -- the exponents of the nilpotent variables -- and
they are stored in a single flat ``array<Coeff, C(M + N, N)>``. The numbering is
*graded*: coefficients are sorted by total order ``|alpha|`` first. For
``<2, 2>`` the six coefficients are:

.. code-block:: text

   index 0:  (0,0)                          order 0   (the real value)
   index 1:  (1,0)    index 2:  (0,1)        order 1
   index 3:  (2,0)    index 4:  (1,1)    index 5:  (0,2)    order 2

**A worked product.** Multiplying the coefficient at index 1, ``alpha = (1,0)``,
by the coefficient at index 2, ``beta = (0,1)``, produces a term at multi-index
``alpha + beta = (1,1)`` -- which is index 4. So that pair contributes
``out[4] += a[1] * b[2]``. Every pair yields one such ``(lhs, rhs, out)`` triple:
index 1 times index 1 gives ``(2,0)`` = index 3, and so on. A pair whose orders
sum above ``N`` is dropped -- index 3, ``(2,0)``, times index 1, ``(1,0)``,
would land at ``(3,0)`` with total order 3 > 2, so it never enters the table.
These triples are identical for every ``<2, 2>`` number, which is exactly why
they can be computed once instead of per multiply.

**Finding the output index.** Turning a multi-index such as ``(1,1)`` into its
array position (4) is the job of ``rank<M, N>(alpha)``. Rather than looking the
position up, it *computes* it directly by counting how many multi-indices
precede ``alpha`` in the graded ordering (a closed-form sum of binomials). A
direct lookup array addressed by the exponent vector would instead need
``(N + 1)^M`` slots -- fine for ``<2, 2>``, but astronomically large once ``M``
grows (``M = 20`` is already hopeless) -- so the library always uses the
formula. ``rank`` builds the tables below at compile time, and also backs the
user-facing ``coeff(alpha)`` and ``partial(alpha)`` accessors, which take a
multi-index.

**Truncation is a prefix.** Because entries are grouped by order, "keep
everything up to order k" is exactly the first ``C(M + k, k)`` entries. This
makes order-bounded operations (``trunc_mul``, the order-skipping in Taylor
composition) simple index ranges rather than per-element order tests.

**The precomputed tables in** ``detail::tables<M, N>`` **.** The ``(lhs, rhs,
out)`` triples above are stored as ``product_terms``. All tables are
``constexpr`` and built once per ``(M, N)``:

* ``product_terms`` -- the convolution as a flat list of ``(lhs, rhs, out)``
  index triples. ``operator*`` and ``trunc_mul`` simply walk it and accumulate
  ``out[k] += a[i] * b[j]``, with no multi-index arithmetic at run time.
* ``product_terms_by_output`` with ``product_offset`` -- the same products,
  grouped by their output coefficient, so that every product landing on a given
  output index can be visited together. ``product_offset[k] .. product_offset[k
  + 1]`` is the group for output ``k`` (see *Multiply versus divide* below).
* ``order_offset`` and ``order_of`` -- the first flat index of each total
  order, and the order of each coefficient. These make truncation and
  order-skipping (for example, ``h^k`` cannot contribute below order ``k``) a
  range or a cheap comparison.
* ``factorial_alpha`` -- the multi-index factorial ``alpha!`` per coefficient,
  used to convert between the stored normalized coefficient and the ordinary
  derivative returned by ``partial``.

**Multiply versus divide.** The two product tables serve opposite access
patterns. ``operator*`` *scatters*: it walks ``product_terms`` once and adds
each product into its output slot. ``inv`` (the reciprocal ``1 / x``) instead
*gathers*. There is no direct formula for its coefficients, so it solves
``x * inv(x) = 1`` one coefficient at a time. For each output coefficient ``k``
in graded order it takes that output's group from ``product_terms_by_output``
(every product landing on ``k``), sets the product coefficient to its target --
``1`` for the real part and ``0`` for every higher-order coefficient -- and
solves for the single unknown ``inv`` coefficient. The graded layout guarantees
this unknown depends only on lower-order coefficients that are already computed,
so one forward pass over the coefficients suffices. Division then needs no
further machinery: ``a / b`` is ``a * inv(b)``. The same gather pattern drives
the nilpotent powering used by the Taylor-composed functions (``exp``, ``log``,
``sin``, and so on).

**Why compile-time.** Because the tables are ``constexpr`` and ``M`` and ``N``
are template parameters, an optimizing build constant-folds them: at ``-O2``
the product loops become straight-line, fully unrolled arithmetic and the
tables do not appear in the binary at all. The cost moves entirely to compile
time and grows with ``tables<M, N>::nproducts``. See *Choosing M and N* in the
README for measured compile-time and memory guidance and recommended shape
ceilings.

Generated C++ API Docs
----------------------

Doxygen extracts API information from the headers and Breathe renders that
information inside Sphinx. Generated XML is written to ``docs/api/xml/`` and is
ignored by Git.

Generate the XML before building Sphinx:

.. code-block:: console

   doxygen docs/Doxyfile
   sphinx-build -E -a -W -b html docs docs/_build/html

The generated reference intentionally includes ``oti::detail``. Those internals
define the coefficient layout, multi-index ranking, factorial tables, and
Kokkos compatibility layer, so they are part of the design story even though
ordinary callers should prefer the public ``oti`` API.

.. toctree::
   :maxdepth: 2

   core
   functions
   detail
   profile
   generated
