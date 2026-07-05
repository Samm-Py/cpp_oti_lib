Choosing M and N
================

Every ``otinum<M, N>`` builds its multi-index and truncated-product tables at
compile time, so each distinct shape you instantiate adds to the *build*, not
to the runtime (see :ref:`api/index:Coefficient Layout and Lookup Tables` for
why the tables exist and how they fold away at ``-O2``). The flip
side of that zero runtime cost is a real compile-time cost that grows quickly
with the shape, so the shape is worth choosing deliberately.

Measured Cost By Shape
----------------------

The build cost is driven by the number of product terms,
``detail::tables<M, N>::nproducts``, which grows with the coefficient count
``C(M + N, N)``. The following are measured per translation unit with
``g++ 11 -O2`` and are a rough guide rather than hard limits:

.. list-table::
   :header-rows: 1

   * - Shape
     - Coeffs
     - Product terms
     - Compile
     - Peak compiler RAM
   * - ``<3,3>``
     - 20
     - 84
     - ~1 s
     - <0.1 GB
   * - ``<4,4>``
     - 70
     - 495
     - ~1.5 s
     - ~0.12 GB
   * - ``<5,4>``
     - 126
     - 1001
     - ~3 s
     - ~0.16 GB
   * - ``<5,5>``
     - 252
     - 3003
     - ~12 s
     - ~0.33 GB
   * - ``<6,6>``
     - 924
     - 18564
     - tens of minutes
     - ~1.5 GB

Peak compile memory runs roughly 0.1 MB per product term. Memory is no longer
the limiting resource (the sparse table builders removed the old
multi-gigabyte peaks); *constexpr evaluation time* is, and it grows
super-linearly with the product count.

At ``<6,6>`` GCC first stops with ``'constexpr' evaluation operation count
exceeds limit`` — raise it with ``-fconstexpr-ops-limit`` (e.g.
``-fconstexpr-ops-limit=268435456``) if you genuinely need such a shape, and
budget tens of minutes per translation unit that instantiates it.

Practical Guidance
------------------

* Keep ``C(M + N, N)`` at or below ~70 (up to about ``<4,4>``) for fast,
  interactive builds on any machine.
* Treat ~250 coefficients (``<5,5>``) as a soft ceiling: it builds in
  ordinary memory but takes over ten seconds per shape.
* Shapes with coefficient counts in the high hundreds (``<6,6>`` and beyond)
  hit GCC's constexpr operation limit and, with the limit raised, cost tens
  of minutes of compile time; avoid them unless you have measured the cost on
  your build host.

Because each instantiated shape pays this cost independently, prefer reusing a
small set of ``(M, N)`` shapes over scattering many large ones across a build.

Clang Fold-Expression Limit
---------------------------

Clang limits fold-expression expansion to 256 arguments by default, and the
unrolled product folds have one argument per product term, so shapes with more
than 256 product terms (``<5,3>`` and larger) need that limit raised. The CMake
target applies ``-fbracket-depth=65536`` automatically for Clang; pass the same
flag manually when compiling directly with ``clang++ -I include``. GCC and NVCC
have no such limit.
