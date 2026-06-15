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
``g++ -O2`` and are a rough guide rather than hard limits:

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
     - <1 s
     - <0.1 GB
   * - ``<4,4>``
     - 70
     - 495
     - ~1 s
     - <0.3 GB
   * - ``<5,4>``
     - 126
     - 1001
     - ~3 s
     - ~0.8 GB
   * - ``<5,5>``
     - 252
     - 3003
     - ~13 s
     - ~4.2 GB
   * - ``<6,6>``
     - 924
     - (large)
     - ~90 s
     - >11 GB, often OOM-killed

Peak compile memory runs very roughly 1-1.5 MB per product term, and the curve
is super-linear, so it climbs steeply.

Practical Guidance
------------------

* Keep ``C(M + N, N)`` at or below ~70 (up to about ``<4,4>``) for fast,
  interactive builds on any machine.
* Treat ~250 coefficients (``<5,5>``) as a soft ceiling: it builds, but wants
  a large-RAM machine and over ten seconds per shape.
* Shapes with coefficient counts in the high hundreds (``<6,6>`` and beyond)
  can exhaust compiler memory and should be avoided unless you have measured
  the cost on your build host.

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
