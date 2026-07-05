Coefficient-Major Storage Tutorial
==================================

This tutorial covers ``oti::soa_span``, a non-owning view that stores an
*array* of OTI numbers coefficient-major (structure-of-arrays) instead of the
default element-major (array-of-structs) layout. The view changes where the
bytes live, never what is computed: results are bit-identical to the default
layout.

It matters for one specific situation — large arrays of jets processed by GPU
kernels — and it is a measured loss everywhere else, so the second half of
this tutorial is guidance on when *not* to use it.

The Two Layouts
---------------

A ``Kokkos::View<otinum<3, 3>*>`` (or a plain C++ array of ``otinum``) stores
each element's 20 coefficients contiguously, then the next element's:

.. code-block:: text

   element 0: [c0 c1 ... c19] | element 1: [c0 c1 ... c19] | ...

``oti::soa_span`` views the transposed arrangement: all elements' coefficient
``k`` are contiguous, so coefficient ``k`` of element ``i`` lives at
``data[k * extent + i]``:

.. code-block:: text

   c0 of every element | c1 of every element | ... | c19 of every element

When one GPU thread per element walks the coefficients in lockstep — which is
what the unrolled OTI arithmetic does — the first layout makes "every thread
reads coefficient k" a strided access spanning one cache line per thread,
while the second makes it a single coalesced transaction.

Object Alignment
----------------

Layout is one axis of where the coefficients live; *alignment* is the other,
and it applies to the default array-of-structs layout. Independently of the
AoS/SoA choice, the library promotes each ``otinum``'s address boundary
whenever it can do so for free. The rule looks only at the object's byte size:

.. code-block:: text

   bytes = ncoeffs * sizeof(Coeff)

   if bytes is a multiple of 32: align the object to 32 bytes
   else if bytes is a multiple of 16: align the object to 16 bytes
   else if bytes is a multiple of 8: align it to at least 8 bytes
   else: keep the natural alignment of the coefficient type

(The 32-byte rung targets the 256-bit single-instruction loads of NVIDIA
Blackwell / CUDA 13+; on earlier GPUs it compiles to the same code as 16-byte
alignment, so it is never a cost.)

The size test is what makes the promotion free. C++ requires ``sizeof`` to be a
multiple of ``alignof``, so conditioning on the byte size guarantees the
alignment never changes ``sizeof(otinum)`` and never inserts padding. A
four-``float`` ``otinum<3, 1>`` is 16 bytes, so it is promoted to a 16-byte
boundary and a whole jet loads or stores as one 128-bit transaction; a
three-``double`` ``otinum<2, 1>`` is 24 bytes and keeps ``alignof(double)``:

.. code-block:: cpp

   static_assert(alignof(oti::otinum<3, 1, float>) == 16);   // 16 bytes
   static_assert(alignof(oti::otinum<2, 1>) == alignof(double));  // 24 bytes

This is why the default layout is less naive than it first appears: an aligned
jet is already a single wide vector access. It is also what lets the default
layout hold the bandwidth peak for small first-order jets, and why the
gather-heavy float ``otinum<3, 1>`` case below *prefers* AoS — the
coefficient-major view gives up that one-load-per-jet property. The alignment
rule and the layout choice are measured independently in the
:doc:`../benchmarks/gpu_optimization_workflow`.

A Complete Host Example
-----------------------

The span itself has no Kokkos dependency; it works over any caller-owned
buffer. The program below evaluates ``f(x, y) = x*x + 3*y`` for four elements,
scatters the jets into a coefficient-major buffer, and gathers one back. The
same source is available in the repository as ``examples/soa_layout.cpp``.

.. literalinclude:: ../../examples/soa_layout.cpp
   :language: cpp

Compile and run it from the repository root:

.. code-block:: console

   c++ -std=c++17 -I include examples/soa_layout.cpp -o /tmp/soa_layout
   /tmp/soa_layout

Expected output:

.. code-block:: text

   real parts (buffer[0..3]):  30 31 34 39
   df/dx parts (buffer[4..7]): 0 2 4 6
   element 3: f = 39, df/dx = 6, d2f/dx2 = 2

The first two printed lines are the point of the example: the buffer really is
grouped by coefficient, not by element.

The API surface is deliberately small:

* ``Span::required_size(n)`` — coefficients a backing buffer must hold for
  ``n`` elements.
* ``span.load(i)`` — gather element ``i`` into a register-resident ``otinum``.
* ``span.store(i, value)`` — scatter an ``otinum`` back to element ``i``.
* ``span.coeff(i, k)`` — writable reference to one coefficient of one element,
  for initialization and coefficient-wise kernels.

There is no ``operator()`` returning a reference: no ``otinum`` object exists
in memory to refer to, and a proxy reference would hide the gather/scatter
cost that this layout is all about. The explicit ``load``/``store`` pair keeps
every memory crossing visible in the kernel source.

Using The Span In A Kokkos Kernel
---------------------------------

The span is trivially copyable, so capture it by value in a device lambda. The
backing buffer is an ordinary flat ``Kokkos::View`` of coefficients:

.. code-block:: cpp

   using T = oti::otinum<3, 3>;
   using Span = oti::soa_span<3, 3>;

   Kokkos::View<double*> xs("xs", Span::required_size(n));
   Kokkos::View<double*> ys("ys", Span::required_size(n));
   Span const x(xs.data(), n);
   Span const y(ys.data(), n);

   Kokkos::parallel_for("jet_update", n, KOKKOS_LAMBDA(std::size_t i) {
       T const xi = x.load(i);   // coalesced gather
       T yi = y.load(i);
       oti::axpy(yi, 0.5, xi);   // ordinary register-resident arithmetic
       y.store(i, yi);           // coalesced scatter
   });

The kernel body between ``load`` and ``store`` is unchanged from what you
would write against a ``View<T*>`` — same operators, same fused helpers, same
results to the last bit.

When It Wins, When It Loses
---------------------------

``tests/benchmark_soa_layout.cpp`` measures both layouts over identical
streaming kernels (two jet reads and one jet write per element, arrays far
larger than cache) and checks that the checksums match bit-exactly. The
benchmark is an extra target that the smoke-test configure in :doc:`kokkos_gpu`
does not build, so reuse that tutorial's ``build-kokkos-gpu`` directory and
just turn the benchmark option on. CMake keeps the compiler and Kokkos prefix
from the original configure, and **Kokkos itself is not rebuilt** — only the
benchmark source is compiled and linked against the already-installed library:

.. code-block:: console

   cmake -S . -B build-kokkos-gpu -DOTI_BUILD_BENCHMARKS=ON
   cmake --build build-kokkos-gpu --target benchmark_soa_layout --parallel
   ./build-kokkos-gpu/benchmark_soa_layout

(If you want CPU/OpenMP numbers instead, do the same on a ``build-kokkos-cpu``
directory configured against an OpenMP Kokkos install.)

Representative output on a CUDA backend (GTX 1650; absolute rates vary by
device, the ratios are the story):

.. code-block:: text

   == mul: y = x*y ==
   <3,1> double      aos  176.32 GB/s   soa  176.05 GB/s   soa/aos  1.00x   checksums match
   <3,3> double      aos   57.82 GB/s   soa  165.37 GB/s   soa/aos  2.86x   checksums match
   <4,4> float       aos   37.75 GB/s   soa  156.94 GB/s   soa/aos  4.16x   checksums match

The measured picture, in both directions:

* **GPU, jets with N >= 2: use the span.** The default layout falls off the
  bandwidth peak for larger jets (down to ~3x-4x below peak in the worst
  cases) while the coefficient-major layout stays at the peak for every shape
  measured. Which default-layout shapes collapse is hard to predict —
  ``<3,3>`` ``double`` collapses while ``<3,3>`` ``float`` does not — so the
  span's real value on GPU is *predictability*: it never lost in any measured
  configuration.
* **GPU, first-order jets (``<3,1>`` and similar): no benefit, and a measured
  loss for gather patterns.** Small aligned jets already coalesce well enough
  that both layouts sit at the bandwidth peak for streaming access. In the
  OTI heat-equation solver (``<3,1>``), the end-to-end double solve is exact
  parity, but the *float* solve runs up to ~20% slower under the span at
  large grids: its stiffness kernel gathers dozens of neighbors' jets per
  thread, and an AoS float ``<3,1>`` jet is a single 16-byte vector load —
  an advantage the coefficient-major layout gives up. Streaming benchmarks
  cannot see this effect; measure kernels with neighbor gathers directly.
* **CPU: keep the default layout.** The span *loses* on every CPU
  configuration measured (0.5x-0.85x, worse for bigger jets). Each CPU thread
  walks elements sequentially, so the default layout is one perfectly
  prefetchable stream per thread, while the coefficient-major gather splits it
  into ``ncoeffs`` strided streams that defeat the hardware prefetcher.
* **Compute-bound kernels: layout cannot help.** A kernel limited by
  arithmetic throughput rather than memory (for example ``<4,4>`` ``double``
  products on consumer GPUs with reduced FP64 rate) runs the same under both
  layouts.

Because the layout choice is backend-dependent, a portable code should treat
it as a per-backend storage decision — for example, selecting the span only in
CUDA builds — rather than a global default.
