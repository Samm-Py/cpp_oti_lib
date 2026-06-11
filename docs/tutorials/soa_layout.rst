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

A Complete Host Example
-----------------------

The span itself has no Kokkos dependency; it works over any caller-owned
buffer. The program below evaluates ``f(x, y) = x*x + 3*y`` for four elements,
scatters the jets into a coefficient-major buffer, and gathers one back.

.. code-block:: cpp

   #include <cstddef>
   #include <iostream>
   #include <vector>

   #include "otinum/otinum.hpp"

   int main()
   {
       using T = oti::otinum<2, 2>;
       using Span = oti::soa_span<2, 2>;

       std::size_t const n = 4;
       std::vector<double> buffer(Span::required_size(n), 0.0);
       Span values(buffer.data(), n);

       // Evaluate f(x, y) = x*x + 3*y at (x, y) = (i, 10) for each element and
       // scatter the resulting jet into the coefficient-major buffer.
       for (std::size_t i = 0; i < n; ++i) {
           T x = T::variable(0, static_cast<double>(i));
           T y = T::variable(1, 10.0);
           values.store(i, x * x + 3.0 * y);
       }

       // The buffer is coefficient-major: the real parts of all four elements
       // are contiguous, then all four df/dx coefficients, and so on.
       std::cout << "real parts (buffer[0..3]):  ";
       for (std::size_t i = 0; i < n; ++i) {
           std::cout << buffer[i] << " ";
       }
       std::cout << "\ndf/dx parts (buffer[4..7]): ";
       for (std::size_t i = 0; i < n; ++i) {
           std::cout << buffer[n + i] << " ";
       }
       std::cout << "\n";

       // Gather one element back into an ordinary otinum and use it as usual.
       T f = values.load(3);
       std::cout << "element 3: f = " << f.real()
                 << ", df/dx = " << f.partial({1, 0})
                 << ", d2f/dx2 = " << f.partial({2, 0}) << "\n";

       return 0;
   }

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
larger than cache) and checks that the checksums match bit-exactly. Build it
with both options enabled and run it on the backend you care about:

.. code-block:: console

   cmake -S . -B build-kokkos-gpu \
     -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_CXX_COMPILER=/path/to/nvcc_wrapper \
     -DCMAKE_PREFIX_PATH=/path/to/kokkos-cuda-install \
     -DOTI_ENABLE_KOKKOS=ON \
     -DOTI_BUILD_BENCHMARKS=ON

   cmake --build build-kokkos-gpu --target benchmark_soa_layout --parallel
   ./build-kokkos-gpu/benchmark_soa_layout

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
* **GPU, first-order jets (``<3,1>`` and similar): no benefit.** Small aligned
  jets already coalesce well enough that both layouts sit at the bandwidth
  peak. The OTI heat-equation study (``<3,1>``) gains nothing from the span.
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
