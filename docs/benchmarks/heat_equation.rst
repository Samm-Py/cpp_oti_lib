Heat Equation: Optimizations Stacked End-to-End
===============================================

The isolation benchmarks each flip exactly one optimization and hold everything
else fixed. This page is the culmination: a real, end-to-end PDE solve -- a 3D
heat equation with an OTI sensitivity jet at every node -- with the same
optimizations switched on one at a time, so the per-isolation results can be
seen compounding in production.

The solver lives in its own repository, built to exercise ``cpp_oti_lib`` as a
drop-in ``Scalar`` type:

   https://github.com/Samm-Py/heat_equation

It solves the heat equation on the unit cube with a Gaussian source. Every nodal
temperature is an ``oti::otinum`` that carries derivatives with respect to the
source parameters, so a single forward solve also returns the sensitivities. The
study below is the ``N=41`` grid -- ``68,921`` nodes, the same working set the
alignment benchmark uses.

The stacked optimization stages
-------------------------------

Each stage turns on exactly one of the isolation benchmarks, cumulatively on top
of the previous ones:

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - stage
     - what it turns on
     - isolation benchmark
   * - no product table
     - naive multi-index reconstruction
     - ``bench_arithmetic`` (naive)
   * - runtime lookup
     - precomputed product tables
     - ``bench_arithmetic`` (lookup)
   * - compile-time fold
     - unrolled product tables
     - ``bench_arithmetic`` (unrolled)
   * - aligned
     - conditional ``otinum`` alignment
     - ``bench_alignment_source_update_gather``
   * - fused AoS
     - ``axpy`` / ``fma_into`` helpers
     - ``bench_fused``
   * - fused SoA
     - coefficient-major ``soa_span``
     - ``bench_layout``

.. image:: ../_static/benchmarks/heat_optimization_speedups.png
   :alt: Cumulative heat-equation application speedup by optimization stage
   :width: 100%

End-to-end result
-----------------

On the GTX 1650, stacking the stages speeds up the full OTI heat solve relative
to the no-product-table baseline by about (``N=41``, source term hoisted):

* ``float``: ``2.1x`` at runtime lookup, jumping to ``3.9x`` at the aligned
  stage and holding through the fused stages. The alignment stage is the big
  jump, exactly as the standalone stencil-gather numbers predict.
* ``double``: ``1.41x`` at lookup easing to ``1.56x`` at fused SoA -- a flatter
  curve, because the larger double jets are less memory-bound at this shape.

The per-node source-division variant (more division work per node) leans harder
on the same optimizations and reaches about ``6.1x`` for ``float``.

The more telling number is the OTI solve cost relative to a plain ``double``
base solve that carries no sensitivities. Stacking the optimizations cuts that
overhead from about ``8.0x`` down to ``1.2x`` for ``float``, and from ``4.6x``
to ``2.5x`` for ``double``. A full-sensitivity ``float`` solve for roughly 20%
over a plain solve is the headline these isolated optimizations add up to.

Two details tie back to the isolation studies:

* The dominant ``float`` speedup arrives at the **aligned** stage, consistent
  with the alignment page: the stiffness gather is the memory-bound kernel, and
  aligning the small ``float`` jets lets its scattered neighbor loads coalesce.
* The **SoA** stage is only marginal here -- and slightly *hurts* the ``float``
  per-node variant -- consistent with the layout page: the stiffness gather is a
  small-jet gather, where AoS wins. So the production heat layout stays AoS; SoA
  is the tool for large-jet streaming, not this solve.

Reproducing
-----------

Clone the heat solver beside ``cpp_oti_lib`` (so its headers resolve at
``../include``) and run the optimization sweep:

.. code-block:: console

   git clone https://github.com/Samm-Py/heat_equation.git
   cd heat_equation
   python3 benchmarks/run_heat_optimization_benchmarks.py --grid-sizes 41

See the heat solver's own README for the CUDA Kokkos build and the plotting
script that produces the figure above.
