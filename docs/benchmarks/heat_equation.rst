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

* ``float``: ``2.7x`` at runtime lookup, jumping to ``5.4x`` at the aligned
  stage and holding through the fused stages. The alignment stage is the big
  jump, exactly as the standalone stencil-gather numbers predict.
* ``double``: ``1.8x`` at lookup easing to ``2.0x`` at fused SoA -- a flatter
  curve, because the larger double jets are less memory-bound at this shape.

The per-node source-division variant (more division work per node) leans harder
on the same optimizations and reaches about ``9.2x`` for ``float``.

The more telling number is the OTI solve cost relative to a plain ``double``
base solve that carries no sensitivities. Stacking the optimizations cuts that
overhead from about ``11x`` down to ``1.1x`` for ``float``, and from ``6.6x``
to ``2.9x`` for ``double``. A full-sensitivity ``float`` solve for roughly 10%
over a plain solve is the headline these isolated optimizations add up to.

Two details tie back to the isolation studies:

* The dominant ``float`` speedup arrives at the **aligned** stage, consistent
  with the alignment page: the stiffness gather is the memory-bound kernel, and
  aligning the small ``float`` jets lets its scattered neighbor loads coalesce.
* The **SoA** stage is only marginal here -- and slightly *hurts* the ``float``
  per-node variant -- consistent with the layout page: the stiffness gather is a
  small-jet gather, where AoS wins. So the production heat layout stays AoS; SoA
  is the tool for large-jet streaming, not this solve.

Why the OTI overhead grows with problem size
---------------------------------------------

A natural question is how the OTI solve's overhead scales as the problem grows.
Sweeping grid size and step count -- so the x-axis is total node-updates
(``num_nodes x num_steps``) -- shows the OTI/base wall-time ratio *rising* and
then plateauing, at about ``3.2x`` for double and ``2.8x`` for float on the
GTX 1650:

.. image:: ../_static/benchmarks/oti_overhead.png
   :alt: OTI/base wall-time ratio vs computational complexity
   :width: 100%

The intuitive guess is that the overhead grows because the larger OTI jet costs
more memory traffic. The per-node-update decomposition shows the mechanism is
actually the opposite of "OTI gets more expensive": it is the *base* solve being
under-utilized at small sizes. Plotting wall time *per node-update* (lower means
better GPU utilization), every curve falls as the problem grows -- everything is
latency- and launch-bound when the GPU is starved of work -- but the base scalar
solve falls much further than the OTI solve:

.. image:: ../_static/benchmarks/oti_overhead_saturation.png
   :alt: GPU per-node-update time, base vs OTI solve
   :width: 100%

* ``float``: the base solve drops from about ``9.2`` to ``0.60`` ns/node-update
  (15x) as it saturates near ``~1e9`` node-updates; the OTI solve drops only
  ``~7.7 -> ~1.6`` ns and flattens by ``~1e8``. At the smallest size the ratio
  is even *below 1* -- the scalar kernel is too small to fill the GPU, while the
  4x-heavier OTI kernel is already closer to saturation.
* ``double``: base ``~7.9 -> ~2.2`` ns (3.6x); OTI ``~13.3 -> ~7.1`` ns (1.9x).

The OTI solve carries four ``otinum<3,1>`` coefficients, so it does roughly 4x
the work per node and saturates the GPU *earlier*; the tiny scalar base solve
needs a much larger problem before it saturates. The ratio climbs because the
base denominator shrinks toward its throughput floor, not because OTI slows
down, and it plateaus once both are saturated -- the regime the existing sweep
already reaches (up to ``N=151``, 3.4M nodes, ``~2.9e10`` node-updates).

The plateau *value* is set by device memory bandwidth, not host-device copies. A
per-kernel profile shows no ``cudaMemcpy`` in the timed loop -- the field stays
resident on the device across timesteps -- and the overhead is concentrated in
the memory-bound stencil gather (``ComputeStiffnessForce``: ``3.9x`` for double,
``5.7x`` for float), exactly the kernel the alignment and layout studies single
out. A bandwidth-bound kernel that moves four coefficients where the base moves
one lands near a 3-4x ratio once saturated.

Reproducing
-----------

Clone the heat solver beside ``cpp_oti_lib`` (so its headers resolve at
``../include``) and run the optimization sweep:

.. code-block:: console

   git clone https://github.com/Samm-Py/heat_equation.git
   cd heat_equation
   python3 benchmarks/run_heat_optimization_benchmarks.py --grid-sizes 41

The stages are **cumulative**: each ``oti_heat_bench_<variant>`` binary bakes in
that stage plus every optimization before it (``aligned`` already uses the
unrolled arithmetic path, ``fused_aos`` is also aligned, and so on -- it is not
the alignment optimization in isolation). ``--variants`` runs a subset (always
keep ``naive`` -- it is the speedup and checksum baseline). In the output CSV,
``oti_over_base`` is that stage's **OTI solve overhead** -- its ``oti_solve``
wall time over the plain ``base_scalar_solve`` on the same grid, valid for any
subset -- ``speedup_vs_naive`` is the cumulative speedup over naive, and
``incremental_speedup`` is the marginal gain from that one stage, which needs the
immediately preceding stage in the same run to be meaningful:

.. code-block:: console

   # cumulative speedup through alignment, plus the marginal alignment gain
   python3 benchmarks/run_heat_optimization_benchmarks.py --grid-sizes 41 \
     --variants naive unrolled aligned

The runner writes a CSV and per-run logs, not a figure; plot the optimization
study separately with ``benchmarks/plot_heat_optimization_benchmarks.py``. For a
single optimization measured *independently* of the others, use the isolation
benchmarks in :doc:`gpu_optimization_workflow`.

The overhead-vs-complexity and per-node-update figures above come from the
grid-size scaling sweep and the per-kernel profile:

.. code-block:: console

   # scaling sweep (cpu/gpu x float/double, base vs OTI over grid sizes)
   benchmarks/run_benchmark.sh
   python3 benchmarks/plot_benchmark.py     # -> oti_overhead.png, oti_overhead_saturation.png

   # per-kernel base-vs-OTI breakdown at one grid size
   benchmarks/profile_kernels.sh 61
   python3 benchmarks/parse_kernel_profile.py 61

See the heat solver's own README for the CUDA Kokkos build and the full set of
runners and plotting scripts.
