GPU Optimization Benchmark Workflow
===================================

The optimization study uses the companion heat-equation sensitivity analysis
as its benchmark. This measures complete Kokkos kernels, memory access, and
solver wall time instead of extrapolating application performance from an
arithmetic microbenchmark.

The analysis uses ``otinum<3,1>`` for thermal diffusivity, source amplitude,
and source width. Six binaries build the same solver as cumulative stages:

``naive``
   Nested coefficient loops rebuild ``alpha + beta`` and call ``rank()``.
   Product lookup tables are disabled, coefficient storage has natural
   alignment, and the timestep uses the original operator chain.

``lookup``
   Multiplication, inverse/division, and Taylor composition use runtime product
   lookup tables.

``unrolled``
   Runtime table loops become the current compile-time folds. The storage is
   still deliberately left naturally aligned.

``aligned``
   The ordinary conditional 8- or 16-byte ``otinum`` alignment is enabled.

``fused_aos``
   The timestep uses ``fma_into`` and ``scale_add`` with the default
   array-of-structs storage.

``fused_soa``
   The current fused implementation uses coefficient-major ``soa_span``
   storage.

These variants are selected by benchmark-only compile definitions. Normal
library builds retain the current unrolled and aligned implementation.

Division Scenarios
------------------

The normal heat solver hoists
``1 / (2 * sigma * sigma)`` outside the timestep loop. That is the correct
application optimization, but it means division is not materially represented
in the timed solve.

The benchmark therefore records two mathematically equivalent scenarios:

``hoisted``
   The normal application.

``per-node``
   The source kernel reevaluates
   ``-r2 / (2 * sigma * sigma)`` at every node. This keeps the heat-equation
   workload and output unchanged while putting OTI inverse/division inside the
   timed kernel.

Build The CUDA Variants
-----------------------

Configure the companion analysis checkout against a CUDA Kokkos installation
and this library's headers:

.. code-block:: console

   cmake -S heat_equation_oti_analysis \
     -B heat_equation_oti_analysis/build-cuda \
     -DCMAKE_CXX_COMPILER=/path/to/kokkos/bin/nvcc_wrapper \
     -DCMAKE_PREFIX_PATH=/path/to/kokkos-cuda-install \
     -DCPP_OTI_LIB_INCLUDE_DIR="$PWD/include"

The collector can build all twelve variant/precision targets serially. Serial
compilation avoids temporary-file collisions in some ``nvcc_wrapper`` builds.

Collect Repeated Runs
---------------------

.. code-block:: console

   python3 \
     heat_equation_oti_analysis/benchmarks/run_heat_optimization_benchmarks.py \
     --build \
     --build-dir heat_equation_oti_analysis/build-cuda \
     --runs 5 \
     --grid-sizes 41 61 \
     --total-time 0.01 \
     --output benchmark_results/heat_optimization_gpu

Runs alternate forward and reverse variant order to reduce systematic thermal
or clock bias. The collector rejects non-CUDA execution by default and saves:

* every application's ``run_config.csv``, ``timing_summary.csv``,
  ``solution_checksum.csv``, and console log;
* ``heat_optimization_results.csv`` with wall times, application speedups, and
  checksum status;
* ``metadata.json`` with revisions, dirty-worktree state, platform, and GPU
  information.

The final four OTI coefficient sums are compared with the naive variant using
precision-appropriate tolerances. A checksum mismatch remains visible in the
CSV and prevents plotting.

Plot The Results
----------------

.. code-block:: console

   python3 \
     heat_equation_oti_analysis/benchmarks/plot_heat_optimization_benchmarks.py \
     benchmark_results/heat_optimization_gpu

The plotting script writes PDF and PNG versions of:

``heat_optimization_stages``
   Median OTI solve wall time through all cumulative stages.

``heat_optimization_speedups``
   End-to-end speedup relative to the no-product-table application.

``heat_optimization_incremental``
   The isolated ratio between adjacent cumulative stages.

``heat_optimization_summary.csv``
   Median and sample standard deviation for each configuration.

Reading The Result
------------------

On the GTX 1650 validation run at ``N=41`` with three repetitions:

* float, normal hoisted source: runtime product tables improved the complete
  solve by about ``2.09x``; alignment raised the cumulative speedup to about
  ``3.90x``;
* float, per-node division: runtime product tables improved the solve by about
  ``3.28x``; alignment raised the cumulative speedup to about ``6.13x``;
* double showed smaller but consistent gains: about ``1.56x`` cumulative for
  the normal source and ``2.09x`` for per-node division.

Compile-time unrolling is nearly neutral for this ``<3,1>`` application because
the runtime table is tiny. Its larger gains belong to higher-order/larger-shape
arithmetic workloads and should not be attributed to this heat solve.

The alignment result is application-dependent in exactly the way a streaming
microbenchmark missed: float ``<3,1>`` improved by roughly ``1.8x`` between the
unaligned and aligned application variants. Conversely, SoA is neutral to
slightly slower for this small first-order jet, matching the guidance in
:doc:`soa_layout`; its main benefit is predictable coalescing for larger jets.
