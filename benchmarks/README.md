# Optimization isolation benchmarks

Each benchmark here isolates **one** of the library's optimizations by flipping
exactly that axis and holding everything else at the default, so its effect can
be read on its own. The companion heat-equation study
(`heat_equation_oti_analysis`) is then the culmination: all of these stacked in
a real, end-to-end solve.

| benchmark | optimization isolated | regime | variants | metric |
|---|---|---|---|---|
| `bench_arithmetic` | product table + compile-time unrolling | compute-bound, register-resident | `naive` / `lookup` / `unrolled` | `ns_per_op` |
| `bench_alignment_source_update_gather` | conditional coefficient alignment | the FE kernels of an explicit PDE step: source/load eval, operator apply (stencil gather), nodal update, mass solve | `natural` / `aligned` | `ns_per_node` |
| `bench_layout` | array-of-structs vs coefficient-major (`soa_span`) | memory-bound **stream + gather** | `aos` / `soa` | `useful_gbps` |
| `bench_fused` | fused `axpy` / `fma_into` vs operator chains | compute-bound | `chain` / `fused` | `ns_per_op` |

All four emit the same tidy CSV schema (see `bench_common.hpp`):

```
backend,coeff_type,M,N,ncoeffs,nproducts,kernel,variant,repetition,metric,value,checksum
```

`value` is the metric (`ns_per_op`/`ns_per_node`: lower is better;
`useful_gbps`: only the useful coefficient bytes are counted, so dead bytes
show as a lower rate). The two variants of each benchmark compute the same
thing, so the `checksum` column must match across them — a mismatch is a bug,
not noise.

## Building

The benchmarks need Kokkos. Configure the library with
`-DOTI_ENABLE_KOKKOS=ON -DOTI_BUILD_BENCHMARKS=ON`; for GPU numbers use a
CUDA Kokkos build (`nvcc_wrapper` as the compiler, g++ 11+ as its host
compiler — see the "Kokkos GPU" tutorial in `docs/`). The arithmetic path is a
compile-time switch over the whole `otinum`, so it builds as three binaries
(`bench_arithmetic_{naive,lookup,unrolled}`); the others are single binaries.

## Running

`run_benchmarks.py` builds (with `--build`) and runs every benchmark, runs the
three arithmetic paths and the two source/update/gather alignment binaries, and
writes one CSV per benchmark plus `metadata.json` (GPU, driver, git revision):

```sh
python3 benchmarks/run_benchmarks.py --build --build-dir build-cuda \
  --output benchmarks/results
```

Each binary measures several repetitions per configuration internally (default
11), all back-to-back in one process. To also smooth out run-to-run variance
(cold start, GPU clock state) on the near-neutral points, pass `--runs N`: it
relaunches every binary `N` times and pools the rows, so the plotter's
per-configuration median is taken over `N * repetitions` samples. `metadata.json`
records the `runs` count.

```sh
python3 benchmarks/run_benchmarks.py --runs 5 --build-dir build-cuda \
  --output benchmarks/results
```

Each binary is also runnable directly for a quick look, e.g.
`./build-cuda/benchmarks/bench_layout 20 3` (timed passes, repetitions).
For `bench_alignment_source_update_gather`, the first argument is the node
count, while the fourth argument is the minimum logical node-updates used for
timing. By default it allocates 68,921 nodes -- the 41x41x41 (N=41) heat-problem
DOF count, so the per-kernel numbers line up with the end-to-end heat
application stage -- and lets the target-time calibration choose the number of
timed repetitions; pass the fourth argument only when you explicitly want a
larger work floor.

## Plotting

`plot_benchmark.py` renders any of the CSVs (or a whole directory): metric vs
algebra size, one line per variant, faceted by precision × kernel.

```sh
python3 benchmarks/plot_benchmark.py benchmarks/results
```

## Reading them together

- **arithmetic** — the unrolled fold's win grows steeply with coefficient count
  (`naive` is `O(ncoeffs^2)` per multiply); `lookup` sits in between.
- **alignment** — `bench_alignment_source_update_gather` runs the real FE
  kernels of an explicit PDE step (source/load eval, operator apply via stencil
  gather, nodal update, and the OTI mass solve) at a representative DOF count
  (the N=41 companion heat grid), with real `otinum` values, so by-value
  parameters and temporaries exercise the generated CUDA register pressure.
  Read it kernel-by-kernel against the
  end-to-end heat stage; do not collapse it into one universal alignment number.
- **layout** — the SoA/AoS decision is access-pattern dependent: SoA wins
  contiguous streaming of larger jets, AoS wins scattered gathers (each neighbor
  is one contiguous load), which is why the heat solver's stiffness gather keeps
  AoS. This benchmark is the data behind that choice.
- **fused** — `fma_into` removes the multiply's temporary (a real win for the
  multiply-accumulate pattern); `axpy` is near-neutral.
