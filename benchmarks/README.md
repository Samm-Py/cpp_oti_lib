# Optimization isolation benchmarks

Each benchmark here isolates **one** of the library's optimizations by flipping
exactly that axis and holding everything else at the default, so its effect can
be read on its own. The companion heat-equation study
(`heat_equation_oti_analysis`) is then the culmination: all of these stacked in
a real, end-to-end solve.

| benchmark | optimization isolated | regime | variants | metric |
|---|---|---|---|---|
| `bench_arithmetic` | product table + compile-time unrolling | compute-bound, register-resident | `naive` / `lookup` / `unrolled` | `ns_per_op` |
| `bench_alignment` | conditional coefficient alignment | memory-bound streaming | `natural` / `aligned` | `useful_gbps` |
| `bench_layout` | array-of-structs vs coefficient-major (`soa_span`) | memory-bound **stream + gather** | `aos` / `soa` | `useful_gbps` |
| `bench_fused` | fused `axpy` / `fma_into` vs operator chains | compute-bound | `chain` / `fused` | `ns_per_op` |

All four emit the same tidy CSV schema (see `bench_common.hpp`):

```
backend,coeff_type,M,N,ncoeffs,nproducts,kernel,variant,repetition,metric,value,checksum
```

`value` is the metric (`ns_per_op`: lower is better; `useful_gbps`: only the
useful coefficient bytes are counted, so dead bytes show as a lower rate). The
two variants of each benchmark compute the same thing, so the `checksum` column
must match across them — a mismatch is a bug, not noise.

## Building

The benchmarks need Kokkos. Configure the library with
`-DOTI_ENABLE_KOKKOS=ON -DOTI_BUILD_BENCHMARKS=ON`; for GPU numbers use a
CUDA Kokkos build (`nvcc_wrapper` as the compiler, g++ 11+ as its host
compiler — see the "Kokkos GPU" tutorial in `docs/`). The arithmetic path is a
compile-time switch over the whole `otinum`, so it builds as three binaries
(`bench_arithmetic_{naive,lookup,unrolled}`); the others are single binaries.

## Running

`run_benchmarks.py` builds (with `--build`) and runs every benchmark, runs the
three arithmetic paths and concatenates them, and writes one CSV per benchmark
plus `metadata.json` (GPU, driver, git revision):

```sh
python3 benchmarks/run_benchmarks.py --build --build-dir build-cuda \
  --output benchmarks/results
```

Each binary is also runnable directly for a quick look, e.g.
`./build-cuda/benchmarks/bench_layout 20 3` (timed passes, repetitions).

## Plotting

`plot_benchmark.py` renders any of the CSVs (or a whole directory): metric vs
algebra size, one line per variant, faceted by precision × kernel.

```sh
python3 benchmarks/plot_benchmark.py benchmarks/results
```

## Reading them together

- **arithmetic** — the unrolled fold's win grows steeply with coefficient count
  (`naive` is `O(ncoeffs^2)` per multiply); `lookup` sits in between.
- **alignment** — a clean win for medium jets whose byte size lets the rule
  reach 16-byte loads; a no-op where it cannot.
- **layout** — the SoA/AoS decision is access-pattern dependent: SoA wins
  contiguous streaming of larger jets, AoS wins scattered gathers (each neighbor
  is one contiguous load), which is why the heat solver's stiffness gather keeps
  AoS. This benchmark is the data behind that choice.
- **fused** — `fma_into` removes the multiply's temporary (a real win for the
  multiply-accumulate pattern); `axpy` is near-neutral.
