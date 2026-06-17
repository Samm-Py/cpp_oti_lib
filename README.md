# cpp_oti_lib

Header-only C++17 library for **order-truncated imaginary (OTI) numbers** —
truncated multivariate Taylor polynomials for forward-mode automatic
differentiation. A single overloaded model evaluation returns the function value
together with its derivatives, so many numerical kernels can be differentiated by
changing selected scalar types instead of rewriting the kernel logic.

The core type is `oti::otinum<M, N, Coeff = double>`: `M` independent variables,
total derivative order `N`, and coefficient type `Coeff` (`double` or `float`).
Shapes are fixed at compile time and coefficients are stored inline — `std::array`
in normal builds, `Kokkos::Array` in Kokkos builds.

## At a glance

```cpp
#include "otinum/otinum.hpp"

using T = oti::otinum<2, 2>;        // 2 variables, derivatives through order 2

T x = T::variable(0, 1.5);
T y = T::variable(1, 0.3);
T f = oti::sin(x * y) + oti::exp(x);

f.real();           // function value
f.partial({1, 0});  // df/dx
f.partial({1, 1});  // d2f/dxdy
```

There is no build step: add `include/` to your compiler search path (`-I include`)
and include the umbrella header `otinum/otinum.hpp`.

## Documentation

The full documentation is in [`docs/`](docs/), built with Sphinx (and published to
GitHub Pages when enabled):

- **Getting started** — [installation and a first example](docs/installation.rst),
  then the [basic usage tutorial](docs/tutorials/basic_usage.rst).
- **Tutorials** — [float coefficients](docs/tutorials/float_coefficients.rst),
  [directional derivatives](docs/tutorials/directional_derivatives.rst),
  [CMake packaging](docs/tutorials/cmake_package.rst),
  [Python bindings](docs/tutorials/python_bindings.rst),
  [Kokkos on CPU](docs/tutorials/kokkos_cpu.rst) and
  [GPU](docs/tutorials/kokkos_gpu.rst), and
  [coefficient-major (SoA) storage](docs/tutorials/soa_layout.rst).
- **API reference** — [core API and coefficient semantics](docs/api/index.rst),
  and [choosing M and N](docs/api/choosing_m_and_n.rst) (the compile-time cost of
  each shape).
- **Benchmarks** — the
  [GPU optimization workflow](docs/benchmarks/gpu_optimization_workflow.rst)
  (arithmetic, alignment, layout, and fused-operation isolation studies) and the
  [end-to-end heat-equation study](docs/benchmarks/heat_equation.rst).

## Features

- Forward-mode AD through operator overloading, plus the full `<cmath>` surface and
  a `std::numeric_limits` specialization, so generic code can substitute
  `oti::otinum` for `double` without edits.
- Single- or double-precision coefficient storage (`oti::otinum<M, N, float>`).
- Optional **Kokkos** path for CPU and GPU backends.
- Optional **Python bindings** for a fixed grid of `(M, N)` instantiations.
- Fused accumulation helpers (`oti::axpy`, `oti::scale_add`, `oti::fma_into`) for
  hot loops.

## Building the tests and benchmarks

The headers need no build. To compile the focused unit tests, the optional Kokkos
paths, or the benchmark suite, see [installation](docs/installation.rst). In brief:

```sh
cmake -S . -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

To build this documentation site locally (Doxygen + Sphinx), see
[Building These Docs](docs/installation.rst) in the installation guide.
