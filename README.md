# cpp_oti_lib

Header-only C++17 library for **order-truncated imaginary (OTI) numbers** —
truncated multivariate Taylor polynomials for forward-mode automatic
differentiation. A single overloaded model evaluation returns the function value
together with its derivatives, so many numerical kernels can be differentiated by
changing selected scalar types instead of rewriting the kernel logic.

For the original OTI library, see
[`mauriaristi/otilib`](https://github.com/mauriaristi/otilib).

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

**The full documentation is published at
[samm-py.github.io/cpp_oti_lib](https://samm-py.github.io/cpp_oti_lib/).**
It is built from [`docs/`](docs/) with Sphinx and redeployed automatically on every
push to `master`.

- **Getting started** —
  [installation and a first example](https://samm-py.github.io/cpp_oti_lib/installation.html),
  then the
  [basic usage tutorial](https://samm-py.github.io/cpp_oti_lib/tutorials/basic_usage.html).
- **Tutorials** —
  [float coefficients](https://samm-py.github.io/cpp_oti_lib/tutorials/float_coefficients.html),
  [directional derivatives](https://samm-py.github.io/cpp_oti_lib/tutorials/directional_derivatives.html),
  [CMake packaging](https://samm-py.github.io/cpp_oti_lib/tutorials/cmake_package.html),
  [Python bindings](https://samm-py.github.io/cpp_oti_lib/tutorials/python_bindings.html),
  [Kokkos on CPU](https://samm-py.github.io/cpp_oti_lib/tutorials/kokkos_cpu.html) and
  [GPU](https://samm-py.github.io/cpp_oti_lib/tutorials/kokkos_gpu.html), and
  [coefficient-major (SoA) storage](https://samm-py.github.io/cpp_oti_lib/tutorials/soa_layout.html).
- **API reference** —
  [core API and coefficient semantics](https://samm-py.github.io/cpp_oti_lib/api/index.html),
  and [choosing M and N](https://samm-py.github.io/cpp_oti_lib/api/choosing_m_and_n.html)
  (the compile-time cost of each shape).
- **Benchmarks** — the
  [GPU optimization workflow](https://samm-py.github.io/cpp_oti_lib/benchmarks/gpu_optimization_workflow.html)
  (arithmetic, alignment, layout, and fused-operation isolation studies) and the
  [end-to-end heat-equation study](https://samm-py.github.io/cpp_oti_lib/benchmarks/heat_equation.html).

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
