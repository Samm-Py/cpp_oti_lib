# cpp_oti_lib

Header-only C++ implementation of static OTI numbers, represented as truncated
multivariate Taylor polynomials.

The core library is the static scalar format: `oti::otinum<M, N>` fixes the
number of variables and truncation order at compile time, uses operator
overloads for arithmetic, and stores coefficients inline. In normal C++ builds
the coefficient container is `std::array`; in Kokkos builds it is
`Kokkos::Array` and the arithmetic entry points are annotated for Kokkos
kernels.

An `oti::otinum<M, N>` stores all Taylor coefficients for `M` variables up to
total order `N`. The coefficient count is:

```text
C(M + N, N)
```

For example, `oti::otinum<2, 2>` stores six coefficients in this order:

```text
(0,0), (1,0), (0,1), (2,0), (1,1), (0,2)
```

This layout matches the usual graded multi-index ordering: coefficients are
grouped by total derivative order, with the real value at flat index zero.

## Quick Start

The library has no build step. Include the umbrella header and add `include/`
to your compiler include path.

```cpp
#include <cmath>
#include <iomanip>
#include <iostream>

#include "otinum/otinum.hpp"

int main()
{
    using T = oti::otinum<2, 2>;

    double x0 = 1.5;
    double y0 = 0.3;

    T x = T::variable(0, x0);
    T y = T::variable(1, y0);
    T f = oti::sin(x * y) + oti::exp(x);

    double xy = x0 * y0;
    double analytic_f = std::sin(xy) + std::exp(x0);
    double analytic_dfdx = y0 * std::cos(xy) + std::exp(x0);
    double analytic_dfdy = x0 * std::cos(xy);
    double analytic_d2fdx2 = -y0 * y0 * std::sin(xy) + std::exp(x0);
    double analytic_d2fdxdy = std::cos(xy) - xy * std::sin(xy);
    double analytic_d2fdy2 = -x0 * x0 * std::sin(xy);

    auto print_check = [](const char* name, double analytic, double ad) {
        std::cout << std::setw(8) << name
                  << " analytic=" << std::setw(16) << analytic
                  << " ad=" << std::setw(16) << ad
                  << " abs_diff=" << std::abs(analytic - ad) << '\n';
    };

    print_check("f", analytic_f, f.real());
    print_check("df/dx", analytic_dfdx, f.partial({1, 0}));
    print_check("df/dy", analytic_dfdy, f.partial({0, 1}));
    print_check("d2f/dx2", analytic_d2fdx2, f.partial({2, 0}));
    print_check("d2f/dxdy", analytic_d2fdxdy, f.partial({1, 1}));
    print_check("d2f/dy2", analytic_d2fdy2, f.partial({0, 2}));
}
```

Compile programs that include the header with `-I include`.

## Core API

`oti::otinum<M, N, Coeff = double>` is a value type backed by a fixed-size
coefficient array:

- normal scalar mode: `std::array<Coeff, ncoeffs>`
- Kokkos mode: `Kokkos::Array<Coeff, ncoeffs>`

Use `oti::otinum<M, N>` for the default `double` coefficients, or instantiate
`oti::otinum<M, N, float>` when single-precision storage and math are preferred.

Common operations:

- `T::variable(i, value)` creates `value + e_i`.
- `real()` returns the scalar coefficient.
- `operator[](flat_index)` accesses the raw Taylor coefficient.
- `coeff(alpha)` returns the stored normalized coefficient for multi-index
  `alpha`.
- `partial(alpha)` returns the ordinary partial derivative value, equal to
  `alpha! * coeff(alpha)`.
- Arithmetic operators support `otinum` and arithmetic scalar combinations.
- `<cmath>`-style functions currently include `exp`, `log`, `log10`, `log_base`,
  `pow`, `sqrt`, `cbrt`, `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`, and
  `abs`.

## Coefficient Semantics

For a smooth scalar function `f(x_1, ..., x_M)`, coefficient `alpha` stores:

```text
(1 / alpha!) * partial^alpha f
```

Use `coeff(alpha)` when you want the normalized Taylor coefficient. Use
`partial(alpha)` when you want the usual derivative value.

Multi-indices outside the configured order return zero from `coeff()` and
`partial()`.

## Choosing M and N

Every `otinum<M, N>` builds its multi-index and truncated-product tables at
compile time, so each distinct shape you instantiate adds to the build, not to
the runtime. The product convolutions (`operator*`, `inv`/division, `trunc_mul`,
and the Taylor composition behind the elementary functions) are emitted as
compile-time `index_sequence` folds that read those tables at constant pack
indices. As a result the `(lhs, rhs, out)` offsets fold to literals and, at
`-O2` or higher, the runtime code is straight-line register arithmetic with the
index tables constant-folded away -- they do not appear in the binary. (Driving
the same convolutions with an ordinary runtime loop over the tables would defeat
this: the compiler then materializes the index tables on the stack and round-
trips every coefficient through memory.)

The build cost is driven by the number of product terms,
`detail::tables<M, N>::nproducts`, which grows quickly with the coefficient
count `C(M + N, N)`. The following are measured per translation unit with
`g++ -O2` and are a rough guide rather than hard limits:

```text
shape    coeffs   product terms   compile     peak compiler RAM
<3,3>        20              84      <1 s                 <0.1 GB
<4,4>        70             495      ~1 s                 <0.3 GB
<5,4>       126            1001      ~3 s                 ~0.8 GB
<5,5>       252            3003     ~13 s                 ~4.2 GB
<6,6>       924          (large)    ~90 s     >11 GB, often OOM-killed
```

Peak compile memory runs very roughly 1-1.5 MB per product term, and the curve
is super-linear, so it climbs steeply. As practical guidance:

- Keep `C(M + N, N)` at or below ~70 (up to about `<4,4>`) for fast,
  interactive builds on any machine.
- Treat ~250 coefficients (`<5,5>`) as a soft ceiling: it builds, but wants a
  large-RAM machine and over ten seconds per shape.
- Shapes with coefficient counts in the high hundreds (`<6,6>` and beyond) can
  exhaust compiler memory and should be avoided unless you have measured the
  cost on your build host.

Because each instantiated shape pays this cost independently, prefer reusing a
small set of `(M, N)` shapes over scattering many large ones across a build.

## Repository Layout

```text
include/otinum/otinum.hpp            umbrella header
include/otinum/core.hpp              otinum value type and arithmetic
include/otinum/functions.hpp         public math functions
include/otinum/taylor.hpp            scalar Taylor composition helpers
include/otinum/detail/kokkos_compat.hpp std::array/Kokkos::Array compatibility
include/otinum/detail/binom.hpp      constexpr binomial/factorial helpers
include/otinum/detail/multi_index.hpp multi-index ranking and lookup tables
CMakeLists.txt                       optional CTest/Kokkos/Python build targets
tests/test_*.cpp                     focused assert-based unit tests
tests/test_kokkos_smoke.cpp          Kokkos kernel smoke test
CPP_HEADER_ONLY_DESIGN.md            design notes and future work
```

## Testing

There is no build system required for the focused C++ unit tests:

```sh
tests/run_unit_tests.sh
```

The runner compiles each `tests/test_*.cpp` file into `/tmp/otinum_unit_tests`
and runs it as a separate executable. This keeps failures localized to one
library area. Each run also writes a timestamped report to `logs/`, including
compiler commands and output from passing tests:

```text
logs/unit_tests_YYYYMMDD_HHMMSS.log
```

You can override the output locations if needed:

```sh
BUILD_DIR=/tmp/my_otinum_tests LOG_DIR=/tmp/my_otinum_logs tests/run_unit_tests.sh
```

The same focused tests are also registered with CTest by default:

```sh
cmake -S . -B build
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

Set `-DOTI_BUILD_TESTS=OFF` when you only want the interface target.

The original broad smoke test can still be compiled directly:

```sh
c++ -std=c++17 -I include tests/smoke.cpp -o /tmp/otinum_smoke
/tmp/otinum_smoke
```

Together, the focused tests cover coefficient layout, construction,
arithmetic, truncated operations, transcendental functions, and a few larger
`(M, N)` shapes.

## Kokkos Support

Kokkos support is opt-in. Define `OTI_ENABLE_KOKKOS` through the CMake option
and provide a Kokkos installation through `CMAKE_PREFIX_PATH`. The CMake target
links against `Kokkos::kokkos`, which supplies the include paths, compile
definitions, backend flags, and link flags.

Example using a Kokkos install at `/root/Research/kokkos-install`:

```sh
cmake -S . -B build-kokkos \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=g++-10 \
  -DCMAKE_PREFIX_PATH=/root/Research/kokkos-install \
  -DOTI_ENABLE_KOKKOS=ON \
  -DOTI_BUILD_PYTHON=OFF

cmake --build build-kokkos --parallel 2
ctest --test-dir build-kokkos --output-on-failure
```

The tested local Kokkos configuration is OpenMP-backed:

```sh
cmake -S /root/Research/kokkos -B /root/Research/kokkos/build-gcc10 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/root/Research/kokkos-install \
  -DCMAKE_CXX_COMPILER=g++-10 \
  -DKokkos_ENABLE_OPENMP=ON \
  -DKokkos_ARCH_NATIVE=ON

cmake --build /root/Research/kokkos/build-gcc10 --target install --parallel 2
```

The current Kokkos `develop`/5.1 line requires GCC 10.4 or newer. On Ubuntu
20.04, `g++-10` 10.5 works for this configuration. If using an older compiler,
pin Kokkos to a compatible release or install a newer compiler.

## Python Bindings

Python bindings are optional and are not required for the static scalar C++
library. The wrapper exposes a small fixed grid of concrete template
instantiations:

```text
OTI_1_1  OTI_1_2  OTI_1_3
OTI_2_1  OTI_2_2  OTI_2_3
OTI_3_1  OTI_3_2  OTI_3_3
```

Install the extension in editable mode:

```sh
python -m pip install -e .
```

Then import it from Python:

```python
import otinum as oti

T = oti.OTI_2_3
x = T.variable(0, 1.5)
y = T.variable(1, 0.3)
f = oti.sin(x * y) + oti.exp(x)

print(f.real())
print(f.partial([1, 0]))
print(f.data())
```

The Python binding smoke tests use `pytest`:

```sh
python -m pip install -e ".[test]"
pytest tests/test_python_bindings.py
```

`partial()` and `coeff()` accept Python lists or tuples whose length matches
the number of variables. `data()` returns a plain Python list of normalized
Taylor coefficients.

## Continuous Integration

The GitHub Actions workflow in `.github/workflows/ci.yml` runs the focused C++
CTest targets, the direct shell test runner, the Kokkos CPU/OpenMP smoke test,
a conditional Kokkos GPU/CUDA smoke test when a CUDA device and `nvcc` are
available, the Python binding tests, and the Doxygen/Sphinx documentation
build. On pushes to `master`, it publishes the documentation site with the
generated coverage report under `generated/coverage/index.html`. The coverage
report is built from the scalar focused tests plus the Kokkos OpenMP smoke test,
so it exercises both ordinary and `OTI_ENABLE_KOKKOS` header paths.

By default, the GPU job runs on `ubuntu-latest` and skips when no CUDA device is
visible. Set the repository variable `KOKKOS_GPU_RUNNER` to a CUDA-capable runner
label to make that job exercise an available GPU.
