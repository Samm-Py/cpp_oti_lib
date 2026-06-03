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
#include <iostream>

#include "otinum/otinum.hpp"

int main()
{
    using T = oti::otinum<2, 2>;

    T x = T::variable(0, 1.5);
    T y = T::variable(1, 0.3);
    T f = oti::sin(x * y) + oti::exp(x);

    std::cout << "f = " << f.real() << '\n';
    std::cout << "df/dx = " << f.partial({1, 0}) << '\n';
    std::cout << "df/dy = " << f.partial({0, 1}) << '\n';
    std::cout << "d2f/dxdy = " << f.partial({1, 1}) << '\n';
}
```

Compile the example directly:

```sh
c++ -std=c++17 -I include examples/basic.cpp -o /tmp/otinum_basic
/tmp/otinum_basic
```

## Core API

`oti::otinum<M, N>` is a value type backed by a fixed-size coefficient array:

- normal scalar mode: `std::array<double, ncoeffs>`
- Kokkos mode: `Kokkos::Array<double, ncoeffs>`

Common operations:

- `T::variable(i, value)` creates `value + e_i`.
- `real()` returns the scalar coefficient.
- `operator[](flat_index)` accesses the raw Taylor coefficient.
- `deriv(alpha)` returns the stored normalized coefficient for multi-index
  `alpha`.
- `partial(alpha)` returns the ordinary partial derivative value, equal to
  `alpha! * deriv(alpha)`.
- Arithmetic operators support `otinum` and `double` combinations.
- `<cmath>`-style functions currently include `exp`, `log`, `log10`, `logb`,
  `pow`, `sqrt`, `cbrt`, `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`, and
  `abs`.

## Coefficient Semantics

For a smooth scalar function `f(x_1, ..., x_M)`, coefficient `alpha` stores:

```text
(1 / alpha!) * partial^alpha f
```

Use `deriv(alpha)` when you want the normalized Taylor coefficient. Use
`partial(alpha)` when you want the usual derivative value.

Multi-indices outside the configured order return zero from `deriv()` and
`partial()`.

## Repository Layout

```text
include/otinum/otinum.hpp            umbrella header
include/otinum/core.hpp              otinum value type and arithmetic
include/otinum/functions.hpp         public math functions
include/otinum/taylor.hpp            scalar Taylor composition helpers
include/otinum/detail/kokkos_compat.hpp std::array/Kokkos::Array compatibility
include/otinum/detail/binom.hpp      constexpr binomial/factorial helpers
include/otinum/detail/multi_index.hpp multi-index ranking and lookup tables
CMakeLists.txt                       optional Kokkos/Python build targets
examples/basic.cpp                   small usage example
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

`partial()` and `deriv()` accept Python lists or tuples whose length matches
the number of variables. `data()` returns a plain Python list of normalized
Taylor coefficients.
