#pragma once

// The suite's checks (expect_near and the tests' direct assert calls) are
// assert-based, and CMake's Release/RelWithDebInfo presets add -DNDEBUG, which
// deletes every assert at the preprocessor -- a Release ctest run would compile
// and launch the tests but verify nothing (and code inside assert() is not even
// type-checked). Re-arm assert for test translation units regardless of build
// flags: the standard specifies that <cassert> redefines assert according to
// the NDEBUG state at each inclusion, so undefining NDEBUG here (before any
// use) keeps every later assert in the TU active. A -UNDEBUG compile option is
// NOT a substitute: Kokkos's nvcc_wrapper forwards -U as -Xcompiler while
// keeping -DNDEBUG an nvcc option, so the define wins back silently.
#undef NDEBUG

#include <cassert>
#include <cmath>

#include "otinum/otinum.hpp"

namespace oti_test {

constexpr double tol = 1e-11;

template <typename T>
void expect_near(T actual, T expected, double tolerance = tol)
{
    assert(std::abs(actual - expected) <= tolerance);
}

template <int M, int N, class Coeff>
void expect_all_near(oti::otinum<M, N, Coeff> const& actual,
                     oti::otinum<M, N, Coeff> const& expected,
                     double tolerance = tol)
{
    for (int i = 0; i < oti::otinum<M, N, Coeff>::ncoeffs; ++i) {
        expect_near(actual[i], expected[i], tolerance);
    }
}

} // namespace oti_test
