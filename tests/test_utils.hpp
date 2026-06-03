#pragma once

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

template <int M, int N>
void expect_all_near(oti::otinum<M, N> const& actual,
                     oti::otinum<M, N> const& expected,
                     double tolerance = tol)
{
    for (int i = 0; i < oti::otinum<M, N>::ncoeffs; ++i) {
        expect_near(actual[i], expected[i], tolerance);
    }
}

} // namespace oti_test
