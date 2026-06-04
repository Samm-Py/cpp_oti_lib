#pragma once

// Small constexpr combinatorial helpers.
//
// The OTI layout depends on binomial coefficients and factorials: binom() counts
// stored multi-indices, factorial() converts normalized Taylor coefficients back
// to ordinary derivatives, and factorials<N>() supplies scalar Taylor-series
// denominators. The functions are annotated through kokkos_compat.hpp so they
// can be used in host and Kokkos/CUDA device code.

#include "otinum/detail/kokkos_compat.hpp"

namespace oti::detail {

OTI_CONSTEXPR_FUNCTION int binom(int n, int k) noexcept
{
    if (k < 0 || n < 0 || k > n) {
        return 0;
    }
    if (k > n - k) {
        k = n - k;
    }

    int result = 1;
    for (int i = 1; i <= k; ++i) {
        result = (result * (n - k + i)) / i;
    }
    return result;
}

OTI_CONSTEXPR_FUNCTION double factorial(int n) noexcept
{
    double result = 1.0;
    for (int i = 2; i <= n; ++i) {
        result *= static_cast<double>(i);
    }
    return result;
}

template <int N>
OTI_CONSTEXPR_FUNCTION array<double, N + 1> factorials() noexcept
{
    array<double, N + 1> out{};
    out[0] = 1.0;
    for (int i = 1; i <= N; ++i) {
        out[i] = out[i - 1] * static_cast<double>(i);
    }
    return out;
}

} // namespace oti::detail
