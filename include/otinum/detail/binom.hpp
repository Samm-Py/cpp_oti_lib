#pragma once

#include <array>

namespace oti::detail {

constexpr int binom(int n, int k) noexcept
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

constexpr double factorial(int n) noexcept
{
    double result = 1.0;
    for (int i = 2; i <= n; ++i) {
        result *= static_cast<double>(i);
    }
    return result;
}

template <int N>
constexpr std::array<double, N + 1> factorials() noexcept
{
    std::array<double, N + 1> out{};
    out[0] = 1.0;
    for (int i = 1; i <= N; ++i) {
        out[i] = out[i - 1] * static_cast<double>(i);
    }
    return out;
}

} // namespace oti::detail
