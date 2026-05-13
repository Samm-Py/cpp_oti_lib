#pragma once

#include <array>
#include <cmath>

#include "otinum/core.hpp"
#include "otinum/detail/binom.hpp"

namespace oti::detail {

template <int M, int N>
otinum<M, N> apply_scalar(std::array<double, N + 1> const& t,
                          otinum<M, N> const& value) noexcept
{
    otinum<M, N> h = value;
    h[0] = 0.0;

    otinum<M, N> out(t[0]);
    otinum<M, N> hk(1.0);
    for (int k = 1; k <= N; ++k) {
        hk = hk * h;
        out += t[static_cast<std::size_t>(k)] * hk;
    }
    return out;
}

template <int N>
std::array<double, N + 1> exp_coeffs(double x) noexcept
{
    std::array<double, N + 1> out{};
    auto const facts = factorials<N>();
    double ex = std::exp(x);
    for (int k = 0; k <= N; ++k) {
        out[static_cast<std::size_t>(k)] = ex / facts[static_cast<std::size_t>(k)];
    }
    return out;
}

template <int N>
std::array<double, N + 1> log_coeffs(double x) noexcept
{
    std::array<double, N + 1> out{};
    out[0] = std::log(x);
    double xpow = x;
    for (int k = 1; k <= N; ++k) {
        double sign = (k % 2 == 1) ? 1.0 : -1.0;
        out[static_cast<std::size_t>(k)] = sign / (static_cast<double>(k) * xpow);
        xpow *= x;
    }
    return out;
}

template <int N>
std::array<double, N + 1> pow_coeffs(double x, double p) noexcept
{
    std::array<double, N + 1> out{};
    double binomial = 1.0;
    for (int k = 0; k <= N; ++k) {
        if (k > 0) {
            binomial *= (p - static_cast<double>(k - 1)) / static_cast<double>(k);
        }
        out[static_cast<std::size_t>(k)] = binomial * std::pow(x, p - static_cast<double>(k));
    }
    return out;
}

template <int N>
std::array<double, N + 1> sin_coeffs(double x) noexcept
{
    std::array<double, N + 1> out{};
    auto const facts = factorials<N>();
    for (int k = 0; k <= N; ++k) {
        out[static_cast<std::size_t>(k)] =
            std::sin(x + static_cast<double>(k) * 1.57079632679489661923) /
            facts[static_cast<std::size_t>(k)];
    }
    return out;
}

template <int N>
std::array<double, N + 1> cos_coeffs(double x) noexcept
{
    std::array<double, N + 1> out{};
    auto const facts = factorials<N>();
    for (int k = 0; k <= N; ++k) {
        out[static_cast<std::size_t>(k)] =
            std::cos(x + static_cast<double>(k) * 1.57079632679489661923) /
            facts[static_cast<std::size_t>(k)];
    }
    return out;
}

} // namespace oti::detail
