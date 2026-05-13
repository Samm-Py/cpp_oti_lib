#pragma once

#include <cmath>

#include "otinum/taylor.hpp"

namespace oti {

template <int M, int N>
otinum<M, N> exp(otinum<M, N> const& value) noexcept
{
    return detail::apply_scalar(detail::exp_coeffs<N>(value.real()), value);
}

template <int M, int N>
otinum<M, N> log(otinum<M, N> const& value) noexcept
{
    return detail::apply_scalar(detail::log_coeffs<N>(value.real()), value);
}

template <int M, int N>
otinum<M, N> log10(otinum<M, N> const& value) noexcept
{
    return log(value) / std::log(10.0);
}

template <int M, int N>
otinum<M, N> logb(otinum<M, N> const& value, double base) noexcept
{
    return log(value) / std::log(base);
}

template <int M, int N>
otinum<M, N> pow(otinum<M, N> const& value, double exponent) noexcept
{
    return detail::apply_scalar(detail::pow_coeffs<N>(value.real(), exponent), value);
}

template <int M, int N>
otinum<M, N> pow(otinum<M, N> const& lhs, otinum<M, N> const& rhs) noexcept
{
    return exp(rhs * log(lhs));
}

template <int M, int N>
otinum<M, N> sqrt(otinum<M, N> const& value) noexcept
{
    return pow(value, 0.5);
}

template <int M, int N>
otinum<M, N> cbrt(otinum<M, N> const& value) noexcept
{
    return pow(value, 1.0 / 3.0);
}

template <int M, int N>
otinum<M, N> sin(otinum<M, N> const& value) noexcept
{
    return detail::apply_scalar(detail::sin_coeffs<N>(value.real()), value);
}

template <int M, int N>
otinum<M, N> cos(otinum<M, N> const& value) noexcept
{
    return detail::apply_scalar(detail::cos_coeffs<N>(value.real()), value);
}

template <int M, int N>
otinum<M, N> tan(otinum<M, N> const& value) noexcept
{
    return sin(value) / cos(value);
}

template <int M, int N>
otinum<M, N> sinh(otinum<M, N> const& value) noexcept
{
    return (exp(value) - exp(-value)) / 2.0;
}

template <int M, int N>
otinum<M, N> cosh(otinum<M, N> const& value) noexcept
{
    return (exp(value) + exp(-value)) / 2.0;
}

template <int M, int N>
otinum<M, N> tanh(otinum<M, N> const& value) noexcept
{
    return sinh(value) / cosh(value);
}

template <int M, int N>
otinum<M, N> abs(otinum<M, N> const& value) noexcept
{
    if (value.real() < 0.0) {
        return -value;
    }
    return value;
}

} // namespace oti
