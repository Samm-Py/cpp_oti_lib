#pragma once

#include "otinum/taylor.hpp"

namespace oti {

// Public <cmath>-style overloads for otinum. Domain behavior follows the
// underlying scalar <cmath> calls on value.real().
template <int M, int N>
OTI_FUNCTION otinum<M, N> exp(otinum<M, N> const& value) noexcept
{
    OTI_PROFILE_COUNT(exp);
    return detail::exp_recurrence(value);
}

template <int M, int N>
OTI_FUNCTION otinum<M, N> log(otinum<M, N> const& value) noexcept
{
    OTI_PROFILE_COUNT(log);
    return detail::apply_scalar(detail::log_coeffs<N>(value.real()), value);
}

template <int M, int N>
OTI_FUNCTION otinum<M, N> log10(otinum<M, N> const& value) noexcept
{
    return log(value) / detail::oti_log(10.0);
}

template <int M, int N>
OTI_FUNCTION otinum<M, N> logb(otinum<M, N> const& value, double base) noexcept
{
    return log(value) / detail::oti_log(base);
}

template <int M, int N>
OTI_FUNCTION otinum<M, N> pow(otinum<M, N> const& value, double exponent) noexcept
{
    OTI_PROFILE_COUNT(pow);
    return detail::apply_scalar(detail::pow_coeffs<N>(value.real(), exponent), value);
}

template <int M, int N>
OTI_FUNCTION otinum<M, N> pow(otinum<M, N> const& lhs, otinum<M, N> const& rhs) noexcept
{
    OTI_PROFILE_COUNT(pow);
    // a^b is defined through exp(b * log(a)), matching the scalar identity.
    return exp(rhs * log(lhs));
}

template <int M, int N>
OTI_FUNCTION otinum<M, N> sqrt(otinum<M, N> const& value) noexcept
{
    return pow(value, 0.5);
}

template <int M, int N>
OTI_FUNCTION otinum<M, N> cbrt(otinum<M, N> const& value) noexcept
{
    return pow(value, 1.0 / 3.0);
}

template <int M, int N>
OTI_FUNCTION otinum<M, N> sin(otinum<M, N> const& value) noexcept
{
    OTI_PROFILE_COUNT(sin);
    return detail::apply_scalar(detail::sin_coeffs<N>(value.real()), value);
}

template <int M, int N>
OTI_FUNCTION otinum<M, N> cos(otinum<M, N> const& value) noexcept
{
    OTI_PROFILE_COUNT(cos);
    return detail::apply_scalar(detail::cos_coeffs<N>(value.real()), value);
}

template <int M, int N>
OTI_FUNCTION otinum<M, N> tan(otinum<M, N> const& value) noexcept
{
    OTI_PROFILE_COUNT(tan);
    return sin(value) / cos(value);
}

template <int M, int N>
OTI_FUNCTION otinum<M, N> sinh(otinum<M, N> const& value) noexcept
{
    OTI_PROFILE_COUNT(sinh);
    return (exp(value) - exp(-value)) / 2.0;
}

template <int M, int N>
OTI_FUNCTION otinum<M, N> cosh(otinum<M, N> const& value) noexcept
{
    OTI_PROFILE_COUNT(cosh);
    return (exp(value) + exp(-value)) / 2.0;
}

template <int M, int N>
OTI_FUNCTION otinum<M, N> tanh(otinum<M, N> const& value) noexcept
{
    OTI_PROFILE_COUNT(tanh);
    return sinh(value) / cosh(value);
}

template <int M, int N>
OTI_FUNCTION otinum<M, N> abs(otinum<M, N> const& value) noexcept
{
    OTI_PROFILE_COUNT(abs);
    // abs is not differentiable at zero. For zero real part this returns value,
    // matching the branch below rather than defining a mathematical derivative.
    if (value.real() < 0.0) {
        return -value;
    }
    return value;
}

} // namespace oti
