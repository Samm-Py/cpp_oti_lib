#pragma once

// Public mathematical functions for oti::otinum.
//
// This header provides <cmath>-style overloads such as exp, log, pow, sin, cos,
// and abs. Primitive analytic functions are lifted to OTI values through the
// scalar Taylor-composition helpers in taylor.hpp. Simpler derived functions
// such as sqrt, tan, sinh, and tanh are expressed in terms of those primitives
// and the arithmetic operators from core.hpp.

#include "otinum/taylor.hpp"

namespace oti {

// Public <cmath>-style overloads for otinum. Domain behavior follows the
// underlying scalar <cmath> calls on value.real().
// TODO: Decide whether the public API should document/recommend qualified
// oti:: calls only, or also explicitly support unqualified ADL-style calls.
template <int M, int N>
OTI_FUNCTION otinum<M, N> exp(otinum<M, N> const& value) noexcept
{
    OTI_PROFILE_COUNT(exp);
    // Use the same scalar Taylor-composition path as the other analytic
    // functions in this header.
    return detail::apply_scalar(detail::exp_coeffs<N>(value.real()), value);
}

template <int M, int N>
OTI_FUNCTION otinum<M, N> log(otinum<M, N> const& value) noexcept
{
    OTI_PROFILE_COUNT(log);
    // apply_scalar evaluates the Taylor expansion of log around value.real()
    // and substitutes the nilpotent part of value into that expansion.
    return detail::apply_scalar(detail::log_coeffs<N>(value.real()), value);
}

template <int M, int N>
OTI_FUNCTION otinum<M, N> log10(otinum<M, N> const& value) noexcept
{
    // Change of base keeps the implementation tied to the generic log path.
    return log(value) / detail::oti_log(10.0);
}

template <int M, int N>
OTI_FUNCTION otinum<M, N> logb(otinum<M, N> const& value, double base) noexcept
{
    // User-specified-base logarithm via log(value) / log(base).
    return log(value) / detail::oti_log(base);
}

template <int M, int N>
OTI_FUNCTION otinum<M, N> pow(otinum<M, N> const& value, double exponent) noexcept
{
    OTI_PROFILE_COUNT(pow);
    // The exponent is scalar, so use the direct Taylor coefficients of x^p
    // instead of composing exp(p * log(x)).
    return detail::apply_scalar(detail::pow_coeffs<N>(value.real(), exponent), value);
}

template <int M, int N>
OTI_FUNCTION otinum<M, N> pow(otinum<M, N> const& lhs, otinum<M, N> const& rhs) noexcept
{
    OTI_PROFILE_COUNT(pow);
    // a^b is defined through exp(b * log(a)); this inherits log(a)'s real-valued
    // domain requirement on lhs.real().
    return exp(rhs * log(lhs));
}

template <int M, int N>
OTI_FUNCTION otinum<M, N> sqrt(otinum<M, N> const& value) noexcept
{
    // sqrt(x) is x^(1/2), so the pow implementation owns the derivative logic.
    return pow(value, 0.5);
}

template <int M, int N>
OTI_FUNCTION otinum<M, N> cbrt(otinum<M, N> const& value) noexcept
{
    // cbrt(x) is x^(1/3), routed through the scalar-exponent pow path.
    return pow(value, 1.0 / 3.0);
}

template <int M, int N>
OTI_FUNCTION otinum<M, N> sin(otinum<M, N> const& value) noexcept
{
    OTI_PROFILE_COUNT(sin);
    // sin/cos use precomputed Taylor coefficients with the same generic
    // scalar-function machinery as log and scalar pow.
    return detail::apply_scalar(detail::sin_coeffs<N>(value.real()), value);
}

template <int M, int N>
OTI_FUNCTION otinum<M, N> cos(otinum<M, N> const& value) noexcept
{
    OTI_PROFILE_COUNT(cos);
    // See sin(): only the scalar Taylor coefficients differ.
    return detail::apply_scalar(detail::cos_coeffs<N>(value.real()), value);
}

template <int M, int N>
OTI_FUNCTION otinum<M, N> tan(otinum<M, N> const& value) noexcept
{
    OTI_PROFILE_COUNT(tan);
    // Division by cos(value) means tan is singular where cos(value.real()) == 0.
    return sin(value) / cos(value);
}

template <int M, int N>
OTI_FUNCTION otinum<M, N> sinh(otinum<M, N> const& value) noexcept
{
    OTI_PROFILE_COUNT(sinh);
    // Hyperbolic functions are expressed with exp to avoid separate recurrence
    // code paths.
    return (exp(value) - exp(-value)) / 2.0;
}

template <int M, int N>
OTI_FUNCTION otinum<M, N> cosh(otinum<M, N> const& value) noexcept
{
    OTI_PROFILE_COUNT(cosh);
    // cosh(x) = (exp(x) + exp(-x)) / 2.
    return (exp(value) + exp(-value)) / 2.0;
}

template <int M, int N>
OTI_FUNCTION otinum<M, N> tanh(otinum<M, N> const& value) noexcept
{
    OTI_PROFILE_COUNT(tanh);
    // For finite real x, cosh(x) is never zero, unlike the cos denominator in tan.
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
