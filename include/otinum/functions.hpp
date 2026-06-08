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
template <int M, int N, class Coeff>
OTI_FUNCTION otinum<M, N, Coeff> exp(otinum<M, N, Coeff> const& value) noexcept
{
    OTI_PROFILE_COUNT(exp);
    // Use the same scalar Taylor-composition path as the other analytic
    // functions in this header.
    return detail::apply_scalar(detail::exp_coeffs<N>(value.real()), value);
}

template <int M, int N, class Coeff>
OTI_FUNCTION otinum<M, N, Coeff> log(otinum<M, N, Coeff> const& value) noexcept
{
    OTI_PROFILE_COUNT(log);
    // apply_scalar evaluates the Taylor expansion of log around value.real()
    // and substitutes the nilpotent part of value into that expansion.
    return detail::apply_scalar(detail::log_coeffs<N>(value.real()), value);
}

template <int M, int N, class Coeff>
OTI_FUNCTION otinum<M, N, Coeff> log10(otinum<M, N, Coeff> const& value) noexcept
{
    // Change of base keeps the implementation tied to the generic log path.
    return log(value) / detail::oti_log(Coeff(10));
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_FUNCTION otinum<M, N, Coeff> logb(otinum<M, N, Coeff> const& value, Scalar base) noexcept
{
    // User-specified-base logarithm via log(value) / log(base).
    return log(value) / detail::oti_log(static_cast<Coeff>(base));
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_FUNCTION otinum<M, N, Coeff> pow(otinum<M, N, Coeff> const& value, Scalar exponent) noexcept
{
    OTI_PROFILE_COUNT(pow);
    // The exponent is scalar, so use the direct Taylor coefficients of x^p
    // instead of composing exp(p * log(x)).
    return detail::apply_scalar(detail::pow_coeffs<N>(value.real(), static_cast<Coeff>(exponent)),
                                value);
}

template <int M, int N, class Coeff>
OTI_FUNCTION otinum<M, N, Coeff> pow(otinum<M, N, Coeff> const& lhs,
                                     otinum<M, N, Coeff> const& rhs) noexcept
{
    OTI_PROFILE_COUNT(pow);
    // a^b is defined through exp(b * log(a)); this inherits log(a)'s real-valued
    // domain requirement on lhs.real().
    return exp(rhs * log(lhs));
}

template <int M, int N, class Coeff>
OTI_FUNCTION otinum<M, N, Coeff> sqrt(otinum<M, N, Coeff> const& value) noexcept
{
    // sqrt(x) is x^(1/2), so the pow implementation owns the derivative logic.
    return pow(value, Coeff(0.5));
}

template <int M, int N, class Coeff>
OTI_FUNCTION otinum<M, N, Coeff> cbrt(otinum<M, N, Coeff> const& value) noexcept
{
    // cbrt(x) is x^(1/3), routed through the scalar-exponent pow path.
    return pow(value, Coeff(1) / Coeff(3));
}

template <int M, int N, class Coeff>
OTI_FUNCTION otinum<M, N, Coeff> sin(otinum<M, N, Coeff> const& value) noexcept
{
    OTI_PROFILE_COUNT(sin);
    // sin/cos use precomputed Taylor coefficients with the same generic
    // scalar-function machinery as log and scalar pow.
    return detail::apply_scalar(detail::sin_coeffs<N>(value.real()), value);
}

template <int M, int N, class Coeff>
OTI_FUNCTION otinum<M, N, Coeff> cos(otinum<M, N, Coeff> const& value) noexcept
{
    OTI_PROFILE_COUNT(cos);
    // See sin(): only the scalar Taylor coefficients differ.
    return detail::apply_scalar(detail::cos_coeffs<N>(value.real()), value);
}

template <int M, int N, class Coeff>
OTI_FUNCTION otinum<M, N, Coeff> tan(otinum<M, N, Coeff> const& value) noexcept
{
    OTI_PROFILE_COUNT(tan);
    // Division by cos(value) means tan is singular where cos(value.real()) == 0.
    return sin(value) / cos(value);
}

template <int M, int N, class Coeff>
OTI_FUNCTION otinum<M, N, Coeff> sinh(otinum<M, N, Coeff> const& value) noexcept
{
    OTI_PROFILE_COUNT(sinh);
    // Hyperbolic functions are expressed with exp to avoid separate recurrence
    // code paths.
    return (exp(value) - exp(-value)) / Coeff(2);
}

template <int M, int N, class Coeff>
OTI_FUNCTION otinum<M, N, Coeff> cosh(otinum<M, N, Coeff> const& value) noexcept
{
    OTI_PROFILE_COUNT(cosh);
    // cosh(x) = (exp(x) + exp(-x)) / 2.
    return (exp(value) + exp(-value)) / Coeff(2);
}

template <int M, int N, class Coeff>
OTI_FUNCTION otinum<M, N, Coeff> tanh(otinum<M, N, Coeff> const& value) noexcept
{
    OTI_PROFILE_COUNT(tanh);
    // For finite real x, cosh(x) is never zero, unlike the cos denominator in tan.
    return sinh(value) / cosh(value);
}

template <int M, int N, class Coeff>
OTI_FUNCTION otinum<M, N, Coeff> abs(otinum<M, N, Coeff> const& value) noexcept
{
    OTI_PROFILE_COUNT(abs);
    // Away from zero abs is locally linear: for real() != 0,
    //   abs(a + h) = sgn(a) * (a + h),
    // since every derivative of abs above first order vanishes there. So the
    // result is just value scaled by the sign of its real part.
    Coeff const r = value.real();
    if (r > Coeff(0)) {
        return value;
    }
    if (r < Coeff(0)) {
        return -value;
    }
    if (r != r) {
        // NaN real part: propagate it unchanged rather than treat it as a kink.
        return value;
    }
    // real() == 0 exactly: |0| = 0 is well defined, but abs is not
    // differentiable here, and the correct local behavior cannot be recovered
    // from the real part alone -- abs(x) (a genuine kink) and abs(x^2) (smooth,
    // tangent to zero) both arrive with real() == 0 yet differ. Rather than
    // guess a subgradient, signal non-differentiability with NaN derivatives. A
    // value that is exactly zero has no perturbation direction and is returned
    // unchanged.
    constexpr int n = otinum<M, N, Coeff>::ncoeffs;
    for (int i = 1; i < n; ++i) {
        if (value[i] != Coeff(0)) {
            otinum<M, N, Coeff> out; // out[0] == 0 == |0|
            for (int j = 1; j < n; ++j) {
                out[j] = static_cast<Coeff>(NAN);
            }
            return out;
        }
    }
    return value;
}

} // namespace oti
