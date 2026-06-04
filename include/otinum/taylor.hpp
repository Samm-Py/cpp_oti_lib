#pragma once

// Scalar-function Taylor composition helpers.
//
// Public functions such as oti::log(x) and oti::sin(x) reduce to this file's
// machinery. For an OTI value value = a + h, where a is the real coefficient and
// h has zero real part, apply_scalar() evaluates
//
//   f(a + h) = sum_k (f^(k)(a) / k!) h^k
//
// inside the truncated multi-index algebra. This header also provides the
// scalar Taylor coefficient generators for exp, log, pow, sin, and cos.

#include "otinum/core.hpp"
#include "otinum/detail/binom.hpp"

namespace oti::detail {

template <int M, int N>
OTI_CONSTEXPR_FUNCTION void add_scaled_orders(otinum<M, N>& out,
                                              otinum<M, N> const& value,
                                              double scale,
                                              int min_order) noexcept
{
    using tables = typename otinum<M, N>::table_type;
    // Add scale * value into out, skipping coefficients below min_order. In
    // scalar Taylor composition this is used for h^k, which cannot contribute
    // below total order k because h has zero real part.
    if (scale == 0.0 || min_order > N) {
        return;
    }

    if (min_order < 0) {
        min_order = 0;
    }

    int begin = tables::order_offset_value(min_order);
    for (int i = begin; i < otinum<M, N>::ncoeffs; ++i) {
        out[i] += scale * value[i];
    }
}

template <int M, int N>
OTI_CONSTEXPR_FUNCTION otinum<M, N> multiply_by_nilpotent(otinum<M, N> const& lhs,
                                                          otinum<M, N> const& nilpotent,
                                                          int min_output_order) noexcept
{
    using tables = typename otinum<M, N>::table_type;
    // Used by apply_scalar to advance h^(k-1) to h^k, where h is the nilpotent
    // part of value and h[0] == 0. Therefore h^k cannot contribute below total
    // order k, and product terms involving the right-hand real coefficient
    // nilpotent[0] vanish.
    otinum<M, N> out;

    if (min_output_order < 0) {
        min_output_order = 0;
    }
    if (min_output_order > N) {
        return out;
    }

    int first_output = tables::order_offset_value(min_output_order);
    for (int out_index = first_output; out_index < otinum<M, N>::ncoeffs; ++out_index) {
        double accum = 0.0;
        int begin = tables::product_offset_value(out_index);
        int end = tables::product_offset_value(out_index + 1);
        for (int p = begin; p < end; ++p) {
            auto const term = tables::product_term_by_output_value(p);
            if (term.rhs == 0) {
                continue;
            }
            accum += lhs[term.lhs] * nilpotent[term.rhs];
        }
        out[out_index] = accum;
    }

    return out;
}

// Compose a scalar function with an OTI value using its Taylor coefficients at
// value.real(). The nilpotent part h has finite order, so the series stops at N.
// Although this has the shape of a one-variable Taylor series in h, h is an OTI
// value. Powers of h are multiplied in the multi-index algebra, which produces
// the mixed-variable coefficients for general <M, N>.
template <int M, int N>
OTI_FUNCTION otinum<M, N> apply_scalar(array<double, N + 1> const& t,
                                       otinum<M, N> const& value) noexcept
{
    otinum<M, N> h = value;
    h[0] = 0.0;

    otinum<M, N> out(t[0]);
    if constexpr (N > 0) {
        otinum<M, N> hk = h;
        add_scaled_orders(out, hk, t[1], 1);

        for (int k = 2; k <= N; ++k) {
            hk = multiply_by_nilpotent(hk, h, k);
            add_scaled_orders(out, hk, t[static_cast<std::size_t>(k)], k);
        }
    }
    return out;
}

// Coefficients are f^(k)(x) / k!, ready for apply_scalar().
template <int N>
OTI_FUNCTION array<double, N + 1> exp_coeffs(double x) noexcept
{
    array<double, N + 1> out{};
    auto const facts = factorials<N>();
    double ex = oti_exp(x);
    for (int k = 0; k <= N; ++k) {
        out[static_cast<std::size_t>(k)] = ex / facts[static_cast<std::size_t>(k)];
    }
    return out;
}

template <int N>
OTI_FUNCTION array<double, N + 1> log_coeffs(double x) noexcept
{
    // Real-valued log coefficients require x > 0; invalid x follows scalar
    // floating-point behavior through oti_log() and divisions by x^k.
    array<double, N + 1> out{};
    out[0] = oti_log(x);
    double xpow = x;
    for (int k = 1; k <= N; ++k) {
        double sign = (k % 2 == 1) ? 1.0 : -1.0;
        out[static_cast<std::size_t>(k)] = sign / (static_cast<double>(k) * xpow);
        xpow *= x;
    }
    return out;
}

template <int N>
OTI_FUNCTION array<double, N + 1> pow_coeffs(double x, double p) noexcept
{
    // Domain follows scalar pow() and the requested derivatives. Non-integer or
    // negative powers can require x > 0 or x != 0.
    array<double, N + 1> out{};
    double binomial = 1.0;
    for (int k = 0; k <= N; ++k) {
        if (k > 0) {
            binomial *= (p - static_cast<double>(k - 1)) / static_cast<double>(k);
        }
        out[static_cast<std::size_t>(k)] = binomial * oti_pow(x, p - static_cast<double>(k));
    }
    return out;
}

template <int N>
OTI_FUNCTION array<double, N + 1> sin_coeffs(double x) noexcept
{
    // d^k/dx^k sin(x) equals sin(x + k*pi/2), then divided by k! for Taylor
    // coefficients.
    array<double, N + 1> out{};
    auto const facts = factorials<N>();
    for (int k = 0; k <= N; ++k) {
        out[static_cast<std::size_t>(k)] =
            oti_sin(x + static_cast<double>(k) * 1.57079632679489661923) /
            facts[static_cast<std::size_t>(k)];
    }
    return out;
}

template <int N>
OTI_FUNCTION array<double, N + 1> cos_coeffs(double x) noexcept
{
    // d^k/dx^k cos(x) equals cos(x + k*pi/2), then divided by k! for Taylor
    // coefficients.
    array<double, N + 1> out{};
    auto const facts = factorials<N>();
    for (int k = 0; k <= N; ++k) {
        out[static_cast<std::size_t>(k)] =
            oti_cos(x + static_cast<double>(k) * 1.57079632679489661923) /
            facts[static_cast<std::size_t>(k)];
    }
    return out;
}

} // namespace oti::detail
