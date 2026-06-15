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

#include <utility>

#include "otinum/core.hpp"
#include "otinum/detail/binom.hpp"

namespace oti::detail {

template <int M, int N, class Coeff>
OTI_CONSTEXPR_FUNCTION void add_scaled_orders(otinum<M, N, Coeff>& out,
                                              otinum<M, N, Coeff> const& value,
                                              Coeff scale,
                                              int min_order) noexcept
{
    using tables = typename otinum<M, N, Coeff>::table_type;
    // Add scale * value into out, skipping coefficients below min_order. In
    // scalar Taylor composition this is used for h^k, which cannot contribute
    // below total order k because h has zero real part.
    if (scale == Coeff(0) || min_order > N) {
        return;
    }

    if (min_order < 0) {
        min_order = 0;
    }

    int begin = tables::order_offset_value(min_order);
    for (int i = begin; i < otinum<M, N, Coeff>::ncoeffs; ++i) {
        out[i] += scale * value[i];
    }
}

// One output coefficient of lhs * nilpotent, summing product terms (lhs, rhs)
// -> OUT but skipping rhs == 0 (the nilpotent's real coefficient is zero). The
// indices come from the static by-output table at compile-time pack positions,
// so they fold to literals -- straight-line FMAs, no table materialization.
template <int M, int N, class Coeff, std::size_t OUT, std::size_t... Q>
OTI_CONSTEXPR_FUNCTION Coeff nilpotent_output(otinum<M, N, Coeff> const& lhs,
                                              otinum<M, N, Coeff> const& nilpotent,
                                              std::index_sequence<Q...>) noexcept
{
    using tb = tables<M, N>;
    constexpr int begin = tb::product_offset[OUT];
    Coeff acc = Coeff(0);
    ((acc += (tb::product_terms_by_output[begin + Q].rhs == 0)
                 ? Coeff(0)
                 : lhs[tb::product_terms_by_output[begin + Q].lhs] *
                       nilpotent[tb::product_terms_by_output[begin + Q].rhs]),
     ...);
    return acc;
}

// Compute every output coefficient whose total order is at least
// min_output_order; lower outputs stay zero (h^k has no terms below order k).
// The order of each output is a compile-time literal, so the guard is just a
// runtime compare against min_output_order with no table lookup.
template <int M, int N, class Coeff, std::size_t... OUT>
OTI_CONSTEXPR_FUNCTION void nilpotent_mul_into(otinum<M, N, Coeff>& out,
                                               otinum<M, N, Coeff> const& lhs,
                                               otinum<M, N, Coeff> const& nilpotent,
                                               int min_output_order,
                                               std::index_sequence<OUT...>) noexcept
{
    using tb = tables<M, N>;
    ((out[static_cast<int>(OUT)] =
          (tb::order_of[OUT] >= min_output_order)
              ? nilpotent_output<M, N, Coeff, OUT>(
                    lhs, nilpotent,
                    std::make_index_sequence<tb::product_offset[OUT + 1] -
                                             tb::product_offset[OUT]>{})
              : Coeff(0)),
     ...);
}

template <int M, int N, class Coeff>
OTI_CONSTEXPR_FUNCTION otinum<M, N, Coeff>
multiply_by_nilpotent(otinum<M, N, Coeff> const& lhs,
                      otinum<M, N, Coeff> const& nilpotent,
                      int min_output_order) noexcept
{
    // Used by apply_scalar to advance h^(k-1) to h^k, where h is the nilpotent
    // part of value and h[0] == 0. Therefore h^k cannot contribute below total
    // order k, and product terms involving the right-hand real coefficient
    // nilpotent[0] vanish.
    otinum<M, N, Coeff> out;

    if (min_output_order < 0) {
        min_output_order = 0;
    }
    if (min_output_order > N) {
        return out;
    }

#if OTI_BENCHMARK_ARITHMETIC_PATH == 0
    using tb = tables<M, N>;
    for (int i = 0; i < otinum<M, N, Coeff>::ncoeffs; ++i) {
        auto const alpha = tb::alpha_at(i);
        int const order_i = tb::order_of_value(i);
        for (int j = 1; j < otinum<M, N, Coeff>::ncoeffs; ++j) {
            int const output_order = order_i + tb::order_of_value(j);
            if (output_order < min_output_order || output_order > N) {
                continue;
            }
            auto gamma = alpha;
            auto const beta = tb::alpha_at(j);
            for (int m = 0; m < M; ++m) {
                gamma[static_cast<std::size_t>(m)] += beta[static_cast<std::size_t>(m)];
            }
            int const k = rank<M, N>(gamma);
            out[k] += lhs[i] * nilpotent[j];
        }
    }
#elif OTI_BENCHMARK_ARITHMETIC_PATH == 1
    using tb = tables<M, N>;
    int const first_output = tb::order_offset_value(min_output_order);
    for (int out_index = first_output;
         out_index < otinum<M, N, Coeff>::ncoeffs; ++out_index) {
        Coeff accum = Coeff(0);
        int const begin = tb::product_offset_value(out_index);
        int const end = tb::product_offset_value(out_index + 1);
        for (int p = begin; p < end; ++p) {
            auto const term = tb::product_term_by_output_value(p);
            if (term.rhs != 0) {
                accum += lhs[term.lhs] * nilpotent[term.rhs];
            }
        }
        out[out_index] = accum;
    }
#else
    nilpotent_mul_into(out, lhs, nilpotent, min_output_order,
                       std::make_index_sequence<otinum<M, N, Coeff>::ncoeffs>{});
#endif
    return out;
}

// Compose a scalar function with an OTI value using its Taylor coefficients at
// value.real(). The nilpotent part h has finite order, so the series stops at N.
// Although this has the shape of a one-variable Taylor series in h, h is an OTI
// value. Powers of h are multiplied in the multi-index algebra, which produces
// the mixed-variable coefficients for general <M, N>.
template <int M, int N, class Coeff>
OTI_FUNCTION otinum<M, N, Coeff> apply_scalar(array<Coeff, N + 1> const& t,
                                              otinum<M, N, Coeff> const& value) noexcept
{
    // The truncated Taylor jet is valid only where the function is analytic. If
    // any scalar coefficient is non-finite, the expansion point is singular: an
    // undefined value (log(-1)), a pole (log(0)), or a vertical tangent (sqrt(0),
    // cbrt(0)). There is no valid finite jet there, so report the scalar value
    // f(x0) and NaN for every derivative -- one consistent signal across all
    // singularities, instead of the mix of inf and nan that arithmetic with
    // infinities would otherwise produce (e.g. the inf * 0 cross terms that turn
    // the true infinite second derivative of log at 0 into a NaN).
    bool singular = false;
    for (int k = 0; k <= N; ++k) {
        if (!oti_isfinite(t[static_cast<std::size_t>(k)])) {
            singular = true;
        }
    }
    if (singular) {
        otinum<M, N, Coeff> out;
        out[0] = t[0];
        for (int i = 1; i < otinum<M, N, Coeff>::ncoeffs; ++i) {
            out[i] = static_cast<Coeff>(NAN);
        }
        return out;
    }

    otinum<M, N, Coeff> h = value;
    h[0] = Coeff(0);

    otinum<M, N, Coeff> out(t[0]);
    if constexpr (N > 0) {
        otinum<M, N, Coeff> hk = h;
        add_scaled_orders(out, hk, t[1], 1);

        for (int k = 2; k <= N; ++k) {
            hk = multiply_by_nilpotent(hk, h, k);
            add_scaled_orders(out, hk, t[static_cast<std::size_t>(k)], k);
        }
    }
    return out;
}

// Coefficients are f^(k)(x) / k!, ready for apply_scalar().
template <int N, class Coeff>
OTI_FUNCTION array<Coeff, N + 1> exp_coeffs(Coeff x) noexcept
{
    array<Coeff, N + 1> out{};
    auto const facts = factorials<N>();
    Coeff ex = oti_exp(x);
    for (int k = 0; k <= N; ++k) {
        out[static_cast<std::size_t>(k)] =
            ex / static_cast<Coeff>(facts[static_cast<std::size_t>(k)]);
    }
    return out;
}

template <int N, class Coeff>
OTI_FUNCTION array<Coeff, N + 1> log_coeffs(Coeff x) noexcept
{
    // Real-valued log coefficients require x > 0; invalid x follows scalar
    // floating-point behavior through oti_log() and divisions by x^k.
    array<Coeff, N + 1> out{};
    out[0] = oti_log(x);
    Coeff xpow = x;
    for (int k = 1; k <= N; ++k) {
        Coeff sign = (k % 2 == 1) ? Coeff(1) : Coeff(-1);
        out[static_cast<std::size_t>(k)] = sign / (static_cast<Coeff>(k) * xpow);
        xpow *= x;
    }
    return out;
}

template <int N, class Coeff>
OTI_FUNCTION array<Coeff, N + 1> pow_coeffs(Coeff x, Coeff p) noexcept
{
    // Domain follows scalar pow(). A negative base with a non-integer exponent
    // returns NaN by design, matching std::pow: a double exponent cannot encode
    // whether the intended rational power has an odd denominator (real, e.g.
    // x^(1/3)) or an even one (complex, e.g. x^(1/2)). cbrt() is the dedicated
    // real cube root for negative bases.
    array<Coeff, N + 1> out{};
    Coeff binomial = Coeff(1);
    for (int k = 0; k <= N; ++k) {
        if (k > 0) {
            binomial *= (p - static_cast<Coeff>(k - 1)) / static_cast<Coeff>(k);
        }
        // For an integer exponent the generalized binomial becomes exactly zero
        // once k exceeds p, so the term vanishes. Short-circuit it: otherwise an
        // x == 0 base gives 0 * oti_pow(0, negative) == 0 * Inf == NaN instead of
        // the correct 0 (e.g. pow(x, 2) at x == 0 in an order >= 3 algebra).
        out[static_cast<std::size_t>(k)] =
            binomial == Coeff(0)
                ? Coeff(0)
                : binomial * oti_pow(x, p - static_cast<Coeff>(k));
    }
    return out;
}

template <int N, class Coeff>
OTI_FUNCTION array<Coeff, N + 1> cbrt_coeffs(Coeff x) noexcept
{
    // Real cube root, valid for x < 0 (unlike pow(x, 1/3), which would route
    // through std::pow(negative, non-integer) and return NaN). The k-th Taylor
    // coefficient is C(1/3, k) * x^(1/3 - k). For x < 0 that power is evaluated
    // on the real branch as cbrt(x) / x^k; for x >= 0 oti_pow is used directly,
    // so x == 0 yields the correct infinite-slope derivatives.
    array<Coeff, N + 1> out{};
    Coeff const cr = oti_cbrt(x);
    Coeff const third = Coeff(1) / Coeff(3);
    out[0] = cr;
    Coeff binomial = Coeff(1);
    Coeff xpow = Coeff(1);
    for (int k = 1; k <= N; ++k) {
        binomial *= (third - static_cast<Coeff>(k - 1)) / static_cast<Coeff>(k);
        xpow *= x;
        Coeff const power =
            (x < Coeff(0)) ? cr / xpow : oti_pow(x, third - static_cast<Coeff>(k));
        out[static_cast<std::size_t>(k)] = binomial * power;
    }
    return out;
}

template <int N, class Coeff>
OTI_FUNCTION array<Coeff, N + 1> sin_coeffs(Coeff x) noexcept
{
    // d^k/dx^k sin(x) equals sin(x + k*pi/2), then divided by k! for Taylor
    // coefficients.
    array<Coeff, N + 1> out{};
    auto const facts = factorials<N>();
    for (int k = 0; k <= N; ++k) {
        out[static_cast<std::size_t>(k)] =
            oti_sin(x + static_cast<Coeff>(k) * static_cast<Coeff>(1.57079632679489661923)) /
            static_cast<Coeff>(facts[static_cast<std::size_t>(k)]);
    }
    return out;
}

template <int N, class Coeff>
OTI_FUNCTION array<Coeff, N + 1> cos_coeffs(Coeff x) noexcept
{
    // d^k/dx^k cos(x) equals cos(x + k*pi/2), then divided by k! for Taylor
    // coefficients.
    array<Coeff, N + 1> out{};
    auto const facts = factorials<N>();
    for (int k = 0; k <= N; ++k) {
        out[static_cast<std::size_t>(k)] =
            oti_cos(x + static_cast<Coeff>(k) * static_cast<Coeff>(1.57079632679489661923)) /
            static_cast<Coeff>(facts[static_cast<std::size_t>(k)]);
    }
    return out;
}

// atan coefficients at x: a_0 = atan(x); for k >= 1, a_k = g_{k-1} / k, where g
// are the Taylor coefficients of atan'(x) = 1/(1+x^2). Those follow from series
// division by d = 1 + x^2 (only d0, d1, d2 are nonzero). 1 + x^2 >= 1, so atan
// is entire on the reals -- no singular case here.
template <int N, class Coeff>
OTI_FUNCTION array<Coeff, N + 1> atan_coeffs(Coeff x) noexcept
{
    array<Coeff, N + 1> out{};
    out[0] = oti_atan(x);
    if constexpr (N > 0) {
        Coeff const d0 = Coeff(1) + x * x;
        Coeff const d1 = Coeff(2) * x;
        Coeff const d2 = Coeff(1);
        array<Coeff, N + 1> g{};
        for (int j = 0; j <= N - 1; ++j) {
            Coeff acc = (j == 0) ? Coeff(1) : Coeff(0);
            if (j >= 1) {
                acc -= d1 * g[static_cast<std::size_t>(j - 1)];
            }
            if (j >= 2) {
                acc -= d2 * g[static_cast<std::size_t>(j - 2)];
            }
            g[static_cast<std::size_t>(j)] = acc / d0;
            out[static_cast<std::size_t>(j + 1)] =
                g[static_cast<std::size_t>(j)] / static_cast<Coeff>(j + 1);
        }
    }
    return out;
}

// asin coefficients at x: a_0 = asin(x); for k >= 1, a_k = w_{k-1} / k, where w
// are the Taylor coefficients of asin'(x) = (1 - x^2)^(-1/2). With s = 1 - x^2
// (only s0, s1, s2 nonzero) and p = -1/2, w = s^p follows the standard
// power-of-a-series recurrence. For |x| >= 1 the leading coefficient is
// non-finite, and apply_scalar turns that into value + NaN-derivatives (the
// vertical-tangent / out-of-domain convention shared with sqrt and log).
template <int N, class Coeff>
OTI_FUNCTION array<Coeff, N + 1> asin_coeffs(Coeff x) noexcept
{
    array<Coeff, N + 1> out{};
    out[0] = oti_asin(x);
    if constexpr (N > 0) {
        Coeff const p = Coeff(-0.5);
        Coeff const s0 = Coeff(1) - x * x;
        Coeff const s1 = Coeff(-2) * x;
        Coeff const s2 = Coeff(-1);
        array<Coeff, N + 1> w{};
        w[0] = oti_pow(s0, p);
        for (int k = 1; k <= N - 1; ++k) {
            Coeff acc = (p * Coeff(1) - static_cast<Coeff>(k - 1)) * s1 *
                        w[static_cast<std::size_t>(k - 1)];
            if (k >= 2) {
                acc += (p * Coeff(2) - static_cast<Coeff>(k - 2)) * s2 *
                       w[static_cast<std::size_t>(k - 2)];
            }
            w[static_cast<std::size_t>(k)] = acc / (static_cast<Coeff>(k) * s0);
        }
        for (int k = 1; k <= N; ++k) {
            out[static_cast<std::size_t>(k)] =
                w[static_cast<std::size_t>(k - 1)] / static_cast<Coeff>(k);
        }
    }
    return out;
}

// acos coefficients at x: acos = pi/2 - asin, so the real part is acos(x) and
// every derivative coefficient is the negation of asin's.
template <int N, class Coeff>
OTI_FUNCTION array<Coeff, N + 1> acos_coeffs(Coeff x) noexcept
{
    array<Coeff, N + 1> out = asin_coeffs<N, Coeff>(x);
    out[0] = oti_acos(x);
    for (int k = 1; k <= N; ++k) {
        out[static_cast<std::size_t>(k)] = -out[static_cast<std::size_t>(k)];
    }
    return out;
}

} // namespace oti::detail
