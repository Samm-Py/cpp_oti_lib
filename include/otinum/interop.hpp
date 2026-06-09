#pragma once

// Standard-library interoperability for oti::otinum.
//
// This header makes otinum behave like a built-in floating-point type as far as
// the standard library and ordinary "scalar-shaped" C++ code are concerned, so
// that replacing `double` with `otinum` in an existing codebase needs as few
// edits as possible. It provides:
//
//   * std::numeric_limits<otinum<...>>          (limits/sentinels in generic code)
//   * operator<< / print_coeffs                 (streaming and inspection)
//   * <cmath>-style overloads not in functions.hpp:
//       floor, ceil, trunc, round, nearbyint, rint, fabs,
//       fmin, fmax, copysign, signbit, isnan, isinf, isfinite,
//       hypot, fmod, remainder, exp2, log2, expm1, log1p
//
// The analytic elementary functions (exp, log, pow, sin, ...) live in
// functions.hpp; this header is the non-analytic / library-glue companion.
//
// All overloads are found by argument-dependent lookup, matching how existing
// code calls unqualified `floor(x)` / `fmin(a, b)` after `#include <cmath>`.

#include <cmath>
#include <limits>
#include <ostream>

#include "otinum/core.hpp"
#include "otinum/functions.hpp"

// ---------------------------------------------------------------------------
// std::numeric_limits specialization
//
// Forwards every query to the backing scalar's std::numeric_limits, lifting the
// value-returning members into an otinum (real part = the scalar limit, all
// derivatives = 0, since these are constants of the number system). This lets
// generic numerical code -- thresholds, sentinels, NaN/inf guards written as
// std::numeric_limits<T>::max()/epsilon()/quiet_NaN() -- compile and behave
// exactly as it does for the underlying scalar once T becomes an otinum.
// ---------------------------------------------------------------------------

namespace std {

template <int M, int N, class Coeff>
class numeric_limits<oti::otinum<M, N, Coeff>> {
    using T = oti::otinum<M, N, Coeff>;
    using B = std::numeric_limits<Coeff>;

public:
    static constexpr bool is_specialized = true;

    // Value queries: lifted to an otinum constant (derivatives are zero). Marked
    // with the library's constexpr annotation so they stay usable in constant
    // expressions in both host and Kokkos builds, matching otinum's own ctor.
    static OTI_CONSTEXPR_FUNCTION T min() noexcept { return T(B::min()); }
    static OTI_CONSTEXPR_FUNCTION T max() noexcept { return T(B::max()); }
    static OTI_CONSTEXPR_FUNCTION T lowest() noexcept { return T(B::lowest()); }
    static OTI_CONSTEXPR_FUNCTION T epsilon() noexcept { return T(B::epsilon()); }
    static OTI_CONSTEXPR_FUNCTION T round_error() noexcept { return T(B::round_error()); }
    static OTI_CONSTEXPR_FUNCTION T infinity() noexcept { return T(B::infinity()); }
    static OTI_CONSTEXPR_FUNCTION T quiet_NaN() noexcept { return T(B::quiet_NaN()); }
    static OTI_CONSTEXPR_FUNCTION T signaling_NaN() noexcept { return T(B::signaling_NaN()); }
    static OTI_CONSTEXPR_FUNCTION T denorm_min() noexcept { return T(B::denorm_min()); }

    // Trait queries: forwarded unchanged from the scalar, except is_iec559 --
    // an otinum is not a bit-for-bit IEEE-754 float, so we report false to stop
    // callers assuming IEEE bit-layout tricks are valid on it.
    static constexpr int digits = B::digits;
    static constexpr int digits10 = B::digits10;
    static constexpr int max_digits10 = B::max_digits10;
    static constexpr bool is_signed = B::is_signed;
    static constexpr bool is_integer = B::is_integer;
    static constexpr bool is_exact = B::is_exact;
    static constexpr int radix = B::radix;
    static constexpr int min_exponent = B::min_exponent;
    static constexpr int max_exponent = B::max_exponent;
    static constexpr int min_exponent10 = B::min_exponent10;
    static constexpr int max_exponent10 = B::max_exponent10;
    static constexpr bool has_infinity = B::has_infinity;
    static constexpr bool has_quiet_NaN = B::has_quiet_NaN;
    static constexpr bool has_signaling_NaN = B::has_signaling_NaN;
    static constexpr float_denorm_style has_denorm = B::has_denorm;
    static constexpr bool has_denorm_loss = B::has_denorm_loss;
    static constexpr float_round_style round_style = B::round_style;
    static constexpr bool is_iec559 = false;
    static constexpr bool is_bounded = B::is_bounded;
    static constexpr bool is_modulo = B::is_modulo;
    static constexpr bool traps = B::traps;
    static constexpr bool tinyness_before = B::tinyness_before;
};

} // namespace std

// ---------------------------------------------------------------------------
// Streaming and inspection
// ---------------------------------------------------------------------------

namespace oti {

// Stream insertion. Prints ONLY the real part, forwarding to the scalar's
// operator<< so the stream's precision/width/flags are honored. This keeps
// existing textual output (logs, CSV writers) byte-identical to a double build
// after a double -> otinum swap; use print_coeffs() to see the derivatives.
template <int M, int N, class Coeff>
std::ostream& operator<<(std::ostream& os, otinum<M, N, Coeff> const& v)
{
    return os << v.real();
}

// Full inspection: real part followed by the stored derivative coefficients in
// flat multi-index order, e.g. "2 | [1, 0, 0.5]". For N == 0 (no derivatives)
// the bracketed list is empty.
template <int M, int N, class Coeff>
std::ostream& print_coeffs(std::ostream& os, otinum<M, N, Coeff> const& v)
{
    os << v.real() << " | [";
    for (int i = 1; i < otinum<M, N, Coeff>::ncoeffs; ++i) {
        if (i > 1) {
            os << ", ";
        }
        os << v[i];
    }
    return os << "]";
}

} // namespace oti

namespace oti {

// ---------------------------------------------------------------------------
// Rounding / step functions.
//
// These are piecewise constant: their derivative is 0 everywhere except at the
// integer jumps, where it is undefined. Following the universal AD convention,
// the result keeps the rounded real value and reports zero derivatives (the
// otinum scalar constructor zeroes every derivative coefficient).
// ---------------------------------------------------------------------------

template <int M, int N, class Coeff>
OTI_FUNCTION otinum<M, N, Coeff> floor(otinum<M, N, Coeff> const& v) noexcept
{
    return otinum<M, N, Coeff>(detail::oti_floor(v.real()));
}

template <int M, int N, class Coeff>
OTI_FUNCTION otinum<M, N, Coeff> ceil(otinum<M, N, Coeff> const& v) noexcept
{
    return otinum<M, N, Coeff>(detail::oti_ceil(v.real()));
}

template <int M, int N, class Coeff>
OTI_FUNCTION otinum<M, N, Coeff> trunc(otinum<M, N, Coeff> const& v) noexcept
{
    return otinum<M, N, Coeff>(detail::oti_trunc(v.real()));
}

template <int M, int N, class Coeff>
OTI_FUNCTION otinum<M, N, Coeff> round(otinum<M, N, Coeff> const& v) noexcept
{
    return otinum<M, N, Coeff>(detail::oti_round(v.real()));
}

template <int M, int N, class Coeff>
OTI_FUNCTION otinum<M, N, Coeff> nearbyint(otinum<M, N, Coeff> const& v) noexcept
{
    return otinum<M, N, Coeff>(detail::oti_nearbyint(v.real()));
}

template <int M, int N, class Coeff>
OTI_FUNCTION otinum<M, N, Coeff> rint(otinum<M, N, Coeff> const& v) noexcept
{
    return otinum<M, N, Coeff>(detail::oti_rint(v.real()));
}

// fabs is the C-style spelling of abs for floating types; forward to the
// existing abs(), which already handles the kink at zero correctly.
template <int M, int N, class Coeff>
OTI_FUNCTION otinum<M, N, Coeff> fabs(otinum<M, N, Coeff> const& v) noexcept
{
    return abs(v);
}

// ---------------------------------------------------------------------------
// Classification predicates. Decided on the real part only, consistent with
// the comparison operators, and returning bool exactly like their scalar
// counterparts so guard code (if (!isfinite(x)) ...) compiles unchanged.
// NaN/finiteness are tested with device-safe primitives (r != r, oti_isfinite)
// so the predicates remain callable from Kokkos device code.
// ---------------------------------------------------------------------------

template <int M, int N, class Coeff>
OTI_FUNCTION bool signbit(otinum<M, N, Coeff> const& v) noexcept
{
    return detail::oti_signbit(v.real());
}

template <int M, int N, class Coeff>
OTI_FUNCTION bool isnan(otinum<M, N, Coeff> const& v) noexcept
{
    Coeff const r = v.real();
    return r != r;
}

template <int M, int N, class Coeff>
OTI_FUNCTION bool isfinite(otinum<M, N, Coeff> const& v) noexcept
{
    return detail::oti_isfinite(v.real());
}

template <int M, int N, class Coeff>
OTI_FUNCTION bool isinf(otinum<M, N, Coeff> const& v) noexcept
{
    Coeff const r = v.real();
    return !detail::oti_isfinite(r) && !(r != r);
}

// ---------------------------------------------------------------------------
// Selection. fmin/fmax pick a branch by real part and return that operand
// WHOLE, preserving its derivatives -- the correct AD behavior, since the
// function locally equals the selected operand. NaN operands are ignored
// (matching std::fmin/fmax); exact ties return the first argument.
// ---------------------------------------------------------------------------

template <int M, int N, class Coeff>
OTI_FUNCTION otinum<M, N, Coeff> fmax(otinum<M, N, Coeff> const& a,
                                     otinum<M, N, Coeff> const& b) noexcept
{
    Coeff const ar = a.real();
    Coeff const br = b.real();
    if (ar != ar) {
        return b;
    }
    if (br != br) {
        return a;
    }
    return (ar < br) ? b : a;
}

template <int M, int N, class Coeff>
OTI_FUNCTION otinum<M, N, Coeff> fmin(otinum<M, N, Coeff> const& a,
                                     otinum<M, N, Coeff> const& b) noexcept
{
    Coeff const ar = a.real();
    Coeff const br = b.real();
    if (ar != ar) {
        return b;
    }
    if (br != br) {
        return a;
    }
    return (br < ar) ? b : a;
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_FUNCTION otinum<M, N, Coeff> fmax(otinum<M, N, Coeff> const& a, Scalar b) noexcept
{
    return fmax(a, otinum<M, N, Coeff>(static_cast<Coeff>(b)));
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_FUNCTION otinum<M, N, Coeff> fmax(Scalar a, otinum<M, N, Coeff> const& b) noexcept
{
    return fmax(otinum<M, N, Coeff>(static_cast<Coeff>(a)), b);
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_FUNCTION otinum<M, N, Coeff> fmin(otinum<M, N, Coeff> const& a, Scalar b) noexcept
{
    return fmin(a, otinum<M, N, Coeff>(static_cast<Coeff>(b)));
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_FUNCTION otinum<M, N, Coeff> fmin(Scalar a, otinum<M, N, Coeff> const& b) noexcept
{
    return fmin(otinum<M, N, Coeff>(static_cast<Coeff>(a)), b);
}

// copysign(x, y): magnitude of x with the sign of y's real part. Implemented as
// +/- abs(x), so x's derivatives flow through abs (including its kink handling);
// the result is locally constant in y, so y contributes no derivative.
template <int M, int N, class Coeff>
OTI_FUNCTION otinum<M, N, Coeff> copysign(otinum<M, N, Coeff> const& x,
                                          otinum<M, N, Coeff> const& y) noexcept
{
    return detail::oti_signbit(y.real()) ? -abs(x) : abs(x);
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_FUNCTION otinum<M, N, Coeff> copysign(otinum<M, N, Coeff> const& x, Scalar y) noexcept
{
    return detail::oti_signbit(static_cast<Coeff>(y)) ? -abs(x) : abs(x);
}

// ---------------------------------------------------------------------------
// hypot. sqrt(x^2 + y^2 [+ z^2]); differentiable except at the origin. Unlike
// std::hypot this naive form can overflow for extreme magnitudes, but the
// derivatives remain correct. Scalar and 3-argument overloads included.
// ---------------------------------------------------------------------------

template <int M, int N, class Coeff>
OTI_FUNCTION otinum<M, N, Coeff> hypot(otinum<M, N, Coeff> const& x,
                                      otinum<M, N, Coeff> const& y) noexcept
{
    return sqrt(x * x + y * y);
}

template <int M, int N, class Coeff>
OTI_FUNCTION otinum<M, N, Coeff> hypot(otinum<M, N, Coeff> const& x,
                                      otinum<M, N, Coeff> const& y,
                                      otinum<M, N, Coeff> const& z) noexcept
{
    return sqrt(x * x + y * y + z * z);
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_FUNCTION otinum<M, N, Coeff> hypot(otinum<M, N, Coeff> const& x, Scalar y) noexcept
{
    Coeff const yc = static_cast<Coeff>(y);
    return sqrt(x * x + yc * yc);
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_FUNCTION otinum<M, N, Coeff> hypot(Scalar x, otinum<M, N, Coeff> const& y) noexcept
{
    Coeff const xc = static_cast<Coeff>(x);
    return sqrt(xc * xc + y * y);
}

// ---------------------------------------------------------------------------
// fmod / remainder. Both are x - n*y with the integer quotient n frozen from
// the real parts: trunc(x/y) for fmod, round-half-to-even for remainder. This
// gives the correct value and the correct a.e. derivatives (d/dx = 1,
// d/dy = -n) between the jumps.
// ---------------------------------------------------------------------------

template <int M, int N, class Coeff>
OTI_FUNCTION otinum<M, N, Coeff> fmod(otinum<M, N, Coeff> const& x,
                                     otinum<M, N, Coeff> const& y) noexcept
{
    Coeff const n = detail::oti_trunc(x.real() / y.real());
    return x - n * y;
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_FUNCTION otinum<M, N, Coeff> fmod(otinum<M, N, Coeff> const& x, Scalar y) noexcept
{
    Coeff const yc = static_cast<Coeff>(y);
    Coeff const n = detail::oti_trunc(x.real() / yc);
    return x - n * yc;
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_FUNCTION otinum<M, N, Coeff> fmod(Scalar x, otinum<M, N, Coeff> const& y) noexcept
{
    Coeff const xc = static_cast<Coeff>(x);
    Coeff const n = detail::oti_trunc(xc / y.real());
    return xc - n * y;
}

template <int M, int N, class Coeff>
OTI_FUNCTION otinum<M, N, Coeff> remainder(otinum<M, N, Coeff> const& x,
                                          otinum<M, N, Coeff> const& y) noexcept
{
    Coeff const n = detail::oti_nearbyint(x.real() / y.real());
    return x - n * y;
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_FUNCTION otinum<M, N, Coeff> remainder(otinum<M, N, Coeff> const& x, Scalar y) noexcept
{
    Coeff const yc = static_cast<Coeff>(y);
    Coeff const n = detail::oti_nearbyint(x.real() / yc);
    return x - n * yc;
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_FUNCTION otinum<M, N, Coeff> remainder(Scalar x, otinum<M, N, Coeff> const& y) noexcept
{
    Coeff const xc = static_cast<Coeff>(x);
    Coeff const n = detail::oti_nearbyint(xc / y.real());
    return xc - n * y;
}

// ---------------------------------------------------------------------------
// Base-2 exp/log via the natural-log primitives (derivatives come for free).
// ---------------------------------------------------------------------------

template <int M, int N, class Coeff>
OTI_FUNCTION otinum<M, N, Coeff> exp2(otinum<M, N, Coeff> const& v) noexcept
{
    // 2^x = exp(x * ln 2).
    return exp(v * static_cast<Coeff>(0.69314718055994530942));
}

template <int M, int N, class Coeff>
OTI_FUNCTION otinum<M, N, Coeff> log2(otinum<M, N, Coeff> const& v) noexcept
{
    return log(v) / static_cast<Coeff>(0.69314718055994530942);
}

// ---------------------------------------------------------------------------
// expm1 / log1p. The derivatives equal those of exp(x) and log(1+x), so the
// composition supplies the derivative coefficients; only the real part is
// replaced with the dedicated accurate scalar call to avoid cancellation near
// zero (the entire reason these functions exist).
// ---------------------------------------------------------------------------

template <int M, int N, class Coeff>
OTI_FUNCTION otinum<M, N, Coeff> expm1(otinum<M, N, Coeff> const& v) noexcept
{
    otinum<M, N, Coeff> out = exp(v);
    out[0] = detail::oti_expm1(v.real());
    return out;
}

template <int M, int N, class Coeff>
OTI_FUNCTION otinum<M, N, Coeff> log1p(otinum<M, N, Coeff> const& v) noexcept
{
    otinum<M, N, Coeff> out = log(otinum<M, N, Coeff>(Coeff(1)) + v);
    out[0] = detail::oti_log1p(v.real());
    return out;
}

} // namespace oti
