#ifndef OTINUM_VALIDITY_HPP
#define OTINUM_VALIDITY_HPP

// Surrogate evaluation and truncation-error / validity analysis for otinum jets.
//
// An otinum<M, P, Coeff> built by seeding the M perturbed parameters as OTI
// variables is a local truncated-Taylor surrogate of some quantity f in
// PARAMETER space, around the point where it was computed. Given a candidate
// step h = (Δparam_0, ..., Δparam_{M-1}) you can:
//
//   * predict          -- evaluate the surrogate at h without re-running anything
//   * check trust      -- is the prediction within a relative tolerance tau?
//   * size the region  -- how far along each axis is the model good (the reach)
//   * attribute blame  -- which parameter drives the error when it is not trusted
//
// The model order / computed order contract (THE thing to get right):
//   You CERTIFY an order-`model_order` surrogate. Its leading truncation error is
//   the order-(model_order+1) term, so the jet must be computed at order
//   P >= model_order+1 for that term to exist in storage. By default model_order
//   = P-1, i.e. you run the simulation at ONE order above the model you trust and
//   the jet's top order is the dropped-term error estimate.
//
//     otinum<3,2> default -> model_order 1: certified LINEAR surrogate, quadratic
//                            terms spent as the trust budget.
//     otinum<3,3> default -> model_order 2: certified QUADRATIC surrogate, cubic
//                            terms as the error.
//
//   evaluate() and the trust functions SHARE model_order, so "what you predict
//   with" and "what you certify" never describe different models. evaluate() may
//   also be called with model_order = P to use the full jet for a more accurate
//   but UNcertified prediction (no higher term to bound it).
//
// Everything here is OTI_CONSTEXPR_FUNCTION (host + device, allocation-free), so
// it runs inside an ensemble/control kernel on the live jet, or in a post-solve
// Kokkos parallel_for over a stored field -- same primitives either way.

#include <limits>

#include "otinum/core.hpp"

namespace oti {
namespace validity {

namespace detail {

// Puts a type in a non-deduced context so template args (M, Coeff) are deduced
// only from the otinum argument, never from the step `h` -- otherwise deducing
// the int M from std::array's size_t extent fails. (std::type_identity is C++20.)
template <class T>
struct identity {
    using type = T;
};
template <class T>
using identity_t = typename identity<T>::type;

template <class Coeff>
OTI_CONSTEXPR_FUNCTION Coeff abs_scalar(Coeff x) noexcept
{
    return x < Coeff(0) ? -x : x;
}

// base^e for a small non-negative integer exponent (the multi-index entries),
// by repeated multiply -- exact for integer powers and cheaper than pow().
template <class Coeff>
OTI_CONSTEXPR_FUNCTION Coeff pow_int(Coeff base, int e) noexcept
{
    Coeff r = Coeff(1);
    for (int k = 0; k < e; ++k) r *= base;
    return r;
}

// h^alpha = prod_i h[i]^alpha[i].
template <int M, class Coeff>
OTI_CONSTEXPR_FUNCTION Coeff monomial(oti::detail::array<Coeff, M> const& h,
                                      oti::detail::alpha_t<M> const& alpha) noexcept
{
    Coeff r = Coeff(1);
    for (int i = 0; i < M; ++i) r *= pow_int(h[i], alpha[i]);
    return r;
}

} // namespace detail

// Surrogate prediction: f(x0 + h) ~= sum_{|beta| <= model_order} c_beta h^beta.
// The stored coefficients ARE the normalized Taylor coefficients, so no
// factorials enter. model_order defaults to P-1 (the certified model); pass P for
// the full, more accurate, uncertified prediction.
template <int M, int N, class Coeff>
OTI_CONSTEXPR_FUNCTION Coeff evaluate(otinum<M, N, Coeff> const& jet,
                                      detail::identity_t<oti::detail::array<Coeff, M>> const& h,
                                      int model_order = N - 1) noexcept
{
    using table = oti::detail::tables<M, N>;
    OTI_ASSERT(model_order >= 0 && model_order <= N);
    int const end = table::order_offset_value(model_order + 1);  // exclusive
    Coeff acc = Coeff(0);
    for (int idx = 0; idx < end; ++idx)
        acc += jet[idx] * detail::monomial<M>(h, table::alpha_at(idx));
    return acc;
}

// Signed leading truncation error of the order-`model_order` surrogate at h:
// sum_{|beta| = model_order+1} c_beta h^beta. Requires model_order < N so the
// (model_order+1) band exists.
template <int M, int N, class Coeff>
OTI_CONSTEXPR_FUNCTION Coeff truncation_error(otinum<M, N, Coeff> const& jet,
                                              detail::identity_t<oti::detail::array<Coeff, M>> const& h,
                                              int model_order = N - 1) noexcept
{
    using table = oti::detail::tables<M, N>;
    OTI_ASSERT(model_order >= 0 && model_order < N);
    int const begin = table::order_offset_value(model_order + 1);
    int const end = table::order_offset_value(model_order + 2);
    Coeff acc = Coeff(0);
    for (int idx = begin; idx < end; ++idx)
        acc += jet[idx] * detail::monomial<M>(h, table::alpha_at(idx));
    return acc;
}

// Trust check: is the order-`model_order` prediction at h within relative
// tolerance tau of the value? |truncation_error| <= tau * |f|, f = jet real part.
template <int M, int N, class Coeff>
OTI_CONSTEXPR_FUNCTION bool is_trusted(otinum<M, N, Coeff> const& jet,
                                       detail::identity_t<oti::detail::array<Coeff, M>> const& h, Coeff tau,
                                       int model_order = N - 1) noexcept
{
    Coeff const budget = tau * detail::abs_scalar(jet[0]);
    return detail::abs_scalar(truncation_error(jet, h, model_order)) <= budget;
}

// Per-variable reach: the largest single-axis step r_i along parameter i for
// which the order-`model_order` model stays within tau, from the pure
// order-(model_order+1) term:  |c_{(d) e_i}| r_i^d = tau |f|, d = model_order+1.
// A zero pure term (model exact in that variable at this order) yields +inf.
//
// NOTE these are ellipsoid SEMI-AXES, not box sides: combine a multi-axis step
// with sum_i (h_i / r_i)^2 <= 1, NOT |h_i| <= r_i (the box corner overshoots).
template <int M, int N, class Coeff>
OTI_CONSTEXPR_FUNCTION oti::detail::array<Coeff, M> validity_radius(
    otinum<M, N, Coeff> const& jet, Coeff tau, int model_order = N - 1) noexcept
{
    OTI_ASSERT(model_order >= 0 && model_order < N);
    int const d = model_order + 1;
    Coeff const budget = tau * detail::abs_scalar(jet[0]);
    oti::detail::array<Coeff, M> r{};
    for (int i = 0; i < M; ++i) {
        oti::detail::alpha_t<M> a{};
        a[i] = d;
        Coeff const c = detail::abs_scalar(jet[oti::detail::rank<M, N>(a)]);
        r[i] = c > Coeff(0)
                   ? static_cast<Coeff>(oti::detail::oti_pow<Coeff>(budget / c, Coeff(1) / Coeff(d)))
                   : std::numeric_limits<Coeff>::infinity();
    }
    return r;
}

// Coupling-aware per-variable blame: the gradient of the truncation error,
// g_i = dE/dh_i = sum_{|beta| = model_order+1} c_beta * beta_i * h^{beta - e_i}.
// This is the steepest-descent direction for pulling h back under tolerance and
// distributes interaction (mixed-partial) terms across their variables for free.
template <int M, int N, class Coeff>
OTI_CONSTEXPR_FUNCTION oti::detail::array<Coeff, M> error_sensitivity(
    otinum<M, N, Coeff> const& jet, detail::identity_t<oti::detail::array<Coeff, M>> const& h,
    int model_order = N - 1) noexcept
{
    using table = oti::detail::tables<M, N>;
    OTI_ASSERT(model_order >= 0 && model_order < N);
    int const begin = table::order_offset_value(model_order + 1);
    int const end = table::order_offset_value(model_order + 2);
    oti::detail::array<Coeff, M> g{};
    for (int i = 0; i < M; ++i) g[i] = Coeff(0);
    for (int idx = begin; idx < end; ++idx) {
        oti::detail::alpha_t<M> a = table::alpha_at(idx);
        Coeff const c = jet[idx];
        for (int i = 0; i < M; ++i) {
            if (a[i] > 0) {
                oti::detail::alpha_t<M> b = a;
                b[i] -= 1;
                g[i] += c * Coeff(a[i]) * detail::monomial<M>(h, b);
            }
        }
    }
    return g;
}

} // namespace validity
} // namespace oti

#endif // OTINUM_VALIDITY_HPP
