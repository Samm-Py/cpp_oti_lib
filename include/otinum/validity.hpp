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

#include <cmath>
#include <utility>

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

// +infinity, host- AND device-safe. The INFINITY macro is a constant expression
// usable in device code, unlike std::numeric_limits<T>::infinity(), which is a
// host-only constexpr function under CUDA (rejected without --expt-relaxed-constexpr).
// Returned when a pure-axis term vanishes (model exact in that variable -> unbounded reach).
template <class Coeff>
OTI_CONSTEXPR_FUNCTION Coeff pos_inf() noexcept
{
    return static_cast<Coeff>(INFINITY);
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

// h^alpha for the compile-time coefficient index IDX, folded over the sparse
// entries of alpha (at most N of them). Every position/exponent below is a
// literal subscript into the static idx_to_sparse table, so this compiles to a
// straight-line product h[p0]^e0 * h[p1]^e1 * ... -- the same index-folding
// trick as otinum_mul_into. A runtime-indexed alpha_at() here instead forces
// the compiler to materialize the whole sparse table in (device local) memory
// per call: measured 2.4 KB of stack frame per thread at <6,3> double.
// Skipped zero-exponent positions contribute an exact *1, so the value is
// bit-identical to the dense product.
template <int M, int N, class Coeff, std::size_t IDX, std::size_t... T>
OTI_CONSTEXPR_FUNCTION Coeff monomial_at(oti::detail::array<Coeff, M> const& h,
                                         std::index_sequence<T...>) noexcept
{
    using tb = oti::detail::tables<M, N>;
    Coeff r = Coeff(1);
    (((static_cast<int>(T) < tb::idx_to_sparse[IDX].k)
          ? (void)(r *= pow_int(
                h[static_cast<std::size_t>(tb::idx_to_sparse[IDX].pos[T])],
                tb::idx_to_sparse[IDX].exp[T]))
          : (void)0),
     ...);
    return r;
}

// Sum of jet[idx] * h^alpha(idx) over every coefficient whose total order lies
// in [min_order, max_order]. Each coefficient's order is a compile-time
// literal, so the band test is one scalar compare per term; excluded terms are
// skipped outright (not accumulated as +0), preserving bit-exactness with a
// plain prefix loop over the graded layout.
template <int M, int N, class Coeff, std::size_t... IDX>
OTI_CONSTEXPR_FUNCTION Coeff band_sum(otinum<M, N, Coeff> const& jet,
                                      identity_t<oti::detail::array<Coeff, M>> const& h,
                                      int min_order, int max_order,
                                      std::index_sequence<IDX...>) noexcept
{
    using tb = oti::detail::tables<M, N>;
    Coeff acc = Coeff(0);
    (((tb::order_of[IDX] >= min_order && tb::order_of[IDX] <= max_order)
          ? (void)(acc += jet[static_cast<int>(IDX)] *
                          monomial_at<M, N, Coeff, IDX>(
                              h, std::make_index_sequence<
                                     oti::detail::sparse_index<N>::cap>{}))
          : (void)0),
     ...);
    return acc;
}

// The reduced monomial h^(alpha - e_p) where p is sparse entry T's position:
// like monomial_at, but entry T's exponent is lowered by one. Used by the
// error-gradient fold; all indices are again compile-time literals.
template <int M, int N, class Coeff, std::size_t IDX, std::size_t T, std::size_t... W>
OTI_CONSTEXPR_FUNCTION Coeff dmonomial_at(oti::detail::array<Coeff, M> const& h,
                                          std::index_sequence<W...>) noexcept
{
    using tb = oti::detail::tables<M, N>;
    Coeff r = Coeff(1);
    (((static_cast<int>(W) < tb::idx_to_sparse[IDX].k)
          ? (void)(r *= pow_int(
                h[static_cast<std::size_t>(tb::idx_to_sparse[IDX].pos[W])],
                tb::idx_to_sparse[IDX].exp[W] - (W == T ? 1 : 0)))
          : (void)0),
     ...);
    return r;
}

// One coefficient's contribution to the error gradient: for each nonzero
// exponent (p, e) of alpha(IDX), g[p] += jet[IDX] * e * h^(alpha - e_p).
// Folding over the sparse entries visits exactly the variables with a[i] > 0,
// in ascending position order -- the same accumulation order as a dense scan.
template <int M, int N, class Coeff, std::size_t IDX, std::size_t... T>
OTI_CONSTEXPR_FUNCTION void sensitivity_add(identity_t<oti::detail::array<Coeff, M>>& g,
                                            otinum<M, N, Coeff> const& jet,
                                            identity_t<oti::detail::array<Coeff, M>> const& h,
                                            std::index_sequence<T...>) noexcept
{
    using tb = oti::detail::tables<M, N>;
    (((static_cast<int>(T) < tb::idx_to_sparse[IDX].k)
          ? (void)(g[static_cast<std::size_t>(tb::idx_to_sparse[IDX].pos[T])] +=
                jet[static_cast<int>(IDX)] *
                static_cast<Coeff>(tb::idx_to_sparse[IDX].exp[T]) *
                dmonomial_at<M, N, Coeff, IDX, T>(
                    h, std::make_index_sequence<
                           oti::detail::sparse_index<N>::cap>{}))
          : (void)0),
     ...);
}

template <int M, int N, class Coeff, std::size_t... IDX>
OTI_CONSTEXPR_FUNCTION void sensitivity_fold(identity_t<oti::detail::array<Coeff, M>>& g,
                                             otinum<M, N, Coeff> const& jet,
                                             identity_t<oti::detail::array<Coeff, M>> const& h,
                                             int min_order,
                                             std::index_sequence<IDX...>) noexcept
{
    using tb = oti::detail::tables<M, N>;
    (((tb::order_of[IDX] >= min_order)
          ? (void)sensitivity_add<M, N, Coeff, IDX>(
                g, jet, h,
                std::make_index_sequence<oti::detail::sparse_index<N>::cap>{})
          : (void)0),
     ...);
}

// |E_i(r)|: magnitude of the pure-axis truncation error for a step r along
// variable i, summed over every stored order above the model,
// |sum_{d=model_order+1}^{N} c_{d e_i} r^d|. (Restricting h to one axis kills
// all mixed monomials, so this equals |truncation_error(jet, r*e_i)|.) Used by
// validity_radius to size the per-axis reach when more than one band is present.
template <int M, int N, class Coeff>
OTI_CONSTEXPR_FUNCTION Coeff pure_axis_error(otinum<M, N, Coeff> const& jet,
                                             int i, Coeff r, int model_order) noexcept
{
    Coeff acc = Coeff(0);
    for (int d = model_order + 1; d <= N; ++d) {
        oti::detail::alpha_t<M> a{};
        a[i] = d;
        acc += jet[oti::detail::rank<M, N>(a)] * pow_int(r, d);
    }
    return abs_scalar(acc);
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
    OTI_ASSERT(model_order >= 0 && model_order <= N);
    return detail::band_sum(jet, h, 0, model_order,
                            std::make_index_sequence<otinum<M, N, Coeff>::ncoeffs>{});
}

// Signed truncation error of the order-`model_order` surrogate at h: the sum of
// EVERY stored order above the model, sum_{model_order+1 <= |beta| <= N} c_beta
// h^beta. Requires model_order < N so at least one such band exists. For the
// default model_order = N-1 this is just the single top band; with spare
// computed orders (e.g. <M,3> certifying a linear model) it folds them all in
// for a tighter, less optimistic error estimate.
template <int M, int N, class Coeff>
OTI_CONSTEXPR_FUNCTION Coeff truncation_error(otinum<M, N, Coeff> const& jet,
                                              detail::identity_t<oti::detail::array<Coeff, M>> const& h,
                                              int model_order = N - 1) noexcept
{
    OTI_ASSERT(model_order >= 0 && model_order < N);
    // Every stored order above the model, through the top stored order N.
    return detail::band_sum(jet, h, model_order + 1, N,
                            std::make_index_sequence<otinum<M, N, Coeff>::ncoeffs>{});
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
// which the order-`model_order` model stays within tau. The pure-axis error
// |sum_{d=model_order+1}^{N} c_{d e_i} r^d| is matched to the budget tau|f|.
// With a single error band this is the closed form r_i = (tau|f|/|c_{d e_i}|)^{1/d};
// with several stored bands there is no closed form, so r_i is found by an
// allocation-free bracket-and-bisection on the pure-axis error (device-callable
// like the rest, consistent with truncation_error / is_trusted along the axis).
// A zero pure-axis error in a variable yields +inf.
//
// NOTE these are ellipsoid SEMI-AXES, not box sides: combine a multi-axis step
// with sum_i (h_i / r_i)^2 <= 1, NOT |h_i| <= r_i (the box corner overshoots).
template <int M, int N, class Coeff>
OTI_CONSTEXPR_FUNCTION oti::detail::array<Coeff, M> validity_radius(
    otinum<M, N, Coeff> const& jet, Coeff tau, int model_order = N - 1) noexcept
{
    OTI_ASSERT(model_order >= 0 && model_order < N);
    int const d0 = model_order + 1;
    Coeff const budget = tau * detail::abs_scalar(jet[0]);
    oti::detail::array<Coeff, M> r{};
    for (int i = 0; i < M; ++i) {
        // Seed the upper bracket from the leading pure term's closed form; fall
        // back to 1 when that coefficient vanishes.
        oti::detail::alpha_t<M> a0{};
        a0[i] = d0;
        Coeff const c0 = detail::abs_scalar(jet[oti::detail::rank<M, N>(a0)]);
        Coeff hi = c0 > Coeff(0)
                       ? static_cast<Coeff>(oti::detail::oti_pow<Coeff>(budget / c0, Coeff(1) / Coeff(d0)))
                       : Coeff(1);
        // Grow hi until the full pure-axis error reaches the budget; if it never
        // does, the model is exact enough in this variable -> reach is +inf.
        int guard = 0;
        while (detail::pure_axis_error<M, N>(jet, i, hi, model_order) < budget && guard < 64) {
            hi *= Coeff(2);
            ++guard;
        }
        if (detail::pure_axis_error<M, N>(jet, i, hi, model_order) < budget) {
            r[i] = detail::pos_inf<Coeff>();
            continue;
        }
        // Bisect [0, hi] for |E_i(r)| = budget (E_i monotone in magnitude near 0,
        // where the leading term dominates -> this is the first/inner crossing).
        Coeff lo = Coeff(0);
        for (int it = 0; it < 64; ++it) {
            Coeff const mid = Coeff(0.5) * (lo + hi);
            if (detail::pure_axis_error<M, N>(jet, i, mid, model_order) < budget)
                lo = mid;
            else
                hi = mid;
        }
        r[i] = Coeff(0.5) * (lo + hi);
    }
    return r;
}

// Coupling-aware per-variable blame: the gradient of the truncation error,
// g_i = dE/dh_i = sum_{model_order+1 <= |beta| <= N} c_beta * beta_i * h^{beta - e_i}
// over every stored order above the model. This is the steepest-descent
// direction for pulling h back under tolerance and distributes interaction
// (mixed-partial) terms across their variables for free.
template <int M, int N, class Coeff>
OTI_CONSTEXPR_FUNCTION oti::detail::array<Coeff, M> error_sensitivity(
    otinum<M, N, Coeff> const& jet, detail::identity_t<oti::detail::array<Coeff, M>> const& h,
    int model_order = N - 1) noexcept
{
    OTI_ASSERT(model_order >= 0 && model_order < N);
    oti::detail::array<Coeff, M> g{};
    for (int i = 0; i < M; ++i) g[i] = Coeff(0);
    // Every stored order above the model, through the top stored order N.
    detail::sensitivity_fold(g, jet, h, model_order + 1,
                             std::make_index_sequence<otinum<M, N, Coeff>::ncoeffs>{});
    return g;
}

} // namespace validity
} // namespace oti

#endif // OTINUM_VALIDITY_HPP
