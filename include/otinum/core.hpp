#pragma once

// Core OTI value type and arithmetic.
//
// This header defines oti::otinum<M, N, Coeff>, a fixed-size truncated Taylor
// algebra with M infinitesimal directions, maximum total order N, and selectable
// floating-point coefficient storage. The class owns the coefficient storage,
// scalar/variable construction, coefficient accessors, and arithmetic operators.
// Coefficients are stored in the flat multi-index layout defined by
// detail/multi_index.hpp, and arithmetic uses those compile-time tables to
// implement truncated polynomial algebra.

#include <cstddef>
#include <initializer_list>
#include <type_traits>
#include <vector>

#include "otinum/detail/multi_index.hpp"
#include "otinum/profile.hpp"

namespace oti {

// Sparse multi-index helpers: specify only the nonzero (variable, order) pairs,
// e.g. sparse({{0, 2}, {1, 1}}) for alpha = (2, 1, 0, ...). Backed by
// std::vector, so the coeff()/set_coeff()/partial()/set_partial() overloads that
// take a sparse_alpha are host-only -- unlike the alpha_type (array) overloads,
// which are device-callable.
struct sparse_index {
    int variable;
    int order;
};

struct sparse_alpha {
    std::vector<sparse_index> terms;
};

inline sparse_alpha sparse(std::initializer_list<sparse_index> terms)
{
    return sparse_alpha{terms};
}

// Static OTI number with M infinitesimal variables, truncated at total order N.
//
// Coefficients are normalized Taylor coefficients:
//   c[alpha] = (1 / alpha!) * partial^alpha f
// Use partial(alpha) to recover the ordinary derivative value.
template <class Coeff, class Scalar>
using scalar_enable_t =
    std::enable_if_t<std::is_arithmetic<Scalar>::value && std::is_convertible<Scalar, Coeff>::value,
                     int>;

template <int M, int N, class Coeff = double>
class otinum {
    static_assert(std::is_floating_point<Coeff>::value,
                  "otinum coefficient type must be float, double, or another floating-point type");

public:
    using table_type = detail::tables<M, N>;
    using alpha_type = detail::alpha_t<M>;
    using coeff_type = Coeff;

    static constexpr int nvars = M;
    static constexpr int order = N;
    // Number of stored coefficients is the count of multi-indices alpha with
    // |alpha| <= N, i.e. binomial(M + N, N).
    static constexpr int ncoeffs = table_type::ncoeffs;

    // c_ has a brace initializer, so the empty constructor still creates the
    // zero element. Keeping the body explicit avoids NVCC warnings on annotated
    // defaulted constructors.
    OTI_CONSTEXPR_FUNCTION otinum() noexcept {}

    // Lift a scalar into the OTI algebra. All derivative coefficients are zero.
    // The c_ member initializer zeros every coefficient before c_[0] is assigned.
    OTI_CONSTEXPR_FUNCTION otinum(Coeff real)
    {
        c_[0] = real;
    }

    // Create value + e_i, where e_i is the first-order nilpotent for variable i.
    static OTI_CONSTEXPR_FUNCTION otinum variable(int i, Coeff value = Coeff(0))
    {
        otinum out(value);
        OTI_ASSERT(i >= 0 && i < M);
        if constexpr (N > 0) {
            // For N == 0 the algebra stores only the real coefficient. Otherwise
            // seed the first-order coefficient for epsilon_i with derivative 1.
            alpha_type alpha{};
            alpha[static_cast<std::size_t>(i)] = 1;
            out[detail::rank<M, N>(alpha)] = Coeff(1);
        }
        return out;
    }

    // Construct directly from the library's flat graded multi-index layout.
    // This is mainly for tests, serialization, and advanced callers that already
    // know the rank ordering used by detail::tables<M, N>.
    static OTI_CONSTEXPR_FUNCTION otinum from_coeffs(detail::array<Coeff, ncoeffs> const& coeffs)
    {
        otinum out;
        out.c_ = coeffs;
        return out;
    }

    OTI_CONSTEXPR_FUNCTION Coeff real() const noexcept
    {
        return c_[0];
    }

    // Raw normalized coefficient access by flat multi-index rank. These accessors
    // do not bounds-check; callers that start from alpha should prefer coeff() or
    // partial(), which go through detail::rank().
    OTI_CONSTEXPR_FUNCTION Coeff operator[](int flat_index) const noexcept
    {
        return c_[static_cast<std::size_t>(flat_index)];
    }

    OTI_CONSTEXPR_FUNCTION Coeff& operator[](int flat_index) noexcept
    {
        return c_[static_cast<std::size_t>(flat_index)];
    }

    // Return the normalized Taylor coefficient for alpha, or zero if alpha is
    // outside this otinum's configured total order.
    OTI_CONSTEXPR_FUNCTION Coeff coeff(alpha_type const& alpha) const noexcept
    {
        int idx = detail::rank<M, N>(alpha);
        return idx < 0 ? Coeff(0) : c_[static_cast<std::size_t>(idx)];
    }

    Coeff coeff(sparse_alpha const& alpha) const noexcept
    {
        int idx = rank_sparse(alpha);
        return idx < 0 ? Coeff(0) : c_[static_cast<std::size_t>(idx)];
    }

    // Set the normalized Taylor coefficient for alpha. If alpha is outside this
    // otinum's configured total order, the request is ignored.
    OTI_CONSTEXPR_FUNCTION void set_coeff(alpha_type const& alpha, Coeff value) noexcept
    {
        int idx = detail::rank<M, N>(alpha);
        if (idx >= 0) {
            c_[static_cast<std::size_t>(idx)] = value;
        }
    }

    void set_coeff(sparse_alpha const& alpha, Coeff value) noexcept
    {
        int idx = rank_sparse(alpha);
        if (idx >= 0) {
            c_[static_cast<std::size_t>(idx)] = value;
        }
    }

    // Return the ordinary partial derivative value alpha! * c[alpha], or zero if
    // alpha is outside this otinum's configured total order.
    OTI_CONSTEXPR_FUNCTION Coeff partial(alpha_type const& alpha) const noexcept
    {
        int idx = detail::rank<M, N>(alpha);
        if (idx < 0) {
            return Coeff(0);
        }
        return c_[static_cast<std::size_t>(idx)] *
               table_type::template factorial_alpha_as<Coeff>(idx);
    }

    Coeff partial(sparse_alpha const& alpha) const noexcept
    {
        int idx = rank_sparse(alpha);
        if (idx < 0) {
            return Coeff(0);
        }
        return c_[static_cast<std::size_t>(idx)] *
               table_type::template factorial_alpha_as<Coeff>(idx);
    }

    // Set the ordinary partial derivative value for alpha. The stored normalized
    // Taylor coefficient is value / alpha!. If alpha is outside this otinum's
    // configured total order, the request is ignored.
    OTI_CONSTEXPR_FUNCTION void set_partial(alpha_type const& alpha, Coeff value) noexcept
    {
        int idx = detail::rank<M, N>(alpha);
        if (idx >= 0) {
            c_[static_cast<std::size_t>(idx)] =
                value / table_type::template factorial_alpha_as<Coeff>(idx);
        }
    }

    void set_partial(sparse_alpha const& alpha, Coeff value) noexcept
    {
        int idx = rank_sparse(alpha);
        if (idx >= 0) {
            c_[static_cast<std::size_t>(idx)] =
                value / table_type::template factorial_alpha_as<Coeff>(idx);
        }
    }

    OTI_CONSTEXPR_FUNCTION detail::array<Coeff, ncoeffs> const& data() const noexcept
    {
        return c_;
    }

    OTI_CONSTEXPR otinum& operator+=(otinum const& rhs) noexcept
    {
        OTI_PROFILE_COUNT(add);
        OTI_PROFILE_COUNT(add_oti);
        for (int i = 0; i < ncoeffs; ++i) {
            c_[static_cast<std::size_t>(i)] += rhs.c_[static_cast<std::size_t>(i)];
        }
        return *this;
    }

    OTI_CONSTEXPR otinum& operator-=(otinum const& rhs) noexcept
    {
        OTI_PROFILE_COUNT(sub);
        OTI_PROFILE_COUNT(sub_oti);
        for (int i = 0; i < ncoeffs; ++i) {
            c_[static_cast<std::size_t>(i)] -= rhs.c_[static_cast<std::size_t>(i)];
        }
        return *this;
    }

    OTI_CONSTEXPR otinum& operator*=(otinum const& rhs) noexcept
    {
        *this = (*this) * rhs;
        return *this;
    }

    otinum& operator/=(otinum const& rhs) noexcept
    {
        *this = (*this) / rhs;
        return *this;
    }

    template <class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
    OTI_CONSTEXPR otinum& operator+=(Scalar rhs) noexcept
    {
        OTI_PROFILE_COUNT(add);
        OTI_PROFILE_COUNT(add_scalar);
        c_[0] += static_cast<Coeff>(rhs);
        return *this;
    }

    template <class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
    OTI_CONSTEXPR otinum& operator-=(Scalar rhs) noexcept
    {
        OTI_PROFILE_COUNT(sub);
        OTI_PROFILE_COUNT(sub_scalar);
        c_[0] -= static_cast<Coeff>(rhs);
        return *this;
    }

    template <class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
    OTI_CONSTEXPR otinum& operator*=(Scalar rhs) noexcept
    {
        OTI_PROFILE_COUNT(mul);
        OTI_PROFILE_COUNT(mul_scalar);
        Coeff factor = static_cast<Coeff>(rhs);
        for (int i = 0; i < ncoeffs; ++i) {
            c_[static_cast<std::size_t>(i)] *= factor;
        }
        return *this;
    }

    template <class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
    OTI_CONSTEXPR otinum& operator/=(Scalar rhs) noexcept
    {
        OTI_PROFILE_COUNT(div);
        OTI_PROFILE_COUNT(div_scalar);
        Coeff divisor = static_cast<Coeff>(rhs);
        if constexpr (N > 0) {
            bool singular_divisor = !detail::oti_isfinite(Coeff(1) / divisor);
            c_[0] /= divisor;
            if (singular_divisor) {
                for (int i = 1; i < ncoeffs; ++i) {
                    c_[static_cast<std::size_t>(i)] = static_cast<Coeff>(NAN);
                }
                return *this;
            }
            for (int i = 1; i < ncoeffs; ++i) {
                c_[static_cast<std::size_t>(i)] /= divisor;
            }
            return *this;
        }
        for (int i = 0; i < ncoeffs; ++i) {
            c_[static_cast<std::size_t>(i)] /= divisor;
        }
        return *this;
    }

private:
    static int rank_sparse(sparse_alpha const& sparse) noexcept
    {
        alpha_type alpha{};
        for (sparse_index term : sparse.terms) {
            if (term.variable < 0 || term.variable >= M || term.order < 0) {
                return -1;
            }
            alpha[static_cast<std::size_t>(term.variable)] += term.order;
        }
        return detail::rank<M, N>(alpha);
    }

    detail::array<Coeff, ncoeffs> c_{};
};

// Comparisons intentionally use only the real coefficient. This makes OTI values
// usable in ordinary branch/control-flow code while leaving derivative
// propagation to the arithmetic on the branch that is actually taken.
template <int M, int N, class Coeff>
OTI_CONSTEXPR_FUNCTION bool operator==(otinum<M, N, Coeff> const& lhs,
                                       otinum<M, N, Coeff> const& rhs) noexcept
{
    return lhs.real() == rhs.real();
}

template <int M, int N, class Coeff>
OTI_CONSTEXPR_FUNCTION bool operator!=(otinum<M, N, Coeff> const& lhs,
                                       otinum<M, N, Coeff> const& rhs) noexcept
{
    return !(lhs == rhs);
}

template <int M, int N, class Coeff>
OTI_CONSTEXPR_FUNCTION bool operator<(otinum<M, N, Coeff> const& lhs,
                                      otinum<M, N, Coeff> const& rhs) noexcept
{
    return lhs.real() < rhs.real();
}

template <int M, int N, class Coeff>
OTI_CONSTEXPR_FUNCTION bool operator<=(otinum<M, N, Coeff> const& lhs,
                                       otinum<M, N, Coeff> const& rhs) noexcept
{
    return lhs.real() <= rhs.real();
}

template <int M, int N, class Coeff>
OTI_CONSTEXPR_FUNCTION bool operator>(otinum<M, N, Coeff> const& lhs,
                                      otinum<M, N, Coeff> const& rhs) noexcept
{
    return lhs.real() > rhs.real();
}

template <int M, int N, class Coeff>
OTI_CONSTEXPR_FUNCTION bool operator>=(otinum<M, N, Coeff> const& lhs,
                                       otinum<M, N, Coeff> const& rhs) noexcept
{
    return lhs.real() >= rhs.real();
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_CONSTEXPR_FUNCTION bool operator==(otinum<M, N, Coeff> const& lhs, Scalar rhs) noexcept
{
    return lhs.real() == static_cast<Coeff>(rhs);
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_CONSTEXPR_FUNCTION bool operator==(Scalar lhs, otinum<M, N, Coeff> const& rhs) noexcept
{
    return static_cast<Coeff>(lhs) == rhs.real();
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_CONSTEXPR_FUNCTION bool operator!=(otinum<M, N, Coeff> const& lhs, Scalar rhs) noexcept
{
    return !(lhs == rhs);
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_CONSTEXPR_FUNCTION bool operator!=(Scalar lhs, otinum<M, N, Coeff> const& rhs) noexcept
{
    return !(lhs == rhs);
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_CONSTEXPR_FUNCTION bool operator<(otinum<M, N, Coeff> const& lhs, Scalar rhs) noexcept
{
    return lhs.real() < static_cast<Coeff>(rhs);
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_CONSTEXPR_FUNCTION bool operator<(Scalar lhs, otinum<M, N, Coeff> const& rhs) noexcept
{
    return static_cast<Coeff>(lhs) < rhs.real();
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_CONSTEXPR_FUNCTION bool operator<=(otinum<M, N, Coeff> const& lhs, Scalar rhs) noexcept
{
    return lhs.real() <= static_cast<Coeff>(rhs);
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_CONSTEXPR_FUNCTION bool operator<=(Scalar lhs, otinum<M, N, Coeff> const& rhs) noexcept
{
    return static_cast<Coeff>(lhs) <= rhs.real();
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_CONSTEXPR_FUNCTION bool operator>(otinum<M, N, Coeff> const& lhs, Scalar rhs) noexcept
{
    return lhs.real() > static_cast<Coeff>(rhs);
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_CONSTEXPR_FUNCTION bool operator>(Scalar lhs, otinum<M, N, Coeff> const& rhs) noexcept
{
    return static_cast<Coeff>(lhs) > rhs.real();
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_CONSTEXPR_FUNCTION bool operator>=(otinum<M, N, Coeff> const& lhs, Scalar rhs) noexcept
{
    return lhs.real() >= static_cast<Coeff>(rhs);
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_CONSTEXPR_FUNCTION bool operator>=(Scalar lhs, otinum<M, N, Coeff> const& rhs) noexcept
{
    return static_cast<Coeff>(lhs) >= rhs.real();
}

template <int M, int N, class Coeff>
OTI_CONSTEXPR otinum<M, N, Coeff> operator+(otinum<M, N, Coeff> lhs,
                                            otinum<M, N, Coeff> const& rhs) noexcept
{
    lhs += rhs;
    return lhs;
}

template <int M, int N, class Coeff>
OTI_CONSTEXPR otinum<M, N, Coeff> operator-(otinum<M, N, Coeff> lhs,
                                            otinum<M, N, Coeff> const& rhs) noexcept
{
    lhs -= rhs;
    return lhs;
}

template <int M, int N, class Coeff>
OTI_CONSTEXPR otinum<M, N, Coeff> operator-(otinum<M, N, Coeff> value) noexcept
{
    OTI_PROFILE_COUNT(neg);
    value *= Coeff(-1);
    return value;
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_CONSTEXPR otinum<M, N, Coeff> operator+(otinum<M, N, Coeff> lhs, Scalar rhs) noexcept
{
    lhs += rhs;
    return lhs;
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_CONSTEXPR otinum<M, N, Coeff> operator+(Scalar lhs, otinum<M, N, Coeff> rhs) noexcept
{
    rhs += lhs;
    return rhs;
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_CONSTEXPR otinum<M, N, Coeff> operator-(otinum<M, N, Coeff> lhs, Scalar rhs) noexcept
{
    lhs -= rhs;
    return lhs;
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_CONSTEXPR otinum<M, N, Coeff> operator-(Scalar lhs, otinum<M, N, Coeff> rhs) noexcept
{
    rhs *= Coeff(-1);
    rhs += lhs;
    return rhs;
}

template <int M, int N, class Coeff>
OTI_CONSTEXPR otinum<M, N, Coeff> operator*(otinum<M, N, Coeff> const& lhs,
                                            otinum<M, N, Coeff> const& rhs) noexcept
{
    using tables = detail::tables<M, N>;
    otinum<M, N, Coeff> out;
    OTI_PROFILE_COUNT(mul);
    OTI_PROFILE_COUNT(mul_oti);

    // Polynomial convolution in the truncated multi-index algebra:
    // c[alpha + beta] += lhs[alpha] * rhs[beta] when |alpha + beta| <= N.
    for (int p = 0; p < tables::nproducts; ++p) {
        auto const term = tables::product_term_value(p);
        out[term.out] += lhs[term.lhs] * rhs[term.rhs];
    }

    return out;
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_CONSTEXPR otinum<M, N, Coeff> operator*(otinum<M, N, Coeff> lhs, Scalar rhs) noexcept
{
    lhs *= rhs;
    return lhs;
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_CONSTEXPR otinum<M, N, Coeff> operator*(Scalar lhs, otinum<M, N, Coeff> rhs) noexcept
{
    rhs *= lhs;
    return rhs;
}

template <int M, int N, class Coeff>
OTI_CONSTEXPR otinum<M, N, Coeff> trunc_mul(otinum<M, N, Coeff> const& lhs,
                                            otinum<M, N, Coeff> const& rhs,
                                            int max_order) noexcept
{
    using tables = detail::tables<M, N>;
    otinum<M, N, Coeff> out;
    OTI_PROFILE_COUNT(trunc_mul);
    if (max_order < 0) {
        return out;
    }
    if (max_order > N) {
        max_order = N;
    }

    // Same convolution as operator*, but discards every term above max_order.
    for (int p = 0; p < tables::nproducts; ++p) {
        auto const term = tables::product_term_value(p);
        if (tables::order_of_value(term.out) <= max_order) {
            out[term.out] += lhs[term.lhs] * rhs[term.rhs];
        }
    }

    return out;
}

template <int M, int N, class Coeff>
OTI_CONSTEXPR otinum<M, N, Coeff> trunc_add(otinum<M, N, Coeff> const& lhs,
                                            otinum<M, N, Coeff> const& rhs,
                                            int max_order) noexcept
{
    using tables = detail::tables<M, N>;
    otinum<M, N, Coeff> out;
    OTI_PROFILE_COUNT(trunc_add);
    if (max_order < 0) {
        return out;
    }
    if (max_order > N) {
        max_order = N;
    }

    for (int i = 0; i < otinum<M, N, Coeff>::ncoeffs; ++i) {
        if (tables::order_of_value(i) <= max_order) {
            out[i] = lhs[i] + rhs[i];
        }
    }
    return out;
}

template <int M, int N, class Coeff>
OTI_CONSTEXPR otinum<M, N, Coeff> gem(otinum<M, N, Coeff> const& a,
                                      otinum<M, N, Coeff> const& b,
                                      otinum<M, N, Coeff> const& c) noexcept
{
    OTI_PROFILE_COUNT(gem);
    return a * b + c;
}

template <int M, int N, class Coeff>
OTI_FUNCTION otinum<M, N, Coeff> inv(otinum<M, N, Coeff> const& value) noexcept
{
    using tables = detail::tables<M, N>;
    // The inverse is expanded around the real coefficient. A valid real-valued
    // Taylor inverse requires value.real() != 0.
    Coeff r = value.real();
    OTI_PROFILE_COUNT(inv);

    otinum<M, N, Coeff> out;
    out[0] = Coeff(1) / r;

    if constexpr (N > 0) {
        if (!detail::oti_isfinite(out[0])) {
            // value.real() == 0 (or non-finite): 1/x has a pole here, so there
            // is no valid Taylor jet. Report the scalar value (out[0], typically
            // inf) and NaN for every derivative -- the same singular-point
            // contract that the elementary functions use (see apply_scalar).
            for (int i = 1; i < otinum<M, N, Coeff>::ncoeffs; ++i) {
                out[i] = static_cast<Coeff>(NAN);
            }
            return out;
        }

        // Solve value * out = 1 coefficient-by-coefficient.
        //
        // For every non-real coefficient k, the target coefficient of the
        // product is zero:
        //
        //   (value * out)[k] = value[0] * out[k] + known_terms = 0
        //
        // Product terms are grouped by output coefficient. The term
        // value[0] * out[k] contains the unknown coefficient being solved, so
        // it is skipped while accumulating known_terms. All other dependencies
        // involve out coefficients of lower total order and have already been
        // computed because the coefficient layout is graded by total order.
        for (int k = 1; k < otinum<M, N, Coeff>::ncoeffs; ++k) {
            Coeff accum = Coeff(0);
            int begin = tables::product_offset_value(k);
            int end = tables::product_offset_value(k + 1);
            for (int p = begin; p < end; ++p) {
                auto const term = tables::product_term_by_output_value(p);
                if (term.lhs == 0 && term.rhs == k) {
                    continue;
                }
                accum += value[term.lhs] * out[term.rhs];
            }
            out[k] = -accum / r;
        }
    }
    return out;
}

template <int M, int N, class Coeff>
OTI_FUNCTION otinum<M, N, Coeff> operator/(otinum<M, N, Coeff> const& lhs,
                                           otinum<M, N, Coeff> const& rhs) noexcept
{
    OTI_PROFILE_COUNT(div);
    OTI_PROFILE_COUNT(div_oti);
    return lhs * inv(rhs);
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_CONSTEXPR otinum<M, N, Coeff> operator/(otinum<M, N, Coeff> lhs, Scalar rhs) noexcept
{
    lhs /= rhs;
    return lhs;
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_FUNCTION otinum<M, N, Coeff> operator/(Scalar lhs, otinum<M, N, Coeff> const& rhs) noexcept
{
    return otinum<M, N, Coeff>(static_cast<Coeff>(lhs)) / rhs;
}

} // namespace oti
