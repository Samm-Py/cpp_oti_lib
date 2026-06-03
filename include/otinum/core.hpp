#pragma once

#include <cstddef>

#include "otinum/detail/multi_index.hpp"
#include "otinum/profile.hpp"

namespace oti {

// Static OTI number with M infinitesimal variables, truncated at total order N.
//
// Coefficients are normalized Taylor coefficients:
//   c[alpha] = (1 / alpha!) * partial^alpha f
// Use partial(alpha) to recover the ordinary derivative value.
template <int M, int N>
class otinum {
public:
    using table_type = detail::tables<M, N>;
    using alpha_type = detail::alpha_t<M>;

    static constexpr int nvars = M;
    static constexpr int order = N;
    static constexpr int ncoeffs = table_type::ncoeffs;

    OTI_CONSTEXPR_FUNCTION otinum() = default;

    // Lift a scalar into the OTI algebra. All derivative coefficients are zero.
    OTI_CONSTEXPR_FUNCTION otinum(double real)
    {
        c_[0] = real;
    }

    // Create value + e_i, where e_i is the first-order nilpotent for variable i.
    static OTI_CONSTEXPR_FUNCTION otinum variable(int i, double value = 0.0)
    {
        otinum out(value);
        OTI_ASSERT(i >= 0 && i < M);
        if constexpr (N > 0) {
            alpha_type alpha{};
            alpha[static_cast<std::size_t>(i)] = 1;
            out[detail::rank<M, N>(alpha)] = 1.0;
        }
        return out;
    }

    // Construct directly from the library's flat graded multi-index layout.
    static OTI_CONSTEXPR_FUNCTION otinum from_coeffs(detail::array<double, ncoeffs> const& coeffs)
    {
        otinum out;
        out.c_ = coeffs;
        return out;
    }

    OTI_CONSTEXPR_FUNCTION double real() const noexcept
    {
        return c_[0];
    }

    // Raw normalized coefficient access by flat multi-index rank.
    OTI_CONSTEXPR_FUNCTION double operator[](int flat_index) const noexcept
    {
        return c_[static_cast<std::size_t>(flat_index)];
    }

    OTI_CONSTEXPR_FUNCTION double& operator[](int flat_index) noexcept
    {
        return c_[static_cast<std::size_t>(flat_index)];
    }

    // Return the normalized Taylor coefficient for alpha, or zero if alpha is
    // outside this otinum's configured total order.
    OTI_CONSTEXPR_FUNCTION double deriv(alpha_type const& alpha) const noexcept
    {
        int idx = detail::rank<M, N>(alpha);
        return idx < 0 ? 0.0 : c_[static_cast<std::size_t>(idx)];
    }

    // Return the ordinary partial derivative value alpha! * c[alpha].
    OTI_CONSTEXPR_FUNCTION double partial(alpha_type const& alpha) const noexcept
    {
        int idx = detail::rank<M, N>(alpha);
        if (idx < 0) {
            return 0.0;
        }
        return c_[static_cast<std::size_t>(idx)] *
               table_type::factorial_alpha[static_cast<std::size_t>(idx)];
    }

    OTI_CONSTEXPR_FUNCTION detail::array<double, ncoeffs> const& data() const noexcept
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

    OTI_CONSTEXPR otinum& operator+=(double rhs) noexcept
    {
        OTI_PROFILE_COUNT(add);
        OTI_PROFILE_COUNT(add_scalar);
        c_[0] += rhs;
        return *this;
    }

    OTI_CONSTEXPR otinum& operator-=(double rhs) noexcept
    {
        OTI_PROFILE_COUNT(sub);
        OTI_PROFILE_COUNT(sub_scalar);
        c_[0] -= rhs;
        return *this;
    }

    OTI_CONSTEXPR otinum& operator*=(double rhs) noexcept
    {
        OTI_PROFILE_COUNT(mul);
        OTI_PROFILE_COUNT(mul_scalar);
        for (int i = 0; i < ncoeffs; ++i) {
            c_[static_cast<std::size_t>(i)] *= rhs;
        }
        return *this;
    }

    OTI_CONSTEXPR otinum& operator/=(double rhs) noexcept
    {
        OTI_PROFILE_COUNT(div);
        OTI_PROFILE_COUNT(div_scalar);
        for (int i = 0; i < ncoeffs; ++i) {
            c_[static_cast<std::size_t>(i)] /= rhs;
        }
        return *this;
    }

private:
    detail::array<double, ncoeffs> c_{};
};

template <int M, int N>
OTI_CONSTEXPR otinum<M, N> operator+(otinum<M, N> lhs, otinum<M, N> const& rhs) noexcept
{
    lhs += rhs;
    return lhs;
}

template <int M, int N>
OTI_CONSTEXPR otinum<M, N> operator-(otinum<M, N> lhs, otinum<M, N> const& rhs) noexcept
{
    lhs -= rhs;
    return lhs;
}

template <int M, int N>
OTI_CONSTEXPR otinum<M, N> operator-(otinum<M, N> value) noexcept
{
    OTI_PROFILE_COUNT(neg);
    value *= -1.0;
    return value;
}

template <int M, int N>
OTI_CONSTEXPR otinum<M, N> operator+(otinum<M, N> lhs, double rhs) noexcept
{
    lhs += rhs;
    return lhs;
}

template <int M, int N>
OTI_CONSTEXPR otinum<M, N> operator+(double lhs, otinum<M, N> rhs) noexcept
{
    rhs += lhs;
    return rhs;
}

template <int M, int N>
OTI_CONSTEXPR otinum<M, N> operator-(otinum<M, N> lhs, double rhs) noexcept
{
    lhs -= rhs;
    return lhs;
}

template <int M, int N>
OTI_CONSTEXPR otinum<M, N> operator-(double lhs, otinum<M, N> rhs) noexcept
{
    rhs *= -1.0;
    rhs += lhs;
    return rhs;
}

template <int M, int N>
OTI_CONSTEXPR otinum<M, N> operator*(otinum<M, N> const& lhs, otinum<M, N> const& rhs) noexcept
{
    using tables = detail::tables<M, N>;
    otinum<M, N> out;
    OTI_PROFILE_COUNT(mul);
    OTI_PROFILE_COUNT(mul_oti);

    // Polynomial convolution in the truncated multi-index algebra:
    // c[alpha + beta] += lhs[alpha] * rhs[beta] when |alpha + beta| <= N.
    for (auto const& term : tables::product_terms) {
        out[term.out] += lhs[term.lhs] * rhs[term.rhs];
    }

    return out;
}

template <int M, int N>
OTI_CONSTEXPR otinum<M, N> operator*(otinum<M, N> lhs, double rhs) noexcept
{
    lhs *= rhs;
    return lhs;
}

template <int M, int N>
OTI_CONSTEXPR otinum<M, N> operator*(double lhs, otinum<M, N> rhs) noexcept
{
    rhs *= lhs;
    return rhs;
}

template <int M, int N>
OTI_CONSTEXPR otinum<M, N> trunc_mul(otinum<M, N> const& lhs,
                                    otinum<M, N> const& rhs,
                                    int max_order) noexcept
{
    using tables = detail::tables<M, N>;
    otinum<M, N> out;
    OTI_PROFILE_COUNT(trunc_mul);
    if (max_order < 0) {
        return out;
    }
    if (max_order > N) {
        max_order = N;
    }

    // Same convolution as operator*, but discards every term above max_order.
    for (auto const& term : tables::product_terms) {
        if (tables::order_of[static_cast<std::size_t>(term.out)] <= max_order) {
            out[term.out] += lhs[term.lhs] * rhs[term.rhs];
        }
    }

    return out;
}

template <int M, int N>
OTI_CONSTEXPR otinum<M, N> trunc_add(otinum<M, N> const& lhs,
                                    otinum<M, N> const& rhs,
                                    int max_order) noexcept
{
    using tables = detail::tables<M, N>;
    otinum<M, N> out;
    OTI_PROFILE_COUNT(trunc_add);
    if (max_order < 0) {
        return out;
    }
    if (max_order > N) {
        max_order = N;
    }

    for (int i = 0; i < otinum<M, N>::ncoeffs; ++i) {
        if (tables::order_of[static_cast<std::size_t>(i)] <= max_order) {
            out[i] = lhs[i] + rhs[i];
        }
    }
    return out;
}

template <int M, int N>
OTI_CONSTEXPR otinum<M, N> gem(otinum<M, N> const& a,
                              otinum<M, N> const& b,
                              otinum<M, N> const& c) noexcept
{
    OTI_PROFILE_COUNT(gem);
    return a * b + c;
}

template <int M, int N>
OTI_FUNCTION otinum<M, N> inv(otinum<M, N> const& value) noexcept
{
    using tables = detail::tables<M, N>;
    double r = value.real();
    OTI_PROFILE_COUNT(inv);

    otinum<M, N> out;
    out[0] = 1.0 / r;

    // Solve value * out = 1 coefficient-by-coefficient. Product terms are
    // grouped by output coefficient, and all dependencies for coefficient k
    // have lower total order than k except value[0] * out[k].
    for (int k = 1; k < otinum<M, N>::ncoeffs; ++k) {
        double accum = 0.0;
        int begin = tables::product_offset[static_cast<std::size_t>(k)];
        int end = tables::product_offset[static_cast<std::size_t>(k + 1)];
        for (int p = begin; p < end; ++p) {
            auto const& term = tables::product_terms_by_output[static_cast<std::size_t>(p)];
            if (term.lhs == 0 && term.rhs == k) {
                continue;
            }
            accum += value[term.lhs] * out[term.rhs];
        }
        out[k] = -accum / r;
    }
    return out;
}

template <int M, int N>
OTI_FUNCTION otinum<M, N> operator/(otinum<M, N> const& lhs, otinum<M, N> const& rhs) noexcept
{
    OTI_PROFILE_COUNT(div);
    OTI_PROFILE_COUNT(div_oti);
    return lhs * inv(rhs);
}

template <int M, int N>
OTI_CONSTEXPR otinum<M, N> operator/(otinum<M, N> lhs, double rhs) noexcept
{
    lhs /= rhs;
    return lhs;
}

template <int M, int N>
OTI_FUNCTION otinum<M, N> operator/(double lhs, otinum<M, N> const& rhs) noexcept
{
    return otinum<M, N>(lhs) / rhs;
}

} // namespace oti
