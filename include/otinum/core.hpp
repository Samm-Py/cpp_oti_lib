#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>

#include "otinum/detail/multi_index.hpp"

namespace oti {

template <int M, int N>
class otinum {
public:
    using table_type = detail::tables<M, N>;
    using alpha_type = detail::alpha_t<M>;

    static constexpr int nvars = M;
    static constexpr int order = N;
    static constexpr int ncoeffs = table_type::ncoeffs;

    constexpr otinum() = default;

    constexpr otinum(double real)
    {
        c_[0] = real;
    }

    static constexpr otinum variable(int i, double value = 0.0)
    {
        otinum out(value);
        assert(i >= 0 && i < M);
        if constexpr (N > 0) {
            alpha_type alpha{};
            alpha[static_cast<std::size_t>(i)] = 1;
            out[detail::rank<M, N>(alpha)] = 1.0;
        }
        return out;
    }

    static constexpr otinum from_coeffs(std::array<double, ncoeffs> const& coeffs)
    {
        otinum out;
        out.c_ = coeffs;
        return out;
    }

    constexpr double real() const noexcept
    {
        return c_[0];
    }

    constexpr double operator[](int flat_index) const noexcept
    {
        return c_[static_cast<std::size_t>(flat_index)];
    }

    constexpr double& operator[](int flat_index) noexcept
    {
        return c_[static_cast<std::size_t>(flat_index)];
    }

    constexpr double deriv(alpha_type const& alpha) const noexcept
    {
        int idx = detail::rank<M, N>(alpha);
        return idx < 0 ? 0.0 : c_[static_cast<std::size_t>(idx)];
    }

    constexpr double partial(alpha_type const& alpha) const noexcept
    {
        int idx = detail::rank<M, N>(alpha);
        if (idx < 0) {
            return 0.0;
        }
        return c_[static_cast<std::size_t>(idx)] *
               table_type::factorial_alpha[static_cast<std::size_t>(idx)];
    }

    constexpr std::array<double, ncoeffs> const& data() const noexcept
    {
        return c_;
    }

    constexpr otinum& operator+=(otinum const& rhs) noexcept
    {
        for (int i = 0; i < ncoeffs; ++i) {
            c_[static_cast<std::size_t>(i)] += rhs.c_[static_cast<std::size_t>(i)];
        }
        return *this;
    }

    constexpr otinum& operator-=(otinum const& rhs) noexcept
    {
        for (int i = 0; i < ncoeffs; ++i) {
            c_[static_cast<std::size_t>(i)] -= rhs.c_[static_cast<std::size_t>(i)];
        }
        return *this;
    }

    constexpr otinum& operator*=(otinum const& rhs) noexcept
    {
        *this = (*this) * rhs;
        return *this;
    }

    otinum& operator/=(otinum const& rhs) noexcept
    {
        *this = (*this) / rhs;
        return *this;
    }

    constexpr otinum& operator+=(double rhs) noexcept
    {
        c_[0] += rhs;
        return *this;
    }

    constexpr otinum& operator-=(double rhs) noexcept
    {
        c_[0] -= rhs;
        return *this;
    }

    constexpr otinum& operator*=(double rhs) noexcept
    {
        for (double& value : c_) {
            value *= rhs;
        }
        return *this;
    }

    constexpr otinum& operator/=(double rhs) noexcept
    {
        for (double& value : c_) {
            value /= rhs;
        }
        return *this;
    }

private:
    std::array<double, ncoeffs> c_{};
};

template <int M, int N>
constexpr otinum<M, N> operator+(otinum<M, N> lhs, otinum<M, N> const& rhs) noexcept
{
    lhs += rhs;
    return lhs;
}

template <int M, int N>
constexpr otinum<M, N> operator-(otinum<M, N> lhs, otinum<M, N> const& rhs) noexcept
{
    lhs -= rhs;
    return lhs;
}

template <int M, int N>
constexpr otinum<M, N> operator-(otinum<M, N> value) noexcept
{
    value *= -1.0;
    return value;
}

template <int M, int N>
constexpr otinum<M, N> operator+(otinum<M, N> lhs, double rhs) noexcept
{
    lhs += rhs;
    return lhs;
}

template <int M, int N>
constexpr otinum<M, N> operator+(double lhs, otinum<M, N> rhs) noexcept
{
    rhs += lhs;
    return rhs;
}

template <int M, int N>
constexpr otinum<M, N> operator-(otinum<M, N> lhs, double rhs) noexcept
{
    lhs -= rhs;
    return lhs;
}

template <int M, int N>
constexpr otinum<M, N> operator-(double lhs, otinum<M, N> rhs) noexcept
{
    rhs *= -1.0;
    rhs += lhs;
    return rhs;
}

template <int M, int N>
constexpr otinum<M, N> operator*(otinum<M, N> const& lhs, otinum<M, N> const& rhs) noexcept
{
    using tables = detail::tables<M, N>;
    otinum<M, N> out;

    for (int i = 0; i < otinum<M, N>::ncoeffs; ++i) {
        auto const& alpha = tables::idx_to_alpha[static_cast<std::size_t>(i)];
        int order_i = tables::order_of[static_cast<std::size_t>(i)];

        for (int j = 0; j < otinum<M, N>::ncoeffs; ++j) {
            int order_j = tables::order_of[static_cast<std::size_t>(j)];
            if (order_i + order_j > N) {
                continue;
            }

            detail::alpha_t<M> gamma{};
            auto const& beta = tables::idx_to_alpha[static_cast<std::size_t>(j)];
            for (int m = 0; m < M; ++m) {
                gamma[static_cast<std::size_t>(m)] =
                    alpha[static_cast<std::size_t>(m)] + beta[static_cast<std::size_t>(m)];
            }

            int k = detail::rank<M, N>(gamma);
            out[k] += lhs[i] * rhs[j];
        }
    }

    return out;
}

template <int M, int N>
constexpr otinum<M, N> operator*(otinum<M, N> lhs, double rhs) noexcept
{
    lhs *= rhs;
    return lhs;
}

template <int M, int N>
constexpr otinum<M, N> operator*(double lhs, otinum<M, N> rhs) noexcept
{
    rhs *= lhs;
    return rhs;
}

template <int M, int N>
constexpr otinum<M, N> trunc_mul(otinum<M, N> const& lhs,
                                 otinum<M, N> const& rhs,
                                 int max_order) noexcept
{
    using tables = detail::tables<M, N>;
    otinum<M, N> out;
    if (max_order < 0) {
        return out;
    }
    if (max_order > N) {
        max_order = N;
    }

    for (int i = 0; i < otinum<M, N>::ncoeffs; ++i) {
        auto const& alpha = tables::idx_to_alpha[static_cast<std::size_t>(i)];
        int order_i = tables::order_of[static_cast<std::size_t>(i)];
        if (order_i > max_order) {
            continue;
        }

        for (int j = 0; j < otinum<M, N>::ncoeffs; ++j) {
            int order_j = tables::order_of[static_cast<std::size_t>(j)];
            if (order_j > max_order || order_i + order_j > max_order) {
                continue;
            }

            detail::alpha_t<M> gamma{};
            auto const& beta = tables::idx_to_alpha[static_cast<std::size_t>(j)];
            for (int m = 0; m < M; ++m) {
                gamma[static_cast<std::size_t>(m)] =
                    alpha[static_cast<std::size_t>(m)] + beta[static_cast<std::size_t>(m)];
            }

            int k = detail::rank<M, N>(gamma);
            out[k] += lhs[i] * rhs[j];
        }
    }

    return out;
}

template <int M, int N>
constexpr otinum<M, N> trunc_add(otinum<M, N> const& lhs,
                                 otinum<M, N> const& rhs,
                                 int max_order) noexcept
{
    using tables = detail::tables<M, N>;
    otinum<M, N> out;
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
constexpr otinum<M, N> gem(otinum<M, N> const& a,
                           otinum<M, N> const& b,
                           otinum<M, N> const& c) noexcept
{
    return a * b + c;
}

template <int M, int N>
otinum<M, N> inv(otinum<M, N> const& value) noexcept
{
    double r = value.real();
    otinum<M, N> h = value;
    h[0] = 0.0;
    h /= r;

    otinum<M, N> out(1.0 / r);
    otinum<M, N> hk(1.0);
    double sign = -1.0;
    for (int k = 1; k <= N; ++k) {
        hk = hk * h;
        out += (sign / r) * hk;
        sign = -sign;
    }
    return out;
}

template <int M, int N>
otinum<M, N> operator/(otinum<M, N> const& lhs, otinum<M, N> const& rhs) noexcept
{
    return lhs * inv(rhs);
}

template <int M, int N>
constexpr otinum<M, N> operator/(otinum<M, N> lhs, double rhs) noexcept
{
    lhs /= rhs;
    return lhs;
}

template <int M, int N>
otinum<M, N> operator/(double lhs, otinum<M, N> const& rhs) noexcept
{
    return otinum<M, N>(lhs) / rhs;
}

} // namespace oti
