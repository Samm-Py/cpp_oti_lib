#pragma once

#include <array>
#include <cstddef>

#include "otinum/detail/binom.hpp"

namespace oti::detail {

template <int M>
using alpha_t = std::array<int, M>;

constexpr int composition_count(int parts, int total) noexcept
{
    if (parts < 0 || total < 0) {
        return 0;
    }
    if (parts == 0) {
        return total == 0 ? 1 : 0;
    }
    return binom(parts + total - 1, total);
}

template <int M>
constexpr int total_order(alpha_t<M> const& alpha) noexcept
{
    int order = 0;
    for (int i = 0; i < M; ++i) {
        order += alpha[i];
    }
    return order;
}

template <int M, int N>
constexpr int rank(alpha_t<M> const& alpha) noexcept
{
    int order = 0;
    for (int i = 0; i < M; ++i) {
        if (alpha[i] < 0) {
            return -1;
        }
        order += alpha[i];
    }
    if (order > N) {
        return -1;
    }

    int index = 0;
    for (int o = 0; o < order; ++o) {
        index += composition_count(M, o);
    }

    int remaining = order;
    for (int pos = 0; pos < M; ++pos) {
        for (int value = remaining; value > alpha[pos]; --value) {
            index += composition_count(M - pos - 1, remaining - value);
        }
        remaining -= alpha[pos];
    }
    return index;
}

template <int M, int N>
struct tables {
    static_assert(M > 0, "otinum requires at least one variable");
    static_assert(N >= 0, "otinum order cannot be negative");

    static constexpr int nvars = M;
    static constexpr int order = N;
    static constexpr int ncoeffs = binom(M + N, N);

private:
    static constexpr void fill_order(std::array<alpha_t<M>, ncoeffs>& out,
                                     alpha_t<M>& current,
                                     int& index,
                                     int pos,
                                     int remaining) noexcept
    {
        if (pos == M - 1) {
            current[pos] = remaining;
            out[static_cast<std::size_t>(index)] = current;
            ++index;
            return;
        }

        for (int value = remaining; value >= 0; --value) {
            current[pos] = value;
            fill_order(out, current, index, pos + 1, remaining - value);
        }
    }

    static constexpr std::array<alpha_t<M>, ncoeffs> make_idx_to_alpha() noexcept
    {
        std::array<alpha_t<M>, ncoeffs> out{};
        alpha_t<M> current{};
        int index = 0;
        for (int degree = 0; degree <= N; ++degree) {
            fill_order(out, current, index, 0, degree);
        }
        return out;
    }

    static constexpr std::array<int, ncoeffs> make_order_of() noexcept
    {
        std::array<int, ncoeffs> out{};
        constexpr auto alphas = make_idx_to_alpha();
        for (int i = 0; i < ncoeffs; ++i) {
            int degree = 0;
            for (int j = 0; j < M; ++j) {
                degree += alphas[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
            }
            out[static_cast<std::size_t>(i)] = degree;
        }
        return out;
    }

    static constexpr std::array<int, N + 2> make_order_offset() noexcept
    {
        std::array<int, N + 2> out{};
        int offset = 0;
        for (int degree = 0; degree <= N; ++degree) {
            out[static_cast<std::size_t>(degree)] = offset;
            offset += composition_count(M, degree);
        }
        out[static_cast<std::size_t>(N + 1)] = offset;
        return out;
    }

    static constexpr std::array<double, ncoeffs> make_factorial_alpha() noexcept
    {
        std::array<double, ncoeffs> out{};
        constexpr auto alphas = make_idx_to_alpha();
        for (int i = 0; i < ncoeffs; ++i) {
            double value = 1.0;
            for (int j = 0; j < M; ++j) {
                value *= factorial(alphas[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]);
            }
            out[static_cast<std::size_t>(i)] = value;
        }
        return out;
    }

public:
    static constexpr std::array<alpha_t<M>, ncoeffs> idx_to_alpha = make_idx_to_alpha();
    static constexpr std::array<int, ncoeffs> order_of = make_order_of();
    static constexpr std::array<int, N + 2> order_offset = make_order_offset();
    static constexpr std::array<double, ncoeffs> factorial_alpha = make_factorial_alpha();
};

template <int M, int N>
constexpr std::array<alpha_t<M>, tables<M, N>::ncoeffs> tables<M, N>::idx_to_alpha;

template <int M, int N>
constexpr std::array<int, tables<M, N>::ncoeffs> tables<M, N>::order_of;

template <int M, int N>
constexpr std::array<int, N + 2> tables<M, N>::order_offset;

template <int M, int N>
constexpr std::array<double, tables<M, N>::ncoeffs> tables<M, N>::factorial_alpha;

} // namespace oti::detail
