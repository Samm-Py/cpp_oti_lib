#pragma once

// Compile-time multi-index layout and multiplication tables.
//
// This header maps between mathematical multi-indices alpha and the flat
// coefficient indices used by otinum<M, N>. Coefficients are grouped by total
// order, which makes truncation and order-based loops cheap. The tables also
// precompute every valid truncated product contribution
//
//   alpha_lhs + alpha_rhs = alpha_out, |alpha_out| <= N
//
// so multiplication and coefficient-by-coefficient algorithms can use compact
// integer lookup tables instead of rebuilding multi-index sums at runtime.

#include <cstddef>

#include "otinum/detail/binom.hpp"

namespace oti::detail {

template <int M>
using alpha_t = array<int, M>;

struct product_term {
    int lhs;
    int rhs;
    int out;
};

// Number of weak compositions of total into parts slots.
OTI_CONSTEXPR_FUNCTION int composition_count(int parts, int total) noexcept
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
OTI_CONSTEXPR_FUNCTION int total_order(alpha_t<M> const& alpha) noexcept
{
    int order = 0;
    for (int i = 0; i < M; ++i) {
        order += alpha[i];
    }
    return order;
}

template <int M, int N>
OTI_CONSTEXPR_FUNCTION int rank(alpha_t<M> const& alpha) noexcept
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

    // First skip every lower total order. This makes truncation by order a
    // prefix operation over the flat coefficient array.
    int index = 0;
    for (int o = 0; o < order; ++o) {
        index += composition_count(M, o);
    }

    // Within a fixed total order, entries are generated with earlier positions
    // descending. Count the entries that would appear before alpha.
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
    // Generate all multi-indices of one total order in the same order rank()
    // expects: high exponent in the current position first.
    static OTI_CONSTEXPR_FUNCTION void fill_order(array<alpha_t<M>, ncoeffs>& out,
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

    static OTI_CONSTEXPR_FUNCTION array<alpha_t<M>, ncoeffs> make_idx_to_alpha() noexcept
    {
        array<alpha_t<M>, ncoeffs> out{};
        alpha_t<M> current{};
        int index = 0;
        for (int degree = 0; degree <= N; ++degree) {
            fill_order(out, current, index, 0, degree);
        }
        return out;
    }

    static OTI_CONSTEXPR_FUNCTION array<int, ncoeffs> make_order_of() noexcept
    {
        array<int, ncoeffs> out{};
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

    static OTI_CONSTEXPR_FUNCTION array<int, N + 2> make_order_offset() noexcept
    {
        array<int, N + 2> out{};
        int offset = 0;
        for (int degree = 0; degree <= N; ++degree) {
            out[static_cast<std::size_t>(degree)] = offset;
            offset += composition_count(M, degree);
        }
        out[static_cast<std::size_t>(N + 1)] = offset;
        return out;
    }

    // alpha! per coefficient, narrowed to the requested type. The product is
    // accumulated in double for accuracy and converted to Coeff only once, at
    // compile time. Keeping this templated lets otinum<M, N, float> read a float
    // table, so partial()/set_partial() introduce no double load or double-to-
    // float conversion in single-precision (e.g. Kokkos/CUDA) builds.
    template <class Coeff>
    static OTI_CONSTEXPR_FUNCTION array<Coeff, ncoeffs> make_factorial_alpha_typed() noexcept
    {
        array<Coeff, ncoeffs> out{};
        constexpr auto alphas = make_idx_to_alpha();
        for (int i = 0; i < ncoeffs; ++i) {
            double value = 1.0;
            for (int j = 0; j < M; ++j) {
                value *= factorial(alphas[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]);
            }
            out[static_cast<std::size_t>(i)] = static_cast<Coeff>(value);
        }
        return out;
    }

    static OTI_CONSTEXPR_FUNCTION array<double, ncoeffs> make_factorial_alpha() noexcept
    {
        return make_factorial_alpha_typed<double>();
    }

    static OTI_CONSTEXPR_FUNCTION array<int, ncoeffs> make_product_counts_by_output() noexcept
    {
        array<int, ncoeffs> out{};
        constexpr auto alphas = make_idx_to_alpha();
        constexpr auto orders = make_order_of();

        for (int i = 0; i < ncoeffs; ++i) {
            auto const& alpha = alphas[static_cast<std::size_t>(i)];
            int order_i = orders[static_cast<std::size_t>(i)];

            for (int j = 0; j < ncoeffs; ++j) {
                int order_j = orders[static_cast<std::size_t>(j)];
                if (order_i + order_j > N) {
                    continue;
                }

                alpha_t<M> gamma{};
                auto const& beta = alphas[static_cast<std::size_t>(j)];
                for (int m = 0; m < M; ++m) {
                    gamma[static_cast<std::size_t>(m)] =
                        alpha[static_cast<std::size_t>(m)] + beta[static_cast<std::size_t>(m)];
                }

                int k = rank<M, N>(gamma);
                ++out[static_cast<std::size_t>(k)];
            }
        }

        return out;
    }

    static OTI_CONSTEXPR_FUNCTION int count_product_terms() noexcept
    {
        int count = 0;
        constexpr auto counts = make_product_counts_by_output();
        for (int k = 0; k < ncoeffs; ++k) {
            count += counts[static_cast<std::size_t>(k)];
        }
        return count;
    }

    static OTI_CONSTEXPR_FUNCTION array<int, ncoeffs + 1> make_product_offset() noexcept
    {
        array<int, ncoeffs + 1> out{};
        constexpr auto counts = make_product_counts_by_output();
        int offset = 0;
        for (int k = 0; k < ncoeffs; ++k) {
            out[static_cast<std::size_t>(k)] = offset;
            offset += counts[static_cast<std::size_t>(k)];
        }
        out[static_cast<std::size_t>(ncoeffs)] = offset;
        return out;
    }

    static OTI_CONSTEXPR_FUNCTION array<product_term, count_product_terms()> make_product_terms() noexcept
    {
        array<product_term, count_product_terms()> out{};
        constexpr auto alphas = make_idx_to_alpha();
        constexpr auto orders = make_order_of();
        int index = 0;

        for (int i = 0; i < ncoeffs; ++i) {
            auto const& alpha = alphas[static_cast<std::size_t>(i)];
            int order_i = orders[static_cast<std::size_t>(i)];

            for (int j = 0; j < ncoeffs; ++j) {
                int order_j = orders[static_cast<std::size_t>(j)];
                if (order_i + order_j > N) {
                    continue;
                }

                alpha_t<M> gamma{};
                auto const& beta = alphas[static_cast<std::size_t>(j)];
                for (int m = 0; m < M; ++m) {
                    gamma[static_cast<std::size_t>(m)] =
                        alpha[static_cast<std::size_t>(m)] + beta[static_cast<std::size_t>(m)];
                }

                int ranked = rank<M, N>(gamma);
                out[static_cast<std::size_t>(index)] = {i, j, ranked};
                ++index;
            }
        }

        return out;
    }

    static OTI_CONSTEXPR_FUNCTION array<product_term, count_product_terms()>
    make_product_terms_by_output() noexcept
    {
        array<product_term, count_product_terms()> out{};
        constexpr auto alphas = make_idx_to_alpha();
        constexpr auto orders = make_order_of();
        constexpr auto offsets = make_product_offset();
        array<int, ncoeffs> cursor{};

        for (int i = 0; i < ncoeffs; ++i) {
            auto const& alpha = alphas[static_cast<std::size_t>(i)];
            int order_i = orders[static_cast<std::size_t>(i)];

            for (int j = 0; j < ncoeffs; ++j) {
                int order_j = orders[static_cast<std::size_t>(j)];
                if (order_i + order_j > N) {
                    continue;
                }

                alpha_t<M> gamma{};
                auto const& beta = alphas[static_cast<std::size_t>(j)];
                for (int m = 0; m < M; ++m) {
                    gamma[static_cast<std::size_t>(m)] =
                        alpha[static_cast<std::size_t>(m)] + beta[static_cast<std::size_t>(m)];
                }

                int ranked = rank<M, N>(gamma);
                int index = offsets[static_cast<std::size_t>(ranked)] +
                            cursor[static_cast<std::size_t>(ranked)];
                out[static_cast<std::size_t>(index)] = {i, j, ranked};
                ++cursor[static_cast<std::size_t>(ranked)];
            }
        }

        return out;
    }

public:
    static constexpr int nproducts = count_product_terms();
    static constexpr array<alpha_t<M>, ncoeffs> idx_to_alpha = make_idx_to_alpha();
    static constexpr array<int, ncoeffs> order_of = make_order_of();
    static constexpr array<int, N + 2> order_offset = make_order_offset();
    static constexpr array<double, ncoeffs> factorial_alpha = make_factorial_alpha();
    static constexpr array<product_term, nproducts> product_terms = make_product_terms();
    static constexpr array<product_term, nproducts> product_terms_by_output =
        make_product_terms_by_output();
    static constexpr array<int, ncoeffs + 1> product_offset = make_product_offset();

    // Prefer these accessors in device-callable code. The table expressions are
    // constant-evaluated locally, avoiding direct references to class static
    // constexpr arrays that can be problematic for NVCC/Kokkos device builds.
    // TODO: Inspect CUDA codegen/profiling for large <M, N>. Runtime-indexed
    // local constexpr arrays may be materialized in hot device loops; if so,
    // revisit direct static table access or another single-table device layout.
    static OTI_CONSTEXPR_FUNCTION alpha_t<M> alpha_at(int index) noexcept
    {
        constexpr auto values = make_idx_to_alpha();
        return values[static_cast<std::size_t>(index)];
    }

    static OTI_CONSTEXPR_FUNCTION int order_of_value(int index) noexcept
    {
        constexpr auto values = make_order_of();
        return values[static_cast<std::size_t>(index)];
    }

    static OTI_CONSTEXPR_FUNCTION int order_offset_value(int degree) noexcept
    {
        constexpr auto values = make_order_offset();
        return values[static_cast<std::size_t>(degree)];
    }

    static OTI_CONSTEXPR_FUNCTION double factorial_alpha_value(int index) noexcept
    {
        constexpr auto values = make_factorial_alpha();
        return values[static_cast<std::size_t>(index)];
    }

    // Coefficient-typed alpha! lookup. Prefer this in numeric code: a float
    // instantiation then reads a float table, keeping partial()/set_partial()
    // free of double loads and conversions on single-precision device builds.
    template <class Coeff>
    static OTI_CONSTEXPR_FUNCTION Coeff factorial_alpha_as(int index) noexcept
    {
        constexpr auto values = make_factorial_alpha_typed<Coeff>();
        return values[static_cast<std::size_t>(index)];
    }

    static OTI_CONSTEXPR_FUNCTION int product_count_by_output_value(int output) noexcept
    {
        constexpr auto offsets = make_product_offset();
        return offsets[static_cast<std::size_t>(output + 1)] -
               offsets[static_cast<std::size_t>(output)];
    }

    static OTI_CONSTEXPR_FUNCTION int product_offset_value(int output) noexcept
    {
        constexpr auto values = make_product_offset();
        return values[static_cast<std::size_t>(output)];
    }

    static OTI_CONSTEXPR_FUNCTION product_term product_term_value(int index) noexcept
    {
        constexpr auto values = make_product_terms();
        return values[static_cast<std::size_t>(index)];
    }

    static OTI_CONSTEXPR_FUNCTION product_term product_term_by_output_value(int index) noexcept
    {
        constexpr auto values = make_product_terms_by_output();
        return values[static_cast<std::size_t>(index)];
    }
};

template <int M, int N>
constexpr array<alpha_t<M>, tables<M, N>::ncoeffs> tables<M, N>::idx_to_alpha;

template <int M, int N>
constexpr array<int, tables<M, N>::ncoeffs> tables<M, N>::order_of;

template <int M, int N>
constexpr array<int, N + 2> tables<M, N>::order_offset;

template <int M, int N>
constexpr array<double, tables<M, N>::ncoeffs> tables<M, N>::factorial_alpha;

template <int M, int N>
constexpr array<product_term, tables<M, N>::nproducts> tables<M, N>::product_terms;

template <int M, int N>
constexpr array<product_term, tables<M, N>::nproducts>
    tables<M, N>::product_terms_by_output;

template <int M, int N>
constexpr array<int, tables<M, N>::ncoeffs + 1> tables<M, N>::product_offset;

} // namespace oti::detail
