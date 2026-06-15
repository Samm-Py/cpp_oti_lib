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
    // Build the flat coefficient layout by directly *unranking* each index:
    // for index i (within its total-order band) decode the multi-index whose
    // rank() is i. This is the exact inverse of rank(), so it reproduces the
    // ordering rank() expects (high exponent in the earliest position first)
    // without rank()'s recursion. An earlier version generated the layout with
    // an M-deep recursive helper, which overran the compiler's constexpr
    // evaluation depth once M exceeded a few hundred; this loop is iterative,
    // so high-variable shapes (e.g. <1000, 1>) build without a depth flag.
    static OTI_CONSTEXPR_FUNCTION array<alpha_t<M>, ncoeffs> make_idx_to_alpha() noexcept
    {
        array<alpha_t<M>, ncoeffs> out{};
        int index = 0;
        for (int degree = 0; degree <= N; ++degree) {
            int const count = composition_count(M, degree);
            for (int local = 0; local < count; ++local) {
                alpha_t<M> alpha{};
                int remaining = degree;
                int rest = local;
                for (int pos = 0; pos < M; ++pos) {
                    if (pos == M - 1) {
                        alpha[static_cast<std::size_t>(pos)] = remaining;
                        break;
                    }
                    // Within a fixed order, entries run with this position's
                    // exponent descending; skip whole blocks until `rest` lands.
                    for (int value = remaining; value >= 0; --value) {
                        int const block = composition_count(M - pos - 1, remaining - value);
                        if (rest < block) {
                            alpha[static_cast<std::size_t>(pos)] = value;
                            remaining -= value;
                            break;
                        }
                        rest -= block;
                    }
                }
                out[static_cast<std::size_t>(index)] = alpha;
                ++index;
            }
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
        constexpr auto order_starts = make_order_offset();

        for (int i = 0; i < ncoeffs; ++i) {
            auto const& alpha = alphas[static_cast<std::size_t>(i)];
            int order_i = orders[static_cast<std::size_t>(i)];

            // beta must satisfy order_i + order_j <= N. Because coefficients are
            // graded, those beta are exactly the prefix [0, first index of order
            // N - order_i + 1); iterate it directly instead of scanning all
            // coefficients and discarding the overflow (O(nproducts), not
            // O(ncoeffs^2)).
            int const jmax = order_starts[static_cast<std::size_t>(N - order_i + 1)];
            for (int j = 0; j < jmax; ++j) {
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
        constexpr auto order_starts = make_order_offset();
        int index = 0;

        for (int i = 0; i < ncoeffs; ++i) {
            auto const& alpha = alphas[static_cast<std::size_t>(i)];
            int order_i = orders[static_cast<std::size_t>(i)];

            // Only the graded prefix of beta keeps order_i + order_j <= N; see
            // make_product_counts_by_output. Emission order is unchanged from a
            // full scan with an order test, so the product table is identical.
            int const jmax = order_starts[static_cast<std::size_t>(N - order_i + 1)];
            for (int j = 0; j < jmax; ++j) {
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
        constexpr auto order_starts = make_order_offset();
        array<int, ncoeffs> cursor{};

        for (int i = 0; i < ncoeffs; ++i) {
            auto const& alpha = alphas[static_cast<std::size_t>(i)];
            int order_i = orders[static_cast<std::size_t>(i)];

            int const jmax = order_starts[static_cast<std::size_t>(N - order_i + 1)];
            for (int j = 0; j < jmax; ++j) {
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
