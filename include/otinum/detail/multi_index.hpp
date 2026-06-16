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

// Sparse compile-time multi-index: a total-order-<= N index has at most N
// nonzero exponents, so storing only the nonzero (position, exponent) pairs
// costs O(N) instead of the O(M) of a dense alpha_t<M>. The table builders use
// this internally so that high-variable shapes (large M, small N) no longer pay
// the M factor in constexpr time and memory. The dense alpha_t<M> rank() and the
// alpha_at() reconstruction remain for the public coeff()/partial() accessors
// and the device naive path, which work with dense multi-indices.
template <int N>
struct sparse_index {
    static constexpr int cap = N > 0 ? N : 1;
    array<int, cap> pos;  // nonzero positions, strictly increasing
    array<int, cap> exp;  // their exponents (> 0)
    int k;                // number of nonzero entries (<= N)
};

// Merge two sparse indices (sorted by position) into their sum. Callers only
// merge pairs whose combined total order is <= N, so the result still fits.
template <int N>
OTI_CONSTEXPR_FUNCTION sparse_index<N> merge_sparse(sparse_index<N> const& a,
                                                    sparse_index<N> const& b) noexcept
{
    sparse_index<N> g{};
    g.k = 0;
    int i = 0;
    int j = 0;
    while (i < a.k || j < b.k) {
        if (j >= b.k || (i < a.k && a.pos[i] < b.pos[j])) {
            g.pos[g.k] = a.pos[i];
            g.exp[g.k] = a.exp[i];
            ++g.k;
            ++i;
        } else if (i >= a.k || b.pos[j] < a.pos[i]) {
            g.pos[g.k] = b.pos[j];
            g.exp[g.k] = b.exp[j];
            ++g.k;
            ++j;
        } else {
            g.pos[g.k] = a.pos[i];
            g.exp[g.k] = a.exp[i] + b.exp[j];
            ++g.k;
            ++i;
            ++j;
        }
    }
    return g;
}

// Flat rank of a sparse multi-index, identical to rank(dense) but M-independent.
// The dense rank sums a per-position term over all M positions; here the runs of
// zero positions between nonzeros are collapsed with the hockey-stick identity
//   sum_{q=lo}^{hi} C(q + r - 1, r - 1) = C(hi + r, r) - C(lo - 1 + r, r),
// so the cost is O(N^2) rather than O(M).
template <int M, int N>
OTI_CONSTEXPR_FUNCTION int sparse_rank(sparse_index<N> const& s) noexcept
{
    int order = 0;
    for (int t = 0; t < s.k; ++t) {
        order += s.exp[t];
    }
    int index = 0;
    for (int o = 0; o < order; ++o) {
        index += composition_count(M, o);
    }

    int r = order;
    int prev = 0;
    for (int t = 0; t < s.k && r > 0; ++t) {
        int const p = s.pos[t];
        if (p > prev) {
            // Zero positions [prev, p) each contribute with the same remaining r.
            index += binom(M - prev - 1 + r, r) - binom(M - p - 1 + r, r);
        }
        index += binom(M - p - 1 + (r - s.exp[t] - 1), r - s.exp[t] - 1);
        r -= s.exp[t];
        prev = p + 1;
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
    // Build the flat coefficient layout by directly *unranking* each index: for
    // index i (within its total-order band) decode the multi-index whose rank()
    // is i. This is the exact inverse of rank(), so it reproduces the ordering
    // rank() expects (high exponent in the earliest position first) without
    // rank()'s recursion, and stores only the nonzero exponents so it never
    // materializes an M-wide dense vector.
    //
    // Finding each index's nonzero positions skips the (usually long) runs of
    // zero positions analytically rather than scanning all M positions. Within a
    // band of total order `remaining`, making positions [p, p') all zero consumes
    // a known number of the lower-ranked compositions,
    //   cum_zero(p, p') = C(M - p + remaining - 1, remaining)
    //                   - C(M - p' + remaining - 1, remaining),
    // which is monotonic in p', so the first nonzero position is a binary search.
    // That makes the builder O(ncoeffs * N * log M) instead of O(ncoeffs * M):
    // the per-index O(M) scan was millions of constexpr steps at large M and
    // overran stricter frontends (notably NVCC's device compiler).
    static OTI_CONSTEXPR_FUNCTION array<sparse_index<N>, ncoeffs> make_idx_to_sparse() noexcept
    {
        array<sparse_index<N>, ncoeffs> out{};
        int index = 0;
        for (int degree = 0; degree <= N; ++degree) {
            int const count = composition_count(M, degree);
            for (int local = 0; local < count; ++local) {
                sparse_index<N> s{};
                s.k = 0;
                int remaining = degree;
                int c = local;
                int p = 0;
                while (remaining > 0) {
                    // Largest p' in [p, M-1] whose leading zero run still fits in
                    // c; that p' is the first nonzero position.
                    int const base = binom(M - p + remaining - 1, remaining);
                    int lo = p;
                    int hi = M - 1;
                    while (lo < hi) {
                        int const mid = (lo + hi + 1) / 2;
                        int const cum = base - binom(M - mid + remaining - 1, remaining);
                        if (cum <= c) {
                            lo = mid;
                        } else {
                            hi = mid - 1;
                        }
                    }
                    int const pos = lo;
                    c -= base - binom(M - pos + remaining - 1, remaining);

                    // The exponent at pos: compositions there run high value first.
                    int value = remaining;
                    while (value >= 1) {
                        int const block = composition_count(M - pos - 1, remaining - value);
                        if (c < block) {
                            break;
                        }
                        c -= block;
                        --value;
                    }

                    s.pos[static_cast<std::size_t>(s.k)] = pos;
                    s.exp[static_cast<std::size_t>(s.k)] = value;
                    ++s.k;
                    remaining -= value;
                    p = pos + 1;
                }
                out[static_cast<std::size_t>(index)] = s;
                ++index;
            }
        }
        return out;
    }

    static OTI_CONSTEXPR_FUNCTION array<int, ncoeffs> make_order_of() noexcept
    {
        array<int, ncoeffs> out{};
        auto const sparse = make_idx_to_sparse();
        for (int i = 0; i < ncoeffs; ++i) {
            int degree = 0;
            for (int t = 0; t < sparse[static_cast<std::size_t>(i)].k; ++t) {
                degree += sparse[static_cast<std::size_t>(i)].exp[static_cast<std::size_t>(t)];
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
        auto const sparse = make_idx_to_sparse();
        for (int i = 0; i < ncoeffs; ++i) {
            double value = 1.0;
            // Zero exponents contribute factorial(0) = 1, so only nonzeros matter.
            for (int t = 0; t < sparse[static_cast<std::size_t>(i)].k; ++t) {
                value *= factorial(sparse[static_cast<std::size_t>(i)].exp[static_cast<std::size_t>(t)]);
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
        auto const sparse = make_idx_to_sparse();
        auto const orders = make_order_of();
        auto const order_starts = make_order_offset();

        for (int i = 0; i < ncoeffs; ++i) {
            int order_i = orders[static_cast<std::size_t>(i)];

            // beta must satisfy order_i + order_j <= N. Because coefficients are
            // graded, those beta are exactly the prefix [0, first index of order
            // N - order_i + 1); iterate it directly instead of scanning all
            // coefficients and discarding the overflow (O(nproducts), not
            // O(ncoeffs^2)). The sum and rank run on sparse indices, so neither
            // the M-wide gamma nor the M-position rank loop is materialized.
            int const jmax = order_starts[static_cast<std::size_t>(N - order_i + 1)];
            for (int j = 0; j < jmax; ++j) {
                int k = sparse_rank<M, N>(merge_sparse<N>(sparse[static_cast<std::size_t>(i)],
                                                          sparse[static_cast<std::size_t>(j)]));
                ++out[static_cast<std::size_t>(k)];
            }
        }

        return out;
    }

    static OTI_CONSTEXPR_FUNCTION int count_product_terms() noexcept
    {
        int count = 0;
        auto const counts = make_product_counts_by_output();
        for (int k = 0; k < ncoeffs; ++k) {
            count += counts[static_cast<std::size_t>(k)];
        }
        return count;
    }

    static OTI_CONSTEXPR_FUNCTION array<int, ncoeffs + 1> make_product_offset() noexcept
    {
        array<int, ncoeffs + 1> out{};
        auto const counts = make_product_counts_by_output();
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
        auto const sparse = make_idx_to_sparse();
        auto const orders = make_order_of();
        auto const order_starts = make_order_offset();
        int index = 0;

        for (int i = 0; i < ncoeffs; ++i) {
            int order_i = orders[static_cast<std::size_t>(i)];

            // Only the graded prefix of beta keeps order_i + order_j <= N; see
            // make_product_counts_by_output. Emission order is unchanged from a
            // full scan with an order test, so the product table is identical.
            int const jmax = order_starts[static_cast<std::size_t>(N - order_i + 1)];
            for (int j = 0; j < jmax; ++j) {
                int ranked = sparse_rank<M, N>(merge_sparse<N>(sparse[static_cast<std::size_t>(i)],
                                                               sparse[static_cast<std::size_t>(j)]));
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
        auto const sparse = make_idx_to_sparse();
        auto const orders = make_order_of();
        auto const offsets = make_product_offset();
        auto const order_starts = make_order_offset();
        array<int, ncoeffs> cursor{};

        for (int i = 0; i < ncoeffs; ++i) {
            int order_i = orders[static_cast<std::size_t>(i)];

            int const jmax = order_starts[static_cast<std::size_t>(N - order_i + 1)];
            for (int j = 0; j < jmax; ++j) {
                int ranked = sparse_rank<M, N>(merge_sparse<N>(sparse[static_cast<std::size_t>(i)],
                                                               sparse[static_cast<std::size_t>(j)]));
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
    static constexpr array<int, ncoeffs> order_of = make_order_of();
    static constexpr array<int, N + 2> order_offset = make_order_offset();
    static constexpr array<double, ncoeffs> factorial_alpha = make_factorial_alpha();
    static constexpr array<product_term, nproducts> product_terms = make_product_terms();
    static constexpr array<product_term, nproducts> product_terms_by_output =
        make_product_terms_by_output();
    static constexpr array<int, ncoeffs + 1> product_offset = make_product_offset();

    // Reconstruct the dense multi-index for a flat index from the sparse table.
    // There is deliberately no stored dense idx_to_alpha array: at high variable
    // count that one member (M ints per coefficient) dominated compile memory,
    // and eagerly evaluating it on class instantiation reintroduced the very
    // O(ncoeffs * M) cost the sparse builders remove. Scattering the <= N nonzero
    // exponents into a transient M-vector is cheap and only the device naive path
    // and the small-shape table tests call it.
    //
    // TODO: Inspect CUDA codegen/profiling for large <M, N>. Runtime-indexed
    // local constexpr arrays may be materialized in hot device loops; if so,
    // revisit direct static table access or another single-table device layout.
    static OTI_CONSTEXPR_FUNCTION alpha_t<M> alpha_at(int index) noexcept
    {
        constexpr auto sparse = make_idx_to_sparse();
        alpha_t<M> out{};
        auto const& s = sparse[static_cast<std::size_t>(index)];
        for (int t = 0; t < s.k; ++t) {
            out[static_cast<std::size_t>(s.pos[static_cast<std::size_t>(t)])] =
                s.exp[static_cast<std::size_t>(t)];
        }
        return out;
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
