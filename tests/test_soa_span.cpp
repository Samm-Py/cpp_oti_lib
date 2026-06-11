#include <cassert>
#include <cstddef>
#include <iostream>
#include <vector>

#include "otinum/soa.hpp"
#include "test_utils.hpp"

namespace {

// Fill an otinum with distinct, element-dependent coefficients so layout bugs
// (swapped strides, off-by-one) cannot cancel out.
template <class T>
T make_jet(std::size_t i)
{
    T out;
    for (int k = 0; k < T::ncoeffs; ++k) {
        out[k] = static_cast<typename T::coeff_type>(1.0 + 0.5 * static_cast<double>(k)
                                                     + 0.01 * static_cast<double>(i));
    }
    return out;
}

// Layout contract: coefficient k of element i sits at data[k * extent + i],
// and load/store round-trip bit-exactly.
template <int M, int N, class Coeff>
void check_layout_and_roundtrip()
{
    using T = oti::otinum<M, N, Coeff>;
    constexpr std::size_t n = 7;

    std::vector<Coeff> buffer(oti::soa_span<M, N, Coeff>::required_size(n), Coeff(0));
    oti::soa_span<M, N, Coeff> span(buffer.data(), n);

    assert(span.extent() == n);
    assert(span.data() == buffer.data());
    static_assert(oti::soa_span<M, N, Coeff>::ncoeffs == T::ncoeffs,
                  "span must mirror the value type's coefficient count");

    for (std::size_t i = 0; i < n; ++i) {
        span.store(i, make_jet<T>(i));
    }

    for (std::size_t i = 0; i < n; ++i) {
        T const expected = make_jet<T>(i);
        for (int k = 0; k < T::ncoeffs; ++k) {
            assert(buffer[static_cast<std::size_t>(k) * n + i] == expected[k]);
            assert(span.coeff(i, k) == expected[k]);
        }
        T const loaded = span.load(i);
        for (int k = 0; k < T::ncoeffs; ++k) {
            assert(loaded[k] == expected[k]);
        }
    }

    // coeff() returns a writable reference into the buffer.
    span.coeff(2, 1) = Coeff(42);
    assert(span.load(2)[1] == Coeff(42));
}

// Arithmetic on loaded values must be bit-identical to plain AoS arithmetic:
// the span only changes where the bytes live, never the computation.
template <int M, int N, class Coeff>
void check_arithmetic_matches_aos()
{
    using T = oti::otinum<M, N, Coeff>;
    constexpr std::size_t n = 5;

    std::vector<Coeff> xs(oti::soa_span<M, N, Coeff>::required_size(n));
    std::vector<Coeff> ys(oti::soa_span<M, N, Coeff>::required_size(n));
    oti::soa_span<M, N, Coeff> sx(xs.data(), n);
    oti::soa_span<M, N, Coeff> sy(ys.data(), n);

    for (std::size_t i = 0; i < n; ++i) {
        sx.store(i, make_jet<T>(i));
        sy.store(i, make_jet<T>(i + n));
    }

    for (std::size_t i = 0; i < n; ++i) {
        T const a = make_jet<T>(i);
        T const b = make_jet<T>(i + n);
        T const via_aos = a * b + a;
        T const via_soa = sx.load(i) * sy.load(i) + sx.load(i);
        for (int k = 0; k < T::ncoeffs; ++k) {
            assert(via_soa[k] == via_aos[k]);
        }
    }
}

} // namespace

int main()
{
    check_layout_and_roundtrip<3, 1, double>();
    check_layout_and_roundtrip<3, 3, double>();
    check_layout_and_roundtrip<2, 4, double>();
    check_layout_and_roundtrip<3, 1, float>();
    check_layout_and_roundtrip<3, 3, float>();

    check_arithmetic_matches_aos<3, 1, double>();
    check_arithmetic_matches_aos<3, 3, double>();
    check_arithmetic_matches_aos<3, 3, float>();

    std::cout << "test_soa_span passed\n";
    return 0;
}
