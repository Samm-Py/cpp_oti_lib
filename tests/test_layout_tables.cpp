#include <cassert>
#include <cstddef>
#include <iostream>

#include "test_utils.hpp"

int main()
{
    using T22 = oti::otinum<2, 2>;
    using Tables22 = oti::detail::tables<2, 2>;

    static_assert(T22::ncoeffs == 6, "unexpected coefficient count");
    static_assert(oti::detail::binom(5, 2) == 10, "bad binomial");
    static_assert(oti::detail::rank<2, 2>({0, 0}) == 0, "bad rank");
    static_assert(oti::detail::rank<2, 2>({1, 0}) == 1, "bad rank");
    static_assert(oti::detail::rank<2, 2>({0, 1}) == 2, "bad rank");
    static_assert(oti::detail::rank<2, 2>({2, 0}) == 3, "bad rank");
    static_assert(oti::detail::rank<2, 2>({1, 1}) == 4, "bad rank");
    static_assert(oti::detail::rank<2, 2>({0, 2}) == 5, "bad rank");
    static_assert(oti::detail::rank<2, 2>({3, 0}) == -1, "bad out-of-range rank");
    static_assert(Tables22::order_offset[0] == 0, "bad order offset");
    static_assert(Tables22::order_offset[1] == 1, "bad order offset");
    static_assert(Tables22::order_offset[2] == 3, "bad order offset");
    static_assert(Tables22::order_offset[3] == 6, "bad order offset");

    static_assert(oti::otinum<3, 3>::ncoeffs == 20, "unexpected K(3,3)");
    static_assert(oti::otinum<10, 2>::ncoeffs == 66, "unexpected K(10,2)");

    // The conditional 16-byte alignment (for vectorized GPU/SIMD loads) must
    // never pad the coefficient block: sizeof stays coeffs * sizeof(Coeff),
    // and only shapes whose byte size is a multiple of 16 are promoted.
    static_assert(sizeof(oti::otinum<3, 1, double>) == 4 * sizeof(double), "padded otinum");
    static_assert(alignof(oti::otinum<3, 1, double>) == 16, "expected 16B alignment");
    static_assert(sizeof(oti::otinum<2, 1, double>) == 3 * sizeof(double), "padded otinum");
    static_assert(alignof(oti::otinum<2, 1, double>) == alignof(double), "unexpected promotion");
    static_assert(sizeof(oti::otinum<3, 3, float>) == 20 * sizeof(float), "padded otinum");
    static_assert(alignof(oti::otinum<3, 3, float>) == 16, "expected 16B alignment");
    static_assert(sizeof(oti::otinum<3, 1, float>) == 4 * sizeof(float), "padded otinum");
    static_assert(alignof(oti::otinum<3, 1, float>) == 16, "expected 16B alignment");
    // Middle tier: float shapes sized a multiple of 8 but not 16 get 8-byte
    // alignment (64-bit vector loads), still with no padding.
    static_assert(sizeof(oti::otinum<5, 1, float>) == 6 * sizeof(float), "padded otinum");
    static_assert(alignof(oti::otinum<5, 1, float>) == 8, "expected 8B alignment");
    static_assert(sizeof(oti::otinum<2, 1, float>) == 3 * sizeof(float), "padded otinum");
    static_assert(alignof(oti::otinum<2, 1, float>) == alignof(float), "unexpected promotion");

    for (int i = 0; i < T22::ncoeffs; ++i) {
        int ranked = oti::detail::rank<2, 2>(Tables22::idx_to_alpha[static_cast<std::size_t>(i)]);
        assert(ranked == i);
    }

    std::cout << "layout and table tests passed\n";
}
