#include <iostream>

#include "test_utils.hpp"

int main()
{
    using oti_test::expect_all_near;
    using oti_test::expect_near;

    using T = oti::otinum<2, 2>;
    T positive = T::variable(0, 1.0);
    T negative = T::variable(0, -1.0);
    expect_all_near(oti::abs(positive), positive);
    expect_all_near(oti::abs(negative), -negative);

    // abs has a kink at real() == 0. The value |0| = 0 is defined, but the
    // derivatives are not, so abs signals non-differentiability with NaN rather
    // than guessing a subgradient.
    T at_zero = T::variable(0, 0.0); // 0 + e_0
    T abs_zero = oti::abs(at_zero);
    expect_near(abs_zero.real(), 0.0);
    assert(std::isnan(abs_zero.partial({1, 0})));
    assert(std::isnan(abs_zero.partial({2, 0})));

    // A value that is exactly zero has no perturbation direction; abs returns 0.
    expect_all_near(oti::abs(T{}), T{});

    using Large = oti::otinum<5, 3>;
    Large x0 = Large::variable(0, 1.0);
    Large x4 = Large::variable(4, 2.0);
    Large product = x0 * x4;
    expect_near(product.real(), 2.0);
    expect_near(product.partial({1, 0, 0, 0, 0}), 2.0);
    expect_near(product.partial({0, 0, 0, 0, 1}), 1.0);
    expect_near(product.partial({1, 0, 0, 0, 1}), 1.0);

    std::cout << "abs and large shape tests passed\n";
}
