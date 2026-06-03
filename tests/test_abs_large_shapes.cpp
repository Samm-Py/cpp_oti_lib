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
