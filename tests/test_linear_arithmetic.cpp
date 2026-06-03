#include <iostream>

#include "test_utils.hpp"

int main()
{
    using oti_test::expect_all_near;
    using oti_test::expect_near;
    using T = oti::otinum<2, 2>;

    T x = T::variable(0, 1.25);
    T y = T::variable(1, -0.5);

    expect_near((x + y).real(), 0.75);
    expect_near((x + y).partial({1, 0}), 1.0);
    expect_near((x + y).partial({0, 1}), 1.0);

    expect_all_near((x + 2.0) - 2.0, x);
    expect_all_near((2.0 + x) - x, T(2.0));
    expect_all_near(2.0 - (2.0 - x), x);
    expect_all_near((x * 3.0) / 3.0, x);
    expect_all_near((3.0 * x) / 3.0, x);
    expect_all_near(-(-x), x);

    std::cout << "linear arithmetic tests passed\n";
}
