#include <iostream>

#include "test_utils.hpp"

int main()
{
    using oti_test::expect_all_near;
    using oti_test::expect_near;
    using T = oti::otinum<2, 3>;

    T x = T::variable(0, 1.5);
    T y = T::variable(1, 0.3);

    T p = x * y;
    expect_near(p.real(), 0.45);
    expect_near(p.partial({1, 0}), 0.3);
    expect_near(p.partial({0, 1}), 1.5);
    expect_near(p.partial({1, 1}), 1.0);
    expect_near(p.partial({2, 0}), 0.0);

    T square = x * x;
    expect_near(square.real(), 2.25);
    expect_near(square.partial({1, 0}), 3.0);
    expect_near(square.partial({2, 0}), 2.0);

    T cubic = x * x * x;
    expect_near(cubic.real(), 3.375);
    expect_near(cubic.partial({1, 0}), 6.75);
    expect_near(cubic.partial({2, 0}), 9.0);
    expect_near(cubic.partial({3, 0}), 6.0);

    expect_all_near((x * y) / y, x, 1e-10);
    expect_all_near(x / x, T(1.0), 1e-10);
    expect_all_near(2.0 / x, T(2.0) * oti::inv(x), 1e-10);

    std::cout << "multiplication and division tests passed\n";
}
