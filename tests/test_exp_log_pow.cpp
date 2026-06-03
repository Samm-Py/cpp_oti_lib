#include <cmath>
#include <iostream>

#include "test_utils.hpp"

int main()
{
    using oti_test::expect_all_near;
    using oti_test::expect_near;
    using T = oti::otinum<2, 3>;

    T x = T::variable(0, 1.4);
    T y = T::variable(1, 0.7);
    T z = x + 0.25 * y;

    T ez = oti::exp(z);
    expect_near(ez.real(), std::exp(z.real()));
    expect_near(ez.partial({1, 0}), std::exp(z.real()));
    expect_near(ez.partial({0, 1}), 0.25 * std::exp(z.real()));
    expect_near(ez.partial({2, 0}), std::exp(z.real()));
    expect_near(ez.partial({1, 1}), 0.25 * std::exp(z.real()));

    expect_all_near(oti::log(oti::exp(z)), z, 1e-10);
    expect_all_near(oti::exp(oti::log(z)), z, 1e-10);

    T sq = oti::pow(z, 2.0);
    expect_all_near(sq, z * z, 1e-10);
    expect_all_near(oti::sqrt(z * z), z, 1e-10);
    expect_all_near(oti::pow(z, T(2.0)), z * z, 1e-10);

    T log10z = oti::log10(z);
    expect_all_near(log10z, oti::log(z) / std::log(10.0), 1e-12);
    expect_all_near(oti::logb(z, 2.0), oti::log(z) / std::log(2.0), 1e-12);

    std::cout << "exp/log/pow tests passed\n";
}
