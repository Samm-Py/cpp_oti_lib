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

    // cbrt is the real cube root: defined for negative inputs, where pow(x, 1/3)
    // would return NaN. cbrt(-8) = -2, d/dx = 1/(3*(-8)^(2/3)) = 1/12.
    {
        using T1 = oti::otinum<1, 2>;
        T1 neg = T1::variable(0, -8.0);
        T1 cr = oti::cbrt(neg);
        expect_near(cr.real(), -2.0);
        expect_near(cr.partial({1}), 1.0 / 12.0);
        // Round trip through a negative base: cbrt(w^3) == w.
        T1 w = T1::variable(0, -1.5);
        expect_all_near(oti::cbrt(w * w * w), w, 1e-10);
    }

    // Integer pow at a zero base must not contaminate higher-order coefficients
    // with NaN: pow(x, 2) at x == 0 equals x*x exactly, including order 3.
    {
        using T3 = oti::otinum<1, 3>;
        T3 x0 = T3::variable(0, 0.0);
        expect_all_near(oti::pow(x0, 2.0), x0 * x0, 1e-12);
    }

    // Evaluating a function outside its domain (a NaN value) must report NaN for
    // every coefficient, not finite derivative-formula values: log(-1) is
    // undefined, so its derivatives are undefined too.
    {
        using T2 = oti::otinum<1, 2>;
        T2 bad = oti::log(T2::variable(0, -1.0));
        assert(std::isnan(bad.real()));
        assert(std::isnan(bad.partial({1})));
        assert(std::isnan(bad.partial({2})));
    }

    T log10z = oti::log10(z);
    expect_all_near(log10z, oti::log(z) / std::log(10.0), 1e-12);
    expect_all_near(oti::log_base(z, 2.0), oti::log(z) / std::log(2.0), 1e-12);

    std::cout << "exp/log/pow tests passed\n";
}
