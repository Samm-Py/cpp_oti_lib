#include <cmath>
#include <iostream>

#include "test_utils.hpp"

int main()
{
    using oti_test::expect_all_near;
    using oti_test::expect_near;
    using T = oti::otinum<2, 3>;

    T x = T::variable(0, 0.4);
    T y = T::variable(1, -0.2);
    T z = x - 2.0 * y;

    T s = oti::sin(z);
    T c = oti::cos(z);
    expect_near(s.real(), std::sin(z.real()));
    expect_near(c.real(), std::cos(z.real()));
    expect_near(s.partial({1, 0}), std::cos(z.real()));
    expect_near(c.partial({1, 0}), -std::sin(z.real()));
    expect_near(s.partial({0, 1}), -2.0 * std::cos(z.real()));
    expect_near(c.partial({0, 1}), 2.0 * std::sin(z.real()));
    expect_all_near(s * s + c * c, T(1.0), 1e-10);

    T t = oti::tan(z);
    expect_all_near(t, s / c, 1e-12);

    T sh = oti::sinh(z);
    T ch = oti::cosh(z);
    expect_near(sh.real(), std::sinh(z.real()));
    expect_near(ch.real(), std::cosh(z.real()));
    expect_near(sh.partial({1, 0}), std::cosh(z.real()));
    expect_near(ch.partial({1, 0}), std::sinh(z.real()));
    expect_all_near(ch * ch - sh * sh, T(1.0), 1e-10);
    expect_all_near(oti::tanh(z), sh / ch, 1e-12);

    std::cout << "trigonometric and hyperbolic tests passed\n";
}
