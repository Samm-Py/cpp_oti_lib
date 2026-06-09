#include <cmath>
#include <iomanip>
#include <iostream>

#include "otinum/otinum.hpp"

int main()
{
    using T = oti::otinum<2, 2>;

    double x0 = 1.5;
    double y0 = 0.3;

    T x = T::variable(0, x0);
    T y = T::variable(1, y0);
    T f = oti::sin(x * y) + oti::exp(x);

    double xy = x0 * y0;
    double analytic_f = std::sin(xy) + std::exp(x0);
    double analytic_dfdx = y0 * std::cos(xy) + std::exp(x0);
    double analytic_dfdy = x0 * std::cos(xy);
    double analytic_d2fdx2 = -y0 * y0 * std::sin(xy) + std::exp(x0);
    double analytic_d2fdxdy = std::cos(xy) - xy * std::sin(xy);
    double analytic_d2fdy2 = -x0 * x0 * std::sin(xy);

    auto print_check = [](const char* name, double analytic, double ad) {
        std::cout << std::setw(8) << name
                  << " analytic=" << std::setw(16) << analytic
                  << " ad=" << std::setw(16) << ad
                  << " abs_diff=" << std::abs(analytic - ad) << '\n';
    };

    print_check("f", analytic_f, f.real());
    print_check("df/dx", analytic_dfdx, f.partial({1, 0}));
    print_check("df/dy", analytic_dfdy, f.partial({0, 1}));
    print_check("d2f/dx2", analytic_d2fdx2, f.partial({2, 0}));
    print_check("d2f/dxdy", analytic_d2fdxdy, f.partial({1, 1}));
    print_check("d2f/dy2", analytic_d2fdy2, f.partial({0, 2}));
}
