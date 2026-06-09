#include <cmath>
#include <iomanip>
#include <iostream>

#include "otinum/otinum.hpp"

int main()
{
    using T = oti::otinum<1, 2>;

    double x0 = 1.5;
    double y0 = 0.3;
    double vx = 2.0;
    double vy = -1.0;

    T x(x0);
    T y(y0);

    x.set_coeff({1}, vx);
    y.set_coeff({1}, vy);

    T f = oti::sin(x * y) + oti::exp(x);

    double xy = x0 * y0;
    double fx = y0 * std::cos(xy) + std::exp(x0);
    double fy = x0 * std::cos(xy);
    double fxx = -y0 * y0 * std::sin(xy) + std::exp(x0);
    double fxy = std::cos(xy) - xy * std::sin(xy);
    double fyy = -x0 * x0 * std::sin(xy);

    double analytic_first = fx * vx + fy * vy;
    double analytic_second =
        fxx * vx * vx + 2.0 * fxy * vx * vy + fyy * vy * vy;

    auto print_check = [](const char* name, double analytic, double ad) {
        std::cout << std::setw(18) << name
                  << " analytic=" << std::setw(16) << analytic
                  << " ad=" << std::setw(16) << ad
                  << " abs_diff=" << std::abs(analytic - ad) << '\n';
    };

    print_check("directional d1", analytic_first, f.partial({1}));
    print_check("directional d2", analytic_second, f.partial({2}));
}
