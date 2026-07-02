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

    // --- inverse hyperbolics: values, derivatives, and roundtrips ---
    {
        double const x0 = z.real();  // 0.8, safely inside every domain

        T ash = oti::asinh(z);
        expect_near(ash.real(), std::asinh(x0));
        // asinh'(x) = (1 + x^2)^(-1/2); asinh''(x) = -x (1 + x^2)^(-3/2)
        expect_near(ash.partial({1, 0}), 1.0 / std::sqrt(1.0 + x0 * x0));
        expect_near(ash.partial({2, 0}), -x0 * std::pow(1.0 + x0 * x0, -1.5));
        // d/dy z = -2, so the chain rule scales the first partial by -2
        expect_near(ash.partial({0, 1}), -2.0 / std::sqrt(1.0 + x0 * x0));
        expect_all_near(oti::asinh(sh), z, 1e-10);
        // Log identity: asinh(x) = log(x + sqrt(x^2 + 1)), jet-wide
        expect_all_near(ash, oti::log(z + oti::sqrt(z * z + 1.0)), 1e-10);

        T ath = oti::atanh(z);
        expect_near(ath.real(), std::atanh(x0));
        // atanh'(x) = 1/(1 - x^2); atanh''(x) = 2x/(1 - x^2)^2
        expect_near(ath.partial({1, 0}), 1.0 / (1.0 - x0 * x0));
        expect_near(ath.partial({2, 0}), 2.0 * x0 / std::pow(1.0 - x0 * x0, 2.0));
        expect_all_near(oti::atanh(oti::tanh(z)), z, 1e-10);

        T w = z + 1.0;  // 1.8 > 1, inside acosh's domain
        T ach = oti::acosh(w);
        expect_near(ach.real(), std::acosh(w.real()));
        // acosh'(x) = (x^2 - 1)^(-1/2); acosh''(x) = -x (x^2 - 1)^(-3/2)
        double const w0 = w.real();
        expect_near(ach.partial({1, 0}), 1.0 / std::sqrt(w0 * w0 - 1.0));
        expect_near(ach.partial({2, 0}), -w0 * std::pow(w0 * w0 - 1.0, -1.5));
        expect_all_near(oti::acosh(oti::cosh(w)), w, 1e-9);
    }

    // --- inverse hyperbolic singular points: value + NaN derivatives ---
    {
        using T2 = oti::otinum<1, 2>;
        // acosh at 1: value 0, vertical tangent.
        T2 a1 = oti::acosh(T2::variable(0, 1.0));
        assert(a1.real() == 0.0);
        assert(std::isnan(a1.partial({1})) && std::isnan(a1.partial({2})));
        // acosh below the domain: NaN everywhere.
        T2 a0 = oti::acosh(T2::variable(0, 0.5));
        assert(std::isnan(a0.real()) && std::isnan(a0.partial({1})));
        // atanh at the pole: value +inf, derivatives NaN.
        T2 t1 = oti::atanh(T2::variable(0, 1.0));
        assert(std::isinf(t1.real()) && t1.real() > 0.0);
        assert(std::isnan(t1.partial({1})) && std::isnan(t1.partial({2})));
        // atanh outside the domain: NaN everywhere.
        T2 t2 = oti::atanh(T2::variable(0, 2.0));
        assert(std::isnan(t2.real()) && std::isnan(t2.partial({1})));
    }

    std::cout << "trigonometric and hyperbolic tests passed\n";
}
