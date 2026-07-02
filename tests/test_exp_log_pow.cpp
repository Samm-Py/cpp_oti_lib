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

    // At a singular expansion point the value is the scalar value but every
    // derivative is NaN -- one consistent signal across undefined values, poles,
    // and vertical tangents, instead of a mix of inf and nan.
    {
        using T2 = oti::otinum<1, 2>;
        // Undefined value: log(-1) is NaN everywhere.
        T2 bad = oti::log(T2::variable(0, -1.0));
        assert(std::isnan(bad.real()));
        assert(std::isnan(bad.partial({1})) && std::isnan(bad.partial({2})));
        // Pole: log(0) has value -inf, derivatives NaN.
        T2 l0 = oti::log(T2::variable(0, 0.0));
        assert(std::isinf(l0.real()) && l0.real() < 0.0);
        assert(std::isnan(l0.partial({1})) && std::isnan(l0.partial({2})));
        // Vertical tangent: sqrt(0) has value 0, derivatives NaN.
        T2 s0 = oti::sqrt(T2::variable(0, 0.0));
        expect_near(s0.real(), 0.0);
        assert(std::isnan(s0.partial({1})) && std::isnan(s0.partial({2})));
    }

    T log10z = oti::log10(z);
    expect_all_near(log10z, oti::log(z) / std::log(10.0), 1e-12);
    expect_all_near(oti::log_base(z, 2.0), oti::log(z) / std::log(2.0), 1e-12);

    // --- erf / erfc: Gaussian-recurrence derivatives ---
    {
        using T3 = oti::otinum<1, 3>;
        double const x0 = 0.6;
        double const gauss = std::exp(-x0 * x0);
        double const c = 1.12837916709551257390;  // 2 / sqrt(pi)
        T3 v = T3::variable(0, x0);
        T3 e = oti::erf(v);
        expect_near(e.real(), std::erf(x0));
        // erf'(x) = c e^(-x^2); erf'' = -2x erf'; erf''' = (4x^2 - 2) c e^(-x^2)
        expect_near(e.partial({1}), c * gauss);
        expect_near(e.partial({2}), -2.0 * x0 * c * gauss);
        expect_near(e.partial({3}), (4.0 * x0 * x0 - 2.0) * c * gauss);
        // Odd symmetry, jet-wide: erf(-v) = -erf(v).
        expect_all_near(oti::erf(-v), -e, 1e-12);

        // erfc: real part from the dedicated scalar (no 1 - erf cancellation),
        // derivatives exactly the negation of erf's.
        T3 ec = oti::erfc(v);
        expect_near(ec.real(), std::erfc(x0));
        expect_near(ec.partial({1}), -e.partial({1}));
        expect_near(ec.partial({2}), -e.partial({2}));
        expect_near(ec.partial({3}), -e.partial({3}));
        // Large argument: 1 - erf(x) rounds to exactly 0 in double; erfc must
        // keep the ~4e-23 tail (and its still-nonzero derivatives).
        T3 far = oti::erfc(T3::variable(0, 7.0));
        assert(far.real() > 0.0);
        expect_near(far.real(), std::erfc(7.0));
        assert(far.partial({1}) < 0.0);
    }

    std::cout << "exp/log/pow tests passed\n";
}
