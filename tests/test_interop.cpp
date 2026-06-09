// Tests for the standard-library interoperability layer (otinum/interop.hpp):
// numeric_limits, streaming, and the <cmath>-style overloads that are not in
// functions.hpp. Derivatives of the new analytic functions are checked against
// analytic forms and central finite differences so the generators are verified
// numerically, not just for compilation.

#include <cassert>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>

#include "test_utils.hpp"

namespace {

using oti_test::expect_near;

// Central finite difference of a scalar function, for cross-checking first
// derivatives produced by the OTI machinery.
template <class F>
double central_diff(F f, double x, double h = 1e-6)
{
    return (f(x + h) - f(x - h)) / (2.0 * h);
}

// First derivative of f at x0 via a one-variable, first-order otinum.
template <class F>
double oti_deriv(F f, double x0)
{
    using T = oti::otinum<1, 1>;
    return f(T::variable(0, x0)).partial({1});
}

void test_numeric_limits()
{
    using T = oti::otinum<3, 2>;
    using L = std::numeric_limits<T>;
    using D = std::numeric_limits<double>;

    static_assert(L::is_specialized, "otinum numeric_limits must be specialized");
    static_assert(!L::is_iec559, "an otinum is not a bit-exact IEEE float");
    static_assert(L::digits == D::digits, "digits must forward from the scalar");

    expect_near(L::max().real(), D::max());
    expect_near(L::lowest().real(), D::lowest());
    expect_near(L::min().real(), D::min());
    expect_near(L::epsilon().real(), D::epsilon());
    assert(std::isinf(L::infinity().real()));
    assert(std::isnan(L::quiet_NaN().real()));

    // Lifted constants must carry no derivatives.
    for (int i = 1; i < T::ncoeffs; ++i) {
        expect_near(L::max()[i], 0.0);
        expect_near(L::epsilon()[i], 0.0);
    }
}

void test_streaming()
{
    using T = oti::otinum<2, 1>;
    T x = T::variable(0, 1.5);
    x.set_partial({0, 1}, 2.0);

    // operator<< prints only the real part, identical to streaming the double.
    std::ostringstream got;
    std::ostringstream want;
    got << x;
    want << 1.5;
    assert(got.str() == want.str());

    // print_coeffs exposes the derivative coefficients.
    std::ostringstream full;
    oti::print_coeffs(full, x);
    assert(full.str().find("1.5") != std::string::npos);
    assert(full.str().find('[') != std::string::npos);
}

void test_rounding()
{
    using T = oti::otinum<1, 1>;
    T x = T::variable(0, 2.7);
    T y = T::variable(0, -2.7);

    expect_near(oti::floor(x).real(), 2.0);
    expect_near(oti::ceil(x).real(), 3.0);
    expect_near(oti::trunc(x).real(), 2.0);
    expect_near(oti::round(x).real(), 3.0);
    expect_near(oti::trunc(y).real(), -2.0);
    expect_near(oti::round(y).real(), -3.0);

    // Step functions are locally constant: zero derivative.
    expect_near(oti::floor(x).partial({1}), 0.0);
    expect_near(oti::ceil(x).partial({1}), 0.0);
    expect_near(oti::round(x).partial({1}), 0.0);

    // fabs matches abs.
    expect_near(oti::fabs(y).real(), 2.7);
    expect_near(oti::fabs(y).partial({1}), -1.0);
}

void test_predicates()
{
    using T = oti::otinum<1, 1>;
    double const nan = std::numeric_limits<double>::quiet_NaN();
    double const inf = std::numeric_limits<double>::infinity();

    assert(oti::signbit(T(-3.0)));
    assert(!oti::signbit(T(3.0)));
    assert(oti::isnan(T(nan)));
    assert(!oti::isnan(T(1.0)));
    assert(oti::isinf(T(inf)));
    assert(!oti::isinf(T(1.0)));
    assert(oti::isfinite(T(1.0)));
    assert(!oti::isfinite(T(inf)));
    assert(!oti::isfinite(T(nan)));
}

void test_selection()
{
    using T = oti::otinum<1, 1>;
    T a = T::variable(0, 2.0); // value 2, d/dx = 1
    T b = T::variable(0, 5.0); // value 5, d/dx = 1
    b.set_partial({1}, 3.0);   // make b's derivative distinguishable

    // fmax returns the larger operand WHOLE, keeping its derivative.
    expect_near(oti::fmax(a, b).real(), 5.0);
    expect_near(oti::fmax(a, b).partial({1}), 3.0);
    expect_near(oti::fmin(a, b).real(), 2.0);
    expect_near(oti::fmin(a, b).partial({1}), 1.0);

    // Scalar overloads.
    expect_near(oti::fmax(a, 10.0).real(), 10.0);
    expect_near(oti::fmax(a, 10.0).partial({1}), 0.0);
    expect_near(oti::fmin(a, 10.0).real(), 2.0);
    expect_near(oti::fmin(a, 10.0).partial({1}), 1.0);

    // copysign: magnitude of x with sign of y; derivative follows |x|.
    expect_near(oti::copysign(a, T(-1.0)).real(), -2.0);
    expect_near(oti::copysign(a, T(-1.0)).partial({1}), -1.0);
    expect_near(oti::copysign(a, 1.0).real(), 2.0);
    expect_near(oti::copysign(a, 1.0).partial({1}), 1.0);
}

void test_hypot_mod()
{
    using T = oti::otinum<2, 1>;
    T x = T::variable(0, 3.0);
    T y = T::variable(1, 4.0);

    T h = oti::hypot(x, y);
    expect_near(h.real(), 5.0);
    expect_near(h.partial({1, 0}), 3.0 / 5.0); // d/dx = x/hypot
    expect_near(h.partial({0, 1}), 4.0 / 5.0); // d/dy = y/hypot

    // fmod / remainder: value matches std, d/dx == 1 between jumps.
    T a = T::variable(0, 7.3);
    expect_near(oti::fmod(a, 2.0).real(), std::fmod(7.3, 2.0));
    expect_near(oti::fmod(a, 2.0).partial({1, 0}), 1.0);
    expect_near(oti::remainder(a, 2.0).real(), std::remainder(7.3, 2.0));
    expect_near(oti::remainder(a, 2.0).partial({1, 0}), 1.0);
}

void test_exp_log_variants()
{
    expect_near(oti::exp2(oti::otinum<1, 1>::variable(0, 3.0)).real(), 8.0);
    expect_near(oti::log2(oti::otinum<1, 1>::variable(0, 8.0)).real(), 3.0);

    // expm1/log1p keep full precision in the real part near zero.
    expect_near(oti::expm1(oti::otinum<1, 1>::variable(0, 1e-10)).real(), std::expm1(1e-10));
    expect_near(oti::log1p(oti::otinum<1, 1>::variable(0, 1e-10)).real(), std::log1p(1e-10));

    // Derivatives match finite differences.
    expect_near(oti_deriv([](auto z) { return oti::exp2(z); }, 1.3),
                central_diff([](double z) { return std::exp2(z); }, 1.3), 1e-5);
    expect_near(oti_deriv([](auto z) { return oti::log2(z); }, 1.3),
                central_diff([](double z) { return std::log2(z); }, 1.3), 1e-5);
    expect_near(oti_deriv([](auto z) { return oti::expm1(z); }, 0.7),
                central_diff([](double z) { return std::expm1(z); }, 0.7), 1e-5);
    expect_near(oti_deriv([](auto z) { return oti::log1p(z); }, 0.7),
                central_diff([](double z) { return std::log1p(z); }, 0.7), 1e-5);
}

void test_inverse_trig()
{
    double const x0 = 0.4;

    // Values match std.
    expect_near(oti::atan(oti::otinum<1, 1>::variable(0, x0)).real(), std::atan(x0));
    expect_near(oti::asin(oti::otinum<1, 1>::variable(0, x0)).real(), std::asin(x0));
    expect_near(oti::acos(oti::otinum<1, 1>::variable(0, x0)).real(), std::acos(x0));

    // First derivatives match the analytic forms.
    expect_near(oti_deriv([](auto z) { return oti::atan(z); }, x0), 1.0 / (1.0 + x0 * x0));
    expect_near(oti_deriv([](auto z) { return oti::asin(z); }, x0), 1.0 / std::sqrt(1.0 - x0 * x0));
    expect_near(oti_deriv([](auto z) { return oti::acos(z); }, x0), -1.0 / std::sqrt(1.0 - x0 * x0));

    // Second derivatives (order-2 jet) match finite differences of the first.
    {
        using T2 = oti::otinum<1, 2>;
        double const d2_atan = oti::atan(T2::variable(0, x0)).partial({2});
        double const fd = central_diff([](double z) { return 1.0 / (1.0 + z * z); }, x0);
        expect_near(d2_atan, fd, 1e-5);

        double const d2_asin = oti::asin(T2::variable(0, x0)).partial({2});
        double const fd_asin =
            central_diff([](double z) { return 1.0 / std::sqrt(1.0 - z * z); }, x0);
        expect_near(d2_asin, fd_asin, 1e-5);
    }

    // Out-of-domain: asin(2) has the real value of std::asin (NaN) and NaN
    // derivatives (the singular-point convention).
    {
        oti::otinum<1, 1> s = oti::asin(oti::otinum<1, 1>::variable(0, 2.0));
        assert(std::isnan(s.real()));
        assert(std::isnan(s[1]));
    }
}

void test_atan2()
{
    using T = oti::otinum<2, 1>;
    double const y0 = 1.0;
    double const x0 = 2.0;
    T y = T::variable(0, y0);
    T x = T::variable(1, x0);

    T a = oti::atan2(y, x);
    double const r2 = x0 * x0 + y0 * y0;
    expect_near(a.real(), std::atan2(y0, x0));
    expect_near(a.partial({1, 0}), x0 / r2);  // d/dy =  x / (x^2 + y^2)
    expect_near(a.partial({0, 1}), -y0 / r2); // d/dx = -y / (x^2 + y^2)

    // Quadrant correctness via the real-part override (x < 0).
    expect_near(oti::atan2(T(1.0), T(-1.0)).real(), std::atan2(1.0, -1.0));

    // Scalar overloads.
    expect_near(oti::atan2(y, x0).real(), std::atan2(y0, x0));
    expect_near(oti::atan2(y0, x).real(), std::atan2(y0, x0));
}

} // namespace

int main()
{
    test_numeric_limits();
    test_streaming();
    test_rounding();
    test_predicates();
    test_selection();
    test_hypot_mod();
    test_exp_log_variants();
    test_inverse_trig();
    test_atan2();
    return 0;
}
