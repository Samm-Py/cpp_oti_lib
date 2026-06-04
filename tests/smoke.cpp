#include <array>
#include <cassert>
#include <cmath>
#include <iostream>

#include "otinum/otinum.hpp"

namespace {

constexpr double tol = 1e-11;

template <typename T>
void expect_near(T actual, T expected, double tolerance = tol)
{
    assert(std::abs(actual - expected) <= tolerance);
}

template <int M, int N>
void expect_all_near(oti::otinum<M, N> const& actual,
                     oti::otinum<M, N> const& expected,
                     double tolerance = tol)
{
    for (int i = 0; i < oti::otinum<M, N>::ncoeffs; ++i) {
        expect_near(actual[i], expected[i], tolerance);
    }
}

void test_layout_and_tables()
{
    using T22 = oti::otinum<2, 2>;
    using Tables22 = oti::detail::tables<2, 2>;

    static_assert(T22::ncoeffs == 6, "unexpected coefficient count");
    static_assert(oti::detail::binom(5, 2) == 10, "bad binomial");
    static_assert(oti::detail::rank<2, 2>({0, 0}) == 0, "bad rank");
    static_assert(oti::detail::rank<2, 2>({1, 0}) == 1, "bad rank");
    static_assert(oti::detail::rank<2, 2>({0, 1}) == 2, "bad rank");
    static_assert(oti::detail::rank<2, 2>({2, 0}) == 3, "bad rank");
    static_assert(oti::detail::rank<2, 2>({1, 1}) == 4, "bad rank");
    static_assert(oti::detail::rank<2, 2>({0, 2}) == 5, "bad rank");
    static_assert(oti::detail::rank<2, 2>({3, 0}) == -1, "bad out-of-range rank");
    static_assert(Tables22::order_offset[0] == 0, "bad order offset");
    static_assert(Tables22::order_offset[1] == 1, "bad order offset");
    static_assert(Tables22::order_offset[2] == 3, "bad order offset");
    static_assert(Tables22::order_offset[3] == 6, "bad order offset");

    static_assert(oti::otinum<3, 3>::ncoeffs == 20, "unexpected K(3,3)");
    static_assert(oti::otinum<10, 2>::ncoeffs == 66, "unexpected K(10,2)");

    for (int i = 0; i < T22::ncoeffs; ++i) {
        int ranked = oti::detail::rank<2, 2>(Tables22::idx_to_alpha[static_cast<std::size_t>(i)]);
        assert(ranked == i);
    }
}

void test_construction_and_access()
{
    using T = oti::otinum<2, 3>;

    T zero;
    for (double coeff : zero.data()) {
        expect_near(coeff, 0.0);
    }

    T real(4.25);
    expect_near(real.real(), 4.25);
    for (int i = 1; i < T::ncoeffs; ++i) {
        expect_near(real[i], 0.0);
    }

    T x = T::variable(0, 1.5);
    expect_near(x.real(), 1.5);
    expect_near(x.partial({1, 0}), 1.0);
    expect_near(x.partial({0, 1}), 0.0);

    std::array<double, T::ncoeffs> coeffs{};
    for (int i = 0; i < T::ncoeffs; ++i) {
        coeffs[static_cast<std::size_t>(i)] = static_cast<double>(i) / 10.0;
    }

    T from = T::from_coeffs(coeffs);
    for (int i = 0; i < T::ncoeffs; ++i) {
        expect_near(from[i], coeffs[static_cast<std::size_t>(i)]);
    }

    expect_near(from.coeff({2, 1}), from[oti::detail::rank<2, 3>({2, 1})]);
    expect_near(from.partial({2, 1}), 2.0 * from.coeff({2, 1}));
    expect_near(from.coeff({4, 0}), 0.0);
    expect_near(from.partial({4, 0}), 0.0);

    using ConstantOnly = oti::otinum<2, 0>;
    ConstantOnly c = ConstantOnly::variable(1, 7.0);
    expect_near(c.real(), 7.0);
    expect_near(c.partial({0, 1}), 0.0);
}

void test_linear_arithmetic()
{
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
}

void test_multiplication_and_division()
{
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
}

void test_truncated_operations_and_gem()
{
    using T = oti::otinum<2, 3>;
    T x = T::variable(0, 2.0);
    T y = T::variable(1, 3.0);

    T full = x * y;
    T trunc1 = oti::trunc_mul(x, y, 1);
    expect_near(trunc1.real(), full.real());
    expect_near(trunc1.partial({1, 0}), full.partial({1, 0}));
    expect_near(trunc1.partial({0, 1}), full.partial({0, 1}));
    expect_near(trunc1.partial({1, 1}), 0.0);

    T trunc_add = oti::trunc_add(full, T(1.0), 0);
    expect_near(trunc_add.real(), full.real() + 1.0);
    expect_near(trunc_add.partial({1, 0}), 0.0);

    T negative_cutoff = oti::trunc_mul(x, y, -1);
    expect_all_near(negative_cutoff, T());

    expect_all_near(oti::gem(x, y, T(5.0)), full + 5.0);
}

void test_exp_log_pow_sqrt()
{
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
}

void test_trigonometric_and_hyperbolic()
{
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
}

void test_abs_and_large_shapes()
{
    using T = oti::otinum<2, 2>;
    T positive = T::variable(0, 1.0);
    T negative = T::variable(0, -1.0);
    expect_all_near(oti::abs(positive), positive);
    expect_all_near(oti::abs(negative), -negative);

    using Large = oti::otinum<5, 3>;
    Large x0 = Large::variable(0, 1.0);
    Large x4 = Large::variable(4, 2.0);
    Large product = x0 * x4;
    expect_near(product.real(), 2.0);
    expect_near(product.partial({1, 0, 0, 0, 0}), 2.0);
    expect_near(product.partial({0, 0, 0, 0, 1}), 1.0);
    expect_near(product.partial({1, 0, 0, 0, 1}), 1.0);
}

} // namespace

int main()
{
    test_layout_and_tables();
    test_construction_and_access();
    test_linear_arithmetic();
    test_multiplication_and_division();
    test_truncated_operations_and_gem();
    test_exp_log_pow_sqrt();
    test_trigonometric_and_hyperbolic();
    test_abs_and_large_shapes();

    std::cout << "all otinum smoke tests passed\n";
    return 0;
}
