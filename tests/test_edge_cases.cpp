#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

#include "test_utils.hpp"

int main()
{
    using oti_test::expect_all_near;
    using oti_test::expect_near;

    {
        using T0 = oti::otinum<2, 0>;
        T0 a = T0::variable(1, 6.0);
        T0 b(3.0);

        expect_near(a.real(), 6.0);
        expect_near(a.partial({0, 1}), 0.0);
        expect_near((a + b).real(), 9.0);
        expect_near((a - b).real(), 3.0);
        expect_near((a * b).real(), 18.0);
        expect_near((a / b).real(), 2.0);
        expect_near(oti::inv(b).real(), 1.0 / 3.0);
        expect_near(oti::exp(b).real(), std::exp(3.0));
    }

    {
        using T = oti::otinum<2, 2>;
        T x = T::variable(0, 2.0);
        T y = T::variable(1, 3.0);
        T full_product = x * y;
        T full_sum = x + y;

        expect_all_near(oti::trunc_mul(x, y, -1), T{});
        expect_all_near(oti::trunc_add(x, y, -1), T{});
        expect_all_near(oti::trunc_mul(x, y, 10), full_product);
        expect_all_near(oti::trunc_add(x, y, 10), full_sum);
    }

    {
        using T = oti::otinum<2, 3>;
        T value = T::variable(0, 1.0);
        value.set_partial({1, 1}, 5.0);

        expect_near(value.coeff(oti::sparse({{-1, 1}})), 0.0);
        expect_near(value.partial(oti::sparse({{0, -1}})), 0.0);

        value.set_coeff(oti::sparse({{-1, 1}}), 99.0);
        value.set_partial(oti::sparse({{0, -1}}), 99.0);
        expect_near(value.coeff({1, 1}), 5.0);
        expect_near(value.partial({1, 1}), 5.0);
    }

    {
        using T = oti::otinum<1, 2>;
        double const nan = std::numeric_limits<double>::quiet_NaN();

        T nan_value(nan);
        nan_value.set_partial({1}, 2.0);
        T abs_nan = oti::abs(nan_value);
        assert(std::isnan(abs_nan.real()));
        expect_near(abs_nan.partial({1}), 2.0);

        T cbrt_zero = oti::cbrt(T::variable(0, 0.0));
        expect_near(cbrt_zero.real(), 0.0);
        assert(std::isnan(cbrt_zero.partial({1})));
        assert(std::isnan(cbrt_zero.partial({2})));

        T invalid_pow = oti::pow(T::variable(0, -1.0), 0.5);
        assert(std::isnan(invalid_pow.real()));
        assert(std::isnan(invalid_pow.partial({1})));
        assert(std::isnan(invalid_pow.partial({2})));
    }

    {
        using Tables = oti::detail::tables<2, 2>;

        static_assert(oti::detail::binom(-1, 0) == 0, "negative n should be invalid");
        static_assert(oti::detail::binom(3, -1) == 0, "negative k should be invalid");
        static_assert(oti::detail::binom(2, 3) == 0, "k > n should be invalid");
        static_assert(oti::detail::composition_count(-1, 0) == 0, "negative parts invalid");
        static_assert(oti::detail::composition_count(0, 0) == 1, "empty zero composition");
        static_assert(oti::detail::composition_count(0, 1) == 0, "empty positive composition");
        static_assert(oti::detail::total_order<2>({2, 3}) == 5, "bad total order");
        static_assert(Tables::alpha_at(4)[0] == 1 && Tables::alpha_at(4)[1] == 1,
                      "bad alpha lookup");
        static_assert(Tables::factorial_alpha_value(3) == 2.0, "bad alpha factorial lookup");
        static_assert(Tables::product_count_by_output_value(4) == 4, "bad product count");
    }

    std::cout << "edge case tests passed\n";
}
