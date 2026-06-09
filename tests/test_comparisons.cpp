#include <cassert>
#include <iostream>
#include <type_traits>

#include "test_utils.hpp"

namespace {

template <class Scalar>
Scalar branch_function(Scalar x, Scalar threshold)
{
    if (x >= threshold) {
        return x * x + Scalar(2.0) * x;
    }
    return x * x * x - Scalar(2.0) * x;
}

double finite_difference_branch(double x, double threshold)
{
    double const h = 1.0e-6;
    return (branch_function(x + h, threshold) - branch_function(x - h, threshold)) / (2.0 * h);
}

} // namespace

int main()
{
    using oti_test::expect_near;

    {
        using T = oti::otinum<2, 2>;

        T x = T::variable(0, 2.0);
        T y = T::variable(1, 3.0);
        T same_real_different_derivative = T::variable(1, 2.0);

        assert(x < y);
        assert(x <= y);
        assert(y > x);
        assert(y >= x);
        assert(x != y);

        assert(x == same_real_different_derivative);
        assert(x <= same_real_different_derivative);
        assert(x >= same_real_different_derivative);
        assert(!(x < same_real_different_derivative));
        assert(!(x > same_real_different_derivative));

        assert(x == 2.0);
        assert(2.0 == x);
        assert(x != 3.0);
        assert(3.0 != x);

        assert(x < 3.0);
        assert(1.0 < x);
        assert(x <= 2.0);
        assert(2.0 <= x);
        assert(y > 2.0);
        assert(4.0 > y);
        assert(y >= 3.0);
        assert(3.0 >= y);

        static_assert(std::is_same<decltype(x >= y), bool>::value,
                      "comparisons should return bool");
    }

    {
        using F = oti::otinum<1, 1, float>;

        F value = F::variable(0, 1.25f);
        F threshold(1.5f);

        assert(value < threshold);
        assert(value <= 1.25f);
        assert(1.25f >= value);
        assert(value == 1.25f);
        assert(value != 1.0f);
    }

    {
        constexpr oti::otinum<1, 0> a(4.0);
        constexpr oti::otinum<1, 0> b(5.0);

        static_assert(a < b, "constexpr less-than should use real values");
        static_assert(a <= 4.0, "constexpr mixed comparison should use real values");
        static_assert(4.0 >= a, "constexpr mixed comparison should use real values");
        static_assert(a != b, "constexpr equality should use real values");
    }

    {
        using T = oti::otinum<1, 3>;

        double const threshold = 1.5;
        T below = T::variable(0, 1.0);
        T above = T::variable(0, 2.0);
        T at_threshold = T::variable(0, threshold);

        T below_result = branch_function(below, T(threshold));
        T above_result = branch_function(above, T(threshold));
        T threshold_result = branch_function(at_threshold, T(threshold));

        expect_near(below_result.real(), branch_function(1.0, threshold));
        expect_near(above_result.real(), branch_function(2.0, threshold));
        expect_near(threshold_result.real(), branch_function(threshold, threshold));

        expect_near(below_result.partial({1}), finite_difference_branch(1.0, threshold), 1.0e-9);
        expect_near(above_result.partial({1}), finite_difference_branch(2.0, threshold), 1.0e-9);

        // The discontinuity is not differentiable. The comparison still has a
        // deterministic branch policy: equality takes the >= branch.
        expect_near(threshold_result.partial({1}), 2.0 * threshold + 2.0);
    }

    std::cout << "comparison tests passed\n";
}
