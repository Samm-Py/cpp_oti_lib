#include <cassert>
#include <iostream>
#include <type_traits>

#include "test_utils.hpp"

int main()
{
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

    std::cout << "comparison tests passed\n";
}
