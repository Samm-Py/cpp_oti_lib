#include <iostream>

#include "test_utils.hpp"

int main()
{
    using oti_test::expect_all_near;
    using oti_test::expect_near;
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

    std::cout << "truncated operation tests passed\n";
}
