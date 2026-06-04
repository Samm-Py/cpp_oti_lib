#include <array>
#include <iostream>

#include "test_utils.hpp"

int main()
{
    using oti_test::expect_near;
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

    std::cout << "construction and access tests passed\n";
}
