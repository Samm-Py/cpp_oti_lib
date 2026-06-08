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
    expect_near(from.coeff(oti::sparse({{0, 2}, {1, 1}})), from.coeff({2, 1}));
    expect_near(from.partial(oti::sparse({{0, 2}, {1, 1}})), from.partial({2, 1}));
    expect_near(from.partial(oti::sparse({{0, 1}, {0, 1}, {1, 1}})), from.partial({2, 1}));
    expect_near(from.coeff({4, 0}), 0.0);
    expect_near(from.partial({4, 0}), 0.0);
    expect_near(from.partial(oti::sparse({{0, 4}})), 0.0);
    expect_near(from.partial(oti::sparse({{7, 1}})), 0.0);

    T manual;
    manual.set_coeff({1, 0}, 3.5);
    expect_near(manual.coeff({1, 0}), 3.5);
    expect_near(manual.partial({1, 0}), 3.5);

    manual.set_coeff(oti::sparse({{0, 1}, {1, 1}}), 2.25);
    expect_near(manual.coeff({1, 1}), 2.25);
    expect_near(manual.partial({1, 1}), 2.25);

    manual.set_partial({2, 0}, 8.0);
    expect_near(manual.coeff({2, 0}), 4.0);
    expect_near(manual.partial({2, 0}), 8.0);

    manual.set_partial({2, 1}, 12.0);
    expect_near(manual.coeff({2, 1}), 6.0);
    expect_near(manual.partial({2, 1}), 12.0);

    manual.set_partial(oti::sparse({{1, 3}}), 18.0);
    expect_near(manual.coeff({0, 3}), 3.0);
    expect_near(manual.partial({0, 3}), 18.0);

    manual.set_coeff({4, 0}, 99.0);
    manual.set_partial({4, 0}, 99.0);
    manual.set_coeff(oti::sparse({{9, 1}}), 99.0);
    manual.set_partial(oti::sparse({{9, 1}}), 99.0);
    expect_near(manual.coeff({4, 0}), 0.0);
    expect_near(manual.partial({4, 0}), 0.0);

    using Large = oti::otinum<10, 2>;
    Large a = Large::variable(7, 1.5);
    Large b = Large::variable(9, 2.0);
    Large product = a * b;
    expect_near(product.partial(oti::sparse({{7, 1}})), 2.0);
    expect_near(product.partial(oti::sparse({{9, 1}})), 1.5);
    expect_near(product.partial(oti::sparse({{7, 1}, {9, 1}})), 1.0);

    using ConstantOnly = oti::otinum<2, 0>;
    ConstantOnly c = ConstantOnly::variable(1, 7.0);
    expect_near(c.real(), 7.0);
    expect_near(c.partial({0, 1}), 0.0);

    c.set_coeff({0, 0}, 2.0);
    c.set_partial({0, 1}, 5.0);
    expect_near(c.real(), 2.0);
    expect_near(c.partial({0, 1}), 0.0);

    std::cout << "construction and access tests passed\n";
}
