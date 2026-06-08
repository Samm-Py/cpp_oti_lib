#include <array>
#include <iostream>
#include <type_traits>

#include "test_utils.hpp"

int main()
{
    using oti_test::expect_near;
    using T = oti::otinum<2, 3, float>;

    static_assert(std::is_same<T::coeff_type, float>::value, "coefficient type should be float");
    static_assert(std::is_same<decltype(T{}.real()), float>::value, "real() should return float");
    static_assert(sizeof(T::coeff_type) == sizeof(float), "coefficient storage should use float");

    T x = T::variable(0, 1.25f);
    T y = T::variable(1, 0.5f);
    T f = oti::sin(x) + oti::pow(y + 2.0, 2.0) + 3.0 * x * y;

    expect_near(f.real(), std::sin(1.25f) + (2.5f * 2.5f) + (3.0f * 1.25f * 0.5f), 2e-6);
    expect_near(f.partial({1, 0}), std::cos(1.25f) + 3.0f * 0.5f, 2e-6);
    expect_near(f.partial({0, 1}), 2.0f * 2.5f + 3.0f * 1.25f, 2e-6);
    expect_near(f.partial({1, 1}), 3.0f, 2e-6);

    std::array<float, T::ncoeffs> coeffs{};
    for (int i = 0; i < T::ncoeffs; ++i) {
        coeffs[static_cast<std::size_t>(i)] = static_cast<float>(i) / 8.0f;
    }

    T from = T::from_coeffs(coeffs);
    expect_near(from[3], coeffs[3], 0.0);

    std::cout << "float coefficient tests passed\n";
}
