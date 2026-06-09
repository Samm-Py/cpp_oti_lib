#include <iostream>
#include <type_traits>

#include "otinum/otinum.hpp"

int main()
{
    using T = oti::otinum<2, 3, float>;
    static_assert(std::is_same<T::coeff_type, float>::value);

    T x = T::variable(0, 1.25f);
    T y = T::variable(1, 0.5f);
    T f = oti::sin(x) + oti::pow(y + 2.0f, 2.0f) + 3.0f * x * y;

    std::cout << f.real() << '\n';
    std::cout << f.partial({1, 0}) << '\n';
    std::cout << f.partial({0, 1}) << '\n';
}
