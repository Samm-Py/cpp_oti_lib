#include <iostream>

#include "otinum/otinum.hpp"

int main()
{
    using T = oti::otinum<2, 2>;

    T x = T::variable(0, 1.5);
    T y = T::variable(1, 0.3);
    T f = oti::sin(x * y) + oti::exp(x);

    std::cout << "f = " << f.real() << '\n';
    std::cout << "df/dx = " << f.partial({1, 0}) << '\n';
    std::cout << "df/dy = " << f.partial({0, 1}) << '\n';
    std::cout << "d2f/dxdy = " << f.partial({1, 1}) << '\n';
}
