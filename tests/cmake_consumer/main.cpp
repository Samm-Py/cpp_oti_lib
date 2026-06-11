#include <cmath>

#include <otinum/otinum.hpp>

int main()
{
    using T = oti::otinum<2, 2>;

    T x = T::variable(0, 1.5);
    T y = T::variable(1, 0.25);
    T result = oti::exp(x + y);

    double expected = std::exp(1.75);
    return std::abs(result.real() - expected) < 1e-12 ? 0 : 1;
}
