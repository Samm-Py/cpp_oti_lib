#include <cstddef>
#include <iostream>
#include <vector>

#include "otinum/otinum.hpp"

int main()
{
    using T = oti::otinum<2, 2>;
    using Span = oti::soa_span<2, 2>;

    std::size_t const n = 4;
    std::vector<double> buffer(Span::required_size(n), 0.0);
    Span values(buffer.data(), n);

    // Evaluate f(x, y) = x*x + 3*y at (x, y) = (i, 10) for each element and
    // scatter the resulting jet into the coefficient-major buffer.
    for (std::size_t i = 0; i < n; ++i) {
        T x = T::variable(0, static_cast<double>(i));
        T y = T::variable(1, 10.0);
        values.store(i, x * x + 3.0 * y);
    }

    // The buffer is coefficient-major: the real parts of all four elements
    // are contiguous, then all four df/dx coefficients, and so on.
    std::cout << "real parts (buffer[0..3]):  ";
    for (std::size_t i = 0; i < n; ++i) {
        std::cout << buffer[i] << " ";
    }
    std::cout << "\ndf/dx parts (buffer[4..7]): ";
    for (std::size_t i = 0; i < n; ++i) {
        std::cout << buffer[n + i] << " ";
    }
    std::cout << "\n";

    // Gather one element back into an ordinary otinum and use it as usual.
    T f = values.load(3);
    std::cout << "element 3: f = " << f.real()
              << ", df/dx = " << f.partial({1, 0})
              << ", d2f/dx2 = " << f.partial({2, 0}) << "\n";

    return 0;
}
