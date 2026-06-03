#include <cassert>
#include <array>
#include <cmath>
#include <iostream>

#include <Kokkos_Core.hpp>

#include "otinum/otinum.hpp"

namespace {

constexpr double tol = 1e-10;

void expect_near(double actual, double expected, double tolerance = tol)
{
    assert(std::abs(actual - expected) <= tolerance);
}

void run_kokkos_smoke()
{
    using T = oti::otinum<2, 3>;
    constexpr int nchecks = 9;

    Kokkos::View<double**> device_values("otinum_kokkos_smoke_values", nchecks, T::ncoeffs);

    Kokkos::parallel_for(
        "otinum_kokkos_smoke",
        1,
        KOKKOS_LAMBDA(int) {
            T x = T::variable(0, 1.5);
            T y = T::variable(1, 0.25);
            T z = x + 0.5 * y;

            T checks[nchecks] = {
                x + y,
                x * y,
                (x * y) / y,
                oti::exp(z),
                oti::log(oti::exp(z)),
                oti::pow(z, 2.0),
                oti::sin(z) * oti::sin(z) + oti::cos(z) * oti::cos(z),
                oti::trunc_mul(x, y, 1),
                oti::gem(x, y, T(3.0)),
            };

            for (int row = 0; row < nchecks; ++row) {
                for (int i = 0; i < T::ncoeffs; ++i) {
                    device_values(row, i) = checks[row][i];
                }
            }
        });
    Kokkos::fence();

    auto host_values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), device_values);

    T x = T::variable(0, 1.5);
    T y = T::variable(1, 0.25);
    T z = x + 0.5 * y;
    std::array<T, nchecks> expected = {
        x + y,
        x * y,
        (x * y) / y,
        oti::exp(z),
        oti::log(oti::exp(z)),
        oti::pow(z, 2.0),
        oti::sin(z) * oti::sin(z) + oti::cos(z) * oti::cos(z),
        oti::trunc_mul(x, y, 1),
        oti::gem(x, y, T(3.0)),
    };

    for (int row = 0; row < nchecks; ++row) {
        for (int i = 0; i < T::ncoeffs; ++i) {
            expect_near(host_values(row, i), expected[static_cast<std::size_t>(row)][i]);
        }
    }

    expect_near(host_values(6, 0), 1.0);
    expect_near(host_values(7, 4), 0.0);
}

} // namespace

int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);
    {
        run_kokkos_smoke();
    }
    Kokkos::finalize();

    std::cout << "Kokkos otinum kernel tests passed\n";
    return 0;
}
