// Device smoke test for otinum/validity.hpp: calls every validity primitive from
// inside a Kokkos kernel on a live jet, proving they are genuinely
// device-callable (OTI_CONSTEXPR_FUNCTION -> KOKKOS_FORCEINLINE_FUNCTION,
// allocation-free, Kokkos::Array I/O), then checks the results host-side against
// the same hand-computed values as the host unit test.

#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <Kokkos_Core.hpp>

#include "otinum/otinum.hpp"
#include "otinum/validity.hpp"

namespace {

using J = oti::otinum<2, 2, double>;
namespace v = oti::validity;

void expect_near(double actual, double expected, double tol = 1e-12)
{
    if (std::abs(actual - expected) <= tol * (1.0 + std::abs(expected))) return;
    std::ostringstream m;
    m << "device validity mismatch: actual=" << actual << " expected=" << expected;
    throw std::runtime_error(m.str());
}

} // namespace

int main(int argc, char** argv)
{
    Kokkos::initialize(argc, argv);
    int rc = 0;
    {
        constexpr int NR = 10;
        Kokkos::View<double*> out("validity_out", NR);

        Kokkos::parallel_for(
            "validity_device", 1, KOKKOS_LAMBDA(int) {
                J jet{};
                jet.set_coeff(Kokkos::Array<int, 2>{{0, 0}}, 2.0);
                jet.set_coeff(Kokkos::Array<int, 2>{{1, 0}}, 3.0);
                jet.set_coeff(Kokkos::Array<int, 2>{{0, 1}}, -1.0);
                jet.set_coeff(Kokkos::Array<int, 2>{{2, 0}}, 0.5);
                jet.set_coeff(Kokkos::Array<int, 2>{{1, 1}}, 0.25);
                jet.set_coeff(Kokkos::Array<int, 2>{{0, 2}}, 2.0);

                Kokkos::Array<double, 2> h{{0.1, -0.2}};

                out(0) = v::evaluate(jet, h);              // 2.5  (linear)
                out(1) = v::evaluate(jet, h, 2);           // 2.58 (full quadratic)
                out(2) = v::truncation_error(jet, h);      // 0.08
                out(3) = v::is_trusted(jet, h, 0.01) ? 1.0 : 0.0;  // 0
                out(4) = v::is_trusted(jet, h, 0.05) ? 1.0 : 0.0;  // 1

                auto r = v::validity_radius(jet, 0.01);
                out(5) = r[0];                              // 0.2
                out(6) = r[1];                              // 0.1

                auto g = v::error_sensitivity(jet, h);
                out(7) = g[0];                              // 0.05
                out(8) = g[1];                              // -0.775

                J flat = jet;
                flat.set_coeff(Kokkos::Array<int, 2>{{2, 0}}, 0.0);
                out(9) = v::validity_radius(flat, 0.01)[0]; // +inf
            });
        Kokkos::fence();

        auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
        try {
            expect_near(host(0), 2.5);
            expect_near(host(1), 2.58);
            expect_near(host(2), 0.08);
            expect_near(host(3), 0.0);
            expect_near(host(4), 1.0);
            expect_near(host(5), 0.2);
            expect_near(host(6), 0.1);
            expect_near(host(7), 0.05);
            expect_near(host(8), -0.775);
            if (!std::isinf(host(9))) throw std::runtime_error("expected infinite reach");
            std::cout << "device validity tests passed\n";
        } catch (std::exception const& e) {
            std::cerr << e.what() << "\n";
            rc = 1;
        }

        // Multi-band fold on device: a <1,3> jet certifying the LINEAR model, so
        // truncation_error / error_sensitivity / validity_radius fold orders 2 AND
        // 3 -- and validity_radius runs its bracket-and-bisection inside the kernel.
        {
            using J3 = oti::otinum<1, 3, double>;
            Kokkos::View<double*> out3("validity_out3", 3);
            Kokkos::parallel_for(
                "validity_device_multiband", 1, KOKKOS_LAMBDA(int) {
                    J3 g{};
                    g.set_coeff(Kokkos::Array<int, 1>{{0}}, 2.0);
                    g.set_coeff(Kokkos::Array<int, 1>{{1}}, 0.0);
                    g.set_coeff(Kokkos::Array<int, 1>{{2}}, 1.0);
                    g.set_coeff(Kokkos::Array<int, 1>{{3}}, 0.5);
                    Kokkos::Array<double, 1> h3{{0.1}};
                    out3(0) = v::truncation_error(g, h3, 1);      // c2 h^2 + c3 h^3 = 0.0105
                    out3(1) = v::error_sensitivity(g, h3, 1)[0];  // 2 c2 h + 3 c3 h^2 = 0.215
                    out3(2) = v::validity_radius(g, 0.005, 0.0, 1)[0]; // root of |c2 r^2 + c3 r^3| = 0.01
                });
            Kokkos::fence();
            auto h3 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out3);
            try {
                expect_near(h3(0), 0.0105);
                expect_near(h3(1), 0.215);
                double const rr = h3(2);
                expect_near(rr * rr + 0.5 * rr * rr * rr, 0.01, 1e-6);  // bisection residual
                if (!(rr < 0.1)) throw std::runtime_error("multi-band reach should be < 0.1");
                std::cout << "device multi-band validity tests passed (reach=" << rr << ")\n";
            } catch (std::exception const& e) {
                std::cerr << e.what() << "\n";
                rc = 1;
            }
        }
    }
    Kokkos::finalize();
    return rc;
}
