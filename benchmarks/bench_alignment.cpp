// Alignment isolation benchmark: the library's conditional coefficient
// alignment, in isolation, on a memory-bound streaming kernel. The only thing
// that differs between the two variants is the alignas on the coefficient
// block; layout, size, and the kernel are identical.
//
//   natural : alignof(Coeff)
//   aligned : detail::otinum_alignment (the library's rule: 16/8 when it
//             divides the byte size, so sizeof never grows)
//
// metric = useful_gbps. The two variants move the same bytes, so a difference
// is pure load/store width. They compute the same thing -> checksums match.
//
// Requires OTI_ENABLE_KOKKOS.

#include <Kokkos_Core.hpp>

#include "otinum/otinum.hpp"

#include "bench_common.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace {

template <int NC, class Coeff, std::size_t Align>
struct alignas(Align) jet {
    Coeff c[NC];
};

template <int NC, class Coeff, std::size_t Align>
double time_stream(std::size_t n, int reps, double& checksum)
{
    using J = jet<NC, Coeff, Align>;
    Kokkos::View<J*> x("x", n);
    Kokkos::View<J*> y("y", n);
    Kokkos::parallel_for("init", n, KOKKOS_LAMBDA(std::size_t i) {
        for (int k = 0; k < NC; ++k) {
            x(i).c[k] = static_cast<Coeff>(0.25 + 0.01 * ((i + k) % 13));
            y(i).c[k] = static_cast<Coeff>(0.50 + 0.01 * ((i + 2 * k) % 11));
        }
    });
    Kokkos::fence();

    Coeff const a = static_cast<Coeff>(1.5);
    auto pass = [&] {
        Kokkos::parallel_for("stream", n, KOKKOS_LAMBDA(std::size_t i) {
            J xi = x(i);
            J yi = y(i);
            for (int k = 0; k < NC; ++k) yi.c[k] = a * xi.c[k] + yi.c[k];
            y(i) = yi;
        });
    };
    pass();
    Kokkos::fence();
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) pass();
    Kokkos::fence();
    double const secs = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - t0).count();

    double cs = 0;
    Kokkos::parallel_reduce("checksum", n, KOKKOS_LAMBDA(std::size_t i, double& acc) {
        for (int k = 0; k < NC; ++k) acc += static_cast<double>(y(i).c[k]);
    }, cs);
    checksum = cs;

    double const bytes = 3.0 * double(NC) * sizeof(Coeff) * double(n) * reps;
    return bytes / secs * 1e-9;
}

template <int M, int N, class Coeff>
void run_shape(char const* backend, int reps, int repetitions)
{
    using OT = oti::otinum<M, N, Coeff>;
    constexpr int NC = OT::ncoeffs;
    constexpr std::size_t A_nat = alignof(Coeff);
    constexpr std::size_t A_ali = oti::detail::otinum_alignment<Coeff, NC>();
    std::size_t const n = (96u << 20) / (NC * sizeof(Coeff));

    for (int rep = 1; rep <= repetitions; ++rep) {
        double cs_nat = 0, cs_ali = 0;
        double const nat = time_stream<NC, Coeff, A_nat>(n, reps, cs_nat);
        double const ali = time_stream<NC, Coeff, A_ali>(n, reps, cs_ali);
        bench::print_row(backend, bench::coeff_name<Coeff>(), M, N, NC,
                         OT::table_type::nproducts, "stream", "natural", rep,
                         "useful_gbps", nat, cs_nat);
        bench::print_row(backend, bench::coeff_name<Coeff>(), M, N, NC,
                         OT::table_type::nproducts, "stream", "aligned", rep,
                         "useful_gbps", ali, cs_ali);
    }
}

template <int M, int N>
void run_both(char const* backend, int reps, int repetitions)
{
    run_shape<M, N, double>(backend, reps, repetitions);
    run_shape<M, N, float>(backend, reps, repetitions);
}

} // namespace

int main(int argc, char** argv)
{
    Kokkos::initialize(argc, argv);
    {
        int const reps = (argc > 1) ? std::atoi(argv[1]) : 30;
        int const repetitions = (argc > 2) ? std::atoi(argv[2]) : 3;
        char const* backend = Kokkos::DefaultExecutionSpace::name();
        bench::print_header();

        run_both<3, 1>(backend, reps, repetitions);
        run_both<2, 2>(backend, reps, repetitions);
        run_both<3, 2>(backend, reps, repetitions);
        run_both<3, 3>(backend, reps, repetitions);
        run_both<4, 3>(backend, reps, repetitions);
        run_both<4, 4>(backend, reps, repetitions);
    }
    Kokkos::finalize();
    return 0;
}
