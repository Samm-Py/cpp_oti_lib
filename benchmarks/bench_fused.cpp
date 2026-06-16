// Fused-op isolation benchmark: the fused accumulation helpers vs the
// equivalent operator chains, in isolation, on a compute-bound register-
// resident kernel. Both compute the same value; the fused form avoids the
// intermediate otinum temporaries the operator chain materializes.
//
//   axpy : chain  t = a*x + t   vs  oti::axpy(t, a, x)
//   fma  : chain  t = t + x*y   vs  oti::fma_into(t, x, y)
//
// metric = ns_per_op (lower is better). variant = chain | fused. The two forms
// are arithmetically identical, so the checksums match.
//
// Requires OTI_ENABLE_KOKKOS.

#include <Kokkos_Core.hpp>

#include "otinum/otinum.hpp"

#include "bench_common.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace {

enum Pattern { AXPY, FMA };
enum Mode { CHAIN, FUSED };
char const* pattern_name(Pattern p) { return p == AXPY ? "axpy" : "fma"; }

struct measurement {
    double ns_per_op;
    double checksum;
};

template <class T, Pattern pat, Mode mode>
measurement run_once(int n_elem, int reps)
{
    using Coeff = typename T::coeff_type;
    Kokkos::View<double*> out("out", n_elem);
    Kokkos::fence();
    auto t0 = std::chrono::steady_clock::now();
    Kokkos::parallel_for("fused", n_elem, KOKKOS_LAMBDA(int i) {
        T x, y;
        for (int k = 0; k < T::ncoeffs; ++k) {
            x[k] = static_cast<Coeff>(0.30 + 0.05 * ((i + k) % 5));
            y[k] = static_cast<Coeff>(0.20 + 0.04 * ((i + 2 * k) % 6));
        }
        Coeff const a = static_cast<Coeff>(1.5);
        T acc{};
        for (int r = 0; r < reps; ++r) {
            x[0] += static_cast<Coeff>(1e-9);  // perturb to defeat hoisting/CSE
            T t = y;
            if constexpr (pat == AXPY) {
                if constexpr (mode == CHAIN) {
                    t = a * x + t;
                } else {
                    oti::axpy(t, a, x);
                }
            } else {
                if constexpr (mode == CHAIN) {
                    t = t + x * y;
                } else {
                    oti::fma_into(t, x, y);
                }
            }
            acc += t;
        }
        double s = 0;
        for (int k = 0; k < T::ncoeffs; ++k) s += static_cast<double>(acc[k]);
        out(i) = s;
    });
    Kokkos::fence();
    double const secs = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - t0).count();
    auto host = Kokkos::create_mirror_view(out);
    Kokkos::deep_copy(host, out);
    double total = 0;
    for (int i = 0; i < n_elem; ++i) total += host(i);
    return {secs * 1e9 / (double(n_elem) * reps), total};
}

template <class T, Pattern pat>
int calibrate(int n_elem, double target_ms)
{
    int reps = 1;
    for (;;) {
        auto t0 = std::chrono::steady_clock::now();
        (void)run_once<T, pat, FUSED>(n_elem, reps);
        double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
        if (ms >= target_ms || reps >= (1 << 24)) break;
        int grow = static_cast<int>(target_ms / (ms > 0.05 ? ms : 0.05)) + 1;
        reps *= (grow < 2 ? 2 : grow > 16 ? 16 : grow);
    }
    return reps;
}

template <int M, int N, class Coeff, Pattern pat>
void sweep(char const* backend, int n_elem, double target_ms, int repetitions)
{
    using T = oti::otinum<M, N, Coeff>;
    int const reps = calibrate<T, pat>(n_elem, target_ms);
    for (int rep = 1; rep <= repetitions; ++rep) {
        measurement c = run_once<T, pat, CHAIN>(n_elem, reps);
        measurement f = run_once<T, pat, FUSED>(n_elem, reps);
        bench::print_row(backend, bench::coeff_name<Coeff>(), M, N, T::ncoeffs,
                         T::table_type::nproducts, pattern_name(pat), "chain", rep,
                         "ns_per_op", c.ns_per_op, c.checksum);
        bench::print_row(backend, bench::coeff_name<Coeff>(), M, N, T::ncoeffs,
                         T::table_type::nproducts, pattern_name(pat), "fused", rep,
                         "ns_per_op", f.ns_per_op, f.checksum);
    }
}

template <int M, int N>
void run_shape(char const* backend, int n_elem, double target_ms, int reps)
{
    sweep<M, N, double, AXPY>(backend, n_elem, target_ms, reps);
    sweep<M, N, double, FMA>(backend, n_elem, target_ms, reps);
    sweep<M, N, float, AXPY>(backend, n_elem, target_ms, reps);
    sweep<M, N, float, FMA>(backend, n_elem, target_ms, reps);
}

} // namespace

int main(int argc, char** argv)
{
    Kokkos::initialize(argc, argv);
    {
        int const n_elem = (argc > 1) ? std::atoi(argv[1]) : 16384;
        int const repetitions = (argc > 2) ? std::atoi(argv[2]) : 3;
        double const target_ms = (argc > 3) ? std::atof(argv[3]) : 25.0;
        char const* backend = Kokkos::DefaultExecutionSpace::name();
        bench::print_header();

        run_shape<3, 1>(backend, n_elem, target_ms, repetitions);
        run_shape<2, 2>(backend, n_elem, target_ms, repetitions);
        run_shape<3, 2>(backend, n_elem, target_ms, repetitions);
        run_shape<3, 3>(backend, n_elem, target_ms, repetitions);
        run_shape<4, 3>(backend, n_elem, target_ms, repetitions);
        run_shape<4, 4>(backend, n_elem, target_ms, repetitions);
    }
    Kokkos::finalize();
    return 0;
}
