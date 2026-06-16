// Arithmetic isolation benchmark: the product-table and compile-time-unrolling
// optimizations, in isolation, on a compute-bound register-resident kernel.
//
// The arithmetic implementation is a compile-time switch over the whole otinum
// (OTI_BENCHMARK_ARITHMETIC_PATH): 0 = naive nested-loop multi-index rank
// rebuild, 1 = runtime product-table walk, 2 = compile-time-unrolled fold. Build
// this source once per path (the CMake makes bench_arithmetic_{naive,lookup,
// unrolled}); the runner concatenates their rows. Each thread runs many ops in
// registers, so the reported ns/op reflects the arithmetic itself, not memory.
//
// metric = ns_per_op (lower is better). variant comes from the build's path.
//
// Requires OTI_ENABLE_KOKKOS.

#include <Kokkos_Core.hpp>

#include "otinum/otinum.hpp"

#include "bench_common.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>

#ifndef OTI_BENCHMARK_ARITHMETIC_PATH
#define OTI_BENCHMARK_ARITHMETIC_PATH 2
#endif

#if OTI_BENCHMARK_ARITHMETIC_PATH == 0
#define BENCH_VARIANT "naive"
#elif OTI_BENCHMARK_ARITHMETIC_PATH == 1
#define BENCH_VARIANT "lookup"
#else
#define BENCH_VARIANT "unrolled"
#endif

namespace {

enum Op { MUL, DIV, FUNC };
char const* op_name(Op op) { return op == MUL ? "mul" : op == DIV ? "div" : "func"; }

struct measurement {
    double ns_per_op;
    double checksum;
};

template <class T, Op op>
measurement run_once(int n_elem, int reps)
{
    using Coeff = typename T::coeff_type;
    Kokkos::View<double*> out("out", n_elem);
    Kokkos::fence();
    auto t0 = std::chrono::steady_clock::now();
    Kokkos::parallel_for("arith", n_elem, KOKKOS_LAMBDA(int i) {
        T a, b;
        for (int k = 0; k < T::ncoeffs; ++k) {
            a[k] = static_cast<Coeff>(0.30 + 0.05 * ((i + k) % 5));
            b[k] = static_cast<Coeff>(0.20 + 0.04 * ((i + 2 * k) % 6));
        }
        a[0] = static_cast<Coeff>(1.5) + static_cast<Coeff>(1e-6) * i;
        b[0] = static_cast<Coeff>(2.0) + static_cast<Coeff>(1e-6) * i;
        T acc{};
        for (int r = 0; r < reps; ++r) {
            a[0] += static_cast<Coeff>(1e-9);  // perturb to defeat hoisting/CSE
            if constexpr (op == MUL) {
                acc += a * b;
            } else if constexpr (op == DIV) {
                acc += a / b;
            } else {
                acc += oti::exp(a) + oti::sin(a) + oti::log(b);
            }
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

template <int M, int N, class Coeff, Op op>
void sweep(char const* backend, int n_elem, double target_ms, int repetitions)
{
    using T = oti::otinum<M, N, Coeff>;
    int reps = 1;
    for (;;) {
        auto t0 = std::chrono::steady_clock::now();
        (void)run_once<T, op>(n_elem, reps);
        double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
        if (ms >= target_ms || reps >= (1 << 24)) break;
        int grow = static_cast<int>(target_ms / (ms > 0.05 ? ms : 0.05)) + 1;
        reps *= (grow < 2 ? 2 : grow > 16 ? 16 : grow);
    }
    for (int rep = 1; rep <= repetitions; ++rep) {
        measurement m = run_once<T, op>(n_elem, reps);
        bench::print_row(backend, bench::coeff_name<Coeff>(), M, N, T::ncoeffs,
                         T::table_type::nproducts, op_name(op), BENCH_VARIANT, rep,
                         "ns_per_op", m.ns_per_op, m.checksum);
    }
}

template <int M, int N>
void run_shape(char const* backend, int n_elem, double target_ms, int reps)
{
    sweep<M, N, double, MUL>(backend, n_elem, target_ms, reps);
    sweep<M, N, double, DIV>(backend, n_elem, target_ms, reps);
    sweep<M, N, double, FUNC>(backend, n_elem, target_ms, reps);
    sweep<M, N, float, MUL>(backend, n_elem, target_ms, reps);
    sweep<M, N, float, DIV>(backend, n_elem, target_ms, reps);
    sweep<M, N, float, FUNC>(backend, n_elem, target_ms, reps);
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
