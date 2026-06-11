// Streaming-layout benchmark: array-of-structs (Kokkos::View<otinum*>) versus
// the coefficient-major oti::soa_span over the same arithmetic.
//
// Unlike the register-resident kernel microbenchmark (which reps many ops per
// element and is compute-bound), each kernel here makes exactly one trip
// through memory per element -- two jet reads and one jet write -- so the
// measured rate is dominated by the access pattern the layout produces. The
// reported figure is effective bandwidth (bytes moved / time); the AoS-vs-SoA
// ratio is the layout's win or loss.
//
// Both layouts run identical register-resident otinum arithmetic, so their
// checksums must match bit-exactly; a mismatch means a layout bug, not
// rounding.
//
// Requires OTI_ENABLE_KOKKOS; runs on whatever backend the build targets.

#include <Kokkos_Core.hpp>

#include "otinum/otinum.hpp"
#include "otinum/soa.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace {

enum Op { AXPY, MUL };

template <class T, Op op>
KOKKOS_FORCEINLINE_FUNCTION T apply(T x, T y, typename T::coeff_type s)
{
    if constexpr (op == AXPY) {
        oti::axpy(y, s, x);
        return y;
    } else {
        return x * y;
    }
}

// Deterministic element-dependent fill, same for both layouts.
template <class T>
KOKKOS_FORCEINLINE_FUNCTION T fill_jet(std::size_t i, int salt)
{
    using Coeff = typename T::coeff_type;
    T out;
    for (int k = 0; k < T::ncoeffs; ++k) {
        out[k] = static_cast<Coeff>(0.25 + 0.03 * ((i + static_cast<std::size_t>(k * salt)) % 17));
    }
    return out;
}

struct result {
    double gbps;
    double checksum;
};

template <class T, Op op>
result bench_aos(std::size_t n, int reps)
{
    using Coeff = typename T::coeff_type;
    Kokkos::View<T*> x("x", n);
    Kokkos::View<T*> y("y", n);
    Kokkos::parallel_for("init_aos", n, KOKKOS_LAMBDA(std::size_t i) {
        x(i) = fill_jet<T>(i, 3);
        y(i) = fill_jet<T>(i, 5);
    });
    Kokkos::fence();

    Coeff const s = static_cast<Coeff>(0.5);
    auto pass = [&](char const* label) {
        Kokkos::parallel_for(label, n, KOKKOS_LAMBDA(std::size_t i) {
            T const xi = x(i);
            T const yi = y(i);
            y(i) = apply<T, op>(xi, yi, s);
        });
    };

    pass("aos_warmup");
    Kokkos::fence();
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) {
        pass("aos_stream");
    }
    Kokkos::fence();
    auto t1 = std::chrono::steady_clock::now();
    double const secs = std::chrono::duration<double>(t1 - t0).count();

    double checksum = 0;
    Kokkos::parallel_reduce("aos_checksum", n, KOKKOS_LAMBDA(std::size_t i, double& acc) {
        for (int k = 0; k < T::ncoeffs; ++k) {
            acc += static_cast<double>(y(i)[k]);
        }
    }, checksum);

    double const bytes = 3.0 * double(sizeof(T)) * double(n) * reps;
    return {bytes / secs * 1e-9, checksum};
}

template <class T, Op op>
result bench_soa(std::size_t n, int reps)
{
    using Coeff = typename T::coeff_type;
    using Span = oti::soa_span<T::nvars, T::order, Coeff>;
    Kokkos::View<Coeff*> xs("xs", Span::required_size(n));
    Kokkos::View<Coeff*> ys("ys", Span::required_size(n));
    Span const x(xs.data(), n);
    Span const y(ys.data(), n);
    Kokkos::parallel_for("init_soa", n, KOKKOS_LAMBDA(std::size_t i) {
        x.store(i, fill_jet<T>(i, 3));
        y.store(i, fill_jet<T>(i, 5));
    });
    Kokkos::fence();

    Coeff const s = static_cast<Coeff>(0.5);
    auto pass = [&](char const* label) {
        Kokkos::parallel_for(label, n, KOKKOS_LAMBDA(std::size_t i) {
            T const xi = x.load(i);
            T const yi = y.load(i);
            y.store(i, apply<T, op>(xi, yi, s));
        });
    };

    pass("soa_warmup");
    Kokkos::fence();
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) {
        pass("soa_stream");
    }
    Kokkos::fence();
    auto t1 = std::chrono::steady_clock::now();
    double const secs = std::chrono::duration<double>(t1 - t0).count();

    double checksum = 0;
    Kokkos::parallel_reduce("soa_checksum", n, KOKKOS_LAMBDA(std::size_t i, double& acc) {
        T const yi = y.load(i);
        for (int k = 0; k < T::ncoeffs; ++k) {
            acc += static_cast<double>(yi[k]);
        }
    }, checksum);

    double const bytes = 3.0 * double(sizeof(T)) * double(n) * reps;
    return {bytes / secs * 1e-9, checksum};
}

template <class T, Op op>
void compare(char const* tag, int reps)
{
    // Size each array to ~96 MiB so every pass streams from device/main
    // memory rather than cache, independent of the jet size.
    std::size_t const n = (96u << 20) / sizeof(T);

    result const aos = bench_aos<T, op>(n, reps);
    result const soa = bench_soa<T, op>(n, reps);

    bool const match = aos.checksum == soa.checksum;
    std::printf("%-16s  aos %7.2f GB/s   soa %7.2f GB/s   soa/aos %5.2fx   checksums %s\n",
                tag, aos.gbps, soa.gbps, soa.gbps / aos.gbps,
                match ? "match" : "MISMATCH");
    std::fflush(stdout);
    if (!match) {
        std::exit(1);
    }
}

} // namespace

int main(int argc, char** argv)
{
    Kokkos::initialize(argc, argv);
    {
        int const reps = (argc > 1) ? std::atoi(argv[1]) : 20;
        std::printf("backend: %s   (each array ~96 MiB, %d timed passes)\n",
                    Kokkos::DefaultExecutionSpace::name(), reps);

        std::printf("== axpy: y += s*x ==\n");
        compare<oti::otinum<3, 1, double>, AXPY>("<3,1> double", reps);
        compare<oti::otinum<3, 1, float>, AXPY>("<3,1> float", reps);
        compare<oti::otinum<3, 3, double>, AXPY>("<3,3> double", reps);
        compare<oti::otinum<3, 3, float>, AXPY>("<3,3> float", reps);
        compare<oti::otinum<4, 4, double>, AXPY>("<4,4> double", reps);
        compare<oti::otinum<4, 4, float>, AXPY>("<4,4> float", reps);

        std::printf("== mul: y = x*y ==\n");
        compare<oti::otinum<3, 1, double>, MUL>("<3,1> double", reps);
        compare<oti::otinum<3, 1, float>, MUL>("<3,1> float", reps);
        compare<oti::otinum<3, 3, double>, MUL>("<3,3> double", reps);
        compare<oti::otinum<3, 3, float>, MUL>("<3,3> float", reps);
        compare<oti::otinum<4, 4, double>, MUL>("<4,4> double", reps);
        compare<oti::otinum<4, 4, float>, MUL>("<4,4> float", reps);
    }
    Kokkos::finalize();
    return 0;
}
