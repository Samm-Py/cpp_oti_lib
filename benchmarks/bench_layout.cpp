// Layout isolation benchmark: array-of-structs (View<otinum*>) vs the
// coefficient-major oti::soa_span, over two access patterns, holding the
// arithmetic and everything else at the library default.
//
//   stream : y = a*x + y, one read of x and one read+write of y per element.
//            Sequential, so the question is pure coalescing of contiguous jets.
//   gather : each element sums K neighbor jets at scattered offsets, like a
//            stencil matvec. Each neighbor is a separate jet load -- AoS does it
//            as one (possibly wide) contiguous load per neighbor; SoA does it as
//            ncoeffs coalesced loads per neighbor. This is where small aligned
//            jets favor AoS and the heat solver's stiffness gather lives.
//
// Reports useful GB/s (useful coefficient bytes / time). AoS and SoA compute the
// same thing, so their checksums must match. This is the data behind "when to
// use SoA vs AoS".
//
// Requires OTI_ENABLE_KOKKOS; runs on whatever backend the build targets.

#include <Kokkos_Core.hpp>

#include "otinum/otinum.hpp"
#include "otinum/soa.hpp"

#include "bench_common.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr int K = 8;  // neighbors per gather
KOKKOS_INLINE_FUNCTION int neighbor_offset(int j)
{
    // Scattered offsets so neighbors are distinct memory locations.
    constexpr int off[K] = {1, 3, 7, 15, 31, 63, 127, 255};
    return off[j];
}

// AoS storage adapter with the same load/store surface as oti::soa_span, so one
// templated kernel drives both layouts.
template <class T>
struct aos_view {
    using value_type = T;
    Kokkos::View<T*> data;
    KOKKOS_INLINE_FUNCTION T load(std::size_t i) const { return data(i); }
    KOKKOS_INLINE_FUNCTION void store(std::size_t i, T const& v) const { data(i) = v; }
};

template <class T>
KOKKOS_INLINE_FUNCTION T fill_jet(std::size_t i, int salt)
{
    using Coeff = typename T::coeff_type;
    T out;
    for (int k = 0; k < T::ncoeffs; ++k) {
        out[k] = static_cast<Coeff>(0.25 + 0.03 * ((i + std::size_t(k * salt)) % 17));
    }
    return out;
}

template <class T, class Store>
void init(Store s, std::size_t n, int salt)
{
    Kokkos::parallel_for("init", n, KOKKOS_LAMBDA(std::size_t i) {
        s.store(i, fill_jet<T>(i, salt));
    });
    Kokkos::fence();
}

template <class T, class Store>
double checksum_of(Store s, std::size_t n)
{
    double acc = 0;
    Kokkos::parallel_reduce("checksum", n, KOKKOS_LAMBDA(std::size_t i, double& a) {
        T const v = s.load(i);
        for (int k = 0; k < T::ncoeffs; ++k) {
            a += static_cast<double>(v[k]);
        }
    }, acc);
    return acc;
}

// Returns useful GB/s.
template <class T, class SX, class SY>
double time_stream(SX x, SY y, std::size_t n, int reps)
{
    using Coeff = typename T::coeff_type;
    Coeff const a = static_cast<Coeff>(0.5);
    auto pass = [&] {
        Kokkos::parallel_for("stream", n, KOKKOS_LAMBDA(std::size_t i) {
            T xi = x.load(i);
            T yi = y.load(i);
            for (int k = 0; k < T::ncoeffs; ++k) {
                yi[k] = a * xi[k] + yi[k];
            }
            y.store(i, yi);
        });
    };
    pass();
    Kokkos::fence();
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) pass();
    Kokkos::fence();
    double const secs = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - t0).count();
    double const bytes = 3.0 * double(T::ncoeffs) * sizeof(Coeff) * double(n) * reps;
    return bytes / secs * 1e-9;
}

template <class T, class SX, class SY>
double time_gather(SX x, SY y, std::size_t n, int reps)
{
    using Coeff = typename T::coeff_type;
    auto pass = [&] {
        Kokkos::parallel_for("gather", n, KOKKOS_LAMBDA(std::size_t i) {
            T acc{};
            for (int j = 0; j < K; ++j) {
                std::size_t const idx = (i + std::size_t(neighbor_offset(j))) % n;
                T nb = x.load(idx);
                for (int k = 0; k < T::ncoeffs; ++k) {
                    acc[k] += nb[k];
                }
            }
            y.store(i, acc);
        });
    };
    pass();
    Kokkos::fence();
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) pass();
    Kokkos::fence();
    double const secs = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - t0).count();
    double const bytes = double(K + 1) * double(T::ncoeffs) * sizeof(Coeff) * double(n) * reps;
    return bytes / secs * 1e-9;
}

template <int M, int N, class Coeff>
void run_shape(char const* backend, int reps, int repetitions, std::size_t node_override)
{
    using T = oti::otinum<M, N, Coeff>;
    using Span = oti::soa_span<M, N, Coeff>;
    // Default to a fixed ~96 MiB working set (keeps the stream test memory-bound
    // across shapes); --nodes overrides it with an explicit element count.
    std::size_t const n = node_override ? node_override : (96u << 20) / sizeof(T);

    Kokkos::View<T*> ax("ax", n), ay("ay", n);
    Kokkos::View<Coeff*> sx("sx", Span::required_size(n)), sy("sy", Span::required_size(n));
    aos_view<T> const aos_x{ax}, aos_y{ay};
    Span const soa_x(sx.data(), n), soa_y(sy.data(), n);

    for (int rep = 1; rep <= repetitions; ++rep) {
        // stream
        init<T>(aos_x, n, 3); init<T>(aos_y, n, 5);
        init<T>(soa_x, n, 3); init<T>(soa_y, n, 5);
        double const aos_s = time_stream<T>(aos_x, aos_y, n, reps);
        double const soa_s = time_stream<T>(soa_x, soa_y, n, reps);
        double const cs_as = checksum_of<T>(aos_y, n);
        double const cs_ss = checksum_of<T>(soa_y, n);
        bench::print_row(backend, bench::coeff_name<Coeff>(), M, N, T::ncoeffs,
                         T::table_type::nproducts, "stream", "aos", rep, "useful_gbps", aos_s, cs_as);
        bench::print_row(backend, bench::coeff_name<Coeff>(), M, N, T::ncoeffs,
                         T::table_type::nproducts, "stream", "soa", rep, "useful_gbps", soa_s, cs_ss);

        // gather
        init<T>(aos_x, n, 3); init<T>(soa_x, n, 3);
        double const aos_g = time_gather<T>(aos_x, aos_y, n, reps);
        double const soa_g = time_gather<T>(soa_x, soa_y, n, reps);
        double const cs_ag = checksum_of<T>(aos_y, n);
        double const cs_sg = checksum_of<T>(soa_y, n);
        bench::print_row(backend, bench::coeff_name<Coeff>(), M, N, T::ncoeffs,
                         T::table_type::nproducts, "gather", "aos", rep, "useful_gbps", aos_g, cs_ag);
        bench::print_row(backend, bench::coeff_name<Coeff>(), M, N, T::ncoeffs,
                         T::table_type::nproducts, "gather", "soa", rep, "useful_gbps", soa_g, cs_sg);
    }
}

template <int M, int N>
void run_both(char const* backend, int reps, int repetitions, std::size_t node_override)
{
    run_shape<M, N, double>(backend, reps, repetitions, node_override);
    run_shape<M, N, float>(backend, reps, repetitions, node_override);
}

// Run one precompiled shape only if the runtime --shapes filter selected it.
template <int M, int N>
void run_selected(char const* backend, bench::shape_list const& shapes,
                  int reps, int repetitions, std::size_t node_override)
{
    if (bench::wanted(shapes, M, N)) run_both<M, N>(backend, reps, repetitions, node_override);
}

} // namespace

int main(int argc, char** argv)
{
    Kokkos::initialize(argc, argv);
    {
        int const pargc = bench::positional_argc(argc, argv);
        int const reps = (pargc > 1) ? std::atoi(argv[1]) : 20;
        int const repetitions = (pargc > 2) ? std::atoi(argv[2]) : 11;
        long const node_flag = bench::flag_long(argc, argv, "--nodes", -1);
        std::size_t const node_override = node_flag > 0 ? static_cast<std::size_t>(node_flag) : 0;
        bench::shape_list const shapes = bench::parse_shapes(argc, argv);
        char const* backend = Kokkos::DefaultExecutionSpace::name();
        bench::print_header();

#define OTI_BENCH_SHAPE(M, N) \
    run_selected<M, N>(backend, shapes, reps, repetitions, node_override);
#include "bench_shapes.def"
#undef OTI_BENCH_SHAPE
    }
    Kokkos::finalize();
    return 0;
}
