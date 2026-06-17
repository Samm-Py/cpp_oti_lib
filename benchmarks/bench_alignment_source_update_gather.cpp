// Alignment benchmark over the FE kernels of an explicit PDE step: the
// library's conditional otinum alignment measured on source-expression
// evaluation, a matrix-free operator apply (stencil gather), the nodal update,
// and an OTI consistent-mass solve. The type is the real oti::otinum, so the
// benchmark exposes generated-code and register-pressure changes from by-value
// OTI parameters and temporaries (the fused-op axis lives in bench_fused).
//
// Build twice:
//   natural : -DOTI_BENCHMARK_NATURAL_ALIGNMENT
//   aligned : library default alignment rule
//
// Args: n_elem repetitions target_ms min_node_updates. By default the benchmark
// allocates 68,921 nodes -- the 41x41x41 (N=41) heat-problem DOF count -- and
// lets target_ms calibrate the timed repetitions. Pass min_node_updates
// explicitly to force a larger work floor.
//
// metric = ns_per_node (lower is better).

#include <Kokkos_Core.hpp>

#include "otinum/otinum.hpp"

#include "bench_common.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>

#ifdef OTI_BENCHMARK_NATURAL_ALIGNMENT
#define ALIGN_VARIANT "natural"
#else
#define ALIGN_VARIANT "aligned"
#endif

namespace {

enum Kernel {
    SOURCE_EXPRESSION,
    OPERATOR_CHAIN_UPDATE,
    STENCIL_GATHER,
    MASS_SOLVE
};

char const* kernel_name(Kernel k)
{
    switch (k) {
    case SOURCE_EXPRESSION:
        return "source_expression";
    case OPERATOR_CHAIN_UPDATE:
        return "operator_chain_update";
    case STENCIL_GATHER:
        return "stencil_gather";
    case MASS_SOLVE:
        return "mass_solve";
    }
    return "unknown";
}

struct measurement {
    double ns_per_node;
    double checksum;
};

template <class T>
struct init_source_update_gather_views {
    using Real = typename T::coeff_type;
    Kokkos::View<T*> u;
    Kokkos::View<T*> f;
    Kokkos::View<T*> Ku;
    Kokkos::View<Real*> mass;
    Kokkos::View<T*> m_oti;

    KOKKOS_INLINE_FUNCTION void operator()(int i) const
    {
        T ui;
        T fi;
        T kui;
        T mi;
        for (int k = 0; k < T::ncoeffs; ++k) {
            ui[k] = static_cast<Real>(0.02 + 0.001 * ((i + k) % 11));
            fi[k] = static_cast<Real>(0.10 + 0.002 * ((i + 2 * k) % 13));
            kui[k] = static_cast<Real>(0.05 + 0.003 * ((i + 3 * k) % 17));
            // OTI consistent mass: a well-conditioned real part with small
            // higher-order coefficients, so the per-node inverse stays stable.
            mi[k] = static_cast<Real>(0.001 * ((i + 5 * k) % 7));
        }
        mi[0] = static_cast<Real>(0.8 + 0.001 * (i % 23));
        u(i) = ui;
        f(i) = fi;
        Ku(i) = kui;
        mass(i) = static_cast<Real>(0.5 + 0.001 * (i % 29));
        m_oti(i) = mi;
    }
};

template <class T>
struct source_expression_kernel {
    using Real = typename T::coeff_type;
    Kokkos::View<T*> f;
    Kokkos::View<Real*> mass;
    T amplitude;
    T inv_two_sigma2;

    KOKKOS_INLINE_FUNCTION void operator()(int i) const
    {
        Real x = static_cast<Real>((i % 257) - 128) * static_cast<Real>(0.003);
        Real y = static_cast<Real>(((i / 257) % 257) - 128) * static_cast<Real>(0.003);
        Real z = static_cast<Real>(((i / (257 * 257)) % 257) - 128) *
                 static_cast<Real>(0.003);
        Real r2 = x * x + y * y + z * z;

        T exponent = T(-r2) * inv_two_sigma2;
        T val = amplitude * oti::exp(exponent);
        f(i) = val * mass(i);
    }
};

template <class T>
struct operator_chain_update_kernel {
    using Real = typename T::coeff_type;
    Kokkos::View<T*> u;
    Kokkos::View<T*> u_new;
    Kokkos::View<T*> f;
    Kokkos::View<T*> Ku;
    Kokkos::View<Real*> mass;
    T alpha;
    Real dt;

    KOKKOS_INLINE_FUNCTION void operator()(int i) const
    {
        u_new(i) = u(i) + T(dt) * (Real(1) / mass(i)) * (f(i) - alpha * Ku(i));
    }
};

template <class Real>
struct init_stiffness {
    Kokkos::View<Real*> K;

    KOKKOS_INLINE_FUNCTION void operator()(int i) const
    {
        int row = i / 8;
        int col = i % 8;
        Real diag = row == col ? Real(0.18) : Real(-0.025);
        K(i) = diag + Real(0.001) * Real((row + 2 * col) % 5);
    }
};

template <class T>
struct stencil_gather_kernel {
    using Real = typename T::coeff_type;
    Kokkos::View<T*> u;
    Kokkos::View<T*> Ku;
    Kokkos::View<Real*> K;
    int n_elem;

    KOKKOS_INLINE_FUNCTION void operator()(int i) const
    {
        T sum = T(0);
        for (int elem = 0; elem < 8; ++elem) {
            int base = i + elem * 17 - 64;
            if (base < 0) {
                base += n_elem;
            } else if (base >= n_elem) {
                base -= n_elem;
            }

            for (int col = 0; col < 8; ++col) {
                int j = base + col * 3;
                if (j >= n_elem) {
                    j -= n_elem;
                }
                sum += K(elem * 8 + col) * u(j);
            }
        }
        Ku(i) = sum;
    }
};

// Consistent-mass solve: invert an OTI-valued nodal mass against the load
// vector, u_new = M^{-1} f, one node at a time. This is the FE operation where
// otinum division (inv + multiply) dominates, so it is the kernel where any
// alignment effect on the divide path would show up.
template <class T>
struct mass_solve_kernel {
    Kokkos::View<T*> f;
    Kokkos::View<T*> m_oti;
    Kokkos::View<T*> u_new;

    KOKKOS_INLINE_FUNCTION void operator()(int i) const
    {
        u_new(i) = f(i) / m_oti(i);
    }
};

template <class T>
struct checksum_view {
    Kokkos::View<T*> data;

    KOKKOS_INLINE_FUNCTION void operator()(int i, double& acc) const
    {
        T value = data(i);
        for (int k = 0; k < T::ncoeffs; ++k) {
            acc += static_cast<double>(value[k]);
        }
    }
};

template <class T, Kernel kernel>
measurement run_once(int n_elem, int reps)
{
    using Real = typename T::coeff_type;
    Kokkos::View<T*> u("u", n_elem);
    Kokkos::View<T*> u_new("u_new", n_elem);
    Kokkos::View<T*> f("f", n_elem);
    Kokkos::View<T*> Ku("Ku", n_elem);
    Kokkos::View<Real*> mass("mass", n_elem);
    Kokkos::View<T*> m_oti("m_oti", n_elem);
    Kokkos::View<Real*> K("K", 64);

    Kokkos::parallel_for("AlignmentSourceUpdateGatherInit", n_elem,
                         init_source_update_gather_views<T>{u, f, Ku, mass, m_oti});
    Kokkos::parallel_for("AlignmentSourceUpdateGatherInitStiffness", 64,
                         init_stiffness<Real>{K});
    Kokkos::fence();

    T alpha = T::variable(0, static_cast<Real>(1.0));
    T amplitude = T::variable(1, static_cast<Real>(100.0));
    T sigma = T::variable(2, static_cast<Real>(0.05));
    T inv_two_sigma2 = T(1) / (T(2) * sigma * sigma);
    Real dt = static_cast<Real>(1.0e-4);

    auto pass = [&] {
        if constexpr (kernel == SOURCE_EXPRESSION) {
            Kokkos::parallel_for("AlignmentSourceExpression", n_elem,
                                 source_expression_kernel<T>{f, mass, amplitude,
                                                             inv_two_sigma2});
        } else if constexpr (kernel == OPERATOR_CHAIN_UPDATE) {
            Kokkos::parallel_for("AlignmentOperatorChainUpdate", n_elem,
                                 operator_chain_update_kernel<T>{u, u_new, f, Ku,
                                                                 mass, alpha, dt});
        } else if constexpr (kernel == STENCIL_GATHER) {
            Kokkos::parallel_for("AlignmentStencilGather", n_elem,
                                 stencil_gather_kernel<T>{u, Ku, K, n_elem});
        } else {
            Kokkos::parallel_for("AlignmentMassSolve", n_elem,
                                 mass_solve_kernel<T>{f, m_oti, u_new});
        }
    };

    pass();
    Kokkos::fence();
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) {
        pass();
    }
    Kokkos::fence();
    double const secs = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - t0).count();

    double checksum = 0.0;
    Kokkos::parallel_reduce("AlignmentSourceUpdateGatherChecksum", n_elem,
                            checksum_view<T>{kernel == SOURCE_EXPRESSION ? f
                                             : kernel == STENCIL_GATHER ? Ku
                                                                       : u_new},
                            checksum);
    return {secs * 1e9 / (double(n_elem) * reps), checksum};
}

template <class T, Kernel kernel>
int calibrate(int n_elem, double target_ms, long long min_node_updates)
{
    int reps = 1;
    if (min_node_updates > 0) {
        long long const min_reps =
            (min_node_updates + static_cast<long long>(n_elem) - 1) /
            static_cast<long long>(n_elem);
        if (min_reps > reps) {
            reps = min_reps > static_cast<long long>(1 << 20) ? (1 << 20)
                                                              : static_cast<int>(min_reps);
        }
    }
    for (;;) {
        auto t0 = std::chrono::steady_clock::now();
        (void)run_once<T, kernel>(n_elem, reps);
        double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
        if (ms >= target_ms || reps >= (1 << 20)) {
            break;
        }
        int grow = static_cast<int>(target_ms / (ms > 0.05 ? ms : 0.05)) + 1;
        reps *= (grow < 2 ? 2 : grow > 16 ? 16 : grow);
    }
    return reps;
}

template <int M, int N, class Coeff, Kernel kernel>
void sweep(char const* backend, int n_elem, double target_ms, int repetitions,
           long long min_node_updates)
{
    static_assert(M >= 3, "source-expression benchmark seeds variables 0, 1, and 2");
    using T = oti::otinum<M, N, Coeff>;
    int const reps = calibrate<T, kernel>(n_elem, target_ms, min_node_updates);
    for (int rep = 1; rep <= repetitions; ++rep) {
        measurement m = run_once<T, kernel>(n_elem, reps);
        bench::print_row(backend, bench::coeff_name<Coeff>(), M, N, T::ncoeffs,
                         T::table_type::nproducts, kernel_name(kernel),
                         ALIGN_VARIANT, rep, "ns_per_node", m.ns_per_node,
                         m.checksum);
    }
}

template <int M, int N>
void run_shape(char const* backend, int n_elem, double target_ms, int repetitions,
               long long min_node_updates)
{
    sweep<M, N, double, SOURCE_EXPRESSION>(backend, n_elem, target_ms, repetitions,
                                           min_node_updates);
    sweep<M, N, double, OPERATOR_CHAIN_UPDATE>(backend, n_elem, target_ms, repetitions,
                                               min_node_updates);
    sweep<M, N, double, STENCIL_GATHER>(backend, n_elem, target_ms, repetitions,
                                        min_node_updates);
    sweep<M, N, double, MASS_SOLVE>(backend, n_elem, target_ms, repetitions,
                                    min_node_updates);
    sweep<M, N, float, SOURCE_EXPRESSION>(backend, n_elem, target_ms, repetitions,
                                          min_node_updates);
    sweep<M, N, float, OPERATOR_CHAIN_UPDATE>(backend, n_elem, target_ms, repetitions,
                                              min_node_updates);
    sweep<M, N, float, STENCIL_GATHER>(backend, n_elem, target_ms, repetitions,
                                       min_node_updates);
    sweep<M, N, float, MASS_SOLVE>(backend, n_elem, target_ms, repetitions,
                                   min_node_updates);
}

// Run one precompiled shape only if the runtime --shapes filter selected it.
// The kernels seed spatial variables 0, 1, and 2, so shapes with M < 3 are
// compile-time-skipped here (the `if constexpr` keeps run_shape<M,N> from being
// instantiated for them, which would trip the static_assert in sweep()).
template <int M, int N>
void run_selected(char const* backend, bench::shape_list const& shapes,
                  int n_elem, double target_ms, int repetitions, long long min_node_updates)
{
    if constexpr (M >= 3) {
        if (bench::wanted(shapes, M, N))
            run_shape<M, N>(backend, n_elem, target_ms, repetitions, min_node_updates);
    }
}

} // namespace

int main(int argc, char** argv)
{
    Kokkos::initialize(argc, argv);
    {
        // Default node count mirrors the N=41 heat problem (a 41x41x41 grid is
        // 68,921 nodes), so these per-kernel alignment numbers line up with the
        // end-to-end heat-application stage.
        int const pargc = bench::positional_argc(argc, argv);
        long const node_flag = bench::flag_long(argc, argv, "--nodes", -1);
        int const n_elem = node_flag > 0 ? static_cast<int>(node_flag)
                                          : (pargc > 1 ? std::atoi(argv[1]) : 68921);
        int const repetitions = static_cast<int>(bench::flag_long(
            argc, argv, "--repetitions", (pargc > 2) ? std::atoi(argv[2]) : 11));
        double const target_ms = (pargc > 3) ? std::atof(argv[3]) : 20.0;
        long long const min_node_updates =
            (pargc > 4) ? std::atoll(argv[4]) : 0LL;
        bench::shape_list const shapes = bench::parse_shapes(argc, argv);
        char const* backend = Kokkos::DefaultExecutionSpace::name();
        bench::print_header();

#define OTI_BENCH_SHAPE(M, N) \
    run_selected<M, N>(backend, shapes, n_elem, target_ms, repetitions, min_node_updates);
#include "bench_shapes.def"
#undef OTI_BENCH_SHAPE
    }
    Kokkos::finalize();
    return 0;
}
