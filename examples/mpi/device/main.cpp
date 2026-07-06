// MPI + OTI with DEVICE-RESIDENT buffers: the fifth conversion rung.
//
// Every previous rung changes WHAT the ranks communicate (gather, reduce, halo,
// unstructured); this one changes WHERE the buffers live -- GPU memory -- and
// shows that the OTI surface does not change at all. The same committed
// datatype describes a jet in device memory (the layout is identical, checked
// in gpu_toy), and the same custom MPI_Op folds them.
//
// Each rank evaluates its block of f(x, y; a, b) = sin(a*x) * exp(b*y) on the
// DEVICE (a, b seeded as OTI variables, so every point carries value + gradient
// + Hessian w.r.t. the design parameters), reduces its block to a partial-sum
// jet ON the device, and then runs the same two collectives as the earlier
// rungs -- MPI_Gatherv (movement) and MPI_Allreduce with the custom jet-sum op
// (combining) -- over two transports:
//
//   * host staging   -> deep_copy device->host mirror, MPI on host buffers.
//                       Works under ANY MPI; the portable baseline.
//   * device direct  -> the device pointers go straight into MPI. Needs a
//                       CUDA-aware MPI. Movement collectives move the bytes
//                       device-natively; for the custom-op reduction the
//                       library stages the buffers through host internally,
//                       because an MPI_Op is a HOST callback by the MPI
//                       standard -- correct either way, verified below.
//
// Detected via MPIX_Query_cuda_support() (Open MPI); override with the env var
// OTI_MPI_DEVICE=1/0. Under a host-parallel Kokkos backend (OpenMP/Serial) the
// "device" views are host memory, so the direct path is safe under any MPI and
// OTI_MPI_DEVICE=1 exercises it.
//
// Verification: the two transports must agree BIT-FOR-BIT (same device data,
// same reduction tree -- only the buffer location differs); the staged QoI is
// additionally checked against a serial host recompute to a tight relative
// tolerance (host and device libm may differ in the last bits, and a
// distributed sum is not associative). The raw-coefficient MPI_SUM escape
// hatch (see below) is checked against the custom op to the same tolerance.
//
// Build/run: see CMakeLists.txt in this directory.

#include <Kokkos_Core.hpp>
#include <mpi.h>
#if defined(OPEN_MPI)
#include <mpi-ext.h>   // MPIX_Query_cuda_support (Open MPI extension)
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "otinum/otinum.hpp"
#include "otinum/mpi.hpp"

using Jet = oti::otinum<2, 2, double>;   // value + grad + Hessian w.r.t. (a, b)
static constexpr int  N       = 1000;    // grid is N x N
static constexpr long TOTAL   = static_cast<long>(N) * N;
static constexpr double A0    = 1.0;     // design parameter a
static constexpr double B0    = 1.0;     // design parameter b
static constexpr double SER_TOL = 1.0e-10;  // staged QoI vs serial host recompute

// Kokkos needs the additive identity of a custom scalar to parallel_reduce over
// it: the Sum reducer starts each thread at reduction_identity<Jet>::sum() and
// joins partials with operator+=, which otinum already provides.
namespace Kokkos {
template <>
struct reduction_identity<Jet> {
    KOKKOS_FORCEINLINE_FUNCTION static Jet sum() { return Jet(0.0); }
};
} // namespace Kokkos

// f at flat grid index g, with the design parameters carried as jets. Device-
// callable: everything inside is OTI arithmetic (OTI_ENABLE_KOKKOS is defined).
KOKKOS_INLINE_FUNCTION Jet field_jet(long g)
{
    const double h = 1.0 / (N - 1);
    const double x = (g / N) * h;
    const double y = (g % N) * h;
    const Jet a = Jet::variable(0, A0);   // A0 + e_0
    const Jet b = Jet::variable(1, B0);   // B0 + e_1
    return oti::sin(a * x) * oti::exp(b * y);
}

// Same block decomposition as the gather and reduce rungs.
static void block_range(int rank, int size, long& start, long& count)
{
    const long base = TOTAL / size;
    const long rem  = TOTAL % size;
    count = base + (rank < rem ? 1 : 0);
    start = rank * base + (rank < rem ? rank : rem);
}

// Does this MPI advertise CUDA-aware support? Compile-time gated on Open MPI's
// extension header; OTI_MPI_DEVICE=1/0 overrides (e.g. to force-test a path).
static bool detect_cuda_aware()
{
    bool aware = false;
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    aware = (MPIX_Query_cuda_support() == 1);
#endif
    if (const char* e = std::getenv("OTI_MPI_DEVICE")) {
        aware = (std::atoi(e) != 0);
    }
    return aware;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);
    int failures = 0;
    {
        int rank = 0, size = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        // The SAME datatype and op as the host-only rungs -- nothing about
        // them is host-specific.
        MPI_Datatype MPI_OTINUM = oti::mpi::make_datatype<Jet>();
        MPI_Op MPI_OTI_SUM = oti::mpi::make_sum_op<Jet>();
        const bool device_mpi = detect_cuda_aware();

        // ---- evaluate this rank's block on the device ----------------------
        long start = 0, count = 0;
        block_range(rank, size, start, count);
        Kokkos::View<Jet*> d_local("d_local", count);
        Kokkos::parallel_for(
            "evaluate", count,
            KOKKOS_LAMBDA(long k) { d_local(k) = field_jet(start + k); });

        // ---- reduce the block to ONE partial-sum jet, kept on the device ---
        Kokkos::View<Jet> d_partial("d_partial");
        Kokkos::parallel_reduce(
            "partial_sum", count,
            KOKKOS_LAMBDA(long k, Jet& acc) { acc += d_local(k); }, d_partial);
        Kokkos::fence();

        // Root-side gather bookkeeping, shared by both transports.
        std::vector<int> recvcounts, displs;
        if (rank == 0) {
            recvcounts.resize(static_cast<size_t>(size));
            displs.resize(static_cast<size_t>(size));
            for (int r = 0; r < size; ++r) {
                long s = 0, c = 0;
                block_range(r, size, s, c);
                recvcounts[static_cast<size_t>(r)] = static_cast<int>(c);
                displs[static_cast<size_t>(r)] = static_cast<int>(s);
            }
        }

        // ---- transport 1: host staging (works under ANY MPI) ---------------
        // deep_copy to host mirrors, then the familiar host-buffer collectives.
        auto h_local = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, d_local);
        auto h_partial = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, d_partial);

        Kokkos::View<Jet*, Kokkos::HostSpace> h_field_staged;
        if (rank == 0) {
            h_field_staged =
                Kokkos::View<Jet*, Kokkos::HostSpace>("h_field_staged", TOTAL);
        }
        MPI_Gatherv(h_local.data(), static_cast<int>(count), MPI_OTINUM,
                    rank == 0 ? h_field_staged.data() : nullptr,
                    rank == 0 ? recvcounts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_OTINUM, 0, MPI_COMM_WORLD);

        Jet qoi_staged(0.0);
        MPI_Allreduce(h_partial.data(), &qoi_staged, 1, MPI_OTINUM, MPI_OTI_SUM,
                      MPI_COMM_WORLD);

        // ---- transport 2: device pointers straight into MPI -----------------
        // The staging lines above simply DISAPPEAR; datatype, op, counts, and
        // displacements are untouched.
        bool field_bitexact = true, qoi_bitexact = true;
        double hatch_rel = 0.0;
        if (device_mpi) {
            Kokkos::View<Jet*> d_field;
            if (rank == 0) {
                d_field = Kokkos::View<Jet*>("d_field", TOTAL);
            }
            MPI_Gatherv(d_local.data(), static_cast<int>(count), MPI_OTINUM,
                        rank == 0 ? d_field.data() : nullptr,
                        rank == 0 ? recvcounts.data() : nullptr,
                        rank == 0 ? displs.data() : nullptr,
                        MPI_OTINUM, 0, MPI_COMM_WORLD);

            // Custom-op reduction with device buffers. An MPI_Op is a host
            // callback by the MPI standard, so a CUDA-aware MPI stages these
            // buffers through host internally -- same jets, same reduction
            // tree, so the result must match the staged transport bit-for-bit.
            Kokkos::View<Jet> d_qoi("d_qoi");
            MPI_Allreduce(d_partial.data(), d_qoi.data(), 1, MPI_OTINUM,
                          MPI_OTI_SUM, MPI_COMM_WORLD);

            // The SUM escape hatch: jet addition is coefficient-wise, so the
            // predefined MPI_SUM over the ncoeffs raw doubles is the same
            // mathematics with NO user callback -- the fully device-native
            // path. (MPI may pick a different reduction algorithm for a
            // predefined op, so compare to tolerance, not bits.)
            Kokkos::View<Jet> d_qoi_raw("d_qoi_raw");
            MPI_Allreduce(d_partial.data(), d_qoi_raw.data(), Jet::ncoeffs,
                          MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            // ---- device transport vs staged transport, bit for bit ---------
            auto h_qoi = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace{}, d_qoi);
            auto h_qoi_raw = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace{}, d_qoi_raw);
            qoi_bitexact =
                std::memcmp(h_qoi.data(), &qoi_staged, sizeof(Jet)) == 0;
            for (int c = 0; c < Jet::ncoeffs; ++c) {
                const double s = qoi_staged[c];
                const double rel =
                    std::fabs(h_qoi_raw()[c] - s) / (std::fabs(s) + 1e-300);
                if (rel > hatch_rel) hatch_rel = rel;
            }
            if (rank == 0) {
                auto h_field = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, d_field);
                field_bitexact =
                    std::memcmp(h_field.data(), h_field_staged.data(),
                                static_cast<size_t>(TOTAL) * sizeof(Jet)) == 0;
            }
        }

        // ---- report + verify on rank 0 --------------------------------------
        if (rank == 0) {
            const Jet qoi = qoi_staged * (1.0 / static_cast<double>(TOTAL));
            std::printf("backend            : %s\n",
                        Kokkos::DefaultExecutionSpace::name());
            std::printf("ranks              : %d\n", size);
            std::printf("transport          : %s\n",
                        device_mpi ? "device pointers (CUDA-aware MPI) "
                                     "+ staged reference"
                                   : "host staging only "
                                     "(MPI is not CUDA-aware)");
            std::printf("grid               : %d x %d  (%ld jets, %.1f MB)\n",
                        N, N, TOTAL, TOTAL * sizeof(Jet) / 1048576.0);
            std::printf("QoI = mean of sin(a x) exp(b y)  at a=%.3f, b=%.3f\n",
                        A0, B0);
            std::printf("  value            = % .10f\n",
                        qoi.coeff(oti::sparse({})));
            std::printf("  d/da             = % .10f\n",
                        qoi.coeff(oti::sparse({{0, 1}})));
            std::printf("  d/db             = % .10f\n",
                        qoi.coeff(oti::sparse({{1, 1}})));

            // Staged QoI vs a serial recompute on the HOST. Tolerance-based:
            // the partition changes the summation order, and host libm need
            // not match the device's last bits.
            Jet ser(0.0);
            for (long g = 0; g < TOTAL; ++g) {
                const double h = 1.0 / (N - 1);
                const double x = (g / N) * h;
                const double y = (g % N) * h;
                ser += oti::sin(Jet::variable(0, A0) * x) *
                       oti::exp(Jet::variable(1, B0) * y);
            }
            double ser_rel = 0.0;
            for (int c = 0; c < Jet::ncoeffs; ++c) {
                const double s = ser[c];
                const double rel = std::fabs(qoi_staged[c] - s) /
                                   (std::fabs(s) + 1e-300);
                if (rel > ser_rel) ser_rel = rel;
            }
            const bool ser_ok = ser_rel <= SER_TOL;
            failures += ser_ok ? 0 : 1;
            std::printf("staged vs serial   : %s (max relative diff %.2e)\n",
                        ser_ok ? "PASS" : "FAIL", ser_rel);

            if (device_mpi) {
                failures += field_bitexact ? 0 : 1;
                std::printf("device Gatherv     : %s\n",
                            field_bitexact ? "PASS (bit-exact vs staged)"
                                           : "FAIL");
                failures += qoi_bitexact ? 0 : 1;
                std::printf("device custom-op   : %s\n",
                            qoi_bitexact ? "PASS (bit-exact vs staged)"
                                         : "FAIL");
                const bool hatch_ok = hatch_rel <= SER_TOL;
                failures += hatch_ok ? 0 : 1;
                std::printf("raw MPI_SUM hatch  : %s (max relative diff "
                            "%.2e)\n",
                            hatch_ok ? "PASS" : "FAIL", hatch_rel);
            } else {
                std::printf("device transport   : skipped -- rerun under a "
                            "CUDA-aware MPI (or OTI_MPI_DEVICE=1)\n");
            }
        }

        oti::mpi::free_op(MPI_OTI_SUM);
        oti::mpi::free_datatype(MPI_OTINUM);
        MPI_Bcast(&failures, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    Kokkos::finalize();
    MPI_Finalize();
    return failures == 0 ? 0 : 1;
}
