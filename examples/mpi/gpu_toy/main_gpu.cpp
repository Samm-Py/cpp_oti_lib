// GPU variant of the MPI + OTI toy (step 2): prove the committed MPI datatype
// works for jets produced on the GPU.
//
// Each rank evaluates f(x,y)=sin(x)*exp(y) over its slice of a 1000x1000 grid on
// the DEVICE (Kokkos::View<Jet*> in CUDA space), then MPI_Gatherv's every jet to
// rank 0 using oti::mpi::make_datatype<Jet>().
//
// Two transport paths, chosen at runtime:
//   * CUDA-aware MPI  -> pass the device pointer straight to MPI (what a real
//                        multi-GPU deployment wants; uses GPUDirect / CUDA IPC).
//   * host staging    -> deep_copy device->host mirror, MPI on the host buffer
//                        (always works; the only path under a non-CUDA-aware MPI).
// Detected via MPIX_Query_cuda_support() (Open MPI); override with the env var
// OTI_MPI_DEVICE=1/0. The committed datatype is identical either way -- otinum's
// layout is the same in host and device memory (checked below).
//
// Verification is device-vs-device: rank 0 recomputes the whole grid on its GPU
// and memcmps against the gathered buffer. Both run the SAME kernel on the SAME
// hardware, so they are bit-identical even though GPU sin/exp need not match the
// CPU bit-for-bit. This isolates the MPI transport, which is what is tested.
//
// Build/run: see CMakeLists.txt in this directory.

#include <Kokkos_Core.hpp>
#include <mpi.h>
#if defined(OPEN_MPI)
#include <mpi-ext.h>   // MPIX_Query_cuda_support (Open MPI extension)
#endif

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "otinum/otinum.hpp"
#include "otinum/mpi.hpp"

using Jet = oti::otinum<2, 2, double>;   // value + grad + Hessian w.r.t. (x, y)
static constexpr int N = 1000;
static constexpr long TOTAL = static_cast<long>(N) * N;

KOKKOS_INLINE_FUNCTION Jet evaluate(long g)
{
    const double h = 1.0 / (N - 1);
    const long i = g / N;
    const long j = g % N;
    Jet x = Jet::variable(0, i * h);
    Jet y = Jet::variable(1, j * h);
    return oti::sin(x) * oti::exp(y);
}

static void block_range(int rank, int size, long& start, long& count)
{
    const long base = TOTAL / size;
    const long rem = TOTAL % size;
    count = base + (rank < rem ? 1 : 0);
    start = rank * base + (rank < rem ? rank : rem);
}

// Does this MPI advertise CUDA-aware support? Detected when built against Open
// MPI; an OTI_MPI_DEVICE=1/0 env var overrides (e.g. to force-test the path).
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
    {
        int rank = 0, size = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        MPI_Datatype MPI_OTINUM = oti::mpi::make_datatype<Jet>();
        const bool cuda_aware = detect_cuda_aware();

        // ---- layout agreement: sizeof(Jet) on device must equal host ----------
        Kokkos::View<std::size_t*> d_sz("d_sz", 1);
        Kokkos::parallel_for(
            "sizeof", 1, KOKKOS_LAMBDA(int) { d_sz(0) = sizeof(Jet); });
        auto h_sz = Kokkos::create_mirror_view(d_sz);
        Kokkos::deep_copy(h_sz, d_sz);
        const bool layout_ok = (h_sz(0) == sizeof(Jet));

        // ---- each rank evaluates its slice on the DEVICE ----------------------
        long start = 0, count = 0;
        block_range(rank, size, start, count);
        Kokkos::View<Jet*> d_local("d_local", count);
        Kokkos::parallel_for(
            "evaluate", count,
            KOKKOS_LAMBDA(int k) { d_local(k) = evaluate(start + k); });

        // ---- gather to rank 0; result ends up in h_global (host) either way ---
        std::vector<int> recvcounts, displs;
        Kokkos::View<Jet*, Kokkos::HostSpace> h_global;
        if (rank == 0) {
            recvcounts.resize(static_cast<size_t>(size));
            displs.resize(static_cast<size_t>(size));
            for (int r = 0; r < size; ++r) {
                long s = 0, c = 0;
                block_range(r, size, s, c);
                recvcounts[static_cast<size_t>(r)] = static_cast<int>(c);
                displs[static_cast<size_t>(r)] = static_cast<int>(s);
            }
            h_global = Kokkos::View<Jet*, Kokkos::HostSpace>("h_global", TOTAL);
        }

        if (cuda_aware) {
            // device pointers straight into MPI -- no host staging
            Kokkos::View<Jet*> d_global;
            if (rank == 0) {
                d_global = Kokkos::View<Jet*>("d_global", TOTAL);
            }
            MPI_Gatherv(d_local.data(), static_cast<int>(count), MPI_OTINUM,
                        rank == 0 ? d_global.data() : nullptr,
                        rank == 0 ? recvcounts.data() : nullptr,
                        rank == 0 ? displs.data() : nullptr,
                        MPI_OTINUM, 0, MPI_COMM_WORLD);
            if (rank == 0) {
                Kokkos::deep_copy(h_global, d_global);
            }
        } else {
            // host staging: device -> host mirror, then MPI on the host buffer
            auto h_local = Kokkos::create_mirror_view(d_local);
            Kokkos::deep_copy(h_local, d_local);
            MPI_Gatherv(h_local.data(), static_cast<int>(count), MPI_OTINUM,
                        rank == 0 ? h_global.data() : nullptr,
                        rank == 0 ? recvcounts.data() : nullptr,
                        rank == 0 ? displs.data() : nullptr,
                        MPI_OTINUM, 0, MPI_COMM_WORLD);
        }

        // ---- verify on rank 0: device recompute of the full grid, bit-exact ---
        if (rank == 0) {
            Kokkos::View<Jet*> d_ref("d_ref", TOTAL);
            Kokkos::parallel_for(
                "reference", TOTAL,
                KOKKOS_LAMBDA(long g) { d_ref(g) = evaluate(g); });
            auto h_ref = Kokkos::create_mirror_view(d_ref);
            Kokkos::deep_copy(h_ref, d_ref);

            long mismatches = 0;
            for (long g = 0; g < TOTAL; ++g) {
                if (std::memcmp(&h_global(g), &h_ref(g), sizeof(Jet)) != 0) {
                    ++mismatches;
                }
            }

            const long mid = (TOTAL / 2) + (N / 2);
            const Jet& s = h_global(static_cast<size_t>(mid));
            std::printf("backend          : %s\n",
                        Kokkos::DefaultExecutionSpace::name());
            std::printf("ranks            : %d\n", size);
            std::printf("transport        : %s\n",
                        cuda_aware ? "CUDA-aware MPI (device pointer)"
                                   : "host staging (deep_copy + MPI)");
            std::printf("grid             : %d x %d (%ld points)\n", N, N, TOTAL);
            std::printf("sizeof(Jet) host/device : %zu / %zu  %s\n",
                        sizeof(Jet), h_sz(0), layout_ok ? "OK" : "MISMATCH");
            std::printf("sample @ %ld     : value=% .8f  d/dx=% .8f  d/dy=% .8f\n",
                        mid, s.coeff(oti::sparse({})),
                        s.coeff(oti::sparse({{0, 1}})),
                        s.coeff(oti::sparse({{1, 1}})));
            std::printf("verify (device recompute) : %s (%ld mismatching jets)\n",
                        (mismatches == 0 && layout_ok) ? "PASS (bit-exact)"
                                                       : "FAIL",
                        mismatches);
        }

        oti::mpi::free_datatype(MPI_OTINUM);
    }
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
