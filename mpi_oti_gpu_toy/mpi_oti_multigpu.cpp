// Multi-GPU MPI + Kokkos pattern, simulated on a single GPU.
//
// The 1-rank-per-GPU model: each rank binds to its own device, computes its slice
// of the problem there, and the slices are combined over MPI. On a real
// multi-GPU node the ranks run concurrently (rank r -> GPU r). This machine has
// one GPU, so we *simulate* exclusive per-rank device access with a TOKEN RING:
// a rank blocks until it receives the token from the previous rank, does its
// device work while it alone holds the token, fences, then passes the token on.
// "Wait until the device is free, use it, release it."
//
// The MPI + Kokkos structure here is exactly what you would ship for a real
// multi-GPU run; only the token ring is the single-GPU stand-in (on true
// multi-GPU you drop it and let the distinct devices run in parallel).
//
// Build/run: see CMakeLists.txt in this directory.

#include <Kokkos_Core.hpp>
#include <cuda_runtime.h>
#include <mpi.h>

#include <cstdio>
#include <cstring>
#include <vector>

#include "otinum/otinum.hpp"
#include "otinum/mpi.hpp"

using Jet = oti::otinum<2, 2, double>;
static constexpr int N = 1000;
static constexpr long TOTAL = static_cast<long>(N) * N;
static constexpr int TOKEN_TAG = 99;

KOKKOS_INLINE_FUNCTION Jet evaluate(long g)
{
    const double h = 1.0 / (N - 1);
    const long i = g / N, j = g % N;
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

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Bind this rank to "its" GPU: rank % (devices on this node). With one GPU
    // every rank maps to device 0; on an N-GPU node ranks 0..N-1 get distinct
    // devices. This is the real binding code -- unchanged on multi-GPU hardware.
    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    const int my_device = num_gpus > 0 ? rank % num_gpus : 0;

    Kokkos::InitializationSettings settings;
    settings.set_device_id(my_device);
    Kokkos::initialize(settings);
    {
        MPI_Datatype MPI_OTINUM = oti::mpi::make_datatype<Jet>();

        long start = 0, count = 0;
        block_range(rank, size, start, count);

        // ---- token ring: exclusive device access, one rank at a time --------
        int token = 0;
        if (rank > 0) {
            MPI_Recv(&token, 1, MPI_INT, rank - 1, TOKEN_TAG, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        }
        // --- this rank now holds the GPU to itself ---
        const double t0 = MPI_Wtime();
        Kokkos::View<Jet*> d_local("d_local", count);
        Kokkos::parallel_for(
            "evaluate", count,
            KOKKOS_LAMBDA(int k) { d_local(k) = evaluate(start + k); });
        Kokkos::fence();   // device work must finish before we call it "free"
        const double dt = MPI_Wtime() - t0;

        auto h_local = Kokkos::create_mirror_view(d_local);
        Kokkos::deep_copy(h_local, d_local);

        std::printf("rank %d/%d: bound to GPU %d of %d | exclusive device turn: "
                    "%6.2f ms for %ld jets\n",
                    rank, size, my_device, num_gpus, dt * 1000.0, count);
        std::fflush(stdout);
        // --- release: hand the GPU to the next rank ---
        if (rank < size - 1) {
            MPI_Send(&token, 1, MPI_INT, rank + 1, TOKEN_TAG, MPI_COMM_WORLD);
        }

        // ---- combine slices over MPI (host-staged gather to rank 0) ---------
        std::vector<int> recvcounts, displs;
        std::vector<Jet> global;
        if (rank == 0) {
            recvcounts.resize(static_cast<size_t>(size));
            displs.resize(static_cast<size_t>(size));
            for (int r = 0; r < size; ++r) {
                long s = 0, c = 0;
                block_range(r, size, s, c);
                recvcounts[static_cast<size_t>(r)] = static_cast<int>(c);
                displs[static_cast<size_t>(r)] = static_cast<int>(s);
            }
            global.resize(static_cast<size_t>(TOTAL));
        }
        MPI_Gatherv(h_local.data(), static_cast<int>(count), MPI_OTINUM,
                    rank == 0 ? global.data() : nullptr,
                    rank == 0 ? recvcounts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_OTINUM, 0, MPI_COMM_WORLD);

        // ---- rank 0 verifies the assembled grid (device recompute) ----------
        if (rank == 0) {
            Kokkos::View<Jet*> d_ref("d_ref", TOTAL);
            Kokkos::parallel_for(
                "reference", TOTAL,
                KOKKOS_LAMBDA(long g) { d_ref(g) = evaluate(g); });
            auto h_ref = Kokkos::create_mirror_view(d_ref);
            Kokkos::deep_copy(h_ref, d_ref);
            long mism = 0;
            for (long g = 0; g < TOTAL; ++g) {
                if (std::memcmp(&global[static_cast<size_t>(g)], &h_ref(g),
                                sizeof(Jet)) != 0) {
                    ++mism;
                }
            }
            std::printf("\n%d ranks took exclusive turns on %d physical GPU(s); "
                        "assembled %ld jets.\nverify: %s (%ld mismatches)\n",
                        size, num_gpus, TOTAL,
                        mism == 0 ? "PASS (bit-exact)" : "FAIL", mism);
            if (num_gpus <= 1 && size > 1) {
                std::printf("note: %d ranks shared 1 GPU via the token ring "
                            "(serialized). On a %d-GPU node they would run "
                            "concurrently.\n",
                            size, size);
            }
        }

        oti::mpi::free_datatype(MPI_OTINUM);
    }
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
