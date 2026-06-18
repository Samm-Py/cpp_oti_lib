// BEFORE: an ordinary Kokkos + MPI program in plain `double`.
//
// Evaluates a field model(x,y) = sin(x)*exp(y) over a 1000x1000 grid distributed
// across ranks (one block per rank, computed on the Kokkos device), gathers the
// field to rank 0, and prints a sample value. No derivatives.
//
// The companion convert_after.cpp is the same program with otinum, to show
// exactly what changes to get derivatives. Build/run: see CMakeLists.txt.

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <vector>

using Scalar = double;                       // the type the kernel computes in

static constexpr int N = 1000;
static constexpr long TOTAL = static_cast<long>(N) * N;

// The model. Plain overloaded arithmetic + elementary functions.
KOKKOS_INLINE_FUNCTION Scalar model(Scalar x, Scalar y)
{
    return sin(x) * exp(y);
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
    Kokkos::initialize(argc, argv);
    {
        int rank = 0, size = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        MPI_Datatype field_type = MPI_DOUBLE;        // how MPI describes one element

        long start = 0, count = 0;
        block_range(rank, size, start, count);

        Kokkos::View<Scalar*> d_field("d_field", count);
        Kokkos::parallel_for(
            "evaluate", count, KOKKOS_LAMBDA(int k) {
                const long g = start + k;
                const double h = 1.0 / (N - 1);
                Scalar x = (g / N) * h;              // the inputs
                Scalar y = (g % N) * h;
                d_field(k) = model(x, y);
            });
        auto h_field = Kokkos::create_mirror_view(d_field);
        Kokkos::deep_copy(h_field, d_field);

        std::vector<int> recvcounts, displs;
        std::vector<Scalar> global;
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
        MPI_Gatherv(h_field.data(), static_cast<int>(count), field_type,
                    rank == 0 ? global.data() : nullptr,
                    rank == 0 ? recvcounts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    field_type, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            const long mid = (TOTAL / 2) + (N / 2);
            std::printf("sample value = %.8f\n", global[static_cast<size_t>(mid)]);
        }
    }
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
