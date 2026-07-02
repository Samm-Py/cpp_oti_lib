// BEFORE: a plain double MPI round trip -- no derivatives, no OTI, no Kokkos.
//
// Root assembles an input field, broadcasts a shared parameter, scatters the
// field to all ranks, each rank transforms its block, and the results are
// gathered back. This is the realistic movement pattern (Bcast + Scatter +
// Gather) with NO communication during the compute. The companion
// convert_after.cpp is the same program with otinum, to show exactly what
// changes to get derivatives.
//
// Build/run: see CMakeLists.txt (plain MPI -- mpicxx, no Kokkos).

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <vector>

using Scalar = double;                 // the type the kernel computes in

static constexpr int N = 1000;         // points in the field
static constexpr double P0 = 1.3;      // shared parameter

// THE KERNEL: f(u; p) = sin(p * u). UNCHANGED in convert_after.cpp.
static inline Scalar transform(Scalar u, Scalar p)
{
    using std::sin;       // scalar overload
    return sin(p * u);    // same unqualified call supports OTI through ADL
}

// Even block split for Scatterv / Gatherv (handles any rank count).
static void block_layout(int size, std::vector<int>& counts,
                         std::vector<int>& displs)
{
    counts.resize(size);
    displs.resize(size);
    for (int r = 0, off = 0; r < size; ++r) {
        counts[r] = N / size + (r < N % size ? 1 : 0);
        displs[r] = off;
        off += counts[r];
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Datatype field_type = MPI_DOUBLE;   // how MPI describes one element

    std::vector<int> counts, displs;
    block_layout(size, counts, displs);
    const int local_n = counts[rank];

    // broadcast the shared parameter to every rank
    Scalar p = (rank == 0) ? P0 : 0.0;
    MPI_Bcast(&p, 1, field_type, 0, MPI_COMM_WORLD);

    // root assembles the input field
    std::vector<Scalar> in;
    if (rank == 0) {
        in.resize(N);
        const double h = 1.0 / (N - 1);
        for (int g = 0; g < N; ++g) in[g] = g * h;   // the inputs
    }

    // scatter blocks -> transform -> gather results
    std::vector<Scalar> in_local(local_n);
    MPI_Scatterv(in.data(), counts.data(), displs.data(), field_type,
                 in_local.data(), local_n, field_type, 0, MPI_COMM_WORLD);

    std::vector<Scalar> out_local(local_n);
    for (int k = 0; k < local_n; ++k) out_local[k] = transform(in_local[k], p);

    std::vector<Scalar> out;
    if (rank == 0) out.resize(N);
    MPI_Gatherv(out_local.data(), local_n, field_type,
                out.data(), counts.data(), displs.data(), field_type,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        const int mid = N / 2;
        std::printf("sample value = %.8f\n", out[mid]);
    }

    MPI_Finalize();
    return 0;
}
