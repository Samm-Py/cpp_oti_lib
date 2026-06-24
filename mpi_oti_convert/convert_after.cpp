// AFTER: the same MPI round trip with otinum, so the gathered field now carries
// derivatives. Compare against convert_before.cpp -- the differences are exactly
// what it takes to OTI-enable an existing MPI program:
//
//   1. include the optional headers (otinum core + the MPI datatype helper)
//   2. change the scalar type alias to an otinum
//   3. seed the inputs as infinitesimal variables (the field u and the param p)
//   4. describe the element to MPI with make_datatype instead of MPI_DOUBLE
//   5. read derivatives out of the result
//
// The transform() kernel is UNCHANGED -- the overloaded arithmetic and
// elementary functions carry the derivatives through. No Kokkos, no GPU: plain
// MPI. The committed datatype serves Bcast, Scatter, and Gather alike. Because
// the per-element computation is unchanged, moving the finished jets introduces
// no rounding and the gathered field is bit-identical to the serial run.
// Build/run: see CMakeLists.txt.

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "otinum/otinum.hpp"   // 1. otinum core + oti:: math overloads
#include "otinum/mpi.hpp"      //    optional MPI datatype helper

using Scalar = oti::otinum<2, 1, double>;   // 2. was: using Scalar = double;
                                            //    dir 0 = field u, dir 1 = param p

static constexpr int N = 1000;
static constexpr double P0 = 1.3;

// THE KERNEL: f(u; p) = sin(p * u). UNCHANGED from convert_before.cpp.
static inline Scalar transform(Scalar u, Scalar p)
{
    using std::sin;       // scalar fallback
    return sin(p * u);    // ADL selects oti::sin for an otinum
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

    MPI_Datatype field_type = oti::mpi::make_datatype<Scalar>();   // 4.

    std::vector<int> counts, displs;
    block_layout(size, counts, displs);
    const int local_n = counts[rank];

    // 3. seed + broadcast the shared parameter as a variable
    Scalar p;
    if (rank == 0) p = Scalar::variable(1, P0);   // P0 + e_1 (seed d/dp)
    MPI_Bcast(&p, 1, field_type, 0, MPI_COMM_WORLD);

    // root assembles the input field, each point seeded as a variable
    std::vector<Scalar> in;
    if (rank == 0) {
        in.resize(N);
        const double h = 1.0 / (N - 1);
        for (int g = 0; g < N; ++g)
            in[g] = Scalar::variable(0, g * h);   // 3. seed as e_0 (seed d/du)
    }

    // scatter blocks -> transform -> gather results  (control flow unchanged)
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
        const Scalar& s = out[mid];
        // 5. read value + derivatives out of the jet
        std::printf("sample value = %.8f\n", s.coeff(oti::sparse({})));
        std::printf("        d/du = %.8f\n", s.coeff(oti::sparse({{0, 1}})));
        std::printf("        d/dp = %.8f\n", s.coeff(oti::sparse({{1, 1}})));
    }

    oti::mpi::free_datatype(field_type);   // 4. (release the committed type)
    MPI_Finalize();
    return 0;
}
