// Minimal MPI + OTI toy: evaluate a function over a 2D grid, embarrassingly
// parallel (NO ghost nodes, NO halo exchange), then gather every jet back to
// rank 0 using a *committed* MPI datatype for one otinum.
//
// The point of this toy is to validate exactly one thing: that MPI can move an
// `otinum<M,N>` as a first-class element. Everything else (stencils, halos,
// unstructured ghost lists) builds derived datatypes on top of the base element
// proven here.
//
// Build (standalone, no project CMake needed):
//   mpicxx -std=c++17 -O2 -I ../include main.cpp -o mpi_oti_toy
// Run:
//   mpirun -np 4 ./mpi_oti_toy

#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "otinum/otinum.hpp"
#include "otinum/mpi.hpp"   // optional MPI interop: oti::mpi::make_datatype

// ---- problem definition ----------------------------------------------------

using Jet = oti::otinum<2, 2, double>;   // value + grad + Hessian w.r.t. (x, y)
static constexpr int N = 1000;           // grid is N x N
static constexpr long TOTAL = static_cast<long>(N) * N;

// Map a flat global index to physical coords on the unit square, then evaluate
// f(x,y) = sin(x) * exp(y) with x,y seeded as infinitesimals. The returned jet
// carries f and all of its first/second Taylor coefficients at that point.
//
// This is plain single-process OTI math -- MPI never sees inside it. Serial
// reference and parallel workers call the SAME function, so results are
// bit-identical and we can verify with memcmp.
static Jet evaluate(long global_index)
{
    const double h = 1.0 / (N - 1);
    const long i = global_index / N;     // row
    const long j = global_index % N;     // col
    const double x0 = i * h;
    const double y0 = j * h;

    Jet x = Jet::variable(0, x0);        // x0 + e_0
    Jet y = Jet::variable(1, y0);        // y0 + e_1
    return oti::sin(x) * oti::exp(y);
}

// ---- block decomposition ---------------------------------------------------

// Split TOTAL points across `size` ranks as evenly as possible: the first
// `rem` ranks get one extra point. No communication needed -- every rank
// derives its own [start, start+count) range from (rank, size) arithmetic.
static void block_range(int rank, int size, long& start, long& count)
{
    const long base = TOTAL / size;
    const long rem  = TOTAL % size;
    count = base + (rank < rem ? 1 : 0);
    start = rank * base + (rank < rem ? rank : rem);
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ---- 1. commit the OTI datatype: one jet == ncoeffs contiguous doubles --
    // make_datatype<Jet>() builds + commits MPI_Type_contiguous(ncoeffs, MPI_DOUBLE)
    // and static_asserts the jet is tightly packed -- consumers don't re-roll it.
    MPI_Datatype MPI_OTINUM = oti::mpi::make_datatype<Jet>();

    // ---- 2. each rank evaluates its own slice into a local buffer ----------
    long start = 0, count = 0;
    block_range(rank, size, start, count);

    std::vector<Jet> local_buf(static_cast<size_t>(count));
    for (long k = 0; k < count; ++k) {
        local_buf[static_cast<size_t>(k)] = evaluate(start + k);
    }

    // ---- 3. gather every jet back to rank 0 --------------------------------
    // Counts/displacements are in units of MPI_OTINUM -- no byte arithmetic.
    // (Reduces to a plain MPI_Gather when size divides TOTAL evenly.)
    std::vector<int> recvcounts, displs;
    std::vector<Jet> global_buf;
    if (rank == 0) {
        recvcounts.resize(static_cast<size_t>(size));
        displs.resize(static_cast<size_t>(size));
        for (int r = 0; r < size; ++r) {
            long s = 0, c = 0;
            block_range(r, size, s, c);
            recvcounts[static_cast<size_t>(r)] = static_cast<int>(c);
            displs[static_cast<size_t>(r)]     = static_cast<int>(s);
        }
        global_buf.resize(static_cast<size_t>(TOTAL));
    }

    MPI_Gatherv(local_buf.data(), static_cast<int>(count), MPI_OTINUM,
                rank == 0 ? global_buf.data() : nullptr,
                rank == 0 ? recvcounts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_OTINUM, 0, MPI_COMM_WORLD);

    // ---- 4. verify on rank 0 against a serial recompute --------------------
    if (rank == 0) {
        long mismatches = 0;
        for (long g = 0; g < TOTAL; ++g) {
            Jet ref = evaluate(g);
            if (std::memcmp(&ref, &global_buf[static_cast<size_t>(g)],
                            sizeof(Jet)) != 0) {
                ++mismatches;
            }
        }

        // Show one sample jet (grid centre) with its named Taylor coefficients
        // so it's visible the gather moved the *whole* jet, not just the value.
        const long mid = (TOTAL / 2) + (N / 2);
        const Jet& s = global_buf[static_cast<size_t>(mid)];
        std::printf("ranks            : %d\n", size);
        std::printf("grid             : %d x %d  (%ld points)\n", N, N, TOTAL);
        std::printf("sample @ index %ld (centre):\n", mid);
        std::printf("  value          = % .8f\n", s.coeff(oti::sparse({})));
        std::printf("  d/dx  (Taylor) = % .8f\n", s.coeff(oti::sparse({{0, 1}})));
        std::printf("  d/dy  (Taylor) = % .8f\n", s.coeff(oti::sparse({{1, 1}})));
        std::printf("  d2/dx2(Taylor) = % .8f  (= f_xx / 2)\n",
                    s.coeff(oti::sparse({{0, 2}})));
        std::printf("  d2/dxdy(Taylor)= % .8f  (= f_xy)\n",
                    s.coeff(oti::sparse({{0, 1}, {1, 1}})));
        std::printf("  d2/dy2(Taylor) = % .8f  (= f_yy / 2)\n",
                    s.coeff(oti::sparse({{1, 2}})));
        std::printf("verify vs serial : %s (%ld mismatching jets)\n",
                    mismatches == 0 ? "PASS (bit-exact)" : "FAIL", mismatches);
    }

    // ---- 5. teardown -------------------------------------------------------
    oti::mpi::free_datatype(MPI_OTINUM);
    MPI_Finalize();
    return 0;
}
