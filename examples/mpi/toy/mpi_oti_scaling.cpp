// Rank-scaling benchmark for the embarrassingly-parallel OTI grid evaluation.
//
// Each rank evaluates its block of a 1000x1000 grid of f(x,y)=sin(x)*exp(y)
// (jets carrying derivatives) and we time that compute region -- the
// parallelizable work -- across ranks. The one-time gather is communication, not
// the workload, so it is excluded here and discussed separately in the tutorial.
//
// Runs all four study algebras in one invocation. Emits one CSV row per
// (algebra, repetition): the slowest rank's per-iteration compute time, which is
// what sets the wall clock. A driver sweeps np and a plotter turns it into
// speedup / efficiency curves.
//
// Build: mpicxx -std=c++17 -O2 -I ../../../include mpi_oti_scaling.cpp -o mpi_oti_scaling
// Run:   mpirun -np 4 ./mpi_oti_scaling   (CSV to stdout; header only on rank 0)

#include <mpi.h>

#include <cstdio>
#include <vector>

#include "otinum/otinum.hpp"

static constexpr int N = 1000;
static constexpr long TOTAL = static_cast<long>(N) * N;
static constexpr int REPS = 11;

template <class T>
static T evaluate(long g)
{
    using Coeff = typename T::coeff_type;
    const Coeff h = Coeff(1) / Coeff(N - 1);
    const long i = g / N;
    const long j = g % N;
    T x = T::variable(0, static_cast<Coeff>(i) * h);
    T y = T::variable(1, static_cast<Coeff>(j) * h);
    return oti::sin(x) * oti::exp(y);
}

static void block_range(int rank, int size, long& start, long& count)
{
    const long base = TOTAL / size;
    const long rem = TOTAL % size;
    count = base + (rank < rem ? 1 : 0);
    start = rank * base + (rank < rem ? rank : rem);
}

template <class T>
static void run_algebra(char const* name, int rank, int size)
{
    long start = 0, count = 0;
    block_range(rank, size, start, count);
    std::vector<T> local(static_cast<size_t>(count));

    double sink = 0.0;   // keep the compute from being optimized away
    for (int rep = 0; rep < REPS; ++rep) {
        MPI_Barrier(MPI_COMM_WORLD);
        const double t0 = MPI_Wtime();
        for (long k = 0; k < count; ++k) {
            local[static_cast<size_t>(k)] = evaluate<T>(start + k);
        }
        const double elapsed = MPI_Wtime() - t0;
        sink += static_cast<double>(local[static_cast<size_t>(count / 2)][0]);

        // wall clock is set by the slowest rank
        double max_elapsed = 0.0;
        MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
                   MPI_COMM_WORLD);
        if (rank == 0) {
            std::printf("\"%s\",%d,%d,%d,%.9f\n", name, T::ncoeffs, size, rep + 1,
                        max_elapsed);
        }
    }
    // ensure sink is observable (prevents dead-code elimination of the loop)
    if (rank < 0) {
        std::printf("%g\n", sink);
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        std::printf("algebra,ncoeffs,ranks,repetition,seconds\n");
    }

    run_algebra<oti::otinum<2, 1, float>>("<2,1,float>", rank, size);
    run_algebra<oti::otinum<2, 1, double>>("<2,1,double>", rank, size);
    run_algebra<oti::otinum<2, 2, float>>("<2,2,float>", rank, size);
    run_algebra<oti::otinum<2, 2, double>>("<2,2,double>", rank, size);

    MPI_Finalize();
    return 0;
}
