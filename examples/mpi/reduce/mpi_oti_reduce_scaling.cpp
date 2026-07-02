// Strong-scaling harness for the global-reduction tutorial.
//
// Measures the two algorithmic components separately:
//   1. local accumulation of this rank's fixed-size grid block;
//   2. one MPI_Allreduce of the partial result.
//
// Both the plain-double baseline and the OTI value+gradient+Hessian path are
// timed. The finite-difference and serial verification work from main.cpp is
// intentionally absent: this executable measures the distributed algorithm,
// not its test harness.

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "otinum/otinum.hpp"
#include "otinum/mpi.hpp"

using Jet = oti::otinum<2, 2, double>;

static constexpr int N = 1000;
static constexpr long TOTAL = static_cast<long>(N) * N;
static constexpr double A0 = 1.0;
static constexpr double B0 = 1.0;
static constexpr int REPEATS = 7;
static constexpr int REDUCE_REPEATS = 2000;

static inline void coords(long g, double& x, double& y)
{
    const double h = 1.0 / (N - 1);
    x = (g / N) * h;
    y = (g % N) * h;
}

static inline double field_double(long g)
{
    double x, y;
    coords(g, x, y);
    return std::sin(A0 * x) * std::exp(B0 * y);
}

static inline Jet field_jet(long g, const Jet& a, const Jet& b)
{
    double x, y;
    coords(g, x, y);
    return oti::sin(a * x) * oti::exp(b * y);
}

static void block_range(int rank, int size, long& start, long& count)
{
    const long base = TOTAL / size;
    const long rem = TOTAL % size;
    count = base + (rank < rem ? 1 : 0);
    start = rank * base + (rank < rem ? rank : rem);
}

static double median(std::vector<double> samples)
{
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long start = 0, count = 0;
    block_range(rank, size, start, count);

    MPI_Datatype MPI_OTINUM = oti::mpi::make_datatype<Jet>();
    MPI_Op MPI_OTI_SUM = oti::mpi::make_sum_op<Jet>();
    const Jet a = Jet::variable(0, A0);
    const Jet b = Jet::variable(1, B0);

    std::vector<double> double_compute, jet_compute;
    double_compute.reserve(REPEATS);
    jet_compute.reserve(REPEATS);

    double double_local = 0.0, double_global = 0.0;
    Jet jet_local(0.0), jet_global(0.0);

    for (int rep = 0; rep < REPEATS; ++rep) {
        MPI_Barrier(MPI_COMM_WORLD);
        const double t0 = MPI_Wtime();
        double_local = 0.0;
        for (long k = 0; k < count; ++k) double_local += field_double(start + k);
        const double local_double_time = MPI_Wtime() - t0;
        double max_double_time = 0.0;
        MPI_Reduce(&local_double_time, &max_double_time, 1, MPI_DOUBLE, MPI_MAX,
                   0, MPI_COMM_WORLD);
        if (rank == 0) double_compute.push_back(max_double_time);

        MPI_Barrier(MPI_COMM_WORLD);
        const double t1 = MPI_Wtime();
        jet_local = Jet(0.0);
        for (long k = 0; k < count; ++k)
            jet_local += field_jet(start + k, a, b);
        const double local_jet_time = MPI_Wtime() - t1;
        double max_jet_time = 0.0;
        MPI_Reduce(&local_jet_time, &max_jet_time, 1, MPI_DOUBLE, MPI_MAX,
                   0, MPI_COMM_WORLD);
        if (rank == 0) jet_compute.push_back(max_jet_time);
    }

    // Warm the collectives before timing their steady-state latency.
    MPI_Allreduce(&double_local, &double_global, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&jet_local, &jet_global, 1, MPI_OTINUM, MPI_OTI_SUM,
                  MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int rep = 0; rep < REDUCE_REPEATS; ++rep)
        MPI_Allreduce(&double_local, &double_global, 1, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
    const double double_reduce_local =
        (MPI_Wtime() - t0) / static_cast<double>(REDUCE_REPEATS);

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int rep = 0; rep < REDUCE_REPEATS; ++rep)
        MPI_Allreduce(&jet_local, &jet_global, 1, MPI_OTINUM, MPI_OTI_SUM,
                      MPI_COMM_WORLD);
    const double jet_reduce_local =
        (MPI_Wtime() - t0) / static_cast<double>(REDUCE_REPEATS);

    double double_reduce = 0.0, jet_reduce = 0.0;
    MPI_Reduce(&double_reduce_local, &double_reduce, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&jet_reduce_local, &jet_reduce, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        const double dc = median(double_compute);
        const double jc = median(jet_compute);
        std::printf("ranks,mode,compute_s,allreduce_us,total_s,checksum\n");
        std::printf("%d,double,%.9g,%.9g,%.9g,%.17g\n", size, dc,
                    double_reduce * 1.0e6, dc + double_reduce, double_global);
        std::printf("%d,oti_2_2_double,%.9g,%.9g,%.9g,%.17g\n", size, jc,
                    jet_reduce * 1.0e6, jc + jet_reduce, jet_global[0]);
    }

    oti::mpi::free_op(MPI_OTI_SUM);
    oti::mpi::free_datatype(MPI_OTINUM);
    MPI_Finalize();
    return 0;
}
