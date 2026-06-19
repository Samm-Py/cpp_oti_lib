// The plain-double starting point that mpi_oti_reduce/main.cpp OTI-enables.
//
// It computes the same global quantity of interest -- the mean of
// f(x, y; a, b) = sin(a*x) * exp(b*y) over an N x N grid -- with an ordinary
// MPI_Allreduce(MPI_SUM) over doubles. It gets the *value* and nothing else.
//
// Converting it to OTI (see main.cpp and converting/reduce.rst) is five changes:
// swap the scalar type for a jet, seed a and b as variables, register a custom
// MPI_Op that sums jets, reduce over MPI_OTINUM instead of MPI_DOUBLE, and read
// the derivative coefficients out. The QoI then carries its gradient and Hessian
// w.r.t. (a, b). The summation loop and the decomposition do not change.
//
// Build: mpicxx -std=c++17 -O2 main_before.cpp -o mpi_reduce_before
// Run:   mpirun -np 4 ./mpi_reduce_before

#include <mpi.h>

#include <cmath>
#include <cstdio>

// ---- problem definition ----------------------------------------------------

using Scalar = double;
static constexpr int  N      = 1000;
static constexpr long TOTAL  = static_cast<long>(N) * N;
static constexpr double A0    = 1.0;     // design parameter a
static constexpr double B0    = 1.0;     // design parameter b
static constexpr double SER_TOL = 1.0e-10;

static inline void coords(long g, double& x, double& y)
{
    const double h = 1.0 / (N - 1);
    x = (g / N) * h;
    y = (g % N) * h;
}

static inline Scalar field(long g, Scalar a, Scalar b)
{
    double x, y;
    coords(g, x, y);
    return std::sin(a * x) * std::exp(b * y);
}

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

    // design parameters (plain numbers)
    const Scalar a = A0;
    const Scalar b = B0;

    long start = 0, count = 0;
    block_range(rank, size, start, count);

    Scalar local = 0.0;
    for (long k = 0; k < count; ++k) local += field(start + k, a, b);

    Scalar global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    const Scalar qoi = global / static_cast<double>(TOTAL);   // mean field

    int failures = 0;
    if (rank == 0) {
        std::printf("ranks              : %d\n", size);
        std::printf("grid               : %d x %d  (%ld points)\n", N, N, TOTAL);
        std::printf("QoI = mean of sin(a x) exp(b y)  at a=%.3f, b=%.3f\n", A0, B0);
        std::printf("  value            = % .10f\n", qoi);

        Scalar ser = 0.0;
        for (long g = 0; g < TOTAL; ++g) ser += field(g, a, b);
        ser /= static_cast<double>(TOTAL);
        const double rel = std::fabs(qoi - ser) / (std::fabs(ser) + 1e-300);
        const bool ok = rel <= SER_TOL;
        failures += ok ? 0 : 1;
        std::printf("verify vs serial   : %s (max relative diff %.2e)\n",
                    ok ? "PASS" : "FAIL", rel);
    }

    MPI_Bcast(&failures, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return failures == 0 ? 0 : 1;
}
