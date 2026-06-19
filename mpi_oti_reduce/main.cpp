// MPI + OTI global reduction: a custom MPI_Op that sums jets.
//
// Each rank owns a block of an N x N grid and accumulates a partial sum of
// f(x, y; a, b) = sin(a*x) * exp(b*y), where the design parameters a, b are
// SEEDED as OTI variables. One MPI_Allreduce with a custom jet-sum operator
// folds the per-rank partial-sum jets into the global quantity of interest --
// its value, gradient, AND Hessian w.r.t. (a, b) -- on every rank, from a single
// reduction. That is the gradient and Hessian of a global objective over a
// distributed field, with no adjoint and no extra communication.
//
// MPI has no built-in way to combine an otinum, so we register a custom MPI_Op
// that adds two jets coefficient-wise (otinum::operator+=). For a SUM this is the
// same as MPI_SUM over the raw coefficients, but the custom op is the general
// mechanism -- it would wrap any associative jet combine -- and it lets the
// reduction speak in MPI_OTINUM units, consistent with the rest of the section.
//
// Verification differs from the gather/halo toys: a floating-point sum is not
// associative, so the distributed result is NOT bit-identical to a serial sum --
// the partition changes the summation order. We instead check the global jet
// against a serial recompute to a tight relative tolerance, and check the
// gradient independently against centred finite differences on a, b.
//
// Build: mpicxx -std=c++17 -O2 -I ../include main.cpp -o mpi_oti_reduce
// Run:   mpirun -np 4 ./mpi_oti_reduce

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "otinum/otinum.hpp"
#include "otinum/mpi.hpp"

// ---- problem definition ----------------------------------------------------

using Jet = oti::otinum<2, 2, double>;   // value + grad + Hessian w.r.t. (a, b)
static constexpr int  N      = 1000;     // grid is N x N
static constexpr long TOTAL  = static_cast<long>(N) * N;
static constexpr double A0    = 1.0;     // design parameter a
static constexpr double B0    = 1.0;     // design parameter b
static constexpr double FD_H  = 1.0e-6;  // finite-difference step
static constexpr double FD_TOL = 1.0e-6; // gradient vs finite difference
static constexpr double SER_TOL = 1.0e-10; // distributed vs serial (rel)

// Physical coordinates of a flat grid index on the unit square.
static inline void coords(long g, double& x, double& y)
{
    const double h = 1.0 / (N - 1);
    x = (g / N) * h;
    y = (g % N) * h;
}

// f(x, y; a, b) = sin(a*x) * exp(b*y), with a, b carried as jets.
static inline Jet field_jet(long g, const Jet& a, const Jet& b)
{
    double x, y;
    coords(g, x, y);
    return oti::sin(a * x) * oti::exp(b * y);
}

// Plain-double version for the finite-difference reference.
static inline double field_double(long g, double a, double b)
{
    double x, y;
    coords(g, x, y);
    return std::sin(a * x) * std::exp(b * y);
}

// Mean of the field over the whole grid, in plain double (for finite differences).
static double mean_double(double a, double b)
{
    double s = 0.0;
    for (long g = 0; g < TOTAL; ++g) s += field_double(g, a, b);
    return s / static_cast<double>(TOTAL);
}

// ---- the custom MPI reduction operator -------------------------------------
// Combine two buffers of `*len` jets by adding them coefficient-wise. This is
// exactly otinum::operator+=, so OTI arithmetic plugs straight into MPI.
static void jet_sum(void* in, void* inout, int* len, MPI_Datatype* /*dt*/)
{
    const Jet* a = static_cast<const Jet*>(in);
    Jet* b = static_cast<Jet*>(inout);
    for (int i = 0; i < *len; ++i) b[i] += a[i];
}

// ---- block decomposition (same flat partition as the gather toy) -----------
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

    // ---- 1. commit the datatype and register the custom reduction op -------
    MPI_Datatype MPI_OTINUM = oti::mpi::make_datatype<Jet>();
    MPI_Op MPI_OTI_SUM;
    MPI_Op_create(&jet_sum, /*commute=*/1, &MPI_OTI_SUM);

    // ---- 2. seed the design parameters and sum this rank's block -----------
    const Jet a = Jet::variable(0, A0);   // A0 + e_0
    const Jet b = Jet::variable(1, B0);   // B0 + e_1

    long start = 0, count = 0;
    block_range(rank, size, start, count);

    Jet local(0.0);
    for (long k = 0; k < count; ++k) local += field_jet(start + k, a, b);

    // ---- 3. one Allreduce folds the partial-sum jets into the global QoI ----
    Jet global(0.0);
    MPI_Allreduce(&local, &global, 1, MPI_OTINUM, MPI_OTI_SUM, MPI_COMM_WORLD);
    const Jet qoi = global * (1.0 / static_cast<double>(TOTAL));   // mean field

    // ---- 4. report + verify on rank 0 --------------------------------------
    int failures = 0;
    if (rank == 0) {
        const double val  = qoi.coeff(oti::sparse({}));
        const double d_da = qoi.coeff(oti::sparse({{0, 1}}));
        const double d_db = qoi.coeff(oti::sparse({{1, 1}}));

        std::printf("ranks              : %d\n", size);
        std::printf("grid               : %d x %d  (%ld points)\n", N, N, TOTAL);
        std::printf("QoI = mean of sin(a x) exp(b y)  at a=%.3f, b=%.3f\n", A0, B0);
        std::printf("  value            = % .10f\n", val);
        std::printf("  d/da             = % .10f\n", d_da);
        std::printf("  d/db             = % .10f\n", d_db);
        std::printf("  d2/da2  (Taylor) = % .10f  (= QoI_aa / 2)\n",
                    qoi.coeff(oti::sparse({{0, 2}})));
        std::printf("  d2/dadb (Taylor) = % .10f  (= QoI_ab)\n",
                    qoi.coeff(oti::sparse({{0, 1}, {1, 1}})));
        std::printf("  d2/db2  (Taylor) = % .10f  (= QoI_bb / 2)\n",
                    qoi.coeff(oti::sparse({{1, 2}})));

        // (a) distributed vs serial recompute (relative -- sums are not associative)
        Jet ser(0.0);
        for (long g = 0; g < TOTAL; ++g) ser += field_jet(g, a, b);
        ser = ser * (1.0 / static_cast<double>(TOTAL));
        double max_rel = 0.0;
        for (int c = 0; c < Jet::ncoeffs; ++c) {
            const double s = ser[c];
            const double rel = std::fabs(qoi[c] - s) / (std::fabs(s) + 1e-300);
            if (rel > max_rel) max_rel = rel;
        }
        const bool ser_ok = max_rel <= SER_TOL;
        failures += ser_ok ? 0 : 1;
        std::printf("verify vs serial   : %s (max relative diff %.2e)\n",
                    ser_ok ? "PASS" : "FAIL", max_rel);

        // (b) gradient vs centred finite differences on a, b
        const double fd_da = (mean_double(A0 + FD_H, B0) -
                              mean_double(A0 - FD_H, B0)) / (2 * FD_H);
        const double fd_db = (mean_double(A0, B0 + FD_H) -
                              mean_double(A0, B0 - FD_H)) / (2 * FD_H);
        const double e_da = std::fabs(d_da - fd_da);
        const double e_db = std::fabs(d_db - fd_db);
        const bool fd_ok = e_da <= FD_TOL && e_db <= FD_TOL;
        failures += fd_ok ? 0 : 1;
        std::printf("finite difference  : centred, h = %.1e\n", FD_H);
        std::printf("  d/da : OTI = % .10f, FD = % .10f, |error| = %.2e\n",
                    d_da, fd_da, e_da);
        std::printf("  d/db : OTI = % .10f, FD = % .10f, |error| = %.2e\n",
                    d_db, fd_db, e_db);
        std::printf("verify gradient    : %s (tolerance %.1e)\n",
                    fd_ok ? "PASS" : "FAIL", FD_TOL);
    }

    // ---- 5. teardown -------------------------------------------------------
    MPI_Op_free(&MPI_OTI_SUM);
    oti::mpi::free_datatype(MPI_OTINUM);
    MPI_Bcast(&failures, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return failures == 0 ? 0 : 1;
}
