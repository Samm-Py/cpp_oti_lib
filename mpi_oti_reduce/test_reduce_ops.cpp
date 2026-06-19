// Confidence test for the oti::mpi reduction operators (sum / prod / max / min).
//
// Each rank contributes an array of COUNT jets a[j] = v(rank, j) + e_0 (every
// jet seeds the same first-order variable). All ranks reduce the array four ways
// with the header's committed MPI_Ops, and rank 0 checks each result against a
// serial recompute it derives independently from the rank count (v is a known
// function of rank and index). This validates the value AND the derivative of
// each combine, and exercises the *len > 1 path of the user function.
//
// Build: mpicxx -std=c++17 -O2 -I ../include test_reduce_ops.cpp -o test_reduce_ops
// Run:   mpirun -np 4 ./test_reduce_ops

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "otinum/otinum.hpp"
#include "otinum/mpi.hpp"

using Jet = oti::otinum<1, 1, double>;   // value + d/de_0
static constexpr int COUNT = 3;          // elements per reduction (exercises *len)
static constexpr double TOL = 1.0e-10;

static double v(int r, int j) { return 0.5 + 0.1 * r + 0.01 * j; }

static bool close_jet(const Jet& a, const Jet& b)
{
    for (int c = 0; c < Jet::ncoeffs; ++c) {
        const double s = b[c];
        if (std::fabs(a[c] - s) / (std::fabs(s) + 1e-300) > TOL) return false;
    }
    return true;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Datatype OT = oti::mpi::make_datatype<Jet>();
    MPI_Op SUM  = oti::mpi::make_sum_op<Jet>();
    MPI_Op PROD = oti::mpi::make_prod_op<Jet>();
    MPI_Op MAX  = oti::mpi::make_max_op<Jet>();
    MPI_Op MIN  = oti::mpi::make_min_op<Jet>();

    std::vector<Jet> local(COUNT);
    for (int j = 0; j < COUNT; ++j) local[j] = Jet::variable(0, v(rank, j));

    std::vector<Jet> r_sum(COUNT), r_prod(COUNT), r_max(COUNT), r_min(COUNT);
    MPI_Allreduce(local.data(), r_sum.data(),  COUNT, OT, SUM,  MPI_COMM_WORLD);
    MPI_Allreduce(local.data(), r_prod.data(), COUNT, OT, PROD, MPI_COMM_WORLD);
    MPI_Allreduce(local.data(), r_max.data(),  COUNT, OT, MAX,  MPI_COMM_WORLD);
    MPI_Allreduce(local.data(), r_min.data(),  COUNT, OT, MIN,  MPI_COMM_WORLD);

    int failures = 0;
    if (rank == 0) {
        bool ok_sum = true, ok_prod = true, ok_max = true, ok_min = true;
        for (int j = 0; j < COUNT; ++j) {
            Jet s(0.0), p(1.0), mx, mn;
            bool first = true;
            for (int r = 0; r < size; ++r) {
                const Jet g = Jet::variable(0, v(r, j));
                s = s + g;
                p = p * g;                       // jet product (convolution)
                if (first || g[0] > mx[0]) mx = g;
                if (first || g[0] < mn[0]) mn = g;
                first = false;
            }
            ok_sum  = ok_sum  && close_jet(r_sum[j], s);
            ok_prod = ok_prod && close_jet(r_prod[j], p);
            ok_max  = ok_max  && close_jet(r_max[j], mx);
            ok_min  = ok_min  && close_jet(r_min[j], mn);
        }
        std::printf("oti::mpi reduction-op test (%d ranks, %d elems/reduce)\n",
                    size, COUNT);
        std::printf("  make_sum_op  : %s\n", ok_sum  ? "PASS" : "FAIL");
        std::printf("  make_prod_op : %s\n", ok_prod ? "PASS" : "FAIL");
        std::printf("  make_max_op  : %s\n", ok_max  ? "PASS" : "FAIL");
        std::printf("  make_min_op  : %s\n", ok_min  ? "PASS" : "FAIL");
        std::printf("  sample[0]: sum=%.4f (d=%.1f)  prod=%.6f (d=%.4f)  "
                    "max=%.4f (d=%.1f)\n",
                    r_sum[0][0], r_sum[0][1], r_prod[0][0], r_prod[0][1],
                    r_max[0][0], r_max[0][1]);
        failures = (ok_sum && ok_prod && ok_max && ok_min) ? 0 : 1;
    }

    oti::mpi::free_op(SUM);
    oti::mpi::free_op(PROD);
    oti::mpi::free_op(MAX);
    oti::mpi::free_op(MIN);
    oti::mpi::free_datatype(OT);
    MPI_Bcast(&failures, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return failures == 0 ? 0 : 1;
}
