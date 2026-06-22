// Confidence test for the oti::mpi reduction operators
// (sum / prod / maxloc / minloc).
//
// Each rank contributes an array of COUNT jets whose values and derivatives are
// known functions of rank. All ranks reduce the array four ways
// with the header's committed MPI_Ops, and rank 0 checks each result against a
// serial recompute it derives independently from the rank count (v is a known
// function of rank and index). This validates the value AND the derivative of
// each combine, and exercises the *len > 1 path of the user function.
//
// It also runs MPI_Reduce (root-only) alongside MPI_Allreduce for every op and
// checks that root gets the same result to the test tolerance. MPI may choose
// different reduction trees for the two collectives, so bit identity is not a
// portable requirement for floating-point combines.
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
using Located = oti::mpi::value_loc<Jet>;
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

static bool close_loc(const Located& a, const Located& b)
{
    return a.location == b.location && close_jet(a.value, b.value);
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Datatype OT = oti::mpi::make_datatype<Jet>();
    MPI_Datatype OT_LOC = oti::mpi::make_value_loc_datatype<Jet>();
    MPI_Op SUM  = oti::mpi::make_sum_op<Jet>();
    MPI_Op PROD = oti::mpi::make_prod_op<Jet>();
    MPI_Op MAXLOC = oti::mpi::make_maxloc_op<Jet>();
    MPI_Op MINLOC = oti::mpi::make_minloc_op<Jet>();

    int loc_type_size = 0;
    MPI_Aint loc_lb = 0, loc_extent = 0;
    MPI_Type_size(OT_LOC, &loc_type_size);
    MPI_Type_get_extent(OT_LOC, &loc_lb, &loc_extent);
    const bool ok_loc_type =
        loc_type_size == Jet::ncoeffs * static_cast<int>(sizeof(double)) +
                             static_cast<int>(sizeof(int)) &&
        loc_lb == 0 && loc_extent == static_cast<MPI_Aint>(sizeof(Located));

    std::vector<Jet> local(COUNT);
    std::vector<Located> local_loc(COUNT);
    for (int j = 0; j < COUNT; ++j) {
        Jet jet(v(rank, j));
        jet[1] = static_cast<double>(rank + 1);
        // The last element is an exact value tie across ranks; the lower rank
        // must win even though each tied jet carries a different derivative.
        if (j == COUNT - 1) jet[0] = 1.0;
        local[j] = jet;
        local_loc[j] = Located{jet, rank};
    }

    std::vector<Jet> r_sum(COUNT), r_prod(COUNT);
    std::vector<Located> r_maxloc(COUNT), r_minloc(COUNT);
    MPI_Allreduce(local.data(), r_sum.data(),  COUNT, OT, SUM,  MPI_COMM_WORLD);
    MPI_Allreduce(local.data(), r_prod.data(), COUNT, OT, PROD, MPI_COMM_WORLD);
    MPI_Allreduce(local_loc.data(), r_maxloc.data(), COUNT, OT_LOC, MAXLOC,
                  MPI_COMM_WORLD);
    MPI_Allreduce(local_loc.data(), r_minloc.data(), COUNT, OT_LOC, MINLOC,
                  MPI_COMM_WORLD);

    // The custom MPI_Op is orthogonal to the collective: MPI_Reduce (root-only)
    // and MPI_Allreduce should agree to tolerance on root. Their internal trees
    // need not be identical, so exact floating-point equality is not required.
    std::vector<Jet> q_sum(COUNT), q_prod(COUNT);
    std::vector<Located> q_maxloc(COUNT), q_minloc(COUNT);
    MPI_Reduce(local.data(), q_sum.data(),  COUNT, OT, SUM,  0, MPI_COMM_WORLD);
    MPI_Reduce(local.data(), q_prod.data(), COUNT, OT, PROD, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_loc.data(), q_maxloc.data(), COUNT, OT_LOC, MAXLOC, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(local_loc.data(), q_minloc.data(), COUNT, OT_LOC, MINLOC, 0,
               MPI_COMM_WORLD);

    bool ok_reduce = true;
    if (rank == 0) {
        for (int j = 0; j < COUNT; ++j) {
            ok_reduce = ok_reduce
                && close_jet(q_sum[j], r_sum[j])
                && close_jet(q_prod[j], r_prod[j])
                && close_loc(q_maxloc[j], r_maxloc[j])
                && close_loc(q_minloc[j], r_minloc[j]);
        }
    }

    int failures = 0;
    if (rank == 0) {
        bool ok_sum = true, ok_prod = true, ok_maxloc = true, ok_minloc = true;
        for (int j = 0; j < COUNT; ++j) {
            Jet s(0.0), p(1.0);
            Located mx{}, mn{};
            bool first = true;
            for (int r = 0; r < size; ++r) {
                Jet g(v(r, j));
                g[1] = static_cast<double>(r + 1);
                if (j == COUNT - 1) g[0] = 1.0;
                const Located located{g, r};
                s = s + g;
                p = p * g;                       // jet product (convolution)
                if (first || g[0] > mx.value[0] ||
                    (g[0] == mx.value[0] && r < mx.location))
                    mx = located;
                if (first || g[0] < mn.value[0] ||
                    (g[0] == mn.value[0] && r < mn.location))
                    mn = located;
                first = false;
            }
            ok_sum  = ok_sum  && close_jet(r_sum[j], s);
            ok_prod = ok_prod && close_jet(r_prod[j], p);
            ok_maxloc = ok_maxloc && close_loc(r_maxloc[j], mx);
            ok_minloc = ok_minloc && close_loc(r_minloc[j], mn);
        }
        std::printf("oti::mpi reduction-op test (%d ranks, %d elems/reduce)\n",
                    size, COUNT);
        std::printf("  make_sum_op  : %s\n", ok_sum  ? "PASS" : "FAIL");
        std::printf("  make_prod_op : %s\n", ok_prod ? "PASS" : "FAIL");
        std::printf("  make_maxloc_op: %s\n", ok_maxloc ? "PASS" : "FAIL");
        std::printf("  make_minloc_op: %s\n", ok_minloc ? "PASS" : "FAIL");
        std::printf("  value_loc datatype: %s\n", ok_loc_type ? "PASS" : "FAIL");
        std::printf("  Reduce vs Allreduce on root (tolerance): %s\n",
                    ok_reduce ? "PASS" : "FAIL");
        std::printf("  sample[0]: sum=%.4f (d=%.1f)  prod=%.6f (d=%.4f)  "
                    "max=%.4f at rank %d (d=%.1f)\n",
                    r_sum[0][0], r_sum[0][1], r_prod[0][0], r_prod[0][1],
                    r_maxloc[0].value[0], r_maxloc[0].location,
                    r_maxloc[0].value[1]);
        std::printf("  tie[%d]: maxloc rank=%d, minloc rank=%d (expected 0)\n",
                    COUNT - 1, r_maxloc[COUNT - 1].location,
                    r_minloc[COUNT - 1].location);
        failures =
            (ok_sum && ok_prod && ok_maxloc && ok_minloc && ok_loc_type &&
             ok_reduce)
                ? 0
                : 1;
    }

    oti::mpi::free_op(SUM);
    oti::mpi::free_op(PROD);
    oti::mpi::free_op(MAXLOC);
    oti::mpi::free_op(MINLOC);
    oti::mpi::free_datatype(OT_LOC);
    oti::mpi::free_datatype(OT);
    MPI_Bcast(&failures, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return failures == 0 ? 0 : 1;
}
