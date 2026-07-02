// End-to-end MPI reduction compliance over OVER-ALIGNED (promoted) jet shapes.
//
// The sibling test_reduce_ops.cpp validates the four reduction ops with a small
// alignas(16)-or-less shape. This test specifically covers the shapes the
// 32-byte alignment rung (core.hpp) promotes to alignas(32) -- otinum<3,1,double>,
// otinum<3,3,double>, otinum<7,1,float> -- alongside aligned controls, to confirm
// the over-alignment does not break datatype construction (make_datatype /
// make_value_loc_datatype) or the reduce_fn user callback on real MPI buffers.
//
// Each rank contributes an array of n jets whose coefficients are integer-valued
// deterministic functions of (rank, element). Integer coefficients keep sum and
// (truncated-convolution) product EXACT and order-invariant, so every coefficient
// is checked bit-exact against a serial recompute rank 0 derives independently.
// maxloc/minloc carry the full winning jet plus its location with tie handling.
//
// Build: mpicxx -std=c++17 -O2 -I ../../../include test_reduce_promoted.cpp -o test_reduce_promoted
// Run:   mpirun -np 4 ./test_reduce_promoted

#include <mpi.h>

#include <cstdio>
#include <vector>

#include "otinum/otinum.hpp"
#include "otinum/mpi.hpp"

namespace om = oti::mpi;
template <class T> using VL = om::value_loc<T>;

// Integer-valued coefficients -> all arithmetic is exact and reorder-invariant.
template <class T> static T make_jet(int rank, int e, int kbias)
{
    T v{};
    for (int k = 0; k < T::ncoeffs; ++k) v[k] = double((rank + 1) * 7 + e * 3 + k + kbias);
    return v;
}
// Product factors: real part 1 + one small integer first-order term.
template <class T> static T make_unit_jet(int rank, int e)
{
    T v{};
    v[0] = 1.0;
    if (T::ncoeffs > 1) v[1] = double((rank % 3) + (e % 2));
    return v;
}

template <class T>
static int run_shape(const char* tag, int size, int rank, int n)
{
    MPI_Datatype JET = om::make_datatype<T>();
    MPI_Datatype VLT = om::make_value_loc_datatype<T>();
    MPI_Op SUM = om::make_sum_op<T>(), PROD = om::make_prod_op<T>();
    MPI_Op MAX = om::make_maxloc_op<T>(), MIN = om::make_minloc_op<T>();

    std::vector<T> ls(n), lp(n), gs(n), gp(n);
    std::vector<VL<T>> lmx(n), gmx(n), lmn(n), gmn(n);
    for (int e = 0; e < n; ++e) {
        ls[e] = make_jet<T>(rank, e, 0);
        lp[e] = make_unit_jet<T>(rank, e);
        T jv = make_jet<T>(rank, e, rank * 5);
        jv[0] = double(rank * 100 + e);  // distinct real parts; higher rank wins
        lmx[e] = VL<T>{jv, rank * 1000 + e};
        lmn[e] = lmx[e];
    }
    MPI_Allreduce(ls.data(), gs.data(), n, JET, SUM, MPI_COMM_WORLD);
    MPI_Allreduce(lp.data(), gp.data(), n, JET, PROD, MPI_COMM_WORLD);
    MPI_Allreduce(lmx.data(), gmx.data(), n, VLT, MAX, MPI_COMM_WORLD);
    MPI_Allreduce(lmn.data(), gmn.data(), n, VLT, MIN, MPI_COMM_WORLD);

    int fails = 0;
    if (rank == 0) {
        for (int e = 0; e < n; ++e) {
            T sref{};
            for (int r = 0; r < size; ++r) {
                T x = make_jet<T>(r, e, 0);
                for (int k = 0; k < T::ncoeffs; ++k) sref[k] += x[k];
            }
            for (int k = 0; k < T::ncoeffs; ++k) if (gs[e][k] != sref[k]) ++fails;

            T pref = make_unit_jet<T>(0, e);
            for (int r = 1; r < size; ++r) pref = pref * make_unit_jet<T>(r, e);
            for (int k = 0; k < T::ncoeffs; ++k) if (gp[e][k] != pref[k]) ++fails;

            const int wr = size - 1;
            T jmax = make_jet<T>(wr, e, wr * 5); jmax[0] = double(wr * 100 + e);
            if (gmx[e].location != wr * 1000 + e) ++fails;
            for (int k = 0; k < T::ncoeffs; ++k) if (gmx[e].value[k] != jmax[k]) ++fails;

            T jmin = make_jet<T>(0, e, 0); jmin[0] = double(0 * 100 + e);
            if (gmn[e].location != e) ++fails;
            for (int k = 0; k < T::ncoeffs; ++k) if (gmn[e].value[k] != jmin[k]) ++fails;
        }
        std::printf("  np=%d %-10s n=%d  %s\n", size, tag, n,
                    fails ? "FAIL" : "ok (sum/prod/maxloc/minloc bit-exact)");
    }
    om::free_datatype(JET); om::free_datatype(VLT);
    om::free_op(SUM); om::free_op(PROD); om::free_op(MAX); om::free_op(MIN);
    return fails;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const int n = 13;  // batched count > 1 exercises the *len loop + larger MPI temporaries

    int fails = 0;
    fails += run_shape<oti::otinum<3, 1, double>>("<3,1>d*", size, rank, n);  // promoted alignas(32)
    fails += run_shape<oti::otinum<3, 3, double>>("<3,3>d*", size, rank, n);  // promoted alignas(32)
    fails += run_shape<oti::otinum<7, 1, float>>("<7,1>f*", size, rank, n);   // promoted alignas(32)
    fails += run_shape<oti::otinum<2, 2, double>>("<2,2>d", size, rank, n);   // control alignas(16)
    fails += run_shape<oti::otinum<2, 1, double>>("<2,1>d", size, rank, n);   // control alignas(8)

    int total = 0;
    MPI_Reduce(&fails, &total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    int rc = 0;
    if (rank == 0) {
        std::printf("  (* = promoted to alignas(32) by the 32-byte rung)\n");
        rc = total ? 1 : 0;
    }
    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return rc;
}
