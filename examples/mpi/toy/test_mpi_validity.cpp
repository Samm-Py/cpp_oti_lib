// Confidence test: the oti::validity error-analysis methods run correctly under
// MPI and are decomposition-invariant.
//
// A G x G grid of OTI jets (the surrogate f = exp(x) + exp(y) + 0.5 x y expanded
// at each grid point) is block-distributed across ranks. Every rank runs the
// validity primitives on its own jets -- is_trusted, validity_radius (which here
// folds orders 2 AND 3 via the bracket-and-bisection, since the jet is <2,3>
// certifying the LINEAR model), and truncation_error -- then we reduce two global
// quantities that MUST be identical for any rank count:
//   * trusted-point COUNT          (integer sum -> exact)
//   * global MINIMUM validity reach (MPI_MIN     -> bit-exact, order-independent)
// Running at np = 1, 4, 7 must print byte-identical numbers.
//
//   cmake -S . -B build && cmake --build build --target test_mpi_validity
//   for n in 1 4 7; do mpirun -np $n ./build/test_mpi_validity; done

#include <mpi.h>

#include "otinum/otinum.hpp"
#include "otinum/validity.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>

using J = oti::otinum<2, 3, double>;  // order 3: certify the linear model (m=1), fold orders 2+3
namespace v = oti::validity;

// Local truncated-Taylor surrogate of f = exp(x) + exp(y) + 0.5 x y at (x0, y0).
static J make_jet(double x0, double y0)
{
    J x = J::variable(0, x0), y = J::variable(1, y0);
    return oti::exp(x) + oti::exp(y) + 0.5 * (x * y);
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int const G = 64;
    long const total = static_cast<long>(G) * G;
    double const tau = 0.05;
    int const model_order = 1;                 // certify the LINEAR surrogate
    std::array<double, 2> const step{0.26, 0.26}; // fixed candidate step for the trust check

    // contiguous block decomposition of the flat grid index
    long const lo = total * rank / nprocs;
    long const hi = total * (rank + 1) / nprocs;

    long local_trusted = 0;
    double local_min_reach = std::numeric_limits<double>::infinity();
    for (long idx = lo; idx < hi; ++idx) {
        int i = static_cast<int>(idx % G), j = static_cast<int>(idx / G);
        double x0 = -0.5 + 1.0 * i / (G - 1);
        double y0 = -0.5 + 1.0 * j / (G - 1);
        J jet = make_jet(x0, y0);
        if (v::is_trusted(jet, step, tau, 0.0, model_order)) ++local_trusted;
        auto r = v::validity_radius(jet, tau, 0.0, model_order);   // multi-band bracket-and-bisection
        local_min_reach = std::min(local_min_reach, std::min(r[0], r[1]));
    }

    long total_trusted = 0;
    double global_min_reach = 0.0;
    MPI_Allreduce(&local_trusted, &total_trusted, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_min_reach, &global_min_reach, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

    if (rank == 0) {
        std::printf("np=%d  grid=%dx%d (%ld jets)  trusted=%ld  global_min_reach=%.17g\n",
                    nprocs, G, G, total, total_trusted, global_min_reach);
    }
    MPI_Finalize();
    return 0;
}
