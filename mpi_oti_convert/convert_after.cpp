// AFTER: the same Kokkos + MPI program with otinum, so the gathered field now
// carries derivatives. Compare against convert_before.cpp -- the differences are
// exactly what it takes to OTI-enable an existing Kokkos + MPI code:
//
//   1. include the optional headers (otinum core + the MPI datatype helper)
//   2. change the scalar type alias to an otinum
//   3. seed the inputs as infinitesimal variables
//   4. describe the element to MPI with make_datatype instead of MPI_DOUBLE
//   5. read derivatives out of the result
//
// The model() kernel is UNCHANGED -- the overloaded arithmetic and elementary
// functions carry the derivatives through. (OTI_ENABLE_KOKKOS is set by CMake.)
// Build/run: see CMakeLists.txt.

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "otinum/otinum.hpp"   // 1. otinum core (+ <cmath> overloads)
#include "otinum/mpi.hpp"      //    optional MPI datatype helper

using Scalar = oti::otinum<2, 2, double>;    // 2. was: using Scalar = double;

static constexpr int N = 1000;
static constexpr long TOTAL = static_cast<long>(N) * N;

// The model. UNCHANGED from convert_before.cpp.
KOKKOS_INLINE_FUNCTION Scalar model(Scalar x, Scalar y)
{
    return sin(x) * exp(y);
}

static void block_range(int rank, int size, long& start, long& count)
{
    const long base = TOTAL / size;
    const long rem = TOTAL % size;
    count = base + (rank < rem ? 1 : 0);
    start = rank * base + (rank < rem ? rank : rem);
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);
    {
        int rank = 0, size = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        MPI_Datatype field_type = oti::mpi::make_datatype<Scalar>();  // 4.

        long start = 0, count = 0;
        block_range(rank, size, start, count);

        Kokkos::View<Scalar*> d_field("d_field", count);
        Kokkos::parallel_for(
            "evaluate", count, KOKKOS_LAMBDA(int k) {
                const long g = start + k;
                const double h = 1.0 / (N - 1);
                Scalar x = Scalar::variable(0, (g / N) * h);   // 3. seed as e_0
                Scalar y = Scalar::variable(1, (g % N) * h);   //    seed as e_1
                d_field(k) = model(x, y);
            });
        auto h_field = Kokkos::create_mirror_view(d_field);
        Kokkos::deep_copy(h_field, d_field);

        std::vector<int> recvcounts, displs;
        std::vector<Scalar> global;
        if (rank == 0) {
            recvcounts.resize(static_cast<size_t>(size));
            displs.resize(static_cast<size_t>(size));
            for (int r = 0; r < size; ++r) {
                long s = 0, c = 0;
                block_range(r, size, s, c);
                recvcounts[static_cast<size_t>(r)] = static_cast<int>(c);
                displs[static_cast<size_t>(r)] = static_cast<int>(s);
            }
            global.resize(static_cast<size_t>(TOTAL));
        }
        MPI_Gatherv(h_field.data(), static_cast<int>(count), field_type,
                    rank == 0 ? global.data() : nullptr,
                    rank == 0 ? recvcounts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    field_type, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            const long mid = (TOTAL / 2) + (N / 2);
            const Scalar& s = global[static_cast<size_t>(mid)];
            // 5. read value + derivatives out of the jet
            std::printf("sample value = %.8f\n", s.coeff(oti::sparse({})));
            std::printf("        d/dx = %.8f\n", s.coeff(oti::sparse({{0, 1}})));
            std::printf("        d/dy = %.8f\n", s.coeff(oti::sparse({{1, 1}})));
        }

        oti::mpi::free_datatype(field_type);   // 4. (release the committed type)
    }
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
