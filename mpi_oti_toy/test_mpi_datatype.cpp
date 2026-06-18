// Confidence test for oti::mpi::make_datatype.
//
// Proves the committed MPI datatype matches the C++ otinum layout and moves jets
// faithfully, across the cases the toy did NOT cover: float as well as double,
// and odd-ncoeffs shapes (only 4/8-aligned, the layouts where a padding surprise
// would bite). For each shape it checks:
//   (1) MPI_Type_size   == ncoeffs * sizeof(Coeff)   (payload bytes)
//   (2) MPI_Type extent == sizeof(T), lb == 0        (array stride; what makes
//                                                      count>1 / Gatherv correct)
//   (3) a ring Sendrecv of many jets round-trips bit-exact (real point-to-point,
//       exercising the extent with count>1)
//
// Build: mpicxx -std=c++17 -O2 -I ../include test_mpi_datatype.cpp -o test_mpi_datatype
// Run:   mpirun -np 2 ./test_mpi_datatype   (also valid at np=1 and np>2)

#include <mpi.h>

#include <cstdio>
#include <cstring>
#include <vector>

#include "otinum/otinum.hpp"
#include "otinum/mpi.hpp"

namespace {

constexpr int NJETS = 257;   // prime, >1, so count>1 exercises the extent/stride

// Deterministic, exactly-representable fill so sender and receiver compute the
// identical bit pattern (the round-trip is then a pure transport check).
template <class T>
void fill(T& j, int sender, int i)
{
    using Coeff = typename T::coeff_type;
    for (int k = 0; k < T::ncoeffs; ++k) {
        j[k] = static_cast<Coeff>(sender * 100003 + i * 101 + k);
    }
}

template <class T>
int check_shape(int rank, int size, char const* name)
{
    using Coeff = typename T::coeff_type;
    static_assert(sizeof(T) == T::ncoeffs * sizeof(Coeff),
                  "otinum is padded; a contiguous MPI datatype would desync");

    MPI_Datatype dt = oti::mpi::make_datatype<T>();

    // (1) payload size
    int type_size = 0;
    MPI_Type_size(dt, &type_size);
    int const want_size = T::ncoeffs * static_cast<int>(sizeof(Coeff));
    bool const size_ok = (type_size == want_size);

    // (2) extent == sizeof(T), lower bound 0 -> strides through an array of jets
    MPI_Aint lb = 0, extent = 0;
    MPI_Type_get_extent(dt, &lb, &extent);
    bool const extent_ok =
        (lb == 0 && extent == static_cast<MPI_Aint>(sizeof(T)));

    // (3) ring round-trip: send NJETS jets right, receive from left, verify the
    // received buffer equals what the left neighbor produced (bit-exact memcmp).
    std::vector<T> sendbuf(NJETS), recvbuf(NJETS), expect(NJETS);
    for (int i = 0; i < NJETS; ++i) {
        fill(sendbuf[i], rank, i);
    }
    int const right = (rank + 1) % size;
    int const left = (rank + size - 1) % size;
    MPI_Sendrecv(sendbuf.data(), NJETS, dt, right, 7,
                 recvbuf.data(), NJETS, dt, left, 7,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (int i = 0; i < NJETS; ++i) {
        fill(expect[i], left, i);
    }
    bool const roundtrip_ok =
        std::memcmp(recvbuf.data(), expect.data(),
                    static_cast<size_t>(NJETS) * sizeof(T)) == 0;

    oti::mpi::free_datatype(dt);

    if (rank == 0) {
        std::printf("  %-22s ncoeffs=%2d sizeof=%3zu | size %s extent %s "
                    "roundtrip %s\n",
                    name, T::ncoeffs, sizeof(T),
                    size_ok ? "OK" : "FAIL", extent_ok ? "OK" : "FAIL",
                    roundtrip_ok ? "OK" : "FAIL");
    }
    return (size_ok ? 0 : 1) + (extent_ok ? 0 : 1) + (roundtrip_ok ? 0 : 1);
}

} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        std::printf("oti::mpi datatype confidence test (%d rank%s, %d jets/msg)\n",
                    size, size == 1 ? "" : "s", NJETS);
    }

    int f = 0;
    f += check_shape<oti::otinum<3, 1, float>>(rank, size, "float  <3,1> al16");
    f += check_shape<oti::otinum<2, 2, float>>(rank, size, "float  <2,2> al8");
    f += check_shape<oti::otinum<4, 1, float>>(rank, size, "float  <4,1> al4 ODD");
    f += check_shape<oti::otinum<2, 2, double>>(rank, size, "double <2,2> al16");
    f += check_shape<oti::otinum<4, 1, double>>(rank, size, "double <4,1> al8 ODD");

    int total = 0;
    MPI_Reduce(&f, &total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    int ret = 0;
    if (rank == 0) {
        std::printf("%s (%d check failures across all ranks)\n",
                    total == 0 ? "ALL PASS" : "FAILURES", total);
        ret = (total == 0) ? 0 : 1;
    }
    MPI_Bcast(&ret, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return ret;
}
