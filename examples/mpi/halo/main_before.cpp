// The plain-double starting point that halo/main.cpp OTI-enables.
//
// It is the same distributed Jacobi heat solve -- a 2D Cartesian rank grid,
// row/column halo exchange every iteration, verified bit-exact against a serial
// recompute -- but in ordinary `double`. It computes the temperature field and
// nothing else.
//
// Converting it to OTI (see main.cpp and converting/halo.rst) is a handful of
// changes: swap the scalar type for a jet, seed the two hot-wall temperatures as
// variables, build the halo datatypes on MPI_OTINUM instead of MPI_DOUBLE, and
// read the sensitivities out. The Jacobi stencil and the halo exchange do not
// change.
//
// Build: mpicxx -std=c++17 -O2 main_before.cpp -o mpi_halo_before
// Run:   mpirun -np 4 ./mpi_halo_before

#include <mpi.h>

#include <cstdio>
#include <cstring>
#include <vector>

// ---- problem definition ----------------------------------------------------

using Scalar = double;
static constexpr int N     = 128;        // interior is N x N (boundary excluded)
static constexpr int ITERS = 4000;       // fixed count -> identical serial work

static const Scalar T_WEST  = 1.0;       // hot West wall
static const Scalar T_SOUTH = 1.0;       // hot South wall
static const Scalar T_COLD  = 0.0;       // cold East/North walls + interior

// One Jacobi sweep over the interior [1..nx] x [1..ny] of a ghosted field whose
// row stride is `stride`. Shared verbatim by the serial reference and the
// distributed solver, so the arithmetic is bit-identical.
static inline void jacobi_sweep(const Scalar* cur, Scalar* next,
                                int nx, int ny, int stride)
{
    for (int i = 1; i <= nx; ++i) {
        for (int j = 1; j <= ny; ++j) {
            const int c = i * stride + j;
            next[c] = (cur[c - stride] + cur[c + stride] +
                       cur[c - 1]      + cur[c + 1]) * 0.25;
        }
    }
}

// Serial reference: full (N+2)x(N+2) grid, same BCs, same iteration count.
static std::vector<Scalar> solve_serial()
{
    const int s = N + 2;
    std::vector<Scalar> a(static_cast<size_t>(s) * s, T_COLD);
    std::vector<Scalar> b = a;
    for (int k = 0; k < s; ++k) {
        a[static_cast<size_t>(k) * s + 0] = T_SOUTH;
        a[static_cast<size_t>(0) * s + k] = T_WEST;
        b[static_cast<size_t>(k) * s + 0] = T_SOUTH;
        b[static_cast<size_t>(0) * s + k] = T_WEST;
    }
    Scalar* cur = a.data();
    Scalar* next = b.data();
    for (int it = 0; it < ITERS; ++it) {
        jacobi_sweep(cur, next, N, N, s);
        std::swap(cur, next);
    }
    return (cur == a.data()) ? a : b;
}

static void block_1d(int coord, int parts, int n, int& start, int& count)
{
    const int base = n / parts;
    const int rem  = n % parts;
    count = base + (coord < rem ? 1 : 0);
    start = 1 + coord * base + (coord < rem ? coord : rem);
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // ---- 1. build a 2D Cartesian communicator (non-periodic) ----------------
    int dims[2] = {0, 0};
    MPI_Dims_create(world_size, 2, dims);
    int periods[2] = {0, 0};
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, /*reorder=*/1, &cart);

    int rank = 0;
    MPI_Comm_rank(cart, &rank);
    int coords[2] = {0, 0};
    MPI_Cart_coords(cart, rank, 2, coords);

    int up = MPI_PROC_NULL, down = MPI_PROC_NULL;
    int left = MPI_PROC_NULL, right = MPI_PROC_NULL;
    MPI_Cart_shift(cart, 0, 1, &up, &down);
    MPI_Cart_shift(cart, 1, 1, &left, &right);

    // ---- 2. this rank's interior tile + ghosted local array -----------------
    int gi0 = 0, nx = 0, gj0 = 0, ny = 0;
    block_1d(coords[0], dims[0], N, gi0, nx);
    block_1d(coords[1], dims[1], N, gj0, ny);

    const int stride = ny + 2;
    const size_t local_n = static_cast<size_t>(nx + 2) * stride;
    std::vector<Scalar> a(local_n, T_COLD);
    std::vector<Scalar> b = a;

    auto set_boundary = [&](std::vector<Scalar>& f) {
        if (coords[0] == 0)
            for (int j = 0; j < stride; ++j) f[static_cast<size_t>(0) * stride + j] = T_WEST;
        if (coords[0] == dims[0] - 1)
            for (int j = 0; j < stride; ++j) f[static_cast<size_t>(nx + 1) * stride + j] = T_COLD;
        if (coords[1] == 0)
            for (int i = 0; i < nx + 2; ++i) f[static_cast<size_t>(i) * stride + 0] = T_SOUTH;
        if (coords[1] == dims[1] - 1)
            for (int i = 0; i < nx + 2; ++i) f[static_cast<size_t>(i) * stride + (ny + 1)] = T_COLD;
    };
    set_boundary(a);
    set_boundary(b);

    // ---- 3. halo datatypes: rows are contiguous, columns are strided --------
    MPI_Datatype row_type = MPI_DOUBLE;            // a row is ny contiguous Scalars
    MPI_Datatype MPI_COL;
    MPI_Type_vector(nx, 1, stride, MPI_DOUBLE, &MPI_COL);
    MPI_Type_commit(&MPI_COL);

    auto idx = [&](int i, int j) { return static_cast<size_t>(i) * stride + j; };

    // ---- 4. Jacobi iteration with halo exchange ----------------------------
    Scalar* cur = a.data();
    Scalar* next = b.data();
    for (int it = 0; it < ITERS; ++it) {
        MPI_Sendrecv(&cur[idx(nx, 1)], ny, row_type, down, 0,
                     &cur[idx(0, 1)],  ny, row_type, up,   0,
                     cart, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&cur[idx(1, 1)],      ny, row_type, up,   1,
                     &cur[idx(nx + 1, 1)], ny, row_type, down, 1,
                     cart, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&cur[idx(1, ny)],     1, MPI_COL, right, 2,
                     &cur[idx(1, 0)],      1, MPI_COL, left,  2,
                     cart, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&cur[idx(1, 1)],      1, MPI_COL, left,  3,
                     &cur[idx(1, ny + 1)], 1, MPI_COL, right, 3,
                     cart, MPI_STATUS_IGNORE);

        jacobi_sweep(cur, next, nx, ny, stride);
        std::swap(cur, next);
    }

    // ---- 5. verify this tile against the serial reference (bit-exact) -------
    const std::vector<Scalar> ref = solve_serial();
    const int sref = N + 2;
    long local_mismatch = 0;
    for (int i = 1; i <= nx; ++i) {
        for (int j = 1; j <= ny; ++j) {
            const Scalar got = cur[idx(i, j)];
            const Scalar exp = ref[static_cast<size_t>(gi0 + i - 1) * sref + (gj0 + j - 1)];
            if (std::memcmp(&got, &exp, sizeof(Scalar)) != 0) ++local_mismatch;
        }
    }
    long total_mismatch = 0;
    MPI_Reduce(&local_mismatch, &total_mismatch, 1, MPI_LONG, MPI_SUM, 0, cart);

    const int ci = N / 2, cj = N / 2;
    if (gi0 <= ci && ci < gi0 + nx && gj0 <= cj && cj < gj0 + ny) {
        const Scalar s = cur[idx(ci - gi0 + 1, cj - gj0 + 1)];
        std::printf("sample @ global centre (%d,%d):\n", ci, cj);
        std::printf("  temperature        = % .8f\n", s);
        std::fflush(stdout);
    }
    MPI_Barrier(cart);

    if (rank == 0) {
        std::printf("---\n");
        std::printf("process grid       : %d x %d  (%d ranks)\n", dims[0], dims[1], world_size);
        std::printf("interior grid      : %d x %d  (%d Jacobi iterations)\n", N, N, ITERS);
        std::printf("verify vs serial   : %s (%ld mismatching cells)\n",
                    total_mismatch == 0 ? "PASS (bit-exact)" : "FAIL", total_mismatch);
    }

    MPI_Type_free(&MPI_COL);
    MPI_Comm_free(&cart);
    MPI_Finalize();
    return total_mismatch == 0 ? 0 : 1;
}
