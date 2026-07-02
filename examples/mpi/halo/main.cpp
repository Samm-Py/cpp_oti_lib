// MPI + OTI halo exchange: the first *communicating* solver in the ladder.
//
// We solve steady-state heat (Laplace's equation) on the unit square with a
// 5-point Jacobi stencil, distributed over a 2D Cartesian grid of MPI ranks.
// Each iteration the ranks exchange one ghost layer with their four neighbours,
// then sweep their interior. This is genuine per-iteration communication, unlike
// the embarrassingly-parallel gather toy (../toy).
//
// The OTI twist: the two hot walls carry their temperature as *seeded variables*
// (West = e_0, South = e_1), so every cell's converged value drags along
// d(temperature)/d(West) and d(temperature)/d(South) for free -- the parameter
// sensitivity of the whole field, from one solve.
//
// The MPI-specific surface is two derived datatypes built on the committed jet:
//   * row halos (N/S neighbours) are CONTIGUOUS  -> count of MPI_OTINUM
//   * column halos (E/W neighbours) are STRIDED   -> MPI_Type_vector(MPI_OTINUM)
// If the strided column type were wrong, the columns would corrupt and the
// result would not match the serial reference -- so the bit-exact check below is
// an end-to-end test of both halos.
//
// Build (standalone, no project CMake needed):
//   mpicxx -std=c++17 -O2 -I ../../../include main.cpp -o mpi_oti_halo
// Run:
//   mpirun -np 4 ./mpi_oti_halo

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "otinum/otinum.hpp"
#include "otinum/mpi.hpp"   // optional MPI interop: oti::mpi::make_datatype

// ---- problem definition ----------------------------------------------------

using Jet = oti::otinum<2, 1, double>;   // value + d/dT_west + d/dT_south
static constexpr int N     = 128;        // interior is N x N (boundary excluded)
static constexpr int ITERS = 4000;       // fixed count -> identical serial work
static constexpr double FD_H   = 1.0e-6;
static constexpr double FD_TOL = 1.0e-8;

// The two hot walls, seeded as infinitesimals so the solution carries its
// sensitivity to each wall temperature. The other two walls are cold.
static const Jet T_WEST  = Jet::variable(0, 1.0);   // 1.0 + e_0
static const Jet T_SOUTH = Jet::variable(1, 1.0);   // 1.0 + e_1
static const Jet T_COLD  = Jet(0.0);

// One Jacobi sweep over the interior [1..nx] x [1..ny] of a ghosted field whose
// row stride is `stride`. Written once and used by BOTH the serial reference and
// the distributed solver, so the floating-point arithmetic is bit-identical.
static inline void jacobi_sweep(const Jet* cur, Jet* next,
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

// ---- serial reference: full (N+2)x(N+2) grid, same BCs, same iteration count -
// Every rank runs this redundantly on the whole domain and compares its own tile
// against the matching subblock -- so verification needs no gather, only the
// halo exchange is real communication.
static std::vector<Jet> solve_serial()
{
    const int s = N + 2;                       // stride incl. both boundaries
    std::vector<Jet> a(static_cast<size_t>(s) * s, T_COLD);
    std::vector<Jet> b = a;

    // Dirichlet boundaries: West = column 0, South = row 0 (hot); East/North cold.
    for (int k = 0; k < s; ++k) {
        a[static_cast<size_t>(k) * s + 0] = T_SOUTH;   // row k, col 0  (South)
        a[static_cast<size_t>(0) * s + k] = T_WEST;    // row 0, col k  (West)
        b[static_cast<size_t>(k) * s + 0] = T_SOUTH;
        b[static_cast<size_t>(0) * s + k] = T_WEST;
    }

    Jet* cur = a.data();
    Jet* next = b.data();
    for (int it = 0; it < ITERS; ++it) {
        jacobi_sweep(cur, next, N, N, s);
        std::swap(cur, next);
    }
    // `cur` holds the latest field; copy it into a owned by value if needed.
    return (cur == a.data()) ? a : b;
}

// Independent finite-difference reference: repeat the same Jacobi solve using
// plain doubles and caller-supplied wall temperatures. Four calls at T +/- FD_H
// verify the two OTI boundary-condition derivatives over the full grid.
static std::vector<double> solve_serial_double(double t_west, double t_south)
{
    const int s = N + 2;
    std::vector<double> a(static_cast<size_t>(s) * s, 0.0);
    std::vector<double> b = a;

    for (int k = 0; k < s; ++k) {
        a[static_cast<size_t>(k) * s + 0] = t_south;
        a[static_cast<size_t>(0) * s + k] = t_west;
        b[static_cast<size_t>(k) * s + 0] = t_south;
        b[static_cast<size_t>(0) * s + k] = t_west;
    }

    double* cur = a.data();
    double* next = b.data();
    for (int it = 0; it < ITERS; ++it) {
        for (int i = 1; i <= N; ++i) {
            for (int j = 1; j <= N; ++j) {
                const int c = i * s + j;
                next[c] = (cur[c - s] + cur[c + s] +
                           cur[c - 1] + cur[c + 1]) * 0.25;
            }
        }
        std::swap(cur, next);
    }

    return (cur == a.data()) ? a : b;
}

// 1D block partition of `n` interior cells across `parts` ranks; the first
// `rem` ranks get one extra. Returns this part's [start, start+count) (1-based
// into the global interior 1..n).
static void block_1d(int coord, int parts, int n, int& start, int& count)
{
    const int base = n / parts;
    const int rem  = n % parts;
    count = base + (coord < rem ? 1 : 0);
    start = 1 + coord * base + (coord < rem ? coord : rem);
}

int main(int argc, char** argv)
{
    const char* fd_error_csv = nullptr;
    for (int arg = 1; arg < argc; ++arg) {
        if (std::strcmp(argv[arg], "--fd-error-csv") == 0 && arg + 1 < argc) {
            fd_error_csv = argv[++arg];
        } else if (std::strcmp(argv[arg], "--help") == 0) {
            std::printf("usage: %s [--fd-error-csv PATH]\n", argv[0]);
            return 0;
        }
    }

    MPI_Init(&argc, &argv);
    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // ---- 1. build a 2D Cartesian communicator (non-periodic) ----------------
    int dims[2] = {0, 0};
    MPI_Dims_create(world_size, 2, dims);            // factor ranks into Px x Py
    int periods[2] = {0, 0};
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, /*reorder=*/1, &cart);

    int rank = 0;
    MPI_Comm_rank(cart, &rank);
    int coords[2] = {0, 0};
    MPI_Cart_coords(cart, rank, 2, coords);

    // Neighbours: dim 0 indexes rows (i), dim 1 indexes columns (j).
    int up = MPI_PROC_NULL, down = MPI_PROC_NULL;     // i-1, i+1
    int left = MPI_PROC_NULL, right = MPI_PROC_NULL;   // j-1, j+1
    MPI_Cart_shift(cart, 0, 1, &up, &down);
    MPI_Cart_shift(cart, 1, 1, &left, &right);

    // ---- 2. this rank's interior tile + ghosted local array -----------------
    int gi0 = 0, nx = 0, gj0 = 0, ny = 0;             // global 1-based origins
    block_1d(coords[0], dims[0], N, gi0, nx);
    block_1d(coords[1], dims[1], N, gj0, ny);

    const int stride = ny + 2;                         // local row length w/ ghosts
    const size_t local_n = static_cast<size_t>(nx + 2) * stride;
    std::vector<Jet> a(local_n, T_COLD);
    std::vector<Jet> b = a;

    // Fill ghost layers that coincide with the global Dirichlet boundary. Interior
    // ghosts get overwritten by the halo exchange each iteration; these edge
    // ghosts are set once and never touched, so the BC is held automatically.
    auto set_boundary = [&](std::vector<Jet>& f) {
        if (coords[0] == 0)            // West wall -> top ghost row i=0
            for (int j = 0; j < stride; ++j) f[static_cast<size_t>(0) * stride + j] = T_WEST;
        if (coords[0] == dims[0] - 1)  // East wall -> bottom ghost row i=nx+1
            for (int j = 0; j < stride; ++j) f[static_cast<size_t>(nx + 1) * stride + j] = T_COLD;
        if (coords[1] == 0)            // South wall -> left ghost col j=0
            for (int i = 0; i < nx + 2; ++i) f[static_cast<size_t>(i) * stride + 0] = T_SOUTH;
        if (coords[1] == dims[1] - 1)  // North wall -> right ghost col j=ny+1
            for (int i = 0; i < nx + 2; ++i) f[static_cast<size_t>(i) * stride + (ny + 1)] = T_COLD;
    };
    set_boundary(a);
    set_boundary(b);

    // ---- 3. commit the datatypes -------------------------------------------
    // Base element: one jet == ncoeffs contiguous doubles (static_asserts packed).
    MPI_Datatype MPI_OTINUM = oti::mpi::make_datatype<Jet>();
    // Column halo (E/W): nx jets, one per row, strided by `stride` jets.
    MPI_Datatype MPI_COL;
    MPI_Type_vector(nx, 1, stride, MPI_OTINUM, &MPI_COL);
    MPI_Type_commit(&MPI_COL);
    // Row halo (N/S) is contiguous -> just `ny` of MPI_OTINUM, no derived type.

    auto idx = [&](int i, int j) { return static_cast<size_t>(i) * stride + j; };

    // ---- 4. Jacobi iteration with halo exchange ----------------------------
    Jet* cur = a.data();
    Jet* next = b.data();
    for (int it = 0; it < ITERS; ++it) {
        // Rows (contiguous, count = ny). Send my edge interior row, receive the
        // neighbour's into my ghost row. Sendrecv with MPI_PROC_NULL is a no-op,
        // so edge ranks keep their Dirichlet ghosts.
        MPI_Sendrecv(&cur[idx(nx, 1)], ny, MPI_OTINUM, down, 0,   // send last -> down
                     &cur[idx(0, 1)],  ny, MPI_OTINUM, up,   0,   // recv top ghost
                     cart, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&cur[idx(1, 1)],      ny, MPI_OTINUM, up,   1,   // send first -> up
                     &cur[idx(nx + 1, 1)], ny, MPI_OTINUM, down, 1,   // recv bottom ghost
                     cart, MPI_STATUS_IGNORE);

        // Columns (strided, one MPI_COL). Same pattern in the j direction.
        MPI_Sendrecv(&cur[idx(1, ny)],     1, MPI_COL, right, 2,  // send last col -> right
                     &cur[idx(1, 0)],      1, MPI_COL, left,  2,  // recv left ghost col
                     cart, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&cur[idx(1, 1)],      1, MPI_COL, left,  3,  // send first col -> left
                     &cur[idx(1, ny + 1)], 1, MPI_COL, right, 3,  // recv right ghost col
                     cart, MPI_STATUS_IGNORE);

        jacobi_sweep(cur, next, nx, ny, stride);
        std::swap(cur, next);
    }

    // ---- 5. verify this tile against the serial reference (bit-exact) -------
    const std::vector<Jet> ref = solve_serial();
    const int sref = N + 2;
    long local_mismatch = 0;
    for (int i = 1; i <= nx; ++i) {
        for (int j = 1; j <= ny; ++j) {
            const Jet& got = cur[idx(i, j)];
            const Jet& exp = ref[static_cast<size_t>(gi0 + i - 1) * sref + (gj0 + j - 1)];
            if (std::memcmp(&got, &exp, sizeof(Jet)) != 0) ++local_mismatch;
        }
    }
    long total_mismatch = 0;
    MPI_Reduce(&local_mismatch, &total_mismatch, 1, MPI_LONG, MPI_SUM, 0, cart);

    // The rank owning the global centre cell prints a sample jet.
    const int ci = N / 2, cj = N / 2;   // 1-based global interior centre
    if (gi0 <= ci && ci < gi0 + nx && gj0 <= cj && cj < gj0 + ny) {
        const Jet& s = cur[idx(ci - gi0 + 1, cj - gj0 + 1)];
        std::printf("sample @ global centre (%d,%d), owned by cart rank %d (%d,%d):\n",
                    ci, cj, rank, coords[0], coords[1]);
        std::printf("  temperature        = % .8f\n", s.coeff(oti::sparse({})));
        std::printf("  d/dT_west          = % .8f\n", s.coeff(oti::sparse({{0, 1}})));
        std::printf("  d/dT_south         = % .8f\n", s.coeff(oti::sparse({{1, 1}})));
        std::fflush(stdout);
    }
    MPI_Barrier(cart);

    int fd_pass = 1;
    if (rank == 0) {
        const Jet& centre = ref[static_cast<size_t>(ci) * sref + cj];
        const double oti_west = centre.coeff(oti::sparse({{0, 1}}));
        const double oti_south = centre.coeff(oti::sparse({{1, 1}}));
        const std::vector<double> west_plus =
            solve_serial_double(1.0 + FD_H, 1.0);
        const std::vector<double> west_minus =
            solve_serial_double(1.0 - FD_H, 1.0);
        const std::vector<double> south_plus =
            solve_serial_double(1.0, 1.0 + FD_H);
        const std::vector<double> south_minus =
            solve_serial_double(1.0, 1.0 - FD_H);
        auto fd_value = [&](const std::vector<double>& plus,
                            const std::vector<double>& minus,
                            int i, int j) {
            const size_t k = static_cast<size_t>(i) * sref + j;
            return (plus[k] - minus[k]) / (2.0 * FD_H);
        };
        const double fd_west = fd_value(west_plus, west_minus, ci, cj);
        const double fd_south = fd_value(south_plus, south_minus, ci, cj);
        const double err_west = std::abs(oti_west - fd_west);
        const double err_south = std::abs(oti_south - fd_south);
        double max_err_west = 0.0;
        double max_err_south = 0.0;

        FILE* csv = nullptr;
        if (fd_error_csv != nullptr) {
            csv = std::fopen(fd_error_csv, "w");
            if (csv == nullptr) {
                std::perror("failed to open finite-difference error CSV");
                fd_pass = 0;
            } else {
                std::fprintf(csv, "i,j,error_west,error_south\n");
            }
        }

        for (int i = 1; i <= N; ++i) {
            for (int j = 1; j <= N; ++j) {
                const Jet& cell = ref[static_cast<size_t>(i) * sref + j];
                const double ew = std::abs(
                    cell.coeff(oti::sparse({{0, 1}})) -
                    fd_value(west_plus, west_minus, i, j));
                const double es = std::abs(
                    cell.coeff(oti::sparse({{1, 1}})) -
                    fd_value(south_plus, south_minus, i, j));
                max_err_west = std::max(max_err_west, ew);
                max_err_south = std::max(max_err_south, es);
                if (csv != nullptr)
                    std::fprintf(csv, "%d,%d,%.17g,%.17g\n", i, j, ew, es);
            }
        }
        if (csv != nullptr) std::fclose(csv);

        fd_pass = fd_pass &&
                  max_err_west <= FD_TOL && max_err_south <= FD_TOL;

        std::printf("---\n");
        std::printf("process grid       : %d x %d  (%d ranks)\n", dims[0], dims[1], world_size);
        std::printf("interior grid      : %d x %d  (%d Jacobi iterations)\n", N, N, ITERS);
        std::printf("verify vs serial   : %s (%ld mismatching jets)\n",
                    total_mismatch == 0 ? "PASS (bit-exact)" : "FAIL", total_mismatch);
        std::printf("finite difference  : centred, h = %.1e\n", FD_H);
        std::printf("  d/dT_west        : OTI = %.10f, FD = %.10f, |error| = %.3e\n",
                    oti_west, fd_west, err_west);
        std::printf("  d/dT_south       : OTI = %.10f, FD = %.10f, |error| = %.3e\n",
                    oti_south, fd_south, err_south);
        std::printf("  max grid error   : West = %.3e, South = %.3e\n",
                    max_err_west, max_err_south);
        std::printf("verify sensitivities: %s (tolerance %.1e)\n",
                    fd_pass ? "PASS" : "FAIL", FD_TOL);
        if (fd_error_csv != nullptr && csv != nullptr)
            std::printf("wrote FD error grid: %s\n", fd_error_csv);
    }
    MPI_Bcast(&fd_pass, 1, MPI_INT, 0, cart);

    // ---- 6. teardown -------------------------------------------------------
    MPI_Type_free(&MPI_COL);
    oti::mpi::free_datatype(MPI_OTINUM);
    MPI_Comm_free(&cart);
    MPI_Finalize();
    return total_mismatch == 0 && fd_pass ? 0 : 1;
}
