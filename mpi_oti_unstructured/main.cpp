// MPI + OTI unstructured ghost exchange: the fourth rung of the ladder.
//
// The halo rung (../mpi_oti_halo) communicated over a *structured* grid, where a
// neighbour's ghost data is either contiguous (a row) or strided (a column) -- so
// MPI_Type_vector describes it. Real unstructured meshes have no such regularity:
// the nodes another rank needs from me are an ARBITRARY subset of my owned nodes,
// scattered through my local array. That irregular index list is exactly what
// MPI_Type_indexed is for.
//
// Problem: steady-state diffusion on an unstructured graph (a ring plus a fixed
// set of deterministic "chords", a small-world graph). Each non-source node
// relaxes to the degree-weighted average of its graph neighbours -- the graph
// analogue of the 5-point Jacobi heat solve in the halo rung. Two fixed source
// nodes hold the boundary values.
//
// The OTI twist (identical in spirit to the halo rung): the two sources carry
// their value as *seeded variables* (A = e_0, B = e_1), so every converged node
// drags along d(value)/dA and d(value)/dB -- the source sensitivity of the whole
// field, from one solve. Because the map is linear and both sources are 1.0, every
// node satisfies  value == d/dA + d/dB  (a free self-consistency check).
//
// The MPI-specific surface is one derived datatype per neighbour, built on the
// committed jet element:
//   * RECV: ghosts are laid out grouped by owner -> a contiguous run of MPI_OTINUM
//   * SEND: the owned nodes a neighbour needs are scattered -> MPI_Type_indexed
//           (block lengths all 1) gathers them in place, with no manual packing.
//
// Build (standalone, no project CMake needed):
//   mpicxx -std=c++17 -O2 -I ../include main.cpp -o mpi_oti_unstructured
// Run:
//   mpirun -np 4 ./mpi_oti_unstructured

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <vector>

#include "otinum/otinum.hpp"
#include "otinum/mpi.hpp"   // optional MPI interop: oti::mpi::make_datatype

// ---- problem definition ----------------------------------------------------

using Jet = oti::otinum<2, 1, double>;   // value + d/dA + d/dB
static constexpr int V      = 240;       // global node count
static constexpr int NCHORD = 240;       // deterministic long-range edges
static constexpr int ITERS  = 2000;      // fixed count -> identical serial work
static constexpr int SRC_A  = 0;         // source node carrying A = e_0
static constexpr int SRC_B  = V - 1;     // source node carrying B = e_1
static constexpr std::uint64_t SEED = 0x9E3779B97F4A7C15ull;
static constexpr double FD_H   = 1.0e-6;
static constexpr double FD_TOL = 1.0e-8;

static const Jet T_A = Jet::variable(0, 1.0);   // 1.0 + e_0
static const Jet T_B = Jet::variable(1, 1.0);   // 1.0 + e_1

static inline bool is_source(int g) { return g == SRC_A || g == SRC_B; }

// ---- the (deterministic) graph ---------------------------------------------
// Every rank builds the SAME global adjacency from the same seed -- no RNG
// divergence across ranks. Neighbour lists are sorted by global id so the Jacobi
// sum is accumulated in an identical order everywhere (serial and distributed),
// which is what makes the bit-exact check below meaningful.
static std::vector<std::vector<int>> build_adjacency()
{
    std::set<std::pair<int, int>> edges;
    auto add = [&](int a, int b) {
        if (a == b) return;
        if (a > b) std::swap(a, b);
        edges.insert({a, b});
    };
    for (int i = 0; i < V; ++i) add(i, (i + 1) % V);   // the ring
    std::uint64_t s = SEED;                            // chords (small-world)
    auto rnd = [&]() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<std::uint32_t>(s >> 33);
    };
    for (int k = 0; k < NCHORD; ++k)
        add(static_cast<int>(rnd() % V), static_cast<int>(rnd() % V));

    std::vector<std::vector<int>> adj(V);
    for (const auto& e : edges) {
        adj[e.first].push_back(e.second);
        adj[e.second].push_back(e.first);
    }
    for (auto& nbrs : adj) std::sort(nbrs.begin(), nbrs.end());
    return adj;
}

// One Jacobi sweep over a set of nodes: each non-source node becomes the
// degree-weighted average of its neighbours, read through `at(g)`. Templated on
// the lookup so the serial reference (global array) and the distributed solver
// (owned + ghost slots) share the exact arithmetic, hence are bit-identical.
template <class Jet_, class Lookup>
static inline Jet_ relax_node(const std::vector<std::vector<int>>& adj, int g,
                              const Jet_& self, Lookup at)
{
    if (is_source(g)) return self;
    Jet_ acc(0.0);
    for (int nbr : adj[g]) acc += at(nbr);
    return acc * (1.0 / static_cast<double>(adj[g].size()));
}

// ---- serial reference: full V-node solve, same sources, same iteration count -
// Every rank runs this redundantly and compares its owned nodes against the
// matching global entries -- so verification needs no gather, only the ghost
// exchange is real communication. Templated on the scalar so the OTI reference
// and the plain-double finite-difference references share one body.
template <class S>
static std::vector<S> solve_serial(const std::vector<std::vector<int>>& adj,
                                   const S& a_src, const S& b_src)
{
    std::vector<S> cur(V, S(0.0)), next(V, S(0.0));
    cur[SRC_A] = next[SRC_A] = a_src;
    cur[SRC_B] = next[SRC_B] = b_src;
    for (int it = 0; it < ITERS; ++it) {
        for (int g = 0; g < V; ++g)
            next[g] = relax_node(adj, g, cur[g],
                                 [&](int nbr) -> const S& { return cur[nbr]; });
        std::swap(cur, next);
    }
    return cur;
}

// 1D block partition of `n` items across `parts` ranks; first `rem` get one extra.
static void block_1d(int coord, int parts, int n, int& start, int& count)
{
    const int base = n / parts;
    const int rem  = n % parts;
    count = base + (coord < rem ? 1 : 0);
    start = coord * base + (coord < rem ? coord : rem);
}

// One neighbour of this rank: where its ghosts land on receive, and the indexed
// datatype that gathers the owned nodes it needs on send.
struct Neighbour {
    int rank = MPI_PROC_NULL;
    int recv_start = 0;        // first local slot of this neighbour's ghost block
    int recv_count = 0;        // contiguous ghost count (jets)
    MPI_Datatype send_type = MPI_DATATYPE_NULL;  // MPI_Type_indexed over owned
};

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0, nproc = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    const std::vector<std::vector<int>> adj = build_adjacency();

    // ---- 1. ownership: contiguous block partition of the global nodes -------
    std::vector<int> own_start(nproc), own_count(nproc);
    for (int r = 0; r < nproc; ++r) block_1d(r, nproc, V, own_start[r], own_count[r]);
    std::vector<int> owner(V);
    for (int r = 0; r < nproc; ++r)
        for (int k = 0; k < own_count[r]; ++k) owner[own_start[r] + k] = r;

    const int g0 = own_start[rank];
    const int my_count = own_count[rank];

    // ---- 2. discover ghosts: remote endpoints of edges touching my nodes ----
    // Group ghosts by owning rank and sort by global id, so each neighbour's
    // ghosts occupy ONE contiguous slot range (a plain count on receive).
    std::vector<std::set<int>> ghosts_from(nproc);  // remote rank -> {global ids}
    for (int u = g0; u < g0 + my_count; ++u)
        for (int v : adj[u])
            if (owner[v] != rank) ghosts_from[owner[v]].insert(v);

    std::vector<int> g2l(V, -1);                     // global id -> local slot
    for (int u = 0; u < my_count; ++u) g2l[g0 + u] = u;   // owned slots [0, my_count)

    std::vector<Neighbour> neighbours;
    int slot = my_count;                             // ghost slots follow owned
    for (int r = 0; r < nproc; ++r) {
        if (r == rank || ghosts_from[r].empty()) continue;
        Neighbour nb;
        nb.rank = r;
        nb.recv_start = slot;
        nb.recv_count = static_cast<int>(ghosts_from[r].size());
        for (int gid : ghosts_from[r]) g2l[gid] = slot++;   // sorted by gid (std::set)
        neighbours.push_back(nb);
    }
    const int local_n = slot;                        // owned + all ghosts

    // ---- 3. build the SEND datatype per neighbour --------------------------
    // The owned nodes neighbour r needs are exactly my nodes with an edge into r.
    // They are an arbitrary, scattered subset of my owned slots -> MPI_Type_indexed
    // (block lengths all 1) describes them in place. Sorted by global id so they
    // arrive in the order r laid out its matching ghost block.
    MPI_Datatype MPI_OTINUM = oti::mpi::make_datatype<Jet>();
    for (Neighbour& nb : neighbours) {
        std::set<int> needed;                        // my owned global ids r needs
        for (int u = g0; u < g0 + my_count; ++u)
            for (int v : adj[u])
                if (owner[v] == nb.rank) { needed.insert(u); break; }
        std::vector<int> displ;                      // local slots, sorted by gid
        displ.reserve(needed.size());
        for (int gid : needed) displ.push_back(g2l[gid]);
        std::vector<int> blocklen(displ.size(), 1);
        MPI_Type_indexed(static_cast<int>(displ.size()), blocklen.data(),
                         displ.data(), MPI_OTINUM, &nb.send_type);
        MPI_Type_commit(&nb.send_type);
    }

    // ---- 4. local field: owned slots [0, my_count) + ghost slots after ------
    std::vector<Jet> a(local_n, Jet(0.0)), b(local_n, Jet(0.0));
    auto seed_sources = [&](std::vector<Jet>& f) {
        if (owner[SRC_A] == rank) f[g2l[SRC_A]] = T_A;
        if (owner[SRC_B] == rank) f[g2l[SRC_B]] = T_B;
    };
    seed_sources(a);
    seed_sources(b);

    // ---- 5. Jacobi iteration with unstructured ghost exchange --------------
    Jet* cur = a.data();
    Jet* next = b.data();
    for (int it = 0; it < ITERS; ++it) {
        // Refresh ghosts: send my scattered owned nodes (indexed), receive the
        // neighbour's into my contiguous ghost block. One Sendrecv per neighbour.
        for (const Neighbour& nb : neighbours)
            MPI_Sendrecv(cur, 1, nb.send_type, nb.rank, 7,
                         &cur[nb.recv_start], nb.recv_count, MPI_OTINUM, nb.rank, 7,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int u = 0; u < my_count; ++u)
            next[u] = relax_node(adj, g0 + u, cur[u],
                                 [&](int nbr) -> const Jet& { return cur[g2l[nbr]]; });
        std::swap(cur, next);
    }

    // ---- 6. verify owned nodes against the serial reference (bit-exact) -----
    const std::vector<Jet> ref = solve_serial<Jet>(adj, T_A, T_B);
    long local_mismatch = 0;
    for (int u = 0; u < my_count; ++u)
        if (std::memcmp(&cur[u], &ref[g0 + u], sizeof(Jet)) != 0) ++local_mismatch;
    long total_mismatch = 0;
    MPI_Reduce(&local_mismatch, &total_mismatch, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // A non-source interior node prints a sample jet.
    const int sample = V / 2;
    if (g0 <= sample && sample < g0 + my_count) {
        const Jet& s = cur[g2l[sample]];
        std::printf("sample @ node %d (deg %zu), owned by rank %d:\n",
                    sample, adj[sample].size(), rank);
        std::printf("  value              = % .8f\n", s.coeff(oti::sparse({})));
        std::printf("  d/dA               = % .8f\n", s.coeff(oti::sparse({{0, 1}})));
        std::printf("  d/dB               = % .8f\n", s.coeff(oti::sparse({{1, 1}})));
        std::fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // ---- 7. finite-difference check of the source sensitivities ------------
    int fd_pass = 1;
    if (rank == 0) {
        const Jet& s = ref[sample];
        const double oti_a = s.coeff(oti::sparse({{0, 1}}));
        const double oti_b = s.coeff(oti::sparse({{1, 1}}));
        const auto ap = solve_serial<double>(adj, 1.0 + FD_H, 1.0);
        const auto am = solve_serial<double>(adj, 1.0 - FD_H, 1.0);
        const auto bp = solve_serial<double>(adj, 1.0, 1.0 + FD_H);
        const auto bm = solve_serial<double>(adj, 1.0, 1.0 - FD_H);
        const double fd_a = (ap[sample] - am[sample]) / (2.0 * FD_H);
        const double fd_b = (bp[sample] - bm[sample]) / (2.0 * FD_H);
        double max_err = 0.0;
        for (int g = 0; g < V; ++g) {
            max_err = std::max(max_err, std::abs(
                ref[g].coeff(oti::sparse({{0, 1}})) - (ap[g] - am[g]) / (2.0 * FD_H)));
            max_err = std::max(max_err, std::abs(
                ref[g].coeff(oti::sparse({{1, 1}})) - (bp[g] - bm[g]) / (2.0 * FD_H)));
        }
        fd_pass = max_err <= FD_TOL;

        std::printf("---\n");
        std::printf("graph              : %d nodes, ring + %d chords (small-world)\n",
                    V, NCHORD);
        std::printf("partition          : %d ranks, contiguous blocks\n", nproc);
        std::printf("iterations         : %d Jacobi sweeps\n", ITERS);
        std::printf("verify vs serial   : %s (%ld mismatching jets)\n",
                    total_mismatch == 0 ? "PASS (bit-exact)" : "FAIL", total_mismatch);
        std::printf("finite difference  : centred, h = %.1e\n", FD_H);
        std::printf("  d/dA             : OTI = %.10f, FD = %.10f, |error| = %.3e\n",
                    oti_a, fd_a, std::abs(oti_a - fd_a));
        std::printf("  d/dB             : OTI = %.10f, FD = %.10f, |error| = %.3e\n",
                    oti_b, fd_b, std::abs(oti_b - fd_b));
        std::printf("  max grid error   : %.3e\n", max_err);
        std::printf("verify sensitivities: %s (tolerance %.1e)\n",
                    fd_pass ? "PASS" : "FAIL", FD_TOL);
    }
    MPI_Bcast(&fd_pass, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // ---- 8. teardown -------------------------------------------------------
    for (Neighbour& nb : neighbours) MPI_Type_free(&nb.send_type);
    oti::mpi::free_datatype(MPI_OTINUM);
    MPI_Finalize();
    return total_mismatch == 0 && fd_pass ? 0 : 1;
}
