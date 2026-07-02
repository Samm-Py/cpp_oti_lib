// Plain-double starting point for the unstructured ghost-exchange rung.
//
// The same distributed solver as main.cpp -- steady-state diffusion on a
// ring+chords small-world graph, contiguous block partition, irregular ghost
// lists moved with MPI_Type_indexed -- but in ordinary `double`. It produces the
// node values and nothing else; OTI-enabling it (see main.cpp) adds the source
// sensitivities with no change to the decomposition or the exchange.
//
// Build:
//   mpicxx -std=c++17 -O2 main_before.cpp -o mpi_unstructured_before
// Run:
//   mpirun -np 4 ./mpi_unstructured_before

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>
#include <vector>

using Scalar = double;
static constexpr int V      = 240;
static constexpr int NCHORD = 240;
static constexpr int ITERS  = 2000;
static constexpr int SRC_A  = 0;
static constexpr int SRC_B  = V - 1;
static constexpr std::uint64_t SEED = 0x9E3779B97F4A7C15ull;

static const Scalar T_A = 1.0;
static const Scalar T_B = 1.0;

static inline bool is_source(int g) { return g == SRC_A || g == SRC_B; }

static std::vector<std::vector<int>> build_adjacency()
{
    std::set<std::pair<int, int>> edges;
    auto add = [&](int a, int b) {
        if (a == b) return;
        if (a > b) std::swap(a, b);
        edges.insert({a, b});
    };
    for (int i = 0; i < V; ++i) add(i, (i + 1) % V);
    std::uint64_t s = SEED;
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

template <class Lookup>
static inline Scalar relax_node(const std::vector<std::vector<int>>& adj, int g,
                                Scalar self, Lookup at)
{
    if (is_source(g)) return self;
    Scalar acc = 0.0;
    for (int nbr : adj[g]) acc += at(nbr);
    return acc * (1.0 / static_cast<double>(adj[g].size()));
}

static std::vector<Scalar> solve_serial(const std::vector<std::vector<int>>& adj)
{
    std::vector<Scalar> cur(V, 0.0), next(V, 0.0);
    cur[SRC_A] = next[SRC_A] = T_A;
    cur[SRC_B] = next[SRC_B] = T_B;
    for (int it = 0; it < ITERS; ++it) {
        for (int g = 0; g < V; ++g)
            next[g] = relax_node(adj, g, cur[g],
                                 [&](int nbr) -> Scalar { return cur[nbr]; });
        std::swap(cur, next);
    }
    return cur;
}

static void block_1d(int coord, int parts, int n, int& start, int& count)
{
    const int base = n / parts;
    const int rem  = n % parts;
    count = base + (coord < rem ? 1 : 0);
    start = coord * base + (coord < rem ? coord : rem);
}

struct Neighbour {
    int rank = MPI_PROC_NULL;
    int recv_start = 0;
    int recv_count = 0;
    MPI_Datatype send_type = MPI_DATATYPE_NULL;
};

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0, nproc = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    const std::vector<std::vector<int>> adj = build_adjacency();

    std::vector<int> own_start(nproc), own_count(nproc);
    for (int r = 0; r < nproc; ++r) block_1d(r, nproc, V, own_start[r], own_count[r]);
    std::vector<int> owner(V);
    for (int r = 0; r < nproc; ++r)
        for (int k = 0; k < own_count[r]; ++k) owner[own_start[r] + k] = r;

    const int g0 = own_start[rank];
    const int my_count = own_count[rank];

    std::vector<std::set<int>> ghosts_from(nproc);
    for (int u = g0; u < g0 + my_count; ++u)
        for (int v : adj[u])
            if (owner[v] != rank) ghosts_from[owner[v]].insert(v);

    std::vector<int> g2l(V, -1);
    for (int u = 0; u < my_count; ++u) g2l[g0 + u] = u;

    std::vector<Neighbour> neighbours;
    int slot = my_count;
    for (int r = 0; r < nproc; ++r) {
        if (r == rank || ghosts_from[r].empty()) continue;
        Neighbour nb;
        nb.rank = r;
        nb.recv_start = slot;
        nb.recv_count = static_cast<int>(ghosts_from[r].size());
        for (int gid : ghosts_from[r]) g2l[gid] = slot++;
        neighbours.push_back(nb);
    }
    const int local_n = slot;

    for (Neighbour& nb : neighbours) {
        std::set<int> needed;
        for (int u = g0; u < g0 + my_count; ++u)
            for (int v : adj[u])
                if (owner[v] == nb.rank) { needed.insert(u); break; }
        std::vector<int> displ;
        displ.reserve(needed.size());
        for (int gid : needed) displ.push_back(g2l[gid]);
        std::vector<int> blocklen(displ.size(), 1);
        MPI_Type_indexed(static_cast<int>(displ.size()), blocklen.data(),
                         displ.data(), MPI_DOUBLE, &nb.send_type);
        MPI_Type_commit(&nb.send_type);
    }

    std::vector<Scalar> a(local_n, 0.0), b(local_n, 0.0);
    auto seed_sources = [&](std::vector<Scalar>& f) {
        if (owner[SRC_A] == rank) f[g2l[SRC_A]] = T_A;
        if (owner[SRC_B] == rank) f[g2l[SRC_B]] = T_B;
    };
    seed_sources(a);
    seed_sources(b);

    Scalar* cur = a.data();
    Scalar* next = b.data();
    for (int it = 0; it < ITERS; ++it) {
        for (const Neighbour& nb : neighbours)
            MPI_Sendrecv(cur, 1, nb.send_type, nb.rank, 7,
                         &cur[nb.recv_start], nb.recv_count, MPI_DOUBLE, nb.rank, 7,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int u = 0; u < my_count; ++u)
            next[u] = relax_node(adj, g0 + u, cur[u],
                                 [&](int nbr) -> Scalar { return cur[g2l[nbr]]; });
        std::swap(cur, next);
    }

    const std::vector<Scalar> ref = solve_serial(adj);
    long local_mismatch = 0;
    for (int u = 0; u < my_count; ++u)
        if (std::memcmp(&cur[u], &ref[g0 + u], sizeof(Scalar)) != 0) ++local_mismatch;
    long total_mismatch = 0;
    MPI_Reduce(&local_mismatch, &total_mismatch, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    const int sample = V / 2;
    if (g0 <= sample && sample < g0 + my_count) {
        std::printf("sample @ node %d (deg %zu), owned by rank %d:\n",
                    sample, adj[sample].size(), rank);
        std::printf("  value              = % .8f\n", cur[g2l[sample]]);
        std::fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        std::printf("---\n");
        std::printf("graph              : %d nodes, ring + %d chords (small-world)\n",
                    V, NCHORD);
        std::printf("partition          : %d ranks, contiguous blocks\n", nproc);
        std::printf("iterations         : %d Jacobi sweeps\n", ITERS);
        std::printf("verify vs serial   : %s (%ld mismatching values)\n",
                    total_mismatch == 0 ? "PASS (bit-exact)" : "FAIL", total_mismatch);
    }

    for (Neighbour& nb : neighbours) MPI_Type_free(&nb.send_type);
    MPI_Finalize();
    return total_mismatch == 0 ? 0 : 1;
}
