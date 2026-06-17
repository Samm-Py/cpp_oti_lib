#pragma once

// Shared scaffolding for the otinum isolation benchmarks. Every benchmark emits
// the same tidy CSV schema to stdout so a single plotter can consume all of
// them and they compose into one optimization story:
//
//   backend,coeff_type,M,N,ncoeffs,nproducts,kernel,variant,repetition,metric,value,checksum
//
// Each benchmark isolates one optimization by flipping exactly one axis
// (`variant`) and holding everything else at the library default:
//   arithmetic : variant = naive | lookup | unrolled   metric = ns_per_op
//   alignment  : variant = natural | aligned           metric = useful_gbps
//   layout     : variant = aos | soa                    metric = useful_gbps
//   fused      : variant = chain | fused                metric = ns_per_op
//
// `value` is the measured metric (ns_per_op: lower is better; useful_gbps:
// counts only the useful coefficient bytes, so dead bytes show as a lower rate).
// `checksum` lets the runner confirm the two variants computed the same thing.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace bench {

// ---- command-line helpers shared by every benchmark --------------------
//
// Each benchmark keeps its existing positional arguments (problem size,
// repetitions, ...) and additionally understands two flags:
//   --nodes <int>        override the problem size (element/node count)
//   --shapes <M,N> ...    run only these algebra shapes (default: all compiled)
// Flags may appear after the positional arguments, e.g.
//   bench_fused 16384 11 --shapes 1,1 2,1 3,1
// Positional parsing stops at the first flag, so the two styles compose.

using shape_list = std::vector<std::pair<int, int>>;

// Number of leading positional arguments (those before the first --flag),
// counting argv[0]; mirrors the old `argc` for positional parsing.
inline int positional_argc(int argc, char** argv)
{
    int n = 1;
    for (; n < argc; ++n) {
        if (argv[n][0] == '-') break;
    }
    return n;
}

// Value of an integer flag like --nodes; returns def if the flag is absent.
inline long flag_long(int argc, char** argv, char const* name, long def)
{
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) return std::atol(argv[i + 1]);
    }
    return def;
}

// Collect the M,N pairs following --shapes. Empty result means "all shapes".
inline shape_list parse_shapes(int argc, char** argv)
{
    shape_list shapes;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--shapes") != 0) continue;
        for (int j = i + 1; j < argc; ++j) {
            if (argv[j][0] == '-') break;  // next flag
            int m = 0, n = 0;
            if (std::sscanf(argv[j], "%d,%d", &m, &n) == 2) {
                shapes.emplace_back(m, n);
            }
        }
    }
    return shapes;
}

inline bool wanted(shape_list const& shapes, int M, int N)
{
    if (shapes.empty()) return true;  // no --shapes => run everything
    for (auto const& s : shapes) {
        if (s.first == M && s.second == N) return true;
    }
    return false;
}

template <class Coeff>
inline char const* coeff_name();
template <>
inline char const* coeff_name<double>() { return "double"; }
template <>
inline char const* coeff_name<float>() { return "float"; }

inline void print_header()
{
    std::printf("backend,coeff_type,M,N,ncoeffs,nproducts,kernel,variant,"
                "repetition,metric,value,checksum\n");
    std::fflush(stdout);
}

inline void print_row(char const* backend, char const* coeff, int M, int N,
                      int ncoeffs, int nproducts, char const* kernel,
                      char const* variant, int repetition, char const* metric,
                      double value, double checksum)
{
    std::printf("%s,%s,%d,%d,%d,%d,%s,%s,%d,%s,%.6g,%.10e\n",
                backend, coeff, M, N, ncoeffs, nproducts, kernel, variant,
                repetition, metric, value, checksum);
    std::fflush(stdout);
}

} // namespace bench
