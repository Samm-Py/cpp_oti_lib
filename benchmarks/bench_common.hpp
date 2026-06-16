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

namespace bench {

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
