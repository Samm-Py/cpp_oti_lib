#pragma once

// Optional host-side operation profiling.
//
// When OTI_ENABLE_PROFILE is defined without OTI_ENABLE_KOKKOS, arithmetic and
// public math functions increment counters in a process-local global struct.
// Profiling is disabled in Kokkos mode so device-callable functions can remain
// constexpr/inline and avoid host-only global state. This header also provides
// small CSV helpers for dumping counter snapshots.

#include <cstdint>
#include <ostream>

#include "otinum/detail/kokkos_compat.hpp"

#if defined(OTI_ENABLE_PROFILE) && !defined(OTI_ENABLE_KOKKOS)
#define OTI_CONSTEXPR
#define OTI_PROFILE_COUNT(name) ::oti::profile::global_counters().name += 1
#else
#define OTI_CONSTEXPR OTI_CONSTEXPR_FUNCTION
#define OTI_PROFILE_COUNT(name) ((void)0)
#endif

namespace oti::profile {

struct counters {
    std::uint64_t add = 0;
    std::uint64_t add_oti = 0;
    std::uint64_t add_scalar = 0;
    std::uint64_t sub = 0;
    std::uint64_t sub_oti = 0;
    std::uint64_t sub_scalar = 0;
    std::uint64_t neg = 0;
    std::uint64_t mul = 0;
    std::uint64_t mul_oti = 0;
    std::uint64_t mul_scalar = 0;
    std::uint64_t div = 0;
    std::uint64_t div_oti = 0;
    std::uint64_t div_scalar = 0;
    std::uint64_t inv = 0;
    std::uint64_t trunc_mul = 0;
    std::uint64_t trunc_add = 0;
    std::uint64_t gem = 0;
    std::uint64_t exp = 0;
    std::uint64_t log = 0;
    std::uint64_t pow = 0;
    std::uint64_t sin = 0;
    std::uint64_t cos = 0;
    std::uint64_t tan = 0;
    std::uint64_t sinh = 0;
    std::uint64_t cosh = 0;
    std::uint64_t tanh = 0;
    std::uint64_t abs = 0;
};

// Not thread-safe: the counters are incremented with a plain ++ on a single
// process-global instance. This is acceptable because profiling is compiled out
// in Kokkos builds (see the guard above), so it is only ever used in serial,
// host-only runs. Profiling a host build that itself uses threads (e.g. raw
// OpenMP without Kokkos) would race on these counters.
inline counters& global_counters() noexcept
{
    static counters value;
    return value;
}

inline void reset() noexcept
{
    global_counters() = counters{};
}

inline counters snapshot() noexcept
{
    return global_counters();
}

inline void write_csv_header(std::ostream& os)
{
    os << "run,add,add_oti,add_scalar,sub,sub_oti,sub_scalar,neg,"
          "mul,mul_oti,mul_scalar,div,div_oti,div_scalar,inv,"
          "trunc_mul,trunc_add,gem,exp,log,pow,sin,cos,tan,sinh,cosh,tanh,abs\n";
}

inline void write_csv_row(std::ostream& os, char const* run, counters const& c)
{
    os << run << ','
       << c.add << ','
       << c.add_oti << ','
       << c.add_scalar << ','
       << c.sub << ','
       << c.sub_oti << ','
       << c.sub_scalar << ','
       << c.neg << ','
       << c.mul << ','
       << c.mul_oti << ','
       << c.mul_scalar << ','
       << c.div << ','
       << c.div_oti << ','
       << c.div_scalar << ','
       << c.inv << ','
       << c.trunc_mul << ','
       << c.trunc_add << ','
       << c.gem << ','
       << c.exp << ','
       << c.log << ','
       << c.pow << ','
       << c.sin << ','
       << c.cos << ','
       << c.tan << ','
       << c.sinh << ','
       << c.cosh << ','
       << c.tanh << ','
       << c.abs << '\n';
}

} // namespace oti::profile
