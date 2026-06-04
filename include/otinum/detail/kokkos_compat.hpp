#pragma once

// Compatibility layer for standard C++ and Kokkos/CUDA builds.
//
// This header centralizes the small differences between normal host-only builds
// and OTI_ENABLE_KOKKOS builds. It selects std::array or Kokkos::Array for fixed
// storage, routes scalar math calls through std::<cmath> or Kokkos math, and
// defines the OTI_* annotation/assertion macros used by the rest of the library.

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>

#ifdef OTI_ENABLE_KOKKOS
#include <Kokkos_Core.hpp>
#endif

namespace oti::detail {

#ifdef OTI_ENABLE_KOKKOS
template <class T, std::size_t N>
using array = Kokkos::Array<T, N>;

KOKKOS_FORCEINLINE_FUNCTION double oti_exp(double x) noexcept
{
    return Kokkos::exp(x);
}

KOKKOS_FORCEINLINE_FUNCTION double oti_log(double x) noexcept
{
    return Kokkos::log(x);
}

KOKKOS_FORCEINLINE_FUNCTION double oti_pow(double x, double p) noexcept
{
    return Kokkos::pow(x, p);
}

KOKKOS_FORCEINLINE_FUNCTION double oti_sin(double x) noexcept
{
    return Kokkos::sin(x);
}

KOKKOS_FORCEINLINE_FUNCTION double oti_cos(double x) noexcept
{
    return Kokkos::cos(x);
}
#else
template <class T, std::size_t N>
using array = std::array<T, N>;

inline double oti_exp(double x) noexcept
{
    return std::exp(x);
}

inline double oti_log(double x) noexcept
{
    return std::log(x);
}

inline double oti_pow(double x, double p) noexcept
{
    return std::pow(x, p);
}

inline double oti_sin(double x) noexcept
{
    return std::sin(x);
}

inline double oti_cos(double x) noexcept
{
    return std::cos(x);
}
#endif

} // namespace oti::detail

#ifdef OTI_ENABLE_KOKKOS
#define OTI_FUNCTION KOKKOS_FORCEINLINE_FUNCTION
#define OTI_CONSTEXPR_FUNCTION KOKKOS_FORCEINLINE_FUNCTION constexpr
#define OTI_ASSERT(condition) KOKKOS_ASSERT(condition)
#else
#define OTI_FUNCTION
#define OTI_CONSTEXPR_FUNCTION constexpr
#define OTI_ASSERT(condition) assert(condition)
#endif
