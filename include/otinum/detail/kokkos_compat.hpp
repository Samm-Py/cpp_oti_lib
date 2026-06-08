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

template <class T>
KOKKOS_FORCEINLINE_FUNCTION T oti_exp(T x) noexcept
{
    return static_cast<T>(Kokkos::exp(x));
}

template <class T>
KOKKOS_FORCEINLINE_FUNCTION T oti_log(T x) noexcept
{
    return static_cast<T>(Kokkos::log(x));
}

template <class T>
KOKKOS_FORCEINLINE_FUNCTION T oti_pow(T x, T p) noexcept
{
    return static_cast<T>(Kokkos::pow(x, p));
}

template <class T>
KOKKOS_FORCEINLINE_FUNCTION T oti_cbrt(T x) noexcept
{
    return static_cast<T>(Kokkos::cbrt(x));
}

template <class T>
KOKKOS_FORCEINLINE_FUNCTION bool oti_isfinite(T x) noexcept
{
    return Kokkos::isfinite(x);
}

template <class T>
KOKKOS_FORCEINLINE_FUNCTION T oti_sin(T x) noexcept
{
    return static_cast<T>(Kokkos::sin(x));
}

template <class T>
KOKKOS_FORCEINLINE_FUNCTION T oti_cos(T x) noexcept
{
    return static_cast<T>(Kokkos::cos(x));
}
#else
template <class T, std::size_t N>
using array = std::array<T, N>;

template <class T>
inline T oti_exp(T x) noexcept
{
    return static_cast<T>(std::exp(x));
}

template <class T>
inline T oti_log(T x) noexcept
{
    return static_cast<T>(std::log(x));
}

template <class T>
inline T oti_pow(T x, T p) noexcept
{
    return static_cast<T>(std::pow(x, p));
}

template <class T>
inline T oti_cbrt(T x) noexcept
{
    return static_cast<T>(std::cbrt(x));
}

template <class T>
inline bool oti_isfinite(T x) noexcept
{
    return std::isfinite(x);
}

template <class T>
inline T oti_sin(T x) noexcept
{
    return static_cast<T>(std::sin(x));
}

template <class T>
inline T oti_cos(T x) noexcept
{
    return static_cast<T>(std::cos(x));
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
