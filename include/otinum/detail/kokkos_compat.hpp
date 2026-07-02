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

template <class T>
KOKKOS_FORCEINLINE_FUNCTION T oti_floor(T x) noexcept
{
    return static_cast<T>(Kokkos::floor(x));
}

template <class T>
KOKKOS_FORCEINLINE_FUNCTION T oti_ceil(T x) noexcept
{
    return static_cast<T>(Kokkos::ceil(x));
}

template <class T>
KOKKOS_FORCEINLINE_FUNCTION T oti_trunc(T x) noexcept
{
    return static_cast<T>(Kokkos::trunc(x));
}

template <class T>
KOKKOS_FORCEINLINE_FUNCTION T oti_round(T x) noexcept
{
    return static_cast<T>(Kokkos::round(x));
}

template <class T>
KOKKOS_FORCEINLINE_FUNCTION T oti_nearbyint(T x) noexcept
{
    return static_cast<T>(Kokkos::nearbyint(x));
}

template <class T>
KOKKOS_FORCEINLINE_FUNCTION T oti_rint(T x) noexcept
{
    // Kokkos has no rint; nearbyint differs only in FP-exception behavior, which
    // is not observable on device, so it is the correct device stand-in.
    return static_cast<T>(Kokkos::nearbyint(x));
}

template <class T>
KOKKOS_FORCEINLINE_FUNCTION bool oti_signbit(T x) noexcept
{
    return Kokkos::signbit(x);
}

template <class T>
KOKKOS_FORCEINLINE_FUNCTION T oti_expm1(T x) noexcept
{
    return static_cast<T>(Kokkos::expm1(x));
}

template <class T>
KOKKOS_FORCEINLINE_FUNCTION T oti_log1p(T x) noexcept
{
    return static_cast<T>(Kokkos::log1p(x));
}

template <class T>
KOKKOS_FORCEINLINE_FUNCTION T oti_atan(T x) noexcept
{
    return static_cast<T>(Kokkos::atan(x));
}

template <class T>
KOKKOS_FORCEINLINE_FUNCTION T oti_asin(T x) noexcept
{
    return static_cast<T>(Kokkos::asin(x));
}

template <class T>
KOKKOS_FORCEINLINE_FUNCTION T oti_acos(T x) noexcept
{
    return static_cast<T>(Kokkos::acos(x));
}

template <class T>
KOKKOS_FORCEINLINE_FUNCTION T oti_atan2(T y, T x) noexcept
{
    return static_cast<T>(Kokkos::atan2(y, x));
}

template <class T>
KOKKOS_FORCEINLINE_FUNCTION T oti_asinh(T x) noexcept
{
    return static_cast<T>(Kokkos::asinh(x));
}

template <class T>
KOKKOS_FORCEINLINE_FUNCTION T oti_acosh(T x) noexcept
{
    return static_cast<T>(Kokkos::acosh(x));
}

template <class T>
KOKKOS_FORCEINLINE_FUNCTION T oti_atanh(T x) noexcept
{
    return static_cast<T>(Kokkos::atanh(x));
}

template <class T>
KOKKOS_FORCEINLINE_FUNCTION T oti_erf(T x) noexcept
{
    return static_cast<T>(Kokkos::erf(x));
}

template <class T>
KOKKOS_FORCEINLINE_FUNCTION T oti_erfc(T x) noexcept
{
    return static_cast<T>(Kokkos::erfc(x));
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

template <class T>
inline T oti_floor(T x) noexcept
{
    return static_cast<T>(std::floor(x));
}

template <class T>
inline T oti_ceil(T x) noexcept
{
    return static_cast<T>(std::ceil(x));
}

template <class T>
inline T oti_trunc(T x) noexcept
{
    return static_cast<T>(std::trunc(x));
}

template <class T>
inline T oti_round(T x) noexcept
{
    return static_cast<T>(std::round(x));
}

template <class T>
inline T oti_nearbyint(T x) noexcept
{
    return static_cast<T>(std::nearbyint(x));
}

template <class T>
inline T oti_rint(T x) noexcept
{
    return static_cast<T>(std::rint(x));
}

template <class T>
inline bool oti_signbit(T x) noexcept
{
    return std::signbit(x);
}

template <class T>
inline T oti_expm1(T x) noexcept
{
    return static_cast<T>(std::expm1(x));
}

template <class T>
inline T oti_log1p(T x) noexcept
{
    return static_cast<T>(std::log1p(x));
}

template <class T>
inline T oti_atan(T x) noexcept
{
    return static_cast<T>(std::atan(x));
}

template <class T>
inline T oti_asin(T x) noexcept
{
    return static_cast<T>(std::asin(x));
}

template <class T>
inline T oti_acos(T x) noexcept
{
    return static_cast<T>(std::acos(x));
}

template <class T>
inline T oti_atan2(T y, T x) noexcept
{
    return static_cast<T>(std::atan2(y, x));
}

template <class T>
inline T oti_asinh(T x) noexcept
{
    return static_cast<T>(std::asinh(x));
}

template <class T>
inline T oti_acosh(T x) noexcept
{
    return static_cast<T>(std::acosh(x));
}

template <class T>
inline T oti_atanh(T x) noexcept
{
    return static_cast<T>(std::atanh(x));
}

template <class T>
inline T oti_erf(T x) noexcept
{
    return static_cast<T>(std::erf(x));
}

template <class T>
inline T oti_erfc(T x) noexcept
{
    return static_cast<T>(std::erfc(x));
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
