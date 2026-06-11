#pragma once

// Core OTI value type and arithmetic.
//
// This header defines oti::otinum<M, N, Coeff>, a fixed-size truncated Taylor
// algebra with M infinitesimal directions, maximum total order N, and selectable
// floating-point coefficient storage. The class owns the coefficient storage,
// scalar/variable construction, coefficient accessors, and arithmetic operators.
// Coefficients are stored in the flat multi-index layout defined by
// detail/multi_index.hpp, and arithmetic uses those compile-time tables to
// implement truncated polynomial algebra.

#include <cstddef>
#include <initializer_list>
#include <type_traits>
#include <utility>
#include <vector>

#include "otinum/detail/multi_index.hpp"
#include "otinum/profile.hpp"

namespace oti {

// Sparse multi-index helpers: specify only the nonzero (variable, order) pairs,
// e.g. sparse({{0, 2}, {1, 1}}) for alpha = (2, 1, 0, ...). Backed by
// std::vector, so the coeff()/set_coeff()/partial()/set_partial() overloads that
// take a sparse_alpha are host-only -- unlike the alpha_type (array) overloads,
// which are device-callable.
struct sparse_index {
    int variable;
    int order;
};

struct sparse_alpha {
    std::vector<sparse_index> terms;
};

inline sparse_alpha sparse(std::initializer_list<sparse_index> terms)
{
    return sparse_alpha{terms};
}

// Static OTI number with M infinitesimal variables, truncated at total order N.
//
// Coefficients are normalized Taylor coefficients:
//   c[alpha] = (1 / alpha!) * partial^alpha f
// Use partial(alpha) to recover the ordinary derivative value.
template <class Coeff, class Scalar>
using scalar_enable_t =
    std::enable_if_t<std::is_arithmetic<Scalar>::value && std::is_convertible<Scalar, Coeff>::value,
                     int>;

namespace detail {

// Alignment for the otinum coefficient block. Shapes whose byte size is a
// multiple of 16 (or, failing that, 8) are aligned to that boundary so GPU
// and SIMD compilers can use 128-bit (or 64-bit) vector loads/stores on the
// coefficients; because the promotion is conditional on the size,
// sizeof(otinum) never changes (no padding).
template <class Coeff, int NC>
OTI_CONSTEXPR_FUNCTION std::size_t otinum_alignment() noexcept
{
    return (sizeof(Coeff) * NC) % 16 == 0 ? 16
         : (sizeof(Coeff) * NC) % 8 == 0  ? (alignof(Coeff) < 8 ? 8 : alignof(Coeff))
                                          : alignof(Coeff);
}

} // namespace detail

template <int M, int N, class Coeff = double>
class alignas(detail::otinum_alignment<Coeff, detail::tables<M, N>::ncoeffs>()) otinum {
    static_assert(std::is_floating_point<Coeff>::value,
                  "otinum coefficient type must be float, double, or another floating-point type");

public:
    using table_type = detail::tables<M, N>;
    using alpha_type = detail::alpha_t<M>;
    using coeff_type = Coeff;

    static constexpr int nvars = M;
    static constexpr int order = N;
    // Number of stored coefficients is the count of multi-indices alpha with
    // |alpha| <= N, i.e. binomial(M + N, N).
    static constexpr int ncoeffs = table_type::ncoeffs;

    // c_ has a brace initializer, so the empty constructor still creates the
    // zero element. Keeping the body explicit avoids NVCC warnings on annotated
    // defaulted constructors.
    OTI_CONSTEXPR_FUNCTION otinum() noexcept {}

    // Lift a scalar into the OTI algebra. All derivative coefficients are zero.
    // The c_ member initializer zeros every coefficient before c_[0] is assigned.
    OTI_CONSTEXPR_FUNCTION otinum(Coeff real)
    {
        c_[0] = real;
    }

    // Create value + e_i, where e_i is the first-order nilpotent for variable i.
    static OTI_CONSTEXPR_FUNCTION otinum variable(int i, Coeff value = Coeff(0))
    {
        otinum out(value);
        OTI_ASSERT(i >= 0 && i < M);
        if constexpr (N > 0) {
            // For N == 0 the algebra stores only the real coefficient. Otherwise
            // seed the first-order coefficient for epsilon_i with derivative 1.
            alpha_type alpha{};
            alpha[static_cast<std::size_t>(i)] = 1;
            out[detail::rank<M, N>(alpha)] = Coeff(1);
        }
        return out;
    }

    // Construct directly from the library's flat graded multi-index layout.
    // This is mainly for tests, serialization, and advanced callers that already
    // know the rank ordering used by detail::tables<M, N>.
    static OTI_CONSTEXPR_FUNCTION otinum from_coeffs(detail::array<Coeff, ncoeffs> const& coeffs)
    {
        otinum out;
        out.c_ = coeffs;
        return out;
    }

    OTI_CONSTEXPR_FUNCTION Coeff real() const noexcept
    {
        return c_[0];
    }

    // Explicit conversion to the underlying scalar (the real coefficient). This
    // is deliberately explicit: an implicit otinum -> Coeff conversion would
    // silently discard every derivative and create overload ambiguities, so a
    // drop-in replacement of double with otinum keeps static_cast<double>(x)
    // working while still flagging accidental narrowing at compile time.
    explicit OTI_CONSTEXPR_FUNCTION operator Coeff() const noexcept
    {
        return c_[0];
    }

    // Raw normalized coefficient access by flat multi-index rank. These accessors
    // do not bounds-check; callers that start from alpha should prefer coeff() or
    // partial(), which go through detail::rank().
    OTI_CONSTEXPR_FUNCTION Coeff operator[](int flat_index) const noexcept
    {
        return c_[static_cast<std::size_t>(flat_index)];
    }

    OTI_CONSTEXPR_FUNCTION Coeff& operator[](int flat_index) noexcept
    {
        return c_[static_cast<std::size_t>(flat_index)];
    }

    // Return the normalized Taylor coefficient for alpha, or zero if alpha is
    // outside this otinum's configured total order.
    OTI_CONSTEXPR_FUNCTION Coeff coeff(alpha_type const& alpha) const noexcept
    {
        int idx = detail::rank<M, N>(alpha);
        return idx < 0 ? Coeff(0) : c_[static_cast<std::size_t>(idx)];
    }

    Coeff coeff(sparse_alpha const& alpha) const noexcept
    {
        int idx = rank_sparse(alpha);
        return idx < 0 ? Coeff(0) : c_[static_cast<std::size_t>(idx)];
    }

    // Set the normalized Taylor coefficient for alpha. If alpha is outside this
    // otinum's configured total order, the request is ignored.
    OTI_CONSTEXPR_FUNCTION void set_coeff(alpha_type const& alpha, Coeff value) noexcept
    {
        int idx = detail::rank<M, N>(alpha);
        if (idx >= 0) {
            c_[static_cast<std::size_t>(idx)] = value;
        }
    }

    void set_coeff(sparse_alpha const& alpha, Coeff value) noexcept
    {
        int idx = rank_sparse(alpha);
        if (idx >= 0) {
            c_[static_cast<std::size_t>(idx)] = value;
        }
    }

    // Return the ordinary partial derivative value alpha! * c[alpha], or zero if
    // alpha is outside this otinum's configured total order.
    OTI_CONSTEXPR_FUNCTION Coeff partial(alpha_type const& alpha) const noexcept
    {
        int idx = detail::rank<M, N>(alpha);
        if (idx < 0) {
            return Coeff(0);
        }
        return c_[static_cast<std::size_t>(idx)] *
               table_type::template factorial_alpha_as<Coeff>(idx);
    }

    Coeff partial(sparse_alpha const& alpha) const noexcept
    {
        int idx = rank_sparse(alpha);
        if (idx < 0) {
            return Coeff(0);
        }
        return c_[static_cast<std::size_t>(idx)] *
               table_type::template factorial_alpha_as<Coeff>(idx);
    }

    // Set the ordinary partial derivative value for alpha. The stored normalized
    // Taylor coefficient is value / alpha!. If alpha is outside this otinum's
    // configured total order, the request is ignored.
    OTI_CONSTEXPR_FUNCTION void set_partial(alpha_type const& alpha, Coeff value) noexcept
    {
        int idx = detail::rank<M, N>(alpha);
        if (idx >= 0) {
            c_[static_cast<std::size_t>(idx)] =
                value / table_type::template factorial_alpha_as<Coeff>(idx);
        }
    }

    void set_partial(sparse_alpha const& alpha, Coeff value) noexcept
    {
        int idx = rank_sparse(alpha);
        if (idx >= 0) {
            c_[static_cast<std::size_t>(idx)] =
                value / table_type::template factorial_alpha_as<Coeff>(idx);
        }
    }

    OTI_CONSTEXPR_FUNCTION detail::array<Coeff, ncoeffs> const& data() const noexcept
    {
        return c_;
    }

    OTI_CONSTEXPR otinum& operator+=(otinum const& rhs) noexcept
    {
        OTI_PROFILE_COUNT(add);
        OTI_PROFILE_COUNT(add_oti);
        for (int i = 0; i < ncoeffs; ++i) {
            c_[static_cast<std::size_t>(i)] += rhs.c_[static_cast<std::size_t>(i)];
        }
        return *this;
    }

    OTI_CONSTEXPR otinum& operator-=(otinum const& rhs) noexcept
    {
        OTI_PROFILE_COUNT(sub);
        OTI_PROFILE_COUNT(sub_oti);
        for (int i = 0; i < ncoeffs; ++i) {
            c_[static_cast<std::size_t>(i)] -= rhs.c_[static_cast<std::size_t>(i)];
        }
        return *this;
    }

    OTI_CONSTEXPR otinum& operator*=(otinum const& rhs) noexcept
    {
        *this = (*this) * rhs;
        return *this;
    }

    otinum& operator/=(otinum const& rhs) noexcept
    {
        *this = (*this) / rhs;
        return *this;
    }

    template <class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
    OTI_CONSTEXPR otinum& operator+=(Scalar rhs) noexcept
    {
        OTI_PROFILE_COUNT(add);
        OTI_PROFILE_COUNT(add_scalar);
        c_[0] += static_cast<Coeff>(rhs);
        return *this;
    }

    template <class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
    OTI_CONSTEXPR otinum& operator-=(Scalar rhs) noexcept
    {
        OTI_PROFILE_COUNT(sub);
        OTI_PROFILE_COUNT(sub_scalar);
        c_[0] -= static_cast<Coeff>(rhs);
        return *this;
    }

    template <class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
    OTI_CONSTEXPR otinum& operator*=(Scalar rhs) noexcept
    {
        OTI_PROFILE_COUNT(mul);
        OTI_PROFILE_COUNT(mul_scalar);
        Coeff factor = static_cast<Coeff>(rhs);
        for (int i = 0; i < ncoeffs; ++i) {
            c_[static_cast<std::size_t>(i)] *= factor;
        }
        return *this;
    }

    template <class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
    OTI_CONSTEXPR otinum& operator/=(Scalar rhs) noexcept
    {
        OTI_PROFILE_COUNT(div);
        OTI_PROFILE_COUNT(div_scalar);
        Coeff divisor = static_cast<Coeff>(rhs);
        if constexpr (N > 0) {
            bool singular_divisor = !detail::oti_isfinite(Coeff(1) / divisor);
            c_[0] /= divisor;
            if (singular_divisor) {
                for (int i = 1; i < ncoeffs; ++i) {
                    c_[static_cast<std::size_t>(i)] = static_cast<Coeff>(NAN);
                }
                return *this;
            }
            for (int i = 1; i < ncoeffs; ++i) {
                c_[static_cast<std::size_t>(i)] /= divisor;
            }
            return *this;
        }
        for (int i = 0; i < ncoeffs; ++i) {
            c_[static_cast<std::size_t>(i)] /= divisor;
        }
        return *this;
    }

private:
    static int rank_sparse(sparse_alpha const& sparse) noexcept
    {
        alpha_type alpha{};
        for (sparse_index term : sparse.terms) {
            if (term.variable < 0 || term.variable >= M || term.order < 0) {
                return -1;
            }
            alpha[static_cast<std::size_t>(term.variable)] += term.order;
        }
        return detail::rank<M, N>(alpha);
    }

    detail::array<Coeff, ncoeffs> c_{};
};

// Comparisons intentionally use only the real coefficient. This makes OTI values
// usable in ordinary branch/control-flow code while leaving derivative
// propagation to the arithmetic on the branch that is actually taken.
template <int M, int N, class Coeff>
OTI_CONSTEXPR_FUNCTION bool operator==(otinum<M, N, Coeff> const& lhs,
                                       otinum<M, N, Coeff> const& rhs) noexcept
{
    return lhs.real() == rhs.real();
}

template <int M, int N, class Coeff>
OTI_CONSTEXPR_FUNCTION bool operator!=(otinum<M, N, Coeff> const& lhs,
                                       otinum<M, N, Coeff> const& rhs) noexcept
{
    return !(lhs == rhs);
}

template <int M, int N, class Coeff>
OTI_CONSTEXPR_FUNCTION bool operator<(otinum<M, N, Coeff> const& lhs,
                                      otinum<M, N, Coeff> const& rhs) noexcept
{
    return lhs.real() < rhs.real();
}

template <int M, int N, class Coeff>
OTI_CONSTEXPR_FUNCTION bool operator<=(otinum<M, N, Coeff> const& lhs,
                                       otinum<M, N, Coeff> const& rhs) noexcept
{
    return lhs.real() <= rhs.real();
}

template <int M, int N, class Coeff>
OTI_CONSTEXPR_FUNCTION bool operator>(otinum<M, N, Coeff> const& lhs,
                                      otinum<M, N, Coeff> const& rhs) noexcept
{
    return lhs.real() > rhs.real();
}

template <int M, int N, class Coeff>
OTI_CONSTEXPR_FUNCTION bool operator>=(otinum<M, N, Coeff> const& lhs,
                                       otinum<M, N, Coeff> const& rhs) noexcept
{
    return lhs.real() >= rhs.real();
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_CONSTEXPR_FUNCTION bool operator==(otinum<M, N, Coeff> const& lhs, Scalar rhs) noexcept
{
    return lhs.real() == static_cast<Coeff>(rhs);
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_CONSTEXPR_FUNCTION bool operator==(Scalar lhs, otinum<M, N, Coeff> const& rhs) noexcept
{
    return static_cast<Coeff>(lhs) == rhs.real();
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_CONSTEXPR_FUNCTION bool operator!=(otinum<M, N, Coeff> const& lhs, Scalar rhs) noexcept
{
    return !(lhs == rhs);
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_CONSTEXPR_FUNCTION bool operator!=(Scalar lhs, otinum<M, N, Coeff> const& rhs) noexcept
{
    return !(lhs == rhs);
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_CONSTEXPR_FUNCTION bool operator<(otinum<M, N, Coeff> const& lhs, Scalar rhs) noexcept
{
    return lhs.real() < static_cast<Coeff>(rhs);
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_CONSTEXPR_FUNCTION bool operator<(Scalar lhs, otinum<M, N, Coeff> const& rhs) noexcept
{
    return static_cast<Coeff>(lhs) < rhs.real();
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_CONSTEXPR_FUNCTION bool operator<=(otinum<M, N, Coeff> const& lhs, Scalar rhs) noexcept
{
    return lhs.real() <= static_cast<Coeff>(rhs);
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_CONSTEXPR_FUNCTION bool operator<=(Scalar lhs, otinum<M, N, Coeff> const& rhs) noexcept
{
    return static_cast<Coeff>(lhs) <= rhs.real();
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_CONSTEXPR_FUNCTION bool operator>(otinum<M, N, Coeff> const& lhs, Scalar rhs) noexcept
{
    return lhs.real() > static_cast<Coeff>(rhs);
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_CONSTEXPR_FUNCTION bool operator>(Scalar lhs, otinum<M, N, Coeff> const& rhs) noexcept
{
    return static_cast<Coeff>(lhs) > rhs.real();
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_CONSTEXPR_FUNCTION bool operator>=(otinum<M, N, Coeff> const& lhs, Scalar rhs) noexcept
{
    return lhs.real() >= static_cast<Coeff>(rhs);
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_CONSTEXPR_FUNCTION bool operator>=(Scalar lhs, otinum<M, N, Coeff> const& rhs) noexcept
{
    return static_cast<Coeff>(lhs) >= rhs.real();
}

template <int M, int N, class Coeff>
OTI_CONSTEXPR otinum<M, N, Coeff> operator+(otinum<M, N, Coeff> lhs,
                                            otinum<M, N, Coeff> const& rhs) noexcept
{
    lhs += rhs;
    return lhs;
}

template <int M, int N, class Coeff>
OTI_CONSTEXPR otinum<M, N, Coeff> operator-(otinum<M, N, Coeff> lhs,
                                            otinum<M, N, Coeff> const& rhs) noexcept
{
    lhs -= rhs;
    return lhs;
}

template <int M, int N, class Coeff>
OTI_CONSTEXPR otinum<M, N, Coeff> operator-(otinum<M, N, Coeff> value) noexcept
{
    OTI_PROFILE_COUNT(neg);
    value *= Coeff(-1);
    return value;
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_CONSTEXPR otinum<M, N, Coeff> operator+(otinum<M, N, Coeff> lhs, Scalar rhs) noexcept
{
    lhs += rhs;
    return lhs;
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_CONSTEXPR otinum<M, N, Coeff> operator+(Scalar lhs, otinum<M, N, Coeff> rhs) noexcept
{
    rhs += lhs;
    return rhs;
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_CONSTEXPR otinum<M, N, Coeff> operator-(otinum<M, N, Coeff> lhs, Scalar rhs) noexcept
{
    lhs -= rhs;
    return lhs;
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_CONSTEXPR otinum<M, N, Coeff> operator-(Scalar lhs, otinum<M, N, Coeff> rhs) noexcept
{
    rhs *= Coeff(-1);
    rhs += lhs;
    return rhs;
}

namespace detail {

// Compile-time-unrolled truncated polynomial convolution. Each product term's
// (lhs, rhs, out) indices come from the static product_terms table at a
// compile-time pack index P, so they fold to literal array offsets: the loop
// becomes straight-line register FMAs with no runtime table lookup. Using a
// runtime-indexed accessor here instead forces the compiler to materialize the
// whole index table on the stack and round-trip every coefficient through
// memory (see the TODO in multi_index.hpp).
template <int M, int N, class Coeff, std::size_t... P>
OTI_CONSTEXPR_FUNCTION void otinum_mul_into(otinum<M, N, Coeff>& out,
                                            otinum<M, N, Coeff> const& lhs,
                                            otinum<M, N, Coeff> const& rhs,
                                            std::index_sequence<P...>) noexcept
{
    using tb = tables<M, N>;
    ((out[tb::product_terms[P].out] +=
          lhs[tb::product_terms[P].lhs] * rhs[tb::product_terms[P].rhs]),
     ...);
}

} // namespace detail

template <int M, int N, class Coeff>
OTI_CONSTEXPR otinum<M, N, Coeff> operator*(otinum<M, N, Coeff> const& lhs,
                                            otinum<M, N, Coeff> const& rhs) noexcept
{
    otinum<M, N, Coeff> out;
    OTI_PROFILE_COUNT(mul);
    OTI_PROFILE_COUNT(mul_oti);

    // Polynomial convolution in the truncated multi-index algebra:
    // c[alpha + beta] += lhs[alpha] * rhs[beta] when |alpha + beta| <= N.
    detail::otinum_mul_into(out, lhs, rhs,
                            std::make_index_sequence<detail::tables<M, N>::nproducts>{});

    return out;
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_CONSTEXPR otinum<M, N, Coeff> operator*(otinum<M, N, Coeff> lhs, Scalar rhs) noexcept
{
    lhs *= rhs;
    return lhs;
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_CONSTEXPR otinum<M, N, Coeff> operator*(Scalar lhs, otinum<M, N, Coeff> rhs) noexcept
{
    rhs *= lhs;
    return rhs;
}

// Fused operations: common expression patterns evaluated without the
// intermediate otinum temporaries of the equivalent operator chains
// (y = y + a*b builds a*b, then y + that, then assigns). Operands are
// deliberately taken BY VALUE and results staged through a local: a
// whole-object copy of an aligned otinum compiles to wide vector loads
// and gives the compiler alias-free locals to work on, whereas reading
// coefficients one at a time through references into device/global
// memory forces conservative scalar access (measured: the by-value form
// of fma_into is ~2x the operator chain at <3,3> double on CUDA; the
// by-reference form is slower than the chain).
//
// Rounding: each fused form is bit-identical to performing the same
// mathematical accumulation in the same order, NOT to the operator chain
// it replaces. In particular fma_into(y, a, b) accumulates the product
// terms directly onto y, which can differ in the last ulp from
// y = y + a*b (products summed from zero, y added at the end). Floating-
// point contraction adds another last-ulp source: compilers that contract
// a*b + c into a hardware fma (Clang at -O2 by default) round the fused
// loop once per coefficient but the operator chain twice, so even axpy and
// scale_add only match their chains exactly when contraction treats both
// forms alike (as on GCC and NVCC today).

// y += s * x  (BLAS axpy: "a x plus y").
template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_CONSTEXPR void axpy(otinum<M, N, Coeff>& y, Scalar s,
                        otinum<M, N, Coeff> x) noexcept
{
    OTI_PROFILE_COUNT(axpy);
    otinum<M, N, Coeff> acc = y;
    for (int i = 0; i < otinum<M, N, Coeff>::ncoeffs; ++i) {
        acc[i] += static_cast<Coeff>(s) * x[i];
    }
    y = acc;
}

// a + s * b, returned without materializing s*b.
template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_CONSTEXPR otinum<M, N, Coeff> scale_add(otinum<M, N, Coeff> a, Scalar s,
                                            otinum<M, N, Coeff> b) noexcept
{
    OTI_PROFILE_COUNT(scale_add);
    for (int i = 0; i < otinum<M, N, Coeff>::ncoeffs; ++i) {
        a[i] += static_cast<Coeff>(s) * b[i];
    }
    return a;
}

// y += a * b (truncated product accumulated in place; fused multiply-add).
// The convolution fold in otinum_mul_into accumulates into its output, so
// seeding it with y instead of zero is exactly the fused form.
template <int M, int N, class Coeff>
OTI_CONSTEXPR void fma_into(otinum<M, N, Coeff>& y, otinum<M, N, Coeff> a,
                            otinum<M, N, Coeff> b) noexcept
{
    OTI_PROFILE_COUNT(fma_into);
    otinum<M, N, Coeff> acc = y;
    detail::otinum_mul_into(acc, a, b,
                            std::make_index_sequence<detail::tables<M, N>::nproducts>{});
    y = acc;
}

// Plain-arithmetic overloads of the fused operations, so kernels written
// generically over a Scalar type (double in a baseline build, otinum in an
// AD build) can use one spelling for both:  using oti::axpy;  axpy(y, s, x);
template <class T, class S,
          std::enable_if_t<std::is_arithmetic<T>::value && std::is_convertible<S, T>::value, int> = 0>
OTI_CONSTEXPR void axpy(T& y, S s, T x) noexcept
{
    y += static_cast<T>(s) * x;
}

template <class T, class S,
          std::enable_if_t<std::is_arithmetic<T>::value && std::is_convertible<S, T>::value, int> = 0>
OTI_CONSTEXPR T scale_add(T a, S s, T b) noexcept
{
    return a + static_cast<T>(s) * b;
}

template <class T, std::enable_if_t<std::is_arithmetic<T>::value, int> = 0>
OTI_CONSTEXPR void fma_into(T& y, T a, T b) noexcept
{
    y += a * b;
}

namespace detail {

// Truncated convolution: like otinum_mul_into, but keeps only product terms
// whose output order is <= max_order. The output order of each term is a
// compile-time literal (constexpr index into order_of), so only the comparison
// against the runtime max_order remains; the indices still fold to literals.
template <int M, int N, class Coeff, std::size_t... P>
OTI_CONSTEXPR_FUNCTION void otinum_trunc_mul_into(otinum<M, N, Coeff>& out,
                                                  otinum<M, N, Coeff> const& lhs,
                                                  otinum<M, N, Coeff> const& rhs,
                                                  int max_order,
                                                  std::index_sequence<P...>) noexcept
{
    using tb = tables<M, N>;
    ((tb::order_of[tb::product_terms[P].out] <= max_order
          ? (void)(out[tb::product_terms[P].out] +=
                       lhs[tb::product_terms[P].lhs] * rhs[tb::product_terms[P].rhs])
          : (void)0),
     ...);
}

} // namespace detail

template <int M, int N, class Coeff>
OTI_CONSTEXPR otinum<M, N, Coeff> trunc_mul(otinum<M, N, Coeff> const& lhs,
                                            otinum<M, N, Coeff> const& rhs,
                                            int max_order) noexcept
{
    otinum<M, N, Coeff> out;
    OTI_PROFILE_COUNT(trunc_mul);
    if (max_order < 0) {
        return out;
    }
    if (max_order > N) {
        max_order = N;
    }

    // Same convolution as operator*, but discards every term above max_order.
    detail::otinum_trunc_mul_into(out, lhs, rhs, max_order,
                                  std::make_index_sequence<detail::tables<M, N>::nproducts>{});

    return out;
}

template <int M, int N, class Coeff>
OTI_CONSTEXPR otinum<M, N, Coeff> trunc_add(otinum<M, N, Coeff> const& lhs,
                                            otinum<M, N, Coeff> const& rhs,
                                            int max_order) noexcept
{
    using tables = detail::tables<M, N>;
    otinum<M, N, Coeff> out;
    OTI_PROFILE_COUNT(trunc_add);
    if (max_order < 0) {
        return out;
    }
    if (max_order > N) {
        max_order = N;
    }

    for (int i = 0; i < otinum<M, N, Coeff>::ncoeffs; ++i) {
        if (tables::order_of_value(i) <= max_order) {
            out[i] = lhs[i] + rhs[i];
        }
    }
    return out;
}

template <int M, int N, class Coeff>
OTI_CONSTEXPR otinum<M, N, Coeff> gem(otinum<M, N, Coeff> const& a,
                                      otinum<M, N, Coeff> const& b,
                                      otinum<M, N, Coeff> const& c) noexcept
{
    OTI_PROFILE_COUNT(gem);
    return a * b + c;
}

namespace detail {

// Sum of the known product contributions value[i] * out[j] for terms
// (i, j) -> K, excluding the (0, K) term that carries the unknown out[K]. The
// product indices come from the static by-output table at compile-time pack
// indices, so they fold to literal offsets (no runtime table materialization).
template <int M, int N, class Coeff, std::size_t K, std::size_t... Q>
OTI_CONSTEXPR_FUNCTION Coeff inv_known_sum(otinum<M, N, Coeff> const& value,
                                           otinum<M, N, Coeff> const& out,
                                           std::index_sequence<Q...>) noexcept
{
    using tb = tables<M, N>;
    constexpr int begin = tb::product_offset[K];
    Coeff acc = Coeff(0);
    ((acc += (tb::product_terms_by_output[begin + Q].lhs == 0 &&
              tb::product_terms_by_output[begin + Q].rhs == static_cast<int>(K))
                 ? Coeff(0)
                 : value[tb::product_terms_by_output[begin + Q].lhs] *
                       out[tb::product_terms_by_output[begin + Q].rhs]),
     ...);
    return acc;
}

// Solve out[k] = -known_sum(k) / r for every non-real coefficient k in graded
// order. The comma fold is sequenced left to right, so each out[k] is already
// computed before the higher-order coefficients that depend on it are solved.
template <int M, int N, class Coeff, std::size_t... K>
OTI_CONSTEXPR_FUNCTION void inv_solve(otinum<M, N, Coeff>& out,
                                      otinum<M, N, Coeff> const& value,
                                      Coeff r, std::index_sequence<K...>) noexcept
{
    using tb = tables<M, N>;
    ((out[static_cast<int>(K) + 1] =
          -inv_known_sum<M, N, Coeff, K + 1>(
               value, out,
               std::make_index_sequence<tb::product_offset[K + 2] -
                                        tb::product_offset[K + 1]>{}) /
          r),
     ...);
}

} // namespace detail

template <int M, int N, class Coeff>
OTI_FUNCTION otinum<M, N, Coeff> inv(otinum<M, N, Coeff> const& value) noexcept
{
    // The inverse is expanded around the real coefficient. A valid real-valued
    // Taylor inverse requires value.real() != 0.
    Coeff r = value.real();
    OTI_PROFILE_COUNT(inv);

    otinum<M, N, Coeff> out;
    out[0] = Coeff(1) / r;

    if constexpr (N > 0) {
        if (!detail::oti_isfinite(out[0])) {
            // value.real() == 0 (or non-finite): 1/x has a pole here, so there
            // is no valid Taylor jet. Report the scalar value (out[0], typically
            // inf) and NaN for every derivative -- the same singular-point
            // contract that the elementary functions use (see apply_scalar).
            for (int i = 1; i < otinum<M, N, Coeff>::ncoeffs; ++i) {
                out[i] = static_cast<Coeff>(NAN);
            }
            return out;
        }

        // Solve value * out = 1 coefficient-by-coefficient.
        //
        // For every non-real coefficient k, the target coefficient of the
        // product is zero:
        //
        //   (value * out)[k] = value[0] * out[k] + known_terms = 0
        //
        // The term value[0] * out[k] contains the unknown being solved, so it is
        // excluded while accumulating known_terms. All other dependencies involve
        // out coefficients of lower total order, which are already computed
        // because the coefficient layout is graded by total order (and the fold
        // in inv_solve is sequenced in increasing k).
        detail::inv_solve(out, value, r,
                          std::make_index_sequence<otinum<M, N, Coeff>::ncoeffs - 1>{});
    }
    return out;
}

template <int M, int N, class Coeff>
OTI_FUNCTION otinum<M, N, Coeff> operator/(otinum<M, N, Coeff> const& lhs,
                                           otinum<M, N, Coeff> const& rhs) noexcept
{
    OTI_PROFILE_COUNT(div);
    OTI_PROFILE_COUNT(div_oti);
    return lhs * inv(rhs);
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_CONSTEXPR otinum<M, N, Coeff> operator/(otinum<M, N, Coeff> lhs, Scalar rhs) noexcept
{
    lhs /= rhs;
    return lhs;
}

template <int M, int N, class Coeff, class Scalar, scalar_enable_t<Coeff, Scalar> = 0>
OTI_FUNCTION otinum<M, N, Coeff> operator/(Scalar lhs, otinum<M, N, Coeff> const& rhs) noexcept
{
    return otinum<M, N, Coeff>(static_cast<Coeff>(lhs)) / rhs;
}

} // namespace oti
